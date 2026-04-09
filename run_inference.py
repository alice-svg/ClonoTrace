#!/usr/bin/env python
"""
Run trajectory inference pipeline with updated btraj code.
Produces trajectory_results.h5ad for evaluation.

Features:
  - Structured logging (console + file)
  - Heartbeat threads for long operations
  - Checkpoint/resume support
  - Fail-fast validation (no silent error swallowing)
  - NPY embedding alignment + PCA
  - SOTA pseudotime CSV integration

Usage:
    python run_inference.py --input /path/to/file.h5ad
    python run_inference.py --input /path/to/file.h5ad --embeddings-npy /path/to/full.npy
    python run_inference.py --input /path/to/file.h5ad --resume
"""

import os
import sys
import time
import logging
import resource
import threading
import glob
import warnings
import argparse
from datetime import datetime

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

os.environ["SETUPTOOLS_SCM_PRETEND_VERSION"] = "0.0.0dev0"

# Fix OpenMP threading conflict on macOS (conda + pip BLAS libraries).
# Multiple OpenMP runtimes (conda libgomp/libomp vs pip MKL/OpenBLAS) cause
# pthread_mutex_init failures. Single-threaded BLAS avoids this entirely.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse.csgraph import shortest_path


# ---------------------------------------------------------------------------
# Logging infrastructure
# ---------------------------------------------------------------------------

def setup_logging(log_dir):
    """Configure dual-output logging: console (INFO) + file (DEBUG)."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"pipeline_{timestamp}.log")

    logger = logging.getLogger("btraj_pipeline")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    # Console handler — INFO level, concise format
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(ch)

    # File handler — DEBUG level, full format
    fh = logging.FileHandler(log_file, mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s"))
    logger.addHandler(fh)

    logger.info(f"Log file: {log_file}")
    return logger


def log_memory(logger):
    """Log current RSS memory usage."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    rss_mb = usage.ru_maxrss / (1024 * 1024)  # macOS reports bytes
    logger.debug(f"Peak RSS: {rss_mb:.0f} MB")
    return rss_mb


# ---------------------------------------------------------------------------
# Heartbeat context manager
# ---------------------------------------------------------------------------

class Heartbeat:
    """Thread-based heartbeat that emits periodic status messages during long operations."""

    def __init__(self, logger, step_name, interval=30.0):
        self.logger = logger
        self.step_name = step_name
        self.interval = interval
        self._stop_event = threading.Event()
        self._thread = None
        self._start_time = None

    def _run(self):
        while not self._stop_event.wait(self.interval):
            elapsed = time.time() - self._start_time
            rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
            self.logger.info(f"[{self.step_name}] still running... {elapsed:.0f}s elapsed, RSS={rss_mb:.0f}MB")

    def __enter__(self):
        self._start_time = time.time()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop_event.set()
        self._thread.join(timeout=5)
        return False


# ---------------------------------------------------------------------------
# CheckpointManager
# ---------------------------------------------------------------------------

class CheckpointManager:
    """Save and restore pipeline checkpoints as h5ad files."""

    def __init__(self, checkpoint_dir, logger):
        self.checkpoint_dir = checkpoint_dir
        self.logger = logger
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save(self, adata, step_num, step_name):
        """Write a checkpoint h5ad file."""
        safe_name = step_name.replace(" ", "_").replace("/", "_")
        filename = f"checkpoint_step{step_num}_{safe_name}.h5ad"
        path = os.path.join(self.checkpoint_dir, filename)
        t0 = time.time()
        adata.write_h5ad(path)
        self.logger.info(f"Checkpoint saved: {path} ({time.time() - t0:.1f}s)")
        return path

    def find_latest(self):
        """Find the latest checkpoint. Returns (step_num, path) or None."""
        pattern = os.path.join(self.checkpoint_dir, "checkpoint_step*.h5ad")
        files = sorted(glob.glob(pattern))
        if not files:
            return None
        latest = files[-1]
        basename = os.path.basename(latest)
        # Extract step number from "checkpoint_stepN_..."
        try:
            step_num = int(basename.split("_")[1].replace("step", ""))
        except (IndexError, ValueError):
            step_num = 0
        return step_num, latest

    def load(self, path):
        """Load a checkpoint h5ad file."""
        self.logger.info(f"Loading checkpoint: {path}")
        t0 = time.time()
        adata = sc.read_h5ad(path)
        self.logger.info(f"Checkpoint loaded: {adata.n_obs} cells ({time.time() - t0:.1f}s)")
        return adata


# ---------------------------------------------------------------------------
# Fail-fast validation
# ---------------------------------------------------------------------------

def validate_inputs(adata, args, logger):
    """Collect ALL validation errors before raising, so user sees everything at once."""
    errors = []

    # Required obs columns
    for col in ["celltype", "q_score", "Heavy", "Light"]:
        if col not in adata.obs.columns:
            errors.append(f"Missing required obs column: '{col}'")

    # Required obsm — auto-detect X_scvi or X_scVI
    if "X_scvi" not in adata.obsm and "X_scVI" not in adata.obsm:
        errors.append("Missing required obsm key: 'X_scvi' or 'X_scVI'")
    elif "X_scVI" in adata.obsm and "X_scvi" not in adata.obsm:
        # Normalize to lowercase for pipeline consistency
        logger.info("  Auto-detected 'X_scVI' → renaming to 'X_scvi'")
        adata.obsm["X_scvi"] = adata.obsm["X_scVI"].copy()
        del adata.obsm["X_scVI"]

    # q_score range check
    if "q_score" in adata.obs.columns:
        q = adata.obs["q_score"]
        if not pd.api.types.is_numeric_dtype(q):
            errors.append(f"q_score is not numeric (dtype={q.dtype})")
        else:
            q_min, q_max = q.min(), q.max()
            if q_min < -0.01 or q_max > 1.01:
                errors.append(f"q_score out of range [0, 1]: min={q_min:.4f}, max={q_max:.4f}")

    # Start type must exist
    start_type = args.start_type
    if "celltype" in adata.obs.columns:
        celltypes = set(adata.obs["celltype"].astype(str).unique())
        if start_type not in celltypes:
            errors.append(f"Start type '{start_type}' not found in celltype values: {sorted(celltypes)}")

    # External file checks
    if args.embeddings_npy and not os.path.isfile(args.embeddings_npy):
        errors.append(f"--embeddings-npy file not found: {args.embeddings_npy}")

    if args.pseudotime_csv and not os.path.isfile(args.pseudotime_csv):
        errors.append(f"--pseudotime-csv file not found: {args.pseudotime_csv}")

    # NPY alignment check (if file exists)
    if args.embeddings_npy and os.path.isfile(args.embeddings_npy):
        try:
            npy_shape = np.load(args.embeddings_npy, mmap_mode="r").shape
            # Count BCR+ cells
            if "Heavy" in adata.obs.columns and "Light" in adata.obs.columns:
                heavy = adata.obs["Heavy"].astype(str).str.strip()
                light = adata.obs["Light"].astype(str).str.strip()
                bcr_mask = (
                    ~heavy.isin(["", "nan", "None", "none"])
                    & ~light.isin(["", "nan", "None", "none"])
                    & (heavy.str.len() >= 80)
                    & (light.str.len() >= 80)
                )
                n_bcr = int(bcr_mask.sum())
                n_total = len(adata)
                if npy_shape[0] != n_bcr and npy_shape[0] != n_total:
                    errors.append(
                        f"NPY shape mismatch: {args.embeddings_npy} has {npy_shape[0]} rows, "
                        f"but adata has {n_total} total cells and {n_bcr} BCR+ cells"
                    )
                logger.info(f"NPY validation: shape={npy_shape}, total cells={n_total}, BCR+ cells={n_bcr}")
        except Exception as e:
            errors.append(f"Failed to read NPY file for validation: {e}")

    if errors:
        logger.error("Input validation failed with %d error(s):", len(errors))
        for i, err in enumerate(errors, 1):
            logger.error("  [%d] %s", i, err)
        raise ValueError(f"Input validation failed with {len(errors)} error(s). See log above.")

    logger.info("Input validation passed.")


# ---------------------------------------------------------------------------
# NPY embedding loading + alignment + PCA
# ---------------------------------------------------------------------------

def load_and_align_npy_embeddings(adata, npy_path, logger):
    """
    Load full-length AntiBERTy NPY (BCR+ cells only), align to adata, PCA reduce.

    The NPY file contains embeddings for BCR+ cells only (Heavy+Light both present, len>=80).
    This function:
    1. Identifies BCR+ cells in adata using the same filter
    2. Aligns NPY rows to those cells
    3. PCA reduces to 256 dims
    4. Builds full (n_cells x 256) matrix with NaN for non-BCR cells
    """
    from sklearn.decomposition import PCA

    logger.info(f"Loading NPY embeddings from {npy_path}")
    X_full = np.load(npy_path)
    logger.info(f"NPY shape: {X_full.shape}")

    # Identify BCR+ cells (matching notebook 2 filter)
    heavy = adata.obs["Heavy"].astype(str).str.strip()
    light = adata.obs["Light"].astype(str).str.strip()
    bcr_mask = (
        ~heavy.isin(["", "nan", "None", "none"])
        & ~light.isin(["", "nan", "None", "none"])
        & (heavy.str.len() >= 80)
        & (light.str.len() >= 80)
    )
    n_bcr = int(bcr_mask.sum())
    logger.info(f"BCR+ cells in adata: {n_bcr}")

    if X_full.shape[0] == n_bcr:
        # NPY contains only BCR+ cells — use directly
        X_bcr = X_full
        logger.info("NPY contains BCR+ cells only — using directly")
    elif X_full.shape[0] == adata.n_obs:
        # NPY contains all cells — subset to BCR+ cells
        X_bcr = X_full[bcr_mask.values]
        logger.info(f"NPY contains all cells ({X_full.shape[0]}) — subsetting to {n_bcr} BCR+ cells")
    else:
        raise ValueError(
            f"NPY row count ({X_full.shape[0]}) != BCR+ cell count ({n_bcr}) "
            f"and != total cell count ({adata.n_obs}). Cannot align embeddings."
        )

    # PCA reduce to 256 dims on BCR+ subset
    pca_dim = 256
    effective_dim = min(pca_dim, X_bcr.shape[0], X_bcr.shape[1])
    logger.info(f"PCA reducing {X_bcr.shape[1]} dims -> {effective_dim} dims on {n_bcr} BCR+ cells")
    pca = PCA(n_components=effective_dim, random_state=42)
    X_reduced = pca.fit_transform(X_bcr)
    logger.info(f"PCA explained variance (first 5): {pca.explained_variance_ratio_[:5]}")

    # Build full matrix with NaN for non-BCR cells
    X_h = np.full((adata.n_obs, effective_dim), np.nan, dtype=np.float64)
    X_h[bcr_mask.values] = X_reduced

    adata.obs["has_valid_bcr_embedding"] = bcr_mask.values
    n_nan = int(np.isnan(X_h).any(axis=1).sum())
    logger.info(f"X_h built: shape={X_h.shape}, NaN rows={n_nan}, valid rows={adata.n_obs - n_nan}")

    return X_h


# ---------------------------------------------------------------------------
# SOTA pseudotime integration
# ---------------------------------------------------------------------------

def integrate_pseudotime_comparison(adata, csv_path, logger):
    """Load SOTA pseudotime CSV and merge columns into adata.obs."""
    logger.info(f"Loading SOTA pseudotime CSV from {csv_path}")
    df = pd.read_csv(csv_path, index_col=0)
    logger.info(f"CSV shape: {df.shape}, columns: {list(df.columns)}")

    # Find intersection with adata cell barcodes
    common = adata.obs_names.intersection(df.index)
    logger.info(f"Cell barcode overlap: {len(common)} / {adata.n_obs} adata cells, {len(common)} / {len(df)} CSV rows")

    if len(common) == 0:
        logger.warning("No overlapping cell barcodes between adata and pseudotime CSV. Skipping.")
        return

    # Merge pseudotime columns
    pt_cols = [c for c in df.columns if "pseudotime" in c.lower()]
    if not pt_cols:
        pt_cols = list(df.columns)
        logger.warning(f"No 'pseudotime' columns found, using all: {pt_cols}")

    for col in pt_cols:
        adata.obs[col] = np.nan
        adata.obs.loc[common, col] = df.loc[common, col].values
        n_valid = int(adata.obs[col].notna().sum())
        logger.info(f"  Merged '{col}': {n_valid} valid values")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

DEFAULT_INPUT = os.path.join(
    os.path.dirname(__file__),
    "20260114_fetal_B_celltype_merged-BCR_scVI_v2_qbcr.h5ad",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run btraj trajectory inference pipeline on fetal B cell data."
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help="Path to the input .h5ad file (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path for the output .h5ad file. Defaults to trajectory_results.h5ad beside --input.",
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Skip AntiBERTy embedding computation if X_h already exists in adata.obsm.",
    )
    parser.add_argument(
        "--embeddings-cache",
        default=None,
        help="Path to a .npy file for caching/loading precomputed PCA-reduced AntiBERTy embeddings.",
    )
    parser.add_argument(
        "--embeddings-npy",
        default=None,
        help="Path to full-length AntiBERTy .npy (BCR+ cells only). Triggers alignment + PCA.",
    )
    parser.add_argument(
        "--pseudotime-csv",
        default=None,
        help="Path to SOTA pseudotime CSV for comparison (merged into adata.obs).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest checkpoint.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Directory for checkpoints (default: <output_dir>/.checkpoints).",
    )
    parser.add_argument(
        "--log-dir",
        default=None,
        help="Directory for log files (default: output directory).",
    )
    parser.add_argument(
        "--heartbeat-interval",
        type=float,
        default=30.0,
        help="Seconds between heartbeat messages (default: 30).",
    )
    parser.add_argument(
        "--start-type",
        default="PRE_PRO_B",
        help="Starting cell type for trajectory root (default: PRE_PRO_B).",
    )
    parser.add_argument(
        "--terminal-types",
        default="MATURE_B,B1,PLASMA_B",
        help="Comma-separated terminal cell types for fate probabilities (default: MATURE_B,B1,PLASMA_B).",
    )
    parser.add_argument(
        "--fate-method",
        choices=["direct", "sparse_custom", "gpcca"],
        default="direct",
        help=(
            "Fate probability method. "
            "'direct': bypass GPCCA macrostates, set terminal states from biology, sparse GMRES (fast, default). "
            "'sparse_custom': custom scipy.sparse.linalg solver (fastest, fallback). "
            "'gpcca': full GPCCA with Schur decomposition (very slow without petsc4py)."
        ),
    )
    parser.add_argument(
        "--pseudotime-mode",
        choices=["raw", "q_aware"],
        default="q_aware",
        help=(
            "Pseudotime computation mode. "
            "'raw': geometric projection only (original). "
            "'q_aware': apply q_score monotonicity correction (default)."
        ),
    )
    parser.add_argument(
        "--q-aware-strength",
        choices=["fast", "balanced", "full"],
        default="balanced",
        help=(
            "Blending strength for q_aware pseudotime. "
            "'fast': alpha=0.3 (weak q-score influence — use for cyclical biology like GC). "
            "'balanced': alpha=0.5 (default). "
            "'full': alpha=0.7 (strong q-score influence)."
        ),
    )
    parser.add_argument(
        "--celltype-order",
        default="",
        help=(
            "Comma-separated cell types in expected developmental order (earliest to latest). "
            "Used to validate pseudotime inversion rate. Isotonic alignment is applied ONLY "
            "as a fallback when >50%% of adjacent cell type pairs are inverted, indicating "
            "degenerate pseudotime. Otherwise the raw inferred pseudotime is kept. "
            "Example: 'PRE_PRO_B,PRO_B,LATE_PRO_B,CYCLING_B,LARGE_PRE_B,SMALL_PRE_B,IMMATURE_B,MATURE_B,B1,PLASMA_B'"
        ),
    )
    parser.add_argument(
        "--terminal-types-extended",
        default="",
        help=(
            "Comma-separated extended terminal types for fine-grained branch probabilities. "
            "Stored as prob_ext_* columns. Example: 'PRO_B,LARGE_PRE_B,IMMATURE_B,MATURE_B,B1,PLASMA_B'"
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Fate probability computation — fast alternatives to GPCCA macrostates
# ---------------------------------------------------------------------------

def compute_fate_direct(adata, combined_kernel, terminal_types, logger, hb_interval=30.0, col_prefix="prob_"):
    """
    Compute fate probabilities by directly setting terminal states from biology,
    bypassing GPCCA's expensive Schur decomposition entirely.

    Uses CellRank's GPCCA.set_terminal_states(dict) → compute_fate_probabilities(sparse GMRES).
    This skips compute_macrostates (O(n^3) dense Schur) and goes straight to sparse solving.

    Returns dict of {type_name: probability_array} or None on failure.
    """
    from cellrank.estimators import GPCCA

    celltypes = adata.obs["celltype"].astype(str)

    # Validate terminal types exist in data
    available = set(celltypes.unique())
    missing = [t for t in terminal_types if t not in available]
    if missing:
        logger.error(f"Terminal types not found in data: {missing}. Available: {sorted(available)}")
        raise ValueError(f"Terminal types not found: {missing}")

    # Build dict of {type_name: [barcode_list]} for CellRank
    terminal_dict = {}
    for ttype in terminal_types:
        barcodes = adata.obs_names[celltypes == ttype].tolist()
        terminal_dict[ttype] = barcodes
        logger.info(f"  Terminal type '{ttype}': {len(barcodes)} cells")

    # Create GPCCA estimator — skip compute_macrostates entirely
    g = GPCCA(combined_kernel)

    logger.info("  Setting terminal states directly (skipping Schur decomposition) ...")
    g.set_terminal_states(terminal_dict)
    terminal = list(g.terminal_states.dropna().unique())
    logger.info(f"  Terminal states set: {terminal}")

    # Verify transition matrix is sparse (critical for performance)
    from scipy.sparse import issparse
    if issparse(g.transition_matrix):
        nnz = g.transition_matrix.nnz
        density = nnz / (g.transition_matrix.shape[0] ** 2)
        logger.info(f"  Transition matrix: {g.transition_matrix.shape}, nnz={nnz:,}, density={density:.6f} (sparse)")
    else:
        logger.warning("  Transition matrix is DENSE — fate probabilities will be slow!")

    # Compute fate probabilities — uses sparse GMRES internally
    logger.info("  Computing fate probabilities (sparse GMRES) ...")
    with Heartbeat(logger, "fate-probs-GMRES", interval=hb_interval):
        g.compute_fate_probabilities(solver="gmres", use_petsc=False, tol=1e-6)

    # Extract results
    fate_probs = g.fate_probabilities
    logger.info(f"  Lineage names: {list(fate_probs.names)}")

    result = {}
    for lineage in fate_probs.names:
        col_name = f"{col_prefix}{lineage}"
        prob = fate_probs[lineage].X.flatten()
        adata.obs[col_name] = prob
        result[lineage] = prob
        logger.info(f"  Added: {col_name} (mean={prob.mean():.4f}, min={prob.min():.4f}, max={prob.max():.4f})")

    return result


def compute_fate_sparse_custom(adata, T_sparse, terminal_types, logger, hb_interval=30.0, col_prefix="prob_"):
    """
    Compute absorption probabilities using direct sparse LU factorization.

    Fastest approach — bypasses CellRank entirely, uses scipy.sparse.linalg.splu.
    Solves (I - T_QQ) * x = T_QS for each terminal type.

    Returns dict of {type_name: probability_array}.
    """
    from scipy.sparse import eye, issparse
    from scipy.sparse.linalg import splu

    if not issparse(T_sparse):
        logger.warning("  Converting dense transition matrix to sparse for custom solver")
        from scipy.sparse import csr_matrix
        T_sparse = csr_matrix(T_sparse)

    celltypes = adata.obs["celltype"].astype(str).values
    n = adata.n_obs

    # Build terminal mask
    terminal_mask = np.isin(celltypes, terminal_types)
    transient_mask = ~terminal_mask
    transient_idx = np.where(transient_mask)[0]
    terminal_idx = np.where(terminal_mask)[0]

    n_transient = len(transient_idx)
    n_terminal = len(terminal_idx)
    logger.info(f"  Transient cells: {n_transient:,}, Terminal cells: {n_terminal:,}")

    if n_transient == 0:
        logger.warning("  No transient cells — all cells are terminal")
        result = {}
        for ttype in terminal_types:
            prob = np.zeros(n)
            prob[celltypes == ttype] = 1.0
            col_name = f"{col_prefix}{ttype}"
            adata.obs[col_name] = prob
            result[ttype] = prob
        return result

    # Extract submatrices
    logger.info(f"  Extracting submatrices (T_QQ: {n_transient}x{n_transient}, T_QS: {n_transient}x{n_terminal}) ...")
    T_QQ = T_sparse[np.ix_(transient_idx, transient_idx)]
    T_QS = T_sparse[np.ix_(transient_idx, terminal_idx)]

    # Build A = I - T_QQ (CSC format required for splu)
    A = (eye(n_transient, format="csc") - T_QQ.tocsc())

    # LU factorize once
    logger.info(f"  LU factorizing {n_transient}x{n_transient} sparse matrix ...")
    with Heartbeat(logger, "sparse-LU", interval=hb_interval):
        lu = splu(A)
    logger.info(f"  LU complete (nnz in L+U: {lu.nnz:,})")

    # Solve for each terminal type
    terminal_celltypes = celltypes[terminal_idx]
    result = {}
    for ttype in terminal_types:
        type_in_terminal = (terminal_celltypes == ttype)
        rhs = np.asarray(T_QS[:, type_in_terminal].sum(axis=1)).ravel()

        x = lu.solve(rhs)

        # Build full-length probability vector
        prob = np.zeros(n)
        prob[transient_idx] = np.clip(x, 0.0, 1.0)
        prob[terminal_idx[type_in_terminal]] = 1.0

        col_name = f"{col_prefix}{ttype}"
        adata.obs[col_name] = prob
        result[ttype] = prob
        logger.info(f"  Solved {ttype}: mean={prob.mean():.4f}, min={prob.min():.6f}, max={prob.max():.4f}")

    # Verify probabilities sum to ~1 for transient cells
    prob_sum = sum(result.values())
    mean_sum = prob_sum[transient_idx].mean() if n_transient > 0 else 0.0
    logger.info(f"  Probability sum check (transient cells): mean={mean_sum:.4f} (ideal=1.0)")
    if abs(mean_sum - 1.0) > 0.05:
        logger.warning(f"  Probability sum deviates from 1.0 by {abs(mean_sum - 1.0):.4f} — check transition matrix quality")

    return result


def compute_fate_gpcca_full(adata, combined_kernel, logger, hb_interval=30.0):
    """
    Original GPCCA approach — compute_macrostates + predict_terminal_states + compute_fate_probabilities.

    WARNING: This densifies the transition matrix for Schur decomposition (O(n^3)).
    For 30k cells, this takes 6+ hours and ~7GB RAM without petsc4py/slepc4py.
    """
    from cellrank.estimators import GPCCA

    g = GPCCA(combined_kernel)

    # Try decreasing n_states until compute_macrostates succeeds
    for n_states in [6, 5, 4, 3]:
        try:
            logger.info(f"  Computing macrostates (n_states={n_states}) ...")
            with Heartbeat(logger, f"GPCCA-macrostates-{n_states}", interval=hb_interval):
                g.compute_macrostates(n_states=n_states, cluster_key="celltype")
            logger.info(f"  Macrostates computed: {list(g.macrostates.cat.categories)}")
            break
        except ValueError as ve:
            logger.warning(f"  n_states={n_states} failed: {ve}")
            continue
    else:
        raise RuntimeError("compute_macrostates failed for all n_states values (6, 5, 4, 3)")

    # Terminal states
    logger.info("  Predicting terminal states ...")
    with Heartbeat(logger, "GPCCA-terminal", interval=hb_interval):
        g.predict_terminal_states(method="stability", stability_threshold=0.96)
    terminal = list(g.terminal_states.dropna().unique())
    logger.info(f"  Terminal states: {terminal}")

    if len(terminal) == 0:
        logger.warning("  No terminal states detected. Using all macrostates.")
        all_macrostates = list(g.macrostates.cat.categories)
        g.set_terminal_states(states=all_macrostates, cluster_key="celltype")
        terminal = list(g.terminal_states.dropna().unique())
        logger.info(f"  Terminal states (manual): {terminal}")

    # Fate probabilities
    logger.info("  Computing fate probabilities ...")
    with Heartbeat(logger, "GPCCA-fate-probs", interval=hb_interval):
        g.compute_fate_probabilities()

    fate_probs = g.fate_probabilities
    logger.info(f"  Lineage names: {list(fate_probs.names)}")
    for lineage in fate_probs.names:
        col_name = f"prob_{lineage}"
        adata.obs[col_name] = fate_probs[lineage].X.flatten()
        logger.info(f"  Added: {col_name}")

    return {name: adata.obs[f"prob_{name}"].values for name in fate_probs.names}


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Resolve output path
    if args.output is None:
        out_path = os.path.join(os.path.dirname(args.input), "trajectory_results.h5ad")
    else:
        out_path = args.output

    out_dir = os.path.dirname(out_path) or "."

    # Setup logging
    log_dir = args.log_dir or out_dir
    logger = setup_logging(log_dir)

    # Setup checkpoint manager
    ckpt_dir = args.checkpoint_dir or os.path.join(out_dir, ".checkpoints")
    ckpt = CheckpointManager(ckpt_dir, logger)

    hb_interval = args.heartbeat_interval

    logger.info("=" * 70)
    logger.info("btraj Trajectory Inference Pipeline")
    logger.info("=" * 70)
    logger.info(f"  Input : {args.input}")
    logger.info(f"  Output: {out_path}")
    logger.info(f"  Checkpoints: {ckpt_dir}")
    logger.info(f"  Heartbeat interval: {hb_interval}s")

    pipeline_start = time.time()
    resume_from_step = 0

    # Handle --resume
    if args.resume:
        found = ckpt.find_latest()
        if found is not None:
            resume_from_step, ckpt_path = found
            logger.info(f"Resuming from checkpoint step {resume_from_step}: {ckpt_path}")
            adata = ckpt.load(ckpt_path)
        else:
            logger.warning("--resume specified but no checkpoints found. Starting from scratch.")

    # -------------------------------------------------------------------
    # Step 1: Load h5ad
    # -------------------------------------------------------------------
    if resume_from_step < 1:
        logger.info("[Step 1] Loading h5ad ...")
        t0 = time.time()
        adata = sc.read_h5ad(args.input)
        logger.info(f"  Loaded {adata.n_obs} cells x {adata.n_vars} genes  ({time.time() - t0:.1f}s)")
        logger.info(f"  Cell types: {sorted(adata.obs['celltype'].unique().tolist())}")
        log_memory(logger)

    # -------------------------------------------------------------------
    # Step 1b: Validate inputs — FAIL FAST
    # -------------------------------------------------------------------
    if resume_from_step < 2:
        logger.info("[Step 1b] Validating inputs ...")
        validate_inputs(adata, args, logger)

    # -------------------------------------------------------------------
    # Step 2: Cell type summary
    # -------------------------------------------------------------------
    if resume_from_step < 2:
        logger.info("[Step 2] Cell type summary (all subsets retained) ...")
        ct_counts = adata.obs["celltype"].value_counts()
        for ct, n in ct_counts.items():
            logger.info(f"  {ct:20s}: {n:,}")
        logger.info(f"  Total: {adata.n_obs:,} cells")

    # -------------------------------------------------------------------
    # Step 3: Compute/load AntiBERTy BCR embeddings
    # -------------------------------------------------------------------
    if resume_from_step < 3:
        logger.info("[Step 3] BCR embeddings ...")
        t0 = time.time()

        if args.embeddings_npy:
            # Full-length NPY → alignment + PCA
            logger.info("  Using --embeddings-npy: alignment + PCA path")
            with Heartbeat(logger, "embeddings-align-pca", interval=hb_interval):
                X_h = load_and_align_npy_embeddings(adata, args.embeddings_npy, logger)
            adata.obsm["X_h"] = X_h

        elif args.skip_embeddings and "X_h" in adata.obsm:
            logger.info("  Skipping — X_h already present in adata.obsm (--skip-embeddings).")

        elif args.embeddings_cache:
            # Precomputed PCA-reduced cache
            logger.info(f"  Using --embeddings-cache: {args.embeddings_cache}")
            from btraj.embedding.embed import AntiBERTyEmbedder
            with Heartbeat(logger, "embeddings-cache", interval=hb_interval):
                embedder = AntiBERTyEmbedder(try_gpu=True)
                X_h = embedder.compute_bcr_embeddings(
                    adata,
                    heavy_col="Heavy",
                    light_col="Light",
                    pca_dim=256,
                    precomputed_path=args.embeddings_cache,
                )
            adata.obsm["X_h"] = X_h

        else:
            # Compute fresh (requires GPU)
            logger.info("  Computing fresh AntiBERTy embeddings (no cache specified)")
            from btraj.embedding.embed import AntiBERTyEmbedder
            with Heartbeat(logger, "embeddings-compute", interval=hb_interval):
                embedder = AntiBERTyEmbedder(try_gpu=True)
                X_h = embedder.compute_bcr_embeddings(
                    adata,
                    heavy_col="Heavy",
                    light_col="Light",
                    pca_dim=256,
                )
            adata.obsm["X_h"] = X_h

        if "X_h" in adata.obsm:
            X_h = adata.obsm["X_h"]
            logger.info(
                f"  X_h shape: {X_h.shape} | "
                f"NaN rows: {int(np.isnan(X_h).any(axis=1).sum())}  ({time.time() - t0:.1f}s)"
            )

        # Checkpoint after embeddings
        ckpt.save(adata, 3, "embeddings")
        log_memory(logger)

    # -------------------------------------------------------------------
    # Step 4: Build cluster graph via get_trajectory()
    # -------------------------------------------------------------------
    if resume_from_step < 4:
        logger.info("[Step 4] Running get_trajectory() with q_score ...")
        t0 = time.time()

        from btraj.graph.mst_q_score import get_trajectory, compute_pseudotime, compute_pseudotime_with_q

        with Heartbeat(logger, "get_trajectory", interval=hb_interval):
            (
                clusters_links,
                tree,
                results,
                Lineage_class,
                branch_clusters,
                start_node,
                ids2name,
                name2ids,
                mode,
            ) = get_trajectory(
                cell_labels=adata.obs["celltype"].astype(str).values,
                y_features=adata.obsm["X_scvi"],
                cells=adata.obs_names.values,
                start_type=args.start_type,
                k=30,
                q_score=adata.obs["q_score"].values,
                q_tol=0.01,
            )
        logger.info(f"  get_trajectory() complete — mode={mode}  ({time.time() - t0:.1f}s)")

        # Map cell types to integer cluster IDs
        cell_labels_int = np.array(
            [name2ids[c] for c in adata.obs["celltype"].astype(str).values]
        )
        adata.obs["cluster_int"] = cell_labels_int
        logger.info(f"  Cluster mapping: {name2ids}")

        # -----------------------------------------------------------------
        # Step 4b: Compute pseudotime from trajectory
        # -----------------------------------------------------------------
        logger.info("[Step 4b] Computing pseudotime from trajectory ...")
        t0 = time.time()

        # Always compute geometric pseudotime first
        with Heartbeat(logger, "compute_pseudotime", interval=hb_interval):
            pseudotime_geometric = compute_pseudotime(
                features=adata.obsm["X_scvi"],
                cell_labels=cell_labels_int,
                Lineage_class=Lineage_class,
                start_node=start_node,
            )
        adata.obs["pseudotime_geometric"] = pseudotime_geometric
        logger.info(
            f"  pseudotime_geometric: min={pseudotime_geometric.min():.4f}  "
            f"max={pseudotime_geometric.max():.4f}  ({time.time() - t0:.1f}s)"
        )

        # Optionally apply q_score monotonicity correction
        if args.pseudotime_mode == "q_aware":
            q_mode = args.q_aware_strength
            logger.info(f"  Applying q_score monotonicity correction (strength={q_mode}) ...")
            t1 = time.time()
            with Heartbeat(logger, "compute_pseudotime_with_q", interval=hb_interval):
                pseudotime_corrected = compute_pseudotime_with_q(
                    features=adata.obsm["X_scvi"],
                    cell_labels=cell_labels_int,
                    Lineage_class=Lineage_class,
                    start_node=start_node,
                    q_score=adata.obs["q_score"].values,
                    mode=q_mode,
                    align_cluster=True,
                    shift_strength=0.3,
                )
            adata.obs["pseudotime_raw"] = pseudotime_corrected
            logger.info(
                f"  pseudotime_raw (q_aware): min={pseudotime_corrected.min():.4f}  "
                f"max={pseudotime_corrected.max():.4f}  ({time.time() - t1:.1f}s)"
            )
        else:
            adata.obs["pseudotime_raw"] = pseudotime_geometric
            logger.info("  pseudotime_mode=raw — using geometric pseudotime as pseudotime_raw")

        # Validate pseudotime quality; apply isotonic alignment ONLY as a
        # fallback when the inferred ordering is degenerate (>50% inversions).
        # This avoids circular reasoning: alignment forces the monotonicity
        # that downstream evaluation metrics measure.
        if args.celltype_order:
            ct_order_list = [t.strip() for t in args.celltype_order.split(",") if t.strip()]
            ct_order_map = {ct: i for i, ct in enumerate(ct_order_list)}
            cell_types_str = adata.obs["celltype"].astype(str).values
            known_types = sorted(
                set(cell_types_str) & set(ct_order_map.keys()),
                key=lambda x: ct_order_map[x],
            )
            if len(known_types) >= 3:
                pt = adata.obs["pseudotime_raw"].values.copy()
                # Current medians per type
                current_medians = {}
                for ct in known_types:
                    mask = cell_types_str == ct
                    current_medians[ct] = float(np.nanmedian(pt[mask]))

                # Count inversions (adjacent pairs where median order is wrong)
                medians_ordered = [current_medians[ct] for ct in known_types]
                n_inversions = sum(
                    1 for i in range(len(medians_ordered) - 1)
                    if medians_ordered[i] > medians_ordered[i + 1]
                )
                n_pairs = len(medians_ordered) - 1
                inversion_rate = n_inversions / n_pairs if n_pairs > 0 else 0.0
                logger.info(
                    f"  Pseudotime inversion check: {n_inversions}/{n_pairs} "
                    f"inversions ({inversion_rate:.1%})"
                )

                if inversion_rate > 0.5:
                    # Degenerate ordering — apply isotonic alignment as fallback
                    logger.warning(
                        f"  Inversion rate {inversion_rate:.1%} > 50%: "
                        f"applying isotonic alignment fallback ..."
                    )
                    from sklearn.isotonic import IsotonicRegression

                    x_order = np.array(
                        [ct_order_map[ct] for ct in known_types], dtype=float
                    )
                    y_medians = np.array([current_medians[ct] for ct in known_types])
                    ir = IsotonicRegression(increasing=True)
                    y_target = ir.fit_transform(x_order, y_medians)

                    for ct, target in zip(known_types, y_target):
                        mask = cell_types_str == ct
                        shift = target - current_medians[ct]
                        if abs(shift) > 1e-6:
                            logger.info(
                                f"    {ct}: median {current_medians[ct]:.4f} "
                                f"→ {target:.4f} (shift {shift:+.4f})"
                            )
                            pt[mask] += shift

                    pt = np.clip(pt, 0, 1)
                    adata.obs["pseudotime_raw"] = pt
                    logger.info("  Fallback isotonic alignment applied.")
                else:
                    logger.info(
                        "  Pseudotime ordering acceptable — no alignment applied."
                    )

        # Checkpoint after pseudotime
        # Store tree for later use (as dense in uns since sparse h5ad write can be tricky)
        adata.uns["_tree_dense"] = tree.toarray() if hasattr(tree, "toarray") else np.asarray(tree)
        # h5ad requires string keys in dicts — convert int keys to str
        adata.uns["_name2ids"] = {str(k): int(v) for k, v in name2ids.items()}
        adata.uns["_ids2name"] = {str(k): str(v) for k, v in ids2name.items()}
        ckpt.save(adata, 4, "pseudotime")
        log_memory(logger)

    # -------------------------------------------------------------------
    # Step 5: Compute kNN graph
    # -------------------------------------------------------------------
    if resume_from_step < 5:
        logger.info("[Step 5] Computing kNN graph ...")
        t0 = time.time()

        try:
            with Heartbeat(logger, "kNN-scanpy", interval=hb_interval):
                sc.pp.neighbors(adata, use_rep="X_scvi", key_added="scvi")
            logger.info(f"  sc.pp.neighbors (scanpy) succeeded  ({time.time() - t0:.1f}s)")
        except Exception as e:
            logger.warning(f"  sc.pp.neighbors failed ({e}); using pynndescent fallback ...")
            from pynndescent import NNDescent
            from scipy.sparse import csr_matrix

            n_neighbors = 15
            with Heartbeat(logger, "kNN-pynndescent", interval=hb_interval):
                nnd = NNDescent(
                    adata.obsm["X_scvi"],
                    n_neighbors=n_neighbors,
                    metric="euclidean",
                    random_state=42,
                )
                indices, distances = nnd.neighbor_graph

            n = adata.n_obs
            row = np.repeat(np.arange(n), n_neighbors)
            col = indices.ravel()
            data_ones = np.ones(len(row), dtype=np.float32)
            data_dist = distances.ravel().astype(np.float32)

            conn = csr_matrix((data_ones, (row, col)), shape=(n, n))
            conn = (conn + conn.T > 0).astype(float)
            dist_mat = csr_matrix((data_dist, (row, col)), shape=(n, n))

            adata.obsp["scvi_connectivities"] = conn
            adata.obsp["scvi_distances"] = dist_mat
            adata.uns["scvi"] = {
                "connectivities_key": "scvi_connectivities",
                "distances_key": "scvi_distances",
                "params": {
                    "n_neighbors": n_neighbors,
                    "method": "pynndescent",
                    "use_rep": "X_scvi",
                },
            }
            logger.info(f"  pynndescent fallback succeeded  ({time.time() - t0:.1f}s)")

        # Ensure default neighbors keys exist for CellRank
        if "connectivities" not in adata.obsp:
            adata.obsp["connectivities"] = adata.obsp.get(
                "scvi_connectivities", adata.obsp.get("connectivities")
            )
            adata.obsp["distances"] = adata.obsp.get(
                "scvi_distances", adata.obsp.get("distances")
            )
            adata.uns["neighbors"] = adata.uns.get("scvi", adata.uns.get("neighbors", {}))
        elif "scvi_connectivities" in adata.obsp:
            adata.obsp["connectivities"] = adata.obsp["scvi_connectivities"]
            adata.obsp["distances"] = adata.obsp["scvi_distances"]
            adata.uns["neighbors"] = adata.uns["scvi"]

        log_memory(logger)

    # -------------------------------------------------------------------
    # Step 6: Build three-kernel combined transition matrix
    # -------------------------------------------------------------------
    if resume_from_step < 6:
        logger.info("[Step 6] Building three-kernel combined transition matrix ...")
        t0 = time.time()

        # Recover tree from checkpoint storage if needed
        if "_tree_dense" in adata.uns:
            tree_arr = adata.uns["_tree_dense"]
        else:
            # tree variable should still be in scope from Step 4
            tree_arr = tree.toarray() if hasattr(tree, "toarray") else np.asarray(tree)

        mst_dist = shortest_path(tree_arr, directed=False)
        logger.info(f"  MST shortest-path matrix shape: {mst_dist.shape}")

        from btraj.kernels.kernel_combined import build_three_kernel

        q_threshold = float(adata.obs["q_score"].quantile(0.5))
        logger.info(f"  q_threshold (median q_score): {q_threshold:.4f}")

        with Heartbeat(logger, "build_three_kernel", interval=hb_interval):
            combined_kernel = build_three_kernel(
                adata,
                w_pt=0.5,
                w_bcr=0.3,
                w_conn=0.2,
                time_key="pseudotime_raw",
                clusters_key="cluster_int",
                mst_dist=mst_dist,
                xh_key="X_h",
                q_key="q_score",
                n_neighbors=30,
                q_threshold=q_threshold,
                constraint="soft",
                beta=10.0,
                lambda_skel=0.5,
                b=10.0,
                nu=0.5,
            )
        logger.info(f"  Combined kernel built  ({time.time() - t0:.1f}s)")

        # Store transition matrix in obsp for checkpoint
        if hasattr(combined_kernel, "transition_matrix"):
            from scipy.sparse import issparse
            T = combined_kernel.transition_matrix
            if issparse(T):
                adata.obsp["T_combined"] = T
            else:
                from scipy.sparse import csr_matrix as _csr
                adata.obsp["T_combined"] = _csr(T)
            logger.info(f"  Stored T_combined in obsp: {adata.obsp['T_combined'].shape}")

        ckpt.save(adata, 6, "three_kernel")
        log_memory(logger)

    # -------------------------------------------------------------------
    # Step 7: Fate probabilities
    # -------------------------------------------------------------------
    if resume_from_step < 7:
        fate_method = args.fate_method
        terminal_types = [t.strip() for t in args.terminal_types.split(",")]
        logger.info(f"[Step 7] Computing fate probabilities (method={fate_method}) ...")
        logger.info(f"  Terminal types: {terminal_types}")
        t0 = time.time()

        # If resuming from step 6 checkpoint, rebuild kernel from stored T_combined
        if resume_from_step >= 6 and "T_combined" in adata.obsp:
            logger.info("  Rebuilding kernel from stored T_combined")
            from cellrank.kernels import ConnectivityKernel
            ck = ConnectivityKernel(adata)
            ck.compute_transition_matrix()
            combined_kernel = ck
            combined_kernel._transition_matrix = adata.obsp["T_combined"]

        if fate_method == "direct":
            # FAST: bypass Schur decomposition, set terminal states from biology,
            # use CellRank's sparse GMRES for fate probabilities
            logger.info("  Method: direct (skip macrostates, sparse GMRES)")
            try:
                compute_fate_direct(adata, combined_kernel, terminal_types, logger, hb_interval)
            except Exception as e:
                logger.warning(f"  Direct method failed: {e}")
                logger.info("  Falling back to sparse_custom method ...")
                T = combined_kernel.transition_matrix
                compute_fate_sparse_custom(adata, T, terminal_types, logger, hb_interval)

        elif fate_method == "sparse_custom":
            # FASTEST: direct scipy.sparse.linalg solver, bypasses CellRank entirely
            logger.info("  Method: sparse_custom (direct LU solver)")
            T = combined_kernel.transition_matrix
            compute_fate_sparse_custom(adata, T, terminal_types, logger, hb_interval)

        elif fate_method == "gpcca":
            # SLOW: full GPCCA with Schur decomposition
            logger.warning("  Method: gpcca (SLOW — requires dense Schur decomposition without petsc4py)")
            compute_fate_gpcca_full(adata, combined_kernel, logger, hb_interval)

        logger.info(f"  Fate probabilities complete  ({time.time() - t0:.1f}s)")
        ckpt.save(adata, 7, "fate_probs")
        log_memory(logger)

    # -------------------------------------------------------------------
    # Step 7b: Extended fate probabilities (optional)
    # -------------------------------------------------------------------
    if args.terminal_types_extended:
        ext_types = [t.strip() for t in args.terminal_types_extended.split(",") if t.strip()]
        # Validate all types exist in data
        available_types = set(adata.obs["celltype"].astype(str).unique())
        missing = [t for t in ext_types if t not in available_types]
        if missing:
            logger.warning(f"[Step 7b] Skipping — types not found in data: {missing}")
        else:
            logger.info(f"[Step 7b] Computing extended fate probabilities ({len(ext_types)} terminal types) ...")
            logger.info(f"  Extended terminal types: {ext_types}")
            t0 = time.time()

            # Rebuild kernel if needed (same pattern as Step 7)
            if resume_from_step >= 6 and "T_combined" in adata.obsp:
                logger.info("  Rebuilding kernel from stored T_combined")
                from cellrank.kernels import ConnectivityKernel
                ck = ConnectivityKernel(adata)
                ck.compute_transition_matrix()
                combined_kernel = ck
                combined_kernel._transition_matrix = adata.obsp["T_combined"]

            fate_method = args.fate_method
            if fate_method == "direct":
                try:
                    compute_fate_direct(adata, combined_kernel, ext_types, logger,
                                        hb_interval, col_prefix="prob_ext_")
                except Exception as e:
                    logger.warning(f"  Direct method failed for extended types: {e}")
                    logger.info("  Falling back to sparse_custom method ...")
                    T = combined_kernel.transition_matrix
                    compute_fate_sparse_custom(adata, T, ext_types, logger,
                                              hb_interval, col_prefix="prob_ext_")
            elif fate_method == "sparse_custom":
                T = combined_kernel.transition_matrix
                compute_fate_sparse_custom(adata, T, ext_types, logger,
                                          hb_interval, col_prefix="prob_ext_")
            elif fate_method == "gpcca":
                logger.warning("  Extended types not supported with gpcca method — skipping")

            logger.info(f"  Extended fate probabilities complete  ({time.time() - t0:.1f}s)")
            ckpt.save(adata, 7, "fate_probs_extended")
            log_memory(logger)

    # -------------------------------------------------------------------
    # Step 8: Integrate SOTA pseudotime (if provided)
    # -------------------------------------------------------------------
    if args.pseudotime_csv:
        logger.info("[Step 8] Integrating SOTA pseudotime comparison ...")
        integrate_pseudotime_comparison(adata, args.pseudotime_csv, logger)

    # -------------------------------------------------------------------
    # Step 9: Save final results
    # -------------------------------------------------------------------
    logger.info(f"[Step 9] Saving results to {out_path} ...")
    t0 = time.time()

    # Clean up internal checkpoint keys from uns before final save
    for key in ["_tree_dense", "_name2ids", "_ids2name"]:
        adata.uns.pop(key, None)

    adata.write_h5ad(out_path)
    logger.info(f"  Saved  ({time.time() - t0:.1f}s)")

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------
    total_time = time.time() - pipeline_start
    peak_rss = log_memory(logger)
    logger.info("=" * 70)
    logger.info("Pipeline complete.")
    logger.info(f"  Total time : {total_time:.1f}s")
    logger.info(f"  Peak RSS   : {peak_rss:.0f} MB")
    logger.info(f"  Output file: {out_path}")
    logger.info(f"  adata.obs columns: {list(adata.obs.columns)}")
    logger.info(f"  adata.obsm keys  : {list(adata.obsm.keys())}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
