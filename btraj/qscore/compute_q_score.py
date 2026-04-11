"""
BCR Quality Score Computation Module.

This module provides comprehensive quality scoring for B-cell receptor (BCR) data,
including technical quality (q_tech_bcr), biological quality (q_bio_bcr), and
combined quality score (q_score) with two aggregation strategies.

Design Principles
-----------------
- No arbitrary imputation for missing evidence; use available evidence only.
- Adaptive weighting based on effective information content (IQR + saturation).
- Robust scaling using quantiles to avoid batch-specific thresholds.
- Modular architecture: compute_tech → compute_bio → aggregate_score.

Sections
--------
1. Utility functions (internal helpers)
2. q_tech_bcr computation (from Cell Ranger contig annotations)
3. q_bio_bcr computation (biological features: isotype, SHM, clonality)
4. q_score aggregation (two variants: per-group adaptive vs. global simple)

"""

import os
import sys
import glob
import warnings
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData

warnings.filterwarnings("ignore", category=FutureWarning)


# ============================================================================
# Section 1: Utility Functions (Internal)
# ============================================================================

def _clip01(x: float) -> float:
    """Clip a scalar to [0, 1]."""
    return float(max(0.0, min(1.0, x)))


def _safe_float(x) -> float:
    """Safely convert a value to float; return NaN on failure."""
    if x is None:
        return np.nan
    try:
        x = float(x)
        if np.isnan(x):
            return np.nan
        return x
    except Exception:
        return np.nan


def _safe_bool(x) -> bool:
    """Safely convert a value to bool; return False on failure or ambiguous."""
    if pd.isna(x):
        return False
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        try:
            return bool(int(x))
        except Exception:
            return False
    s = str(x).strip().lower()
    return s in ['true', '1', 't', 'yes', 'y']


def _robust_minmax_scalar(x: float, lo: float, hi: float, eps: float = 1e-8) -> float:
    """Apply robust min-max scaling to a scalar and clip to [0, 1]."""
    if np.isnan(x):
        return np.nan
    return float(np.clip((x - lo) / (hi - lo + eps), 0.0, 1.0))


def _robust_minmax(x: np.ndarray, lo_q: float = 1, hi_q: float = 99, eps: float = 1e-8) -> np.ndarray:
    """Apply robust min-max scaling to an array using lower/upper quantiles."""
    x = np.asarray(x, dtype=float)
    lo, hi = np.nanpercentile(x, lo_q), np.nanpercentile(x, hi_q)
    y = (x - lo) / (hi - lo + eps)
    return np.clip(y, 0.0, 1.0)


def _iqr(x: np.ndarray) -> float:
    """Compute interquartile range."""
    q1, q3 = np.nanpercentile(x, 25), np.nanpercentile(x, 75)
    return float(q3 - q1)


def _mono_amplify_pow(x: np.ndarray, gamma: float = 2.0, eps: float = 1e-12) -> np.ndarray:
    """
    Monotonic contrast amplification using a power transform.

    gamma > 1 increases contrast in the high-value region while keeping output in [0, 1].
    """
    x = np.clip(np.asarray(x, float), 0.0, 1.0)
    if abs(gamma - 1.0) < eps:
        return x
    return x ** gamma


def _mono_amplify_logit(x: np.ndarray, k: float = 6.0, m: float = 0.8, eps: float = 1e-6) -> np.ndarray:
    """
    Monotonic contrast amplification using a logit-sigmoid transform.
    """
    x = np.clip(np.asarray(x, float), eps, 1 - eps)
    z = np.log(x / (1 - x))
    z0 = np.log(m / (1 - m))
    return 1.0 / (1.0 + np.exp(-k * (z - z0)))


def _smooth_alpha(eff_t: float, eff_b: float, k: float = 200.0, eps: float = 1e-8) -> Tuple[float, float]:
    """
    Compute smooth alpha/beta weights from effective information content.

    Parameters
    ----------
    eff_t : float
        Effective information content of q_tech.
    eff_b : float
        Effective information content of q_bio.
    k : float
        Steepness of the sigmoid transition.

    Returns
    -------
    tuple
        (alpha, beta), both clipped to [0.02, 0.98].
    """
    ratio = (eff_t - eff_b) / (eff_t + eff_b + eps)
    exponent = np.clip(-k * ratio, -60, 60)
    alpha = 1.0 / (1.0 + np.exp(exponent))
    alpha = float(np.clip(alpha, 0.02, 0.98))
    beta = 1.0 - alpha
    return alpha, beta


def stretch_01(x: np.ndarray, method: str = "power", **kwargs) -> np.ndarray:
    """
    Apply a monotonic transform to values in [0, 1].

    Supported methods:
    - 'power': Power transform (gamma > 1 increases high-end contrast).
    - 'root': Root transform (gamma > 1 compresses high-end).
    - 'logit': Logit-sigmoid transform.
    - 'sigmoid': Sigmoid transform.
    - 'quantile': Quantile ranking (uniform distribution).

    Parameters
    ----------
    x : np.ndarray
        Input values in [0, 1].
    method : str
        Transform method name.
    **kwargs : dict
        Method-specific parameters (gamma, k, m, etc.).

    Returns
    -------
    np.ndarray
        Transformed values in [0, 1].
    """
    x = np.asarray(x, float)
    x = np.clip(x, 0.0, 1.0)

    if method == "power":
        gamma = kwargs.get("gamma", 2.0)
        return x ** gamma

    if method == "root":
        gamma = kwargs.get("gamma", 2.0)
        return x ** (1.0 / gamma)

    if method == "logit":
        k = kwargs.get("k", 6.0)
        m = kwargs.get("m", float(np.nanmedian(x)))
        eps = kwargs.get("eps", 1e-6)
        z = np.log(np.clip(x, eps, 1 - eps) / np.clip(1 - x, eps, 1 - eps))
        z0 = np.log(np.clip(m, eps, 1 - eps) / np.clip(1 - m, eps, 1 - eps))
        return 1.0 / (1.0 + np.exp(-k * (z - z0)))

    if method == "sigmoid":
        k = kwargs.get("k", 10.0)
        m = kwargs.get("m", float(np.nanmedian(x)))
        return 1.0 / (1.0 + np.exp(-k * (x - m)))

    if method == "quantile":
        order = np.argsort(x)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.linspace(0, 1, len(x))
        return ranks

    raise ValueError(f"Unknown stretch method: {method}")


# ============================================================================
# Section 2: q_tech_bcr Computation (from Cell Ranger contig annotations)
# ============================================================================

# Global state for adaptive support scaling across samples
_SUPPORT_SCALER = {'p05': 0.0, 'p95': 1.0}


def fit_support_scaler(df: pd.DataFrame) -> None:
    """
    Fit global scaler for UMI/reads using log1p transform and 5-95% quantiles.

    This ensures adaptive scaling across different sequencing depths and batches,
    avoiding fixed thresholds (e.g., ">6 UMI") that suffer from batch effects.

    Parameters
    ----------
    df : pd.DataFrame
        Concatenated contig annotations from all samples.
    """
    global _SUPPORT_SCALER

    if df is None or df.empty:
        return

    # Prefer UMIs, fall back to reads
    x = pd.to_numeric(df.get('umis'), errors='coerce')
    if x is None or x.isna().all():
        x = pd.to_numeric(df.get('reads'), errors='coerce')

    x = x.dropna()
    if len(x) == 0:
        return

    lx = np.log1p(x.astype(float))
    p05, p95 = np.nanpercentile(lx, [5, 95])

    if not np.isfinite(p05):
        p05 = 0.0
    if not np.isfinite(p95) or p95 <= p05:
        p95 = p05 + 1.0

    _SUPPORT_SCALER['p05'] = float(p05)
    _SUPPORT_SCALER['p95'] = float(p95)


def _support_log1p(row: pd.Series) -> float:
    """Extract log1p(UMI or reads) from a contig row."""
    x = row.get('umis')
    if pd.isna(x):
        x = row.get('reads')
    if pd.isna(x):
        return 0.0
    try:
        return float(np.log1p(float(x)))
    except Exception:
        return 0.0


def _support_scaled_from_log(lx: float) -> float:
    """Scale log-support value to [0, 1] using fitted global scaler."""
    p05, p95 = _SUPPORT_SCALER['p05'], _SUPPORT_SCALER['p95']
    z = (lx - p05) / (p95 - p05)
    return float(max(0.0, min(1.0, z)))


def contig_quality_score(row: pd.Series, req_cols: Optional[List[str]] = None) -> float:
    """
    Compute technical quality score for a single contig (0-1).

    Design weights:
    - high_confidence: 0.45 (Cell Ranger assembly/annotation confidence)
    - is_cell: 0.20 (barcode is a real cell)
    - productive: 0.15 (no stop codon, in-frame)
    - full_length: 0.10 (complete V to C)
    - cdr_fwr_missing: 0.05 (all 7 regions present: FWR1-3, CDR1-3, FWR4)
    - support (UMI/reads): 0.05 (depth of coverage, adaptively scaled)

    Parameters
    ----------
    row : pd.Series
        Single contig annotation row.
    req_cols : list, optional
        Required CDR/FWR columns to check for completeness.

    Returns
    -------
    float
        Quality score in [0, 1].
    """
    if req_cols is None:
        req_cols = ['fwr1', 'cdr1', 'fwr2', 'cdr2', 'fwr3', 'cdr3', 'fwr4']

    # Core quality flags
    hc = 1.0 if _safe_bool(row.get('high_confidence')) else 0.0
    ic = 1.0 if _safe_bool(row.get('is_cell')) else 0.0
    pr = 1.0 if _safe_bool(row.get('productive')) else 0.0
    fl = 1.0 if _safe_bool(row.get('full_length')) else 0.0

    # Region completeness
    cdr_fwr_missing = any(pd.isna(row.get(c)) for c in req_cols if c in row.index)
    ok = 0.0 if cdr_fwr_missing else 1.0

    # Support depth (adaptive scaling)
    lx = _support_log1p(row)
    sup = _support_scaled_from_log(lx)

    # Weighted sum (weights sum to 1.0)
    q = (
            0.45 * hc +
            0.20 * ic +
            0.15 * pr +
            0.10 * fl +
            0.05 * ok +
            0.05 * sup
    )
    return float(max(0.0, min(1.0, q)))


def pick_one_chain(group: pd.DataFrame) -> Optional[pd.Series]:
    """
    Select the best contig for a chain (IGH/IGK/IGL) within a cell.

    Selection priority:
    1. Higher contig_quality_score
    2. Higher support (log1p UMI/reads)

    Parameters
    ----------
    group : pd.DataFrame
        All contigs for one chain in one cell.

    Returns
    -------
    pd.Series or None
        Best contig row, or None if group is empty.
    """
    if group.empty:
        return None

    g = group.copy()
    g['_q'] = g.apply(contig_quality_score, axis=1)
    g['_lx'] = g.apply(_support_log1p, axis=1)
    g = g.sort_values(by=['_q', '_lx'], ascending=False, kind='mergesort')

    best = g.iloc[0].drop(labels=['_q', '_lx'])
    return best


def summarize_cell_bcr(
        cell_contigs: pd.DataFrame,
        req_cols: Optional[List[str]] = None
) -> Tuple[Dict[str, Any], Optional[pd.Series], Optional[pd.Series]]:
    """
    Aggregate cell-level BCR technical quality from all contigs.

    Computes:
    - q_tech_bcr: Overall technical quality score (0-1)
    - paired_bcr: Whether both heavy and light chains are present
    - has_heavy/light: Chain presence flags
    - n_contigs: Counts of total and productive contigs per chain
    - penalty_multichain: Penalty for multiple productive chains (doublet risk)

    Parameters
    ----------
    cell_contigs : pd.DataFrame
        All contigs for one cell.
    req_cols : list, optional
        Required columns for region completeness check.

    Returns
    -------
    tuple
        (metadata_dict, heavy_best_row, light_best_row)
    """
    # Split by chain
    heavy_all = cell_contigs[cell_contigs['chain'] == 'IGH']
    light_all = cell_contigs[cell_contigs['chain'].isin(['IGK', 'IGL'])]

    # Pick best per chain
    heavy_best = pick_one_chain(heavy_all) if len(heavy_all) else None
    light_best = pick_one_chain(light_all) if len(light_all) else None

    has_heavy = heavy_best is not None
    has_light = light_best is not None
    paired = has_heavy and has_light

    # Quality of best contigs
    def qcontig(r):
        return contig_quality_score(r, req_cols) if r is not None else 0.0

    qH = qcontig(heavy_best)
    qL = qcontig(light_best)

    # Pairing term: full credit for paired, partial for single, zero for none
    pair_term = 1.0 if paired else (0.55 if (has_heavy or has_light) else 0.0)

    # Quality term: average if paired, else best available
    qual_term = (qH + qL) / 2.0 if paired else max(qH, qL)

    # Support term: similarly averaged
    def sup_scaled(r):
        if r is None:
            return 0.0
        return _support_scaled_from_log(_support_log1p(r))

    sup_term = (sup_scaled(heavy_best) + sup_scaled(light_best)) / 2.0 if paired else max(sup_scaled(heavy_best),
                                                                                          sup_scaled(light_best))

    # Count productive contigs (for multi-chain penalty)
    def count_productive(df):
        if df is None or len(df) == 0 or 'productive' not in df.columns:
            return 0
        return int(df['productive'].apply(_safe_bool).sum())

    n_prod_heavy = count_productive(heavy_all)
    n_prod_light = count_productive(light_all)

    # Multi-chain penalty: >1 productive IGH is penalized more heavily (0.55^(n-1))
    # >1 productive light chain: 0.75^(n-1)
    pen_H = 1.0 if n_prod_heavy <= 1 else (0.55 ** (n_prod_heavy - 1))
    pen_L = 1.0 if n_prod_light <= 1 else (0.75 ** (n_prod_light - 1))
    penalty = float(pen_H * pen_L)

    # Final score: pairing is primary, quality secondary, support minor
    q = penalty * (0.60 * pair_term + 0.35 * qual_term + 0.05 * sup_term)
    q = float(max(0.0, min(1.0, q)))

    meta = {
        'paired_bcr': paired,
        'q_tech_bcr': q,
        'has_heavy': has_heavy,
        'has_light': has_light,
        'n_IGH_contigs': int(len(heavy_all)),
        'n_IGKIGL_contigs': int(len(light_all)),
        'n_prod_IGH': int(n_prod_heavy),
        'n_prod_IGKIGL': int(n_prod_light),
        'penalty_multichain': penalty
    }

    return meta, heavy_best, light_best


def compute_q_tech_bcr(
        adata: AnnData,
        bcr_path: str,
        donor_id_col: str = "donor_id",
        cell_id_col: str = "cell_id",
        req_cols: Optional[List[str]] = None,
        output_prefix: str = ""
) -> pd.DataFrame:
    """
    Compute q_tech_bcr for all cells from Cell Ranger contig annotations.

    This is the main entry point for technical quality score calculation.
    It reads all_contig_annotations.csv files, fits adaptive scalers,
    computes per-cell quality metrics, and merges results into adata.obs.

    Parameters
    ----------
    adata : AnnData
        Single-cell data object with cell metadata.
    bcr_path : str
        Directory containing Cell Ranger VDJ output subdirectories.
    donor_id_col : str
        Column in adata.obs containing donor/sample identifiers.
    cell_id_col : str
        Column to use for cell barcodes (will be constructed as donor_id + '-' + barcode).
    req_cols : list, optional
        Required CDR/FWR columns for completeness check.
    output_prefix : str
        Prefix for output column names (e.g., "" for q_tech_bcr, or "raw_" for raw_q_tech_bcr).

    Returns
    -------
    pd.DataFrame         with all computed BCR columns (indexed by cell_id).
    """
    if req_cols is None:
        req_cols = ['fwr1', 'cdr1', 'fwr2', 'cdr2', 'fwr3', 'cdr3', 'fwr4']

    # Ensure required columns exist in contig files
    required_contig_cols = [
                               'productive', 'is_cell', 'high_confidence', 'full_length',
                               'umis', 'reads', 'chain', 'v_gene', 'd_gene', 'j_gene', 'c_gene', 'cdr3'
                           ] + req_cols

    def read_one_sample(donor_id: str) -> Optional[pd.DataFrame]:
        """Read and preprocess one sample's contig annotations."""
        csv_path = os.path.join(bcr_path, f"{donor_id}_all_contig_annotations.csv")
        if not os.path.isfile(csv_path):
            return None

        df = pd.read_csv(csv_path)

        # Ensure all required columns exist
        for col in required_contig_cols:
            if col not in df.columns:
                df[col] = np.nan

        # Mark CDR/FWR completeness
        df['cdr_fwr_missing'] = df[req_cols].isna().any(axis=1)

        # Assemble full amino acid sequence
        df['full_aa'] = (
                df['fwr1'].fillna('') + df['cdr1'].fillna('') +
                df['fwr2'].fillna('') + df['cdr2'].fillna('') +
                df['fwr3'].fillna('') + df['cdr3'].fillna('') +
                df['fwr4'].fillna('')
        )

        # Create cell_id with donor prefix
        df['cell_id'] = f"{donor_id}-" + df['barcode'].astype(str)

        # Select relevant columns
        keep = [
            'cell_id', 'barcode', 'chain', 'v_gene', 'd_gene', 'j_gene', 'c_gene',
            'cdr3', 'full_aa', 'cdr_fwr_missing',
            'productive', 'is_cell', 'high_confidence', 'full_length', 'umis', 'reads'
        ]
        keep = [c for c in keep if c in df.columns]

        return df[keep].copy()

    # Read all samples
    all_dfs = []
    for donor_id in adata.obs[donor_id_col].unique():
        tmp = read_one_sample(donor_id)
        if tmp is not None:
            all_dfs.append(tmp)

    if not all_dfs:
        raise ValueError(f"No all_contig_annotations.csv files found in {bcr_path}")

    # Concatenate and fit global scaler
    big = pd.concat(all_dfs, ignore_index=True)
    fit_support_scaler(big)

    # Process each cell
    rows = []
    for cell_id, g in big.groupby('cell_id'):
        meta, heavy_best, light_best = summarize_cell_bcr(g, req_cols)

        row = {'cell_id': cell_id, **meta}

        # Add heavy chain info
        if heavy_best is not None:
            row.update({
                f'{output_prefix}Heavy': heavy_best.get('full_aa', ''),
                f'{output_prefix}cdrh3': heavy_best.get('cdr3', ''),
                f'{output_prefix}ighv': heavy_best.get('v_gene', ''),
                f'{output_prefix}ighj': heavy_best.get('j_gene', ''),
                f'{output_prefix}ighc': heavy_best.get('c_gene', ''),
                f'{output_prefix}cdr_fwr_missing_H': bool(heavy_best.get('cdr_fwr_missing', True))
            })
        else:
            row.update({
                f'{output_prefix}Heavy': '',
                f'{output_prefix}cdrh3': '',
                f'{output_prefix}ighv': '',
                f'{output_prefix}ighj': '',
                f'{output_prefix}ighc': '',
                f'{output_prefix}cdr_fwr_missing_H': True
            })

        # Add light chain info
        if light_best is not None:
            row.update({
                f'{output_prefix}Light': light_best.get('full_aa', ''),
                f'{output_prefix}cdrl3': light_best.get('cdr3', ''),
                f'{output_prefix}iglv': light_best.get('v_gene', ''),
                f'{output_prefix}iglj': light_best.get('j_gene', ''),
                f'{output_prefix}iglc': light_best.get('c_gene', ''),
                f'{output_prefix}cdr_fwr_missing_L': bool(light_best.get('cdr_fwr_missing', True))
            })
        else:
            row.update({
                f'{output_prefix}Light': '',
                f'{output_prefix}cdrl3': '',
                f'{output_prefix}iglv': '',
                f'{output_prefix}iglj': '',
                f'{output_prefix}iglc': '',
                f'{output_prefix}cdr_fwr_missing_L': True
            })

        rows.append(row)

    cell_bcr = pd.DataFrame(rows)

    # Merge into adata.obs
    orig_idx = adata.obs_names
    new_obs = (adata.obs
               .reset_index()
               .merge(cell_bcr, on='cell_id', how='left', suffixes=('', '_df'))
               .set_index('index')
               .rename_axis(None))
    new_obs = new_obs.loc[orig_idx]
    adata.obs = new_obs

    return cell_bcr


# ============================================================================
# Section 3: q_bio_bcr Computation (Biological Features)
# ============================================================================

def _is_switched_isotype(c_gene: str) -> float:
    """
    Map isotype class to a binary switched/non-switched score.

    Returns
    -------
    float
        1.0 for switched isotypes (IGHG/IGHA/IGHE),
        0.0 for unswitched isotypes (IGHM/IGHD),
        NaN if unavailable or unrecognized.
    """
    if c_gene is None or (isinstance(c_gene, float) and np.isnan(c_gene)):
        return np.nan

    s = str(c_gene).upper()
    if "IGHG" in s or "IGHA" in s or "IGHE" in s:
        return 1.0
    if "IGHM" in s or "IGHD" in s:
        return 0.0
    return np.nan


def fit_shm_scaler_from_shm(
        cell_df: pd.DataFrame,
        shm_col: str = "SHM",
        q: Tuple[int, int] = (5, 95),
) -> Dict[str, Any]:
    """
    Fit a robust scaler for SHM values using quantiles.

    Used when no V-region identity column is available.

    Parameters
    ----------
    cell_df : pd.DataFrame
        Cell metadata DataFrame.
    shm_col : str
        Column name for somatic hypermutation rate.
    q : tuple
        Lower and upper quantiles for scaling.

    Returns
    -------
    dict
        {"available": bool, "col": str, "lo": float, "hi": float}
    """
    if shm_col not in cell_df.columns:
        return {"available": False}

    shm = (
        pd.to_numeric(cell_df[shm_col], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    if len(shm) < 50:
        return {"available": False}

    lo, hi = np.nanpercentile(shm, q)
    if not np.isfinite(lo):
        lo = 0.0
    if not np.isfinite(hi) or hi <= lo:
        hi = lo + 1e-6

    return {
        "available": True,
        "col": shm_col,
        "lo": float(lo),
        "hi": float(hi),
    }


def _shm_score(
        cell_row: pd.Series,
        *,
        shm_col: str = "SHM",
        v_identity_col: Optional[str] = None,
        shm_scaler: Optional[Dict] = None,
) -> float:
    """
    Compute an SHM-based score in [0, 1].

    Priority:
    1. If v_identity_col is provided and scaler is available, use (1 - v_identity).
    2. Otherwise, use the raw SHM column with the fitted scaler.

    Returns NaN if no valid scaler/evidence is available.
    """
    if v_identity_col is not None and shm_scaler and shm_scaler.get("available", False):
        vid = _safe_float(cell_row.get(v_identity_col, np.nan))
        if np.isnan(vid):
            return np.nan
        shm = 1.0 - vid
        return _robust_minmax_scalar(shm, shm_scaler["lo"], shm_scaler["hi"])

    x = _safe_float(cell_row.get(shm_col, np.nan))
    if shm_scaler and shm_scaler.get("available", False):
        return _robust_minmax_scalar(x, shm_scaler["lo"], shm_scaler["hi"])

    return np.nan


def fit_clone_scaler(
        cell_df: pd.DataFrame,
        clone_id_col: str = "clone_id",
        q: int = 95,
) -> Dict[str, Any]:
    """
    Derive clone sizes from clone IDs and fit a robust upper bound for scaling.

    Missing clone IDs and "No_contig" entries are treated as singleton clones
    with size = 1.

    Parameters
    ----------
    cell_df : pd.DataFrame
        Cell metadata DataFrame.
    clone_id_col : str
        Column name for clone identifiers.
    q : int
        Quantile for upper bound.

    Returns
    -------
    dict
        {"available": bool, "hi": float}
    """
    if clone_id_col not in cell_df.columns:
        return {"available": False}

    clone_s = (
        cell_df[clone_id_col]
        .astype(str)
        .replace({"No_contig": np.nan, "nan": np.nan})
    )

    clone_size = (
        clone_s.to_frame("clone_id")
        .groupby("clone_id")["clone_id"]
        .transform("size")
    )
    clone_size = clone_size.fillna(1)

    cs = (
        pd.to_numeric(clone_size, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    if len(cs) == 0:
        return {"available": False}

    hi = float(np.nanpercentile(cs, q))
    if not np.isfinite(hi) or hi <= 1:
        hi = 1.0

    return {"available": True, "hi": hi}


def _clonal_score(clone_size: float, clone_scaler: Optional[Dict]) -> float:
    """
    Compute a log-scaled clonal expansion score in [0, 1].
    """
    if not clone_scaler or not clone_scaler.get("available", False):
        return np.nan

    cs = _safe_float(clone_size)
    if np.isnan(cs) or cs < 1:
        return 0.0

    hi = clone_scaler["hi"]
    return _clip01(np.log1p(cs) / np.log1p(hi))


def compute_q_bio_bcr_cell_v2(
        cell_row: pd.Series,
        *,
        q_tech_bcr: Optional[float] = None,
        igh_c_gene_col: str = "isotype",
        shm_col: str = "SHM",
        v_identity_col: Optional[str] = None,
        clone_size_col: str = "clone_size",
        shm_scaler: Optional[Dict] = None,
        clone_scaler: Optional[Dict] = None,
        weights: Optional[Dict[str, float]] = None,
        tech_gate_power: float = 1.0,
        coverage_power: float = 1.0,
) -> float:
    """
    Compute a biologically informed BCR quality score for a single cell.

    Design Principles
    -----------------
    - No arbitrary imputation for missing evidence.
    - Use weighted averaging over available evidence only.
    - Penalize missingness through a coverage multiplier.
    - Gate the final score by the technical BCR quality score.

    Components
    ----------
    - Isotype switching (switched vs. unswitched).
    - Somatic hypermutation (SHM) level.
    - Clonal expansion (clone size).

    Parameters
    ----------
    cell_row : pd.Series
        Single cell metadata row.
    q_tech_bcr : float, optional
        Technical quality score (gates the final score).
    igh_c_gene_col : str
        Column for constant region gene (isotype).
    shm_col : str
        Column for SHM rate.
    v_identity_col : str, optional
        Column for V-region identity (alternative to SHM).
    clone_size_col : str
        Column for clone size.
    shm_scaler : dict
        Fitted scaler for SHM values.
    clone_scaler : dict
        Fitted scaler for clone sizes.
    weights : dict
        Weights for iso/shm/clonal components.
    tech_gate_power : float
        Exponent for q_tech_gating (1.0 = linear, >1 = stricter).
    coverage_power : float
        Exponent for coverage penalty (0 = no penalty).

    Returns
    -------
    float
        q_bio_bcr score in [0, 1].
    """
    if weights is None:
        weights = {"iso": 0.45, "shm": 0.45, "clonal": 0.10}

    # Compute component scores
    s_iso = _is_switched_isotype(cell_row.get(igh_c_gene_col, None))
    s_shm = _shm_score(
        cell_row,
        shm_col=shm_col,
        v_identity_col=v_identity_col,
        shm_scaler=shm_scaler,
    )
    s_clonal = _clonal_score(cell_row.get(clone_size_col, np.nan), clone_scaler)

    # Weighted average of available components
    num, den = 0.0, 0.0
    for key, score in [("iso", s_iso), ("shm", s_shm), ("clonal", s_clonal)]:
        w = float(weights.get(key, 0.0))
        if w <= 0:
            continue
        if score is None or (isinstance(score, float) and np.isnan(score)):
            continue
        num += w * float(score)
        den += w

    if den == 0:
        q_core = 0.0
        coverage = 0.0
    else:
        q_core = num / den
        coverage = den / sum(weights.values())

    # Get technical quality for gating
    if q_tech_bcr is None:
        q_tech_bcr = cell_row.get("q_tech_bcr", 1.0)

    qt = _safe_float(q_tech_bcr)
    if np.isnan(qt):
        qt = 1.0
    qt = _clip01(qt)

    # Final score: tech-gated, coverage-penalized
    qbio = (qt ** tech_gate_power) * q_core * (coverage ** coverage_power)
    return _clip01(qbio)


def compute_q_bio_bcr(
        adata: AnnData,
        isotype_col: str = "isotype",
        shm_col: str = "SHM",
        clone_id_col: str = "clone_id",
        v_identity_col: Optional[str] = None,
        weights: Optional[Dict[str, float]] = None,
        tech_gate_power: float = 1.0,
        coverage_power: float = 0.0,
        out_key: str = "q_bio_bcr"
) -> np.ndarray:
    """
    Compute q_bio_bcr for all cells in AnnData.

    This is a convenience wrapper that fits scalers and applies
    compute_q_bio_bcr_cell_v2 to all cells.

    Parameters
    ----------
    adata : AnnData
        Single-cell data object.
    isotype_col : str
        Column for isotype (constant region gene).
    shm_col : str
        Column for SHM rate.
    clone_id_col : str
        Column for clone identifiers.
    v_identity_col : str, optional
        Alternative column for V-region identity.
    weights : dict
        Weights for iso/shm/clonal components.
    tech_gate_power : float
        Gating exponent for q_tech_bcr.
    coverage_power : float
        Coverage penalty exponent.
    out_key : str
        Output column name in adata.obs.

    Returns
    -------
    np.ndarray
        Array of q_bio_bcr scores.
    """
    cell_df = adata.obs.copy()

    # Fit scalers
    shm_scaler = fit_shm_scaler_from_shm(cell_df, shm_col=shm_col, q=(1, 100))
    clone_scaler = fit_clone_scaler(cell_df, clone_id_col=clone_id_col, q=95)

    # Compute clone sizes if not present
    if "clone_size" not in cell_df.columns:
        clone_s = (
            cell_df[clone_id_col]
            .astype(str)
            .replace({"No_contig": np.nan, "nan": np.nan})
        )
        clone_size = (
            clone_s.to_frame("clone_id")
            .groupby("clone_id")["clone_id"]
            .transform("size")
        )
        cell_df["clone_size"] = clone_size.fillna(1)

    # Compute per-cell scores
    scores = cell_df.apply(
        lambda r: compute_q_bio_bcr_cell_v2(
            r,
            q_tech_bcr=r.get("q_tech_bcr"),
            igh_c_gene_col=isotype_col,
            shm_col=shm_col,
            v_identity_col=v_identity_col,
            shm_scaler=shm_scaler,
            clone_scaler=clone_scaler,
            weights=weights,
            tech_gate_power=tech_gate_power,
            coverage_power=coverage_power
        ),
        axis=1
    )

    adata.obs[out_key] = scores.reindex(adata.obs.index)
    return adata.obs[out_key].values


# ============================================================================
# Section 4: q_score Aggregation (Two Variants)
# ============================================================================

"""
Q_SCORE COMPUTATION: TWO VARIANTS
==================================

【Variant A: compute_q_score】Per-Group Adaptive Weighting
-----------------------------------------------------------
- Computes adaptive weights (alpha/beta) separately for each group.
- Groups defined by dataset_key (e.g., different batches/donors).
- Supports non-linear stretch_method transforms on q_bio.
- Supports contrast amplification (mono_amplify_pow/logit).
- More complex, suitable for multi-batch data integration.

Key features:
1. Per-group effective information content (IQR * (1 - saturation)).
2. Smooth sigmoid weighting between q_tech and q_bio.
3. Contrast amplification of the more informative component.
4. Optional minimum component weight constraints.

【Variant B: compute_q_score_v2】Global Simple Weighting
--------------------------------------------------------
- Computes single set of weights for entire dataset.
- No group-wise variation.
- No contrast amplification.
- No stretch_method application (assumes linear input).
- Faster and simpler, suitable for single-batch analysis.

Key differences:
1. Global vs. per-group weighting.
2. No stretch transforms.
3. No contrast amplification.
4. Simpler fallback logic (if alpha>0.9, use q_tech only; if beta>0.9, use q_bio only).

CHOOSING BETWEEN VARIANTS:
- Use compute_q_score for multi-batch data where technical quality varies by batch.
- Use compute_q_score_v2 for single, homogeneous dataset where simplicity is preferred.
"""


def compute_q_score(
        adata: AnnData,
        q_tech_key: str = "q_tech_bcr",
        q_bio_key: str = "q_bio_bcr",
        dataset_key: Optional[str] = None,
        sat_thr: float = 0.98,
        eps: float = 1e-3,
        min_component_weight: Optional[float] = None,
        out_q_score_key: str = "q_score",
        out_debug_key: str = "q_score_debug",
        stretch_method: Optional[str] = "root_g2",
        linear_with_raw_qbio: bool = False,
        amplify_gamma: float = 2.0,
        amplify_strength: float = 1.0,
        kernel_weighting: float = 200.0,
) -> np.ndarray:
    """
    Compute final q_score by adaptively combining q_tech and q_bio (Variant A).

    This is the full-featured version with per-group adaptive weighting,
    non-linear transforms, and contrast amplification.

    Parameters
    ----------
    adata : AnnData
        Single-cell data object.
    q_tech_key : str
        Column name for technical quality.
    q_bio_key : str
        Column name for biological quality.
    dataset_key : str, optional
        Grouping key for per-dataset weighting (e.g., "donor_id", "batch").
        If None, treats all cells as one group.
    sat_thr : float
        Saturation threshold for effective information calculation.
    eps : float
        Small constant to avoid division by zero.
    min_component_weight : float, optional
        Minimum weight for each component (enforces balance).
    out_q_score_key : str
        Output column name for q_score.
    out_debug_key : str
        Key in adata.uns for debug DataFrame.
    stretch_method : str, optional
        Non-linear transform applied to q_bio after normalization:
        - "root_g2": Square root (gamma=2)
        - "root_g3": Cube root (gamma=3)
        - "sigmoid_k8/k12": Sigmoid transforms
        - "logit_k3/k5": Logit-sigmoid transforms
        - "power_g2": Power transform
        - "quantile": Quantile ranking
        - None: No transform
    linear_with_raw_qbio : bool
        If True, skip all non-linear transforms and contrast amplification.
    amplify_gamma : float
        Gamma for power-based contrast amplification.
    amplify_strength : float
        Blending strength for amplified component (0-1).
    kernel_weighting : float
        Steepness of sigmoid weight transition (k parameter).

    Returns
    -------
    np.ndarray
        Final q_score values.

    Notes
    -----
    The combination weights are determined from effective information content:
    eff_info = IQR * (1 - saturation_rate)

    Higher effective info → higher weight for that component.
    """
    obs = adata.obs

    # Robust normalization of inputs
    qt_all = _robust_minmax(
        pd.to_numeric(obs[q_tech_key], errors="coerce").values,
        lo_q=1,
        hi_q=100,
    )

    qb_raw = pd.to_numeric(obs[q_bio_key], errors="coerce").values
    qb_all = _robust_minmax(qb_raw, lo_q=1, hi_q=100)

    # Define available stretch methods
    methods = {
        "root_g2": dict(method="root", gamma=2.0),
        "root_g3": dict(method="root", gamma=3.0),
        "sigmoid_k8": dict(method="sigmoid", k=8.0, m=float(np.nanmedian(np.asarray(qb_all)))),
        "sigmoid_k12": dict(method="sigmoid", k=12.0, m=float(np.nanmedian(np.asarray(qb_all)))),
        "logit_k3": dict(method="logit", k=3.0, m=float(np.nanmedian(np.asarray(qb_all)))),
        "logit_k5": dict(method="logit", k=5.0, m=float(np.nanmedian(np.asarray(qb_all)))),
        "power_g2": dict(method="power", gamma=2.0),
        "quantile": dict(method="quantile"),
    }

    # Apply stretch transform to q_bio if requested
    if stretch_method is not None and not linear_with_raw_qbio:
        params = methods[stretch_method]
        qb_all = stretch_01(np.asarray(qb_all), **params)

    # Define groups
    if dataset_key is None:
        groups = pd.Series(["all"] * adata.n_obs, index=obs.index)
    else:
        groups = obs[dataset_key].astype(str)

    q_score = np.full(adata.n_obs, np.nan, dtype=float)
    debug = []

    # Process each group
    for group_name, group_index in groups.groupby(groups).groups.items():
        idx = obs.index.get_indexer(group_index)

        qt = qt_all[idx]
        qb = qb_all[idx]

        # Effective information content: IQR * (1 - saturation)
        iqr_t = _iqr(qt)
        iqr_b = _iqr(qb)
        sat_t = float(np.mean(qt > sat_thr))
        sat_b = float(np.mean(qb > sat_thr))

        eff_t = max(iqr_t * (1.0 - sat_t), eps)
        eff_b = max(iqr_b * (1.0 - sat_b), eps)

        # Smooth weighting between components
        alpha, beta = _smooth_alpha(eff_t, eff_b, k=kernel_weighting)

        # Enforce minimum component weights if requested
        if min_component_weight is not None:
            alpha = max(alpha, min_component_weight)
            beta = max(beta, min_component_weight)
            s = alpha + beta
            alpha, beta = alpha / s, beta / s

        # Contrast amplification of more informative component
        qt_use, qb_use = qt, qb
        if not linear_with_raw_qbio:
            if eff_b >= eff_t:
                qb_amp = _mono_amplify_pow(qb, gamma=amplify_gamma)
                qb_use = (1 - amplify_strength) * qb + amplify_strength * qb_amp
            else:
                qt_amp = _mono_amplify_pow(qt, gamma=amplify_gamma)
                qt_use = (1 - amplify_strength) * qt + amplify_strength * qt_amp

        # Final weighted combination
        q_score[idx] = alpha * qt_use + beta * qb_use

        # Record debug info
        debug.append({
            "group": group_name,
            "alpha_from_qtech": alpha,
            "beta_from_qbio": beta,
            "iqr_qtech": iqr_t,
            "iqr_qbio": iqr_b,
            "sat_qtech(>thr)": sat_t,
            "sat_qbio(>thr)": sat_b,
            "effinfo_qtech": eff_t,
            "effinfo_qbio": eff_b,
            "amplify_gamma": amplify_gamma,
            "amplify_strength": amplify_strength,
            "amplified_component": "q_bio" if eff_b >= eff_t else "q_tech",
            "kernel_weighting_k": kernel_weighting,
        })

    adata.obs[out_q_score_key] = q_score
    adata.uns[out_debug_key] = pd.DataFrame(debug).set_index("group")
    return q_score


def compute_q_score_v2(
        adata: AnnData,
        q_tech_key: str = "q_tech_bcr",
        q_bio_key: str = "q_bio_bcr",
        stretch_method: Optional[str] = "root_g2",
        sat_thr: float = 0.99,
        eps: float = 1e-3,
        out_q_score_key: str = "q_score",
        out_debug_key: str = "q_score_debug",
) -> np.ndarray:
    """
    Compute final q_score using global simple weighting (Variant B).

    This is the simplified version without per-group processing or
    contrast amplification. Suitable for single-batch analysis.

    Parameters
    ----------
    adata : AnnData
        Single-cell data object.
    q_tech_key : str
        Column name for technical quality.
    q_bio_key : str
        Column name for biological quality.
    stretch_method : str, optional
        Non-linear transform applied to q_bio (simple global version).
    sat_thr : float
        Saturation threshold for effective information calculation.
    eps : float
        Small constant to avoid division by zero.
    out_q_score_key : str
        Output column name for q_score.
    out_debug_key : str
        Key in adata.uns for debug DataFrame.

    Returns
    -------
    np.ndarray
        Final q_score values.

    Notes
    -----
    Simplified logic:
    1. Global effective information content (no grouping).
    2. No contrast amplification.
    3. If one component dominates (>90% weight), use it exclusively.
    4. Otherwise, simple weighted average.
    """
    obs = adata.obs

    # Get raw values
    qt_all = pd.to_numeric(obs[q_tech_key], errors="coerce").values
    qb_all = pd.to_numeric(obs[q_bio_key], errors="coerce").values

    # Apply stretch transform if requested
    methods = {
        "root_g2": dict(method="root", gamma=2.0),
        "root_g3": dict(method="root", gamma=3.0),
        "sigmoid_k8": dict(method="sigmoid", k=8.0, m=float(np.nanmedian(np.asarray(qb_all)))),
        "sigmoid_k12": dict(method="sigmoid", k=12.0, m=float(np.nanmedian(np.asarray(qb_all)))),
        "logit_k3": dict(method="logit", k=3.0, m=float(np.nanmedian(np.asarray(qb_all)))),
        "logit_k5": dict(method="logit", k=5.0, m=float(np.nanmedian(np.asarray(qb_all)))),
        "power_g2": dict(method="power", gamma=2.0),
        "quantile": dict(method="quantile"),
    }

    if stretch_method is not None:
        params = methods[stretch_method]
        qb_all = stretch_01(np.asarray(qb_all), **params)

    # Global effective information
    iqr_t = _iqr(qt_all)
    iqr_b = _iqr(qb_all)
    sat_t = float(np.mean(qt_all > sat_thr))
    sat_b = float(np.mean(qb_all > sat_thr))

    eff_t = max(iqr_t * (1.0 - sat_t), eps)
    eff_b = max(iqr_b * (1.0 - sat_b), eps)

    # Simple weighting (no smooth sigmoid)
    alpha = eff_t / (eff_t + eff_b)
    beta = eff_b / (eff_t + eff_b)

    # Dominant component fallback (simplified logic)
    if alpha > 0.9:
        q_score = qt_all
    elif beta > 0.9:
        q_score = qb_all
    else:
        q_score = alpha * qt_all + beta * qb_all

    # Simple debug info (single row)
    debug = pd.DataFrame([{
        "alpha_from_qtech": alpha,
        "beta_from_qbio": beta,
        "iqr_qtech": iqr_t,
        "iqr_qbio": iqr_b,
        "sat_qtech(>thr)": sat_t,
        "sat_qbio(>thr)": sat_b,
        "effinfo_qtech": eff_t,
        "effinfo_qbio": eff_b,
    }])

    adata.obs[out_q_score_key] = q_score
    adata.uns[out_debug_key] = debug
    return q_score


# ============================================================================
# Section 5: High-Level Pipeline Interface
# ============================================================================

def compute_all_q_scores(
        adata: AnnData,
        bcr_path: Optional[str] = None,
        compute_tech: bool = True,
        compute_bio: bool = True,
        compute_aggregate: bool = True,
        q_score_variant: str = "v2",
        **kwargs
) -> AnnData:
    """
    High-level pipeline to compute complete q_score from raw inputs.

    This is a convenience function that runs the full pipeline:
    q_tech_bcr → q_bio_bcr → q_score.

    Parameters
    ----------
    adata : AnnData
        Input single-cell data.
    bcr_path : str, optional
        Path to Cell Ranger VDJ output (required if compute_tech=True).
    compute_tech : bool
        Whether to compute q_tech_bcr from contig files.
    compute_bio : bool
        Whether to compute q_bio_bcr from BCR metadata.
    compute_aggregate : bool
        Whether to compute final q_score aggregation.
    q_score_variant : str
        Which aggregation variant to use ("v1" or "v2").
    **kwargs : dict
        Additional parameters passed to individual functions.

    Returns
    -------
    AnnData
        Updated AnnData with all q_score columns added.
    """
    if compute_tech:
        if bcr_path is None:
            raise ValueError("bcr_path required for technical quality computation")
        print("Computing q_tech_bcr from contig annotations...")
        compute_q_tech_bcr(adata, bcr_path, **kwargs.get('tech_kwargs', {}))

    if compute_bio:
        print("Computing q_bio_bcr from BCR metadata...")
        compute_q_bio_bcr(adata, **kwargs.get('bio_kwargs', {}))

    if compute_aggregate:
        print(f"Computing final q_score (variant: {q_score_variant})...")
        if q_score_variant == "v1":
            compute_q_score(adata, **kwargs.get('aggregate_kwargs', {}))
        else:
            compute_q_score_v2(adata, **kwargs.get('aggregate_kwargs', {}))

    print("Q-score computation complete.")
    print(f"  Columns added: {[c for c in adata.obs.columns if 'q_' in c]}")
    return adata