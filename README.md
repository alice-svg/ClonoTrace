# ClonoTrace

### 1. compute BCR maturation quality score (q-score)

BCR maturation quality scoring module for single-cell BCR data. Computes three scores per cell:

| Column | Description |
|---|---|
| `q_tech_bcr` | Technical confidence from Cell Ranger contig annotations (0–1) |
| `q_bio_bcr` | Biologically informed maturation — isotype, SHM, clonal expansion (0–1) |
| `q_score` | Final aggregated score (0–1) |

---

#### Installation

```bash
pip install numpy pandas scanpy anndata
```

---

#### Usage

#### Full pipeline

```python
import scanpy as sc
from btraj.qscore.compute_q_score import compute_all_q_scores

adata = sc.read_h5ad("your_data.h5ad")

adata = compute_all_q_scores(
    adata,
    bcr_path="/path/to/cellranger_vdj_outputs/",
    compute_tech=True,
    compute_bio=True,
    compute_aggregate=True,
    q_score_variant="v2",   # "v1" or "v2"
)

print(adata.obs[["q_tech_bcr", "q_bio_bcr", "q_score"]].head())
```

#### Step-by-step

```python
from compute_q_score import compute_q_tech_bcr, compute_q_bio_bcr, compute_q_score_v2

compute_q_tech_bcr(adata, bcr_path="/path/to/cellranger_vdj_outputs/")
compute_q_bio_bcr(adata)
compute_q_score_v2(adata)
```

---

#### Input

**`adata.obs`** must contain:
- `donor_id` — sample/donor identifier (used to locate contig files)
- `isotype` — constant region gene (e.g., `IGHG1`)
- `SHM` — somatic hypermutation rate
- `clone_id` — clone identifier

**Cell Ranger VDJ files** under `bcr_path`, named as:
```
{donor_id}_all_contig_annotations.csv
```

---

#### Output

After running, results are stored in:

- `adata.obs["q_tech_bcr"]`
- `adata.obs["q_bio_bcr"]`
- `adata.obs["q_score"]`
- `adata.uns["q_score_debug"]` — per-group weighting details

---

#### Aggregation Variants

| Variant | Function | Best for |
|---|---|---|
| `"v2"` *(default)* | `compute_q_score_v2` | Single-batch, simple global weighting |
| `"v1"` | `compute_q_score` | Multi-batch, per-group adaptive weighting |

#### `v1`-only parameters

| Parameter | Default | Description |
|---|---|---|
| `dataset_key` | `None` | Grouping column for per-batch weights (e.g. `"donor_id"`) |
| `stretch_method` | `"root_g2"` | Transform on `q_bio`: `root_g2/g3`, `sigmoid_k8/k12`, `logit_k3/k5`, `power_g2`, `quantile`, `None` |
| `amplify_gamma` | `2.0` | Contrast amplification exponent |
| `amplify_strength` | `1.0` | Blending strength for amplified component |
| `kernel_weighting` | `200.0` | Sigmoid steepness for weight transition |
| `min_component_weight` | `None` | Minimum weight per component |

---

#### `compute_q_bio_bcr` Parameters

| Parameter | Default | Description |
|---|---|---|
| `isotype_col` | `"isotype"` | Isotype column |
| `shm_col` | `"SHM"` | SHM rate column |
| `clone_id_col` | `"clone_id"` | Clone ID column |
| `v_identity_col` | `None` | V-region identity column (alternative to SHM) |
| `weights` | `{"iso": 0.45, "shm": 0.45, "clonal": 0.10}` | Component weights |
| `tech_gate_power` | `1.0` | Gating exponent for `q_tech_bcr` |
| `coverage_power` | `0.0` | Penalty for missing evidence |

---

#### `compute_q_tech_bcr` Parameters

| Parameter | Default | Description |
|---|---|---|
| `donor_id_col` | `"donor_id"` | Donor column in `adata.obs` |
| `cell_id_col` | `"cell_id"` | Cell barcode column |
| `output_prefix` | `""` | Prefix for output columns |




---

### 2. Trajectory Inference Pipeline

Runs the full BCR trajectory inference pipeline on a q-score-annotated `.h5ad` file.

#### Usage

```bash
python run_inference.py \
  --input data/your_data.h5ad \
  --output results/trajectory_results.h5ad \
  --start-type PRE_PRO_B \
  --terminal-types MATURE_B,B1,PLASMA_B \
  --fate-method direct \
  --pseudotime-mode q_aware \
  --q-aware-strength balanced
```

#### Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--input` | str | *(required)* | Path to input `.h5ad` file |
| `--output` | str | `trajectory_results.h5ad` beside `--input` | Path for output `.h5ad` |
| `--start-type` | str | `PRE_PRO_B` | Root cell type for trajectory |
| `--terminal-types` | str | `MATURE_B,B1,PLASMA_B` | Comma-separated terminal cell types for fate probabilities |
| `--terminal-types-extended` | str | `""` | Extended terminal types for fine-grained branch probabilities |
| `--fate-method` | str | `direct` | Fate method: `direct` / `sparse_custom` / `gpcca` |
| `--pseudotime-mode` | str | `q_aware` | Pseudotime mode: `raw` / `q_aware` |
| `--q-aware-strength` | str | `balanced` | q_aware blending strength: `fast` / `balanced` / `full` |
| `--celltype-order` | str | `""` | Comma-separated cell types in expected developmental order (for validation) |
| `--skip-embeddings` | flag | `False` | Skip AntiBERTy embedding computation if `X_h` already exists in `adata.obsm` |
| `--embeddings-cache` | str | `None` | Path to `.npy` for caching/loading precomputed PCA-reduced AntiBERTy embeddings |
| `--embeddings-npy` | str | `None` | Path to full-length AntiBERTy `.npy` (BCR+ cells only); triggers alignment + PCA |
| `--pseudotime-csv` | str | `None` | Path to SOTA pseudotime CSV for comparison (merged into `adata.obs`) |
| `--resume` | flag | `False` | Resume from the latest checkpoint |
| `--checkpoint-dir` | str | `<output_dir>/.checkpoints` | Directory for step-wise checkpoints |
| `--log-dir` | str | output directory | Directory for log files |
| `--heartbeat-interval` | float | `30.0` | Seconds between heartbeat log messages |

#### Input requirements

`adata.obs` must contain: `celltype`, `q_score`, `Heavy`, `Light`

`adata.obsm` must contain: `X_scvi`

#### Output

| File | Description |
|---|---|
| `trajectory_results.h5ad` | `adata` with `pseudotime_raw`, trajectory tree, and `prob_*` fate probability columns |
| `*.log` | Detailed run log |
| `.checkpoints/*.h5ad` | Step-wise checkpoint files |

#### Pipeline steps

1. Load `.h5ad` and validate required columns
2. Compute or load AntiBERTy embeddings (PCA → 256 dims)
3. Build MST-based trajectory graph; compute initial geometric pseudotime
4. Apply q-score monotonicity correction (if `--pseudotime-mode q_aware`)
5. Build kNN graph
6. Construct tri-component transition matrix (pseudotime + BCR embedding + connectivity kernel)
7. Compute fate probabilities (`--fate-method`)
8. (Optional) Compute extended fate probabilities
9. Merge SOTA pseudotime CSV for comparison; save output

---

### 3. Q-Score Stability Analysis

Validates q_score reproducibility via repeated 80% subsampling and Spearman ρ correlation.

#### Usage

```bash
python qscore_bootstrap.py \
  --h5ad results/trajectory_results.h5ad \
  --n-iter 100 \
  --subsample-frac 0.8 \
  --out-dir results/qscore_bootstrap
```

#### Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--h5ad` | str | *(required)* | Path to AnnData `.h5ad` file |
| `--n-iter` | int | `100` | Number of bootstrap iterations |
| `--subsample-frac` | float | `0.8` | Fraction of cells to subsample per iteration |
| `--q-score-key` | str | `q_score` | Column in `adata.obs` for the reference q_score |
| `--stretch-method` | str | `root_g2` | Transform applied when recomputing q_score on subsamples |
| `--out-dir` | str | `results/qscore_bootstrap` | Output directory |

#### Output

| File | Description |
|---|---|
| `qscore_bootstrap.json` | `mean_rho`, `std_rho`, 95% CI |
| `qscore_bootstrap.pdf` | Histogram of per-iteration Spearman ρ |

#### Python API

```python
from qscore_bootstrap import qscore_bootstrap

results = qscore_bootstrap(adata, n_iter=100, subsample_frac=0.8)
print(results["mean_rho"], results["ci_95_lo"], results["ci_95_hi"])
```

---

### 4. Trajectory Quality Evaluation

Computes 8 quantitative metrics to benchmark trajectory quality against a known developmental order.

#### Usage

```bash
python evaluate_trajectory.py results/trajectory_results.h5ad \
  --pseudotime_col pseudotime_raw \
  --celltype_col celltype \
  --branch_cols prob_MATURE_B,prob_B1,prob_PLASMA_B \
  --label my_run \
  --out_dir results/eval
```

#### Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `h5ad_path` | str | *(positional, required)* | Path to `.h5ad` with trajectory results |
| `--pseudotime_col` | str | `pseudotime_raw` | Pseudotime column in `adata.obs` |
| `--celltype_col` | str | `celltype` | Cell type column in `adata.obs` |
| `--branch_cols` | str | `None` | Comma-separated fate probability columns |
| `--label` | str | `unnamed` | Label for this run (used in output filenames) |
| `--out_dir` | str | `None` | Output directory |
| `--celltype_order` | str | `None` | Comma-separated cell types in expected developmental order |

#### Output

| File | Description |
|---|---|
| `<label>_eval.json` | Full metric scores and composite grade |
| `<label>_eval.png` | Multi-panel evaluation summary figure |

#### Metrics & composite score weights

| Metric | Weight |
|---|---|
| Spearman ρ (cell type median) | 15% |
| Kendall τ (marker genes) | 25% |
| Branch AUC | 15% |
| Variance explained by pseudotime | 10% |
| Pairwise cell type separation | 15% |
| Pseudotime distribution stats | 10% |
| Geodesic distance correlation | 5% |
| F1 branch assignment | 5% |

Final composite score is graded **A–F**.

---

### 5. Pseudotime Order Accuracy

Compares multiple pseudotime methods against a known biological cell type order using Spearman ρ,Kendall τ, and Inversion Rate.

#### Usage

```python
from evaluate_scoring import evaluate_pseudotime_order

metrics = evaluate_pseudotime_order(
    meta=df,                          # DataFrame with celltype + pseudotime columns
    celltype_col="celltype",
    pseudotime_cols=["monocle3_pseudotime", "palantir_pseudotime", "ours_pseudotime"],
    order=["PRE_PRO_B", "PRO_B", "LARGE_PRE_B", "SMALL_PRE_B", "IMMATURE_B", "MATURE_B"],
    sort_by="kendall",                # "spearman" | "kendall" | "inversion"
    save_path="figs/accuracy.pdf",
)
print(metrics)
```

#### Function signature

```python
evaluate_pseudotime_order(
    meta,
    celltype_col="celltype",
    pseudotime_cols=("pseudotime",),
    order=None,                 # required — biological ground-truth order
    figsize=(7, 4),
    sort_by="spearman",         # "spearman" | "kendall" | "inversion"
    save_path=None,
)
```

#### Input

A `pd.DataFrame` (or CSV loaded into one) where each row is a cell type, containing:
- a `celltype` column
- one or more pseudotime columns (one per method being compared)

#### Output

| Output | Description |
|---|---|
| `metrics_df` | DataFrame with `kendall_tau`, `spearman_rho`, `inversion_rate` per method |
| `save_path` (PDF) | Bar + strip plot ranked by the chosen metric |

---

### Recommended Workflow

```
compute_q_score.py          →   q_tech_bcr, q_bio_bcr, q_score added to adata.obs
        ↓
run_inference.py            →   trajectory_results.h5ad  (pseudotime_raw, prob_*)
        ↓
qscore_bootstrap.py         →   stability check (Spearman ρ, 95% CI)
        ↓
evaluate_trajectory.py      →   8-metric evaluation report (JSON + PNG)
        ↓
evaluate_scoring.py         →   compare vs. SOTA pseudotime methods (PDF)
```
