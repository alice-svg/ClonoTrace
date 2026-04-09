"""
BCR Trajectory Quality Evaluation Framework.

Standardized metrics for evaluating trajectory inference methods on B cell data.
Run this script against any trajectory result to get comparable quality scores.

Metrics implemented:
  1. Spearman rank correlation (pseudotime vs known cell type ordering)
  2. Kendall's tau (pseudotime vs marker gene expression)
  3. Branch probability AUC (branch prob discriminates cell types)
  4. Pseudotime variance decomposition (R² of pseudotime ~ cell type)
  5. Pseudotime separation score (effect size between cell types)
  6. Palantir convergence diagnostic (iteration count)
  7. Geodesic distance correlation (dynverse cor_dist)
  8. F1-Branches (dynverse cell assignment accuracy)

Usage:
    python evaluate_trajectory.py <trajectory_h5ad> [--pseudotime_col pseudotime_raw]
                                                     [--celltype_col celltype]
                                                     [--branch_cols prob_B1,prob_PLASMA_B]
                                                     [--label "VDJ-only"]

References:
  - Saelens et al. (2019) Nat. Biotech. — dynverse TI benchmarking
  - Suo et al. (2023) Nat. Biotech. — dandelion
  - Qiu et al. (2024) Nat. Methods — sciCSR
"""

import sys
import os
import argparse
import warnings
import json
from datetime import datetime

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

os.environ["SETUPTOOLS_SCM_PRETEND_VERSION"] = "0.0.0dev0"

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import spearmanr, kendalltau, mannwhitneyu, kruskal
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import roc_auc_score, f1_score
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# =====================================================================
# B cell marker definitions
# =====================================================================

# Marker genes with expected direction along B cell development trajectory
# +1 = should increase with pseudotime (mature/late markers)
# -1 = should decrease with pseudotime (early progenitor markers)
#  0 = control / no expected direction
# Covers bone marrow progenitors through peripheral maturation and terminal fates
MARKER_GENES = {
    # --- Early progenitor markers (should DECREASE along development) ---
    "RAG1": -1,      # V(D)J recombination — peaks PRO_B/SMALL_PRE_B
    "RAG2": -1,      # V(D)J recombination
    "DNTT": -1,      # Terminal deoxynucleotidyl transferase — early
    "VPREB1": -1,    # Pre-BCR surrogate light chain — peaks LARGE_PRE_B
    "IGLL1": -1,     # Lambda5 surrogate light chain — peaks LARGE_PRE_B
    "IGLL5": -1,     # Surrogate light chain component
    "CD34": -1,      # Stem/progenitor marker — earliest stages only
    "SOX4": -1,      # Early B lineage TF — decreases with maturation
    "EBF1": -1,      # Early B cell factor — high early, moderate later
    "TCL1A": -1,     # Naive/immature B marker — decreases with maturation
    "LEF1": -1,      # Wnt pathway TF — early B lineage

    # --- Late / mature B markers (should INCREASE along development) ---
    "MS4A1": +1,     # CD20 — mature B cell marker
    "CR2": +1,       # CD21 — complement receptor, mature B
    "IGHD": +1,      # IgD — surface Ig on mature naive B
    "IGHM": +1,      # IgM — appears at IMMATURE_B, increases
    "FCER2": +1,     # CD23 — mature B activation
    "CD22": +1,      # Siglec-2 — mature B cell
    "SELL": +1,      # CD62L — mature B homing

    # --- Memory markers (should INCREASE) ---
    "CD27": +1,      # Canonical memory B marker
    "IGHG1": +1,     # IgG1 — class-switched memory
    "IGHG2": +1,     # IgG2 — class-switched
    "IGHG3": +1,     # IgG3 — class-switched
    "IGHA1": +1,     # IgA1 — mucosal class-switched
    "IGHA2": +1,     # IgA2 — mucosal class-switched
    "TNFRSF13B": +1, # TACI — memory/plasma survival

    # --- Atypical memory / ABC markers (should INCREASE) ---
    "FCRL5": +1,     # FcRL5 — atypical memory
    "TBX21": +1,     # T-bet — atypical memory / ABC
    "ITGAX": +1,     # CD11c — atypical memory / ABC

    # --- Plasma / terminal markers (should INCREASE) ---
    "SDC1": +1,      # CD138 — plasma cell
    "XBP1": +1,      # Plasma cell UPR
    "PRDM1": +1,     # Blimp-1 — plasma cell TF
    "IRF4": +1,      # Plasma cell TF
    "MZB1": +1,      # Marginal zone / plasma marker

    # --- GC / activation (should INCREASE) ---
    "AICDA": +1,     # AID — SHM enzyme
    "BCL6": +1,      # GC master TF
    "FAS": +1,       # CD95 — GC / activated B

    # --- Pan-B cell controls (no direction expected) ---
    "CD19": 0,       # Pan-B — present throughout
    "CD79A": 0,      # Igalpha — B lineage marker throughout
    "PAX5": 0,       # B cell identity TF — stable
    "CD38": 0,       # Variable across stages
    "CD24": 0,       # Variable across stages
}

# Known biological ordering of B cell development stages
# Lower = earlier in differentiation
# Covers bone marrow development + peripheral maturation + terminal fates
CELLTYPE_ORDER = {
    # --- Bone marrow development ---
    "PRE_PRO_B": 0,
    "PRO_B": 1,
    "LATE_PRO_B": 2,
    "LARGE_PRE_B": 3,
    "SMALL_PRE_B": 4,
    "CYCLING_B": 5,          # proliferating B cells
    "IMMATURE_B": 6,
    # --- Peripheral maturation ---
    "TRANSITIONAL_B": 7,     # T1/T2/T3 transitional
    "MATURE_B": 8,
    "NAIVE_B": 8,            # synonym for mature naive
    "FOLLICULAR_B": 8,       # follicular B cell (same rank as mature)
    "MZ_B": 8,               # marginal zone B cell
    # --- Activation / GC ---
    "ACTIVATED_B": 9,
    "GC_B": 10,              # germinal center B cell
    # --- Effector / terminal fates (parallel) ---
    "MEMORY_B": 11,
    "ATYPICAL_MEMORY_B": 11, # age-associated / atypical memory
    "ABC": 11,               # age-associated B cell (synonym)
    "BREG": 11,              # regulatory B cell
    "B1": 11,                # B-1 cell (innate-like)
    "PLASMABLAST": 12,
    "PLASMA_B": 12,          # long-lived plasma cell
}

# For pairwise separation: which pairs should be distinguishable
EXPECTED_SEPARATIONS = [
    # Bone marrow: early vs late
    ("PRE_PRO_B", "MATURE_B"),
    ("PRE_PRO_B", "IMMATURE_B"),
    ("PRO_B", "MATURE_B"),
    ("LARGE_PRE_B", "IMMATURE_B"),
    ("LATE_PRO_B", "SMALL_PRE_B"),
    ("PRE_PRO_B", "CYCLING_B"),
    # Peripheral: maturation vs terminal
    ("IMMATURE_B", "PLASMA_B"),
    ("MATURE_B", "B1"),
    ("NAIVE_B", "MEMORY_B"),
    ("NAIVE_B", "ATYPICAL_MEMORY_B"),
    ("FOLLICULAR_B", "GC_B"),
    ("GC_B", "PLASMA_B"),
]


# =====================================================================
# Metric 1: Spearman rank correlation
# =====================================================================

def metric_spearman_celltype(obs, pseudotime_col, celltype_col,
                             ordering=None):
    """
    Spearman rank correlation between pseudotime and known cell type order.

    Formula:
        rho = 1 - (6 * sum(d_i^2)) / (n * (n^2 - 1))
        where d_i = rank(pseudotime_i) - rank(celltype_order_i)

    Interpretation:
        |rho| > 0.5  -> strong trajectory signal
        |rho| > 0.3  -> moderate signal
        |rho| < 0.1  -> no signal (trajectory failed)
    """
    if ordering is None:
        ordering = CELLTYPE_ORDER

    valid = obs[celltype_col].isin(ordering) & obs[pseudotime_col].notna()
    if valid.sum() < 50:
        return {"rho": np.nan, "p_value": np.nan, "n": 0, "pass": False}

    ct_order = obs.loc[valid, celltype_col].map(ordering)
    pt = obs.loc[valid, pseudotime_col]

    rho, p = spearmanr(ct_order, pt)
    return {
        "rho": float(rho),
        "p_value": float(p),
        "n": int(valid.sum()),
        "pass": bool(abs(rho) > 0.3 and p < 0.05),
    }


# =====================================================================
# Metric 2: Kendall's tau with marker genes
# =====================================================================

def metric_kendall_markers(adata, pseudotime_col, markers=None):
    """
    Kendall's tau correlation between pseudotime and marker gene expression.

    Formula:
        tau = (concordant - discordant) / (n * (n-1) / 2)
        where concordant/discordant count pairs agreeing/disagreeing in order

    For each marker gene, we check:
        - Whether tau sign matches the expected biological direction
        - Whether the correlation is statistically significant (p < 0.05)

    Score = fraction of tested markers with correct significant correlation.
    """
    if markers is None:
        markers = MARKER_GENES

    pt = adata.obs[pseudotime_col].values
    valid_pt = ~np.isnan(pt)

    results = {}
    n_pass = 0
    n_tested = 0

    for gene, expected_sign in markers.items():
        if gene not in adata.var_names:
            continue

        X = adata[:, gene].X
        expr = np.asarray(X.todense() if hasattr(X, "todense") else X).flatten()

        # Use only cells with valid pseudotime and non-zero expression variance
        mask = valid_pt & np.isfinite(expr)
        if mask.sum() < 50 or np.std(expr[mask]) < 1e-10:
            continue

        tau, p = kendalltau(pt[mask], expr[mask])
        n_tested += 1

        if expected_sign == 0:
            # Control gene — just record, don't count
            correct = True
        else:
            correct = (tau > 0) == (expected_sign > 0) if not np.isnan(tau) else False

        is_pass = correct and p < 0.05 and abs(tau) > 0.02
        if is_pass and expected_sign != 0:
            n_pass += 1

        results[gene] = {
            "tau": float(tau) if not np.isnan(tau) else None,
            "p_value": float(p) if not np.isnan(p) else None,
            "expected_direction": expected_sign,
            "correct_direction": bool(correct),
            "pass": bool(is_pass),
        }

    n_directional = sum(1 for g, s in markers.items()
                        if s != 0 and g in results)
    score = n_pass / n_directional if n_directional > 0 else 0.0

    return {
        "per_gene": results,
        "n_pass": n_pass,
        "n_tested": n_directional,
        "score": float(score),  # fraction of markers passing
        "pass": score > 0.4,    # at least 40% of markers should pass
    }


# =====================================================================
# Metric 3: Branch probability AUC
# =====================================================================

def metric_branch_auc(obs, celltype_col, branch_cols):
    """
    AUC of branch probabilities for discriminating cell types.

    For each branch probability column (e.g. prob_PLASMA_B), compute:
        AUC = P(branch_prob(plasma cell) > branch_prob(non-plasma cell))

    Interpretation:
        AUC > 0.8  -> excellent discrimination
        AUC > 0.65 -> acceptable
        AUC ~ 0.5  -> random (trajectory failed)
    """
    results = {}

    for col in branch_cols:
        if col not in obs.columns:
            continue

        # Infer target cell type from column name
        ct_name = col.replace("prob_", "")
        if ct_name not in obs[celltype_col].values:
            continue

        valid = obs[col].notna() & obs[celltype_col].notna()
        y_true = (obs.loc[valid, celltype_col] == ct_name).astype(int)
        y_score = obs.loc[valid, col]

        if y_true.sum() < 10 or (1 - y_true).sum() < 10:
            continue

        auc = roc_auc_score(y_true, y_score)
        results[ct_name] = {
            "auc": float(auc),
            "n_positive": int(y_true.sum()),
            "n_negative": int((1 - y_true).sum()),
            "pass": bool(auc > 0.65),
        }

    mean_auc = np.mean([v["auc"] for v in results.values()]) if results else 0.0
    return {
        "per_branch": results,
        "mean_auc": float(mean_auc),
        "pass": bool(mean_auc > 0.65),
    }


# =====================================================================
# Metric 4: Pseudotime variance decomposition (R²)
# =====================================================================

def metric_variance_explained(obs, pseudotime_col, celltype_col):
    """
    Fraction of pseudotime variance explained by cell type (eta-squared).

    Formula (one-way ANOVA eta-squared):
        eta² = SS_between / SS_total
        SS_between = sum_k n_k * (mean_k - grand_mean)²
        SS_total   = sum_i (x_i - grand_mean)²

    Also reports Kruskal-Wallis H test (non-parametric).

    Interpretation:
        eta² > 0.14 -> large effect (cell type explains pseudotime well)
        eta² > 0.06 -> medium effect
        eta² < 0.01 -> negligible (trajectory has no cell-type signal)
    """
    valid = obs[pseudotime_col].notna() & obs[celltype_col].notna()
    df = obs.loc[valid, [pseudotime_col, celltype_col]].copy()

    pt = df[pseudotime_col].values
    grand_mean = pt.mean()
    ss_total = np.sum((pt - grand_mean) ** 2)

    ss_between = 0.0
    groups = []
    for ct, grp in df.groupby(celltype_col, observed=True):
        n_k = len(grp)
        if n_k == 0:
            continue
        mean_k = grp[pseudotime_col].mean()
        ss_between += n_k * (mean_k - grand_mean) ** 2
        groups.append(grp[pseudotime_col].values)

    eta_sq = ss_between / ss_total if ss_total > 0 else 0.0

    # Kruskal-Wallis H test (non-parametric)
    valid_groups = [g for g in groups if len(g) > 5]
    if len(valid_groups) >= 2:
        h_stat, h_p = kruskal(*valid_groups)
    else:
        h_stat, h_p = np.nan, np.nan

    return {
        "eta_squared": float(eta_sq),
        "ss_between": float(ss_between),
        "ss_total": float(ss_total),
        "kruskal_h": float(h_stat) if not np.isnan(h_stat) else None,
        "kruskal_p": float(h_p) if not np.isnan(h_p) else None,
        "pass": bool(eta_sq > 0.06),
    }


# =====================================================================
# Metric 5: Pairwise pseudotime separation (effect size)
# =====================================================================

def metric_pairwise_separation(obs, pseudotime_col, celltype_col,
                               pairs=None):
    """
    For each pair of cell types, compute:

    1. Cohen's d (standardized mean difference):
        d = (mean_A - mean_B) / sqrt((s_A² + s_B²) / 2)

    2. Mann-Whitney U test (non-parametric rank test):
        U statistic and p-value

    3. AUROC as effect size:
        AUROC = U / (n_A * n_B)
        (probability that a random cell from A has lower pseudotime than B)

    Interpretation:
        |d| > 0.8  -> large separation
        |d| > 0.5  -> medium separation
        |d| < 0.2  -> negligible (cell types not distinguished)
        AUROC > 0.7 -> good separation
        AUROC ~ 0.5 -> no separation
    """
    if pairs is None:
        pairs = EXPECTED_SEPARATIONS

    results = {}
    for ct_a, ct_b in pairs:
        mask_a = obs[celltype_col] == ct_a
        mask_b = obs[celltype_col] == ct_b

        pt_a = obs.loc[mask_a, pseudotime_col].dropna()
        pt_b = obs.loc[mask_b, pseudotime_col].dropna()

        if len(pt_a) < 10 or len(pt_b) < 10:
            continue

        # Cohen's d
        pooled_std = np.sqrt((pt_a.std() ** 2 + pt_b.std() ** 2) / 2)
        cohens_d = (pt_b.mean() - pt_a.mean()) / pooled_std if pooled_std > 0 else 0.0

        # Mann-Whitney U
        u_stat, u_p = mannwhitneyu(pt_a, pt_b, alternative="two-sided")

        # AUROC as effect size (rank biserial)
        auroc = u_stat / (len(pt_a) * len(pt_b))

        pair_key = f"{ct_a}_vs_{ct_b}"
        results[pair_key] = {
            "mean_A": float(pt_a.mean()),
            "mean_B": float(pt_b.mean()),
            "std_A": float(pt_a.std()),
            "std_B": float(pt_b.std()),
            "n_A": int(len(pt_a)),
            "n_B": int(len(pt_b)),
            "cohens_d": float(cohens_d),
            "mann_whitney_U": float(u_stat),
            "mann_whitney_p": float(u_p),
            "auroc_effect_size": float(auroc),
            "pass": bool(abs(cohens_d) > 0.5 and u_p < 0.05),
        }

    n_pass = sum(1 for v in results.values() if v["pass"])
    return {
        "per_pair": results,
        "n_pass": n_pass,
        "n_tested": len(results),
        "pass": bool(n_pass >= 1),  # at least one pair should separate
    }


# =====================================================================
# Metric 6: Pseudotime distribution statistics
# =====================================================================

def metric_pseudotime_stats(obs, pseudotime_col, celltype_col, ordering=None):
    """
    Descriptive statistics of pseudotime per cell type.

    Reports mean, median, std, IQR, and checks for monotonic ordering:
        PRE_PRO_B < PRO_B < ... < MATURE_B (expected biological ordering).

    Also computes the dynamic range:
        range = max(group_medians) - min(group_medians)
        normalized_range = range / pseudotime_IQR

    Interpretation:
        normalized_range > 1.0 -> clear separation
        normalized_range < 0.3 -> cell types overlap heavily
    """
    valid = obs[pseudotime_col].notna() & obs[celltype_col].notna()
    df = obs.loc[valid, [pseudotime_col, celltype_col]]

    stats = {}
    medians = {}
    for ct, grp in df.groupby(celltype_col):
        pt = grp[pseudotime_col]
        stats[ct] = {
            "mean": float(pt.mean()),
            "median": float(pt.median()),
            "std": float(pt.std()),
            "q25": float(pt.quantile(0.25)),
            "q75": float(pt.quantile(0.75)),
            "n": int(len(pt)),
        }
        medians[ct] = pt.median()

    # Check monotonic ordering (PRE_PRO_B should be earliest)
    _order = ordering if ordering is not None else CELLTYPE_ORDER
    ordered_cts = sorted(medians, key=lambda x: _order.get(x, 99))
    monotonic = all(
        medians.get(ordered_cts[i], 0) <= medians.get(ordered_cts[i + 1], 0)
        for i in range(len(ordered_cts) - 1)
        if ordered_cts[i] in medians and ordered_cts[i + 1] in medians
    )

    # Dynamic range
    all_pt = df[pseudotime_col]
    pt_iqr = all_pt.quantile(0.75) - all_pt.quantile(0.25)
    median_vals = list(medians.values())
    med_range = max(median_vals) - min(median_vals) if median_vals else 0
    norm_range = med_range / pt_iqr if pt_iqr > 0 else 0

    # Inversion rate (from evaluate_scoring.py)
    # Count pairs of cell types where median pseudotime is in wrong order
    ordered_with_medians = [
        (ct, medians[ct]) for ct in ordered_cts if ct in medians
    ]
    pairs = [
        (i, j)
        for i in range(len(ordered_with_medians))
        for j in range(i + 1, len(ordered_with_medians))
    ]
    inversions = sum(
        1 for i, j in pairs
        if ordered_with_medians[i][1] > ordered_with_medians[j][1]
    )
    inversion_rate = 100 * inversions / len(pairs) if pairs else np.nan

    # Group-level Spearman (from evaluate_scoring.py)
    # Spearman correlation of median pseudotime ranks vs expected ranks
    if len(ordered_with_medians) >= 3:
        expected_ranks = np.arange(len(ordered_with_medians))
        median_values = np.array([m for _, m in ordered_with_medians])
        group_rho, group_rho_p = spearmanr(expected_ranks, median_values)
    else:
        group_rho, group_rho_p = np.nan, np.nan

    return {
        "per_celltype": stats,
        "monotonic_ordering": bool(monotonic),
        "median_range": float(med_range),
        "normalized_range": float(norm_range),
        "inversion_rate": float(inversion_rate) if not np.isnan(inversion_rate) else None,
        "group_spearman_rho": float(group_rho) if not np.isnan(group_rho) else None,
        "group_spearman_p": float(group_rho_p) if not np.isnan(group_rho_p) else None,
        "pass": bool(norm_range > 0.3 and monotonic),
    }


# =====================================================================
# Metric 7: Geodesic distance correlation (dynverse cor_dist)
# =====================================================================

def metric_geodesic_correlation(obs, pseudotime_col, celltype_col,
                                n_waypoints=200, ordering=None):
    """
    Proxy for dynverse cor_dist metric using pseudotime distances.

    True cor_dist requires a reference trajectory graph. Without one,
    we approximate by:
    1. Sample waypoint cells (stratified by cell type)
    2. Compute pairwise pseudotime distances
    3. Compute pairwise cell-type-based expected distances
    4. Correlate the two distance matrices

    Expected distance: |celltype_order(i) - celltype_order(j)|

    Interpretation:
        cor > 0.5  -> pseudotime respects expected cell type distances
        cor ~ 0    -> no correlation (trajectory is random)
    """
    _order = ordering if ordering is not None else CELLTYPE_ORDER
    valid = (
        obs[pseudotime_col].notna()
        & obs[celltype_col].isin(_order)
    )
    df = obs.loc[valid, [pseudotime_col, celltype_col]].copy()

    if len(df) < 20:
        return {"cor": np.nan, "p_value": np.nan, "n_waypoints": 0, "pass": False}

    if len(df) < n_waypoints:
        n_waypoints = len(df)

    # Stratified sampling
    np.random.seed(42)
    n_unique = df[celltype_col].nunique()
    n_per_group = max(n_waypoints // n_unique, 10) if n_unique > 0 else 10
    sampled_parts = []
    for ct, grp in df.groupby(celltype_col):
        n_sample = min(len(grp), n_per_group)
        sampled_parts.append(grp.sample(n_sample, random_state=42))
    sampled = pd.concat(sampled_parts)

    if len(sampled) < 20:
        return {"cor": np.nan, "p_value": np.nan, "n_waypoints": 0, "pass": False}

    # Pairwise pseudotime distances
    pt_vals = sampled[pseudotime_col].values
    pt_dists = pdist(pt_vals.reshape(-1, 1), metric="euclidean")

    # Pairwise expected distances from cell type ordering
    ct_vals = sampled[celltype_col].map(_order).values
    ct_dists = pdist(ct_vals.reshape(-1, 1), metric="euclidean")

    cor, p = spearmanr(pt_dists, ct_dists)

    return {
        "cor": float(cor) if not np.isnan(cor) else None,
        "p_value": float(p) if not np.isnan(p) else None,
        "n_waypoints": int(len(sampled)),
        "pass": bool(abs(cor) > 0.3 and p < 0.05),
    }


# =====================================================================
# Metric 8: F1-Branches (dynverse branch assignment)
# =====================================================================

def metric_f1_branches(obs, celltype_col, branch_cols):
    """
    F1-Branches metric from dynverse benchmarking (Saelens et al. 2019).

    Assigns each cell to a branch using argmax of branch probabilities,
    then computes:

        F1_branches = harmonic_mean(Recovery, Relevance)
        Recovery   = mean_k max_j Jaccard(C_k^ref, C_j^pred)
        Relevance  = mean_j max_k Jaccard(C_k^ref, C_j^pred)
        Jaccard(A,B) = |A ∩ B| / |A ∪ B|

    Where C_k^ref are ground-truth cell type sets and C_j^pred are
    predicted branch assignment sets.

    Interpretation:
        F1 > 0.7  -> good branch assignment
        F1 > 0.5  -> moderate
        F1 ~ 0.33 -> random for 3 classes
    """
    if not branch_cols or not all(c in obs.columns for c in branch_cols):
        return {"f1": np.nan, "pass": False}

    # Assign cells to branches by argmax
    prob_df = obs[branch_cols].copy()
    valid = prob_df.notna().all(axis=1)
    prob_df = prob_df.loc[valid]

    # Predicted branch = column name with highest probability
    pred_branch = prob_df.idxmax(axis=1).str.replace("prob_", "")

    # Reference: actual cell types
    ref_types = obs.loc[valid, celltype_col]

    # Get unique sets
    ref_groups = {ct: set(ref_types[ref_types == ct].index)
                  for ct in ref_types.unique()}
    pred_groups = {br: set(pred_branch[pred_branch == br].index)
                   for br in pred_branch.unique()}

    def jaccard(a, b):
        inter = len(a & b)
        union = len(a | b)
        return inter / union if union > 0 else 0.0

    # Recovery: for each ref cluster, find best matching pred cluster
    recovery_scores = []
    for ct, ref_set in ref_groups.items():
        if len(ref_set) == 0:
            continue
        best_j = max(jaccard(ref_set, pred_set)
                     for pred_set in pred_groups.values()) if pred_groups else 0
        recovery_scores.append(best_j)
    recovery = np.mean(recovery_scores) if recovery_scores else 0.0

    # Relevance: for each pred cluster, find best matching ref cluster
    relevance_scores = []
    for br, pred_set in pred_groups.items():
        if len(pred_set) == 0:
            continue
        best_j = max(jaccard(pred_set, ref_set)
                     for ref_set in ref_groups.values()) if ref_groups else 0
        relevance_scores.append(best_j)
    relevance = np.mean(relevance_scores) if relevance_scores else 0.0

    # F1 = harmonic mean
    if recovery + relevance > 0:
        f1 = 2 * recovery * relevance / (recovery + relevance)
    else:
        f1 = 0.0

    return {
        "f1": float(f1),
        "recovery": float(recovery),
        "relevance": float(relevance),
        "predicted_distribution": {k: len(v) for k, v in pred_groups.items()},
        "reference_distribution": {k: len(v) for k, v in ref_groups.items()},
        "pass": bool(f1 > 0.5),
    }


# =====================================================================
# Composite score
# =====================================================================

def compute_composite_score(results):
    """
    Compute overall trajectory quality score (0-100).

    Eight metrics with nominal weights (sum = 1.0):
        Spearman rho           : 15%  (cell type ordering)
        Kendall markers        : 25%  (biological marker concordance)
        Branch AUC             : 15%  (branch discrimination)
        Variance explained     : 10%  (eta-squared)
        Pairwise separation    : 15%  (Cohen's d effect sizes)
        Pseudotime stats       : 10%  (monotonicity + dynamic range)
        Geodesic correlation   : 5%   (distance correlation)
        F1-Branches            : 5%   (branch assignment)

    If a metric cannot be computed (e.g. separation has no valid pairs),
    it is skipped and the remaining weights are renormalized to sum to 1.
    Both nominal and active weights are recorded in the output for
    transparency.
    """
    scores = {}
    skipped = []

    # 1. Spearman (0-1 scale based on |rho|)
    rho = abs(results["spearman"]["rho"]) if results["spearman"]["rho"] is not None else 0
    scores["spearman"] = min(rho / 0.6, 1.0)  # 0.6 = perfect score threshold

    # 2. Kendall markers (already 0-1 fraction)
    scores["kendall"] = results["kendall"]["score"]

    # 3. Branch AUC (rescale from 0.5-1.0 to 0-1)
    auc = results["branch_auc"]["mean_auc"]
    scores["branch_auc"] = max(0, (auc - 0.5) / 0.5)  # 0.5->0, 1.0->1.0

    # 4. Variance explained (eta² rescale, cap at 0.25)
    eta = results["variance"]["eta_squared"]
    scores["variance"] = min(eta / 0.25, 1.0)

    # 5. Pairwise separation (average |Cohen's d|, cap at 1.5)
    d_vals = [abs(v["cohens_d"]) for v in results["separation"]["per_pair"].values()]
    if d_vals:
        avg_d = np.mean(d_vals)
        scores["separation"] = min(avg_d / 1.5, 1.0)
    else:
        skipped.append("separation")

    # 6. Pseudotime stats (normalized range, cap at 1.5)
    nr = results["pseudotime_stats"]["normalized_range"]
    mono = 1.0 if results["pseudotime_stats"]["monotonic_ordering"] else 0.0
    scores["pt_stats"] = (min(nr / 1.5, 1.0) + mono) / 2

    # 7. Geodesic correlation (|cor|, cap at 0.6)
    gc = abs(results["geodesic"]["cor"]) if results["geodesic"]["cor"] is not None else 0
    scores["geodesic"] = min(gc / 0.6, 1.0)

    # 8. F1-Branches
    f1 = results["f1_branches"]["f1"] if not np.isnan(results["f1_branches"]["f1"]) else 0
    scores["f1_branches"] = f1

    nominal_weights = {
        "spearman": 0.15,
        "kendall": 0.25,
        "branch_auc": 0.15,
        "variance": 0.10,
        "separation": 0.15,
        "pt_stats": 0.10,
        "geodesic": 0.05,
        "f1_branches": 0.05,
    }

    # Compute active weights: renormalize over non-skipped metrics
    active_keys = [k for k in nominal_weights if k not in skipped]
    w_sum = sum(nominal_weights[k] for k in active_keys)
    active_weights = {k: round(nominal_weights[k] / w_sum, 4) for k in active_keys}

    composite = sum(scores[k] * active_weights[k] for k in active_keys)
    return {
        "composite_score": float(composite * 100),
        "sub_scores": {k: float(v * 100) for k, v in scores.items()},
        "weights": nominal_weights,
        "active_weights": active_weights,
        "skipped_metrics": skipped,
        "grade": (
            "A" if composite > 0.8 else
            "B" if composite > 0.6 else
            "C" if composite > 0.4 else
            "D" if composite > 0.2 else
            "F"
        ),
    }


# =====================================================================
# Visualization
# =====================================================================

def plot_evaluation(adata, results, pseudotime_col, celltype_col,
                    branch_cols, label, out_dir):
    """Generate evaluation summary figure."""
    fig = plt.figure(figsize=(20, 16))
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.35)

    composite = results["composite"]
    grade = composite["grade"]
    score = composite["composite_score"]

    fig.suptitle(
        f"Trajectory Evaluation: {label}  |  Score: {score:.1f}/100 (Grade {grade})",
        fontsize=14, fontweight="bold", y=0.98,
    )

    # --- Panel 1: Pseudotime by cell type (violin) ---
    ax1 = fig.add_subplot(gs[0, 0:2])
    ct_data = []
    ct_labels = []
    for ct in list(CELLTYPE_ORDER.keys()):
        mask = adata.obs[celltype_col] == ct
        vals = adata.obs.loc[mask, pseudotime_col].dropna()
        if len(vals) > 0:
            ct_data.append(vals.values)
            ct_labels.append(ct)
    if ct_data:
        parts = ax1.violinplot(ct_data, showmeans=True, showmedians=True)
        ax1.set_xticks(range(1, len(ct_labels) + 1))
        ax1.set_xticklabels(ct_labels, fontsize=7, rotation=30, ha="right")
        ax1.set_ylabel("Pseudotime")
        ax1.set_title("Pseudotime by Cell Type")
        # Add mean labels
        for i, d in enumerate(ct_data):
            ax1.text(i + 1, np.mean(d), f"{np.mean(d):.3f}",
                     ha="center", va="bottom", fontsize=8, color="red")

    # --- Panel 2: UMAP colored by pseudotime ---
    ax2 = fig.add_subplot(gs[0, 2:4])
    if "X_umap" in adata.obsm:
        umap = adata.obsm["X_umap"]
        pt = adata.obs[pseudotime_col].values
        valid = ~np.isnan(pt)
        sc_plot = ax2.scatter(
            umap[valid, 0], umap[valid, 1],
            c=pt[valid], cmap="coolwarm", s=1, alpha=0.5, rasterized=True,
        )
        plt.colorbar(sc_plot, ax=ax2, label="Pseudotime", shrink=0.8)
        ax2.set_xlabel("UMAP1")
        ax2.set_ylabel("UMAP2")
        ax2.set_title("UMAP — Pseudotime")

    # --- Panel 3: Kendall tau per marker ---
    ax3 = fig.add_subplot(gs[1, 0:2])
    kendall = results["kendall"]["per_gene"]
    genes = sorted(kendall.keys(), key=lambda g: abs(kendall[g]["tau"] or 0),
                   reverse=True)
    if genes:
        taus = [kendall[g]["tau"] or 0 for g in genes]
        expected = [kendall[g]["expected_direction"] for g in genes]
        colors = []
        for g in genes:
            if kendall[g]["pass"]:
                colors.append("#2ca02c")  # green = pass
            elif kendall[g]["correct_direction"]:
                colors.append("#ff7f0e")  # orange = correct dir but not sig
            else:
                colors.append("#d62728")  # red = wrong direction
        ax3.barh(range(len(genes)), taus, color=colors)
        ax3.set_yticks(range(len(genes)))
        ax3.set_yticklabels(genes, fontsize=7)
        ax3.set_xlabel("Kendall's tau")
        ax3.set_title(
            f"Marker Gene Concordance ({results['kendall']['n_pass']}/{results['kendall']['n_tested']} pass)"
        )
        ax3.axvline(0, color="k", linewidth=0.5)
        # Legend
        from matplotlib.patches import Patch
        ax3.legend(
            handles=[
                Patch(color="#2ca02c", label="Pass"),
                Patch(color="#ff7f0e", label="Correct dir, not sig"),
                Patch(color="#d62728", label="Wrong direction"),
            ],
            fontsize=7, loc="lower right",
        )

    # --- Panel 4: Pairwise separation (Cohen's d) ---
    ax4 = fig.add_subplot(gs[1, 2])
    sep = results["separation"]["per_pair"]
    pair_names = list(sep.keys())
    d_vals = [sep[p]["cohens_d"] for p in pair_names]
    pair_labels = [p.replace("_vs_", "\nvs\n") for p in pair_names]
    bar_colors = ["#2ca02c" if sep[p]["pass"] else "#d62728" for p in pair_names]
    if pair_names:
        ax4.bar(range(len(pair_names)), d_vals, color=bar_colors)
        ax4.set_xticks(range(len(pair_names)))
        ax4.set_xticklabels(pair_labels, fontsize=7)
        ax4.set_ylabel("Cohen's d")
        ax4.set_title("Pairwise Separation")
        ax4.axhline(0.5, color="gray", linestyle="--", linewidth=0.5, label="d=0.5")
        ax4.axhline(-0.5, color="gray", linestyle="--", linewidth=0.5)

    # --- Panel 5: Sub-scores radar / bar chart ---
    ax5 = fig.add_subplot(gs[1, 3])
    sub = composite["sub_scores"]
    metric_names = list(sub.keys())
    metric_vals = [sub[k] for k in metric_names]
    short_names = [n.replace("_", "\n") for n in metric_names]
    bar_colors_5 = ["#2ca02c" if v >= 50 else "#ff7f0e" if v >= 25 else "#d62728"
                    for v in metric_vals]
    ax5.barh(range(len(metric_names)), metric_vals, color=bar_colors_5)
    ax5.set_yticks(range(len(metric_names)))
    ax5.set_yticklabels(short_names, fontsize=7)
    ax5.set_xlabel("Score (0-100)")
    ax5.set_title(f"Sub-Scores | Composite: {score:.1f}")
    ax5.set_xlim(0, 100)
    ax5.axvline(50, color="gray", linestyle="--", linewidth=0.5)

    # --- Panel 6: Branch probability distributions ---
    ax6 = fig.add_subplot(gs[2, 0:2])
    for col in branch_cols:
        if col not in adata.obs.columns:
            continue
        ct_name = col.replace("prob_", "")
        for ct in list(CELLTYPE_ORDER.keys()):
            mask = adata.obs[celltype_col] == ct
            vals = adata.obs.loc[mask, col].dropna()
            if len(vals) > 0:
                ax6.hist(vals, bins=50, alpha=0.4,
                         label=f"{ct} -> {ct_name}",
                         density=True)
    ax6.set_xlabel("Branch Probability")
    ax6.set_ylabel("Density")
    ax6.set_title("Branch Prob Distribution by Cell Type")
    ax6.legend(fontsize=6, ncol=2)

    # --- Panel 7: Summary text ---
    ax7 = fig.add_subplot(gs[2, 2:4])
    ax7.axis("off")
    summary_lines = [
        f"Method: {label}",
        f"Composite Score: {score:.1f} / 100  (Grade {grade})",
        "",
        f"Spearman rho: {results['spearman']['rho']:.3f} (p={results['spearman']['p_value']:.2e})",
        f"Marker concordance: {results['kendall']['n_pass']}/{results['kendall']['n_tested']} genes",
        f"Branch AUC: {results['branch_auc']['mean_auc']:.3f}",
        f"Variance explained (eta2): {results['variance']['eta_squared']:.4f}",
        f"Monotonic ordering: {'Yes' if results['pseudotime_stats']['monotonic_ordering'] else 'No'}",
        f"Inversion rate: {results['pseudotime_stats']['inversion_rate']}%" if results['pseudotime_stats']['inversion_rate'] is not None else "Inversion rate: N/A",
        f"Group Spearman rho: {results['pseudotime_stats']['group_spearman_rho']:.3f}" if results['pseudotime_stats']['group_spearman_rho'] is not None else "Group Spearman rho: N/A",
        f"Normalized range: {results['pseudotime_stats']['normalized_range']:.3f}",
        f"Geodesic cor: {results['geodesic']['cor']:.3f}" if results['geodesic']['cor'] else "Geodesic cor: N/A",
        f"F1-Branches: {results['f1_branches']['f1']:.3f}",
        "",
        "Pass/Fail:",
    ]
    for metric_name in ["spearman", "kendall", "branch_auc", "variance",
                        "separation", "pseudotime_stats", "geodesic", "f1_branches"]:
        passed = results[metric_name].get("pass", False)
        symbol = "PASS" if passed else "FAIL"
        summary_lines.append(f"  {metric_name:20s} : {symbol}")

    ax7.text(0.05, 0.95, "\n".join(summary_lines),
             transform=ax7.transAxes, fontsize=8, fontfamily="monospace",
             verticalalignment="top")

    out_path = f"{out_dir}/eval_{label.replace(' ', '_').lower()}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


# =====================================================================
# Main evaluation runner
# =====================================================================

def evaluate_trajectory(
    adata,
    pseudotime_col="pseudotime_raw",
    celltype_col="celltype",
    branch_cols=None,
    label="unnamed",
    out_dir=None,
    save_json=True,
    celltype_ordering=None,
):
    """
    Run all trajectory evaluation metrics and generate report.

    Parameters
    ----------
    adata : AnnData
        AnnData with pseudotime in .obs
    pseudotime_col : str
        Column name for pseudotime values
    celltype_col : str
        Column name for cell type annotations
    branch_cols : list of str
        Column names for branch probabilities (e.g. ["prob_B1", "prob_PLASMA_B"])
    label : str
        Label for this trajectory method (used in filenames and plots)
    out_dir : str
        Output directory for plots and JSON report
    save_json : bool
        Whether to save JSON report

    Returns
    -------
    dict : All metric results
    """
    if branch_cols is None:
        branch_cols = [c for c in adata.obs.columns if c.startswith("prob_")]

    print(f"\n{'='*70}")
    print(f"  TRAJECTORY EVALUATION: {label}")
    print(f"{'='*70}")
    print(f"  Cells: {adata.shape[0]:,}")
    print(f"  Pseudotime col: {pseudotime_col}")
    print(f"  Cell type col: {celltype_col}")
    print(f"  Branch cols: {branch_cols}")
    print()

    results = {}

    # Use custom ordering if provided, else fall back to built-in CELLTYPE_ORDER
    ordering = celltype_ordering if celltype_ordering is not None else CELLTYPE_ORDER

    # Metric 1
    print("--- Metric 1: Spearman Rank Correlation ---")
    results["spearman"] = metric_spearman_celltype(
        adata.obs, pseudotime_col, celltype_col, ordering=ordering
    )
    r = results["spearman"]
    print(f"  rho = {r['rho']:.4f}, p = {r['p_value']:.2e}, n = {r['n']}")
    print(f"  {'PASS' if r['pass'] else 'FAIL'}")
    print()

    # Metric 2
    print("--- Metric 2: Kendall Tau with Marker Genes ---")
    results["kendall"] = metric_kendall_markers(adata, pseudotime_col)
    k = results["kendall"]
    for gene, info in sorted(k["per_gene"].items(),
                              key=lambda x: abs(x[1]["tau"] or 0), reverse=True):
        tau_str = f"{info['tau']:.4f}" if info["tau"] is not None else "  N/A "
        p_str = f"{info['p_value']:.4f}" if info["p_value"] is not None else " N/A  "
        dir_str = {1: "+", -1: "-", 0: "o"}[info["expected_direction"]]
        status = "PASS" if info["pass"] else "FAIL"
        print(f"  {gene:15s}  tau={tau_str}  p={p_str}  exp={dir_str}  [{status}]")
    print(f"  Score: {k['n_pass']}/{k['n_tested']} markers pass ({k['score']*100:.0f}%)")
    print(f"  {'PASS' if k['pass'] else 'FAIL'}")
    print()

    # Metric 3
    print("--- Metric 3: Branch Probability AUC ---")
    results["branch_auc"] = metric_branch_auc(
        adata.obs, celltype_col, branch_cols
    )
    ba = results["branch_auc"]
    for ct, info in ba["per_branch"].items():
        print(f"  {ct}: AUC = {info['auc']:.4f}  [{('PASS' if info['pass'] else 'FAIL')}]")
    print(f"  Mean AUC: {ba['mean_auc']:.4f}")
    print(f"  {'PASS' if ba['pass'] else 'FAIL'}")
    print()

    # Metric 4
    print("--- Metric 4: Variance Explained (eta2) ---")
    results["variance"] = metric_variance_explained(
        adata.obs, pseudotime_col, celltype_col
    )
    v = results["variance"]
    print(f"  eta2 = {v['eta_squared']:.6f}")
    print(f"  Kruskal-Wallis H = {v['kruskal_h']}, p = {v['kruskal_p']}")
    print(f"  {'PASS' if v['pass'] else 'FAIL'}")
    print()

    # Metric 5
    print("--- Metric 5: Pairwise Separation (Cohen's d) ---")
    results["separation"] = metric_pairwise_separation(
        adata.obs, pseudotime_col, celltype_col
    )
    s = results["separation"]
    for pair, info in s["per_pair"].items():
        print(f"  {pair}: d = {info['cohens_d']:.4f}, "
              f"AUROC = {info['auroc_effect_size']:.4f}, "
              f"p = {info['mann_whitney_p']:.2e}  "
              f"[{'PASS' if info['pass'] else 'FAIL'}]")
    print()

    # Metric 6
    print("--- Metric 6: Pseudotime Distribution Stats ---")
    results["pseudotime_stats"] = metric_pseudotime_stats(
        adata.obs, pseudotime_col, celltype_col, ordering=ordering
    )
    ps = results["pseudotime_stats"]
    for ct, info in ps["per_celltype"].items():
        print(f"  {ct:25s}: mean={info['mean']:.4f}  "
              f"med={info['median']:.4f}  std={info['std']:.4f}  n={info['n']}")
    print(f"  Monotonic ordering: {ps['monotonic_ordering']}")
    print(f"  Normalized range: {ps['normalized_range']:.4f}")
    inv_str = f"{ps['inversion_rate']:.1f}%" if ps['inversion_rate'] is not None else "N/A"
    print(f"  Inversion rate: {inv_str}")
    grho_str = f"{ps['group_spearman_rho']:.4f}" if ps['group_spearman_rho'] is not None else "N/A"
    print(f"  Group-level Spearman rho: {grho_str}")
    print(f"  {'PASS' if ps['pass'] else 'FAIL'}")
    print()

    # Metric 7
    print("--- Metric 7: Geodesic Distance Correlation ---")
    results["geodesic"] = metric_geodesic_correlation(
        adata.obs, pseudotime_col, celltype_col, ordering=ordering
    )
    g = results["geodesic"]
    cor_str = f"{g['cor']:.4f}" if g["cor"] is not None else "N/A"
    print(f"  cor = {cor_str}, n_waypoints = {g['n_waypoints']}")
    print(f"  {'PASS' if g['pass'] else 'FAIL'}")
    print()

    # Metric 8
    print("--- Metric 8: F1-Branches ---")
    results["f1_branches"] = metric_f1_branches(
        adata.obs, celltype_col, branch_cols
    )
    fb = results["f1_branches"]
    print(f"  F1 = {fb['f1']:.4f}, Recovery = {fb.get('recovery', 'N/A')}, "
          f"Relevance = {fb.get('relevance', 'N/A')}")
    if "predicted_distribution" in fb:
        print(f"  Predicted: {fb['predicted_distribution']}")
        print(f"  Reference: {fb['reference_distribution']}")
    print(f"  {'PASS' if fb['pass'] else 'FAIL'}")
    print()

    # Composite score
    print("--- Composite Score ---")
    results["composite"] = compute_composite_score(results)
    c = results["composite"]
    print(f"  GRADE: {c['grade']}")
    print(f"  SCORE: {c['composite_score']:.1f} / 100")
    print(f"  Sub-scores:")
    for name, val in c["sub_scores"].items():
        print(f"    {name:20s}: {val:.1f}")
    print()

    # Save results
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

        # Plot
        plot_path = plot_evaluation(
            adata, results, pseudotime_col, celltype_col,
            branch_cols, label, out_dir
        )
        print(f"  Plot saved: {plot_path}")

        # JSON report
        if save_json:
            json_path = f"{out_dir}/eval_{label.replace(' ', '_').lower()}.json"
            # Make JSON-serializable
            def _clean(obj):
                if isinstance(obj, dict):
                    return {k: _clean(v) for k, v in obj.items()}
                elif isinstance(obj, (np.integer,)):
                    return int(obj)
                elif isinstance(obj, (np.floating,)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.bool_,)):
                    return bool(obj)
                elif obj is None or isinstance(obj, (str, int, float, bool, list)):
                    return obj
                else:
                    return str(obj)

            report = {
                "label": label,
                "timestamp": datetime.now().isoformat(),
                "n_cells": adata.shape[0],
                "pseudotime_col": pseudotime_col,
                "celltype_col": celltype_col,
                "branch_cols": branch_cols,
                "celltype_order": ordering,
                "metrics": _clean(results),
            }
            with open(json_path, "w") as f:
                json.dump(report, f, indent=2)
            print(f"  JSON saved: {json_path}")

    return results


# =====================================================================
# CLI entry point
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate BCR trajectory quality"
    )
    parser.add_argument("h5ad_path", help="Path to AnnData with trajectory results")
    parser.add_argument("--pseudotime_col", default="pseudotime_raw",
                        help="Column name for pseudotime")
    parser.add_argument("--celltype_col", default="celltype",
                        help="Column name for cell type annotations")
    parser.add_argument("--branch_cols", default=None,
                        help="Comma-separated branch probability columns")
    parser.add_argument("--label", default="unnamed",
                        help="Label for this trajectory method")
    parser.add_argument("--out_dir", default=None,
                        help="Output directory (default: same as h5ad)")
    parser.add_argument("--celltype_order", default=None,
                        help=(
                            "Comma-separated cell types in expected developmental "
                            "order (earliest to latest). Overrides the built-in "
                            "CELLTYPE_ORDER for Spearman, pseudotime stats, and "
                            "geodesic metrics. Required for non-fetal-B datasets."
                        ))
    args = parser.parse_args()

    adata = sc.read_h5ad(args.h5ad_path)
    print(f"Loaded: {args.h5ad_path} ({adata.shape[0]:,} cells)")

    if "X_umap" not in adata.obsm and "umap_1" in adata.obs.columns:
        adata.obsm["X_umap"] = adata.obs[["umap_1", "umap_2"]].values

    branch_cols = args.branch_cols.split(",") if args.branch_cols else None
    out_dir = args.out_dir or os.path.dirname(args.h5ad_path)

    # Build custom ordering dict from CLI if provided
    custom_ordering = None
    if args.celltype_order:
        ct_list = [ct.strip() for ct in args.celltype_order.split(",") if ct.strip()]
        custom_ordering = {ct: i for i, ct in enumerate(ct_list)}

    evaluate_trajectory(
        adata,
        pseudotime_col=args.pseudotime_col,
        celltype_col=args.celltype_col,
        branch_cols=branch_cols,
        label=args.label,
        out_dir=out_dir,
        celltype_ordering=custom_ordering,
    )
