#!/usr/bin/env python
"""Q-score bootstrap stability: 80% subsample × N iterations, Spearman ρ."""
import argparse
import json
import os
import numpy as np


def qscore_bootstrap(adata, n_iter=100, subsample_frac=0.8, q_score_key="q_score",
                     q_tech_key="q_tech_bcr", q_bio_key="q_bio_bcr",
                     stretch_method="root_g2", seed=42):
    """
    Bootstrap stability of q_score computation.

    Subsamples cells, recomputes q_score, and measures Spearman ρ
    between full and subsampled q_score on the shared cells.

    Returns dict with correlations, mean, std, 95% CI.
    """
    from scipy.stats import spearmanr
    from btraj.qscore.compute_q_score import compute_q_score

    rng = np.random.RandomState(seed)
    n_cells = adata.n_obs
    n_sub = int(n_cells * subsample_frac)

    # Full q_score
    full_q = np.asarray(adata.obs[q_score_key], dtype=float).copy()

    correlations = []
    for i in range(n_iter):
        idx = rng.choice(n_cells, size=n_sub, replace=False)
        sub = adata[idx].copy()

        compute_q_score(
            sub, q_tech_key=q_tech_key, q_bio_key=q_bio_key,
            out_q_score_key="_q_boot", stretch_method=stretch_method,
        )

        q_boot = np.asarray(sub.obs["_q_boot"], dtype=float)
        q_full_sub = full_q[idx]

        valid = np.isfinite(q_boot) & np.isfinite(q_full_sub)
        if valid.sum() > 10:
            rho, _ = spearmanr(q_full_sub[valid], q_boot[valid])
            correlations.append(float(rho))

    correlations = np.array(correlations)
    ci_lo, ci_hi = np.percentile(correlations, [2.5, 97.5])

    return {
        "n_iter": n_iter,
        "subsample_frac": subsample_frac,
        "mean_rho": float(np.mean(correlations)),
        "std_rho": float(np.std(correlations)),
        "ci_95_lo": float(ci_lo),
        "ci_95_hi": float(ci_hi),
        "correlations": correlations.tolist(),
    }


def plot_bootstrap(results, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(results["correlations"], bins=25, edgecolor="black", alpha=0.7)
    ax.axvline(results["mean_rho"], color="red", linestyle="--",
               label=f"Mean ρ = {results['mean_rho']:.4f}")
    ax.set_xlabel("Spearman ρ (full vs subsampled q_score)")
    ax.set_ylabel("Count")
    ax.set_title(f"Q-Score Bootstrap Stability (n={results['n_iter']})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Q-score bootstrap stability analysis")
    parser.add_argument("--h5ad", required=True, help="Path to AnnData h5ad file")
    parser.add_argument("--n-iter", type=int, default=100, help="Number of bootstrap iterations")
    parser.add_argument("--subsample-frac", type=float, default=0.8)
    parser.add_argument("--q-score-key", default="q_score")
    parser.add_argument("--stretch-method", default="root_g2")
    parser.add_argument("--out-dir", default="results/qscore_bootstrap")
    args = parser.parse_args()

    import anndata
    adata = anndata.read_h5ad(args.h5ad)

    results = qscore_bootstrap(
        adata, n_iter=args.n_iter, subsample_frac=args.subsample_frac,
        q_score_key=args.q_score_key, stretch_method=args.stretch_method,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "qscore_bootstrap.json"), "w") as f:
        json.dump({k: v for k, v in results.items() if k != "correlations"}, f, indent=2)

    plot_bootstrap(results, os.path.join(args.out_dir, "qscore_bootstrap.pdf"))
    print(f"Mean ρ = {results['mean_rho']:.4f} ± {results['std_rho']:.4f}")
    print(f"95% CI: [{results['ci_95_lo']:.4f}, {results['ci_95_hi']:.4f}]")


if __name__ == "__main__":
    main()
