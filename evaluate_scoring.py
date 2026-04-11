import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import kendalltau, spearmanr

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["font.family"] = "Arial"
plt.rcParams["pdf.fonttype"] = 42

def evaluate_pseudotime_order(
    meta,
    celltype_col: str = "celltype",
    pseudotime_cols=("pseudotime",),
    order=None,
    figsize=(7, 4),
    sort_by="spearman",  # 'spearman' | 'kendall' | 'inversion'
    save_path: str | None = None,
):

    if order is None:
        raise ValueError("order (list) of celltypes must be provided.")

    label2rank = {ct: i for i, ct in enumerate(order)}
    results = []

    obs = meta[[celltype_col] + list(pseudotime_cols)].copy()
    obs["gold"] = obs[celltype_col].map(label2rank)
    obs = obs.dropna(subset=["gold"]).astype({"gold": int})
    index_order = pd.Index(order, name=celltype_col)

    for col in pseudotime_cols:
        df = obs[[celltype_col, "gold", col]].dropna(subset=[col])
        if df.empty:
            tau = rho = inv_rate = np.nan
        else:
            tau, _ = kendalltau(df["gold"], df[col])

            med = df.groupby(celltype_col)[col].median().reindex(index_order).dropna()
            ranks = np.arange(len(med))
            rho, _ = spearmanr(ranks, med.values)

            pairs = [(i, j) for i in range(len(med)) for j in range(i + 1, len(med))]
            inver = sum(1 for i, j in pairs if med.iloc[i] > med.iloc[j])
            inv_rate = 100 * inver / len(pairs) if pairs else np.nan

        results.append(
            {"pseudotime_col": col, "kendall_tau": tau,
             "spearman_rho": rho, "inversion_rate": inv_rate}
        )

    metrics_df = pd.DataFrame(results)

    sort_map = {"spearman": ("spearman_rho", False),
                "kendall": ("kendall_tau", False),
                "inversion": ("inversion_rate", True)}
    if sort_by not in sort_map:
        raise ValueError("sort_by must be 'spearman', 'kendall', or 'inversion'.")
    score_col, asc = sort_map[sort_by]
    metrics_df = metrics_df.sort_values(score_col, ascending=asc)

    palette = sns.color_palette("coolwarm_r", len(metrics_df))
    plt.figure(figsize=figsize)
    ax = sns.barplot(
        data=metrics_df,
        y="pseudotime_col",
        x=score_col,
        palette=palette,
        linewidth=0,
    )
    sns.stripplot(
        data=metrics_df,
        y="pseudotime_col",
        x=score_col,
        color="black",
        size=4,
        jitter=False,
        ax=ax,
    )
    sns.despine(left=True, bottom=True)
    ax.set_ylabel("")
    ax.set_xlabel(score_col.replace("_", " ").title())
    ax.set_title(f"Trajectory accuracy ({score_col.replace('_', ' ').title()})", pad=15)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    return metrics_df

### Specify the true developmental order here
order = [
    'PRE_PRO_B',
    'PRO_B',
    'LATE_PRO_B',
    'LARGE_PRE_B',
    'SMALL_PRE_B',
    'IMMATURE_B',
    'MATURE_B',
    'B1',
    'PLASMA_B'
] ### True developmental order, summarized from literature

### sota: each row is the pseudotime median of a cell type,
###       each column is the result from a different method
sota = pd.read_csv('./data/fetal_b_cells_pseudotime_sota.csv')
metrics = evaluate_pseudotime_order(
    sota,
    pseudotime_cols=['monocle3_pseudotime', 'palantir_pseudotime',
        'dpt_pseudotime', 'via_pseudotime',
       'ours_pseudotime'],
    order=order,
    # sort_by="spearman",
    sort_by='kendall', ## different sorting methods can be selected here
    save_path="./figs/fetal_b_cell_pseudotime_accuracy_kendall.pdf"
)

print(metrics)