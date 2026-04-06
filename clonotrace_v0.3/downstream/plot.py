# a) 网格速度场（参考 spaTrack 的 get_velocity_grid）

import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns

def get_velocity_grid_masked(
    coords: np.ndarray,
    V: np.ndarray,
    grid_num: int = 50,
    smooth: float = 0.5,
    density: float = 1.0,
    max_dist_factor: float = 2.0,
):
    """
    在 2D 坐标上构建带掩膜的速度网格，只在细胞附近插值速度。

    参数
    ----
    coords : (n_cells, 2) 细胞的 2D 坐标（UMAP/joint）
    V      : (n_cells, 2) 对应的速度向量
    grid_num : 基础网格数（越大越细）
    smooth   : 控制高斯加权的平滑尺度
    density  : 额外放大/缩小网格密度
    max_dist_factor : 网格点到最近细胞的距离阈值因子
                      d_min > max_dist_factor * d0 时认为无效（d0 是全局最近邻距离的中位数）

    返回
    ----
    Xg, Yg : 2D 网格坐标
    Ug, Vg : 2D 速度分量（在无效区域置 0）
    mask_valid : bool 数组，True 表示该网格点靠近细胞
    """
    coords = np.asarray(coords, float)
    V = np.asarray(V, float)

    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()

    nx = int(grid_num * density)
    ny = int(grid_num * density)

    x_grid = np.linspace(x_min, x_max, nx)
    y_grid = np.linspace(y_min, y_max, ny)
    Xg, Yg = np.meshgrid(x_grid, y_grid)
    grid_points = np.vstack([Xg.ravel(), Yg.ravel()]).T

    # KNN 拿到邻居和距离
    nbrs = NearestNeighbors(n_neighbors=30, metric="euclidean", n_jobs=-1).fit(coords)
    dist, idx = nbrs.kneighbors(grid_points, return_distance=True)

    # 估计一个“典型细胞间距” d0，用于决定多远算“没有细胞”
    # 用细胞之间的最近邻距离的中位数
    nbrs_cells = NearestNeighbors(n_neighbors=2, metric="euclidean", n_jobs=-1).fit(coords)
    dist_cells, _ = nbrs_cells.kneighbors(coords, return_distance=True)
    d0 = np.median(dist_cells[:, 1])  # 每个细胞到最近其它细胞的距离

    d_min = dist[:, 0]  # 每个网格点到最近细胞的距离
    dist_threshold = max_dist_factor * d0
    mask_valid = d_min <= dist_threshold

    # 高斯权重（只对有效网格点做插值）
    sigma = smooth * np.mean(dist[mask_valid, 0]) if np.any(mask_valid) else 1.0
    w = np.exp(-(dist ** 2) / (2 * sigma ** 2))
    w_sum = w.sum(axis=1, keepdims=True) + 1e-9
    w_norm = w / w_sum

    V_neighbors = V[idx]  # (n_grid, k, 2)
    Vg_all = (V_neighbors * w_norm[..., None]).sum(axis=1)  # (n_grid, 2)

    # 对无效网格点速度设成 0（或者 np.nan），但保留 mask，方便绘图时处理
    Vg_all[~mask_valid] = 0.0

    Ug = Vg_all[:, 0].reshape(Xg.shape)
    Vg = Vg_all[:, 1].reshape(Xg.shape)
    mask_valid_grid = mask_valid.reshape(Xg.shape)

    return Xg, Yg, Ug, Vg, mask_valid_grid

def boxplot_by_median(adata, group_col, time_col, figsize=(6, 4), palette=None):
    """
    按 celltype 绘制 time_col 的箱型图，x 轴按该列中位数排序

    Parameters
    ----------
    adata : AnnData
    time_col : str
        adata.obs 中待绘制的连续变量列名（如 'pseudotime_raw'）
    figsize : tuple, optional
    palette : seaborn palette, optional
    """
    # 计算中位数并排序
    order = (adata.obs
             .groupby(group_col)[time_col]
             .median()
             .sort_values()
             .index)

    plt.figure(figsize=figsize)
    sns.boxplot(
        data=adata.obs,
        x=group_col,
        y=time_col,
        order=order,
        palette=palette
    )
    plt.title(f'{time_col} by {group_col} (ordered by median)')
    plt.ylabel(time_col)
    plt.xlabel(group_col)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()