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
    Build a masked velocity grid on 2D coordinates, interpolating velocity
    only near cells.

    Parameters
    ----------
    coords : (n_cells, 2) 2D cell coordinates (UMAP/joint)
    V      : (n_cells, 2) corresponding velocity vectors
    grid_num : base grid resolution (higher = finer)
    smooth   : smoothing scale for Gaussian weighting
    density  : additional scaling factor for grid density
    max_dist_factor : distance threshold factor from grid point to nearest cell;
                      grid points where d_min > max_dist_factor * d0 are considered
                      invalid (d0 is the median nearest-neighbor distance across cells)

    Returns
    -------
    Xg, Yg : 2D grid coordinates
    Ug, Vg : 2D velocity components (set to 0 in invalid regions)
    mask_valid : bool array, True indicates the grid point is near a cell
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

    # KNN to get neighbors and distances
    nbrs = NearestNeighbors(n_neighbors=30, metric="euclidean", n_jobs=-1).fit(coords)
    dist, idx = nbrs.kneighbors(grid_points, return_distance=True)

    # Estimate a "typical inter-cell distance" d0 to determine how far is "no cell nearby"
    # Use the median of each cell's nearest-neighbor distance to another cell

    nbrs_cells = NearestNeighbors(n_neighbors=2, metric="euclidean", n_jobs=-1).fit(coords)
    dist_cells, _ = nbrs_cells.kneighbors(coords, return_distance=True)
    d0 = np.median(dist_cells[:, 1])  # distance from each grid point to its nearest cell

    d_min = dist[:, 0]
    dist_threshold = max_dist_factor * d0
    mask_valid = d_min <= dist_threshold

    # Gaussian weights (only interpolate at valid grid points)
    sigma = smooth * np.mean(dist[mask_valid, 0]) if np.any(mask_valid) else 1.0
    w = np.exp(-(dist ** 2) / (2 * sigma ** 2))
    w_sum = w.sum(axis=1, keepdims=True) + 1e-9
    w_norm = w / w_sum

    V_neighbors = V[idx]  # (n_grid, k, 2)
    Vg_all = (V_neighbors * w_norm[..., None]).sum(axis=1)  # (n_grid, 2)

    # Set velocity to 0 (or np.nan) at invalid grid points, but keep mask for plotting

    Vg_all[~mask_valid] = 0.0

    Ug = Vg_all[:, 0].reshape(Xg.shape)
    Vg = Vg_all[:, 1].reshape(Xg.shape)
    mask_valid_grid = mask_valid.reshape(Xg.shape)

    return Xg, Yg, Ug, Vg, mask_valid_grid

def boxplot_by_median(adata, group_col, time_col, figsize=(6, 4), palette=None):
    """
    Draw a boxplot of time_col grouped by celltype, with x-axis ordered by median.

    Parameters
    ----------
    adata : AnnData
    time_col : str
        Name of the continuous variable column in adata.obs to plot (e.g. 'pseudotime_raw')
    figsize : tuple, optional
    palette : seaborn palette, optional
    """

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