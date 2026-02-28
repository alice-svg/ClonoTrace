
import numpy as np
from typing import Any, Optional

import scipy.sparse as sps
from scipy.sparse import csr_matrix

from cellrank.kernels.utils._pseudotime_scheme import ThresholdSchemeABC
from cellrank._utils._parallelize import parallelize


class SkeletonSoftScheme(ThresholdSchemeABC):
    """
    Soft-thresholding scheme that combines pseudotime bias with a
    cluster-level lineage skeleton prior.

    Parameters
    ----------
    clusters
        1D array of length n_cells, *integer* cluster labels for each cell.
        Must be in [0, n_clusters-1] and consistent with the indexing of
        `mst_adj`.
    mst_adj
        2D array of shape (n_clusters, n_clusters), giving the *shortest-path
        distance* on the lineage skeleton between two clusters. The diagonal
        is expected to be 0. Larger values mean clusters are further apart.
    lambda_skel
        Strength of the skeleton penalty. Larger -> stronger restriction to
        lineage-neighboring clusters.
    b, nu
        Parameters of the generalized logistic function for past-edge
        downweighting (same meaning as in CellRank SoftThresholdScheme).
    """

    def __init__(
        self,
        clusters: np.ndarray,        # (n_cells,) cluster ids (int, 0..n_clust-1)
        mst_adj: np.ndarray,         # (n_clust, n_clust) shortest-path distances
        lambda_skel: float = 2.0,    # skeleton penalty strength
        b: float = 10.0,             # logistic slope
        nu: float = 0.5,             # logistic power
    ):
        super().__init__()

        clusters = np.asarray(clusters)
        mst_adj = np.asarray(mst_adj, dtype=float)

        if clusters.ndim != 1:
            raise ValueError(f"`clusters` must be 1D, found shape {clusters.shape}.")

        if mst_adj.ndim != 2 or mst_adj.shape[0] != mst_adj.shape[1]:
            raise ValueError(
                "`mst_adj` must be a square 2D matrix of shape (n_clusters, n_clusters), "
                f"found shape {mst_adj.shape}."
            )

        # 确保簇编号是从 0..n_clust-1 的整数，并与 mst_adj 对齐
        if not np.issubdtype(clusters.dtype, np.integer):
            raise TypeError(
                "`clusters` must be integer-coded (0..n_clusters-1) and consistent "
                "with the indexing of `mst_adj`."
            )

        if clusters.min() < 0:
            raise ValueError(
                f"`clusters` contains negative indices, found min={clusters.min()}."
            )

        n_clust = mst_adj.shape[0]
        if clusters.max() >= n_clust:
            raise ValueError(
                f"`clusters` has max={clusters.max()}, but `mst_adj` has size {n_clust}. "
                "They must be consistent."
            )

        self._clusters = clusters.astype(np.int32)
        self._mst_adj = mst_adj
        self._lambda_skel = float(lambda_skel)
        self._b = float(b)
        self._nu = float(nu)

    # ---------------------------------------------------------------------
    # 自定义的 bias_knn：唯一目的就是把 cell_index / neigh_indices 传给 __call__
    # ---------------------------------------------------------------------
    def bias_knn(
        self,
        conn: csr_matrix,
        pseudotime: np.ndarray,
        n_jobs: Optional[int] = None,
        backend: str = "loky",
        show_progress_bar: bool = True,
        **kwargs: Any,
    ) -> csr_matrix:
        """
        Bias cell-cell connectivities of a KNN graph, with pseudotime and
        skeleton constraints.

        与父类功能类似，但会将 `cell_index` 和 `neigh_indices` 传给 `__call__`,
        以便在阈值 scheme 内访问簇信息和 MST 距离。
        """
        if not sps.isspmatrix_csr(conn):
            conn = conn.tocsr()

        pseudotime = np.asarray(pseudotime)
        n_cells = conn.shape[0]
        if pseudotime.shape[0] != n_cells:
            raise ValueError(
                f"`pseudotime` length {pseudotime.shape[0]} does not match "
                f"number of cells {n_cells}."
            )
        if self._clusters.shape[0] != n_cells:
            raise ValueError(
                f"`clusters` length {self._clusters.shape[0]} does not match "
                f"number of cells {n_cells}."
            )

        def _worker(ixs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            i_last = 0
            indices: list[int] = []
            indptr: list[int] = []
            data: list[float] = []

            for i in ixs:
                row = conn[i]
                neigh_ix = row.indices

                # 关键：把 cell_index=i 和 neigh_indices=neigh_ix 传进去
                biased_row = self(
                    pseudotime[i],
                    pseudotime[neigh_ix],
                    row.data,
                    cell_index=i,
                    neigh_indices=neigh_ix,
                    **kwargs,
                )

                if np.shape(biased_row) != row.data.shape:
                    raise ValueError(
                        f"Expected row of shape {row.data.shape}, "
                        f"found {np.shape(biased_row)}."
                    )

                data.extend(biased_row)
                indices.extend(neigh_ix)
                indptr.append(conn.indptr[i])
                i_last = i

            # 为该分块补上最后一个指针
            # 注意：这里保持与原 CSR 一致，每行的开始位置来自原始 conn.indptr
            if i_last == conn.shape[0] - 1:
                indptr.append(conn.indptr[-1])

            return (
                np.asarray(data, dtype=float),
                np.asarray(indices, dtype=int),
                np.asarray(indptr, dtype=int),
            )

        # 调用 CellRank 的并行工具
        res = parallelize(
            _worker,
            np.arange(n_cells),
            as_array=False,
            unit="cell",
            n_jobs=n_jobs,
            backend=backend,
            show_progress_bar=show_progress_bar,
        )()
        data, indices, indptr = zip(*res)

        data_cat = np.concatenate(data)
        indices_cat = np.concatenate(indices)
        indptr_cat = np.concatenate(indptr)

        biased = sps.csr_matrix(
            (data_cat, indices_cat, indptr_cat),
            shape=conn.shape,
        )
        biased.eliminate_zeros()

        # ===== 处理全 0 行：补自环 =====
        row_sum = np.asarray(biased.sum(axis=1)).ravel()
        zero_rows = np.where(row_sum == 0)[0]
        if zero_rows.size > 0:
            biased = biased.tolil()
            for i in zero_rows:
                biased[i, i] = 1.0
            biased = biased.tocsr()
            row_sum = np.asarray(biased.sum(axis=1)).ravel()

        # ===== 按行归一化（真正的随机游走矩阵）：每行和为 1 =====
        rows = np.repeat(np.arange(biased.shape[0]), np.diff(biased.indptr))
        biased.data /= row_sum[rows]
        biased.eliminate_zeros()

        return biased

    # ---------------------------------------------------------------------
    # 骨架权重：根据 MST 上的最短路径距离进行惩罚
    # ---------------------------------------------------------------------
    def _skeleton_factor(self, ci: int, neigh_clusters: np.ndarray) -> np.ndarray:
        """
        返回与 neigh_clusters 同长度的权重数组，对应每个邻居细胞所属簇。

        使用基于最短路径距离的指数衰减：
            w = exp(-lambda_skel * d_eff)

        其中 d_eff 是对原始 mst_adj 距离稳定缩放 + 截断后的有效距离。
        """
        neigh_clusters = np.asarray(neigh_clusters, dtype=int)

        # 原始最短路径距离（可能非常大，比如 1e5 级）
        d_raw = self._mst_adj[ci, neigh_clusters].astype(float)  # (n_neighbors,)

        # ------- 数值稳定缩放策略 -------
        # 1. 只在全局非零距离上估计一个代表尺度，减少噪声敏感性
        all_nonzero = self._mst_adj[self._mst_adj > 0]
        if all_nonzero.size > 0:
            # 用对数中位数来缓解极端值影响
            log_med = np.median(np.log(all_nonzero))
            scale = np.exp(log_med)
            if scale <= 0:
                scale = 1.0
        else:
            scale = 1.0

        # 2. 归一化到一个比较小的尺度（例如 0~几），避免 exp(-lambda * d) 下溢
        d_norm = d_raw / scale

        # 3. 对非常远的簇做截断：超过 d_cap 的都看成同样“非常远”
        #    例如 d_norm > 5 时，exp(-lambda * d) 已经非常接近 0
        d_cap = 5.0
        d_eff = np.minimum(d_norm, d_cap)

        # ------- 指数衰减权重 -------
        # 同簇 d_raw=0 -> d_eff=0 -> w=1
        w = np.exp(-self._lambda_skel * d_eff)

        return w

    # ---------------------------------------------------------------------
    # 主体 scheme：伪时间 soft 惩罚 + 骨架惩罚
    # ---------------------------------------------------------------------
    def __call__(
        self,
        cell_pseudotime: float,
        neigh_pseudotime: np.ndarray,
        neigh_conn: np.ndarray,
        cell_index: Optional[int] = None,
        neigh_indices: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Skeleton-aware soft thresholding of kNN connectivities.

        Parameters
        ----------
        cell_pseudotime
            Pseudotime of the current cell.
        neigh_pseudotime
            Pseudotime of neighbors (1D array, same length as neigh_conn).
        neigh_conn
            kNN connectivities to neighbors (1D array).
        cell_index
            Index of current cell in `adata`.
        neigh_indices
            Indices of neighbor cells in `adata`.

        Returns
        -------
        Biased connectivities (1D array of same shape as `neigh_conn`).
        """
        neigh_pseudotime = np.asarray(neigh_pseudotime, dtype=float)
        neigh_conn = np.asarray(neigh_conn, dtype=float)

        if neigh_pseudotime.shape != neigh_conn.shape:
            raise ValueError(
                f"`neigh_pseudotime` shape {neigh_pseudotime.shape} does not "
                f"match `neigh_conn` shape {neigh_conn.shape}."
            )

        weights = np.ones_like(neigh_conn, dtype=float)

        # --- 1) 伪时间 soft 惩罚（和 CellRank 的 SoftThresholdScheme 类似）---
        past_ixs = np.where(neigh_pseudotime < cell_pseudotime)[0]
        if past_ixs.size > 0:
            dt = cell_pseudotime - neigh_pseudotime[past_ixs]  # > 0

            # 数值稳定：裁剪 exponent，避免 exp 溢出
            x = self._b * dt
            x = np.clip(x, -60.0, 60.0)  # 防止 overflow / underflow
            weights[past_ixs] = 2.0 / ((1.0 + np.exp(x)) ** (1.0 / self._nu))

        # --- 2) skeleton gating based on cluster-level lineage ---
        if cell_index is not None and neigh_indices is not None:
            ci = int(self._clusters[cell_index])
            neigh_clusters = self._clusters[np.asarray(neigh_indices, dtype=int)]
            skel_w = self._skeleton_factor(ci, neigh_clusters)
            weights *= skel_w

        # 返回最终的 biased connectivities
        return neigh_conn * weights
