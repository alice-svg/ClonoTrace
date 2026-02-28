from typing import Any, Optional
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from anndata import AnnData

from cellrank.kernels import PseudotimeKernel
from cellrank._utils._enum import Backend_t, DEFAULT_BACKEND
from cellrank._utils._utils import _connected, _irreducible
from cellrank._utils._parallelize import parallelize
from sklearn.neighbors import NearestNeighbors
from cellrank import logging as logg
from cellrank.kernels._base_kernel import BidirectionalKernel
from cellrank.kernels.mixins import ConnectivityMixin

# 导入 SkeletonSoftScheme
from btraj.kernels.bias_q import SkeletonSoftScheme


class LineagePriorPseudotimeKernel(PseudotimeKernel):
    """
    Pseudotime kernel that incorporates a cluster-level lineage skeleton
    as a strong prior, while still allowing CellRank to learn transitions
    at single-cell resolution.

    Parameters
    ----------
    adata
        Annotated data matrix.
    time_key
        Key in `adata.obs` for pseudotime.
    clusters_key
        Key in `adata.obs` for cluster labels (must be integer-coded and
        consistent with `mst_dist` indices).
    mst_dist
        Precomputed (n_clusters x n_clusters) shortest-path distance matrix
        on the lineage skeleton (e.g. q-MST).
    backward
        As in `PseudotimeKernel`.
    """

    def __init__(
        self,
        adata: AnnData,
        time_key: str,
        clusters_key: str,
        mst_dist: np.ndarray,
        backward: bool = False,
        **kwargs: Any,
    ):
        self._clusters_key = clusters_key
        self._mst_dist = np.asarray(mst_dist, dtype=float)

        super().__init__(
            adata=adata,
            time_key=time_key,
            backward=backward,
            **kwargs,
        )

        if clusters_key not in self.adata.obs:
            raise KeyError(f"Unable to find clusters in `adata.obs[{clusters_key!r}]`.")

        clusters = np.asarray(self.adata.obs[clusters_key])
        if clusters.ndim != 1:
            raise ValueError(
                f"`adata.obs[{clusters_key!r}]` must be 1D, found shape {clusters.shape}."
            )

        # 强制要求已经是整数编码，且与 mst_dist 的索引对应
        if not np.issubdtype(clusters.dtype, np.integer):
            raise TypeError(
                f"`adata.obs[{clusters_key!r}]` must contain integer cluster IDs "
                "consistent with `mst_dist` indexing."
            )

        n_clust = self._mst_dist.shape[0]
        if self._mst_dist.ndim != 2 or self._mst_dist.shape[0] != self._mst_dist.shape[1]:
            raise ValueError(
                "`mst_dist` must be a square 2D matrix of shape (n_clusters, n_clusters), "
                f"found shape {self._mst_dist.shape}."
            )

        if clusters.min() < 0:
            raise ValueError(
                f"Cluster IDs must be >= 0, found min={clusters.min()}."
            )
        if clusters.max() >= n_clust:
            raise ValueError(
                f"Max cluster ID {clusters.max()} exceeds `mst_dist` size {n_clust}."
            )

        self._clusters = clusters.astype(np.int32)

        # 额外检查：细胞数和 connectivities 形状一致
        if self._clusters.shape[0] != self.connectivities.shape[0]:
            raise ValueError(
                f"`clusters` length {self._clusters.shape[0]} does not match "
                f"number of cells {self.connectivities.shape[0]}."
            )

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    @property
    def clusters(self) -> np.ndarray:
        """Integer cluster IDs per cell."""
        return self._clusters

    def compute_transition_matrix(
        self,
        lambda_skel: float = 2.0,
        b: float = 10.0,
        nu: float = 0.5,
        check_irreducibility: bool = False,
        n_jobs: Optional[int] = None,
        backend: Backend_t = DEFAULT_BACKEND,
        show_progress_bar: bool = True,
        **kwargs: Any,
    ) -> "LineagePriorPseudotimeKernel":
        """
        Compute a skeleton-aware transition matrix.

        Uses `SkeletonSoftScheme`, which:
        - penalizes edges going against the pseudotime;
        - additionally gates transitions according to the given lineage skeleton.
        """
        if self.pseudotime is None:
            raise ValueError("Compute pseudotime first or provide `time_key` in `adata.obs`.")

        start = logg.info("Computing skeleton-aware pseudotime transition matrix")

        scheme = SkeletonSoftScheme(
            clusters=self.clusters,
            mst_adj=self._mst_dist,
            lambda_skel=lambda_skel,
            b=b,
            nu=nu,
        )

        biased_conn = self._bias_knn_with_indices(
            scheme,
            n_jobs=n_jobs,
            backend=backend,
            show_progress_bar=show_progress_bar,
            **kwargs,
        )

        # 确保仍然连通 / 可约
        if not _connected(biased_conn):
            logg.warning("Biased k-NN graph is disconnected.")
        if check_irreducibility and not _irreducible(biased_conn):
            logg.warning("Biased k-NN graph is not irreducible.")

        # 这里会自动做行归一化（如果需要）
        self.transition_matrix = biased_conn
        logg.info("    Finish", time=start)

        return self

    # ------------------------------------------------------------------
    # 内部：带 indices 的 bias_knn 实现
    # ------------------------------------------------------------------
    def _bias_knn_with_indices(
        self,
        scheme: SkeletonSoftScheme,
        n_jobs: Optional[int] = None,
        backend: Backend_t = DEFAULT_BACKEND,
        show_progress_bar: bool = True,
        **kwargs: Any,
    ) -> csr_matrix:
        """
        Variant of `bias_knn` that passes global cell index and neighbor indices
        to the scheme for skeleton gating.
        """
        conn = self.connectivities
        pseudotime = np.asarray(self.pseudotime)
        n_cells = conn.shape[0]

        if pseudotime.shape[0] != n_cells:
            raise ValueError(
                f"`pseudotime` length {pseudotime.shape[0]} does not match "
                f"number of cells {n_cells}."
            )

        def _helper(ixs: np.ndarray, conn, pseudotime, queue=None, **kwargs_inner):
            data: list[float] = []
            indices: list[int] = []

            for i in ixs:
                row = conn[i]               # csr_matrix row
                neigh_ix = row.indices      # neighbor indices

                biased_row = scheme(
                    cell_pseudotime=pseudotime[i],
                    neigh_pseudotime=pseudotime[neigh_ix],
                    neigh_conn=row.data,
                    cell_index=i,
                    neigh_indices=neigh_ix,
                    **kwargs_inner,
                )
                if np.shape(biased_row) != row.data.shape:
                    raise ValueError(
                        f"Expected row of shape `{row.data.shape}`, "
                        f"found `{np.shape(biased_row)}`."
                    )

                data.extend(biased_row)
                indices.extend(neigh_ix)

                if queue is not None:
                    queue.put(1)

            if queue is not None:
                queue.put(None)

            return np.asarray(data, dtype=float), np.asarray(indices, dtype=int)

        # 并行计算 data/indices；indptr 直接沿用原 conn.indptr
        res = parallelize(
            _helper,
            np.arange(n_cells),
            as_array=False,
            unit="cell",
            n_jobs=n_jobs,
            backend=backend,
            show_progress_bar=show_progress_bar,
        )(conn, pseudotime, **kwargs)

        data_chunks, indices_chunks = zip(*res)
        data_cat = np.concatenate(data_chunks)
        indices_cat = np.concatenate(indices_chunks)

        # 注意：这里用原始 conn.indptr，这在每行非零个数未改变时是安全的
        biased_conn = sp.csr_matrix(
            (data_cat, indices_cat, conn.indptr.copy()),
            shape=conn.shape,
        )
        biased_conn.eliminate_zeros()

        return biased_conn


class BCRKernel(ConnectivityMixin, BidirectionalKernel):
    """
    BCR-aware kernel with q_score-based directional constraint.

    - 对有 X_h 的细胞：在 X_h 空间构 kNN，按 1/dist 加权；
    - 再用 q_score 做硬/软约束，禁止或惩罚向更低 q_score 的转移；
    - 对没有 X_h 的细胞：保持单位自环，对其它 kernel“中性”。
    """

    def __init__(
        self,
        adata: AnnData,
        xh_key: str = "X_h",
        q_key: str = "q_score",
        **kwargs: Any,
    ):
        super().__init__(adata, **kwargs)
        self._xh_key = xh_key
        self._q_key = q_key

        if xh_key not in self.adata.obsm:
            raise KeyError(f"Expected `adata.obsm['{xh_key}']` to exist for BCRKernel.")
        if q_key not in self.adata.obs:
            raise KeyError(f"Expected `adata.obs['{q_key}']` (q_score) to exist for BCRKernel.")

        self._X_h = np.asarray(self.adata.obsm[xh_key])
        self._q = np.asarray(self.adata.obs[q_key]).astype(float)

    def __invert__(self) -> "BCRKernel":
        # 没有时间方向概念，直接复制即可
        bk = self._copy_ignore("_transition_matrix")
        bk._params = {}
        return bk

    def compute_transition_matrix(
        self,
        n_neighbors: int = 30,
        q_threshold=0.5,
        metric: str = "euclidean",
        constraint: str = "hard",  # "hard" or "soft"
        beta: float = 10.0,        # 软约束强度
        check_irreducibility: bool = False,
        eps=1e-8,
        **kwargs: Any,
    ) -> "BCRKernel":
        """
        基于 X_h + q_score 计算 BCR-aware 有向转移矩阵。

        - constraint = "hard":
            仅保留 q_j >= q_i 的边；
        - constraint = "soft":
            对 q_j < q_i 的边按 logistic 权重衰减；
        - 没有 X_h 的细胞保留单位自环。
        """
        start = logg.info("Computing BCR-aware, q_score-constrained transition matrix from `X_h`")

        X_h = self._X_h
        q = self._q
        n_cells = self.adata.n_obs

        # 这里你可以换成 obs['q_score']==1 之类的逻辑
        has_xh = (~np.isnan(X_h).any(axis=1)) & (q > q_threshold)
        idx_has = np.where(has_xh)[0]
        idx_no = np.where(~has_xh)[0]

        if idx_has.size == 0:
            raise ValueError("No cells with valid `X_h` found for BCRKernel.")

        # 1) 对有 X_h 的细胞构建 kNN 图（在 X_h 空间）
        X_sub = X_h[idx_has]
        nn = NearestNeighbors(
            n_neighbors=min(n_neighbors, idx_has.size - 1),
            metric=metric,
        )
        nn.fit(X_sub)
        dists, neigh = nn.kneighbors(X_sub, return_distance=True)

        rows = []
        cols = []
        vals = []

        for row_pos, cell_idx in enumerate(idx_has):
            neigh_idx = idx_has[neigh[row_pos]]  # 映射回全局索引
            q_i = q[cell_idx]
            q_j = q[neigh_idx]

            d = dists[row_pos]
            d[d == 0] = d[d > 0].min() if np.any(d > 0) else 1.0
            w = 1.0 / d  # 基于距离的相似度

            if constraint == "hard":
                # 公式： w_ij' = w_ij * 1(q_j >= q_i)
                # mask = q_j >= q_i

                mask = q_j >= q_i - eps
                w = w * mask
            elif constraint == "soft":
                # 公式（示例 logistic 软约束）:
                #   Δq_ij = q_j - q_i
                #   weight_factor_ij = 1 / (1 + exp(-β * Δq_ij))
                #   w_ij' = w_ij * weight_factor_ij
                dq = q_j - q_i
                weight_factor = 1.0 / (1.0 + np.exp(-beta * dq))
                w = w * weight_factor
            else:
                raise ValueError(f"Unknown constraint type `{constraint}`, expected 'hard' or 'soft'.")

            # 去除被完全抑制的邻居
            valid = w > 0
            if not np.any(valid):
                # 若该细胞所有邻居都被抑制，则保留对自己的自环
                rows.append(cell_idx)
                cols.append(cell_idx)
                vals.append(1.0)
            else:
                rows.extend([cell_idx] * np.sum(valid))
                cols.extend(neigh_idx[valid].tolist())
                vals.extend(w[valid].tolist())

        # 2) 对没有 X_h 的细胞，设置对自身的权重=1（自环）
        rows.extend(idx_no.tolist())
        cols.extend(idx_no.tolist())
        vals.extend([1.0] * idx_no.size)

        mat = sp.csr_matrix((vals, (rows, cols)), shape=(n_cells, n_cells))

        # 3) 行归一化为行随机矩阵 P
        row_sums = np.array(mat.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1.0
        inv_row = sp.diags(1.0 / row_sums)
        T_bcr = inv_row @ mat

        self.transition_matrix = T_bcr

        if check_irreducibility:
            # 如需可在此调用 _irreducible 检查
            pass

        logg.info("    Finish", time=start)
        return self