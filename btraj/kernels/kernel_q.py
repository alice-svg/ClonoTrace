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
from btraj.kernels.bias_q import SkeletonSoftScheme

# Isotype maturation ordering: IgM/IgD (naive) → IgG/IgA/IgE (class-switched)
ISO_RANK = {
    "IGHM": 0, "IGHD": 1,
    "IGHG3": 2, "IGHG1": 3, "IGHA1": 4,
    "IGHG2": 5, "IGHG4": 6, "IGHA2": 7, "IGHE": 8,
}


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

        # Enforce integer encoding that matches mst_dist indexing
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

        # Additional check: ensure cell count matches connectivities shape
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

        # Ensure graph remains connected/irreducible
        if not _connected(biased_conn):
            logg.warning("Biased k-NN graph is disconnected.")
        if check_irreducibility and not _irreducible(biased_conn):
            logg.warning("Biased k-NN graph is not irreducible.")

        # Row normalization is applied automatically if needed
        self.transition_matrix = biased_conn
        logg.info("    Finish", time=start)

        return self

    # ------------------------------------------------------------------
    # Internal: bias_knn implementation with indices
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

        # Parallel computation of data/indices; indptr is reused from original conn.indptr
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

        # Note: using original conn.indptr is safe when the number of non-zeros per row is unchanged
        biased_conn = sp.csr_matrix(
            (data_cat, indices_cat, conn.indptr.copy()),
            shape=conn.shape,
        )
        biased_conn.eliminate_zeros()

        return biased_conn


class BCRKernel(ConnectivityMixin, BidirectionalKernel):
    """
    BCR-aware kernel with q_score-based directional constraint.

    - For cells with X_h: construct kNN in X_h space, weighted by 1/dist;
    - Apply q_score hard/soft constraints to prohibit or penalize transitions
      toward lower q_score;
    - For cells without X_h: maintain self-loop, remain neutral to other kernels.
    """

    def __init__(
        self,
        adata: AnnData,
        xh_key: str = "X_h",
        q_key: str = "q_score",
        isotype_key: Optional[str] = None,
        gamma_iso: float = 0.0,
        **kwargs: Any,
    ):
        super().__init__(adata, **kwargs)
        self._xh_key = xh_key
        self._q_key = q_key
        self._isotype_key = isotype_key
        self._gamma_iso = gamma_iso

        if xh_key not in self.adata.obsm:
            raise KeyError(f"Expected `adata.obsm['{xh_key}']` to exist for BCRKernel.")
        if q_key not in self.adata.obs:
            raise KeyError(f"Expected `adata.obs['{q_key}']` (q_score) to exist for BCRKernel.")

        self._X_h = np.asarray(self.adata.obsm[xh_key])
        self._q = np.asarray(self.adata.obs[q_key]).astype(float)

        # Optional: isotype ordering constraint
        if isotype_key is not None:
            if isotype_key not in self.adata.obs:
                raise KeyError(
                    f"Expected `adata.obs['{isotype_key}']` to exist for isotype ordering constraint."
                )
            iso_series = self.adata.obs[isotype_key]
            # Map each cell's isotype to ISO_RANK value; unknown isotypes set to np.nan
            self._iso_ranks = np.array(
                [ISO_RANK.get(iso, np.nan) for iso in iso_series],
                dtype=float,
            )

    def __invert__(self) -> "BCRKernel":
        # No temporal direction concept, simply copy
        bk = self._copy_ignore("_transition_matrix")
        bk._params = {}
        return bk

    def compute_transition_matrix(
        self,
        n_neighbors: int = 30,
        q_threshold=0.5,
        metric: str = "euclidean",
        constraint: str = "soft",  # "hard" or "soft"
        beta: float = 10.0,        # soft constraint strength
        check_irreducibility: bool = True,
        eps=1e-8,
        **kwargs: Any,
    ) -> "BCRKernel":
        """
        Compute BCR-aware directed transition matrix based on X_h + q_score.

        - constraint = "hard":
            Only retain edges where q_j >= q_i;
        - constraint = "soft":
            Apply logistic weight decay to edges where q_j < q_i;
        - Cells without X_h retain self-loop.
        """
        start = logg.info("Computing BCR-aware, q_score-constrained transition matrix from `X_h`")

        X_h = self._X_h
        q = self._q
        n_cells = self.adata.n_obs

        # Logic can be changed to obs['q_score']==1 etc.
        has_xh = (~np.isnan(X_h).any(axis=1)) & (q > q_threshold)
        idx_has = np.where(has_xh)[0]
        idx_no = np.where(~has_xh)[0]

        if idx_has.size == 0:
            raise ValueError("No cells with valid `X_h` found for BCRKernel.")

        # 1) Build kNN graph for cells with X_h (in X_h space)
        X_sub = X_h[idx_has]
        k = min(n_neighbors, idx_has.size - 1)
        if k < 1:
            # Single valid cell — no neighbors possible, assign self-loop only
            rows = [idx_has[0]]
            cols = [idx_has[0]]
            vals = [1.0]
            # Also add self-loops for cells without X_h
            rows.extend(idx_no.tolist())
            cols.extend(idx_no.tolist())
            vals.extend([1.0] * idx_no.size)
            mat = sp.csr_matrix((vals, (rows, cols)), shape=(n_cells, n_cells))
            row_sums = np.array(mat.sum(axis=1)).flatten()
            row_sums[row_sums == 0] = 1.0
            inv_row = sp.diags(1.0 / row_sums)
            self.transition_matrix = inv_row @ mat
            logg.info("    Finish", time=start)
            return self
        nn = NearestNeighbors(
            n_neighbors=k,
            metric=metric,
        )
        nn.fit(X_sub)
        dists, neigh = nn.kneighbors(X_sub, return_distance=True)

        rows = []
        cols = []
        vals = []

        for row_pos, cell_idx in enumerate(idx_has):
            neigh_idx = idx_has[neigh[row_pos]]  # Map back to global indices
            q_i = q[cell_idx]
            q_j = q[neigh_idx]

            d = dists[row_pos]
            d[d == 0] = d[d > 0].min() if np.any(d > 0) else 1.0
            sigma = np.median(d)
            sigma = max(sigma, 1e-12)
            exponent = np.clip(-d**2 / (2 * sigma**2), -60, 60)
            w = np.exp(exponent)  # Gaussian kernel — adaptive bandwidth per cell

            if constraint == "hard":
                #  w_ij' = w_ij * 1(q_j >= q_i)
                # mask = q_j >= q_i

                mask = q_j >= q_i - eps
                w = w * mask
            elif constraint == "soft":
                # Formula (example logistic soft constraint):
                #   Δq_ij = q_j - q_i
                #   weight_factor_ij = 1 / (1 + exp(-β * Δq_ij))
                #   w_ij' = w_ij * weight_factor_ij
                dq = q_j - q_i
                weight_factor = 1.0 / (1.0 + np.exp(np.clip(-beta * dq, -60, 60)))
                w = w * weight_factor
            else:
                raise ValueError(f"Unknown constraint type `{constraint}`, expected 'hard' or 'soft'.")

            # Optional: isotype ordering constraint (applied after q-score constraint)
            if self._gamma_iso > 0 and self._isotype_key is not None:
                iso_i = self._iso_ranks[cell_idx]
                iso_j = self._iso_ranks[neigh_idx]
                # Only apply where both cells have known isotypes
                both_known = np.isfinite(iso_i) & np.isfinite(iso_j)
                if np.any(both_known):
                    delta_iso = np.zeros_like(w)
                    delta_iso[both_known] = iso_j[both_known] - iso_i
                    iso_factor = 1.0 / (1.0 + np.exp(np.clip(-self._gamma_iso * delta_iso, -60, 60)))
                    # Only modify weights where both isotypes are known
                    w[both_known] *= iso_factor[both_known]

            # Remove completely suppressed neighbors
            valid = w > 0
            if not np.any(valid):
                # If all neighbors are suppressed, retain self-loop
                rows.append(cell_idx)
                cols.append(cell_idx)
                vals.append(1.0)
            else:
                rows.extend([cell_idx] * np.sum(valid))
                cols.extend(neigh_idx[valid].tolist())
                vals.extend(w[valid].tolist())

        # 2) For cells without X_h, set self-loop weight=1
        rows.extend(idx_no.tolist())
        cols.extend(idx_no.tolist())
        vals.extend([1.0] * idx_no.size)

        mat = sp.csr_matrix((vals, (rows, cols)), shape=(n_cells, n_cells))

        # 3) Row normalize to row-stochastic matrix P
        row_sums = np.array(mat.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1.0
        inv_row = sp.diags(1.0 / row_sums)
        T_bcr = inv_row @ mat

        self.transition_matrix = T_bcr

        if check_irreducibility:
            from scipy.sparse.csgraph import connected_components
            n_components, labels = connected_components(T_bcr, directed=True, connection='strong')
            if n_components > 1:
                logg.warning(
                    f"BCRKernel transition matrix has {n_components} strongly connected components. "
                    "The chain is not irreducible — some cells may act as absorbing states."
                )
            # Detect absorbing cells (self-loop = 1.0, no outgoing transitions)
            diag_vals = T_bcr.diagonal()
            row_nnz = np.diff(T_bcr.indptr)
            absorbing_mask = (np.abs(diag_vals - 1.0) < 1e-8) & (row_nnz == 1)
            n_absorbing = int(absorbing_mask.sum())
            if n_absorbing > 0:
                logg.warning(
                    f"BCRKernel: {n_absorbing} absorbing cells detected "
                    "(self-loop only, no outgoing transitions)."
                )

        logg.info("    Finish", time=start)
        return self