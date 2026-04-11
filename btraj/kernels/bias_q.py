
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

        # Ensure cluster indices are integers from 0..n_clust-1 and aligned with mst_adj
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
    # Custom bias_knn: the sole purpose is to pass cell_index / neigh_indices to __call__
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

        Similar to parent class functionality, but passes cell_index and
        neigh_indices to __call__ to access cluster information and MST
        distances within the thresholding scheme.
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

                # Key: pass cell_index=i and neigh_indices=neigh_ix
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

            # Append the last pointer for this chunk
            # Note: maintain consistency with original CSR, row start positions
            # come from original conn.indptr
            if i_last == conn.shape[0] - 1:
                indptr.append(conn.indptr[-1])

            return (
                np.asarray(data, dtype=float),
                np.asarray(indices, dtype=int),
                np.asarray(indptr, dtype=int),
            )

        # Call CellRank's parallelization utility
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

        # Handle all-zero rows: add self-loops
        row_sum = np.asarray(biased.sum(axis=1)).ravel()
        zero_rows = np.where(row_sum == 0)[0]
        if zero_rows.size > 0:
            biased = biased.tolil()
            for i in zero_rows:
                biased[i, i] = 1.0
            biased = biased.tocsr()
            row_sum = np.asarray(biased.sum(axis=1)).ravel()

        # Row-wise normalization (true random walk matrix): each row sums to 1
        rows = np.repeat(np.arange(biased.shape[0]), np.diff(biased.indptr))
        biased.data /= row_sum[rows]
        biased.eliminate_zeros()

        return biased

    # ---------------------------------------------------------------------
    # Skeleton weight: penalize based on shortest path distance on MST
    # ---------------------------------------------------------------------
    def _skeleton_factor(self, ci: int, neigh_clusters: np.ndarray) -> np.ndarray:
        """
        Return weight array of same length as neigh_clusters, corresponding
        to each neighbor cell's cluster.

        Uses exponential decay based on shortest path distance:
            w = exp(-lambda_skel * d_eff)

        where d_eff is the effective distance after stable scaling + truncation
        of raw mst_adj distances.
        """
        neigh_clusters = np.asarray(neigh_clusters, dtype=int)

        # Raw shortest path distances (can be very large, e.g., 1e5 scale)
        d_raw = self._mst_adj[ci, neigh_clusters].astype(float)  # (n_neighbors,)

        # ------- Numerical stability scaling strategy -------
        # 1. Estimate a representative scale only on global non-zero distances
        #    to reduce noise sensitivity
        all_nonzero = self._mst_adj[self._mst_adj > 0]
        if all_nonzero.size > 0:
            # Use log median to mitigate outlier effects
            log_med = np.median(np.log(all_nonzero))
            scale = np.exp(log_med)
            if scale <= 0:
                scale = 1.0
        else:
            scale = 1.0

        # 2. Normalize to a smaller scale (e.g., 0 to several) to avoid
        #    underflow of exp(-lambda * d)
        d_norm = d_raw / scale

        # 3. Truncate very distant clusters: treat those beyond d_cap as
        #    equally "very far"
        #    For example, when d_norm > 5, exp(-lambda * d) is already very close to 0
        d_cap = 5.0
        d_eff = np.minimum(d_norm, d_cap)

        # ------- Exponential decay weights -------
        # Same cluster d_raw=0 -> d_eff=0 -> w=1
        w = np.exp(-self._lambda_skel * d_eff)

        return w

    # ---------------------------------------------------------------------
    # Main scheme: pseudotime soft penalty + skeleton penalty
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

        # --- 1) Pseudotime soft penalty (similar to CellRank's SoftThresholdScheme) ---
        past_ixs = np.where(neigh_pseudotime < cell_pseudotime)[0]
        if past_ixs.size > 0:
            dt = cell_pseudotime - neigh_pseudotime[past_ixs]  # > 0

            # Numerical stability: clip exponent to avoid exp overflow
            x = self._b * dt
            x = np.clip(x, -60.0, 60.0)  # prevent overflow / underflow
            weights[past_ixs] = 2.0 / ((1.0 + np.exp(x)) ** (1.0 / self._nu))

        # --- 2) skeleton gating based on cluster-level lineage ---
        if cell_index is not None and neigh_indices is not None:
            ci = int(self._clusters[cell_index])
            neigh_clusters = self._clusters[np.asarray(neigh_indices, dtype=int)]
            skel_w = self._skeleton_factor(ci, neigh_clusters)
            weights *= skel_w

        # Return final biased connectivities
        return neigh_conn * weights
