"""
Three-kernel combined transition matrix for BCR-informed trajectory inference.

Combines:
  1. LineagePriorPseudotimeKernel (or PseudotimeKernel fallback) — pseudotime + skeleton
  2. BCRKernel — BCR embedding space with q_score directional constraint
  3. ConnectivityKernel — graph regularizer for smoothness

Usage:
    combined_kernel = build_three_kernel(adata, time_key="palantir_pseudotime", ...)
    combined_kernel.compute_transition_matrix()
"""

from typing import Any, Optional
import numpy as np
from anndata import AnnData

from cellrank.kernels import PseudotimeKernel, ConnectivityKernel
from cellrank import logging as logg

from btraj.kernels.kernel_q import LineagePriorPseudotimeKernel, BCRKernel


def build_three_kernel(
    adata: AnnData,
    # --- weights ---
    w_pt: float = 0.5,
    w_bcr: float = 0.3,
    w_conn: float = 0.2,
    # --- pseudotime kernel params ---
    time_key: str = "palantir_pseudotime",
    clusters_key: Optional[str] = None,
    mst_dist: Optional[np.ndarray] = None,
    lambda_skel: float = 2.0,
    b: float = 10.0,
    nu: float = 0.5,
    # --- BCR kernel params ---
    xh_key: str = "X_h",
    q_key: str = "q_score",
    n_neighbors: int = 30,
    q_threshold: float = 0.5,
    constraint: str = "soft",
    beta: float = 10.0,
    check_irreducibility: bool = True,
    # --- shared ---
    backward: bool = False,
    **kwargs: Any,
):
    """
    Build a combined three-kernel transition matrix.

    Parameters
    ----------
    adata
        Annotated data matrix with pseudotime, BCR embeddings, and q_score.
    w_pt, w_bcr, w_conn
        Weights for pseudotime, BCR, and connectivity kernels. Must sum to 1.
    time_key
        Key in ``adata.obs`` for pseudotime values.
    clusters_key
        Key in ``adata.obs`` for integer cluster labels. If provided along with
        ``mst_dist``, uses ``LineagePriorPseudotimeKernel`` with skeleton constraint.
        Otherwise falls back to plain ``PseudotimeKernel``.
    mst_dist
        Precomputed shortest-path distance matrix on the lineage skeleton.
    lambda_skel, b, nu
        Parameters for ``SkeletonSoftScheme`` (only used with skeleton kernel).
    xh_key
        Key in ``adata.obsm`` for BCR embeddings.
    q_key
        Key in ``adata.obs`` for q_score values.
    n_neighbors
        Number of neighbors for BCR kNN graph.
    q_threshold
        Minimum q_score for a cell to participate in BCR kNN.
    constraint
        ``"soft"`` or ``"hard"`` constraint mode for BCR kernel.
    beta
        Logistic steepness for soft constraint.
    check_irreducibility
        Whether to check for absorbing states in BCR kernel.
    backward
        Direction of the kernel.

    Returns
    -------
    combined_kernel
        CellRank kernel combining all three sub-kernels.
    """
    start = logg.info("Building three-kernel combined transition matrix")

    # Validate weights
    if w_pt < 0 or w_bcr < 0 or w_conn < 0:
        raise ValueError(
            f"Kernel weights must be non-negative, got w_pt={w_pt}, w_bcr={w_bcr}, w_conn={w_conn}"
        )
    w_sum = w_pt + w_bcr + w_conn
    if w_sum < 1e-10:
        raise ValueError(
            f"Kernel weights sum to {w_sum:.2e}, which is effectively zero"
        )
    if abs(w_sum - 1.0) > 1e-6:
        logg.warning(f"Kernel weights sum to {w_sum:.4f}, normalizing to 1.0")
        w_pt, w_bcr, w_conn = w_pt / w_sum, w_bcr / w_sum, w_conn / w_sum

    # 1) Pseudotime kernel — use skeleton variant if mst_dist is available
    if clusters_key is not None and mst_dist is not None:
        logg.info("  Using LineagePriorPseudotimeKernel with skeleton constraint")
        pk = LineagePriorPseudotimeKernel(
            adata,
            time_key=time_key,
            clusters_key=clusters_key,
            mst_dist=mst_dist,
            backward=backward,
        )
        pk.compute_transition_matrix(
            lambda_skel=lambda_skel,
            b=b,
            nu=nu,
        )
    else:
        logg.info("  Using PseudotimeKernel (no skeleton constraint)")
        pk = PseudotimeKernel(
            adata,
            time_key=time_key,
            backward=backward,
        )
        pk.compute_transition_matrix()

    # 2) BCR kernel
    logg.info(f"  Building BCRKernel with {constraint} constraint")
    bk = BCRKernel(
        adata,
        xh_key=xh_key,
        q_key=q_key,
        backward=backward,
    )
    bk.compute_transition_matrix(
        n_neighbors=n_neighbors,
        q_threshold=q_threshold,
        constraint=constraint,
        beta=beta,
        check_irreducibility=check_irreducibility,
    )

    # 3) Connectivity kernel — graph regularizer
    logg.info("  Building ConnectivityKernel")
    ck = ConnectivityKernel(adata)
    ck.compute_transition_matrix()

    # 4) Combine
    combined_kernel = w_pt * pk + w_bcr * bk + w_conn * ck

    logg.info(
        f"  Combined kernel: w_pt={w_pt:.2f}, w_bcr={w_bcr:.2f}, w_conn={w_conn:.2f}",
        time=start,
    )

    return combined_kernel
