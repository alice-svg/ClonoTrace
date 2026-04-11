"""Batch correction for BCR embeddings using Harmony."""
import numpy as np
import pandas as pd
from anndata import AnnData
from sklearn.decomposition import PCA


def batch_correct_bcr_embeddings(
    adata,
    bcr_key="X_bcr_antiberty",
    batch_key="batch",
    out_key="X_bcr_corrected",
    n_pca=50,
):
    """
    Apply Harmony batch correction to BCR embeddings.

    Handles NaN rows (cells without BCR data) by preserving them
    as NaN in the output and only correcting valid embeddings.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with BCR embeddings in obsm.
    bcr_key : str
        Key in adata.obsm for the BCR embeddings.
    batch_key : str
        Column in adata.obs identifying batches/samples.
    out_key : str
        Key in adata.obsm for the corrected embeddings.
    n_pca : int or None
        Number of PCA components before Harmony. None = skip PCA.

    Returns
    -------
    AnnData
        Modified in-place with corrected embeddings in obsm[out_key].
    """
    if bcr_key not in adata.obsm:
        raise KeyError(f"BCR embedding key '{bcr_key}' not found in adata.obsm")
    if batch_key not in adata.obs.columns:
        raise KeyError(f"Batch key '{batch_key}' not found in adata.obs")

    X = np.asarray(adata.obsm[bcr_key], dtype=float)
    n_cells, n_dims = X.shape

    # Identify valid (non-NaN) rows
    valid_mask = ~np.isnan(X).any(axis=1)
    n_valid = int(valid_mask.sum())

    if n_valid == 0:
        raise ValueError("No valid (non-NaN) BCR embeddings found.")

    X_valid = X[valid_mask]
    batch_valid = adata.obs[batch_key].values[valid_mask]

    # Optional PCA reduction
    if n_pca is not None and n_pca < X_valid.shape[1]:
        effective_pca = min(n_pca, n_valid, X_valid.shape[1])
        pca = PCA(n_components=effective_pca, random_state=42)
        X_input = pca.fit_transform(X_valid)
    else:
        X_input = X_valid.copy()

    # Run Harmony
    try:
        import harmonypy
    except ImportError:
        raise ImportError(
            "harmonypy is required for batch correction. "
            "Install with: pip install harmonypy"
        )

    meta = pd.DataFrame({"batch": batch_valid})
    ho = harmonypy.run_harmony(X_input, meta, "batch", max_iter_harmony=20)
    X_corrected_valid = ho.Z_corr.T  # Harmony returns (n_dims, n_cells)

    # Reconstruct full array with NaN rows
    out_dim = X_corrected_valid.shape[1]
    X_out = np.full((n_cells, out_dim), np.nan, dtype=float)
    X_out[valid_mask] = X_corrected_valid

    adata.obsm[out_key] = X_out
    return adata
