"""CDR3-only fallback embedding when AntiBERTy is unavailable."""
import numpy as np
from anndata import AnnData
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Sequence, Optional


def build_cdr3_fallback_embedding(
    adata,
    cdr3_cols=("cdrh3", "cdrl3"),
    v_cols=("ighv", "iglv"),
    j_cols=("ighj", "iglj"),
    pca_dim=50,
    out_key="X_bcr_cdr3",
):
    """
    Build BCR embedding from CDR3 sequences and V/J gene usage.

    A lightweight fallback when full AntiBERTy embeddings are not
    available. Uses physicochemical CDR3 features + V/J gene encoding
    from build_bcr_numeric_features_rich(), then applies PCA reduction.

    Parameters
    ----------
    adata : AnnData
        Must have CDR3 and V/J columns in .obs.
    cdr3_cols : tuple of str
        CDR3 amino acid sequence columns.
    v_cols : tuple of str
        V gene columns.
    j_cols : tuple of str
        J gene columns.
    pca_dim : int or None
        PCA output dimensions. None = no reduction.
    out_key : str
        Key in adata.obsm for output embeddings.

    Returns
    -------
    AnnData
        Modified in-place with embeddings in obsm[out_key].
    """
    from btraj.embedding.embed import build_bcr_numeric_features_rich

    X_numeric, feature_names = build_bcr_numeric_features_rich(
        adata,
        cdr3_cols=cdr3_cols,
        bcr_v_cols=v_cols,
        bcr_j_cols=j_cols,
        use_aa_2mer=True,
        max_aa_2mer_dim=100,
    )

    # Handle NaN rows (cells missing CDR3/VJ data)
    nan_rows = np.isnan(X_numeric).any(axis=1)
    valid_mask = ~nan_rows
    n_valid = int(valid_mask.sum())

    if n_valid == 0:
        # All NaN: store full NaN matrix
        out_dim = pca_dim if pca_dim is not None else X_numeric.shape[1]
        adata.obsm[out_key] = np.full((adata.n_obs, out_dim), np.nan)
        return adata

    # StandardScaler on valid rows
    scaler = StandardScaler()
    X_scaled = np.full_like(X_numeric, np.nan)
    X_scaled[valid_mask] = scaler.fit_transform(X_numeric[valid_mask])

    # Replace any remaining NaN/inf from scaling
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    # PCA reduction
    if pca_dim is not None and pca_dim < X_scaled.shape[1] and n_valid > 1:
        effective_dim = min(pca_dim, n_valid, X_scaled.shape[1])
        pca = PCA(n_components=effective_dim, random_state=42)
        X_out = np.full((adata.n_obs, effective_dim), np.nan)
        X_out[valid_mask] = pca.fit_transform(X_scaled[valid_mask])
    else:
        X_out = X_scaled
        # Restore NaN for invalid rows
        X_out[nan_rows] = np.nan

    adata.obsm[out_key] = X_out
    return adata
