"""Tests for batch correction of BCR embeddings."""
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData


def _make_batch_adata(n=100, n_dim=20):
    """Create test AnnData with batch labels and BCR embeddings."""
    rng = np.random.RandomState(42)
    X = rng.randn(n, 10).astype(np.float32)

    # Create BCR embeddings with batch effect
    batch = np.array(["batch1"] * 50 + ["batch2"] * 50)
    bcr = rng.randn(n, n_dim).astype(np.float64)
    bcr[50:] += 3.0  # artificial batch effect

    obs = pd.DataFrame({"batch": batch}, index=[f"cell_{i}" for i in range(n)])
    adata = AnnData(X=X, obs=obs)
    adata.obsm["X_bcr"] = bcr
    return adata


class TestBatchCorrection:
    def test_output_created(self):
        pytest.importorskip("harmonypy")
        from btraj.preprocess.batch_correct_bcr import batch_correct_bcr_embeddings
        adata = _make_batch_adata()
        batch_correct_bcr_embeddings(adata, bcr_key="X_bcr", batch_key="batch")
        assert "X_bcr_corrected" in adata.obsm

    def test_shape_with_pca(self):
        pytest.importorskip("harmonypy")
        from btraj.preprocess.batch_correct_bcr import batch_correct_bcr_embeddings
        adata = _make_batch_adata()
        batch_correct_bcr_embeddings(adata, bcr_key="X_bcr", batch_key="batch", n_pca=10)
        assert adata.obsm["X_bcr_corrected"].shape == (100, 10)

    def test_shape_without_pca(self):
        pytest.importorskip("harmonypy")
        from btraj.preprocess.batch_correct_bcr import batch_correct_bcr_embeddings
        adata = _make_batch_adata()
        batch_correct_bcr_embeddings(adata, bcr_key="X_bcr", batch_key="batch", n_pca=None)
        assert adata.obsm["X_bcr_corrected"].shape[0] == 100


class TestNaNPreservation:
    def test_nan_rows_preserved(self):
        pytest.importorskip("harmonypy")
        from btraj.preprocess.batch_correct_bcr import batch_correct_bcr_embeddings
        adata = _make_batch_adata()
        # Set some rows to NaN
        adata.obsm["X_bcr"][5] = np.nan
        adata.obsm["X_bcr"][95] = np.nan
        batch_correct_bcr_embeddings(adata, bcr_key="X_bcr", batch_key="batch")
        assert np.all(np.isnan(adata.obsm["X_bcr_corrected"][5]))
        assert np.all(np.isnan(adata.obsm["X_bcr_corrected"][95]))
        assert np.all(np.isfinite(adata.obsm["X_bcr_corrected"][0]))


class TestErrorHandling:
    def test_missing_bcr_key(self):
        from btraj.preprocess.batch_correct_bcr import batch_correct_bcr_embeddings
        adata = _make_batch_adata()
        with pytest.raises(KeyError):
            batch_correct_bcr_embeddings(adata, bcr_key="nonexistent")

    def test_missing_batch_key(self):
        from btraj.preprocess.batch_correct_bcr import batch_correct_bcr_embeddings
        adata = _make_batch_adata()
        with pytest.raises(KeyError):
            batch_correct_bcr_embeddings(adata, bcr_key="X_bcr", batch_key="nonexistent")

    def test_all_nan_raises(self):
        from btraj.preprocess.batch_correct_bcr import batch_correct_bcr_embeddings
        adata = _make_batch_adata()
        adata.obsm["X_bcr"][:] = np.nan
        with pytest.raises(ValueError):
            batch_correct_bcr_embeddings(adata, bcr_key="X_bcr", batch_key="batch")
