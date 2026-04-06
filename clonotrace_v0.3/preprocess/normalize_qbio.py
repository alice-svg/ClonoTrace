"""Per-patient rank normalization of q_bio components to remove batch effects."""
import numpy as np
import pandas as pd
from scipy.stats import rankdata


def normalize_qbio_per_patient(
    adata,
    patient_key,
    shm_col="SHM",
    clone_size_col="clone_size",
    out_shm_col="SHM_norm",
    out_clone_size_col="clone_size_norm",
):
    """
    Rank-normalize SHM and clone_size within each patient group.

    Removes inter-patient batch effects by converting raw values to
    within-patient rank percentiles scaled to [0, 1].

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    patient_key : str
        Column in adata.obs identifying patient/sample groups.
    shm_col : str
        Column with raw SHM values.
    clone_size_col : str
        Column with raw clone size values.
    out_shm_col : str
        Output column name for normalized SHM.
    out_clone_size_col : str
        Output column name for normalized clone size.

    Returns
    -------
    AnnData
        Modified in-place with normalized columns added to .obs.
    """
    if patient_key not in adata.obs.columns:
        raise KeyError(f"Patient key '{patient_key}' not found in adata.obs")

    obs = adata.obs

    for in_col, out_col in [(shm_col, out_shm_col), (clone_size_col, out_clone_size_col)]:
        if in_col not in obs.columns:
            raise KeyError(f"Column '{in_col}' not found in adata.obs")

        raw = pd.to_numeric(obs[in_col], errors="coerce").values.astype(float)
        normed = np.full(len(raw), np.nan, dtype=float)

        groups = obs[patient_key]
        for patient, idx in obs.groupby(patient_key).groups.items():
            pos = obs.index.get_indexer(idx)
            vals = raw[pos]

            # Preserve NaN positions
            valid = np.isfinite(vals)
            n_valid = int(valid.sum())

            if n_valid <= 1:
                # Single cell or all NaN: set to 0.5 (neutral)
                normed[pos[valid]] = 0.5
                continue

            # Rank within patient, scale to [0, 1]
            ranks = rankdata(vals[valid], method="average")
            normed[pos[valid]] = (ranks - 1) / (n_valid - 1)

        adata.obs[out_col] = normed

    return adata
