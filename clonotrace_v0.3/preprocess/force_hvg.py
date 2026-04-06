"""Force-include known B cell marker genes in the highly variable gene set."""
import numpy as np
from anndata import AnnData

# Curated B cell developmental markers across all stages
B_CELL_MARKERS = [
    # Early progenitor
    "RAG1", "RAG2", "DNTT", "VPREB1", "IGLL1", "IGLL5", "CD34", "SOX4",
    # Lineage TFs
    "EBF1", "PAX5", "TCL1A", "LEF1",
    # Mature / naive
    "MS4A1", "CR2", "IGHD", "IGHM", "FCER2", "CD22", "SELL",
    # Memory
    "CD27", "IGHG1", "IGHA1", "TNFRSF13B",
    # Plasma
    "SDC1", "XBP1", "PRDM1", "IRF4", "MZB1",
    # GC
    "AICDA", "BCL6", "FAS",
]


def force_include_markers(adata, markers=None, hvg_key="highly_variable"):
    """
    Force-include known B cell markers in the highly variable gene set.

    Modifies `adata.var[hvg_key]` in-place by setting True for any
    marker genes that are present in `adata.var_names` but not yet
    marked as highly variable.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix. Must have `adata.var[hvg_key]`.
    markers : list of str, optional
        Gene names to force-include. Defaults to B_CELL_MARKERS.
    hvg_key : str
        Key in `adata.var` for the HVG boolean mask.

    Returns
    -------
    int
        Number of genes newly added to the HVG set.
    """
    if markers is None:
        markers = B_CELL_MARKERS

    if hvg_key not in adata.var.columns:
        raise KeyError(f"`adata.var['{hvg_key}']` not found. Run HVG selection first.")

    current_hvg = adata.var[hvg_key].values.astype(bool)
    gene_names = adata.var_names

    n_added = 0
    for gene in markers:
        if gene in gene_names:
            idx = gene_names.get_loc(gene)
            if not current_hvg[idx]:
                current_hvg[idx] = True
                n_added += 1

    adata.var[hvg_key] = current_hvg
    return n_added
