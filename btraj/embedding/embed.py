import os
import time
from typing import List, Sequence, Tuple

import numpy as np
import scanpy as sc
import torch
from Bio.SeqUtils import ProtParam
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm import tqdm

# Patch transformers for compatibility with AntiBERTy on newer transformers
import transformers.modeling_utils as _mu
_orig_ptm_init = _mu.PreTrainedModel.__init__

def _patched_ptm_init(self, *args, **kwargs):
    _orig_ptm_init(self, *args, **kwargs)
    if not hasattr(self, 'all_tied_weights_keys'):
        self.all_tied_weights_keys = {}
_mu.PreTrainedModel.__init__ = _patched_ptm_init

from antiberty import AntiBERTyRunner


class AntiBERTyEmbedder:
    def __init__(
        self,
        max_length: int = 256,
        batch_size: int = 256,
        try_gpu=True
    ):

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() and try_gpu else "cpu")
        self.max_length = max_length
        self.batch_size = batch_size

        self.antiberty= AntiBERTyRunner()
        self.antiberty.model.eval()
        self.antiberty.model.to(self.device)
        print("Loaded AntiBERTy model.")
        self.hidden_dim = self.antiberty.model.config.hidden_size  # 512

    # ========== 1. Sequence-level progress bar ==========
    def _embed_seqs(self, seqs):
        seqs = ["" if (s is None or str(s).lower() in ("nan", "none")) else str(s) for s in seqs]
        all_embs = []
        n_total = len(seqs)
        n_batch = (n_total - 1) // self.batch_size + 1
        start_t = time.time()

        with torch.no_grad():
            # wrap with tqdm to show batch-level progress
            for i in tqdm(range(0, n_total, self.batch_size),
                          total=n_batch,
                          desc=f"AntiBERTy embed  {len(seqs)} seqs"):
                batch_seqs = seqs[i: i + self.batch_size]
                embeddings = self.antiberty.embed(batch_seqs, return_attention=False)
                pooled = [e[1:-1].mean(dim=0, keepdim=True).cpu() for e in embeddings]
                all_embs += pooled

        cost = time.time() - start_t
        print(f"[AntiBERTy] Embedded {n_total} sequences | Time: {cost:.1f} s | Speed: {n_total/cost:.1f} seq/s")
        return torch.cat(all_embs, dim=0)

    def compute_bcr_embeddings(
        self,
        adata,
        heavy_col: str = "Heavy",
        light_col: str = "Light",
        pca_dim: int = 256,
        precomputed_path: str = None,
    ):
        """
        Takes adata.obs['heavy', 'light'] as input and outputs:
          - X_bcr_bert: np.array [n_cells, pca_dim] or [n_cells, 2*hidden_dim] (if pca_dim=None)
        Complexity:
          - O(N * L * d), fully leverages GPU via batching for significant speedup.
        If precomputed_path is provided, attempts to load/save .npy file.
        """
        n_cells = adata.n_obs

        # If precomputed, load directly
        if precomputed_path is not None and os.path.isfile(precomputed_path):
            print(f"Loading precomputed AntiBERTy BCR embeddings from {precomputed_path}")
            X_bcr = np.load(precomputed_path)
            return X_bcr

        print("Computing AntiBERTy embeddings for heavy/light sequences...")

        heavy_seqs = adata.obs[heavy_col].astype("str").to_list()
        light_seqs = adata.obs[light_col].astype("str").to_list()

        # embedding
        heavy_emb = self._embed_seqs(heavy_seqs)  # [N, d]
        light_emb = self._embed_seqs(light_seqs)  # [N, d]

        # Mark missing heavy/light with NaN (not zero) to avoid spurious kNN edges
        # Catch both actual NaN/None and string representations ("nan", "None", "")
        _heavy_str = adata.obs[heavy_col].astype(str).str.strip().str.lower()
        _light_str = adata.obs[light_col].astype(str).str.strip().str.lower()
        mask_heavy_nan = adata.obs[heavy_col].isna().to_numpy() | _heavy_str.isin(["nan", "none", ""]).to_numpy()
        mask_light_nan = adata.obs[light_col].isna().to_numpy() | _light_str.isin(["nan", "none", ""]).to_numpy()
        heavy_emb[mask_heavy_nan] = float('nan')
        light_emb[mask_light_nan] = float('nan')

        # Concatenate: heavy + light -> [N, 2d]
        X_concat = torch.cat([heavy_emb, light_emb], dim=1).numpy()

        # Identify rows with any NaN (missing BCR data)
        nan_rows = np.isnan(X_concat).any(axis=1)
        adata.obs['has_valid_bcr_embedding'] = ~nan_rows

        # PCA on valid rows only; NaN rows stay NaN
        if pca_dim is not None and pca_dim < X_concat.shape[1]:
            valid_mask = ~nan_rows
            n_valid = int(valid_mask.sum())
            effective_pca_dim = min(pca_dim, n_valid) if n_valid > 0 else pca_dim
            print(f"Applying PCA to reduce AntiBERTy BCR embedding to {effective_pca_dim} dimensions...")
            pca = PCA(n_components=effective_pca_dim, random_state=42)
            X_bcr = np.full((X_concat.shape[0], effective_pca_dim), np.nan, dtype=np.float64)
            if n_valid > 0:
                X_bcr[valid_mask] = pca.fit_transform(X_concat[valid_mask])
        else:
            X_bcr = X_concat  # [N, 2*hidden_dim] — NaN rows preserved

        if precomputed_path is not None:
            np.save(precomputed_path, X_bcr)
            print(f"Saved AntiBERTy BCR embeddings to {precomputed_path}")

        return X_bcr


# ---------------- 1. V/J usage (label encoding + standardization) ----------------
def build_vj_numeric_features(
        adata,
        bcr_v_cols: Sequence[str] = ('ighv', 'iglv'),
        bcr_j_cols: Sequence[str] = ('ighj', 'iglj'),
) -> Tuple[np.ndarray, List[str]]:
    """
    Build numeric features for V/J genes:
    - Apply label encoding (string -> integer) for each column separately
    - Then standardize all columns together
    Returns:
    - vj_scaled: (n_cells, n_vj_cols) array
    - vj_names:  list of feature names
    """
    obs = adata.obs

    vj_cols = list(bcr_v_cols) + list(bcr_j_cols)
    vj_feats = []
    vj_names = []

    for col in vj_cols:
        if col not in obs.columns:
            raise KeyError(f"{col} not found in adata.obs")

        raw_vals = obs[col].astype('string').fillna('NA').to_numpy()

        # Preprocessing: handle multi-hits, keep up to allele level
        processed_vals = []
        for v in raw_vals:
            s = str(v)
            if s == '' or s.lower() == 'nan':
                s = 'NA'
            if '*' in s:
                s = s.split('*')[0]
            processed_vals.append(s)

        processed_vals = np.array(processed_vals, dtype=str)

        le = LabelEncoder()
        int_encoded = le.fit_transform(processed_vals).astype(float)
        vj_feats.append(int_encoded)
        vj_names.append(f"{col}_label")

    if vj_feats:
        # now vj_feats is a list of length n_vj_cols, each element shape=(n_cells,)
        vj_mat = np.vstack(vj_feats).T  # (n_cells, n_vj)
        vj_scaled = StandardScaler().fit_transform(vj_mat)
    else:
        vj_scaled = np.zeros((obs.shape[0], 0), dtype=float)

    return vj_scaled, vj_names



def build_weighted_joint_embedding(
    adata,
    rna_key="X_scVI",
    bcr_key="X_bcr_antiberty",
    bcr_reduced_dim=20,
    w_rna=1.0,
    w_bcr=1.0,
    joint_key="X_joint_rna_bcr",
):
    X_rna = np.asarray(adata.obsm[rna_key])
    X_bcr = np.asarray(adata.obsm[bcr_key])
    # Normalize each separately
    X_rna_scaled = StandardScaler().fit_transform(X_rna)
    X_bcr_scaled = StandardScaler().fit_transform(X_bcr)
    # Reduce BCR dimensionality to avoid overly high dimensions
    if bcr_reduced_dim is not None and X_bcr.shape[1] > bcr_reduced_dim:
        # from sklearn.decomposition import PCA
        pca_bcr = PCA(n_components=bcr_reduced_dim, random_state=0)
        X_bcr_reduced = pca_bcr.fit_transform(X_bcr_scaled)
    else:
        X_bcr_reduced = X_bcr_scaled
    # weight: multiply by sqrt(w) to ensure Euclidean distance is equivalent to weighted subspace direct sum
    X_rna_w = np.sqrt(w_rna) * X_rna_scaled
    X_bcr_w = np.sqrt(w_bcr) * X_bcr_reduced
    X_joint = np.hstack([X_rna_w, X_bcr_w])
    adata.obsm[joint_key] = X_joint
    return adata



def build_bcr_numeric_features_rich(
    adata,
    cdr3_cols: Sequence[str] = ('cdrh3', 'cdrl3'),
    bcr_v_cols: Sequence[str] = ('ighv', 'iglv'),
    bcr_j_cols: Sequence[str] = ('ighj', 'iglj'),
    use_aa_2mer: bool = True,
    max_aa_2mer_dim: int = 100,
) -> Tuple[np.ndarray, List[str]]:
    """
    Build a richer BCR numeric feature matrix:
      - V/J usage (label encoding + standardization)
      - AA sequence features for each CDR3:
        * length
        * physicochemical properties: gravy, charge, aromaticity, instability_index, pI
        * 20 AA frequencies
        * optional: common 2-mer frequencies (compressed to limited dimensions)

    Parameters
    ----------
    adata : AnnData
    cdr3_cols : CDR3 AA sequence column names
    bcr_v_cols, bcr_j_cols : V/J gene column names
    use_aa_2mer : whether to include AA-level bigram features
    max_aa_2mer_dim : limit 2-mer dimensions to reduce sparsity and high dimensionality

    Returns
    -------
    X_bcr_numeric : (n_cells, d_bcr) numeric matrix
    feature_names : list of feature names
    """

    obs = adata.obs
    # ---------------- 1. V/J usage (label encoding + standardization) ----------------

    vj_cols = list(bcr_v_cols) + list(bcr_j_cols)
    vj_feats = []
    vj_names = []

    for col in vj_cols:
        if col not in obs.columns:
            raise KeyError(f"{col} not found in adata.obs")

        vals = obs[col].astype('string').fillna('NA').to_numpy()
        le = LabelEncoder()
        int_encoded = le.fit_transform(vals).astype(float)
        vj_feats.append(int_encoded)
        vj_names.append(f"{col}_label")

    if vj_feats:
        vj_mat = np.vstack(vj_feats).T  # (n_cells, n_vj)
        vj_scaled = StandardScaler().fit_transform(vj_mat)
    else:
        vj_scaled = np.zeros((obs.shape[0], 0), dtype=float)

    # ---------------- 2. CDR3 AA sequence features ----------------
    # 20 standard AAs

    AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")

    def safe_seq_aa(s):
        if s is None:
            return ""
        s = str(s)
        if s.lower() in ("nan", "none"):
            return ""
        s = "".join([c for c in s.upper() if c in AA_LIST])
        return s

    def compute_cdr3_features(seq: str):
        """
        Return:
          length, gravy, charge, aromaticity, instability, pI,
          aa_freqs[20], (optional) 2mer_freqs[k]
        """
        seq = safe_seq_aa(seq)
        if len(seq) == 0:
            # physicochemical properties & frequencies all set to 0
            length = 0.0
            gravy = 0.0
            charge = 0.0
            aromaticity = 0.0
            instability = 0.0
            pi = 0.0
            aa_freq = np.zeros(len(AA_LIST), dtype=float)
            return length, gravy, charge, aromaticity, instability, pi, aa_freq, []
        try:
            pa = ProtParam.ProteinAnalysis(seq)
        except Exception:
            # unconventional AA, fall back to length + frequency
            length = float(len(seq))
            aa_freq = np.zeros(len(AA_LIST), dtype=float)
            if length > 0:
                for c in seq:
                    if c in AA_LIST:
                        aa_freq[AA_LIST.index(c)] += 1.0
                aa_freq /= length
            return length, 0.0, 0.0, 0.0, 0.0, 0.0, aa_freq, []

        length = float(len(seq))
        gravy = float(pa.gravy())
        charge = float(pa.charge_at_pH(7.4))
        aromaticity = float(pa.aromaticity())
        instability = float(pa.instability_index())
        pi = float(pa.isoelectric_point())

        aa_freq = np.zeros(len(AA_LIST), dtype=float)
        for c in seq:
            if c in AA_LIST:
                aa_freq[AA_LIST.index(c)] += 1.0
        aa_freq /= max(length, 1.0)
        return length, gravy, charge, aromaticity, instability, pi, aa_freq, seq

    # 2-mer computation deferred to outer scope for vocabulary control
    cdr3_seqs_by_col = {}
    for col in cdr3_cols:
        if col not in obs.columns:
            raise KeyError(f"{col} not found in adata.obs")
        cdr3_seqs_by_col[col] = [safe_seq_aa(x) for x in obs[col].astype('string').fillna('').to_list()]

    # -------- 2.1 Build global 2-mer vocabulary (optional) --------
    aa_2mer_vocab = []
    if use_aa_2mer:
        from collections import Counter
        counter_2mer = Counter()
        for col in cdr3_cols:
            seqs = cdr3_seqs_by_col[col]
            for s in seqs:
                for i in range(len(s) - 1):
                    kmer = s[i:i+2]
                    counter_2mer[kmer] += 1
        # sort by frequency, take top max_aa_2mer_dim entries
        common_2mers = [k for k, v in counter_2mer.most_common(max_aa_2mer_dim)]
        aa_2mer_vocab = common_2mers  # list of strings
    vocab_2mer_index = {k: i for i, k in enumerate(aa_2mer_vocab)}

    # -------- 2.2 Compute features for each CDR3 --------
    cdr3_feature_blocks = []  # stack feature blocks for each col
    cdr3_feature_names = []

    for col in cdr3_cols:
        seqs = cdr3_seqs_by_col[col]
        n = len(seqs)

        length_list = []
        gravy_list = []
        charge_list = []
        aromaticity_list = []
        instability_list = []
        pi_list = []
        aa_freq_mat = []          # (n, 20)
        if use_aa_2mer and aa_2mer_vocab:
            twomer_mat = np.zeros((n, len(aa_2mer_vocab)), dtype=float)
        else:
            twomer_mat = None

        for idx, s in enumerate(seqs):
            length, gravy, charge, aro, instab, pi, aa_freq, seq_used = compute_cdr3_features(s)
            length_list.append(length)
            gravy_list.append(gravy)
            charge_list.append(charge)
            aromaticity_list.append(aro)
            instability_list.append(instab)
            pi_list.append(pi)
            aa_freq_mat.append(aa_freq)

            # count 2-mer frequencies
            if twomer_mat is not None and isinstance(seq_used, str):
                l = len(seq_used)
                if l > 1:
                    for i in range(l - 1):
                        kmer = seq_used[i:i+2]
                        idx2 = vocab_2mer_index.get(kmer, None)
                        if idx2 is not None:
                            twomer_mat[idx, idx2] += 1.0
                    twomer_mat[idx] /= (l - 1.0)

        base_feats = np.vstack(
            [
                np.array(length_list, dtype=float),
                np.array(gravy_list, dtype=float),
                np.array(charge_list, dtype=float),
                np.array(aromaticity_list, dtype=float),
                np.array(instability_list, dtype=float),
                np.array(pi_list, dtype=float),
            ]
        ).T  # (n, 6)

        aa_freq_mat = np.vstack(aa_freq_mat)  # (n, 20)

        # standardize base + aa_freq separately
        base_scaled = StandardScaler().fit_transform(base_feats)
        aa_freq_scaled = StandardScaler().fit_transform(aa_freq_mat)

        blocks = [base_scaled, aa_freq_scaled]
        names = [
            f"{col}_len",
            f"{col}_gravy",
            f"{col}_charge",
            f"{col}_aromaticity",
            f"{col}_instability",
            f"{col}_pI",
        ] + [f"{col}_aa_freq_{aa}" for aa in AA_LIST]

        if twomer_mat is not None:
            twomer_scaled = StandardScaler().fit_transform(twomer_mat)
            blocks.append(twomer_scaled)
            names += [f"{col}_2mer_{k}" for k in aa_2mer_vocab]

        cdr3_block = np.hstack(blocks)  # (n, d_cdr3_for_this_col)
        cdr3_feature_blocks.append(cdr3_block)
        cdr3_feature_names.extend(names)

    if cdr3_feature_blocks:
        cdr3_all = np.hstack(cdr3_feature_blocks)  # (n_cells, sum over cols)
    else:
        cdr3_all = np.zeros((obs.shape[0], 0), dtype=float)

    # ---------------- 3. Concatenate VJ + CDR3 ----------------
    X_bcr_numeric = np.hstack([vj_scaled, cdr3_all])
    feature_names = vj_names + cdr3_feature_names

    return X_bcr_numeric, feature_names
