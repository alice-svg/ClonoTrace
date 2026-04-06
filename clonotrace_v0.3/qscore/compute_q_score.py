import numpy as np
import pandas as pd

def _clip01(x):
    return float(max(0.0, min(1.0, x)))

def _safe_float(x):
    if x is None:
        return np.nan
    try:
        x = float(x)
        if np.isnan(x):
            return np.nan
        return x
    except Exception:
        return np.nan

def _is_switched_isotype(c_gene: str):
    if c_gene is None or (isinstance(c_gene, float) and np.isnan(c_gene)):
        return np.nan
    s = str(c_gene).upper()
    if "IGHG" in s or "IGHA" in s or "IGHE" in s:
        return 1.0
    if "IGHM" in s or "IGHD" in s:
        return 0.0
    return np.nan

def _robust_minmax_scalar(x, lo, hi, eps=1e-8):
    if np.isnan(x):
        return np.nan
    return float(np.clip((x - lo) / (hi - lo + eps), 0, 1))

def fit_shm_scaler_from_shm(cell_df: pd.DataFrame, shm_col="SHM", q=(5,95)):
    """当没有 v_identity 时，用你已有的 SHM 列（原始比例/计数）做分位数缩放到[0,1]"""
    if shm_col not in cell_df.columns:
        return {"available": False}
    shm = pd.to_numeric(cell_df[shm_col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(shm) < 50:
        return {"available": False}
    lo, hi = np.nanpercentile(shm, q)
    if not np.isfinite(lo): lo = 0.0
    if not np.isfinite(hi) or hi <= lo: hi = lo + 1e-6
    return {"available": True, "col": shm_col, "lo": float(lo), "hi": float(hi)}

def _shm_score(cell_row, *, shm_col="SHM", v_identity_col=None, shm_scaler=None):
    """
    返回 S_shm in [0,1]，优先用 v_identity -> (1-v_identity) 分位数缩放；
    否则用 shm_col 的分位数缩放。
    """
    if v_identity_col is not None and shm_scaler and shm_scaler.get("available", False):
        vid = _safe_float(cell_row.get(v_identity_col, np.nan))
        if np.isnan(vid):
            return np.nan
        shm = 1.0 - vid
        return _robust_minmax_scalar(shm, shm_scaler["lo"], shm_scaler["hi"])

    # fallback: use raw SHM column + scaler
    x = _safe_float(cell_row.get(shm_col, np.nan))
    if shm_scaler and shm_scaler.get("available", False):
        return _robust_minmax_scalar(x, shm_scaler["lo"], shm_scaler["hi"])
    # 没有 scaler：不瞎用（返回缺失，让 coverage 处理）
    return np.nan

# def fit_clone_scaler(cell_df: pd.DataFrame, clone_size_col="clone_size", q=95):
#     if clone_size_col not in cell_df.columns:
#         return {"available": False}
#     cs = pd.to_numeric(cell_df[clone_size_col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
#     if len(cs) == 0:
#         return {"available": False}
#     hi = float(np.nanpercentile(cs, q))
#     if not np.isfinite(hi) or hi <= 1:
#         hi = 1.0
#     return {"available": True, "hi": hi}

def fit_clone_scaler(
        cell_df: pd.DataFrame,
        clone_id_col: str = "clone_id",
        q: int = 95
):
    """
    输入 clone_id 列，现场计算 clone_size，再按 0-q% 分位数做 [0,1] 缩放。
    对 "No_contig" 或缺失视为单细胞单克隆（size=1），不影响分位数上限。
    返回 dict:
        available : bool
        hi        : float   # 用于后续 Min-Max 归一化的上限
    """
    if clone_id_col not in cell_df.columns:
        return {"available": False}

    # 1. 把 No_contig / NaN 统一成缺失标记，方便 groupby
    clone_s = (
        cell_df[clone_id_col]
        .astype(str)
        .replace({"No_contig": np.nan, "nan": np.nan})  # 一句搞定
    )

    # 2. 现场计算 clone_size
    clone_size = (
        clone_s
        .to_frame("clone_id")
        .groupby("clone_id")["clone_id"]  # 显式指定列，防止未来 pandas 警告
        .transform("size")
    )
    clone_size = clone_size.fillna(1)  # 缺失 clone_id 的设为 1

    # 3. 清洗异常
    cs = pd.to_numeric(clone_size, errors="coerce") \
        .replace([np.inf, -np.inf], np.nan).dropna()
    if len(cs) == 0:
        return {"available": False}

    # 4. 计算分位上限
    hi = float(np.nanpercentile(cs, q))
    if not np.isfinite(hi) or hi <= 1:
        hi = 1.0

    return {"available": True, "hi": hi}

def _clonal_score(clone_size, clone_scaler):
    if not clone_scaler or not clone_scaler.get("available", False):
        return np.nan
    cs = _safe_float(clone_size)
    if np.isnan(cs) or cs < 1:
        return 0.0
    hi = clone_scaler["hi"]
    return _clip01(np.log1p(cs) / np.log1p(hi))

def compute_q_bio_bcr_cell_v2(
    cell_row,
    *,
    q_tech_bcr=None,
    igh_c_gene_col="isotype",
    shm_col="SHM",
    v_identity_col=None,
    clone_size_col="clone_size",
    shm_scaler=None,
    clone_scaler=None,
    weights=None,
    # 技术门控与缺失惩罚
    tech_gate_power=1.0,     # 1.0=线性；更严格可 1.5~2
    coverage_power=1.0,      # 缺失越多，分数越低
):
    """
    改进点：
    - 不再用 missing_fill=0.5 注入假信号
    - 对可用证据做加权平均，缺失项不计入，并用 coverage 乘子惩罚
    - SHM 强制缩放到 [0,1]（用 v_identity 或 SHM 列的分位数 scaler）
    - 最后用 q_tech^k 门控
    """
    if weights is None:
        # 建议：clonal 权重更低一点，避免采样/批次影响；SHM+isotype 更稳
        weights = {"iso": 0.45, "shm": 0.45, "clonal": 0.10}

    # 1) 三个证据项
    S_iso = _is_switched_isotype(cell_row.get(igh_c_gene_col, None))
    S_shm = _shm_score(cell_row, shm_col=shm_col, v_identity_col=v_identity_col, shm_scaler=shm_scaler)
    S_clonal = _clonal_score(cell_row.get(clone_size_col, np.nan), clone_scaler)

    # 2) 忽略缺失，做加权平均
    num, den = 0.0, 0.0
    for k, S in [("iso", S_iso), ("shm", S_shm), ("clonal", S_clonal)]:
        w = float(weights.get(k, 0.0))
        if w <= 0:
            continue
        if S is None or (isinstance(S, float) and np.isnan(S)):
            continue
        num += w * float(S)
        den += w

    if den == 0:
        q_core = 0.0
        coverage = 0.0
    else:
        q_core = num / den
        coverage = den / sum(weights.values())

    # 3) 技术门控
    if q_tech_bcr is None:
        q_tech_bcr = cell_row.get("q_tech_bcr", 1.0)
    qt = _safe_float(q_tech_bcr)
    if np.isnan(qt):
        qt = 1.0
    qt = _clip01(qt)

    qbio = (qt ** tech_gate_power) * (q_core) * (coverage ** coverage_power)
    return _clip01(qbio)




def _robust_minmax(x, lo_q=1, hi_q=99, eps=1e-8):
    x = np.asarray(x, dtype=float)
    lo, hi = np.nanpercentile(x, lo_q), np.nanpercentile(x, hi_q)
    y = (x - lo) / (hi - lo + eps)
    return np.clip(y, 0.0, 1.0)

def _iqr(x):
    q1, q3 = np.nanpercentile(x, 25), np.nanpercentile(x, 75)
    return float(q3 - q1)

def _mono_amplify_pow(x, gamma=2.0, eps=1e-12):
    """单调且放大差距：gamma>1 会拉开高值区差距；保持 [0,1]"""
    x = np.clip(np.asarray(x, float), 0.0, 1.0)
    if abs(gamma - 1.0) < eps:
        return x
    return x ** gamma

def _mono_amplify_logit(x, k=6.0, m=0.8, eps=1e-6):
    x = np.clip(np.asarray(x, float), eps, 1-eps)
    z = np.log(x/(1-x))
    z0 = np.log(m/(1-m))
    return 1/(1+np.exp(-k*(z-z0)))


def _smooth_alpha(eff_t, eff_b, k=200.0, eps=1e-8):
    """
    Sigmoid transition for entropy-weighted alpha/beta.
    Returns (alpha, beta) in [0.02, 0.98] with smooth transition
    instead of hard NOISE_THR snap.

    alpha = 1/(1 + exp(-k * (eff_t - eff_b) / (eff_t + eff_b + eps)))
    """
    ratio = (eff_t - eff_b) / (eff_t + eff_b + eps)
    exponent = np.clip(-k * ratio, -60, 60)
    alpha = 1.0 / (1.0 + np.exp(exponent))
    alpha = float(np.clip(alpha, 0.02, 0.98))
    beta = 1.0 - alpha
    return alpha, beta


def stretch_01(x, method="power", **kw):
    x = np.asarray(x, float)
    x = np.clip(x, 0.0, 1.0)

    if method == "power":
        # gamma > 1: 强化高值差距、压低低值（整体会更“低”一些）
        gamma = kw.get("gamma", 2.0)
        return x ** gamma

    if method == "root":
        # gamma > 1: 用幂的倒数，相当于“抬高整体”，同时拉开低值段差距
        gamma = kw.get("gamma", 2.0)
        return x ** (1.0 / gamma)

    if method == "logit":
        # 中间对比度增强：把(0,1)映射到R再sigmoid回[0,1]
        # k 越大越“硬”，m 是中点（建议用中位数）
        k = kw.get("k", 6.0)
        m = kw.get("m", float(np.nanmedian(x)))
        eps = kw.get("eps", 1e-6)
        z = np.log(np.clip(x, eps, 1-eps) / np.clip(1-x, eps, 1-eps))
        z0 = np.log(np.clip(m, eps, 1-eps) / np.clip(1-m, eps, 1-eps))
        return 1.0 / (1.0 + np.exp(-k * (z - z0)))

    if method == "sigmoid":
        # 经典S型：围绕 m 拉开差距；k越大越陡
        k = kw.get("k", 10.0)
        m = kw.get("m", float(np.nanmedian(x)))
        return 1.0 / (1.0 + np.exp(-k * (x - m)))

    if method == "quantile":
        # 分位数拉伸（rank-based）：强行“铺满”[0,1]，最能拉开但会改变绝对间距语义
        # 保序，但会把分布变成近似均匀
        r = kw.get("rankdata", None)
        # 不引scipy的话用简单rank（对ties处理一般）
        order = np.argsort(x)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.linspace(0, 1, len(x))
        return ranks

    raise ValueError(f"Unknown method: {method}")


def compute_q_score(
    adata,
    q_tech_key="q_tech_bcr",
    q_bio_key="q_bio_bcr",
    dataset_key=None,
    sat_thr=0.98,
    eps=1e-3,
    min_component_weight=None,
    out_q_score_key="q_score",
    out_debug_key="q_score_debug",
    stretch_method="root_g2",
    linear_with_raw_qbio=False,
    # --- 新增：差距放大参数 ---
    amplify_gamma=2.0,   # >1 放大，=1 不变
    amplify_strength=1.0, # 0~1：只放大一部分（更稳），1：全量放大
    kernel_weighting=200.0,  # sigmoid steepness for smooth alpha/beta transition
):
    # Smooth alpha/beta via sigmoid — no hard NOISE_THR snap


    obs = adata.obs
    qt_all = _robust_minmax(pd.to_numeric(obs[q_tech_key], errors="coerce").values,lo_q=1, hi_q=100,)
    # qb_all = _robust_minmax(pd.to_numeric(obs[q_bio_key], errors="coerce").values,lo_q=1, hi_q=100,)
    qb_raw = pd.to_numeric(obs[q_bio_key], errors="coerce").values
    qb_all = _robust_minmax(qb_raw, lo_q=1, hi_q=100)  # 线性归一化后的版本（和raw单调线性对应）

    methods = {
        "root_g2": dict(method="root", gamma=2.0),
        "root_g3": dict(method="root", gamma=3.0),
        "sigmoid_k8": dict(method="sigmoid", k=8.0, m=float(np.nanmedian(np.asarray(qb_all)))),
        "sigmoid_k12": dict(method="sigmoid", k=12.0, m=float(np.nanmedian(np.asarray(qb_all)))),
        "logit_k3": dict(method="logit", k=3.0, m=float(np.nanmedian(np.asarray(qb_all)))),
        "logit_k5": dict(method="logit", k=5.0, m=float(np.nanmedian(np.asarray(qb_all)))),
        "power_g2": dict(method="power", gamma=2.0),
        "quantile": dict(method="quantile"),
    }
    if stretch_method is not None and not linear_with_raw_qbio:
        params = methods[stretch_method]
        qb_all = stretch_01(np.asarray(qb_all), **params)

    if dataset_key is None:
        groups = pd.Series(["all"] * adata.n_obs, index=obs.index)
    else:
        groups = obs[dataset_key].astype(str)

    q_score = np.full(adata.n_obs, np.nan, dtype=float)
    debug = []

    for g, idx in groups.groupby(groups).groups.items():
        idx = obs.index.get_indexer(idx)
        qt = qt_all[idx]
        qb = qb_all[idx]

        iqr_t = _iqr(qt)
        iqr_b = _iqr(qb)
        sat_t = float(np.mean(qt > sat_thr))
        sat_b = float(np.mean(qb > sat_thr))

        eff_t = max(iqr_t * (1.0 - sat_t), 0.0)
        eff_b = max(iqr_b * (1.0 - sat_b), 0.0)
        eff_t = max(eff_t, eps)
        eff_b = max(eff_b, eps)

        alpha, beta = _smooth_alpha(eff_t, eff_b, k=kernel_weighting)

        if min_component_weight is not None:
            alpha = max(alpha, min_component_weight)
            beta  = max(beta,  min_component_weight)
            s = alpha + beta
            alpha, beta = alpha / s, beta / s

        # --- 关键修改：对“信息量更大”的分量做单调放大 ---
        qt_use, qb_use = qt, qb
        if not linear_with_raw_qbio:
            # 只有在不追求线性时才启用放大

            if eff_b >= eff_t:
                qb_amp = _mono_amplify_pow(qb, gamma=amplify_gamma)
                qb_use = (1 - amplify_strength) * qb + amplify_strength * qb_amp

                # qb_amp = _mono_amplify_logit(qb, k=6.0, m=0.85)
                # qb_use = (1 - amplify_strength) * qb + amplify_strength * qb_amp
            else:
                qt_amp = _mono_amplify_pow(qt, gamma=amplify_gamma)
                qt_use = (1 - amplify_strength) * qt + amplify_strength * qt_amp

                # qt_amp = _mono_amplify_logit(qt, k=6.0, m=0.85)
                # qt_use = (1 - amplify_strength) * qt + amplify_strength * qt_amp

        q_score[idx] = alpha * qt_use + beta * qb_use

        debug.append({
            "group": g,
            "alpha_from_qtech": alpha,
            "beta_from_qbio": beta,
            "iqr_qtech": iqr_t,
            "iqr_qbio": iqr_b,
            "sat_qtech(>thr)": sat_t,
            "sat_qbio(>thr)": sat_b,
            "effinfo_qtech": eff_t,
            "effinfo_qbio": eff_b,
            "amplify_gamma": amplify_gamma,
            "amplify_strength": amplify_strength,
            "amplified_component": "q_bio" if eff_b >= eff_t else "q_tech",
            "kernel_weighting_k": kernel_weighting,
        })

    adata.obs[out_q_score_key] = q_score
    adata.uns[out_debug_key] = pd.DataFrame(debug).set_index("group")
    return q_score

def compute_q_score_v2(
    adata,
    q_tech_key="q_tech_bcr",
    q_bio_key="q_bio_bcr",
    stretch_method="root_g2",
    sat_thr=0.99,
    eps=1e-3,
    out_q_score_key="q_score",
    out_debug_key="q_score_debug",
):

    obs = adata.obs
    qt_all = pd.to_numeric(obs[q_tech_key], errors="coerce").values
    qb_all = pd.to_numeric(obs[q_bio_key], errors="coerce").values

    methods = {
        "root_g2": dict(method="root", gamma=2.0),
        "root_g3": dict(method="root", gamma=3.0),
        "sigmoid_k8": dict(method="sigmoid", k=8.0, m=float(np.nanmedian(np.asarray(qb_all)))),
        "sigmoid_k12": dict(method="sigmoid", k=12.0, m=float(np.nanmedian(np.asarray(qb_all)))),
        "logit_k3": dict(method="logit", k=3.0, m=float(np.nanmedian(np.asarray(qb_all)))),
        "logit_k5": dict(method="logit", k=5.0, m=float(np.nanmedian(np.asarray(qb_all)))),
        "power_g2": dict(method="power", gamma=2.0),
        "quantile": dict(method="quantile"),
    }
    if stretch_method is not None :
        params = methods[stretch_method]
        qb_all = stretch_01(np.asarray(qb_all), **params)

    debug = []

    iqr_t = _iqr(qt_all)
    iqr_b = _iqr(qb_all)
    sat_t = float(np.mean(qt_all > sat_thr))
    sat_b = float(np.mean(qb_all > sat_thr))

    eff_t = max(iqr_t * (1.0 - sat_t), 0.0)
    eff_b = max(iqr_b * (1.0 - sat_b), 0.0)
    eff_t = max(eff_t, eps)
    eff_b = max(eff_b, eps)

    alpha = eff_t / (eff_t + eff_b)
    beta  = eff_b / (eff_t + eff_b)

    if alpha > 0.9 :
        q_score = qt_all
    elif beta > 0.9 :
        q_score = qb_all
    else:
        q_score = alpha * qt_all + beta * qb_all

    debug.append({
        "alpha_from_qtech": alpha,
        "beta_from_qbio": beta,
        "iqr_qtech": iqr_t,
        "iqr_qbio": iqr_b,
        "sat_qtech(>thr)": sat_t,
        "sat_qbio(>thr)": sat_b,
        "effinfo_qtech": eff_t,
        "effinfo_qbio": eff_b,
    })

    adata.obs[out_q_score_key] = q_score
    adata.uns[out_debug_key] = pd.DataFrame(debug)
    return q_score
