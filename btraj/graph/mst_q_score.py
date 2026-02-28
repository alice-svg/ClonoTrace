import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path, minimum_spanning_tree, breadth_first_order
from scipy.sparse.linalg import eigs
import networkx as nx

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.neighbors import kneighbors_graph, NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.isotonic import IsotonicRegression

from typing import Tuple, Optional, Dict, List, Any, Sequence
from collections import defaultdict, deque
from itertools import product
from tqdm import tqdm

import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev

# 可选社区检测
try:
    import igraph as ig
    import leidenalg as la
    HAS_LEIDEN = True
except Exception:
    HAS_LEIDEN = False



def inner_cluster_knn_sparse_new(labels, features, k=20, norm=False, metric='cosine'):
    """
    用 kNN 图近似 cluster 间连通性，避免构造 n_cells×n_cells 的完整距离矩阵。

    参数
    ----
    labels : array-like, shape (n_cells,)
        每个细胞的 cluster ID（int 或可转成 int）
    features : array-like, shape (n_cells, n_features)
        低维特征（如 X_scVI），用于 kNN。
    k : int
        每个细胞保留的最近邻个数（不包括自己）。
    norm : bool
        是否对 cluster-links 做行归一化（和你原来函数保持接口一致）。
    metric : str
        距离度量，默认为 'cosine'。

    返回
    ----
    links : ndarray, shape (n_clusters, n_clusters)
        簇间的连通强度矩阵。links[i, j] 越大表示 i→j 的跨簇 kNN 边越多/越近。
    """
    labels = np.asarray(labels)
    features = np.asarray(features)
    n_cells = features.shape[0]

    # 用 NearestNeighbors 构建 kNN；用稀疏存储，避免大矩阵
    nn = NearestNeighbors(
        n_neighbors=min(k + 1, n_cells),  # 包含自己，所以 k+1
        metric=metric,
        n_jobs=-1  # 多线程
    )
    nn.fit(features)
    distances, indices = nn.kneighbors(features, return_distance=True)

    # labels[i] -> cluster_i
    uniq = np.unique(labels)
    n_clusters = uniq.shape[0]
    # 构造簇 ID 到 0..(n_clusters-1) 的映射，防止簇 ID 不是连续整数
    cluster_map = {u: idx for idx, u in enumerate(uniq)}
    inv_cluster_map = {idx: u for u, idx in cluster_map.items()}
    cluster_ids = np.vectorize(cluster_map.get)(labels)

    # links_raw[i, j] 汇总 “从 cluster i 到 cluster j 的跨簇 kNN 关系”
    links_raw = np.zeros((n_clusters, n_clusters), dtype=float)

    # 遍历每个细胞的 kNN
    # distances[i, 0] 是该细胞本身（距离0），所以从 1 开始
    for i in range(n_cells):
        ci = cluster_ids[i]
        for nn_idx, dist in zip(indices[i, 1:], distances[i, 1:]):
            cj = cluster_ids[nn_idx]
            if ci == cj:
                continue  # 不考虑簇内连边
            # 距离越小，权重越大；可以用 1/(dist+eps)
            w = 1.0 / (dist + 1e-6)
            links_raw[ci, cj] += w

    # 可选：归一化行，使每个 cluster 出发的边总和为 1
    if norm:
        row_sums = links_raw.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        links = links_raw / row_sums
    else:
        links = links_raw

    # 如果后续代码假定簇 ID 是原始标签顺序而不是 0..n_clusters-1，
    # 可以在外层用 uniq / mapping 再做一次转换，这里先保持 0..n_clusters-1。
    return links, cluster_map, inv_cluster_map

def compute_cluster_q_stats(q_score: np.ndarray,
                            cell_labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算每个 cluster 的 q_score 中位数与有效样本数量。

    参数
    ----
    q_score : (n_cells,) 浮点数组，可能有 NaN
    cell_labels : (n_cells,) 整数簇 ID

    返回
    ----
    q_median : (n_clusters,) 各簇 q_score 中位数（若无有效值则为 np.nan）
    q_count : (n_clusters,) 各簇 q_score 非 NaN 样本数
    """
    cell_labels = np.asarray(cell_labels)
    q_score = np.asarray(q_score, dtype=float)
    n_clusters = int(cell_labels.max()) + 1
    q_median = np.full(n_clusters, np.nan)
    q_count = np.zeros(n_clusters, dtype=int)

    for k in range(n_clusters):
        mask = (cell_labels == k) & np.isfinite(q_score)
        if np.any(mask):
            q_median[k] = np.nanmedian(q_score[mask])
            q_count[k] = mask.sum()
    return q_median, q_count

def adjust_links_with_qscore(links: np.ndarray,
                             cell_labels_int: np.ndarray,
                             q_score: np.ndarray,
                             mode: str = "balanced",
                             penalty_factor = None,
                             boost_factor = None,
                             q_tol: float = 0.02,
                             min_q_count: int = 20,
                             ids2name = None) -> np.ndarray:
    """
    基于 cluster 级 q_score 中位数，对 links 做方向性惩罚/增强，使 MST 更偏向 q_score 单调方向。

    参数
    ----
    links : (n_clusters, n_clusters) cluster-level 连通强度（越大越近）
    cell_labels_int : (n_cells,) 整数簇 ID（0..n_clusters-1，与 links 对应）
    q_score : (n_cells,) 浮点数组，可有 NaN
    mode : {"fast", "balanced", "full"} 控制惩罚力度
    q_tol : float，认为中位数差距显著的阈值
    min_q_count : int，cluster 至少有多少个有效 q_score 才参与约束

    返回
    ----
    new_links : (n_clusters, n_clusters) 调整后的连通强度
    """
    links = np.asarray(links, dtype=float)
    n_clusters = links.shape[0]
    cell_labels_int = np.asarray(cell_labels_int)
    q_score = np.asarray(q_score, dtype=float)

    q_median, q_count = compute_cluster_q_stats(q_score, cell_labels_int)
    if ids2name is not None:
        print(pd.DataFrame({"cluster": [ids2name[i] for i in range(n_clusters)],
                            "q_median": q_median,
                            "q_count": q_count}))

    # 没有足够 q_score 信息时，直接返回原 links
    if np.all(~np.isfinite(q_median)):
        return links

    # 拷贝
    new_links = links.copy()
    # 惩罚系数设置
    if mode == "fast":
        penalty_factor = 0.3   # 逆向减弱到原来的 30%
        boost_factor = 1.0     # 顺向不额外增强
    elif mode == "full" :
        if penalty_factor is None:
            penalty_factor = 0.1   # 逆向连边强度砍到 10%
        if boost_factor is None:
            boost_factor = 1.2     # 顺向稍微增强
    else:  # balanced
        penalty_factor = 0.2
        boost_factor = 1.1

    for i in range(n_clusters):
        for j in range(n_clusters):
            if i == j:
                continue
            if links[i, j] <= 0:
                continue
            # 两端都需要有足够 q_score
            if q_count[i] < min_q_count or q_count[j] < min_q_count:
                continue
            qi, qj = q_median[i], q_median[j]
            if not (np.isfinite(qi) and np.isfinite(qj)):
                continue
            diff = qi - qj
            # 希望方向：低 q -> 高 q
            # 若 i->j 是逆向（qi >> qj），惩罚 i->j
            if diff > q_tol:
                new_links[i, j] *= penalty_factor
                new_links[j, i] *= boost_factor
            elif diff < -q_tol:
                new_links[j, i] *= penalty_factor
                new_links[i, j] *= boost_factor
            # abs(diff) <= q_tol 视为不可靠，不处理

    return new_links






def recurse_branches(path, v, tree, branch_clusters):
    num_children = len(tree[v])
    if num_children == 0:  # at leaf, add a None token
        return path + [v, None]
    elif num_children == 1:
        return recurse_branches(path + [v], tree[v][0], tree, branch_clusters)
    else:  # at branch
        branch_clusters.append(v)
        return [recurse_branches(path + [v], tree[v][i], tree, branch_clusters) for i in range(num_children)]

def flatten(li):
    if li[-1] is None:  # special None token indicates a leaf
        yield li[:-1]
    else:  # otherwise yield from children
        for l in li:
            yield from flatten(l)


class Lineage:
    class SingleLineage:
        def __init__(self, lineage, lineage_seq, lineage_index):
            self.lineage = lineage
            self.lineage_seq = lineage_seq
            self.lineage_index = lineage_index

        def __repr__(self):
            return 'lineage:' + str(self.lineage) + ', ' + \
                   'lineage_seq:' + str(self.lineage_seq) + ', ' + \
                   'lineage_index:' + str(self.lineage_index) + '\n'

    def __init__(self):

        self.all_lineage_seq = None
        self.lineages = []

        self.cluster_lineages = None
        self.cluster_lineseq_mat = None

    def add_lineage(self, lineage, lineage_seq, lineage_index):
        self.lineages.append(self.SingleLineage(lineage, lineage_seq, lineage_index))

    def __repr__(self):
        return 'all_lineage_seq: ' + \
               str(self.all_lineage_seq.tolist()) + \
               '\nlineages: ' + '\n' + str(self.lineages)

def get_mst_tree(hidden_clusters_links, num_clusters, start_node):
    tree = minimum_spanning_tree(np.max(hidden_clusters_links) + 1 - hidden_clusters_links)
    index_mapping = np.array([c for c in range(num_clusters)])
    connections = {k: list() for k in range(num_clusters)}
    cx = tree.tocoo()
    for i, j, v in zip(cx.row, cx.col, cx.data):
        i = index_mapping[i]
        j = index_mapping[j]
        connections[i].append(j)
        connections[j].append(i)
    visited = [False for _ in range(num_clusters)]
    queue = list()
    queue.append(start_node)
    children = {k: list() for k in range(num_clusters)}
    while len(queue) > 0: # BFS to construct children dict
        current_node = queue.pop()
        visited[current_node] = True
        for child in connections[current_node]:
            if not visited[child]:
                children[current_node].append(child)
                queue.append(child)
    return children,tree


def getLineage(links, cell_labels, start_node=0,terminal_clusters=None):
    num_clusters = len(set(cell_labels))
    children,tree = get_mst_tree(links, num_clusters, start_node=start_node)
    branch_clusters = deque()  # 产生分支的节点 cluster
    lineages_list = recurse_branches([], start_node, children, branch_clusters)
    lineages_list = list(flatten(lineages_list))

    if terminal_clusters is not None and len(terminal_clusters) > 0:
        terminal_clusters = set(terminal_clusters)

        def truncate_to_terminal(line):
            # 在一条 lineage 中，如果出现 terminal 节点，
            # 从第一个 terminal 节点开始截断后半段
            for idx, c in enumerate(line):
                if c in terminal_clusters:
                    return line[: idx + 1]
            return line

        new_lineages_list = []
        for line in lineages_list:
            truncated = truncate_to_terminal(line)
            # 去掉截断后长度<2的线
            if len(truncated) >= 2:
                new_lineages_list.append(truncated)

        lineages_list = new_lineages_list

    Infered_lineages = Lineage()
    lineage_seq = []
    adj_cluster2lineage_seq = dict()

    for each_line in lineages_list:
        for i in range(len(each_line) - 1):
            str_seq = str(each_line[i]) + str(each_line[i + 1])
            if str_seq in adj_cluster2lineage_seq.keys():
                continue
            adj_cluster2lineage_seq[str_seq] = len(lineage_seq)  # 存储lineage_seq的index
            lineage_seq.append([each_line[i], each_line[i + 1]])
    lineage_seq = np.array(lineage_seq)
    Infered_lineages.all_lineage_seq = lineage_seq

    cluster_lineages = {k: list() for k in range(num_clusters)}
    for l_idx, lineage in enumerate(lineages_list):
        for k in lineage:
            cluster_lineages[k].append(l_idx)
    Infered_lineages.cluster_lineages = cluster_lineages

    cluster_lineseq_mat = np.zeros([len(set(cell_labels)), len(lineage_seq)])
    for each_line in lineages_list:
        each_line_index = []
        each_line_seq = []
        for i in range(len(each_line) - 1):
            each_line_seq.append([each_line[i], each_line[i + 1]])
            str_seq = str(each_line[i]) + str(each_line[i + 1])
            each_line_index.append(adj_cluster2lineage_seq[str_seq])

        each_line_index = np.array(each_line_index)

        # 添加 lineage class 的顺序
        Infered_lineages.add_lineage(each_line, each_line_seq, each_line_index)  # 要加相邻的东西

        for i in each_line:
            cluster_lineseq_mat[i, each_line_index] = 1
    Infered_lineages.cluster_lineseq_mat = cluster_lineseq_mat

    return Infered_lineages, branch_clusters, tree  # ← 只加 tree


# ===== 3. 从有向图和 root 生成 raw lineages =====
def build_lineages_from_directed(adj_dir, root_clusters):
    visited = set()
    lineages = []

    def dfs(path, current):
        visited.add(current)
        children = [nbr for nbr in adj_dir[current] if nbr not in visited]
        if not children:
            lineages.append(path.copy())
            return
        for nxt in children:
            path.append(nxt)
            dfs(path, nxt)
            path.pop()

    for root in root_clusters:
        if root in adj_dir:
            dfs([root], root)

    return lineages

# ===== 4. q 单调性截断 =====
def enforce_lineage_q_monotone(lineage, q_median, q_tol_cluster=0.02):
    """
    版本2：尽量保留路径，只跳过明显逆向的节点，而不是整段截断。
    """
    if not lineage:
        return lineage

    new_lineage = [lineage[0]]
    last_q = q_median[lineage[0]]

    for c in lineage[1:]:
        qc = q_median[c]
        if not (np.isfinite(last_q) and np.isfinite(qc)):
            # 任一端没有 q 信息：保留
            new_lineage.append(c)
            last_q = qc
            continue

        # 若出现明显下降：qc < last_q - tol，则跳过这个点，但不断整条线
        if qc < last_q - q_tol_cluster:
            continue

        new_lineage.append(c)
        last_q = qc

    return new_lineage

def getLineage_v2(links, cell_labels, q_score, cell_labels_int, start_node=0,q_tol_cluster=0.02):
    """
    1. 用 links 算 MST；
    2. 用 q_median 给 MST 边定向（低 q → 高 q）；
    3. 从有向图和 start_node 构造 lineages；
    4. 做 q 单调性截断；

    links: cluster-level similarity/weight matrix (hidden_clusters_links)
    cell_labels: per-cell cluster labels (int or can be mapped to int)
    q_score: per-cell q (e.g. SHM-based) array
    cell_labels_int: same shape as q_score, int cluster IDs
    start_node: root cluster ID (e.g. naive B)
    """
    from collections import defaultdict
    num_clusters = len(set(cell_labels))

    # ===== 1. cluster 层的 q 统计 =====
    q_median, q_count = compute_cluster_q_stats(q_score, cell_labels_int)

    # ===== 2. MST + 用 q_median 定向：低 q -> 高 q =====
    tree = minimum_spanning_tree(np.max(links) + 1 - links)
    cx = tree.tocoo()
    mst_edges = []
    for i, j, _ in zip(cx.row, cx.col, cx.data):
        if i < j:
            mst_edges.append((int(i), int(j)))

    directed_edges = []
    for u, v in mst_edges:
        qu, qv = q_median[u], q_median[v]
        if (not np.isfinite(qu)) or (not np.isfinite(qv)):
            # 缺 q 信息 → 双向
            directed_edges.append((u, v))
            directed_edges.append((v, u))
        else:
            if qu <= qv:
                directed_edges.append((u, v))  # u -> v
            else:
                directed_edges.append((v, u))  # v -> u

    adj_dir = defaultdict(list)
    for a, b in directed_edges:
        adj_dir[a].append(b)

    root_clusters = [start_node]
    lineages_raw = build_lineages_from_directed(adj_dir, root_clusters)

    lineages = []
    for lin in lineages_raw:
        fixed = enforce_lineage_q_monotone(lin, q_median, q_tol_cluster=q_tol_cluster)
        if len(fixed) >= 2:
            lineages.append(fixed)

    # ===== 5. 构建 Lineage 对象（保持你原来的结构） =====
    Infered_lineages = Lineage()
    lineage_seq = []
    adj_cluster2lineage_seq = dict()

    # 注意这里用 lineages，而不是 lineages_list
    for each_line in lineages:
        for i in range(len(each_line) - 1):
            str_seq = str(each_line[i]) + str(each_line[i + 1])
            if str_seq in adj_cluster2lineage_seq:
                continue
            adj_cluster2lineage_seq[str_seq] = len(lineage_seq)
            lineage_seq.append([each_line[i], each_line[i + 1]])

    lineage_seq = np.array(lineage_seq)
    Infered_lineages.all_lineage_seq = lineage_seq

    cluster_lineages = {k: [] for k in range(num_clusters)}
    for l_idx, lineage in enumerate(lineages):
        for k in lineage:
            cluster_lineages[k].append(l_idx)
    Infered_lineages.cluster_lineages = cluster_lineages

    cluster_lineseq_mat = np.zeros([num_clusters, len(lineage_seq)])
    for each_line in lineages:
        each_line_index = []
        each_line_seq = []
        for i in range(len(each_line) - 1):
            each_line_seq.append([each_line[i], each_line[i + 1]])
            str_seq = str(each_line[i]) + str(each_line[i + 1])
            each_line_index.append(adj_cluster2lineage_seq[str_seq])

        each_line_index = np.array(each_line_index)

        Infered_lineages.add_lineage(each_line, each_line_seq, each_line_index)

        for i in each_line:
            cluster_lineseq_mat[i, each_line_index] = 1
    Infered_lineages.cluster_lineseq_mat = cluster_lineseq_mat

    # ===== 6. branch_clusters：所有出度>1的节点视为 branch =====
    branch_clusters = []
    for each_line in lineages:
        for c in each_line:
            # 在 mst 的有向图中，出度>1 的点可以视为 branch
            if len(adj_dir[c]) > 1 and c not in branch_clusters:
                branch_clusters.append(c)

    return Infered_lineages, branch_clusters


def convert_output(
    predict_order: np.ndarray,
    features: np.ndarray,
    cell_labels: np.ndarray,
    cells: np.ndarray,
    ids2name: Optional[Dict[int, Any]] = None,
    sep_points: int = 199
):
    """
    基于已推断的簇间轨迹（predict_order）和细胞特征，生成
    monocle 风格的 milestone_network、milestone_percentages、progressions。

    与原版相比的改进：
    - 不再显式构造所有插值点 line_points 并用 pairwise_distances 计算最近点；
    - 改为：按簇归类细胞，对每条边 (i, j) 仅投影属于簇 i 或 j 的细胞；
    - 使用解析几何直接计算细胞在该边上的相对位置 percentage（0~1），
      时间复杂度 O(N_cells × d)，适合 10 万+ 细胞。

    参数
    ----
    predict_order : ndarray, shape (n_edges, 2)
        每一行是 [start_cluster, end_cluster]，簇 ID 为 0..n_clusters-1。
    features : ndarray, shape (n_cells, n_features)
        细胞的低维特征。
    cell_labels : array-like, shape (n_cells,)
        整数簇 ID（必须与 predict_order 使用的簇编号一致）。
    cells : array-like, shape (n_cells,)
        细胞 ID（字符串或其他可转为 str 的类型）。
    ids2name : dict[int, Any], optional
        将整数簇 ID 映射回原始标签（如 'A', 'B' 等）。如果为 None，则直接使用整数。
    sep_points : int
        用于计算 milestone_network 中每条边的“长度”权重的插值粒度（默认 199），
        实际细胞投影不再显式使用这些插值点，只用于保持与原来长度定义兼容。

    返回
    ----
    [milestone_network, milestone_percentages, progressions]
        milestone_network : DataFrame(columns=['from', 'to', 'length', 'directed'])
        milestone_percentages : DataFrame(columns=['cell_id', 'milestone_id', 'percentage'])
        progressions : DataFrame(columns=['cell_id', 'from', 'to', 'percentage'])
    """
    features = np.asarray(features, dtype=float)
    cell_labels = np.asarray(cell_labels)
    cells = np.asarray(cells)

    # 1. 将 cell_labels 归一化为从 0 开始的整数簇 ID（保持与 predict_order 对齐）
    #   如果 cell_labels 已经是 0..n_clusters-1，则这一步不会改变顺序，
    #   但会保证后面 centers 的索引安全。
    uniq_labels = np.unique(cell_labels)
    label_map = {lab: idx for idx, lab in enumerate(uniq_labels)}
    inv_label_map = {idx: lab for lab, idx in label_map.items()}
    normalized_labels = np.vectorize(label_map.get)(cell_labels)
    n_clusters = len(uniq_labels)

    # 2. 计算每个簇的中心
    centers = np.zeros((n_clusters, features.shape[1]), dtype=float)
    for k in range(n_clusters):
        mask = (normalized_labels == k)
        if np.any(mask):
            centers[k] = features[mask].mean(axis=0)
        else:
            centers[k] = 0.0  # 理论上不会出现

    predict_order = np.asarray(predict_order, dtype=int)
    n_edges = predict_order.shape[0]

    # 3. 按簇归类细胞索引，方便按边处理
    cluster_to_cell_indices: Dict[int, List[int]] = {k: [] for k in range(n_clusters)}
    for idx, lab in enumerate(normalized_labels):
        cluster_to_cell_indices[int(lab)].append(idx)

    # 4. 为每个细胞确定其所属 edge 和在该 edge 上的相对位置 percentage
    #    如果一个细胞属于多个 edge 端点的簇，选择投影距离最近的那条 edge。
    n_cells = features.shape[0]
    best_dist = np.full(n_cells, np.inf, dtype=float)       # 到最近边的投影距离
    best_edge_idx = np.full(n_cells, -1, dtype=int)         # 使用哪条边
    best_percentage = np.zeros(n_cells, dtype=float)        # 在该边上的位置 [0,1]

    # 5. 对每条边进行投影计算
    for edge_idx, (ci, cj) in enumerate(predict_order):
        ci = int(ci)
        cj = int(cj)

        # 当前 edge 两端簇的中心
        pi = centers[ci]
        pj = centers[cj]
        edge_vec = pj - pi
        edge_len = np.linalg.norm(edge_vec)

        # 极端情况：两个簇中心重合，跳过或设为长度极小
        if edge_len == 0.0:
            continue

        direction = edge_vec / edge_len

        # 属于 ci 或 cj 的细胞索引
        idx_ci = cluster_to_cell_indices.get(ci, [])
        idx_cj = cluster_to_cell_indices.get(cj, [])
        idx_all = np.array(sorted(set(idx_ci) | set(idx_cj)), dtype=int)
        if idx_all.size == 0:
            continue

        X = features[idx_all]  # (n_subcells, d)
        vec = X - pi           # 从点 pi 出发的向量
        # 沿边方向的投影长度（可能 <0 或 >edge_len）
        proj_len = np.dot(vec, direction)  # (n_subcells,)

        # 限制在 [0, edge_len]
        proj_len_clipped = np.clip(proj_len, 0.0, edge_len)
        # 边上的最近点坐标
        proj_points = pi[None, :] + proj_len_clipped[:, None] * direction[None, :]
        # 最近点到细胞点的距离，作为评价哪条边更适合该细胞
        residual_vec = X - proj_points
        dist_to_edge = np.linalg.norm(residual_vec, axis=1)

        # 相对位置百分比
        frac = proj_len_clipped / edge_len  # (0~1)

        # 如果该边对某个细胞的距离更近，则更新该细胞的最佳边选择
        better_mask = dist_to_edge < best_dist[idx_all]
        if np.any(better_mask):
            update_indices = idx_all[better_mask]
            best_dist[update_indices] = dist_to_edge[better_mask]
            best_edge_idx[update_indices] = edge_idx
            best_percentage[update_indices] = frac[better_mask]

    # 6. 构建 progressions DataFrame
    #    每个细胞：cell_id, from, to, percentage
    #    如果某个细胞没有被任何 edge 捕获（理论上极少），则 percentage=0，from/to 用该细胞簇自身。
    edge_for_cell = best_edge_idx
    percentage = best_percentage.copy()

    # 对于未匹配 edge 的细胞，设置默认
    no_edge_mask = (edge_for_cell < 0)
    if np.any(no_edge_mask):
        # 将它们的 from/to 设置为自身簇，percentage=0
        # 这里我们给一个虚拟 edge（起点=终点=所在簇），只用于输出一致性
        edge_for_cell[no_edge_mask] = 0
        percentage[no_edge_mask] = 0.0

    # 对应每个细胞的 edge 起点和终点簇 ID（使用 predict_order）
    edge_starts = predict_order[:, 0]
    edge_ends = predict_order[:, 1]
    cell_from_clusters = edge_starts[edge_for_cell]
    cell_to_clusters = edge_ends[edge_for_cell]

    # 还原到原始簇标签（字符串）或整数
    if ids2name is not None:
        # 注意：predict_order 和 normalized_labels 都是 [0..n_clusters-1]，
        #      需要先映射回原始 label，再用 ids2name 转成最终名字。
        # 先将内部簇 ID -> 原始标签值
        cell_from_orig = np.vectorize(inv_label_map.get)(cell_from_clusters)
        cell_to_orig = np.vectorize(inv_label_map.get)(cell_to_clusters)
        # 再用 ids2name 映射到用户希望的名字
        cell_starts = [ids2name[int(lab)] for lab in cell_from_orig]
        cell_targets = [ids2name[int(lab)] for lab in cell_to_orig]

        net_from_orig = np.vectorize(inv_label_map.get)(edge_starts)
        net_to_orig = np.vectorize(inv_label_map.get)(edge_ends)
        network_starts = [ids2name[int(lab)] for lab in net_from_orig]
        network_targets = [ids2name[int(lab)] for lab in net_to_orig]
    else:
        # 直接使用内部整数簇 ID
        cell_starts = cell_from_clusters.tolist()
        cell_targets = cell_to_clusters.tolist()
        network_starts = edge_starts.tolist()
        network_targets = edge_ends.tolist()

    # progressions
    progressions = pd.DataFrame(
        {
            "cell_id": cells.astype(str),
            "from": np.asarray(cell_starts).astype(str),
            "to": np.asarray(cell_targets).astype(str),
            "percentage": percentage.astype(float).astype(str),
        }
    )

    # 7. milestone_percentages
    #    与原版本保持兼容：cell_id, milestone_id, percentage（目前全部为 0）
    milestone_percentages = pd.DataFrame(
        {
            "cell_id": cells.astype(str),
            "milestone_id": np.asarray(cell_starts).astype(str),
            "percentage": np.zeros_like(cells, dtype=float).astype(str),
        }
    )

    # 8. milestone_network
    #    需要 length 和 directed。length 仍然用“细胞数量归一化×2”方式，
    #    为了兼容原函数，我们根据每条边被分配的细胞数量来确定长度。
    # 统计每条边被细胞选择的次数
    edge_counts = np.zeros(n_edges, dtype=int)
    for e_idx in edge_for_cell:
        if 0 <= e_idx < n_edges:
            edge_counts[e_idx] += 1

    if edge_counts.max() > 0:
        network_length = edge_counts / edge_counts.max() * 2.0
    else:
        network_length = np.ones(n_edges, dtype=float)

    network_directed = np.ones(n_edges, dtype=bool)

    milestone_network = pd.DataFrame(
        {
            "from": np.asarray(network_starts).astype(str),
            "to": np.asarray(network_targets).astype(str),
            "length": network_length,
            "directed": network_directed,
        }
    )

    return [milestone_network, milestone_percentages, progressions]

def describe_lineages(Lineage_class, ids2name, print_out: bool = True):
    """
    将 Lineage_class 中的各条 lineage 转成人类可读的
    “celltypeA --> celltypeB --> ...” 文本描述。

    参数
    ----
    Lineage_class : get_trajectory 返回的 Lineage 对象
    ids2name : dict[int -> str]，簇整数 ID 到原始名称的映射
    print_out : 是否直接 print；若 False，仅返回列表

    返回
    ----
    descriptions : list[str]
        每个元素形如 "lineage1: A --> B --> C"
    """
    descriptions = []
    for idx, single in enumerate(Lineage_class.lineages, start=1):
        # single.lineage 是一个簇 ID 序列，例如 [0, 3, 5]
        labels = [ids2name[int(c)] for c in single.lineage]
        desc = f"lineage{idx}: " + " --> ".join(labels)
        descriptions.append(desc)
        if print_out:
            print(desc)
    return descriptions

def get_trajectory(
    cell_labels,
    y_features,
    cells,
    start_type=0,
    terminal_types=None,
    norm=False,
    k=20,
    q_score=None,
    mode: str = None,
    q_tol: float = 0.02,
    q_tol_cluster=0.02,
    penalty_factor = None,
    boost_factor = None,
):
    """
    拓展版：支持 q_score 约束 + 自动选择运行模式。

    mode:
        None -> 根据细胞数自动判定 {"fast", "balanced", "full"}
    """
    n_cells = y_features.shape[0]
    if mode is None:
        if n_cells >= 100000:
            mode = "fast"
        elif n_cells >= 30000:
            mode = "balanced"
        else:
            mode = "full"

    # 1. clusters_links: 仍用 cell-level kNN 近似 cluster 间连通性
    clusters_links, _, _ = inner_cluster_knn_sparse_new(cell_labels, y_features, k=k, norm=norm)
    # print(clusters_links)
    # 2. 将 cell_labels 规整为 0..C-1 的整数 ID
    uniq = np.unique(cell_labels)
    ids2name = {i: lab for i, lab in enumerate(uniq)}
    name2ids = {lab: i for i, lab in enumerate(uniq)}
    cell_labels_int = np.array([name2ids[i] for i in cell_labels])

    # 3. 如果有 q_score，则对 links 做方向约束调整
    if q_score is not None:
        q_score_arr = np.asarray(q_score, dtype=float)
        clusters_links = adjust_links_with_qscore(
            clusters_links,
            cell_labels_int=cell_labels_int,
            q_score=q_score_arr,
            mode=mode,
            q_tol=q_tol,
            ids2name=ids2name,
            penalty_factor=penalty_factor,
            boost_factor=boost_factor,
        )
        print()
        # print(clusters_links)

    # 4. MST 与 lineage，仍用你原来的逻辑
    start_node = int(np.where(uniq == start_type)[0][0])
    terminal_clusters = []
    if terminal_types is not None:

        for t in terminal_types:
            idx = np.where(uniq == t)[0]
            if len(idx) == 0:
                print(f'Warning: terminal type {t} not found in uniq')
                continue
            terminal_clusters.append(int(idx[0]))


    # if q_score is not None:
    #     Lineage_class, branch_clusters = getLineage_v2(clusters_links, cell_labels, q_score, cell_labels_int, start_node=start_node,
    #                                                    q_tol_cluster=q_tol_cluster)
    # else:
    #     Lineage_class, branch_clusters = getLineage(clusters_links, cell_labels, start_node=start_node)

    Lineage_class, branch_clusters, tree = getLineage(clusters_links, cell_labels, start_node=start_node,terminal_clusters=terminal_clusters)
    # 5. 转成 monocle 格式（这里 cell_labels 用整数 ID）
    results = convert_output(
        Lineage_class.all_lineage_seq,
        y_features,
        cell_labels_int,
        cells,
        ids2name
    )

    return clusters_links, tree, results, Lineage_class, branch_clusters, start_node, ids2name, name2ids, mode



def compute_cluster_centers(features: np.ndarray,
                            cell_labels: np.ndarray) -> np.ndarray:
    """
    根据细胞特征和簇标签计算每个簇的中心 (均值向量)。

    参数
    ----
    features : (n_cells, n_features)
    cell_labels : (n_cells,) 整数簇 ID，必须是 0..n_clusters-1 连续整数

    返回
    ----
    centers : (n_clusters, n_features)
    """
    cell_labels = np.asarray(cell_labels)
    features = np.asarray(features)
    n_clusters = int(cell_labels.max()) + 1

    centers = np.zeros((n_clusters, features.shape[1]), dtype=float)
    for k in range(n_clusters):
        mask = (cell_labels == k)
        if np.any(mask):
            centers[k] = features[mask].mean(axis=0)
        else:
            # 理论上不会出现；保险起见给一个很小扰动
            centers[k] = 0.0
    return centers


def build_cluster_graph_from_lineages(lineage_seq: np.ndarray,
                                      num_clusters: int) -> Dict[int, list]:
    """
    根据 Lineage_class.all_lineage_seq (簇边列表) 构建无向图 adjacency。

    参数
    ----
    lineage_seq : (n_edges, 2) 每行是 [start_cluster, end_cluster]
    num_clusters : 簇总数

    返回
    ----
    graph : dict[int, list[int]] 邻接表
    """
    graph = {k: [] for k in range(num_clusters)}
    for u, v in lineage_seq:
        u = int(u)
        v = int(v)
        if v not in graph[u]:
            graph[u].append(v)
        if u not in graph[v]:
            graph[v].append(u)
    return graph


def compute_tree_distances(start_node: int,
                           graph: Dict[int, list],
                           centers: np.ndarray) -> Tuple[np.ndarray, Dict[Tuple[int, int], float]]:
    """
    在簇图 (MST) 上从 start_node 出发，计算每个簇到 root 的累积距离 (树距)，
    以及每条无向边的几何长度。

    参数
    ----
    start_node : 根簇 ID (整数)
    graph : 簇图邻接表 (MST 结构)
    centers : (n_clusters, n_features) 簇中心

    返回
    ----
    tree_dist : (n_clusters,) 从 root 到每个簇的树距离
    edge_len : dict[(min(u,v), max(u,v))] -> edge_length
    """
    n_clusters = centers.shape[0]
    tree_dist = np.full(n_clusters, np.inf, dtype=float)
    tree_dist[start_node] = 0.0

    # 边长度（无向）
    edge_len: Dict[Tuple[int, int], float] = {}

    # 先预计算所有边几何长度
    for u in range(n_clusters):
        for v in graph[u]:
            if u < v:
                length = np.linalg.norm(centers[u] - centers[v])
                edge_len[(u, v)] = float(length)

    # BFS / DFS 均可，这里用 BFS
    dq = deque([start_node])
    visited = np.zeros(n_clusters, dtype=bool)
    visited[start_node] = True

    while dq:
        u = dq.popleft()
        for v in graph[u]:
            if not visited[v]:
                # 无向边，注意键顺序
                key = (u, v) if u < v else (v, u)
                length = edge_len[key]
                tree_dist[v] = tree_dist[u] + length
                visited[v] = True
                dq.append(v)

    return tree_dist, edge_len


def project_cells_onto_edge(features: np.ndarray,
                            centers: np.ndarray,
                            cell_labels: np.ndarray,
                            edge: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    将属于 edge 两端簇的细胞投影到该簇中心连线段上，得到：
    - 每个细胞的在该边上的位置 (0~1)；不属于该簇对的细胞返回 np.nan。
    - 边的欧式长度。

    参数
    ----
    features : (n_cells, n_features)
    centers : (n_clusters, n_features)
    cell_labels : (n_cells,) 整数簇 ID
    edge : (u, v) 整数簇 ID，对应 centers[u] -> centers[v]

    返回
    ----
    frac : (n_cells,) 细胞在该边上的位置 [0,1]，不在该边的为 np.nan
    edge_length : float 该边几何长度
    """
    u, v = edge
    cu = centers[u]
    cv = centers[v]
    diff = cv - cu
    edge_length = float(np.linalg.norm(diff))

    # 防止零长度边 (极少数情况下簇中心刚好相同)
    if edge_length == 0.0:
        # 全部给 0
        frac = np.full(features.shape[0], np.nan, dtype=float)
        mask = (cell_labels == u) | (cell_labels == v)
        frac[mask] = 0.0
        return frac, 0.0

    # 单位向量
    direction = diff / edge_length  # shape (d,)

    # 只对属于 u 或 v 的细胞进行投影
    mask = (cell_labels == u) | (cell_labels == v)
    frac = np.full(features.shape[0], np.nan, dtype=float)
    if not np.any(mask):
        return frac, edge_length

    X = features[mask]  # (n_subcells, d)
    # 向 cu 为原点的坐标系投影
    # proj_len = (X - cu) · direction
    proj_vec = X - cu  # (n_subcells, d)
    proj_len = np.dot(proj_vec, direction)  # (n_subcells,)

    # 限制在 [0, edge_length]
    proj_len_clip = np.clip(proj_len, 0.0, edge_length)
    frac_edge = proj_len_clip / edge_length  # 0~1

    frac[mask] = frac_edge
    return frac, edge_length


def compute_pseudotime(features: np.ndarray,
                       cell_labels: np.ndarray,
                       Lineage_class: Any,
                       start_node: int) -> np.ndarray:
    """
    在已经得到的 trajectory (Lineage_class, start_node) 上计算每个细胞的 pseudotime。

    整体思想：
    1) 用簇中心连线作为轨迹主曲线；
    2) 在 MST 上从根簇 start_node 计算每个簇的树距 tree_dist；
    3) 对每条边 (u, v)，将属于 u 或 v 的细胞投影到该边上，得到边内位置 frac (0~1)；
    4) 细胞 pseudotime = tree_dist[u] + frac * edge_length；若属于 v，则按与 u 同一边计算；
    5) 对所有细胞取 min(所在边上的 pseudotime)（一般只会所在簇边），最后全局归一化到 [0, 1]。

    参数
    ----
    features : (n_cells, n_features)
    cell_labels : (n_cells,) 整数簇 ID，必须与 get_trajectory 中 name2ids 映射后的一致
    Lineage_class : get_trajectory 返回的 Lineage 对象
    start_node : 根簇的整数 ID（get_trajectory 返回的 start_node）

    返回
    ----
    pseudotime : (n_cells,) 浮点数，已归一化到 [0, 1]
    """
    features = np.asarray(features)
    cell_labels = np.asarray(cell_labels)

    # 1) 簇中心
    centers = compute_cluster_centers(features, cell_labels)
    n_clusters = centers.shape[0]

    # 2) 从 Lineage_class.all_lineage_seq 构建 MST 图 (无向)
    lineage_seq = np.asarray(Lineage_class.all_lineage_seq, dtype=int)
    graph = build_cluster_graph_from_lineages(lineage_seq, n_clusters)

    # 3) 在 MST 上计算每簇到根的树距
    tree_dist, edge_len = compute_tree_distances(start_node, graph, centers)

    n_cells = features.shape[0]
    # 初始化为 +inf，用于求 min
    cell_pseudo = np.full(n_cells, np.inf, dtype=float)

    # 4) 遍历每条边，进行投影并更新细胞 pseudotime
    #    为了与 tree_dist 保持一致，边长度使用 edge_len 中的值
    for u, v in lineage_seq:
        u = int(u)
        v = int(v)
        key = (u, v) if u < v else (v, u)
        length = edge_len.get(key, None)
        if length is None:
            # 理论上不应该发生
            continue

        # 对属于 u 或 v 的细胞进行投影
        frac, _ = project_cells_onto_edge(features, centers, cell_labels, (u, v))

        # 对 u -> v 方向上的树距起点，用 tree_dist[u]
        # 注意：对于 MST，u 和 v 在树上的远近可能不同，但对一条边来说
        # 任取一端作为起点即可，只要 tree_dist 正确。
        base_u = tree_dist[u]
        base_v = tree_dist[v]

        # 对属于 u 簇的细胞：从 u 出发，位置是 frac
        mask_u = (cell_labels == u) & (~np.isnan(frac))
        if np.any(mask_u):
            pseudo_u = base_u + frac[mask_u] * length
            cell_pseudo[mask_u] = np.minimum(cell_pseudo[mask_u], pseudo_u)

        # 对属于 v 簇的细胞：从 v 出发，也可以写成 (1 - frac)*length，从树距更小的一端出发
        mask_v = (cell_labels == v) & (~np.isnan(frac))
        if np.any(mask_v):
            # 我们采用“从 v 看过去”的对称写法：
            # 在这条边上，u->v 方向的 frac 已知，如果从 v 作为起点，则位置是 (1 - frac)
            frac_v = 1.0 - frac[mask_v]
            pseudo_v = base_v + frac_v * length
            cell_pseudo[mask_v] = np.minimum(cell_pseudo[mask_v], pseudo_v)

    # 如果有未被赋值的 (np.inf)，说明有簇不在 lineage 上，退而用该簇的 tree_dist
    inf_mask = ~np.isfinite(cell_pseudo)
    if np.any(inf_mask):
        cell_pseudo[inf_mask] = tree_dist[cell_labels[inf_mask]]

    # 5) 全局归一化到 [0, 1]
    pt_min = float(np.nanmin(cell_pseudo))
    pt_max = float(np.nanmax(cell_pseudo))
    if pt_max > pt_min:
        pseudotime = (cell_pseudo - pt_min) / (pt_max - pt_min)
    else:
        pseudotime = np.zeros_like(cell_pseudo)

    return pseudotime



def enforce_qscore_monotonicity(
    pseudotime: np.ndarray,
    q_score: np.ndarray,
    mode: str = "balanced"
) -> np.ndarray:
    """
    利用细胞级 q_score 对原始 pseudotime 做轻量级单调化约束。

    思路：
    - 在 q_score 轴上做一维 isotonic regression，
      拟合 pseudo ≈ f(q_score)，其中 f 单调递增；
    - 再用 f(q) 作为“q_score 支持下的伪时间”，
      然后与原始 pseudotime 做 convex combination。

    mode 控制融合强度：
      fast:   强度=0.3
      balanced: 0.5
      full:  0.7
    """
    pseudo = np.asarray(pseudotime, dtype=float)
    q = np.asarray(q_score, dtype=float)

    # 仅使用有有效 q_score 的细胞做拟合
    mask = np.isfinite(q)
    if mask.sum() < 50:  # 有效 BCR 太少时不做
        return pseudo

    q_valid = q[mask]
    pseudo_valid = pseudo[mask]

    # 防止 q_valid 全相同，避免 isotonic 退化
    if np.allclose(q_valid.min(), q_valid.max()):
        return pseudo

    ir = IsotonicRegression(increasing=True, out_of_bounds='clip')
    pseudo_fit = ir.fit_transform(q_valid, pseudo_valid)

    # 将拟合结果映射回全体细胞：只对有 q 的细胞预测
    pseudo_q = np.full_like(q, np.nan, dtype=float)
    q_clip = np.clip(q[mask], q_valid.min(), q_valid.max())
    pseudo_q[mask] = ir.predict(q_clip)

    # 融合强度
    if mode == "fast":
        alpha = 0.3
    elif mode == "full":
        alpha = 0.7
    else:
        alpha = 0.5

    # 默认保留原始 pseudotime；仅对有 q 的细胞做 convex combination
    pseudo_new = pseudo.copy()
    pseudo_new[mask] = (1 - alpha) * pseudo[mask] + alpha * pseudo_q[mask]

    # 归一化到 [0, 1]
    mn, mx = np.nanmin(pseudo_new), np.nanmax(pseudo_new)
    if mx > mn:
        pseudo_new = (pseudo_new - mn) / (mx - mn)
    else:
        pseudo_new = np.zeros_like(pseudo_new)

    return pseudo_new


def compute_pseudotime_with_q(
    features: np.ndarray,
    cell_labels: np.ndarray,
    Lineage_class: Any,
    start_node: int,
    q_score: Optional[np.ndarray] = None,
    mode: str = "balanced",
    align_cluster = False,
    shift_strength=0.3
) -> np.ndarray:
    """
    在原 compute_pseudotime 基础上，增加 q_score 单调约束后处理。
    """
    pseudo_raw = compute_pseudotime(
        features=features,
        cell_labels=cell_labels,
        Lineage_class=Lineage_class,
        start_node=start_node
    )

    if q_score is None:
        return pseudo_raw

    pseudo_q = enforce_qscore_monotonicity(
        pseudotime=pseudo_raw,
        q_score=q_score,
        mode=mode
    )

    if align_cluster:
        pseudo_q = align_cluster_median_with_q(pseudo_q, cell_labels, q_score, shift_strength=shift_strength)
        return pseudo_q
    else:
        return pseudo_q


### 可选：在cluster级别硬保证中位数顺序
def align_cluster_median_with_q(
    pseudotime: np.ndarray,
    cell_labels: np.ndarray,
    q_score: np.ndarray,
    shift_strength: float = 0.3
) -> np.ndarray:
    """
    使得 cluster 级 median(pseudo) 的排序不严重违背 median(q) 的排序。
    通过对每个 cluster 的伪时间整体做小幅线性平移实现。
    """
    pseudo = np.asarray(pseudotime, dtype=float)
    labels = np.asarray(cell_labels)
    q = np.asarray(q_score, dtype=float)

    n_clusters = int(labels.max()) + 1
    q_median, _ = compute_cluster_q_stats(q, labels)

    # cluster 伪时间中位数
    p_median = np.full(n_clusters, np.nan)
    for k in range(n_clusters):
        mask = (labels == k)
        if np.any(mask):
            p_median[k] = np.nanmedian(pseudo[mask])

    # 只看有 q_median & p_median 的 cluster
    valid_mask = np.isfinite(q_median) & np.isfinite(p_median)
    idx = np.where(valid_mask)[0]
    if idx.size < 3:
        return pseudo

    qm = q_median[idx]
    pm = p_median[idx]

    # 理想关系：pm_sorted 的顺序与 qm_sorted 顺序一致
    # 简化处理：在 qm 上做线性回归拟合目标伪时间中心，再对每个 cluster 做 shift
    from sklearn.linear_model import LinearRegression
    qm_2d = qm.reshape(-1, 1)
    lr = LinearRegression()
    lr.fit(qm_2d, pm)
    pm_target = lr.predict(qm_2d)

    # 对每个 cluster 的伪时间整体 shift
    pseudo_new = pseudo.copy()
    for k, qk, pk, pkt in zip(idx, qm, pm, pm_target):
        delta = (pkt - pk) * shift_strength
        mask = (labels == k)
        pseudo_new[mask] += delta

    # 再次归一化
    mn, mx = np.nanmin(pseudo_new), np.nanmax(pseudo_new)
    if mx > mn:
        pseudo_new = (pseudo_new - mn) / (mx - mn)
    else:
        pseudo_new = np.zeros_like(pseudo_new)
    return pseudo_new

def draw_lineage_graph_simple(Lineage_class, ids2name, seed=42, k=0.8, iterations=500,
                              out_pdf="lineage_graph.pdf"):
    import networkx as nx
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.patches as mpatches

    n_cluster = len(ids2name)

    # 1) 建 有向图
    G = nx.DiGraph()
    G.add_nodes_from(range(n_cluster))

    # 2) 把每条 lineage 按顺序加“有向”边
    for lineage in Lineage_class.lineages:
        nodes = lineage.lineage          # 假设是 [c0, c1, c2, ...]
        for u, v in zip(nodes, nodes[1:]):
            G.add_edge(int(u), int(v))

    pos = nx.spring_layout(G, seed=seed, k=k, iterations=iterations)

    fig, ax = plt.subplots(figsize=(6, 6))

    # 3) 画带箭头的边
    nx.draw_networkx_edges(G, pos, edge_color='gray', width=2,
                           arrowsize=20, arrowstyle='->', arrows=True)

    # 节点
    node_color = np.arange(n_cluster)
    nx.draw_networkx_nodes(G, pos,
                           node_color=node_color,
                           node_size=400,
                           cmap='tab20',
                           ax=ax)

    # 标签
    labels = {i: ids2name[i] for i in range(n_cluster)}
    nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)

    # legend
    cmap = plt.cm.tab20
    legend_handles = [mpatches.Patch(color=cmap(i % 20),
                                     label=f'{i}: {ids2name[i]}')
                      for i in range(n_cluster)]
    ax.legend(handles=legend_handles,
              bbox_to_anchor=(1.05, 1),
              loc='upper left',
              fontsize=8)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out_pdf, dpi=300, bbox_inches='tight')
    plt.close(fig)

def draw_lineage_graph_3d(Lineage_class, cell_labels,
                          ids2name=None, name2ids=None,
                          seed=42, k=0.8, iterations=500,
                          figsize=(6, 6), legend_font=8):
    import networkx as nx
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.patches as mpatches
    from mpl_toolkits.mplot3d import Axes3D   #  noqa: F401 (仅注册 3d)

    uniq = np.unique(cell_labels)
    n_cluster = len(uniq)

    if name2ids is None:
        name2ids = {lab: i for i, lab in enumerate(uniq)}
    if ids2name is None:
        ids2name = {i: lab for i, lab in enumerate(uniq)}

    G = nx.Graph()
    G.add_nodes_from(range(n_cluster))
    G.add_edges_from(Lineage_class.all_lineage_seq)

    pos_3d = nx.spring_layout(G, dim=3, seed=seed, k=k, iterations=iterations)

    xs = [pos_3d[i][0] for i in range(n_cluster)]
    ys = [pos_3d[i][1] for i in range(n_cluster)]
    zs = [pos_3d[i][2] for i in range(n_cluster)]

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # 画边
    for u, v in G.edges():
        x = [pos_3d[u][0], pos_3d[v][0]]
        y = [pos_3d[u][1], pos_3d[v][1]]
        z = [pos_3d[u][2], pos_3d[v][2]]
        ax.plot(x, y, z, color='gray', linewidth=1)

    # 画点
    node_colors = np.arange(n_cluster)
    cmap = plt.cm.tab20
    sc = ax.scatter(xs, ys, zs,
                    c=node_colors,
                    s=60,
                    cmap=cmap,
                    depthshade=True)

    # 节点标签：真实簇名
    for i in range(n_cluster):
        ax.text(xs[i], ys[i], zs[i],
                f'{i}\n{ids2name[i]}',   # 第二行放簇名
                fontsize=7,
                ha='center', va='bottom')

    # 生成 legend
    legend_handles = []
    for i in range(n_cluster):
        color = cmap(i % 20)
        legend_handles.append(
            mpatches.Patch(color=color, label=f'{i}: {ids2name[i]}')
        )
    ax.legend(handles=legend_handles,
              loc='upper left',
              bbox_to_anchor=(1.05, 1),
              fontsize=legend_font)

    ax.set_axis_off()
    plt.tight_layout()
    plt.show()





def plot_lineage_skeleton_on_umap_v2(
    umap: np.ndarray,
    cell_labels_int: np.ndarray,
    Lineage_class: Any,
    ids2name: Dict[int, str],
    ax: Optional[plt.Axes] = None,
    linewidth: float = 3.0,
    alpha_line: float = 0.9,
    alpha_points: float = 0.3,
    s_points: float = 4.0,
    colors: Optional[Sequence[str]] = None,
):
    """
    在 UMAP(2D) 上绘制各条 lineage 的平滑样条骨架曲线（而不是折线）。
    """

    umap = np.asarray(umap)
    labels = np.asarray(cell_labels_int)
    n_clusters = int(labels.max()) + 1

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    # 背景散点
    ax.scatter(
        umap[:, 0],
        umap[:, 1],
        s=s_points,
        c="lightgray",
        alpha=alpha_points,
        edgecolors="none",
        rasterized=True,
    )

    # cluster 中心（在 UMAP 空间）
    centers = np.zeros((n_clusters, 2), dtype=float)
    for k in range(n_clusters):
        mask = labels == k
        if np.any(mask):
            centers[k] = umap[mask].mean(axis=0)
        else:
            centers[k] = np.nan

    # 颜色
    num_lineages = len(Lineage_class.lineages)
    if colors is None:
        base_cmap = plt.get_cmap("tab10")
        colors = [base_cmap(i % 10) for i in range(num_lineages)]

    for idx, single in enumerate(Lineage_class.lineages, start=1):
        lineage_clusters = [int(c) for c in single.lineage]
        pts = centers[lineage_clusters]  # (L, 2)
        valid = np.all(np.isfinite(pts), axis=1)
        pts = pts[valid]
        L = pts.shape[0]
        if L < 2:
            continue

        # 用 B-spline 对簇中心进行平滑插值
        # k 为样条阶数：L<=3 用一次/二次曲线，L>=4 用三次样条
        k_spline = min(3, L-1 )
        # 为避免数值问题，确保点之间有足够变化
        try:
            tck, u = splprep(pts.T, s=0.3 * L, k=k_spline)  # s 为平滑度，可按需要调节
            u_fine = np.linspace(0, 1, 100)
            x_smooth, y_smooth = splev(u_fine, tck)
        except Exception:
            # 如果样条拟合失败，退回简单折线
            x_smooth, y_smooth = pts[:, 0], pts[:, 1]

        ax.plot(
            x_smooth,
            y_smooth,
            "-",
            color=colors[idx - 1],
            linewidth=linewidth,
            alpha=alpha_line,
            label=f"lineage{idx}",
        )

        # 起点/终点
        ax.scatter(
            pts[0, 0], pts[0, 1],
            s=30, color=colors[idx - 1], edgecolor="black", zorder=5
        )
        ax.scatter(
            pts[-1, 0], pts[-1, 1],
            s=30, color=colors[idx - 1], edgecolor="black", zorder=5
        )

        # 起点→终点用箭头
        # ax.annotate('', xy=(x_smooth[-1], y_smooth[-1]),
        #             xytext=(x_smooth[-2], y_smooth[-2]),
        #             arrowprops=dict(arrowstyle='->', lw=0.8, color=colors[idx-1]))
        # 中间段用虚线，可进一步弱化视觉干扰
        # ax.plot(x_smooth[1:-1], y_smooth[1:-1], "--",
        #         color=colors[idx-1], lw=lw*0.6, alpha=alpha_line*0.8)

    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(frameon=False)
    ax.set_title("Lineage skeletons on UMAP (smoothed)")
    plt.tight_layout()
    return ax