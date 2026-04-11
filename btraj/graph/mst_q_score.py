import numpy as np
import pandas as pd
import scipy.sparse as sp
from numpy import ndarray, dtype
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

try:
    import igraph as ig
    import leidenalg as la
    HAS_LEIDEN = True
except Exception:
    HAS_LEIDEN = False



def inner_cluster_knn_sparse_new(labels, features, k=20, norm=False, metric='cosine'):
    """Approximate inter-cluster connectivity using kNN graph, avoiding construction of n_cells×n_cells full distance matrix.

    Parameters
    ----
    labels : array-like, shape (n_cells,)
    features : array-like, shape (n_cells, n_features)
    k : int
    norm : bool
    metric : str
    Returns
    ----
    links : ndarray, shape (n_clusters, n_clusters)
    """
    labels = np.asarray(labels)
    features = np.asarray(features)
    n_cells = features.shape[0]

    # Build kNN using NearestNeighbors with sparse storage to avoid large matrices
    nn = NearestNeighbors(
        n_neighbors=min(k + 1, n_cells),  # include self, so k+1
        metric=metric,
        n_jobs=-1  # multi-threading
    )
    nn.fit(features)
    distances, indices = nn.kneighbors(features, return_distance=True)

    # labels[i] -> cluster_i
    uniq = np.unique(labels)
    n_clusters = uniq.shape[0]
    # Map cluster ID to 0..(n_clusters-1) to handle non-consecutive cluster IDs
    cluster_map = {u: idx for idx, u in enumerate(uniq)}
    inv_cluster_map = {idx: u for u, idx in cluster_map.items()}
    cluster_ids = np.vectorize(cluster_map.get)(labels)

    # links_raw[i, j] accumulates cross-cluster kNN connections from cluster i to cluster j
    links_raw = np.zeros((n_clusters, n_clusters), dtype=float)

    # Iterate over each cell's kNN neighbors
    # distances[i, 0] is the cell itself (distance=0), so start from index 1

    for i in range(n_cells):
        ci = cluster_ids[i]
        for nn_idx, dist in zip(indices[i, 1:], distances[i, 1:]):
            cj = cluster_ids[nn_idx]
            if ci == cj:
                continue  # skip intra-cluster edges
            # smaller distance -> larger weight: use 1/(dist+eps)
            w = 1.0 / (dist + 1e-6)
            links_raw[ci, cj] += w

    # Optional: row-normalize so that outgoing edge weights sum to 1 per cluster
    if norm:
        row_sums = links_raw.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        links = links_raw / row_sums
    else:
        links = links_raw

    # If downstream code expects original label ordering instead of 0..n_clusters-1,
    # use uniq / mapping for re-conversion. Here we keep 0..n_clusters-1.

    return links, cluster_map, inv_cluster_map

def compute_cluster_q_stats(q_score: np.ndarray,
                            cell_labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-cluster median q_score and valid sample count.

    Parameters
    ----------
    q_score : (n_cells,) float array, may contain NaN
    cell_labels : (n_cells,) integer cluster IDs

    Returns
    -------
    q_median : (n_clusters,) median q_score per cluster (np.nan if no valid values)
    q_count : (n_clusters,) number of non-NaN q_score samples per cluster
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
    Apply directional penalty/boost to links based on cluster-level q_score medians,
    encouraging the MST to follow the monotonic direction of q_score.

    Parameters
    ----------
    links : (n_clusters, n_clusters) cluster-level connectivity strength (higher = closer)
    cell_labels_int : (n_cells,) integer cluster IDs (0..n_clusters-1, aligned with links)
    q_score : (n_cells,) float array, may contain NaN
    mode : {"fast", "balanced", "full"} controls penalty strength
    q_tol : float, threshold for considering median difference significant
    min_q_count : int, minimum valid q_score samples required per cluster to apply constraint

    Returns
    -------
    new_links : (n_clusters, n_clusters) adjusted connectivity strength
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

    # Return original links if no sufficient q_score information
    if np.all(~np.isfinite(q_median)):
        return links

    new_links = links.copy()
    # Set penalty/boost factors by mode
    if mode == "fast":
        penalty_factor = 0.3   # weaken reverse direction to 30%
        boost_factor = 1.0    # no extra boost for forward direction
    elif mode == "full" :
        if penalty_factor is None:
            penalty_factor = 0.1   # cut reverse edge strength to 10%
        if boost_factor is None:
            boost_factor = 1.2     # slightly boost forward direction
    else:  # balanced
        penalty_factor = 0.2
        boost_factor = 1.1

    for i in range(n_clusters):
        for j in range(n_clusters):
            if i == j:
                continue
            if links[i, j] <= 0:
                continue
            # Both ends need sufficient q_score samples
            if q_count[i] < min_q_count or q_count[j] < min_q_count:
                continue
            qi, qj = q_median[i], q_median[j]
            if not (np.isfinite(qi) and np.isfinite(qj)):
                continue
            diff = qi - qj
            # Desired direction: low q -> high q
            # If i->j is reverse (qi >> qj), penalize i->j
            if diff > q_tol:
                new_links[i, j] *= penalty_factor
                new_links[j, i] *= boost_factor
            elif diff < -q_tol:
                new_links[j, i] *= penalty_factor
                new_links[i, j] *= boost_factor
            # abs(diff) <= q_tol: treated as unreliable, no adjustment

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
    branch_clusters = deque()  # clusters that produce branches
    lineages_list = recurse_branches([], start_node, children, branch_clusters)
    lineages_list = list(flatten(lineages_list))

    if terminal_clusters is not None and len(terminal_clusters) > 0:
        terminal_clusters = set(terminal_clusters)

        def truncate_to_terminal(line):
            # In a lineage, if a terminal node appears,
            # truncate everything after the first terminal node
            for idx, c in enumerate(line):
                if c in terminal_clusters:
                    return line[: idx + 1]
            return line

        new_lineages_list = []
        for line in lineages_list:
            truncated = truncate_to_terminal(line)
            # discard truncated lineages with fewer than 2 nodes
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

        # add lineage in order
        Infered_lineages.add_lineage(each_line, each_line_seq, each_line_index)  # 要加相邻的东西

        for i in each_line:
            cluster_lineseq_mat[i, each_line_index] = 1
    Infered_lineages.cluster_lineseq_mat = cluster_lineseq_mat

    return Infered_lineages, branch_clusters, tree  # return tree as well


# ===== 3. Generate raw lineages from directed graph and root =====
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

# ===== 4. q monotonicity truncation =====
def enforce_lineage_q_monotone(lineage, q_median, q_tol_cluster=0.02):
    """
    Preserve the path as much as possible, only skip nodes with obvious
    reverse direction rather than truncating the entire segment.
    """
    if not lineage:
        return lineage

    new_lineage = [lineage[0]]
    last_q = q_median[lineage[0]]

    for c in lineage[1:]:
        qc = q_median[c]
        if not (np.isfinite(last_q) and np.isfinite(qc)):
            # Either end has no q info: keep the node
            new_lineage.append(c)
            last_q = qc
            continue

        # If there is a clear drop: qc < last_q - tol, skip this node but don't break the line
        if qc < last_q - q_tol_cluster:
            continue

        new_lineage.append(c)
        last_q = qc

    return new_lineage

def getLineage_v2(links, cell_labels, q_score, cell_labels_int, start_node=0,q_tol_cluster=0.02):
    """
    1. Compute MST from links;
    2. Orient MST edges using q_median (low q -> high q);
    3. Build lineages from directed graph starting at start_node;
    4. Apply q monotonicity truncation.

    links: cluster-level similarity/weight matrix (hidden_clusters_links)
    cell_labels: per-cell cluster labels (int or can be mapped to int)
    q_score: per-cell q (e.g. SHM-based) array
    cell_labels_int: same shape as q_score, int cluster IDs
    start_node: root cluster ID (e.g. naive B)
    """
    from collections import defaultdict
    num_clusters = len(set(cell_labels))

    # ===== 1. Cluster-level q statistics =====
    q_median, q_count = compute_cluster_q_stats(q_score, cell_labels_int)

    # ===== 2. MST + orient edges using q_median: low q -> high q =====
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
            # Missing q info -> bidirectional
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

    # ===== 5. Build Lineage object (preserving original structure) =====
    Infered_lineages = Lineage()
    lineage_seq = []
    adj_cluster2lineage_seq = dict()

    # Note: use lineages here, not lineages_list
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

    # ===== 6. branch_clusters: nodes with out-degree > 1 in directed graph =====
    branch_clusters = []
    for each_line in lineages:
        for c in each_line:
            # nodes with out-degree > 1 in the directed MST are treated as branches
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
    Generate monocle-style milestone_network, milestone_percentages, and progressions
    based on inferred inter-cluster trajectory (predict_order) and cell features.

    Improvements over original version:
    - No longer explicitly constructs all interpolation points line_points and uses
      pairwise_distances to find nearest points;
    - Instead: group cells by cluster, for each edge (i, j) only project cells
      belonging to cluster i or j;
    - Use analytic geometry to directly compute relative position (percentage, 0~1)
      of each cell on the edge. Time complexity O(N_cells x d), suitable for 100k+ cells.

    Parameters
    ----------
    predict_order : ndarray, shape (n_edges, 2)
        Each row is [start_cluster, end_cluster], cluster IDs are 0..n_clusters-1.
    features : ndarray, shape (n_cells, n_features)
        Low-dimensional cell features.
    cell_labels : array-like, shape (n_cells,)
        Integer cluster IDs (must match cluster numbering used in predict_order).
    cells : array-like, shape (n_cells,)
        Cell IDs (strings or other types convertible to str).
    ids2name : dict[int, Any], optional
        Maps integer cluster IDs back to original labels (e.g. 'A', 'B'). If None, use integers directly.
    sep_points : int
        Interpolation granularity for computing edge "length" weights in milestone_network (default 199).
        Actual cell projection no longer uses these interpolation points explicitly,
        kept only for compatibility with original length definition.

    Returns
    -------
    [milestone_network, milestone_percentages, progressions]
        milestone_network : DataFrame(columns=['from', 'to', 'length', 'directed'])
        milestone_percentages : DataFrame(columns=['cell_id', 'milestone_id', 'percentage'])
        progressions : DataFrame(columns=['cell_id', 'from', 'to', 'percentage'])
    """
    features = np.asarray(features, dtype=float)
    cell_labels = np.asarray(cell_labels)
    cells = np.asarray(cells)

    # 1. Normalize cell_labels to integer cluster IDs starting from 0 (aligned with predict_order).
    #    If cell_labels is already 0..n_clusters-1, this step is a no-op,
    #    but ensures safe indexing for centers below.
    uniq_labels = np.unique(cell_labels)
    label_map = {lab: idx for idx, lab in enumerate(uniq_labels)}
    inv_label_map = {idx: lab for lab, idx in label_map.items()}
    normalized_labels = np.vectorize(label_map.get)(cell_labels)
    n_clusters = len(uniq_labels)

    # 2. Compute cluster centers
    centers = np.zeros((n_clusters, features.shape[1]), dtype=float)
    for k in range(n_clusters):
        mask = (normalized_labels == k)
        if np.any(mask):
            centers[k] = features[mask].mean(axis=0)
        else:
            centers[k] = 0.0  # should not occur in practice

    predict_order = np.asarray(predict_order, dtype=int)
    n_edges = predict_order.shape[0]

    # 3. Group cell indices by cluster for efficient per-edge processing
    cluster_to_cell_indices: Dict[int, List[int]] = {k: [] for k in range(n_clusters)}
    for idx, lab in enumerate(normalized_labels):
        cluster_to_cell_indices[int(lab)].append(idx)

    # 4. Determine each cell's assigned edge and relative position (percentage) on that edge.
    #    If a cell belongs to multiple edge endpoint clusters, select the edge with the smallest
    #    projection residual distance.
    n_cells = features.shape[0]
    best_dist = np.full(n_cells, np.inf, dtype=float)       # distance to nearest edge
    best_edge_idx = np.full(n_cells, -1, dtype=int)         # which edge is used
    best_percentage = np.zeros(n_cells, dtype=float)        # position on that edge [0,1]

    # 5. Project cells onto each edge
    for edge_idx, (ci, cj) in enumerate(predict_order):
        ci = int(ci)
        cj = int(cj)

        # Centers of the two endpoint clusters
        pi = centers[ci]
        pj = centers[cj]
        edge_vec = pj - pi
        edge_len = np.linalg.norm(edge_vec)

        # Edge case: two cluster centers coincide, skip
        if edge_len == 0.0:
            continue

        direction = edge_vec / edge_len

        # Cell indices belonging to ci or cj
        idx_ci = cluster_to_cell_indices.get(ci, [])
        idx_cj = cluster_to_cell_indices.get(cj, [])
        idx_all = np.array(sorted(set(idx_ci) | set(idx_cj)), dtype=int)
        if idx_all.size == 0:
            continue

        X = features[idx_all]  # (n_subcells, d)
        vec = X - pi           # vectors from pi
        # Projection length along edge direction (may be <0 or >edge_len)
        proj_len = np.dot(vec, direction)  # (n_subcells,)

        # Clip to [0, edge_len]
        proj_len_clipped = np.clip(proj_len, 0.0, edge_len)
        # Nearest point coordinates on the edge
        proj_points = pi[None, :] + proj_len_clipped[:, None] * direction[None, :]
        # Residual distance from cell to nearest point, used to rank edge fitness
        residual_vec = X - proj_points
        dist_to_edge = np.linalg.norm(residual_vec, axis=1)

        # Relative position percentage
        frac = proj_len_clipped / edge_len  # (0~1)

        # Update best edge assignment if this edge is closer
        better_mask = dist_to_edge < best_dist[idx_all]
        if np.any(better_mask):
            update_indices = idx_all[better_mask]
            best_dist[update_indices] = dist_to_edge[better_mask]
            best_edge_idx[update_indices] = edge_idx
            best_percentage[update_indices] = frac[better_mask]

    # 6. Build progressions DataFrame
    #    Each cell: cell_id, from, to, percentage
    #    If a cell is not captured by any edge (extremely rare), set percentage=0,
    #    from/to use that cell's own cluster.
    edge_for_cell = best_edge_idx
    percentage = best_percentage.copy()

    # Handle cells with no matched edge
    no_edge_mask = (edge_for_cell < 0)
    if np.any(no_edge_mask):
        # Set from/to to their own cluster, percentage=0
        # Assign a dummy edge (index 0) just for output consistency
        edge_for_cell[no_edge_mask] = 0
        percentage[no_edge_mask] = 0.0

    # Map each cell to its edge's start and end cluster IDs
    edge_starts = predict_order[:, 0]
    edge_ends = predict_order[:, 1]
    cell_from_clusters = edge_starts[edge_for_cell]
    cell_to_clusters = edge_ends[edge_for_cell]

    # Restore original cluster labels (strings) or integers
    if ids2name is not None:
        # predict_order and normalized_labels both use [0..n_clusters-1],
        # first map internal cluster ID -> original label value,
        # then apply ids2name to get final user-facing names.
        cell_from_orig = np.vectorize(inv_label_map.get)(cell_from_clusters)
        cell_to_orig = np.vectorize(inv_label_map.get)(cell_to_clusters)
        cell_starts = [ids2name[int(lab)] for lab in cell_from_orig]
        cell_targets = [ids2name[int(lab)] for lab in cell_to_orig]

        net_from_orig = np.vectorize(inv_label_map.get)(edge_starts)
        net_to_orig = np.vectorize(inv_label_map.get)(edge_ends)
        network_starts = [ids2name[int(lab)] for lab in net_from_orig]
        network_targets = [ids2name[int(lab)] for lab in net_to_orig]
    else:
        # Use internal integer cluster IDs directly
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
    milestone_percentages = pd.DataFrame(
        {
            "cell_id": cells.astype(str),
            "milestone_id": np.asarray(cell_starts).astype(str),
            "percentage": np.zeros_like(cells, dtype=float).astype(str),
        }
    )

    # 8. milestone_network
    #    Requires length and directed. Length is still determined by number of cells
    #    assigned to each edge, normalized to [0, 2] for compatibility with original function.
    # Count how many cells are assigned to each edge
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
    Convert each lineage in Lineage_class into a human-readable
    "celltypeA --> celltypeB --> ..." text description.

    Parameters
    ----------
    Lineage_class : Lineage object returned by get_trajectory
    ids2name : dict[int -> str], mapping from integer cluster ID to original name
    print_out : whether to print directly; if False, only return the list

    Returns
    -------
    descriptions : list[str]
        Each element looks like "lineage1: A --> B --> C"
    """
    descriptions = []
    for idx, single in enumerate(Lineage_class.lineages, start=1):
        # single.lineage is a sequence of cluster IDs, e.g. [0, 3, 5]
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
    Extended version: supports q_score constraints + automatic mode selection.

    mode:
        None -> automatically determined based on cell count: {"fast", "balanced", "full"}
    """
    n_cells = y_features.shape[0]
    if mode is None:
        if n_cells >= 100000:
            mode = "fast"
        elif n_cells >= 30000:
            mode = "balanced"
        else:
            mode = "full"

    # 1. clusters_links: approximate inter-cluster connectivity using cell-level kNN

    clusters_links, _, _ = inner_cluster_knn_sparse_new(cell_labels, y_features, k=k, norm=norm)
    # print(clusters_links)
    # 2. Normalize cell_labels to consecutive integer IDs 0..C-1
    uniq = np.unique(cell_labels)
    ids2name = {i: lab for i, lab in enumerate(uniq)}
    name2ids = {lab: i for i, lab in enumerate(uniq)}
    cell_labels_int = np.array([name2ids[i] for i in cell_labels])

    # 3. If q_score is provided, apply directional constraint adjustment to links

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

    # 4. MST and lineage construction using original logic
    start_node = int(np.where(uniq == start_type)[0][0])
    terminal_clusters = []
    if terminal_types is not None:

        for t in terminal_types:
            idx = np.where(uniq == t)[0]
            if len(idx) == 0:
                print(f'Warning: terminal type {t} not found in uniq')
                continue
            terminal_clusters.append(int(idx[0]))
    Lineage_class, branch_clusters, tree = getLineage(clusters_links, cell_labels, start_node=start_node,terminal_clusters=terminal_clusters)

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
    Compute the center (mean vector) of each cluster from cell features and cluster labels.

    Parameters
    ----------
    features : (n_cells, n_features)
    cell_labels : (n_cells,) integer cluster IDs, must be consecutive integers 0..n_clusters-1

    Returns
    -------
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
            centers[k] = 0.0
    return centers


def build_cluster_graph_from_lineages(lineage_seq: np.ndarray,
                                      num_clusters: int) -> Dict[int, list]:
    """
    Build an undirected adjacency graph from Lineage_class.all_lineage_seq (list of cluster edges).

    Parameters
    ----------
    lineage_seq : (n_edges, 2) each row is [start_cluster, end_cluster]
    num_clusters : total number of clusters

    Returns
    -------
    graph : dict[int, list[int]] adjacency list
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
    Compute cumulative distance from start_node to every cluster on the MST,
    and the geometric length of each undirected edge.

    Parameters
    ----------
    start_node : root cluster ID (integer)
    graph : cluster graph adjacency list (MST structure)
    centers : (n_clusters, n_features) cluster centers

    Returns
    -------
    tree_dist : (n_clusters,) tree distance from root to each cluster
    edge_len : dict[(min(u,v), max(u,v))] -> edge_length
    """
    n_clusters = centers.shape[0]
    tree_dist = np.full(n_clusters, np.inf, dtype=float)
    tree_dist[start_node] = 0.0

    edge_len: Dict[Tuple[int, int], float] = {}

    for u in range(n_clusters):
        for v in graph[u]:
            if u < v:
                length = np.linalg.norm(centers[u] - centers[v])
                edge_len[(u, v)] = float(length)

    dq = deque([start_node])
    visited = np.zeros(n_clusters, dtype=bool)
    visited[start_node] = True

    while dq:
        u = dq.popleft()
        for v in graph[u]:
            if not visited[v]:
                key = (u, v) if u < v else (v, u)
                length = edge_len[key]
                tree_dist[v] = tree_dist[u] + length
                visited[v] = True
                dq.append(v)

    return tree_dist, edge_len


def project_cells_onto_edge(features: np.ndarray,
                            centers: np.ndarray,
                            cell_labels: np.ndarray,
                            edge: Tuple[int, int]) -> tuple[ndarray[Any, dtype[Any]], float]:
    """
    Project cells belonging to either endpoint cluster of an edge onto the
    line segment connecting the two cluster centers, yielding:
    - relative position on the edge (0~1) for each cell; np.nan for cells not in either cluster.
    - Euclidean length of the edge.

    Parameters
    ----------
    features : (n_cells, n_features)
    centers : (n_clusters, n_features)
    cell_labels : (n_cells,) integer cluster IDs
    edge : (u, v) integer cluster IDs, corresponding to centers[u] -> centers[v]

    Returns
    -------
    frac : (n_cells,) relative position on edge [0,1], np.nan for non-member cells
    edge_length : float geometric length of the edge
    """
    u, v = edge
    cu = centers[u]
    cv = centers[v]
    diff = cv - cu
    edge_length = float(np.linalg.norm(diff))

    if edge_length == 0.0:
        frac = np.full(features.shape[0], np.nan, dtype=float)
        mask = (cell_labels == u) | (cell_labels == v)
        frac[mask] = 0.0
        return frac, 0.0

    direction = diff / edge_length  # shape (d,)

    mask = (cell_labels == u) | (cell_labels == v)
    frac = np.full(features.shape[0], np.nan, dtype=float)
    if not np.any(mask):
        return frac, edge_length

    X = features[mask]  # (n_subcells, d)
    # proj_len = (X - cu) · direction
    proj_vec = X - cu  # (n_subcells, d)
    proj_len = np.dot(proj_vec, direction)  # (n_subcells,)

    proj_len_clip = np.clip(proj_len, 0.0, edge_length)
    frac_edge = proj_len_clip / edge_length  # 0~1

    frac[mask] = frac_edge
    return frac, edge_length


def compute_pseudotime(features: np.ndarray,
                       cell_labels: np.ndarray,
                       Lineage_class: Any,
                       start_node: int) -> np.ndarray:
    """
    Compute pseudotime for each cell given an inferred trajectory (Lineage_class, start_node).

    Overall approach:
    1) Use cluster center connections as the trajectory backbone;
    2) Compute tree distance tree_dist from root cluster start_node on the MST;
    3) For each edge (u, v), project cells belonging to u or v onto the edge to get
       intra-edge position frac (0~1);
    4) Cell pseudotime = tree_dist[u] + frac * edge_length;
    5) Take min pseudotime across all edges a cell participates in; then globally
       normalize to [0, 1].

    Parameters
    ----------
    features : (n_cells, n_features)
    cell_labels : (n_cells,) integer cluster IDs, must match name2ids mapping from get_trajectory
    Lineage_class : Lineage object returned by get_trajectory
    start_node : integer ID of the root cluster (start_node returned by get_trajectory)

    Returns
    -------
    pseudotime : (n_cells,) float array, normalized to [0, 1]
    """
    features = np.asarray(features)
    cell_labels = np.asarray(cell_labels)

    # 1) Cluster centers
    centers = compute_cluster_centers(features, cell_labels)
    n_clusters = centers.shape[0]

    # 2) Build undirected MST graph from Lineage_class.all_lineage_seq
    lineage_seq = np.asarray(Lineage_class.all_lineage_seq, dtype=int)
    graph = build_cluster_graph_from_lineages(lineage_seq, n_clusters)

    # 3) Compute tree distance from root to each cluster on the MST
    tree_dist, edge_len = compute_tree_distances(start_node, graph, centers)

    n_cells = features.shape[0]
    # Initialize to +inf; use min to select best pseudotime per cell
    cell_pseudo = np.full(n_cells, np.inf, dtype=float)

    # 4) Iterate over each edge, project cells, and update pseudotime
    #    Edge lengths are taken from edge_len for consistency with tree_dist
    for u, v in lineage_seq:
        u = int(u)
        v = int(v)
        key = (u, v) if u < v else (v, u)
        length = edge_len.get(key, None)
        if length is None:
            # Should not occur in practice
            continue

        # Project cells belonging to u or v onto this edge
        frac, _ = project_cells_onto_edge(features, centers, cell_labels, (u, v))

        base_u = tree_dist[u]
        base_v = tree_dist[v]

        # Cells in cluster u: pseudotime = tree_dist[u] + frac * length
        mask_u = (cell_labels == u) & (~np.isnan(frac))
        if np.any(mask_u):
            pseudo_u = base_u + frac[mask_u] * length
            cell_pseudo[mask_u] = np.minimum(cell_pseudo[mask_u], pseudo_u)

        # Cells in cluster v: use symmetric formulation from v's perspective
        mask_v = (cell_labels == v) & (~np.isnan(frac))
        if np.any(mask_v):
            # On this edge, frac is defined as u->v direction,
            # so from v's perspective the position is (1 - frac)
            frac_v = 1.0 - frac[mask_v]
            pseudo_v = base_v + frac_v * length
            cell_pseudo[mask_v] = np.minimum(cell_pseudo[mask_v], pseudo_v)

    # Fall back to tree_dist for any cells not assigned to any lineage edge (np.inf remaining)
    inf_mask = ~np.isfinite(cell_pseudo)
    if np.any(inf_mask):
        cell_pseudo[inf_mask] = tree_dist[cell_labels[inf_mask]]

    # 5) Globally normalize to [0, 1]
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
    Apply lightweight monotonic constraint to raw pseudotime using cell-level q_score.

    Approach:
    - Fit a 1D isotonic regression on the q_score axis,
      fitting pseudo ≈ f(q_score) where f is monotonically increasing;
    - Use f(q) as the "q_score-supported pseudotime",
      then blend with original pseudotime via convex combination.

    mode controls blending strength:
      fast:     strength = 0.3
      balanced: strength = 0.5
      full:     strength = 0.7
    """
    pseudo = np.asarray(pseudotime, dtype=float)
    q = np.asarray(q_score, dtype=float)

    # Only use cells with valid q_score for fitting
    mask = np.isfinite(q)
    if mask.sum() < 50:  # skip if too few valid BCR cells
        return pseudo

    q_valid = q[mask]
    pseudo_valid = pseudo[mask]

    # Guard against degenerate case where all q values are identical
    if np.allclose(q_valid.min(), q_valid.max()):
        return pseudo

    ir = IsotonicRegression(increasing=True, out_of_bounds='clip')
    pseudo_fit = ir.fit_transform(q_valid, pseudo_valid)

    # Map fitted results back to all cells: only predict for cells with valid q
    pseudo_q = np.full_like(q, np.nan, dtype=float)
    q_clip = np.clip(q[mask], q_valid.min(), q_valid.max())
    pseudo_q[mask] = ir.predict(q_clip)

    # Blending strength by mode
    if mode == "fast":
        alpha = 0.3
    elif mode == "full":
        alpha = 0.7
    else:
        alpha = 0.5

    # Keep original pseudotime by default; apply convex combination only for cells with valid q
    pseudo_new = pseudo.copy()
    pseudo_new[mask] = (1 - alpha) * pseudo[mask] + alpha * pseudo_q[mask]

    # Normalize to [0, 1]
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
    Extends compute_pseudotime with optional q_score monotonic post-processing.
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


### Optional: hard-enforce cluster-level median ordering
def align_cluster_median_with_q(
    pseudotime: np.ndarray,
    cell_labels: np.ndarray,
    q_score: np.ndarray,
    shift_strength: float = 0.3
) -> np.ndarray:
    """
    Ensure that cluster-level median(pseudotime) ordering does not seriously
    violate median(q_score) ordering.
    Achieved by applying small linear shifts to each cluster's pseudotime values.
    """
    pseudo = np.asarray(pseudotime, dtype=float)
    labels = np.asarray(cell_labels)
    q = np.asarray(q_score, dtype=float)

    n_clusters = int(labels.max()) + 1
    q_median, _ = compute_cluster_q_stats(q, labels)

    # Cluster pseudotime medians
    p_median = np.full(n_clusters, np.nan)
    for k in range(n_clusters):
        mask = (labels == k)
        if np.any(mask):
            p_median[k] = np.nanmedian(pseudo[mask])

    # Only consider clusters with both valid q_median and p_median
    valid_mask = np.isfinite(q_median) & np.isfinite(p_median)
    idx = np.where(valid_mask)[0]
    if idx.size < 3:
        return pseudo

    qm = q_median[idx]
    pm = p_median[idx]

    # Ideal relationship: ordering of pm should match ordering of qm.
    # Simplified approach: fit linear regression of pm ~ qm to get target pseudotime centers,
    # then shift each cluster's pseudotime toward its target.
    from sklearn.linear_model import LinearRegression
    qm_2d = qm.reshape(-1, 1)
    lr = LinearRegression()
    lr.fit(qm_2d, pm)
    pm_target = lr.predict(qm_2d)

    # Apply per-cluster pseudotime shift
    pseudo_new = pseudo.copy()
    for k, qk, pk, pkt in zip(idx, qm, pm, pm_target):
        delta = (pkt - pk) * shift_strength
        mask = (labels == k)
        pseudo_new[mask] += delta

    # Re-normalize to [0, 1]
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

    #1) Build directed graph
    G = nx.DiGraph()
    G.add_nodes_from(range(n_cluster))

    # 2) Add directed edges in order for each lineage
    for lineage in Lineage_class.lineages:
        nodes = lineage.lineage          # 假设是 [c0, c1, c2, ...]
        for u, v in zip(nodes, nodes[1:]):
            G.add_edge(int(u), int(v))

    pos = nx.spring_layout(G, seed=seed, k=k, iterations=iterations)

    fig, ax = plt.subplots(figsize=(6, 6))

    # 3) Draw edges with arrows
    nx.draw_networkx_edges(G, pos, edge_color='gray', width=2,
                           arrowsize=20, arrowstyle='->', arrows=True)

    # Draw nodes
    node_color = np.arange(n_cluster)
    nx.draw_networkx_nodes(G, pos,
                           node_color=node_color,
                           node_size=400,
                           cmap='tab20',
                           ax=ax)

    # Draw labels
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
    from mpl_toolkits.mplot3d import Axes3D

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

    for u, v in G.edges():
        x = [pos_3d[u][0], pos_3d[v][0]]
        y = [pos_3d[u][1], pos_3d[v][1]]
        z = [pos_3d[u][2], pos_3d[v][2]]
        ax.plot(x, y, z, color='gray', linewidth=1)

    node_colors = np.arange(n_cluster)
    cmap = plt.cm.tab20
    sc = ax.scatter(xs, ys, zs,
                    c=node_colors,
                    s=60,
                    cmap=cmap,
                    depthshade=True)

    for i in range(n_cluster):
        ax.text(xs[i], ys[i], zs[i],
                f'{i}\n{ids2name[i]}',
                fontsize=7,
                ha='center', va='bottom')

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
    Draw smoothed spline skeleton curves for each lineage on a 2D UMAP embedding
    (instead of polylines connecting cluster centers directly).
    """

    umap = np.asarray(umap)
    labels = np.asarray(cell_labels_int)
    n_clusters = int(labels.max()) + 1

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    ax.scatter(
        umap[:, 0],
        umap[:, 1],
        s=s_points,
        c="lightgray",
        alpha=alpha_points,
        edgecolors="none",
        rasterized=True,
    )

    centers = np.zeros((n_clusters, 2), dtype=float)
    for k in range(n_clusters):
        mask = labels == k
        if np.any(mask):
            centers[k] = umap[mask].mean(axis=0)
        else:
            centers[k] = np.nan

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

        # Fit B-spline through cluster centers for smooth interpolation
        # spline degree k: use linear/quadratic for L<=3, cubic for L>=4
        k_spline = min(3, L-1 )
        try:
            tck, u = splprep(pts.T, s=0.3 * L, k=k_spline)
            u_fine = np.linspace(0, 1, 100)
            x_smooth, y_smooth = splev(u_fine, tck)
        except Exception:
            # Fall back to simple polyline if spline fitting fails
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

        ax.scatter(
            pts[0, 0], pts[0, 1],
            s=30, color=colors[idx - 1], edgecolor="black", zorder=5
        )
        ax.scatter(
            pts[-1, 0], pts[-1, 1],
            s=30, color=colors[idx - 1], edgecolor="black", zorder=5
        )

    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(frameon=False)
    ax.set_title("Lineage skeletons on UMAP (smoothed)")
    plt.tight_layout()
    return ax