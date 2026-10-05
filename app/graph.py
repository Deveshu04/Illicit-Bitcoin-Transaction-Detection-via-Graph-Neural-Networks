import networkx as nx
import numpy as np
from scipy import sparse

STRUCTURAL = ["in_degree", "out_degree", "pagerank", "core_number", "clustering", "avg_neighbor_degree", "two_hop_size"]
AGGREGATES = ["in_mean", "in_max", "out_mean", "out_max", "hop2_mean"]
STACKING = ["gnn_logit", "gnn_in_mean", "gnn_in_max", "gnn_out_mean", "gnn_out_max"]


def engineered_names(attrs):
    return STRUCTURAL + [f"{attr}_{agg}" for attr in attrs for agg in AGGREGATES]


def directed(n, src, dst):
    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    keep = src != dst
    matrix = sparse.csr_matrix((np.ones(int(keep.sum())), (src[keep], dst[keep])), shape=(n, n))
    matrix.sum_duplicates()
    matrix.data[:] = 1.0
    return matrix


def undirected(forward):
    matrix = (forward + forward.T).tocsr()
    matrix.data[:] = 1.0
    return matrix


def two_hop(und):
    n = und.shape[0]
    reach = (und + und @ und).tocoo()
    off = reach.row != reach.col
    return sparse.csr_matrix((np.ones(int(off.sum())), (reach.row[off], reach.col[off])), shape=(n, n))


def segment_mean(matrix, values):
    counts = np.diff(matrix.indptr).astype(float)
    sums = np.asarray(matrix @ values, dtype=float)
    out = np.full(sums.shape, np.nan)
    has = counts > 0
    out[has] = sums[has] / counts[has, None]
    return out


def segment_max(matrix, values):
    out = np.full((matrix.shape[0], values.shape[1]), np.nan)
    has = np.diff(matrix.indptr) > 0
    if has.any():
        out[has] = np.maximum.reduceat(values[matrix.indices], matrix.indptr[:-1][has], axis=0)
    return out


def graph_features(n, src, dst, sel):
    sel = np.asarray(sel, dtype=float).reshape(n, -1)
    forward = directed(n, src, dst)
    backward = forward.T.tocsr()
    reach = two_hop(undirected(forward))
    rows, cols = forward.nonzero()
    digraph = nx.DiGraph()
    digraph.add_nodes_from(range(n))
    digraph.add_edges_from(zip(rows.tolist(), cols.tolist()))
    graph = digraph.to_undirected()
    rank = nx.pagerank(digraph, alpha=0.85)
    core = nx.core_number(graph)
    clustering = nx.clustering(graph)
    neighbour_degree = nx.average_neighbor_degree(graph)
    structural = np.column_stack([
        np.diff(backward.indptr),
        np.diff(forward.indptr),
        [rank[i] for i in range(n)],
        [core[i] for i in range(n)],
        [clustering[i] for i in range(n)],
        [neighbour_degree[i] for i in range(n)],
        np.diff(reach.indptr),
    ]).astype(float)
    blocks = np.stack([
        segment_mean(backward, sel),
        segment_max(backward, sel),
        segment_mean(forward, sel),
        segment_max(forward, sel),
        segment_mean(reach, sel),
    ], axis=2)
    return np.hstack([structural, blocks.reshape(n, -1)])


def stacking_features(n, src, dst, logits):
    forward = directed(n, src, dst)
    backward = forward.T.tocsr()
    values = np.asarray(logits, dtype=float).reshape(n, 1)
    return np.hstack([
        values,
        segment_mean(backward, values),
        segment_max(backward, values),
        segment_mean(forward, values),
        segment_max(forward, values),
    ])
