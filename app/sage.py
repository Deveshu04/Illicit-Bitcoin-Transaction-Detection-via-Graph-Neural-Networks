import numpy as np
from scipy import sparse

from graph import STRUCTURAL

LOG1P = ("in_degree", "out_degree", "core_number", "avg_neighbor_degree", "two_hop_size")
LOG = ("pagerank",)


def gnn_matrix(raw, structural):
    values = np.array(structural, dtype=float)
    for name in LOG1P:
        i = STRUCTURAL.index(name)
        values[:, i] = np.log1p(values[:, i])
    for name in LOG:
        i = STRUCTURAL.index(name)
        values[:, i] = np.log(values[:, i])
    return np.hstack([np.asarray(raw, dtype=float), values])


def symmetric(n, src, dst):
    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    rows = np.concatenate([dst, src])
    cols = np.concatenate([src, dst])
    keep = rows != cols
    adj = sparse.csr_matrix((np.ones(int(keep.sum())), (rows[keep], cols[keep])), shape=(n, n))
    adj.sum_duplicates()
    adj.data[:] = 1.0
    return adj


def normalise(adj):
    degree = np.diff(adj.indptr).astype(float)
    scale = np.zeros(adj.shape[0])
    scale[degree > 0] = 1.0 / degree[degree > 0]
    return (sparse.diags(scale) @ adj).tocsr()


def ball(adj, center, hops):
    seen = {int(center)}
    frontier = [int(center)]
    for _ in range(hops):
        reached = []
        for node in frontier:
            for other in adj.indices[adj.indptr[node]:adj.indptr[node + 1]].tolist():
                if other not in seen:
                    seen.add(other)
                    reached.append(other)
        frontier = reached
    return np.array(sorted(seen), dtype=np.int64)


def relu(values):
    return np.maximum(values, 0.0)


class GraphSAGE:
    def __init__(self, path):
        data = np.load(path)
        self.mean = data["mean"].astype(float)
        self.std = data["std"].astype(float)
        self.layers = int(data["layers"])
        count = int(data["n_models"])
        self.models = [
            {key.split(".", 1)[1]: data[key].astype(float) for key in data.files if key.startswith(f"m{k}.")}
            for k in range(count)
        ]

    def forward(self, params, x, adj):
        h = relu(x @ params["inp.weight"].T + params["inp.bias"])
        hidden = [h]
        for layer in range(self.layers):
            prefix = f"convs.{layer}."
            h = relu((adj @ h) @ params[prefix + "lin_l.weight"].T + params[prefix + "lin_l.bias"] + h @ params[prefix + "lin_r.weight"].T)
            hidden.append(h)
        return (np.hstack(hidden) @ params["head.weight"].T + params["head.bias"]).ravel()

    def logits(self, features, adj):
        x = (np.asarray(features, dtype=float) - self.mean) / self.std
        return np.mean([self.forward(params, x, adj) for params in self.models], axis=0)

    def node_logit(self, features, src, dst, node):
        features = np.asarray(features, dtype=float)
        adj = symmetric(len(features), src, dst)
        keep = ball(adj, node, self.layers)
        sub = normalise(adj[keep][:, keep])
        center = int(np.searchsorted(keep, node))
        return float(self.logits(features[keep], sub)[center])
