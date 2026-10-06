import json
import threading
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np

from graph import STRUCTURAL, graph_features, stacking_features
from sage import GraphSAGE, gnn_matrix, symmetric

TOP_DRIVERS = 8
NEIGHBOURHOOD_CAP = 60
NEIGHBOURHOOD_HOPS = 2
MAX_LINKS = 50
MODELS = ("hybrid", "graphsage", "raw_eng", "rf")
LABELS = {1: "illicit", 0: "licit", -1: "unknown"}
FIELDS = {"time_step", "features", "inputs", "outputs"}


class ValidationError(ValueError):
    pass


class NotFoundError(LookupError):
    pass


def number(value):
    value = float(value)
    return value if np.isfinite(value) else None


def sigmoid(value):
    return float(1.0 / (1.0 + np.exp(-float(value))))


def integer(value, field):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(f"{field} must be an integer")
    return value


class Scorer:
    def __init__(self, artifact_dir):
        root = Path(artifact_dir)
        self.meta = json.loads((root / "feature_meta.json").read_text(encoding="utf-8"))
        store = np.load(root / "store.npz")
        self.tx_id = store["tx_id"]
        self.time_step = store["time_step"]
        self.label = store["label"]
        self.raw = store["raw"]
        self.eng = store["eng"]
        self.stack = store["stack"]
        self.scores = {name: store[f"score_{name}"] for name in MODELS}
        src = store["src"].astype(np.int64)
        dst = store["dst"].astype(np.int64)
        self.row = {t: i for i, t in enumerate(self.tx_id.tolist())}
        self.steps = {}
        for step in np.unique(self.time_step).tolist():
            nodes = np.flatnonzero(self.time_step == step)
            mask = self.time_step[src] == step
            local_src = np.searchsorted(nodes, src[mask])
            local_dst = np.searchsorted(nodes, dst[mask])
            self.steps[int(step)] = (nodes, local_src, local_dst, symmetric(len(nodes), local_src, local_dst))
        self.raw_names = self.meta["raw"]
        self.raw_index = {name: i for i, name in enumerate(self.raw_names)}
        self.attr_index = [self.raw_index[a] for a in self.meta["attrs"]]
        self.names = self.meta["hybrid"]
        self.kinds = ["raw"] * len(self.raw_names) + ["graph"] * len(self.meta["engineered"]) + ["gnn"] * len(self.meta["stacking"])
        self.headline = self.meta["headline"]
        self.thresholds = {k: float(v) for k, v in self.meta["thresholds"].items()}
        self.sage = GraphSAGE(root / "graphsage.npz")
        self.booster = lgb.Booster(model_file=str(root / "lgbm.txt"))
        self.lock = threading.Lock()
        self.by_label = {name: np.flatnonzero(self.label == code) for code, name in LABELS.items()}

    def index_of(self, tx_id):
        if tx_id not in self.row:
            raise NotFoundError(f"transaction {tx_id} is not in the test period store")
        return self.row[tx_id]

    def predict(self, vector):
        with self.lock:
            return float(self.booster.predict(vector[None, :])[0])

    def drivers(self, vector):
        with self.lock:
            contrib = self.booster.predict(vector[None, :], pred_contrib=True)[0][:-1]
        order = np.argsort(-np.abs(contrib), kind="stable")[:TOP_DRIVERS]
        return [{"feature": self.names[i], "kind": self.kinds[i], "contribution": float(contrib[i]), "value": number(vector[i])} for i in order]

    def neighbourhood(self, adjacency, src, dst, center, describe):
        hop = {center: 0}
        order = [center]
        frontier = [center]
        for depth in range(1, NEIGHBOURHOOD_HOPS + 1):
            reached = []
            for node in frontier:
                for other in adjacency.indices[adjacency.indptr[node]:adjacency.indptr[node + 1]].tolist():
                    if other not in hop:
                        hop[other] = depth
                        reached.append(other)
            reached.sort()
            order.extend(reached)
            frontier = reached
        kept = order[:NEIGHBOURHOOD_CAP]
        keep = set(kept)
        info = {i: describe(i) for i in kept}
        nodes = [{"id": info[i]["id"], "hop": hop[i], "risk": info[i]["risk"], "label": info[i]["label"], "flagged": info[i]["flagged"]} for i in kept]
        pairs = dict.fromkeys((info[s]["id"], info[d]["id"]) for s, d in zip(src.tolist(), dst.tolist()) if s in keep and d in keep and s != d)
        return {"nodes": nodes, "edges": [list(p) for p in pairs], "truncated": len(order) > len(kept)}

    def describe_stored(self, nodes):
        threshold = self.thresholds[self.headline]
        scores = self.scores[self.headline]

        def describe(i):
            g = int(nodes[i])
            risk = float(scores[g])
            return {"id": int(self.tx_id[g]), "label": LABELS[int(self.label[g])], "risk": risk, "flagged": risk >= threshold}

        return describe

    def transaction(self, tx_id):
        g = self.index_of(tx_id)
        step = int(self.time_step[g])
        nodes, src, dst, adjacency = self.steps[step]
        center = int(np.searchsorted(nodes, g))
        vector = np.concatenate([self.raw[g], self.eng[g], self.stack[g]]).astype(float)
        risk = float(self.scores[self.headline][g])
        threshold = self.thresholds[self.headline]
        return {
            "tx_id": int(tx_id),
            "time_step": step,
            "label": LABELS[int(self.label[g])],
            "headline": self.headline,
            "risk": risk,
            "threshold": threshold,
            "flagged": risk >= threshold,
            "scores": {name: float(self.scores[name][g]) for name in MODELS},
            "drivers": self.drivers(vector),
            "neighbourhood": self.neighbourhood(adjacency, src, dst, center, self.describe_stored(nodes)),
        }

    def compute(self, raw, src, dst, logits, node, replace):
        n = len(raw)
        eng = graph_features(n, src, dst, raw[:, self.attr_index]).astype(np.float32)
        x = gnn_matrix(raw, eng[:, :len(STRUCTURAL)].astype(float))
        logit = np.float32(self.sage.node_logit(x, src, dst, node))
        values = np.array(logits, dtype=np.float32)
        if replace:
            values[node] = logit
        stack = stacking_features(n, src, dst, values)[node].astype(np.float32)
        vector = np.concatenate([raw[node].astype(np.float32), eng[node], stack]).astype(float)
        return {"eng": eng[node], "stack": stack, "logit": float(logit), "vector": vector}

    def recompute(self, tx_id):
        g = self.index_of(tx_id)
        nodes, src, dst, _ = self.steps[int(self.time_step[g])]
        center = int(np.searchsorted(nodes, g))
        out = self.compute(self.raw[nodes].astype(float), src, dst, self.stack[nodes, 0], center, replace=False)
        out["hybrid"] = self.predict(out["vector"])
        return out

    def parse(self, body):
        if not isinstance(body, dict):
            raise ValidationError("body must be a JSON object")
        extra = sorted(set(body) - FIELDS)
        if extra:
            raise ValidationError(f"unknown fields: {', '.join(extra)}")
        step = integer(body.get("time_step"), "time_step")
        if step not in self.steps:
            raise ValidationError(f"time_step must be one of the test steps {min(self.steps)} to {max(self.steps)}")
        features = body.get("features")
        if isinstance(features, dict):
            unknown = sorted(set(features) - set(self.raw_index))
            if unknown:
                raise ValidationError(f"unknown feature names: {', '.join(unknown[:5])}")
            absent = [name for name in self.raw_names if name not in features]
            if absent:
                raise ValidationError(f"missing {len(absent)} features, starting with {absent[0]}")
            values = [features[name] for name in self.raw_names]
        elif isinstance(features, list):
            if len(features) != len(self.raw_names):
                raise ValidationError(f"features must hold {len(self.raw_names)} values")
            values = features
        else:
            raise ValidationError("features must be a list or an object keyed by feature name")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in values):
            raise ValidationError("features must be numbers")
        try:
            raw = np.array(values, dtype=float)
        except OverflowError:
            raise ValidationError("features must fit in 32-bit floats") from None
        if not np.isfinite(raw).all():
            raise ValidationError("features must be finite numbers")
        with np.errstate(over="ignore"):
            raw32 = raw.astype(np.float32)
        if not np.isfinite(raw32).all():
            raise ValidationError("features must fit in 32-bit floats")
        links = {}
        for field in ("inputs", "outputs"):
            value = body.get(field, [])
            if not isinstance(value, list):
                raise ValidationError(f"{field} must be a list of transaction ids")
            links[field] = [integer(v, f"{field} entries") for v in value]
        linked = links["inputs"] + links["outputs"]
        if not linked:
            raise ValidationError("give at least one input or output transaction")
        if len(linked) > MAX_LINKS:
            raise ValidationError(f"at most {MAX_LINKS} linked transactions")
        if len(set(linked)) != len(linked):
            raise ValidationError("linked transactions must be distinct")
        for tx in linked:
            g = self.row.get(tx)
            if g is None:
                raise ValidationError(f"transaction {tx} is not in the test period store")
            if int(self.time_step[g]) != step:
                raise ValidationError(f"transaction {tx} is in time step {int(self.time_step[g])}, not {step}")
        return step, raw32, links["inputs"], links["outputs"]

    def score_new(self, body):
        started = time.perf_counter()
        step, raw32, inputs, outputs = self.parse(body)
        nodes, src, dst, _ = self.steps[step]
        n = len(nodes)
        ins = np.searchsorted(nodes, np.array([self.row[t] for t in inputs], dtype=np.int64)).astype(np.int64)
        outs = np.searchsorted(nodes, np.array([self.row[t] for t in outputs], dtype=np.int64)).astype(np.int64)
        all_src = np.concatenate([src, ins, np.full(len(outs), n, dtype=np.int64)])
        all_dst = np.concatenate([dst, np.full(len(ins), n, dtype=np.int64), outs])
        raw = np.vstack([self.raw[nodes].astype(float), raw32.astype(float)[None, :]])
        logits = np.append(self.stack[nodes, 0], np.float32(0.0))
        out = self.compute(raw, all_src, all_dst, logits, n, replace=True)
        scores = {"hybrid": self.predict(out["vector"]), "graphsage": sigmoid(out["logit"])}
        risk = scores[self.headline]
        threshold = self.thresholds[self.headline]
        stored = self.describe_stored(nodes)

        def describe(i):
            if i == n:
                return {"id": "new", "label": "unknown", "risk": risk, "flagged": risk >= threshold}
            return stored(i)

        return {
            "time_step": step,
            "headline": self.headline,
            "risk": risk,
            "threshold": threshold,
            "flagged": risk >= threshold,
            "scores": scores,
            "drivers": self.drivers(out["vector"]),
            "neighbourhood": self.neighbourhood(symmetric(n + 1, all_src, all_dst), all_src, all_dst, n, describe),
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
        }

    def sample(self, label, rng):
        if label not in self.by_label:
            raise ValidationError("label must be illicit, licit or unknown")
        pool = self.by_label[label]
        if len(pool) == 0:
            raise NotFoundError(f"no {label} transactions in the store")
        return int(self.tx_id[pool[int(rng.integers(len(pool)))]])
