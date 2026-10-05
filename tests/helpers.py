import json
from pathlib import Path

import lightgbm as lgb
import numpy as np

from graph import STACKING, STRUCTURAL, engineered_names, graph_features, stacking_features
from sage import GraphSAGE, gnn_matrix, normalise, symmetric

RAW = [f"local_{i}" for i in range(1, 94)] + [f"agg_{i}" for i in range(1, 73)]
ATTRS = [f"local_{i}" for i in range(1, 35)]
ENG = engineered_names(ATTRS)
HYBRID = RAW + ENG + STACKING
STEPS = (35, 36, 37)
PER_STEP = 10
LABELS = [1, 0, 0, -1, 1, 0, -1, 0, 0, -1]
MODELS = ("hybrid", "graphsage", "raw_eng", "rf")
MODEL_NAMES = {"rf": "Random Forest", "raw_eng": "LightGBM, raw + graph features", "hybrid": "GraphSAGE + LightGBM", "graphsage": "GraphSAGE"}
CONFIGS = [("bce_natural", "Cross-entropy, natural ratio"), ("bce_balanced", "Cross-entropy, balanced seeds"), ("focal_natural", "Focal loss, natural ratio"), ("focal_balanced", "Focal loss, balanced seeds")]


def sage_arrays(rng, d_in, hidden, layers, count):
    arrays = {"layers": np.array(layers), "hidden": np.array(hidden), "n_models": np.array(count)}
    for k in range(count):
        arrays[f"m{k}.inp.weight"] = rng.normal(scale=0.2, size=(hidden, d_in)).astype(np.float32)
        arrays[f"m{k}.inp.bias"] = rng.normal(scale=0.1, size=hidden).astype(np.float32)
        for layer in range(layers):
            arrays[f"m{k}.convs.{layer}.lin_l.weight"] = rng.normal(scale=0.3, size=(hidden, hidden)).astype(np.float32)
            arrays[f"m{k}.convs.{layer}.lin_l.bias"] = rng.normal(scale=0.1, size=hidden).astype(np.float32)
            arrays[f"m{k}.convs.{layer}.lin_r.weight"] = rng.normal(scale=0.3, size=(hidden, hidden)).astype(np.float32)
        arrays[f"m{k}.head.weight"] = rng.normal(scale=0.3, size=(1, hidden * (layers + 1))).astype(np.float32)
        arrays[f"m{k}.head.bias"] = np.zeros(1, np.float32)
    return arrays


def build_bundle(root, seed=0):
    rng = np.random.default_rng(seed)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    n = len(STEPS) * PER_STEP
    tx_id = np.arange(1000, 1000 + n, dtype=np.int64)
    time_step = np.repeat(np.array(STEPS, dtype=np.int16), PER_STEP)
    label = np.tile(np.array(LABELS, dtype=np.int8), len(STEPS))
    raw = rng.normal(size=(n, len(RAW))).astype(np.float32)
    src, dst = [], []
    for s in range(len(STEPS)):
        base = s * PER_STEP
        for k in range(1, PER_STEP):
            src.append(base + int(rng.integers(0, k)))
            dst.append(base + k)
        src.append(base + 3)
        dst.append(base + 8)
    src = np.array(src, dtype=np.int32)
    dst = np.array(dst, dtype=np.int32)
    attr_index = [RAW.index(a) for a in ATTRS]
    eng = np.zeros((n, len(ENG)), dtype=np.float32)
    local = {}
    for s in range(len(STEPS)):
        nodes = np.arange(s * PER_STEP, (s + 1) * PER_STEP)
        mask = (src >= nodes[0]) & (src <= nodes[-1])
        local[s] = (nodes, src[mask] - nodes[0], dst[mask] - nodes[0])
        eng[nodes] = graph_features(len(nodes), local[s][1], local[s][2], raw[nodes].astype(float)[:, attr_index]).astype(np.float32)
    x_all = gnn_matrix(raw.astype(float), eng[:, :len(STRUCTURAL)].astype(float))
    arrays = sage_arrays(rng, x_all.shape[1], 8, 2, 2)
    std = x_all.std(axis=0)
    std[std == 0] = 1.0
    arrays["mean"] = x_all.mean(axis=0).astype(np.float32)
    arrays["std"] = std.astype(np.float32)
    np.savez(root / "graphsage.npz", **arrays)
    sage = GraphSAGE(root / "graphsage.npz")
    logits = np.zeros(n)
    for nodes, ls, ld in local.values():
        logits[nodes] = sage.logits(x_all[nodes], normalise(symmetric(len(nodes), ls, ld)))
    logits = logits.astype(np.float32)
    stack = np.zeros((n, len(STACKING)), dtype=np.float32)
    for nodes, ls, ld in local.values():
        stack[nodes] = stacking_features(len(nodes), ls, ld, logits[nodes]).astype(np.float32)
    x_fit = rng.normal(size=(400, len(HYBRID)))
    target = (x_fit[:, 0] + x_fit[:, len(RAW) + 7] + x_fit[:, len(RAW) + len(ENG)] > 0).astype(int)
    params = {"objective": "binary", "verbose": -1, "num_leaves": 7, "min_data_in_leaf": 5, "seed": 0, "deterministic": True, "num_threads": 1}
    booster = lgb.train(params, lgb.Dataset(x_fit, target, feature_name=HYBRID), num_boost_round=15)
    booster.save_model(str(root / "lgbm.txt"))
    hybrid = booster.predict(np.hstack([raw, eng, stack]).astype(float)).astype(np.float32)
    np.savez(
        root / "store.npz", tx_id=tx_id, time_step=time_step, label=label, raw=raw, eng=eng, stack=stack,
        score_hybrid=hybrid, score_graphsage=(1.0 / (1.0 + np.exp(-logits.astype(float)))).astype(np.float32),
        score_raw_eng=rng.uniform(size=n).astype(np.float32), score_rf=rng.uniform(size=n).astype(np.float32),
        src=src, dst=dst,
    )
    meta = {
        "raw": RAW, "attrs": ATTRS, "structural": STRUCTURAL, "engineered": ENG, "stacking": STACKING,
        "hybrid": HYBRID, "gnn_inputs": RAW + STRUCTURAL, "headline": "hybrid",
        "thresholds": {m: 0.5 for m in MODELS}, "steps": list(STEPS), "form_features": RAW[:8],
        "medians": {name: float(np.median(raw[:, i])) for i, name in enumerate(RAW)},
        "model_names": MODEL_NAMES,
        "counts": {"test_nodes": n, "test_labelled": int((label >= 0).sum()), "test_illicit": int((label == 1).sum())},
    }
    (root / "feature_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    charts = {
        "model_names": MODEL_NAMES, "headline": "hybrid",
        "roc": {m: {"x": [0.0, 0.1, 1.0], "y": [0.0, 0.8, 1.0], "area": 0.9} for m in MODELS},
        "pr": {m: {"x": [0.0, 0.5, 1.0], "y": [1.0, 0.7, 0.1], "area": 0.6} for m in MODELS},
        "per_step": {"steps": list(STEPS), "illicit": [2, 2, 2], "f1": {m: [0.8, 0.6, 0.1] for m in MODELS}},
        "fn": {"rf": {"own": 30, "matched": None}, "raw_eng": {"own": 25, "matched": 26}, "hybrid": {"own": 22, "matched": 24}, "graphsage": {"own": 35, "matched": 40}},
        "ablation": [{"config": c, "label": text, "oof_pr_auc": 0.5, "oof_f1": 0.5, "test_pr_auc": 0.4, "test_f1": 0.4, "test_recall": 0.4} for c, text in CONFIGS],
        "contrast": [{"model": m, "metric": k, "temporal": 0.7, "random": 0.9} for m in MODELS for k in ("auc", "pr_auc", "f1", "recall")],
        "benchmarks": [
            {"name": "AUC-ROC", "target": 0.962, "achieved": 0.93, "low": 0.92, "high": 0.94, "met": False},
            {"name": "Illicit F1", "target": 0.84, "achieved": 0.85, "low": 0.83, "high": 0.87, "met": True},
            {"name": "Fewer false negatives than Random Forest, own threshold", "target": 0.22, "achieved": 0.25, "low": 0.1, "high": 0.35, "met": True},
        ],
        "counts": {"train_labelled": 100, "train_illicit": 10, "test_nodes": n, "test_labelled": 21, "test_illicit": 6},
    }
    (root / "charts.json").write_text(json.dumps(charts), encoding="utf-8")
    (root / "metrics.json").write_text(json.dumps({"headline": "hybrid"}), encoding="utf-8")
    (root / "versions.json").write_text(json.dumps({"python": "3.13"}), encoding="utf-8")
    return root


def strict_loads(text):
    def reject(token):
        raise ValueError(f"non-standard JSON token {token}")
    return json.loads(text, parse_constant=reject)


def longest_number_list(obj):
    if isinstance(obj, dict):
        return max((longest_number_list(v) for v in obj.values()), default=0)
    if isinstance(obj, list):
        own = len(obj) if obj and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in obj) else 0
        return max([own] + [longest_number_list(v) for v in obj])
    return 0


def valid_body(scorer):
    nodes = scorer.steps[36][0]
    return {
        "time_step": 36,
        "features": [float(scorer.meta["medians"][name]) for name in scorer.raw_names],
        "inputs": [int(scorer.tx_id[nodes[0]])],
        "outputs": [int(scorer.tx_id[nodes[5]])],
    }
