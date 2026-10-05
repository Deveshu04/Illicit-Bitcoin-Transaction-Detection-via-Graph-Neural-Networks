import numpy as np

from graph import STRUCTURAL
from sage import LOG, LOG1P, GraphSAGE, ball, gnn_matrix, normalise, symmetric


def save_model(path, models, layers, d_in):
    arrays = {"mean": np.zeros(d_in, np.float32), "std": np.ones(d_in, np.float32), "layers": np.array(layers), "hidden": np.array(models[0]["inp.weight"].shape[0]), "n_models": np.array(len(models))}
    for k, params in enumerate(models):
        for name, value in params.items():
            arrays[f"m{k}.{name}"] = np.asarray(value, dtype=np.float32)
    np.savez(path, **arrays)
    return path


def hand_model(bias):
    eye = np.eye(2)
    return {"inp.weight": eye, "inp.bias": np.zeros(2), "convs.0.lin_l.weight": eye, "convs.0.lin_l.bias": np.zeros(2), "convs.0.lin_r.weight": eye, "head.weight": np.array([[1.0, 0.0, 0.0, 1.0]]), "head.bias": np.array([bias])}


def random_model(rng, d_in, hidden, layers):
    params = {"inp.weight": rng.normal(scale=0.5, size=(hidden, d_in)), "inp.bias": rng.normal(scale=0.1, size=hidden), "head.weight": rng.normal(scale=0.5, size=(1, hidden * (layers + 1))), "head.bias": rng.normal(scale=0.1, size=1)}
    for layer in range(layers):
        params[f"convs.{layer}.lin_l.weight"] = rng.normal(scale=0.5, size=(hidden, hidden))
        params[f"convs.{layer}.lin_l.bias"] = rng.normal(scale=0.1, size=hidden)
        params[f"convs.{layer}.lin_r.weight"] = rng.normal(scale=0.5, size=(hidden, hidden))
    return params


X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.0]])
SRC, DST = [0, 1, 2], [1, 2, 3]


def test_hand_computed_forward(tmp_path):
    sage = GraphSAGE(save_model(tmp_path / "m.npz", [hand_model(0.5)], 1, 2))
    out = sage.logits(X, normalise(symmetric(4, SRC, DST)))
    assert np.allclose(out, [2.5, 2.0, 3.0, 3.5])


def test_models_are_averaged(tmp_path):
    sage = GraphSAGE(save_model(tmp_path / "m.npz", [hand_model(0.5), hand_model(1.5)], 1, 2))
    assert np.allclose(sage.logits(X, normalise(symmetric(4, SRC, DST))), [3.0, 2.5, 3.5, 4.0])


def test_standardisation_is_applied(tmp_path):
    path = save_model(tmp_path / "m.npz", [hand_model(0.5)], 1, 2)
    data = dict(np.load(path))
    data["mean"] = np.array([1.0, 0.0], np.float32)
    data["std"] = np.array([2.0, 1.0], np.float32)
    np.savez(path, **data)
    sage = GraphSAGE(path)
    expected = GraphSAGE(save_model(tmp_path / "e.npz", [hand_model(0.5)], 1, 2)).logits((X - [1.0, 0.0]) / [2.0, 1.0], normalise(symmetric(4, SRC, DST)))
    assert np.allclose(sage.logits(X, normalise(symmetric(4, SRC, DST))), expected)


def test_duplicate_and_reverse_edges_do_not_change_the_mean(tmp_path):
    sage = GraphSAGE(save_model(tmp_path / "m.npz", [hand_model(0.5)], 1, 2))
    adj = normalise(symmetric(4, [0, 1, 2, 1, 0, 3], [1, 2, 3, 0, 1, 3]))
    assert np.allclose(sage.logits(X, adj), [2.5, 2.0, 3.0, 3.5])


def test_ball():
    adj = symmetric(6, [0, 1, 2, 3, 4], [1, 2, 3, 4, 5])
    assert ball(adj, 2, 1).tolist() == [1, 2, 3]
    assert ball(adj, 0, 2).tolist() == [0, 1, 2]
    assert ball(adj, 5, 0).tolist() == [5]


def test_node_logit_matches_full_graph(tmp_path):
    rng = np.random.default_rng(7)
    n, d_in, hidden, layers = 30, 5, 4, 3
    src = rng.integers(0, n, 45)
    dst = rng.integers(0, n, 45)
    features = rng.normal(size=(n, d_in))
    sage = GraphSAGE(save_model(tmp_path / "m.npz", [random_model(rng, d_in, hidden, layers) for _ in range(2)], layers, d_in))
    full = sage.logits(features, normalise(symmetric(n, src, dst)))
    for node in range(n):
        assert np.isclose(sage.node_logit(features, src, dst, node), full[node])


def test_gnn_matrix_transforms():
    raw = np.array([[1.0, -2.0]])
    structural = np.array([[3.0, 0.0, 0.25, 2.0, 0.5, 1.5, 7.0]])
    out = gnn_matrix(raw, structural)
    assert out.shape == (1, 2 + len(STRUCTURAL))
    assert np.allclose(out[0, :2], [1.0, -2.0])
    expected = structural[0].copy()
    for name in LOG1P:
        i = STRUCTURAL.index(name)
        expected[i] = np.log1p(expected[i])
    for name in LOG:
        i = STRUCTURAL.index(name)
        expected[i] = np.log(expected[i])
    assert np.allclose(out[0, 2:], expected)
    assert structural[0, 0] == 3.0
