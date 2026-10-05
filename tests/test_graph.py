import networkx as nx
import numpy as np
import pytest

from graph import AGGREGATES, STACKING, STRUCTURAL, engineered_names, graph_features, stacking_features

SRC = [0, 0, 1, 3, 2, 4, 0, 4]
DST = [1, 2, 2, 2, 4, 5, 1, 4]
N = 6
SEL = np.column_stack([np.arange(1.0, 7.0), np.arange(10.0, 70.0, 10.0)])
NAN = np.nan


@pytest.fixture(scope="module")
def table():
    return dict(zip(engineered_names(["a", "b"]), graph_features(N, SRC, DST, SEL).T))


def test_names_and_shape():
    names = engineered_names(["a", "b"])
    assert names[:7] == STRUCTURAL
    assert names[7:12] == [f"a_{agg}" for agg in AGGREGATES]
    assert len(names) == 7 + 5 * 2
    assert graph_features(N, SRC, DST, SEL).shape == (N, len(names))
    assert STACKING == ["gnn_logit", "gnn_in_mean", "gnn_in_max", "gnn_out_mean", "gnn_out_max"]


def test_degrees_ignore_duplicates_and_self_loops(table):
    assert table["in_degree"].tolist() == [0, 1, 3, 0, 1, 1]
    assert table["out_degree"].tolist() == [2, 1, 1, 1, 1, 0]


def test_undirected_structure(table):
    assert table["core_number"].tolist() == [2, 2, 2, 1, 1, 1]
    assert np.allclose(table["clustering"], [1.0, 1.0, 1 / 6, 0.0, 0.0, 0.0])
    assert np.allclose(table["avg_neighbor_degree"], [3.0, 3.0, 1.75, 4.0, 2.5, 2.0])
    assert table["two_hop_size"].tolist() == [4, 4, 5, 4, 5, 2]


def test_pagerank_matches_networkx(table):
    graph = nx.DiGraph()
    graph.add_nodes_from(range(N))
    graph.add_edges_from((s, d) for s, d in zip(SRC, DST) if s != d)
    expected = nx.pagerank(graph, alpha=0.85)
    assert np.allclose(table["pagerank"], [expected[i] for i in range(N)])


def test_directed_aggregates(table):
    assert np.allclose(table["a_in_mean"], [NAN, 1, 7 / 3, NAN, 3, 5], equal_nan=True)
    assert np.allclose(table["a_in_max"], [NAN, 1, 4, NAN, 3, 5], equal_nan=True)
    assert np.allclose(table["a_out_mean"], [2.5, 3, 5, 3, 6, NAN], equal_nan=True)
    assert np.allclose(table["a_out_max"], [3, 3, 5, 3, 6, NAN], equal_nan=True)
    assert np.allclose(table["b_out_max"], [30, 30, 50, 30, 60, NAN], equal_nan=True)


def test_two_hop_mean(table):
    assert np.allclose(table["a_hop2_mean"], [3.5, 3.25, 3.6, 2.75, 3.2, 4.0])
    assert np.allclose(table["b_hop2_mean"], [35, 32.5, 36, 27.5, 32, 40])


def test_isolated_node_gets_nan_aggregates():
    out = graph_features(3, [0], [1], np.ones((3, 1)))
    names = engineered_names(["a"])
    row = dict(zip(names, out[2]))
    assert row["in_degree"] == 0 and row["two_hop_size"] == 0
    assert np.isnan(row["a_in_mean"]) and np.isnan(row["a_hop2_mean"])


def test_stacking_features():
    logits = np.array([0.5, -1.0, 2.0, 0.0, 1.0, 3.0])
    out = stacking_features(N, SRC, DST, logits)
    assert np.allclose(out[:, 0], logits)
    assert np.allclose(out[:, 1], [NAN, 0.5, -1 / 6, NAN, 2.0, 1.0], equal_nan=True)
    assert np.allclose(out[:, 2], [NAN, 0.5, 0.5, NAN, 2.0, 1.0], equal_nan=True)
    assert np.allclose(out[:, 3], [0.5, 2.0, 1.0, 2.0, 3.0, NAN], equal_nan=True)
    assert np.allclose(out[:, 4], [2.0, 2.0, 1.0, 2.0, 3.0, NAN], equal_nan=True)
