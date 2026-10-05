import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from helpers import RAW, longest_number_list, strict_loads, valid_body
from scoring import NEIGHBOURHOOD_CAP, TOP_DRIVERS, NotFoundError, Scorer, ValidationError, number


def test_store_loads(scorer):
    assert len(scorer.tx_id) == 30
    assert sorted(scorer.steps) == [35, 36, 37]
    assert scorer.headline == "hybrid"
    assert len(scorer.names) == 165 + 177 + 5


def test_recompute_matches_store(scorer):
    for tx in scorer.tx_id.tolist():
        g = scorer.row[tx]
        out = scorer.recompute(tx)
        assert np.allclose(out["eng"], scorer.eng[g], rtol=1e-5, atol=1e-6, equal_nan=True)
        assert np.allclose(out["stack"], scorer.stack[g], rtol=1e-5, atol=1e-6, equal_nan=True)
        assert abs(out["logit"] - float(scorer.stack[g, 0])) < 1e-4
        assert abs(out["hybrid"] - float(scorer.scores["hybrid"][g])) < 1e-6


def test_transaction_payload(scorer):
    out = scorer.transaction(1004)
    assert out["tx_id"] == 1004 and out["time_step"] == 35 and out["label"] == "illicit"
    assert set(out["scores"]) == {"hybrid", "graphsage", "raw_eng", "rf"}
    assert out["risk"] == pytest.approx(out["scores"]["hybrid"])
    assert out["flagged"] == (out["risk"] >= out["threshold"])
    sizes = [abs(d["contribution"]) for d in out["drivers"]]
    assert len(sizes) == TOP_DRIVERS and sizes == sorted(sizes, reverse=True)
    nodes = out["neighbourhood"]["nodes"]
    assert nodes[0] == {"id": 1004, "hop": 0, "risk": out["risk"], "label": "illicit", "flagged": out["flagged"]}
    assert all(n["hop"] <= 2 for n in nodes) and len(nodes) <= NEIGHBOURHOOD_CAP
    ids = {n["id"] for n in nodes}
    assert all(a in ids and b in ids for a, b in out["neighbourhood"]["edges"])


def test_raw_drivers_are_verbatim(scorer):
    out = scorer.transaction(1007)
    raw = [d for d in out["drivers"] if d["kind"] == "raw"]
    assert raw
    for d in raw:
        assert d["value"] == float(scorer.raw[scorer.row[1007], scorer.raw_index[d["feature"]]])


def test_missing_aggregates_become_null(scorer):
    out = scorer.transaction(1000)
    driver = next(d for d in out["drivers"] if d["feature"] == "local_1_in_mean")
    assert driver["value"] is None and driver["kind"] == "graph"


def test_payloads_are_strict_json(scorer):
    for tx in scorer.tx_id.tolist():
        strict_loads(json.dumps(scorer.transaction(tx), allow_nan=False))
    strict_loads(json.dumps(scorer.score_new(valid_body(scorer)), allow_nan=False))


def test_licence_guard(scorer):
    for tx in scorer.tx_id.tolist():
        out = scorer.transaction(tx)
        assert sum(d["kind"] == "raw" for d in out["drivers"]) <= 8
        assert longest_number_list(out) <= 8
    assert longest_number_list(scorer.score_new(valid_body(scorer))) <= 8


def test_score_new(scorer):
    body = valid_body(scorer)
    out = scorer.score_new(body)
    assert 0.0 <= out["risk"] <= 1.0 and out["time_step"] == 36
    assert set(out["scores"]) == {"hybrid", "graphsage"}
    nodes = out["neighbourhood"]["nodes"]
    assert nodes[0]["id"] == "new" and nodes[0]["hop"] == 0
    assert {body["inputs"][0], body["outputs"][0]} <= {n["id"] for n in nodes}
    assert [body["inputs"][0], "new"] in out["neighbourhood"]["edges"]
    assert ["new", body["outputs"][0]] in out["neighbourhood"]["edges"]
    assert len(out["drivers"]) == TOP_DRIVERS and out["latency_ms"] >= 0


def test_score_new_accepts_named_features(scorer):
    body = valid_body(scorer)
    named = dict(body, features=dict(zip(scorer.raw_names, body["features"])))
    assert scorer.score_new(named)["risk"] == pytest.approx(scorer.score_new(body)["risk"])


def test_score_new_reacts_to_features(scorer):
    body = valid_body(scorer)
    other = valid_body(scorer)
    other["features"][0] = body["features"][0] + 5.0
    assert scorer.score_new(other)["risk"] != scorer.score_new(body)["risk"]


def put(index, value):
    return lambda b: b["features"].__setitem__(index, value)


@pytest.mark.parametrize("change, message", [
    (lambda b: b.update(time_step=True), "time_step must be an integer"),
    (lambda b: b.pop("time_step"), "time_step must be an integer"),
    (lambda b: b.update(time_step=34), "time_step must be one of"),
    (lambda b: b.update(features=b["features"][:-1]), "features must hold 165"),
    (put(0, True), "features must be numbers"),
    (put(0, "1.0"), "features must be numbers"),
    (put(0, None), "features must be numbers"),
    (put(0, float("nan")), "finite"),
    (put(0, float("inf")), "finite"),
    (put(0, 1e300), "32-bit"),
    (lambda b: b.update(features={"local_1": 1.0}), "missing 164 features"),
    (lambda b: b.update(features={**{n: 0.0 for n in RAW}, "bogus": 1.0}), "unknown feature names: bogus"),
    (lambda b: b.update(features="abc"), "features must be a list or an object"),
    (lambda b: b.update(inputs=[], outputs=[]), "at least one"),
    (lambda b: b.update(inputs=list(range(51))), "at most 50"),
    (lambda b: b.update(outputs=list(b["inputs"])), "distinct"),
    (lambda b: b.update(inputs=[999999]), "not in the test period store"),
    (lambda b: b.update(inputs=[1000]), "is in time step 35"),
    (lambda b: b.update(inputs="1010"), "inputs must be a list"),
    (lambda b: b.update(inputs=[True]), "inputs entries must be an integer"),
    (lambda b: b.update(inputs=[1010.0]), "inputs entries must be an integer"),
    (lambda b: b.update(extra=1), "unknown fields: extra"),
])
def test_score_new_rejects(scorer, change, message):
    body = valid_body(scorer)
    change(body)
    with pytest.raises(ValidationError, match=message):
        scorer.score_new(body)


@pytest.mark.parametrize("body", [None, [1, 2], "text", 3])
def test_score_new_rejects_non_objects(scorer, body):
    with pytest.raises(ValidationError, match="JSON object"):
        scorer.score_new(body)


def test_concurrent_scoring_matches_sequential(scorer):
    bodies = []
    for k in range(8):
        body = valid_body(scorer)
        body["features"][0] = k / 4
        bodies.append(body)
    sequential = [scorer.score_new(b)["risk"] for b in bodies]
    with ThreadPoolExecutor(max_workers=4) as pool:
        concurrent = list(pool.map(lambda b: scorer.score_new(b)["risk"], bodies))
    assert concurrent == sequential
    txs = scorer.tx_id.tolist()
    expected = [scorer.transaction(t)["drivers"] for t in txs]
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert list(pool.map(lambda t: scorer.transaction(t)["drivers"], txs)) == expected


@pytest.mark.parametrize("label", ["illicit", "licit", "unknown"])
def test_sample(scorer, label):
    tx = scorer.sample(label, np.random.default_rng(0))
    assert scorer.transaction(tx)["label"] == label


def test_sample_rejects_bad_label(scorer):
    with pytest.raises(ValidationError, match="label must be"):
        scorer.sample("fraud", np.random.default_rng(0))


def test_sample_from_empty_pool(artifact_dir):
    fresh = Scorer(artifact_dir)
    fresh.by_label["illicit"] = np.array([], dtype=np.int64)
    with pytest.raises(NotFoundError, match="no illicit"):
        fresh.sample("illicit", np.random.default_rng(0))


def test_unknown_transaction(scorer):
    with pytest.raises(NotFoundError, match="not in the test period store"):
        scorer.transaction(5)


def test_number():
    assert number(float("nan")) is None and number(float("inf")) is None
    assert number(np.float32(1.5)) == 1.5
