import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from helpers import valid_body
from scoring import MODELS, Scorer

APP = Path(__file__).resolve().parents[1] / "app"
REAL = APP / "artifacts"

pytestmark = pytest.mark.skipif(not (REAL / "store.npz").exists(), reason="real artifacts not present")


@pytest.fixture(scope="module")
def real():
    return Scorer(REAL)


def sample_ids(real, count=200):
    rng = np.random.default_rng(42)
    return [int(t) for t in rng.choice(real.tx_id, size=count, replace=False)]


def test_real_recompute_matches_store(real):
    close = 0
    worst_logit = 0.0
    for tx in sample_ids(real):
        g = real.row[tx]
        out = real.recompute(tx)
        assert np.allclose(out["eng"], real.eng[g], rtol=1e-5, atol=1e-6, equal_nan=True), tx
        assert np.allclose(out["stack"], real.stack[g], rtol=1e-5, atol=1e-6, equal_nan=True), tx
        worst_logit = max(worst_logit, abs(out["logit"] - float(real.stack[g, 0])))
        close += abs(out["hybrid"] - float(real.scores["hybrid"][g])) < 1e-6
    print(f"worst GraphSAGE logit difference {worst_logit:.2e}, hybrid scores within 1e-6: {close} of 200")
    assert worst_logit < 1e-4
    assert close >= 199


def test_real_counts_match_notebook(real):
    metrics = json.loads((REAL / "metrics.json").read_text(encoding="utf-8"))
    labelled = real.label >= 0
    truth = real.label[labelled] == 1
    for model in MODELS:
        flagged = real.scores[model][labelled].astype(float) >= metrics["thresholds"][model]
        for key, value in (("tp", flagged & truth), ("fp", flagged & ~truth), ("fn", ~flagged & truth)):
            assert abs(int(value.sum()) - metrics["test"][model][key]) <= 1, (model, key)


def test_real_payloads(real):
    tx = int(real.tx_id[real.by_label["illicit"][0]])
    out = real.transaction(tx)
    assert len(out["drivers"]) == 8 and out["label"] == "illicit"
    assert len(out["neighbourhood"]["nodes"]) <= 60


def test_memory_fits_render():
    code = "import psutil, server; server.create_app(); print(psutil.Process().memory_info().rss)"
    out = subprocess.run([sys.executable, "-c", code], cwd=APP, capture_output=True, text=True, check=True)
    rss = int(out.stdout.strip().splitlines()[-1])
    print(f"resident memory after loading the app: {rss / 1e6:.0f} MB")
    assert rss < 350e6


def test_latency(real):
    body = valid_body(real)
    timings = []
    for _ in range(30):
        started = time.perf_counter()
        real.score_new(body)
        timings.append((time.perf_counter() - started) * 1000)
    lookups = []
    for tx in sample_ids(real, 30):
        started = time.perf_counter()
        real.transaction(tx)
        lookups.append((time.perf_counter() - started) * 1000)
    print(f"POST /api/score p50 {np.median(timings):.0f} ms, p95 {np.percentile(timings, 95):.0f} ms; lookup p50 {np.median(lookups):.0f} ms, p95 {np.percentile(lookups, 95):.0f} ms")
    assert np.percentile(timings, 95) < 2000
