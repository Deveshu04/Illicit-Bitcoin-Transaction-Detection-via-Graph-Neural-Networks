import json
import shutil

import pytest

import server
from helpers import strict_loads, valid_body


def test_health(client):
    out = client.get("/api/health")
    assert out.status_code == 200
    assert out.get_json() == {"status": "ok", "headline": "hybrid", "transactions": 30, "steps": [35, 36, 37]}


def test_transaction_route(client):
    out = client.get("/api/transactions/1004")
    assert out.status_code == 200
    body = out.get_json()
    assert body["tx_id"] == 1004 and body["label"] == "illicit" and len(body["drivers"]) == 8


def test_unknown_transaction_is_404_json(client):
    out = client.get("/api/transactions/5")
    assert out.status_code == 404 and "not in the test period store" in out.get_json()["error"]


@pytest.mark.parametrize("method, path, code", [
    ("get", "/api/transactions/abc", 404),
    ("get", "/api/transactions/-3", 404),
    ("get", "/api/nothing", 404),
    ("get", "/api/score", 405),
    ("post", "/api/health", 405),
])
def test_api_errors_are_json(client, method, path, code):
    out = getattr(client, method)(path)
    assert out.status_code == code and out.is_json and "error" in out.get_json()


@pytest.mark.parametrize("label", ["illicit", "licit", "unknown"])
def test_sample_route(client, label):
    tx = client.get(f"/api/transactions/sample?label={label}").get_json()["tx_id"]
    assert client.get(f"/api/transactions/{tx}").get_json()["label"] == label


def test_sample_bad_label(client):
    out = client.get("/api/transactions/sample?label=fraud")
    assert out.status_code == 400 and "label must be" in out.get_json()["error"]


def test_score_route(client, scorer):
    out = client.post("/api/score", json=valid_body(scorer))
    assert out.status_code == 200
    body = out.get_json()
    assert 0 <= body["risk"] <= 1 and body["neighbourhood"]["nodes"][0]["id"] == "new"


def test_responses_are_strict_json(client, scorer):
    for tx in scorer.tx_id.tolist():
        strict_loads(client.get(f"/api/transactions/{tx}").get_data(as_text=True))
    strict_loads(client.post("/api/score", json=valid_body(scorer)).get_data(as_text=True))


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity", "1e999"])
def test_score_bad_bodies(client, scorer, token):
    body = valid_body(scorer)
    body["features"][0] = 0.123456789
    text = json.dumps(body).replace("0.123456789", token)
    out = client.post("/api/score", data=text, content_type="application/json")
    assert out.status_code == 400 and "finite" in strict_loads(out.get_data(as_text=True))["error"]


@pytest.mark.parametrize("data, content_type", [
    ("not json", "application/json"),
    ('{"time_step": 36,', "application/json"),
    ("[1, 2]", "application/json"),
    ('{"time_step": 36}', "text/plain"),
])
def test_score_unreadable_bodies(client, data, content_type):
    out = client.post("/api/score", data=data, content_type=content_type)
    assert out.status_code == 400 and "JSON object" in strict_loads(out.get_data(as_text=True))["error"]


def test_score_field_errors_are_400(client, scorer):
    body = valid_body(scorer)
    body["inputs"] = [999999]
    out = client.post("/api/score", json=body)
    assert out.status_code == 400 and "not in the test period store" in out.get_json()["error"]


def test_create_app_downloads_when_repo_set(tmp_path, artifact_dir, monkeypatch):
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        shutil.copytree(artifact_dir, kwargs["local_dir"])

    monkeypatch.setattr(server, "hub_download", fake_download)
    monkeypatch.setenv("ARTIFACT_REPO", "owner/repo")
    monkeypatch.setenv("HF_TOKEN", "token-value")
    app = server.create_app(tmp_path / "artifacts")
    assert calls[0]["repo_id"] == "owner/repo" and calls[0]["token"] == "token-value"
    assert app.test_client().get("/api/health").status_code == 200


def test_create_app_fails_fast_without_artifacts(tmp_path, monkeypatch):
    monkeypatch.delenv("ARTIFACT_REPO", raising=False)
    with pytest.raises(FileNotFoundError, match="ARTIFACT_REPO"):
        server.create_app(tmp_path / "empty")


def test_metric_filter():
    assert server.metric(0.96234) == "0.962"
    assert server.metric(0.2213, share=True) == "22.1%"
    assert server.metric(None) == "n/a"
