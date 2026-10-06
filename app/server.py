import json
import os
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template, request
from flask.json.provider import DefaultJSONProvider
from werkzeug.exceptions import HTTPException
from werkzeug.middleware.proxy_fix import ProxyFix

from artifacts import ensure_artifacts, hub_download
from scoring import MAX_LINKS, NotFoundError, Scorer, ValidationError

ROOT = Path(__file__).resolve().parent
MODEL_KEYS = ("model_names", "headline", "roc", "pr", "per_step", "fn", "ablation", "contrast", "benchmarks", "counts")


class StrictJSON(DefaultJSONProvider):
    sort_keys = False

    def dumps(self, obj, **kwargs):
        kwargs.setdefault("allow_nan", False)
        return super().dumps(obj, **kwargs)


class App(Flask):
    json_provider_class = StrictJSON


def metric(value, share=False):
    if value is None:
        return "n/a"
    return f"{value:.1%}" if share else f"{value:.3f}"


def create_app(artifact_dir=None):
    artifact_dir = Path(artifact_dir or os.environ.get("ARTIFACT_DIR", ROOT / "artifacts"))
    ensure_artifacts(artifact_dir, os.environ.get("ARTIFACT_REPO"), os.environ.get("HF_TOKEN"), hub_download)
    scorer = Scorer(artifact_dir)
    charts = json.loads((artifact_dir / "charts.json").read_text(encoding="utf-8"))
    meta = scorer.meta
    app = App(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 64 * 1024
    app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)
    app.add_template_filter(metric, "metric")
    rng = np.random.default_rng()
    common = {"headline": scorer.headline, "model_names": meta["model_names"]}

    @app.get("/")
    def console():
        data = {
            **common,
            "thresholds": scorer.thresholds,
            "steps": meta["steps"],
            "form_features": meta["form_features"],
            "medians": meta["medians"],
            "raw_names": meta["raw"],
            "max_links": MAX_LINKS,
            "counts": meta["counts"],
        }
        return render_template("console.html", active="console", data=data, base_url=request.url_root.rstrip("/"), **common)

    @app.get("/model")
    def model_page():
        data = {key: charts[key] for key in MODEL_KEYS}
        return render_template("model.html", active="model", data=data, charts=charts, **common)

    @app.get("/api/health")
    def health():
        return jsonify(status="ok", headline=scorer.headline, transactions=int(len(scorer.tx_id)), steps=meta["steps"])

    @app.get("/api/transactions/sample")
    def sample():
        return jsonify(tx_id=scorer.sample(request.args.get("label", "illicit"), rng))

    @app.get("/api/transactions/<int:tx_id>")
    def transaction(tx_id):
        return jsonify(scorer.transaction(tx_id))

    @app.post("/api/score")
    def score():
        try:
            body = request.get_json(silent=True)
        except RecursionError:
            body = None
        return jsonify(scorer.score_new(body))

    @app.errorhandler(ValidationError)
    def bad_request(error):
        return jsonify(error=str(error)), 400

    @app.errorhandler(NotFoundError)
    def not_found(error):
        return jsonify(error=str(error)), 404

    @app.errorhandler(HTTPException)
    def http_error(error):
        if request.path == "/api" or request.path.startswith("/api/"):
            return jsonify(error=error.description), error.code
        return error

    return app
