import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_runner():
    spec = importlib.util.spec_from_file_location("run_notebook", ROOT / "scripts" / "run_notebook.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_skips_when_source_is_absent(tmp_path, monkeypatch):
    runner = load_runner()
    folder = tmp_path / "notebooks" / "01-eda"
    folder.mkdir(parents=True)
    (folder / "kernel-metadata.json").write_text(json.dumps({"id": "someone/elliptic-gnn-01-eda"}), encoding="utf-8")
    (folder / "01-eda.ipynb").write_text("{}", encoding="utf-8")
    calls = []
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **k: calls.append(a))
    runner.build("01-eda")
    assert calls == []


def test_build_runs_jupytext_when_source_exists(tmp_path, monkeypatch):
    runner = load_runner()
    folder = tmp_path / "notebooks" / "01-eda"
    folder.mkdir(parents=True)
    (folder / "kernel-metadata.json").write_text(json.dumps({"id": "someone/elliptic-gnn-01-eda"}), encoding="utf-8")
    (folder / "01-eda.py").write_text("", encoding="utf-8")
    calls = []
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **k: calls.append(a))
    runner.build("01-eda")
    assert len(calls) == 1 and "jupytext" in calls[0][0]


def test_smoke_push_restores_notebook_without_source(tmp_path, monkeypatch):
    runner = load_runner()
    folder = tmp_path / "notebooks" / "01-eda"
    folder.mkdir(parents=True)
    (folder / "kernel-metadata.json").write_text(json.dumps({"id": "someone/elliptic-gnn-01-eda"}), encoding="utf-8")
    notebook = folder / "01-eda.ipynb"
    original = b'{"cells": [{"source": ["SMOKE = False"]}]}\n'
    notebook.write_bytes(original)
    pushed = []

    def fake_run(command, **kwargs):
        pushed.append(notebook.read_text(encoding="utf-8"))
        return type("Result", (), {"stdout": "Kernel version 1 successfully pushed.", "stderr": "", "returncode": 0})()

    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    runner.push("01-eda", smoke=True)
    assert "SMOKE = True" in pushed[0]
    assert notebook.read_bytes() == original
