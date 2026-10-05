import shutil

import pytest

from artifacts import REQUIRED, ensure_artifacts, missing


def test_bundle_has_every_required_file(artifact_dir):
    assert missing(artifact_dir) == []
    assert "store.npz" in REQUIRED and "lgbm.txt" in REQUIRED


def test_existing_artifacts_skip_download(artifact_dir):
    calls = []
    assert ensure_artifacts(artifact_dir, "owner/repo", None, lambda **kw: calls.append(kw)) == artifact_dir
    assert calls == []


def test_missing_artifacts_are_downloaded(tmp_path, artifact_dir):
    target = tmp_path / "artifacts"
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        shutil.copytree(artifact_dir, kwargs["local_dir"])

    assert ensure_artifacts(target, "owner/repo", "token-value", fake_download) == target
    assert calls == [{"repo_id": "owner/repo", "repo_type": "model", "local_dir": str(target), "token": "token-value"}]


def test_missing_artifacts_without_repo_fail_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match="ARTIFACT_REPO"):
        ensure_artifacts(tmp_path / "artifacts", None, None, None)


def test_incomplete_download_fails_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match="did not provide"):
        ensure_artifacts(tmp_path / "artifacts", "owner/repo", None, lambda **kw: None)
