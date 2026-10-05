from pathlib import Path

REQUIRED = ("store.npz", "graphsage.npz", "lgbm.txt", "feature_meta.json", "charts.json")


def missing(artifact_dir):
    return [name for name in REQUIRED if not (Path(artifact_dir) / name).exists()]


def ensure_artifacts(artifact_dir, repo_id, token, download):
    artifact_dir = Path(artifact_dir)
    if not missing(artifact_dir):
        return artifact_dir
    if not repo_id:
        raise FileNotFoundError(f"missing {', '.join(missing(artifact_dir))} in {artifact_dir}; set ARTIFACT_REPO to download them")
    download(repo_id=repo_id, repo_type="model", local_dir=str(artifact_dir), token=token)
    absent = missing(artifact_dir)
    if absent:
        raise FileNotFoundError(f"download from {repo_id} did not provide {', '.join(absent)}")
    return artifact_dir


def hub_download(**kwargs):
    from huggingface_hub import snapshot_download

    return snapshot_download(**kwargs)
