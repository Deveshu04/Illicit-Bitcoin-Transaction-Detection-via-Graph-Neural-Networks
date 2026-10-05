import pytest

from helpers import build_bundle


@pytest.fixture(scope="session")
def artifact_dir(tmp_path_factory):
    return build_bundle(tmp_path_factory.mktemp("bundle") / "artifacts")


@pytest.fixture(scope="session")
def scorer(artifact_dir):
    from scoring import Scorer

    return Scorer(artifact_dir)


@pytest.fixture(scope="session")
def client(artifact_dir):
    from server import create_app

    return create_app(artifact_dir).test_client()
