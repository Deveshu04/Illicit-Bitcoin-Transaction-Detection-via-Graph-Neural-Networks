from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
REPO = "deveshu/elliptic-gnn-artifacts"


def main():
    api = HfApi()
    api.create_repo(REPO, repo_type="model", private=True, exist_ok=True)
    api.upload_folder(folder_path=str(ROOT / "app" / "artifacts"), repo_id=REPO, repo_type="model", commit_message="Upload serving artifacts")
    info = api.model_info(REPO)
    print(f"private={info.private}, files={sorted(s.rfilename for s in info.siblings)}")


if __name__ == "__main__":
    main()
