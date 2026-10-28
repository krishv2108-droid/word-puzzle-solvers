"""Stage everything the Space needs into build/space/ (and optionally upload it).

    python deploy/build_space.py                      # stage only
    python deploy/build_space.py --upload USER/NAME   # stage, create the Space, upload

The staged folder is self-contained: `docker build build/space` works locally too.
Requires the trained model: run `python -m src.connections.train` first.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "build" / "space"
MODEL = ROOT / "data" / "cache" / "connections_model.joblib"


def stage() -> Path:
    if not MODEL.exists():
        raise SystemExit(f"{MODEL} missing; run `python -m src.connections.train` first")
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    shutil.copytree(ROOT / "src", OUT / "src", ignore=ignore)
    shutil.copytree(ROOT / "web", OUT / "web", ignore=ignore)
    (OUT / "data" / "wordle").mkdir(parents=True)
    for name in ("answers.txt", "allowed.txt"):
        shutil.copy(ROOT / "data" / "wordle" / name, OUT / "data" / "wordle" / name)
    (OUT / "data" / "cache").mkdir(parents=True)
    shutil.copy(MODEL, OUT / "data" / "cache" / MODEL.name)
    for name, dest in (("Dockerfile", "Dockerfile"), ("requirements-app.txt", "requirements-app.txt"),
                       (".dockerignore", ".dockerignore"), ("space_README.md", "README.md")):
        shutil.copy(ROOT / "deploy" / name, OUT / dest)
    return OUT


def upload(repo_id: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, repo_type="space", space_sdk="docker", exist_ok=True)
    api.upload_folder(folder_path=str(OUT), repo_id=repo_id, repo_type="space",
                      commit_message="Deploy Word Puzzle Solvers")
    print(f"uploaded: https://huggingface.co/spaces/{repo_id}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--upload", metavar="USER/NAME")
    args = ap.parse_args()
    print(f"staged at {stage()}")
    if args.upload:
        upload(args.upload)
