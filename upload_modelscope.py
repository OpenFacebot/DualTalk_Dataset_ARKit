#!/usr/bin/env python3
"""Upload DualTalk ARKit artifacts to ModelScope dataset repo.

Repo: yiwenhao/Dualtalk_Dataset_ARKit

Uploads:
    - ARKit_npy.zip        (converted ARKit blendshape data, single large file)
    - metadata/            (train/test/ood *.jsonl)
    - script/              (conversion & metadata generation scripts)
    - DualTalk_Dataset/    (raw dataset zips: train/test/ood)

Usage:
    export MODELSCOPE_SDK_TOKEN=<your token>
    python upload_modelscope.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from modelscope.hub.api import HubApi

# ------------------------- config -------------------------
TOKEN = os.environ.get("MODELSCOPE_SDK_TOKEN", "")
REPO_ID = "yiwenhao/Dualtalk_Dataset_ARKit"
REPO_TYPE = "dataset"
REVISION = "master"

DATASET_ROOT = Path(__file__).resolve().parent

# (local path, path in repo); folders keep their own name as top dir
UPLOAD_ITEMS: list[tuple[Path, str]] = [
    (DATASET_ROOT / "ARKit_npy.zip", "ARKit_npy.zip"),
    (DATASET_ROOT / "metadata", "metadata"),
    (DATASET_ROOT / "script", "script"),
    (DATASET_ROOT / "DualTalk_Dataset", "DualTalk_Dataset"),
]
# ----------------------------------------------------------


def fmt_size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def main() -> None:
    if not TOKEN:
        sys.exit("[error] set MODELSCOPE_SDK_TOKEN environment variable first")
    api = HubApi()
    api.login(TOKEN)
    print(f"[login] ok, target repo: {REPO_ID} ({REPO_TYPE})")

    for local_path, path_in_repo in UPLOAD_ITEMS:
        if not local_path.exists():
            print(f"[skip] {local_path} does not exist")
            continue

        size = (
            local_path.stat().st_size
            if local_path.is_file()
            else sum(f.stat().st_size for f in local_path.rglob("*") if f.is_file())
        )
        print(f"\n[upload] {local_path} -> {path_in_repo} ({fmt_size(size)}) ...")

        t0 = time.time()
        try:
            if local_path.is_file():
                api.upload_file(
                    repo_id=REPO_ID,
                    path_or_fileobj=str(local_path),
                    path_in_repo=path_in_repo,
                    repo_type=REPO_TYPE,
                    revision=REVISION,
                )
            else:
                api.upload_folder(
                    repo_id=REPO_ID,
                    folder_path=str(local_path),
                    path_in_repo=path_in_repo,
                    repo_type=REPO_TYPE,
                    revision=REVISION,
                )
        except Exception as e:  # keep going with remaining items
            print(f"[fail] {path_in_repo}: {e}", file=sys.stderr)
            continue
        print(f"[done] {path_in_repo} in {time.time() - t0:.1f}s")

    print("\nAll upload tasks finished.")


if __name__ == "__main__":
    main()
