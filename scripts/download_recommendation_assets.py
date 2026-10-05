#!/usr/bin/env python3
"""Download recommendation datasets or model snapshots defined in the asset manifest."""

from __future__ import annotations

import argparse
import json
import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path
import sys

ESER_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = ESER_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from recommendation_exposure.utils import load_json


MANIFEST_PATH = ESER_ROOT / "experiments" / "recommendation_eser" / "assets_manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List known datasets and models.")
    list_parser.add_argument("--kind", choices=["datasets", "models", "all"], default="all")

    dataset_parser = subparsers.add_parser("download-dataset", help="Download one or more datasets.")
    dataset_parser.add_argument("dataset_ids", nargs="+")

    model_parser = subparsers.add_parser("download-model", help="Download one or more model snapshots.")
    model_parser.add_argument("model_ids", nargs="+")

    return parser.parse_args()


def _manifest() -> dict:
    return load_json(MANIFEST_PATH)


def _path_from_manifest(relative_path: str) -> Path:
    return (ESER_ROOT / relative_path).resolve()


def _download_url(url: str, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 Codex Recommendation Bootstrap"})
    with urllib.request.urlopen(request, timeout=1200) as response, target_path.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def _download_with_fallbacks(primary_url: str, fallback_urls: list[str], target_path: Path) -> str:
    attempted: list[str] = []
    for url in [primary_url, *fallback_urls]:
        attempted.append(url)
        try:
            _download_url(url, target_path)
            return url
        except Exception:
            if target_path.exists():
                target_path.unlink()
            continue
    raise RuntimeError(f"All download sources failed for {target_path.name}: {attempted}")


def _extract_archive(archive_path: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    archive_name = archive_path.name.lower()
    if archive_name.endswith(".zip"):
      with zipfile.ZipFile(archive_path) as handle:
            handle.extractall(target_dir)
      return
    if archive_name.endswith(".tar.gz") or archive_name.endswith(".tgz"):
        with tarfile.open(archive_path, "r:gz") as handle:
            handle.extractall(target_dir)
        return
    raise ValueError(f"Unsupported archive format: {archive_path}")


def download_dataset(dataset_id: str, dataset_info: dict) -> dict:
    raw_dir = _path_from_manifest(dataset_info["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    result = {"dataset_id": dataset_id, "raw_dir": str(raw_dir)}

    if dataset_info["kind"] == "url_archive":
        archive_path = raw_dir / dataset_info["archive_name"]
        if not archive_path.exists():
            used_url = _download_with_fallbacks(dataset_info["url"], dataset_info.get("fallback_urls", []), archive_path)
        else:
            used_url = dataset_info["url"]
        extract_dir = raw_dir / dataset_info["extract_dir_name"]
        if not extract_dir.exists():
            _extract_archive(archive_path, raw_dir)
        result["archive_path"] = str(archive_path)
        result["extract_dir"] = str(extract_dir)
        result["download_url"] = used_url
        return result

    if dataset_info["kind"] == "url_files":
        downloaded_files = []
        for file_info in dataset_info["files"]:
            target_path = raw_dir / file_info["name"]
            if not target_path.exists():
                used_url = _download_with_fallbacks(file_info["url"], file_info.get("fallback_urls", []), target_path)
            else:
                used_url = file_info["url"]
            downloaded_files.append(str(target_path))
            result.setdefault("download_urls", []).append(used_url)
        result["files"] = downloaded_files
        return result

    raise ValueError(f"Unsupported dataset kind: {dataset_info['kind']}")


def download_model(model_id: str, model_info: dict) -> dict:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "Model downloads require the optional 'huggingface_hub' package. "
            "Install it with: pip install huggingface_hub"
        ) from exc

    local_dir = _path_from_manifest(model_info["local_dir"])
    local_dir.mkdir(parents=True, exist_ok=True)
    downloaded_path = snapshot_download(
        repo_id=model_info["repo_id"],
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
    )
    return {
        "model_id": model_id,
        "repo_id": model_info["repo_id"],
        "local_dir": str(local_dir),
        "downloaded_path": downloaded_path,
    }


def main() -> None:
    args = parse_args()
    manifest = _manifest()

    if args.command == "list":
        payload = {}
        if args.kind in {"datasets", "all"}:
            payload["datasets"] = manifest["datasets"]
        if args.kind in {"models", "all"}:
            payload["models"] = manifest["models"]
        print(json.dumps(payload, indent=2))
        return

    if args.command == "download-dataset":
        payload = [download_dataset(dataset_id, manifest["datasets"][dataset_id]) for dataset_id in args.dataset_ids]
        print(json.dumps(payload, indent=2))
        return

    if args.command == "download-model":
        payload = [download_model(model_id, manifest["models"][model_id]) for model_id in args.model_ids]
        print(json.dumps(payload, indent=2))
        return


if __name__ == "__main__":
    main()
