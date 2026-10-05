#!/usr/bin/env python3
"""Run a recommendation experiment using an explicit precomputed score JSONL."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
import sys

ESER_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = ESER_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from recommendation_exposure import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--scores-jsonl", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--methods")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    scores_path = Path(args.scores_jsonl).resolve()
    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    config["scorer"] = {"kind": "precomputed_jsonl", "scores_path": str(scores_path)}

    method_names = None
    if args.methods:
        method_names = {name.strip() for name in args.methods.split(",") if name.strip()}

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        prefix=f"{config_path.stem}-scores-",
        dir=str(config_path.parent),
        encoding="utf-8",
        delete=False,
    ) as handle:
        json.dump(config, handle, indent=2)
        runtime_config = handle.name

    result = run_experiment(runtime_config, output_dir=args.output_dir, method_names=method_names)
    print(json.dumps({key: result[key] for key in ["output_dir", "summary_path", "csv_path"]}, indent=2))


if __name__ == "__main__":
    main()
