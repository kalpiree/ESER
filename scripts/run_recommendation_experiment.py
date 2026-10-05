#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ESER_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = ESER_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from recommendation_exposure import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run recommendation selective-exposure experiments.")
    parser.add_argument("--config", required=True, help="Path to the experiment JSON config.")
    parser.add_argument("--output-dir", help="Optional output directory override.")
    parser.add_argument(
        "--methods",
        help="Optional comma-separated list of method names to run from the config.",
    )
    args = parser.parse_args()

    method_names = None
    if args.methods:
        method_names = {name.strip() for name in args.methods.split(",") if name.strip()}
    result = run_experiment(args.config, output_dir=args.output_dir, method_names=method_names)
    print(json.dumps({key: result[key] for key in ["output_dir", "summary_path", "csv_path"]}, indent=2))


if __name__ == "__main__":
    main()
