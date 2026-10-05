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

from audit_recommendation_results import _build_rows, _evaluate_pass, _print_table


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one recommendation smoke check and immediately audit the resulting table rows."
    )
    parser.add_argument("--config", required=True, help="Path to the experiment JSON config.")
    parser.add_argument("--output-dir", help="Optional output directory override.")
    args = parser.parse_args()

    result = run_experiment(args.config, output_dir=args.output_dir)
    summary_path = Path(result["summary_path"]).resolve()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = _build_rows(summary)
    _print_table(rows)
    passed, issues = _evaluate_pass(rows)
    print()
    print(json.dumps({"summary_path": str(summary_path), "audit_pass": passed, "issues": issues}, indent=2))
    if not passed:
        sys.exit(2)


if __name__ == "__main__":
    main()
