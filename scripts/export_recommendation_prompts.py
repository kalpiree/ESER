#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ESER_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = ESER_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from recommendation_exposure.data import prepare_dataset
from recommendation_exposure.prompting import export_prompt_batch
from recommendation_exposure.utils import load_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Export prompt batches for recommendation scoring.")
    parser.add_argument("--config", required=True, help="Path to the experiment JSON config.")
    parser.add_argument("--output", required=True, help="JSONL output path.")
    parser.add_argument(
        "--split",
        choices=["calibration", "test", "both"],
        default="both",
        help="Which split to export.",
    )
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=None,
        help="Stop exporting after this many prompt rows.",
    )
    parser.add_argument(
        "--max-history-items",
        type=int,
        default=None,
        help="Keep only the most recent N history items in each prompt.",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = load_json(config_path)
    config["_config_path"] = str(config_path)
    dataset = prepare_dataset(config)

    if args.split == "calibration":
        cases = dataset.calibration_cases
    elif args.split == "test":
        cases = dataset.test_cases
    else:
        cases = dataset.calibration_cases + dataset.test_cases

    total_prompts = sum(len(case.candidates) for case in cases)
    print(
        "[rec:prompts] starting_export "
        f"split={args.split} cases={len(cases)} prompts={total_prompts} output={Path(args.output).resolve()}",
        flush=True,
    )
    export_prompt_batch(
        dataset,
        cases,
        Path(args.output).resolve(),
        max_prompts=args.max_prompts,
        max_history_items=args.max_history_items,
    )
    print(Path(args.output).resolve())


if __name__ == "__main__":
    main()
