#!/usr/bin/env python3
"""Export listwise recommendation prompts for black-box LLM ranking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ESER_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = ESER_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from recommendation_exposure.data import prepare_dataset
from recommendation_exposure.types import ItemRecord, PreparedDataset, UserEvalCase
from recommendation_exposure.utils import load_json


PROMPT_TEMPLATE = """Task: Rank candidate items for the user's next recommendation.

Use only the interaction history and candidate item information below. Do not use demographic or group information. Return deterministic JSON only.

User interaction history in chronological order:
{history_block}

Candidate items:
{candidate_block}

Return exactly this JSON shape and no extra text:
{{"ranking":["C001","C002"]}}

The ranking list must contain every candidate key exactly once, ordered from most likely to least likely next preference."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to the experiment JSON config.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument(
        "--split",
        choices=["calibration", "test", "both"],
        default="both",
        help="Which split to export.",
    )
    parser.add_argument("--max-cases", type=int, default=None, help="Stop after this many user-round cases.")
    parser.add_argument(
        "--max-history-items",
        type=int,
        default=50,
        help="Keep only the most recent N history items in each prompt.",
    )
    args = parser.parse_args()
    if args.max_cases is not None and args.max_cases <= 0:
        raise ValueError("--max-cases must be positive when provided.")
    if args.max_history_items is not None and args.max_history_items <= 0:
        raise ValueError("--max-history-items must be positive when provided.")
    return args


def _item_summary(item: ItemRecord) -> str:
    parts = [item.title or item.item_id, item.category, item.description, item.tags]
    return " | ".join(part for part in parts if part)


def _history_block(dataset: PreparedDataset, case: UserEvalCase, max_history_items: int | None) -> str:
    item_ids = case.history_item_ids
    if max_history_items is not None:
        item_ids = item_ids[-max_history_items:]
    lines = []
    for index, item_id in enumerate(item_ids, start=1):
        item = dataset.items.get(item_id)
        if item is None:
            continue
        lines.append(f"{index}. {_item_summary(item)}")
    return "\n".join(lines) if lines else "No previous interactions."


def _candidate_key(index: int) -> str:
    return f"C{index:03d}"


def _prompt_for_case(dataset: PreparedDataset, case: UserEvalCase, max_history_items: int | None) -> tuple[str, list[dict]]:
    candidate_rows: list[dict] = []
    candidate_lines: list[str] = []
    for index, candidate in enumerate(case.candidates, start=1):
        key = _candidate_key(index)
        item = dataset.items[candidate.item_id]
        summary = _item_summary(item)
        candidate_rows.append(
            {
                "candidate_key": key,
                "item_id": candidate.item_id,
                "label": candidate.label,
                "summary": summary,
            }
        )
        candidate_lines.append(f"{key}. item_id={candidate.item_id} | {summary}")

    prompt = PROMPT_TEMPLATE.format(
        history_block=_history_block(dataset, case, max_history_items),
        candidate_block="\n".join(candidate_lines),
    )
    return prompt, candidate_rows


def _select_cases(dataset: PreparedDataset, split: str) -> tuple[UserEvalCase, ...]:
    if split == "calibration":
        return dataset.calibration_cases
    if split == "test":
        return dataset.test_cases
    return dataset.calibration_cases + dataset.test_cases


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_json(config_path)
    config["_config_path"] = str(config_path)
    dataset = prepare_dataset(config)
    cases = _select_cases(dataset, args.split)

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = len(cases) if args.max_cases is None else min(len(cases), args.max_cases)
    print(
        "[rec:rank-prompts] starting_export "
        f"split={args.split} cases={total} output={output_path}",
        flush=True,
    )

    written = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for case in cases:
            if args.max_cases is not None and written >= args.max_cases:
                break
            prompt, candidates = _prompt_for_case(dataset, case, args.max_history_items)
            payload = {
                "case_id": f"{case.split}:{case.user_id}",
                "split": case.split,
                "user_id": case.user_id,
                "group": case.group,
                "prompt": prompt,
                "candidates": candidates,
            }
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
            written += 1
            if written % 1000 == 0 or written == total:
                print(f"[rec:rank-prompts] wrote={written}/{total}", flush=True)

    print(output_path)


if __name__ == "__main__":
    main()
