from __future__ import annotations

import json
import os
from pathlib import Path

from .types import PreparedDataset, UserEvalCase


PROMPT_TEMPLATE = """Task: Estimate whether the candidate item matches the user's future preference based on the given interaction history. Do not use any demographic or group information.

User interaction history in chronological order:
{history_block}

Candidate item: {candidate_title}
Candidate information: {candidate_info}

Output exactly one score from this grid and nothing else:
0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00

Interpretation:
- 0.00 = not relevant
- 0.50 = uncertain or moderately relevant
- 1.00 = highly relevant"""


def _item_summary(dataset: PreparedDataset, item_id: str) -> str:
    item = dataset.items[item_id]
    parts = [item.title, item.category, item.description, item.tags]
    return " | ".join(part for part in parts if part)


def _prompt_for_case(
    case: UserEvalCase,
    dataset: PreparedDataset,
    candidate_item_id: str,
    *,
    max_history_items: int | None = None,
) -> str:
    history_lines = []
    history_item_ids = case.history_item_ids
    if max_history_items is not None and max_history_items > 0:
        history_item_ids = history_item_ids[-max_history_items:]
    for index, item_id in enumerate(history_item_ids, start=1):
        history_lines.append(f"{index}. {_item_summary(dataset, item_id)}")
    history_block = "\n".join(history_lines) if history_lines else "No previous interactions."
    candidate = dataset.items[candidate_item_id]
    return PROMPT_TEMPLATE.format(
        history_block=history_block,
        candidate_title=candidate.title or candidate.item_id,
        candidate_info=_item_summary(dataset, candidate_item_id),
    )


def export_prompt_batch(
    dataset: PreparedDataset,
    cases: tuple[UserEvalCase, ...],
    output_path: Path,
    *,
    max_prompts: int | None = None,
    max_history_items: int | None = None,
) -> None:
    if max_history_items is None:
        env_limit = os.environ.get("REC_PROMPT_HISTORY_LIMIT", "50")
        max_history_items = int(env_limit) if env_limit not in ("", "none", "None") else None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_prompts = sum(len(case.candidates) for case in cases)
    if max_prompts is not None:
        total_prompts = min(total_prompts, max_prompts)
    written = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for case in cases:
            for candidate in case.candidates:
                if max_prompts is not None and written >= max_prompts:
                    print(f"[rec:prompts] wrote={written}/{total_prompts}", flush=True)
                    return
                payload = {
                    "split": case.split,
                    "user_id": case.user_id,
                    "group": case.group,
                    "item_id": candidate.item_id,
                    "label": candidate.label,
                    "prompt": _prompt_for_case(
                        case,
                        dataset,
                        candidate.item_id,
                        max_history_items=max_history_items,
                    ),
                }
                handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
                written += 1
                if written % 100_000 == 0 or written == total_prompts:
                    print(f"[rec:prompts] wrote={written}/{total_prompts}", flush=True)
