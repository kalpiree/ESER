#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_NAMES = [
    "movielens1m_qwen25_0_5b_precomputed_50neg",
    "movielens1m_qwen25_1_5b_precomputed_50neg",
    "movielens1m_qwen25_3b_precomputed_50neg",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect small-LLM robustness rows.")
    parser.add_argument(
        "--outputs-root",
        default="experiments/recommendation_eser/outputs",
        help="Recommendation experiment outputs root.",
    )
    parser.add_argument(
        "--output-csv",
        default="experiments/recommendation_eser/outputs/rq1_tables/small_llm_robustness_rows.csv",
    )
    return parser.parse_args()


def _format_float(value: Any) -> Any:
    if isinstance(value, (int, float)):
        return f"{float(value):.6f}"
    return value


def _metric(payload: dict[str, Any], key: str) -> Any:
    return payload.get("overall", {}).get(key)


def main() -> None:
    args = parse_args()
    outputs_root = Path(args.outputs_root)
    rows: list[dict[str, object]] = []
    for name in DEFAULT_NAMES:
        summary_path = outputs_root / name / "summary.json"
        if not summary_path.exists():
            rows.append({"experiment": name, "status": "missing"})
            continue
        with summary_path.open(encoding="utf-8") as handle:
            summary = json.load(handle)
        dataset = summary.get("dataset", {})
        scorer = summary.get("scorer", {}).get("kind", "")
        for method_name, method_payload in summary.get("methods", {}).items():
            test = method_payload.get("test", {})
            row = {
                "experiment": name,
                "status": "complete",
                "scorer": scorer,
                "method": method_name,
                "avg_size": _metric(test, "avg_size"),
                "hr": _metric(test, "hr"),
                "ndcg": _metric(test, "ndcg"),
                "ber": _metric(test, "ber"),
                "mor": _metric(test, "mor"),
                "utility_disparity": test.get("utility_disparity"),
                "burden_disparity": test.get("burden_disparity"),
                "delta_e": test.get("delta_e"),
                "num_users": dataset.get("num_users"),
                "num_items": dataset.get("num_items"),
                "num_test_cases": dataset.get("num_test_cases"),
            }
            rows.append({key: _format_float(value) for key, value in row.items()})

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "experiment",
        "status",
        "scorer",
        "method",
        "avg_size",
        "hr",
        "ndcg",
        "ber",
        "mor",
        "utility_disparity",
        "burden_disparity",
        "delta_e",
        "num_users",
        "num_items",
        "num_test_cases",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[small-llm] wrote {output_path}")


if __name__ == "__main__":
    main()
