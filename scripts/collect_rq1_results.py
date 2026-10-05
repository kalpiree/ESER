#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_EXPERIMENTS = [
    "movielens1m_qwen25_3b_precomputed_50neg",
    "lastfm1k_qwen25_3b_precomputed_50neg",
    "bookcrossing_qwen25_3b_precomputed_50neg",
    "movielens1m_token_overlap",
    "amazon_books_token_overlap",
    "bookcrossing_token_overlap",
    "lastfm1k_token_overlap",
]


def _metric(payload: dict[str, Any], key: str) -> Any:
    return payload.get("overall", {}).get(key)


def _format_float(value: Any) -> Any:
    if isinstance(value, (int, float)):
        return f"{float(value):.6f}"
    return value


def _rows_from_summary(summary_path: Path) -> list[dict[str, Any]]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    experiment = summary.get("experiment", summary_path.parent.name)
    scorer = summary.get("scorer", {}).get("kind", "")
    dataset = summary.get("dataset", {})
    rows: list[dict[str, Any]] = []
    for method_name, method_payload in summary.get("methods", {}).items():
        test = method_payload.get("test", {})
        row = {
            "experiment": experiment,
            "scorer": scorer,
            "method": method_name,
            "num_users": dataset.get("num_users"),
            "num_items": dataset.get("num_items"),
            "num_test_cases": dataset.get("num_test_cases"),
            "ber": _metric(test, "ber"),
            "mor": _metric(test, "mor"),
            "avg_size": _metric(test, "avg_size"),
            "hr": _metric(test, "hr"),
            "ndcg": _metric(test, "ndcg"),
            "utility_disparity": test.get("utility_disparity"),
            "burden_disparity": test.get("burden_disparity"),
            "delta_e": test.get("delta_e"),
            "summary_path": str(summary_path),
        }
        rows.append({key: _format_float(value) for key, value in row.items()})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect RQ1 recommendation outputs into table-ready CSV rows.")
    parser.add_argument(
        "--outputs-root",
        default="experiments/recommendation_eser/outputs",
        help="Root containing per-experiment output directories.",
    )
    parser.add_argument(
        "--experiments",
        default=",".join(DEFAULT_EXPERIMENTS),
        help="Comma-separated experiment output directory names to collect.",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/recommendation_eser/outputs/rq1_tables",
        help="Directory for collected RQ1 artifacts.",
    )
    args = parser.parse_args()

    outputs_root = Path(args.outputs_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    experiments = [name.strip() for name in args.experiments.split(",") if name.strip()]

    manifest: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for experiment in experiments:
        summary_path = outputs_root / experiment / "summary.json"
        csv_path = outputs_root / experiment / "test_metrics.csv"
        present = summary_path.is_file() and csv_path.is_file()
        manifest.append(
            {
                "experiment": experiment,
                "summary_path": str(summary_path),
                "test_metrics_path": str(csv_path),
                "complete": present,
            }
        )
        if present:
            rows.extend(_rows_from_summary(summary_path))

    table_path = output_dir / "rq1_table_rows.csv"
    fieldnames = [
        "experiment",
        "scorer",
        "method",
        "num_users",
        "num_items",
        "num_test_cases",
        "ber",
        "mor",
        "avg_size",
        "hr",
        "ndcg",
        "utility_disparity",
        "burden_disparity",
        "delta_e",
        "summary_path",
    ]
    with table_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    manifest_path = output_dir / "rq1_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps({"table_path": str(table_path), "manifest_path": str(manifest_path), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
