#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


EXPERIMENTS = [
    "movielens1m_qwen25_0_5b_precomputed_50neg",
    "movielens1m_qwen25_1_5b_precomputed_50neg",
    "movielens1m_qwen25_3b_precomputed_50neg",
    "lastfm1k_qwen25_3b_precomputed_50neg",
    "lastfm1k_qwen25_0_5b_precomputed_50neg",
    "lastfm1k_qwen25_1_5b_precomputed_50neg",
    "bookcrossing_qwen25_3b_precomputed_50neg",
    "bookcrossing_qwen25_0_5b_precomputed_50neg",
    "bookcrossing_qwen25_1_5b_precomputed_50neg",
    "amazon_books_qwen25_0_5b_precomputed_50neg",
    "amazon_books_qwen25_1_5b_precomputed_50neg",
    "amazon_books_qwen25_3b_precomputed_50neg",
]


def _format_float(value: Any) -> Any:
    if isinstance(value, (int, float)):
        return f"{float(value):.6f}"
    return value


def _metric(payload: dict[str, Any], key: str) -> Any:
    return payload.get("overall", {}).get(key)


def _dataset_from_experiment(experiment: str) -> str:
    if experiment.startswith("amazon_books"):
        return "amazon_books"
    if experiment.startswith("bookcrossing"):
        return "bookcrossing"
    if experiment.startswith("lastfm1k"):
        return "lastfm1k"
    if experiment.startswith("movielens1m"):
        return "movielens1m"
    return ""


def _model_from_experiment(experiment: str) -> str:
    if "qwen25_0_5b" in experiment:
        return "qwen25_0_5b"
    if "qwen25_1_5b" in experiment:
        return "qwen25_1_5b"
    if "qwen25_3b" in experiment:
        return "qwen25_3b"
    return ""


def main() -> None:
    outputs_root = Path("experiments/recommendation_eser/outputs")
    output_dir = outputs_root / "rq1_tables"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    for experiment in EXPERIMENTS:
        summary_path = outputs_root / experiment / "summary.json"
        csv_path = outputs_root / experiment / "test_metrics.csv"
        complete = summary_path.is_file() and csv_path.is_file()
        manifest.append(
            {
                "experiment": experiment,
                "dataset": _dataset_from_experiment(experiment),
                "model": _model_from_experiment(experiment),
                "summary_path": str(summary_path),
                "test_metrics_path": str(csv_path),
                "complete": complete,
            }
        )
        if not complete:
            rows.append(
                {
                    "experiment": experiment,
                    "dataset": _dataset_from_experiment(experiment),
                    "model": _model_from_experiment(experiment),
                    "status": "missing",
                }
            )
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        dataset_meta = summary.get("dataset", {})
        scorer = summary.get("scorer", {}).get("kind", "")
        for method_name, payload in summary.get("methods", {}).items():
            test = payload.get("test", {})
            row = {
                "experiment": experiment,
                "dataset": _dataset_from_experiment(experiment),
                "model": _model_from_experiment(experiment),
                "status": "complete",
                "scorer": scorer,
                "method": method_name,
                "num_users": dataset_meta.get("num_users"),
                "num_items": dataset_meta.get("num_items"),
                "num_test_cases": dataset_meta.get("num_test_cases"),
                "avg_size": _metric(test, "avg_size"),
                "hr": _metric(test, "hr"),
                "ndcg": _metric(test, "ndcg"),
                "ber": _metric(test, "ber"),
                "mor": _metric(test, "mor"),
                "utility_disparity": test.get("utility_disparity"),
                "burden_disparity": test.get("burden_disparity"),
                "delta_e": test.get("delta_e"),
                "summary_path": str(summary_path),
            }
            rows.append({key: _format_float(value) for key, value in row.items()})

    fieldnames = [
        "experiment",
        "dataset",
        "model",
        "status",
        "scorer",
        "method",
        "num_users",
        "num_items",
        "num_test_cases",
        "avg_size",
        "hr",
        "ndcg",
        "ber",
        "mor",
        "utility_disparity",
        "burden_disparity",
        "delta_e",
        "summary_path",
    ]
    table_path = output_dir / "multillm_recommendation_rows.csv"
    with table_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    manifest_path = output_dir / "multillm_recommendation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"table_path": str(table_path), "manifest_path": str(manifest_path), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
