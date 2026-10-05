#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


SCORERS = [
    ("popularity", "Conventional", "Popularity"),
    ("bpr_mf", "MF", "BPR-MF"),
    ("neumf", "Neural CF", "NeuMF"),
    ("sasrec_lite", "Sequential", "SASRec-lite"),
    ("qwen25_0_5b", "LLM", "Qwen2.5-0.5B"),
    ("qwen25_1_5b", "LLM", "Qwen2.5-1.5B"),
    ("qwen25_3b", "LLM", "Qwen2.5-3B"),
]

METHOD_LABELS = {
    "topk": "Top-k",
    "global_thr": "Global-Thr.",
    "group_thr": "Group-Thr.",
    "coverage_set": "Coverage-Set",
    "user_fair": "User-Fair",
    "far": "FAR",
    "fairrec_style": "FairRec-style",
    "ltt": "LTT",
}


def _fmt(value: Any) -> Any:
    if isinstance(value, (int, float)):
        return f"{float(value):.6f}"
    return value


def _summary_path(outputs_root: Path, scorer_name: str) -> Path:
    return outputs_root / f"rq_motivation_movielens1m_{scorer_name}_50neg" / "summary.json"


def collect(outputs_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    aggregate_rows: list[dict[str, Any]] = []
    user_rows: list[dict[str, Any]] = []
    for scorer_name, family, display_name in SCORERS:
        path = _summary_path(outputs_root, scorer_name)
        if not path.is_file():
            aggregate_rows.append(
                {
                    "dataset": "MovieLens",
                    "scorer_family": family,
                    "scorer": display_name,
                    "scorer_key": scorer_name,
                    "method": "MISSING",
                    "method_label": "MISSING",
                    "complete": "no",
                }
            )
            continue
        summary = json.loads(path.read_text(encoding="utf-8"))
        for method_name, payload in summary.get("methods", {}).items():
            test = payload.get("test", {})
            overall = test.get("overall", {})
            row = {
                "dataset": "MovieLens",
                "scorer_family": family,
                "scorer": display_name,
                "scorer_key": scorer_name,
                "method": method_name,
                "method_label": METHOD_LABELS.get(method_name, method_name),
                "complete": "yes",
                "ber": overall.get("ber"),
                "mor": overall.get("mor"),
                "size": overall.get("avg_size"),
                "exposure_rate": overall.get("avg_exposure_rate"),
                "hr": overall.get("hr"),
                "ndcg": overall.get("ndcg"),
                "u_diff": test.get("utility_disparity"),
                "b_diff": test.get("burden_disparity"),
                "b_diff_raw": test.get("burden_disparity_raw"),
                "item_diff": test.get("item_exposure_disparity"),
                "delta_e": test.get("delta_e"),
                "sat": "yes" if all(test.get("constraints_satisfied", {}).values()) else "no",
                "summary_path": str(path),
            }
            aggregate_rows.append({key: _fmt(value) for key, value in row.items()})
            for user_row in test.get("user_rows", []):
                user_rows.append(
                    {
                        "dataset": "MovieLens",
                        "scorer_family": family,
                        "scorer": display_name,
                        "scorer_key": scorer_name,
                        "method": method_name,
                        "method_label": METHOD_LABELS.get(method_name, method_name),
                        "user_id": user_row.get("user_id"),
                        "group": user_row.get("group"),
                        "bad_exposed": user_row.get("bad_exposed"),
                        "exposed": user_row.get("exposed"),
                        "positive": user_row.get("positive"),
                        "bad_candidate_count": user_row.get("bad_candidate_count"),
                        "missed_positive": user_row.get("missed_positive"),
                        "hit_rate": user_row.get("hit_rate"),
                        "ndcg": user_row.get("ndcg"),
                        "size": user_row.get("avg_size"),
                        "candidate_count": user_row.get("candidate_count"),
                        "exposure_rate": user_row.get("exposure_rate"),
                    }
                )
    return aggregate_rows, user_rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    headers = ["Family", "Scorer", "Policy", "HR", "NDCG", "BER", "MOR", "Size", "U-Diff", "B-Diff", "Item-Diff", "Sat."]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        if row.get("complete") != "yes":
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["scorer_family"]),
                    str(row["scorer"]),
                    str(row["method_label"]),
                    str(row["hr"]),
                    str(row["ndcg"]),
                    str(row["ber"]),
                    str(row["mor"]),
                    str(row["size"]),
                    str(row["u_diff"]),
                    str(row["b_diff"]),
                    str(row["item_diff"]),
                    str(row["sat"]),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect MovieLens motivation analysis outputs.")
    parser.add_argument("--outputs-root", default="experiments/recommendation_eser/outputs")
    parser.add_argument("--output-dir", default="experiments/recommendation_eser/outputs/motivation_analysis_tables")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregate_rows, user_rows = collect(Path(args.outputs_root))
    _write_csv(output_dir / "motivation_analysis_rows.csv", aggregate_rows)
    _write_csv(output_dir / "motivation_analysis_user_rows.csv", user_rows)
    _write_markdown(output_dir / "motivation_analysis_rows.md", aggregate_rows)
    print(
        json.dumps(
            {
                "aggregate_rows": len(aggregate_rows),
                "user_rows": len(user_rows),
                "aggregate_csv": str(output_dir / "motivation_analysis_rows.csv"),
                "user_csv": str(output_dir / "motivation_analysis_user_rows.csv"),
                "markdown": str(output_dir / "motivation_analysis_rows.md"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
