#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DATASETS = [
    ("movielens1m", "MovieLens", "rq_mainfair_movielens1m_qwen25_3b_50neg"),
    ("amazon_books", "Amazon", "rq_mainfair_amazon_books_qwen25_3b_50neg"),
    ("bookcrossing", "BookCrossing", "rq_mainfair_bookcrossing_qwen25_3b_50neg"),
    ("lastfm1k", "LastFM", "rq_mainfair_lastfm1k_qwen25_3b_50neg"),
]

METHOD_ORDER = [
    "topk",
    "topk_matched_eser",
    "global_thr",
    "group_thr",
    "coverage_set",
    "user_fair",
    "item_fair",
    "two_sided_fair",
    "ltt",
    "eser_s",
    "eser",
]

METHOD_LABELS = {
    "topk": "Top-k",
    "topk_matched_eser": "Top-k-M",
    "global_thr": "Global-Thr.",
    "group_thr": "Group-Thr.",
    "coverage_set": "Coverage-Set",
    "user_fair": "User-Fair",
    "item_fair": "Item-Fair",
    "two_sided_fair": "Two-Sided-Fair",
    "ltt": "LTT",
    "eser_s": "ESER w/o SC",
    "eser": "ESER",
}


def _load_summary(outputs_root: Path, experiment: str) -> dict[str, Any]:
    path = outputs_root / experiment / "summary.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value: Any, digits: int = 3, dagger: bool = False) -> str:
    if value is None:
        text = "--"
    elif isinstance(value, (int, float)):
        text = f"{float(value):.{digits}f}"
    else:
        text = str(value)
    return f"{text}$^\\dagger$" if dagger else text


def _group_pair(groups: dict[str, Any]) -> tuple[str, dict[str, Any], str, dict[str, Any]]:
    if not groups:
        return "G1", {}, "G2", {}
    names = sorted(groups)
    if len(names) == 1:
        return names[0], groups[names[0]], "G2", {}
    return names[0], groups[names[0]], names[1], groups[names[1]]


def _constraints_satisfied(test: dict[str, Any]) -> bool:
    constraints = test.get("constraints_satisfied", {})
    return bool(constraints) and all(bool(value) for value in constraints.values())


def _rows(outputs_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset_key, dataset_label, experiment in DATASETS:
        summary = _load_summary(outputs_root, experiment)
        methods = summary.get("methods", {})
        for method_name in METHOD_ORDER:
            if method_name not in methods:
                continue
            payload = methods[method_name]
            test = payload.get("test", {})
            overall = test.get("overall", {})
            group1_name, group1, group2_name, group2 = _group_pair(test.get("groups", {}))
            sat = _constraints_satisfied(test)
            violation = not sat
            rows.append(
                {
                    "dataset_key": dataset_key,
                    "dataset": dataset_label,
                    "experiment": experiment,
                    "method": method_name,
                    "method_label": METHOD_LABELS.get(method_name, method_name),
                    "group1": group1_name,
                    "group2": group2_name,
                    "ber1": group1.get("ber"),
                    "ber2": group2.get("ber"),
                    "mor1": group1.get("mor"),
                    "mor2": group2.get("mor"),
                    "size": overall.get("avg_size"),
                    "hr": overall.get("hr"),
                    "ndcg": overall.get("ndcg"),
                    "u_diff": test.get("utility_disparity"),
                    "b_diff": test.get("burden_disparity"),
                    "b_diff_raw": test.get("burden_disparity_raw"),
                    "item_exposure_disparity": test.get("item_exposure_disparity"),
                    "delta_e": test.get("delta_e"),
                    "sat": "yes" if sat else "no",
                    "violation": "yes" if violation else "no",
                }
            )
    return rows


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "dataset",
        "method_label",
        "method",
        "group1",
        "group2",
        "ber1",
        "ber2",
        "mor1",
        "mor2",
        "size",
        "hr",
        "ndcg",
        "u_diff",
        "b_diff",
        "b_diff_raw",
        "item_exposure_disparity",
        "delta_e",
        "sat",
        "violation",
        "experiment",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fieldnames} for row in rows)


def _write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    headers = ["Dataset", "Method", "BER1", "BER2", "MOR1", "MOR2", "Size", "HR", "NDCG", "U-Diff", "B-Diff", "DeltaE", "Sat."]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        dagger = row["violation"] == "yes"
        lines.append(
            "| "
            + " | ".join(
                [
                    row["dataset"],
                    row["method_label"],
                    _fmt(row["ber1"], dagger=dagger),
                    _fmt(row["ber2"], dagger=dagger),
                    _fmt(row["mor1"], dagger=dagger),
                    _fmt(row["mor2"], dagger=dagger),
                    _fmt(row["size"], digits=2),
                    _fmt(row["hr"]),
                    _fmt(row["ndcg"]),
                    _fmt(row["u_diff"]),
                    _fmt(row["b_diff"]),
                    _fmt(row["delta_e"], dagger=dagger),
                    row["sat"],
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_latex_rows(rows: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "% Columns: Dataset & Method & BER1 & BER2 & MOR1 & MOR2 & Size & HR & NDCG & U-Diff & B-Diff & DeltaE & Sat.",
    ]
    last_dataset = None
    for row in rows:
        dagger = row["violation"] == "yes"
        dataset = row["dataset"] if row["dataset"] != last_dataset else ""
        last_dataset = row["dataset"]
        cells = [
            dataset,
            row["method_label"],
            _fmt(row["ber1"], dagger=dagger),
            _fmt(row["ber2"], dagger=dagger),
            _fmt(row["mor1"], dagger=dagger),
            _fmt(row["mor2"], dagger=dagger),
            _fmt(row["size"], digits=2),
            _fmt(row["hr"]),
            _fmt(row["ndcg"]),
            _fmt(row["u_diff"]),
            _fmt(row["b_diff"]),
            _fmt(row["delta_e"], dagger=dagger),
            row["sat"],
        ]
        lines.append(" & ".join(cells) + r" \\")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect fairness comparison metrics into CSV, Markdown, and LaTeX tables.")
    parser.add_argument("--outputs-root", default="experiments/recommendation_eser/outputs", help="Directory containing experiment outputs.")
    parser.add_argument("--output-dir", default="experiments/recommendation_eser/outputs/tables", help="Directory for collected tables.")
    args = parser.parse_args()

    outputs_root = Path(args.outputs_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _rows(outputs_root)
    _write_csv(rows, output_dir / "main_fairness_table_rows.csv")
    _write_markdown(rows, output_dir / "main_fairness_table.md")
    _write_latex_rows(rows, output_dir / "main_fairness_table_latex_rows.tex")
    print(
        json.dumps(
            {
                "rows": len(rows),
                "csv": str(output_dir / "main_fairness_table_rows.csv"),
                "markdown": str(output_dir / "main_fairness_table.md"),
                "latex_rows": str(output_dir / "main_fairness_table_latex_rows.tex"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
