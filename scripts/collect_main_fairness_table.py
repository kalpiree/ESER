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
    "item_fair": "FAR",
    "two_sided_fair": "FairRec-style",
    "ltt": "LTT",
    "eser_s": "ESER-S",
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


def _write_references(path: Path) -> None:
    text = """# Baseline Naming And Citation Notes

## Recommended framing

These rows should be described as **LLM-score post-processing baselines**: every method consumes the same Qwen-generated relevance scores and changes only the selection/calibration policy. This is cleaner than claiming exact reproductions of recent LLM-RS training methods.

## Modern LLM-RS fairness citations to foreground

- User-side LLM-RS fairness: `User-Fair` can be motivated by UP5 / counterfactually fair prompting for fairness-aware LLM recommendation: Hua et al., "UP5: Unbiased Foundation Model for Fairness-aware Recommendation" (2023), arXiv:2305.12090.
- Consumer fairness evaluation in LLM recommenders: cite Deldjoo and di Noia, "CFaiRLLM: Consumer Fairness Evaluation in Large-Language Model Recommender System" (2024), arXiv:2403.05668.
- Item/provider-side LLM-RS fairness: `FAR` / item-exposure reranking can be motivated by Jiang et al., "Item-side Fairness of Large Language Model-based Recommendation System" / IFairLRS (2024), arXiv:2402.15215.
- LLM reranking with fairness/diversity criteria: cite Gao et al., "LLM4Rerank: LLM-based Auto-Reranking Framework for Recommendations" (2024), arXiv:2406.12433.
- Recent LLM-enhanced fairness training reference: cite Zhang et al., "BiFair: A Fairness-aware Training Framework for LLM-enhanced Recommender Systems via Bi-level Optimization" (2025), arXiv:2507.04294.
- Recent no-retraining LLM-RS debiasing reference: cite Li et al., "Towards Fair Large Language Model-based Recommender Systems without Costly Retraining" / FUDLR (2026), arXiv:2601.17492.

## Classical objective-lineage citations, only if space allows

- `User-Fair`: Yao and Huang, "Beyond Parity: Fairness Objectives for Collaborative Filtering" (2017), arXiv:1705.08804.
- `FAR`: Liu and Burke, "Personalizing Fairness-aware Re-ranking" (2018), arXiv:1809.02921; optionally Biega et al., "Equity of Attention" (2018).
- `FairRec-style`: Patro et al., "FairRec: Two-Sided Fairness for Personalized Recommendations in Two-Sided Platforms" (2020), arXiv:2002.10764, and/or Biswas et al., "Towards Fair Recommendation in Two-Sided Platforms" (2021), arXiv:2201.01180.
- `LTT`: Angelopoulos et al., "Learn then Test: Calibrating Predictive Algorithms to Achieve Risk Control" (2021), arXiv:2110.01052; optionally pair with "Conformal Risk Control" (2022).

## Suggested manuscript sentence

"Because several recent LLM-RS fairness methods require model retraining or prompt-level intervention, we include comparable post-processing baselines that operate on the same LLM score matrix: user-side fairness calibration (UP5/CFaiRLLM-motivated), item-exposure fairness reranking (IFairLRS/FAR-motivated), two-sided fairness reranking (FairRec-style), and LTT risk-control calibration."

Implementation note: these are lightweight post-processing/reranking baselines inspired by the above lines of work, not exact reproductions of every paper algorithm. Use "motivated by", "style", or "post-processing analogue" unless we implement the full external method.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect main fairness baseline outputs into paper-ready rows.")
    parser.add_argument("--outputs-root", default="results/eser_tables", help="Directory containing pulled output folders.")
    parser.add_argument("--output-dir", default="results/eser_tables", help="Directory for table artifacts.")
    args = parser.parse_args()

    outputs_root = Path(args.outputs_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _rows(outputs_root)
    _write_csv(rows, output_dir / "main_fairness_table_rows.csv")
    _write_markdown(rows, output_dir / "main_fairness_table.md")
    _write_latex_rows(rows, output_dir / "main_fairness_table_latex_rows.tex")
    _write_references(output_dir / "baseline_citation_notes.md")
    print(
        json.dumps(
            {
                "rows": len(rows),
                "csv": str(output_dir / "main_fairness_table_rows.csv"),
                "markdown": str(output_dir / "main_fairness_table.md"),
                "latex_rows": str(output_dir / "main_fairness_table_latex_rows.tex"),
                "citation_notes": str(output_dir / "baseline_citation_notes.md"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
