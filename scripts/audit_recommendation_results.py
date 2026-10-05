#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _format_float(value: float) -> str:
    return f"{value:.4f}"


def _load_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _flags_for_method(method_name: str, payload: dict) -> list[str]:
    flags: list[str] = []
    test = payload["test"]
    overall = test["overall"]
    metadata = payload.get("fit", {}).get("metadata", {}) or {}

    if overall["avg_size"] <= 0.0:
        flags.append("empty_exposure")
    if overall["mor"] >= 0.95:
        flags.append("near_total_miss")
    if overall["ber"] == 0.0 and overall["avg_size"] <= 0.0:
        flags.append("degenerate_zero_set")
    if metadata.get("warning"):
        flags.append("infeasible_fit")

    constraints = test.get("constraints_satisfied", {})
    violated = [name for name, ok in constraints.items() if not ok]
    if violated:
        flags.append("violates:" + ",".join(sorted(violated)))

    if method_name in {"eser", "eser_s"}:
        correction = metadata.get("fairness_correction")
        if isinstance(correction, dict):
            mode = correction.get("mode")
            total = float(correction.get("total", 0.0))
            if total > 0.25:
                flags.append(f"large_fairness_bound:{mode}")

    return flags


def _build_rows(summary: dict) -> list[dict]:
    rows: list[dict] = []
    for method_name, payload in summary["methods"].items():
        overall = payload["test"]["overall"]
        rows.append(
            {
                "method": method_name,
                "kind": payload["kind"],
                "ber": float(overall["ber"]),
                "mor": float(overall["mor"]),
                "avg_size": float(overall["avg_size"]),
                "avg_exposure_rate": float(overall.get("avg_exposure_rate", 0.0)),
                "hr": float(overall["hr"]),
                "ndcg": float(overall["ndcg"]),
                "delta_e": float(payload["test"]["delta_e"]),
                "flags": _flags_for_method(method_name, payload),
            }
        )
    return rows


def _print_table(rows: list[dict]) -> None:
    header = (
        "method".ljust(18)
        + "ber".rjust(10)
        + "mor".rjust(10)
        + "size".rjust(10)
        + "rate".rjust(10)
        + "hr".rjust(10)
        + "ndcg".rjust(10)
        + "delta_e".rjust(10)
        + "  flags"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            row["method"].ljust(18)
            + _format_float(row["ber"]).rjust(10)
            + _format_float(row["mor"]).rjust(10)
            + _format_float(row["avg_size"]).rjust(10)
            + _format_float(row["avg_exposure_rate"]).rjust(10)
            + _format_float(row["hr"]).rjust(10)
            + _format_float(row["ndcg"]).rjust(10)
            + _format_float(row["delta_e"]).rjust(10)
            + "  "
            + (", ".join(row["flags"]) if row["flags"] else "ok")
        )


def _evaluate_pass(rows: list[dict]) -> tuple[bool, list[str]]:
    issues: list[str] = []
    method_map = {row["method"]: row for row in rows}
    eser = method_map.get("eser")
    if eser is None:
        issues.append("missing_eser_row")
        return False, issues

    if "degenerate_zero_set" in eser["flags"]:
        issues.append("eser_degenerate_zero_set")
    if "infeasible_fit" in eser["flags"]:
        issues.append("eser_infeasible_fit")
    if any(flag.startswith("violates:") for flag in eser["flags"]):
        issues.append("eser_constraint_violation")
    if eser["mor"] >= 0.95:
        issues.append("eser_near_total_miss")
    if eser["avg_size"] <= 0.0:
        issues.append("eser_empty_exposure")

    topk = method_map.get("topk")
    if topk is not None and eser["ndcg"] + 1e-9 < 0.25 * topk["ndcg"]:
        issues.append("eser_ndcg_far_below_topk")

    coverage = method_map.get("coverage_set")
    if coverage is not None and eser["avg_size"] > coverage["avg_size"] + 1e-9:
        issues.append("eser_not_more_selective_than_coverage_set")

    return (len(issues) == 0), issues


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit recommendation experiment outputs for degenerate rows.")
    parser.add_argument("--summary", required=True, help="Path to summary.json.")
    args = parser.parse_args()

    summary_path = Path(args.summary).resolve()
    summary = _load_summary(summary_path)
    rows = _build_rows(summary)
    _print_table(rows)
    passed, issues = _evaluate_pass(rows)
    print()
    if passed:
        print("audit=pass")
        return

    print("audit=fail")
    print("issues=" + ",".join(issues))
    sys.exit(2)


if __name__ == "__main__":
    main()
