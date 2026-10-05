from __future__ import annotations

import math
from collections import defaultdict

from .types import ScoredCandidate, ScoredUserEvalCase


def _ranked_exposed(candidates: tuple[ScoredCandidate, ...], exposed_ids: set[str]) -> list[ScoredCandidate]:
    return sorted(
        [candidate for candidate in candidates if candidate.item_id in exposed_ids],
        key=lambda candidate: (-candidate.score, candidate.item_id),
    )


def summarize_exposures(
    cases: tuple[ScoredUserEvalCase, ...],
    exposures_by_user: dict[str, list[str]],
    *,
    omega: float = 1.0,
    fairness_metric: str = "hit_rate",
) -> dict:
    grouped = defaultdict(list)
    totals = {
        "exposed": 0,
        "bad_exposed": 0,
        "positive": 0,
        "missed_positive": 0,
    }
    item_group_available = defaultdict(int)
    item_group_exposed = defaultdict(int)

    for case in cases:
        exposed_ids = set(exposures_by_user.get(case.user_id, []))
        exposed_ranked = _ranked_exposed(case.candidates, exposed_ids)
        for candidate in case.candidates:
            item_group_available[candidate.item_group or "unknown"] += 1
        for candidate in exposed_ranked:
            item_group_exposed[candidate.item_group or "unknown"] += 1

        bad_exposed = sum(1 for candidate in exposed_ranked if candidate.label == 0)
        exposed = len(exposed_ranked)
        candidate_count = len(case.candidates)
        positive = sum(candidate.label for candidate in case.candidates)
        bad_candidate_count = candidate_count - positive
        hit = 1 if any(candidate.label == 1 for candidate in exposed_ranked) else 0

        ndcg = 0.0
        if hit:
            for rank, candidate in enumerate(exposed_ranked, start=1):
                if candidate.label == 1:
                    ndcg = 1.0 / math.log2(rank + 1)
                    break

        missed_positive = positive - hit
        row = {
            "user_id": case.user_id,
            "group": case.group,
            "bad_exposed": bad_exposed,
            "exposed": exposed,
            "positive": positive,
            "bad_candidate_count": bad_candidate_count,
            "missed_positive": missed_positive,
            "hit_rate": float(hit),
            "ndcg": ndcg,
            "avg_size": float(exposed),
            "candidate_count": candidate_count,
            "exposure_rate": (exposed / candidate_count) if candidate_count else 0.0,
        }
        grouped[case.group].append(row)
        totals["exposed"] += exposed
        totals["bad_exposed"] += bad_exposed
        totals["positive"] += positive
        totals["missed_positive"] += missed_positive

    def aggregate(rows: list[dict]) -> dict:
        if not rows:
            return {
                "ber": 0.0,
                "mor": 0.0,
                "avg_size": 0.0,
                "avg_candidate_count": 0.0,
                "avg_exposure_rate": 0.0,
                "hr": 0.0,
                "ndcg": 0.0,
                "num_users": 0,
            }
        exposed = sum(row["exposed"] for row in rows)
        positive = sum(row["positive"] for row in rows)
        bad_candidate_total = sum(row["bad_candidate_count"] for row in rows)
        return {
            "ber": sum(row["bad_exposed"] for row in rows) / bad_candidate_total if bad_candidate_total else 0.0,
            "mor": sum(row["missed_positive"] for row in rows) / positive if positive else 0.0,
            "avg_size": sum(row["avg_size"] for row in rows) / len(rows),
            "avg_candidate_count": sum(row["candidate_count"] for row in rows) / len(rows),
            "avg_exposure_rate": sum(row["exposure_rate"] for row in rows) / len(rows),
            "hr": sum(row["hit_rate"] for row in rows) / len(rows),
            "ndcg": sum(row["ndcg"] for row in rows) / len(rows),
            "num_users": len(rows),
        }

    groups = {group: aggregate(rows) for group, rows in grouped.items()}
    all_rows = [row for rows in grouped.values() for row in rows]
    overall = aggregate(all_rows)

    utility_key = "hr" if fairness_metric == "hit_rate" else "ndcg"
    group_names = sorted(groups)
    utility_disparity = 0.0
    burden_disparity = 0.0
    burden_disparity_raw = 0.0
    if len(group_names) >= 2:
        utility_values = [groups[group_name][utility_key] for group_name in group_names]
        burden_values = [groups[group_name]["avg_exposure_rate"] for group_name in group_names]
        burden_values_raw = [groups[group_name]["avg_size"] for group_name in group_names]
        utility_disparity = max(utility_values) - min(utility_values)
        burden_disparity = max(burden_values) - min(burden_values)
        burden_disparity_raw = max(burden_values_raw) - min(burden_values_raw)

    item_groups = {}
    for item_group, available in sorted(item_group_available.items()):
        exposed = item_group_exposed[item_group]
        item_groups[item_group] = {
            "available": available,
            "exposed": exposed,
            "exposure_rate": exposed / available if available else 0.0,
        }
    item_exposure_disparity = 0.0
    if item_groups:
        rates = [payload["exposure_rate"] for payload in item_groups.values()]
        item_exposure_disparity = max(rates) - min(rates)

    return {
        "overall": overall,
        "groups": groups,
        "user_rows": all_rows,
        "item_groups": item_groups,
        "utility_disparity": utility_disparity,
        "burden_disparity": burden_disparity,
        "burden_disparity_raw": burden_disparity_raw,
        "item_exposure_disparity": item_exposure_disparity,
        "delta_e": utility_disparity + omega * burden_disparity,
    }
