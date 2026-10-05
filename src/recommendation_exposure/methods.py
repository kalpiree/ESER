"""Exposure-policy implementations for the ESER recommendation paper.

All methods in this module operate on already-computed relevance scores. The
LLM or conventional recommender is treated as a black-box scorer; ESER only
calibrates the final displayed set by selecting exposure thresholds.
"""

from __future__ import annotations

import itertools
import math
import os
from dataclasses import dataclass
from typing import Any

from .metrics import summarize_exposures
from .types import ScoredUserEvalCase


@dataclass(frozen=True, slots=True)
class FittedMethod:
    name: str
    kind: str
    thresholds: dict[str, float] | None = None
    global_threshold: float | None = None
    top_k: int | None = None
    metadata: dict[str, Any] | None = None


def _positive_part(value: float) -> float:
    return value if value > 0.0 else 0.0


def _is_better_candidate(
    *,
    violation: float,
    objective: float,
    best_violation: float,
    best_objective: float,
) -> bool:
    if violation < best_violation:
        return True
    if math.isclose(violation, best_violation) and objective < best_objective:
        return True
    return False


def _compute_epsilons(
    grouped: dict[str, list[ScoredUserEvalCase]],
    *,
    pair_count: int,
    delta_hat: float,
) -> dict[str, float]:
    """Finite-sample correction used when selecting among threshold pairs."""
    epsilons: dict[str, float] = {}
    for group, cases in grouped.items():
        epsilons[group] = math.sqrt(math.log(8 * pair_count / max(delta_hat, 1e-9)) / (2 * max(1, len(cases))))
    return epsilons


def _compute_fairness_correction(
    *,
    group_names: list[str],
    epsilons: dict[str, float],
    omega: float,
    mode: str,
) -> dict[str, float | str]:
    pairwise_eps = epsilons[group_names[0]] + epsilons[group_names[1]]
    if mode == "none":
        utility = 0.0
        burden = 0.0
    elif mode == "normalized":
        utility = pairwise_eps
        burden = pairwise_eps
    else:
        raise ValueError(
            f"Unsupported fairness correction mode: {mode!r}. "
            "Expected one of: 'none', 'normalized'."
        )
    total = utility + omega * burden
    return {
        "mode": mode,
        "utility": utility,
        "burden": burden,
        "total": total,
    }


def _selection_objective(summary: dict, method_cfg: dict) -> float:
    mode = method_cfg.get("selection_objective", "min_size")
    overall = summary["overall"]
    if mode == "min_size":
        return float(overall["avg_size"])
    if mode == "max_utility_size_tradeoff":
        utility_key = method_cfg.get("selection_metric")
        if not utility_key:
            utility_key = "ndcg" if method_cfg.get("utility_metric", "hit_rate") == "ndcg" else "hr"
        size_key = method_cfg.get("size_penalty_metric", "avg_exposure_rate")
        size_penalty = float(method_cfg.get("size_penalty", 0.0))
        objective = -float(overall[utility_key]) + size_penalty * float(overall[size_key])
        target_exposure_rate = method_cfg.get("target_exposure_rate")
        if target_exposure_rate is not None:
            excess_penalty = float(method_cfg.get("excess_exposure_penalty", max(1.0, size_penalty)))
            objective += excess_penalty * _positive_part(
                float(overall["avg_exposure_rate"]) - float(target_exposure_rate)
            )
        return objective
    raise ValueError(
        f"Unsupported selection objective: {mode!r}. "
        "Expected one of: 'min_size', 'max_utility_size_tradeoff'."
    )


def _select_eser_candidate(
    *,
    feasible: bool,
    objective: float,
    total_violation: float,
    best_thresholds: dict[str, float] | None,
    best_score: float,
    best_violation: float,
    method_cfg: dict,
) -> bool:
    priority = method_cfg.get("constraint_priority", "violation_first")
    if priority == "objective_only":
        return best_thresholds is None or objective < best_score
    if priority == "feasible_then_objective":
        if feasible and best_violation > 0.0:
            return True
        if feasible and math.isclose(best_violation, 0.0):
            return objective < best_score
        if math.isclose(best_violation, 0.0):
            return False
        return best_thresholds is None or _is_better_candidate(
            violation=total_violation,
            objective=objective,
            best_violation=best_violation,
            best_objective=best_score,
        )
    if priority == "violation_first":
        return best_thresholds is None or _is_better_candidate(
            violation=total_violation,
            objective=objective,
            best_violation=best_violation,
            best_objective=best_score,
        )
    raise ValueError(
        f"Unsupported constraint priority: {priority!r}. "
        "Expected one of: 'violation_first', 'feasible_then_objective', 'objective_only'."
    )


def _group_cases(cases: tuple[ScoredUserEvalCase, ...]) -> dict[str, list[ScoredUserEvalCase]]:
    grouped: dict[str, list[ScoredUserEvalCase]] = {}
    for case in cases:
        grouped.setdefault(case.group, []).append(case)
    return grouped


def _threshold_grid(cases: list[ScoredUserEvalCase], max_thresholds: int) -> list[float]:
    """Build a candidate threshold grid from empirical scores."""
    scores = sorted({candidate.score for case in cases for candidate in case.candidates})
    if not scores:
        return [1.0]
    scores.append(max(scores) + 1e-9)
    if len(scores) <= max_thresholds:
        return scores
    picked: list[float] = []
    for index in range(max_thresholds):
        position = round(index * (len(scores) - 1) / max(1, max_thresholds - 1))
        picked.append(scores[position])
    return sorted(set(picked))


def _effective_threshold_limit(case_count: int, configured: int, method_cfg: dict) -> int:
    """Cap large threshold grids so full-dataset calibration stays tractable."""
    if not method_cfg.get("adaptive_threshold_cap", True):
        return configured
    if "auto_threshold_cap" in method_cfg:
        return min(configured, int(method_cfg["auto_threshold_cap"]))
    if case_count >= 250_000:
        return min(configured, 8)
    if case_count >= 50_000:
        return min(configured, 12)
    if case_count >= 5_000:
        return min(configured, 16)
    if case_count >= 1_000:
        return min(configured, 24)
    return configured


def _exposures_from_thresholds(
    cases: tuple[ScoredUserEvalCase, ...],
    *,
    thresholds: dict[str, float] | None = None,
    global_threshold: float | None = None,
) -> dict[str, list[str]]:
    """Apply the fitted exposure rule; this is the online ESER decision."""
    exposures: dict[str, list[str]] = {}
    for case in cases:
        threshold = global_threshold if global_threshold is not None else thresholds[case.group]  # type: ignore[index]
        exposures[case.user_id] = [
            candidate.item_id for candidate in case.candidates if candidate.score >= threshold
        ]
    return exposures


def _exposures_from_topk(cases: tuple[ScoredUserEvalCase, ...], top_k: int) -> dict[str, list[str]]:
    exposures: dict[str, list[str]] = {}
    for case in cases:
        ranked = sorted(case.candidates, key=lambda candidate: (-candidate.score, candidate.item_id))
        exposures[case.user_id] = [candidate.item_id for candidate in ranked[:top_k]]
    return exposures


def _exposures_from_ltt(
    cases: tuple[ScoredUserEvalCase, ...],
    *,
    lambda_value: float,
    min_items: int = 1,
) -> dict[str, list[str]]:
    """Local-thresholding baseline using each user's score distribution."""
    exposures: dict[str, list[str]] = {}
    for case in cases:
        scores = [candidate.score for candidate in case.candidates]
        mean_score = sum(scores) / len(scores) if scores else 0.0
        variance = sum((score - mean_score) ** 2 for score in scores) / len(scores) if scores else 0.0
        threshold = mean_score + lambda_value * math.sqrt(variance)
        selected = [candidate for candidate in case.candidates if candidate.score >= threshold]
        if len(selected) < min_items:
            selected = sorted(case.candidates, key=lambda candidate: (-candidate.score, candidate.item_id))[:min_items]
        exposures[case.user_id] = [candidate.item_id for candidate in selected]
    return exposures


def _exposures_from_item_fair_rerank(
    cases: tuple[ScoredUserEvalCase, ...],
    *,
    top_k: int,
    item_penalty: float,
) -> dict[str, list[str]]:
    """Greedy item-exposure baseline for item-side and two-sided fairness."""
    exposures: dict[str, list[str]] = {}
    item_group_counts: dict[str, int] = {}
    total_exposed = 0
    for case in sorted(cases, key=lambda value: value.user_id):
        selected = []
        remaining = list(case.candidates)
        while remaining and len(selected) < top_k:
            def adjusted(candidate):
                item_group = candidate.item_group or "unknown"
                exposure_rate = item_group_counts.get(item_group, 0) / max(1, total_exposed)
                return (candidate.score - item_penalty * exposure_rate, candidate.score, candidate.item_id)

            best = max(remaining, key=adjusted)
            selected.append(best)
            remaining.remove(best)
            item_group = best.item_group or "unknown"
            item_group_counts[item_group] = item_group_counts.get(item_group, 0) + 1
            total_exposed += 1
        exposures[case.user_id] = [candidate.item_id for candidate in selected]
    return exposures


def _constraint_violation(summary: dict, alpha: float | None, beta: float | None, eta: float | None = None) -> float:
    violation = 0.0
    if alpha is not None:
        violation += _positive_part(summary["overall"]["ber"] - alpha)
    if beta is not None:
        violation += _positive_part(summary["overall"]["mor"] - beta)
    if eta is not None:
        violation += _positive_part(summary["delta_e"] - eta)
    return violation


def _fit_user_fair(method_cfg: dict, calibration_cases: tuple[ScoredUserEvalCase, ...]) -> FittedMethod:
    """Fit a user-side fairness baseline over group-specific thresholds."""
    alpha = float(method_cfg.get("alpha", 0.15))
    beta = float(method_cfg.get("beta", 0.15))
    eta = method_cfg.get("eta")
    eta_value = float(eta) if eta is not None else None
    omega = float(method_cfg.get("omega", 1.0))
    utility_metric = method_cfg.get("utility_metric", "ndcg")
    max_thresholds = int(method_cfg.get("max_thresholds_per_group", 64))
    grouped = _group_cases(calibration_cases)
    group_names = sorted(grouped)
    if len(group_names) != 2:
        raise ValueError("user_fair expects exactly two user groups.")
    threshold_grids = {
        group: _threshold_grid(cases, _effective_threshold_limit(len(cases), max_thresholds, method_cfg))
        for group, cases in grouped.items()
    }
    best_thresholds = None
    best_violation = math.inf
    best_objective = math.inf
    best_summary = {}
    for left, right in itertools.product(threshold_grids[group_names[0]], threshold_grids[group_names[1]]):
        thresholds = {group_names[0]: left, group_names[1]: right}
        exposures = _exposures_from_thresholds(calibration_cases, thresholds=thresholds)
        summary = summarize_exposures(calibration_cases, exposures, omega=omega, fairness_metric=utility_metric)
        violation = _constraint_violation(summary, alpha, beta, eta_value)
        objective = summary["utility_disparity"] + omega * summary["burden_disparity"]
        objective += float(method_cfg.get("size_penalty", 0.01)) * summary["overall"]["avg_exposure_rate"]
        if _is_better_candidate(
            violation=violation,
            objective=objective,
            best_violation=best_violation,
            best_objective=best_objective,
        ):
            best_thresholds = thresholds
            best_violation = violation
            best_objective = objective
            best_summary = summary
    return FittedMethod(
        name=method_cfg["name"],
        kind=method_cfg["kind"],
        thresholds=best_thresholds,
        metadata={"calibration": best_summary, "constraint_violation": best_violation, "objective": best_objective},
    )


def _fit_local_policy(method_cfg: dict, calibration_cases: tuple[ScoredUserEvalCase, ...]) -> FittedMethod:
    """Fit baselines whose policy is local to a user's candidate list."""
    kind = method_cfg["kind"]
    alpha = float(method_cfg.get("alpha", 0.15))
    beta = float(method_cfg.get("beta", 0.15))
    eta = method_cfg.get("eta")
    eta_value = float(eta) if eta is not None else None
    omega = float(method_cfg.get("omega", 1.0))
    utility_metric = method_cfg.get("utility_metric", "ndcg")
    best_metadata = {}
    best_violation = math.inf
    best_objective = math.inf

    if kind == "ltt":
        lambdas = method_cfg.get("lambda_grid", [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])
        for lambda_value in [float(value) for value in lambdas]:
            exposures = _exposures_from_ltt(
                calibration_cases,
                lambda_value=lambda_value,
                min_items=int(method_cfg.get("min_items", 1)),
            )
            summary = summarize_exposures(calibration_cases, exposures, omega=omega, fairness_metric=utility_metric)
            violation = _constraint_violation(summary, alpha, beta, eta_value)
            objective = -summary["overall"]["ndcg"] + float(method_cfg.get("size_penalty", 0.01)) * summary["overall"]["avg_exposure_rate"]
            if _is_better_candidate(
                violation=violation,
                objective=objective,
                best_violation=best_violation,
                best_objective=best_objective,
            ):
                best_violation = violation
                best_objective = objective
                best_metadata = {"lambda": lambda_value, "calibration": summary}
    else:
        top_ks = [int(value) for value in method_cfg.get("top_k_grid", [3, 5, 10, 15, 20])]
        penalties = [float(value) for value in method_cfg.get("item_penalty_grid", [0.0, 0.25, 0.5, 1.0, 2.0])]
        for top_k, penalty in itertools.product(top_ks, penalties):
            exposures = _exposures_from_item_fair_rerank(calibration_cases, top_k=top_k, item_penalty=penalty)
            summary = summarize_exposures(calibration_cases, exposures, omega=omega, fairness_metric=utility_metric)
            violation = _constraint_violation(summary, alpha, beta, eta_value)
            if kind == "item_fair":
                objective = summary["item_exposure_disparity"]
            elif kind == "two_sided_fair":
                objective = summary["delta_e"] + summary["item_exposure_disparity"]
            else:
                raise ValueError(f"Unsupported local policy kind: {kind!r}")
            objective += float(method_cfg.get("utility_penalty", 1.0)) * (1.0 - summary["overall"]["ndcg"])
            if _is_better_candidate(
                violation=violation,
                objective=objective,
                best_violation=best_violation,
                best_objective=best_objective,
            ):
                best_violation = violation
                best_objective = objective
                best_metadata = {"top_k": top_k, "item_penalty": penalty, "calibration": summary}
    best_metadata["constraint_violation"] = best_violation
    best_metadata["objective"] = best_objective
    return FittedMethod(name=method_cfg["name"], kind=kind, metadata=best_metadata)


def _search_best_group_threshold(
    cases: list[ScoredUserEvalCase],
    *,
    alpha: float | None,
    beta: float | None,
    omega: float,
    utility_metric: str,
    max_thresholds: int,
) -> tuple[float, dict[str, float]]:
    best_threshold = max(candidate.score for case in cases for candidate in case.candidates) + 1e-9
    best_metrics = {"avg_size": math.inf, "ber": 0.0, "mor": 1.0}
    best_violation = math.inf
    for threshold in _threshold_grid(cases, max_thresholds):
        exposures = _exposures_from_thresholds(tuple(cases), thresholds={cases[0].group: threshold})
        metrics = summarize_exposures(tuple(cases), exposures, omega=omega, fairness_metric=utility_metric)["overall"]
        violation = 0.0
        if alpha is not None:
            violation += _positive_part(metrics["ber"] - alpha)
        if beta is not None:
            violation += _positive_part(metrics["mor"] - beta)
        if alpha is not None or beta is not None:
            if not _is_better_candidate(
                violation=violation,
                objective=metrics["avg_size"],
                best_violation=best_violation,
                best_objective=best_metrics["avg_size"],
            ):
                continue
            best_violation = violation
            best_threshold = threshold
            best_metrics = metrics
        elif metrics["avg_size"] < best_metrics["avg_size"] or (
            math.isclose(metrics["avg_size"], best_metrics["avg_size"]) and threshold > best_threshold
        ):
            best_threshold = threshold
            best_metrics = metrics
    return best_threshold, best_metrics


def _estimate_group_stability(
    cases: list[ScoredUserEvalCase],
    *,
    alpha: float,
    beta: float,
    omega: float,
    utility_metric: str,
    max_thresholds: int,
    max_leave_one_out_cases: int = 200,
) -> dict[str, float]:
    """Estimate threshold instability with a bounded leave-one-out proxy.

    This implements the stability-correction idea from GSEFA without requiring
    an exhaustive leave-one-out pass on very large calibration splits.
    """
    thresholds = _threshold_grid(cases, max_thresholds)
    def rank_for(threshold: float) -> int:
        return min(range(len(thresholds)), key=lambda index: abs(thresholds[index] - threshold))

    full_threshold, _ = _search_best_group_threshold(
        cases,
        alpha=alpha,
        beta=beta,
        omega=omega,
        utility_metric=utility_metric,
        max_thresholds=max_thresholds,
    )
    full_rank = rank_for(full_threshold)
    max_rank_shift = 0
    if len(cases) <= 1:
        return {"k": 0.0, "xi_bad": 0.0, "xi_opp": 0.0}

    leave_one_out_indices = list(range(len(cases)))
    if len(leave_one_out_indices) > max_leave_one_out_cases:
        step = len(leave_one_out_indices) / max_leave_one_out_cases
        leave_one_out_indices = sorted(
            {
                min(len(cases) - 1, int(round(position * step)))
                for position in range(max_leave_one_out_cases)
            }
        )

    for index in leave_one_out_indices:
        subset = [case for offset, case in enumerate(cases) if offset != index]
        subset_threshold, _ = _search_best_group_threshold(
            subset,
            alpha=alpha,
            beta=beta,
            omega=omega,
            utility_metric=utility_metric,
            max_thresholds=max_thresholds,
        )
        subset_rank = rank_for(subset_threshold)
        max_rank_shift = max(max_rank_shift, abs(subset_rank - full_rank))

    n = len(cases)
    sample_scale = len(leave_one_out_indices) / n
    xi_bad = 2.0 * max(alpha, 1.0 - alpha) * max_rank_shift / (n + 1)
    xi_opp = 2.0 * max_rank_shift / (n + 1)
    return {
        "k": float(max_rank_shift),
        "xi_bad": xi_bad,
        "xi_opp": xi_opp,
        "sampled_cases": float(len(leave_one_out_indices)),
        "sample_fraction": sample_scale,
    }


def fit_method(
    method_cfg: dict,
    calibration_cases: tuple[ScoredUserEvalCase, ...],
) -> FittedMethod:
    name = method_cfg["name"]
    kind = method_cfg["kind"]
    grouped = _group_cases(calibration_cases)

    if kind == "top_k":
        return FittedMethod(name=name, kind=kind, top_k=int(method_cfg["k"]), metadata={})

    if kind == "coverage_set":
        beta = float(method_cfg["beta"])
        max_thresholds = int(method_cfg.get("max_thresholds_per_group", 64))
        thresholds = {}
        per_group = {}
        for group, cases in grouped.items():
            group_thresholds = _effective_threshold_limit(len(cases), max_thresholds, method_cfg)
            threshold, metrics = _search_best_group_threshold(
                cases,
                alpha=None,
                beta=beta,
                omega=float(method_cfg.get("omega", 1.0)),
                utility_metric=method_cfg.get("utility_metric", "hit_rate"),
                max_thresholds=group_thresholds,
            )
            thresholds[group] = threshold
            per_group[group] = metrics
        return FittedMethod(name=name, kind=kind, thresholds=thresholds, metadata={"per_group": per_group})

    if kind == "user_fair":
        return _fit_user_fair(method_cfg, calibration_cases)

    if kind in {"item_fair", "two_sided_fair", "ltt"}:
        return _fit_local_policy(method_cfg, calibration_cases)

    if kind == "group_threshold":
        alpha = float(method_cfg["alpha"])
        beta = float(method_cfg["beta"])
        max_thresholds = int(method_cfg.get("max_thresholds_per_group", 64))
        thresholds = {}
        per_group = {}
        for group, cases in grouped.items():
            group_thresholds = _effective_threshold_limit(len(cases), max_thresholds, method_cfg)
            threshold, metrics = _search_best_group_threshold(
                cases,
                alpha=alpha,
                beta=beta,
                omega=float(method_cfg.get("omega", 1.0)),
                utility_metric=method_cfg.get("utility_metric", "hit_rate"),
                max_thresholds=group_thresholds,
            )
            thresholds[group] = threshold
            per_group[group] = metrics
        return FittedMethod(name=name, kind=kind, thresholds=thresholds, metadata={"per_group": per_group})

    if kind == "global_threshold":
        alpha = float(method_cfg["alpha"])
        beta = float(method_cfg["beta"])
        max_thresholds = int(method_cfg.get("max_thresholds_global", 128))
        max_thresholds = _effective_threshold_limit(len(calibration_cases), max_thresholds, method_cfg)
        best_threshold = max(candidate.score for case in calibration_cases for candidate in case.candidates) + 1e-9
        best_metrics = {"avg_size": math.inf, "ber": 0.0, "mor": 1.0}
        best_violation = math.inf
        pooled_cases = list(calibration_cases)
        for threshold in _threshold_grid(pooled_cases, max_thresholds):
            exposures = _exposures_from_thresholds(calibration_cases, global_threshold=threshold)
            metrics = summarize_exposures(
                calibration_cases,
                exposures,
                omega=float(method_cfg.get("omega", 1.0)),
                fairness_metric=method_cfg.get("utility_metric", "hit_rate"),
            )["overall"]
            violation = _positive_part(metrics["ber"] - alpha) + _positive_part(metrics["mor"] - beta)
            if not _is_better_candidate(
                violation=violation,
                objective=metrics["avg_size"],
                best_violation=best_violation,
                best_objective=best_metrics["avg_size"],
            ):
                continue
            best_violation = violation
            best_threshold = threshold
            best_metrics = metrics
        return FittedMethod(
            name=name,
            kind=kind,
            global_threshold=best_threshold,
            metadata={"overall": best_metrics, "constraint_violation": best_violation},
        )

    if kind == "eser":
        alpha = float(method_cfg["alpha"])
        beta = float(method_cfg["beta"])
        eta = float(method_cfg["eta"])
        delta_hat = float(method_cfg.get("delta_hat", 0.05))
        omega = float(method_cfg.get("omega", 1.0))
        utility_metric = method_cfg.get("utility_metric", "hit_rate")
        max_thresholds = int(method_cfg.get("max_thresholds_per_group", 64))
        stability_mode = os.environ.get("REC_ESER_STABILITY_MODE", method_cfg.get("stability_mode", "user_leave_one_out"))
        fairness_correction_mode = method_cfg.get("fairness_correction_mode", "none")

        if len(grouped) != 2:
            raise ValueError("ESER currently expects exactly two groups in calibration.")

        if stability_mode == "auto":
            disable_above_cases = int(
                os.environ.get(
                    "REC_ESER_STABILITY_DISABLE_ABOVE_CASES",
                    method_cfg.get("stability_disable_above_cases", 5_000),
                )
            )
            stability_mode = "none" if len(calibration_cases) >= disable_above_cases else "user_leave_one_out"
            print(
                f"[rec:eser] resolved_stability_mode={stability_mode} "
                f"calibration_cases={len(calibration_cases)} disable_above={disable_above_cases}",
                flush=True,
            )

        group_names = sorted(grouped)
        threshold_grids = {
            group: _threshold_grid(
                cases,
                _effective_threshold_limit(len(cases), max_thresholds, method_cfg),
            )
            for group, cases in grouped.items()
        }
        corrections = {}
        for group, cases in grouped.items():
            if stability_mode == "none":
                corrections[group] = {"k": 0.0, "xi_bad": 0.0, "xi_opp": 0.0}
            else:
                corrections[group] = _estimate_group_stability(
                    cases,
                    alpha=alpha,
                    beta=beta,
                    omega=omega,
                    utility_metric=utility_metric,
                    max_thresholds=max_thresholds,
                    max_leave_one_out_cases=int(
                        os.environ.get(
                            "REC_ESER_STABILITY_MAX_CASES",
                            method_cfg.get("stability_max_cases", 200),
                        )
                    ),
                )

        pair_count = max(1, len(threshold_grids[group_names[0]]) * len(threshold_grids[group_names[1]]))
        epsilons = _compute_epsilons(
            grouped,
            pair_count=pair_count,
            delta_hat=delta_hat,
        )
        fairness_correction = _compute_fairness_correction(
            group_names=group_names,
            epsilons=epsilons,
            omega=omega,
            mode=fairness_correction_mode,
        )

        best_thresholds: dict[str, float] | None = None
        best_score = math.inf
        best_metadata: dict[str, Any] = {}
        best_violation = math.inf

        # GSEFA-style threshold-pair search: evaluate every group-threshold
        # pair on calibration data, apply reliability/opportunity/fairness
        # corrections, and keep the smallest feasible exposed set.
        for left_threshold, right_threshold in itertools.product(
            threshold_grids[group_names[0]], threshold_grids[group_names[1]]
        ):
            threshold_map = {group_names[0]: left_threshold, group_names[1]: right_threshold}
            exposures = _exposures_from_thresholds(calibration_cases, thresholds=threshold_map)
            summary = summarize_exposures(
                calibration_cases,
                exposures,
                omega=omega,
                fairness_metric=utility_metric,
            )
            feasible = True
            corrected = {}
            for group_name in group_names:
                group_metrics = summary["groups"][group_name]
                corrected_bad = group_metrics["ber"] + corrections[group_name]["xi_bad"]
                corrected_opp = group_metrics["mor"] + corrections[group_name]["xi_opp"]
                corrected[group_name] = {"ber": corrected_bad, "mor": corrected_opp}
                if corrected_bad > alpha or corrected_opp > beta:
                    feasible = False
            corrected_fairness = summary["delta_e"] + float(fairness_correction["total"])
            fairness_violation = _positive_part(corrected_fairness - eta)
            if corrected_fairness > eta:
                feasible = False

            total_violation = fairness_violation
            for group_name in group_names:
                total_violation += _positive_part(corrected[group_name]["ber"] - alpha)
                total_violation += _positive_part(corrected[group_name]["mor"] - beta)
            objective = _selection_objective(summary, method_cfg)
            if _select_eser_candidate(
                feasible=feasible,
                objective=objective,
                total_violation=total_violation,
                best_thresholds=best_thresholds,
                best_score=best_score,
                best_violation=best_violation,
                method_cfg=method_cfg,
            ):
                best_violation = 0.0 if feasible else total_violation
                best_score = objective
                best_thresholds = threshold_map
                best_metadata = {
                    "calibration": summary,
                    "corrected": corrected,
                    "fairness_correction": fairness_correction,
                    "corrected_fairness": corrected_fairness,
                    "epsilons": epsilons,
                    "stability": corrections,
                    "selection_objective": method_cfg.get("selection_objective", "min_size"),
                    "constraint_priority": method_cfg.get("constraint_priority", "violation_first"),
                    "constraint_violation": 0.0 if feasible else total_violation,
                }
                if not feasible:
                    best_metadata["warning"] = (
                        "Selected an infeasible ESER threshold pair under the configured selection policy."
                    )

        return FittedMethod(name=name, kind=kind, thresholds=best_thresholds, metadata=best_metadata)

    if kind == "top_k_matched":
        return FittedMethod(name=name, kind=kind, metadata={"match_to": method_cfg["match_to"]})

    raise ValueError(f"Unsupported method kind: {kind}")


def evaluate_method(
    fitted: FittedMethod,
    cases: tuple[ScoredUserEvalCase, ...],
    *,
    method_cfg: dict,
    reference_results: dict[str, dict] | None = None,
) -> dict:
    """Evaluate a fitted exposure policy on calibration or test cases."""
    omega = float(method_cfg.get("omega", 1.0))
    utility_metric = method_cfg.get("utility_metric", "hit_rate")

    if fitted.kind == "top_k":
        exposures = _exposures_from_topk(cases, fitted.top_k or 0)
    elif fitted.kind == "ltt":
        exposures = _exposures_from_ltt(
            cases,
            lambda_value=float((fitted.metadata or {}).get("lambda", 0.0)),
            min_items=int(method_cfg.get("min_items", 1)),
        )
    elif fitted.kind in {"item_fair", "two_sided_fair"}:
        exposures = _exposures_from_item_fair_rerank(
            cases,
            top_k=int((fitted.metadata or {}).get("top_k", method_cfg.get("k", 10))),
            item_penalty=float((fitted.metadata or {}).get("item_penalty", 0.0)),
        )
    elif fitted.kind == "top_k_matched":
        match_to = (fitted.metadata or {}).get("match_to")
        if not reference_results or match_to not in reference_results:
            raise ValueError(f"Matched-size baseline '{fitted.name}' needs prior result '{match_to}'.")
        matched_size_raw = float(reference_results[match_to]["test"]["overall"]["avg_size"])
        matched_size = 0 if matched_size_raw <= 0 else max(1, int(math.floor(matched_size_raw + 0.5)))
        exposures = _exposures_from_topk(cases, matched_size)
    elif fitted.global_threshold is not None:
        exposures = _exposures_from_thresholds(cases, global_threshold=fitted.global_threshold)
    else:
        exposures = _exposures_from_thresholds(cases, thresholds=fitted.thresholds)

    summary = summarize_exposures(cases, exposures, omega=omega, fairness_metric=utility_metric)
    constraints = {}
    if "alpha" in method_cfg:
        alpha = float(method_cfg["alpha"])
        constraints["ber"] = all(group["ber"] <= alpha for group in summary["groups"].values())
    if "beta" in method_cfg:
        beta = float(method_cfg["beta"])
        constraints["mor"] = all(group["mor"] <= beta for group in summary["groups"].values())
    if "eta" in method_cfg:
        constraints["delta_e"] = summary["delta_e"] <= float(method_cfg["eta"])
    summary["constraints_satisfied"] = constraints
    return summary
