from __future__ import annotations

import csv
import json
from pathlib import Path

from .data import prepare_dataset
from .methods import evaluate_method, fit_method
from .scorers import build_scorer, score_cases
from .types import ScoredCandidate, ScoredUserEvalCase
from .utils import load_json


def _resolve_config(config_path: str | Path) -> dict:
    path = Path(config_path).resolve()
    config = load_json(path)
    config["_config_path"] = str(path)
    return config


def _apply_score_view(
    cases: tuple[ScoredUserEvalCase, ...],
    method_cfg: dict,
) -> tuple[ScoredUserEvalCase, ...]:
    score_view = method_cfg.get("score_view") or {"kind": "identity"}
    kind = score_view.get("kind", "identity")
    if kind == "identity":
        return cases
    if kind == "quantize":
        bins = max(2, int(score_view.get("bins", 8)))
        denom = bins - 1
        transformed: list[ScoredUserEvalCase] = []
        for case in cases:
            transformed.append(
                ScoredUserEvalCase(
                    split=case.split,
                    user_id=case.user_id,
                    group=case.group,
                    history_item_ids=case.history_item_ids,
                    candidates=tuple(
                        ScoredCandidate(
                            item_id=candidate.item_id,
                            label=candidate.label,
                            score=round(candidate.score * denom) / denom,
                            item_group=candidate.item_group,
                        )
                        for candidate in case.candidates
                    ),
                )
            )
        return tuple(transformed)
    if kind == "constant":
        value = float(score_view.get("value", 0.5))
        transformed: list[ScoredUserEvalCase] = []
        for case in cases:
            transformed.append(
                ScoredUserEvalCase(
                    split=case.split,
                    user_id=case.user_id,
                    group=case.group,
                    history_item_ids=case.history_item_ids,
                    candidates=tuple(
                        ScoredCandidate(
                            item_id=candidate.item_id,
                            label=candidate.label,
                            score=value,
                            item_group=candidate.item_group,
                        )
                        for candidate in case.candidates
                    ),
                )
            )
        return tuple(transformed)
    raise ValueError(f"Unsupported score view: {kind!r}")


def run_experiment(
    config_path: str | Path,
    output_dir: str | Path | None = None,
    method_names: set[str] | None = None,
) -> dict:
    config = _resolve_config(config_path)
    print(f"[rec] config={config['name']}")
    dataset = prepare_dataset(config)
    print(
        "[rec] dataset_ready "
        f"users={len(dataset.train_histories)} items={len(dataset.items)} "
        f"calibration_cases={len(dataset.calibration_cases)} test_cases={len(dataset.test_cases)}"
    )
    scorer = build_scorer(config, dataset)
    normalize_scores = bool(config.get("normalize_scores", True))

    print(f"[rec] scoring_calibration normalize={normalize_scores}")
    calibration_scored = score_cases(dataset.calibration_cases, dataset, scorer, normalize=normalize_scores)
    print(f"[rec] scoring_test normalize={normalize_scores}")
    test_scored = score_cases(dataset.test_cases, dataset, scorer, normalize=normalize_scores)

    output_root = Path(output_dir) if output_dir else (Path(config["_config_path"]).parent.parent / "outputs" / config["name"])
    output_root.mkdir(parents=True, exist_ok=True)
    print(f"[rec] output_dir={output_root}")

    results: dict = {
        "experiment": config["name"],
        "config_path": config["_config_path"],
        "dataset": {
            "num_users": len(dataset.train_histories),
            "num_items": len(dataset.items),
            "num_calibration_cases": len(calibration_scored),
            "num_test_cases": len(test_scored),
        },
        "scorer": config["scorer"],
        "methods": {},
    }

    ordered_method_names: list[str] = []
    methods = config["methods"]
    if method_names is not None:
        methods = [method_cfg for method_cfg in methods if method_cfg["name"] in method_names]
        missing = sorted(method_names.difference({method_cfg["name"] for method_cfg in methods}))
        if missing:
            raise ValueError(f"Requested methods not found in config: {missing}")
        print(f"[rec] selected_methods={','.join(method_cfg['name'] for method_cfg in methods)}")

    for method_cfg in methods:
        calibration_method_cases = _apply_score_view(calibration_scored, method_cfg)
        test_method_cases = _apply_score_view(test_scored, method_cfg)
        print(f"[rec] fitting_method name={method_cfg['name']} kind={method_cfg['kind']}")
        fitted = fit_method(method_cfg, calibration_method_cases)
        calibration_summary = evaluate_method(
            fitted,
            calibration_method_cases,
            method_cfg=method_cfg,
            reference_results=results["methods"],
        )
        test_summary = evaluate_method(
            fitted,
            test_method_cases,
            method_cfg=method_cfg,
            reference_results=results["methods"],
        )
        results["methods"][fitted.name] = {
            "kind": fitted.kind,
            "score_view": method_cfg.get("score_view", {"kind": "identity"}),
            "fit": {
                "thresholds": fitted.thresholds,
                "global_threshold": fitted.global_threshold,
                "top_k": fitted.top_k,
                "metadata": fitted.metadata or {},
            },
            "calibration": calibration_summary,
            "test": test_summary,
        }
        ordered_method_names.append(fitted.name)
        print(f"[rec] finished_method name={fitted.name}", flush=True)

    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2))
    print(f"[rec] wrote_summary={summary_path}")

    csv_path = output_root / "test_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "method",
            "kind",
            "ber",
            "mor",
            "avg_size",
            "avg_exposure_rate",
            "hr",
            "ndcg",
            "utility_disparity",
            "burden_disparity",
            "burden_disparity_raw",
            "item_exposure_disparity",
            "delta_e",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for method_name in ordered_method_names:
            row = results["methods"][method_name]
            overall = row["test"]["overall"]
            writer.writerow(
                {
                    "method": method_name,
                    "kind": row["kind"],
                    "ber": overall["ber"],
                    "mor": overall["mor"],
                    "avg_size": overall["avg_size"],
                    "avg_exposure_rate": overall["avg_exposure_rate"],
                    "hr": overall["hr"],
                    "ndcg": overall["ndcg"],
                    "utility_disparity": row["test"]["utility_disparity"],
                    "burden_disparity": row["test"]["burden_disparity"],
                    "burden_disparity_raw": row["test"]["burden_disparity_raw"],
                    "item_exposure_disparity": row["test"].get("item_exposure_disparity", 0.0),
                    "delta_e": row["test"]["delta_e"],
                }
            )
    print(f"[rec] wrote_csv={csv_path}")

    return {
        "output_dir": str(output_root),
        "summary_path": str(summary_path),
        "csv_path": str(csv_path),
        "results": results,
    }
