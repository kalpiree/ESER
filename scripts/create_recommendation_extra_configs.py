#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any


ESER_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ESER_ROOT / "experiments/recommendation_eser/configs"

BASE_CONFIGS = {
    "movielens1m": "movielens1m_token_overlap.json",
    "lastfm1k": "lastfm1k_token_overlap.json",
    "bookcrossing": "bookcrossing_token_overlap.json",
    "amazon_books": "amazon_books_token_overlap.json",
}


def _load_base(dataset: str) -> dict[str, Any]:
    return json.loads((CONFIG_DIR / BASE_CONFIGS[dataset]).read_text())


def _write_config(config: dict[str, Any], paths: list[str]) -> None:
    path = CONFIG_DIR / f"{config['name']}.json"
    path.write_text(json.dumps(config, indent=2) + "\n")
    paths.append(str(path.relative_to(ESER_ROOT)))


def _with_common(config: dict[str, Any], name: str, negative_sample_size: int = 50) -> dict[str, Any]:
    cfg = copy.deepcopy(config)
    cfg["name"] = name
    cfg["split"]["negative_sample_size"] = negative_sample_size
    return cfg


def _base_methods(config: dict[str, Any]) -> list[dict[str, Any]]:
    return copy.deepcopy(config["methods"])


def _method(config: dict[str, Any], name: str) -> dict[str, Any]:
    for method in config["methods"]:
        if method["name"] == name:
            return copy.deepcopy(method)
    raise KeyError(f"Missing method {name!r} in {config['name']}")


def _ablation_methods(config: dict[str, Any]) -> list[dict[str, Any]]:
    eser = _method(config, "eser")
    eser_s = _method(config, "eser_s")
    group_thr = _method(config, "group_thr")
    global_thr = _method(config, "global_thr")

    no_opp = copy.deepcopy(eser)
    no_opp["name"] = "eser_wo_opportunity"
    no_opp["beta"] = 1.0

    no_fair = copy.deepcopy(eser)
    no_fair["name"] = "eser_wo_fairness"
    no_fair["eta"] = 1.0

    eser_global = copy.deepcopy(global_thr)
    eser_global["name"] = "eser_global"

    return [
        copy.deepcopy(eser_s),
        no_opp,
        no_fair,
        copy.deepcopy(group_thr),
        eser_global,
        copy.deepcopy(eser),
    ]


def _stability_methods(config: dict[str, Any]) -> list[dict[str, Any]]:
    return [_method(config, "group_thr"), _method(config, "eser_s"), _method(config, "eser")]


def _rq5_sensitivity_methods(config: dict[str, Any]) -> list[dict[str, Any]]:
    base = _method(config, "eser")
    specs = [
        ("eser_base", {}),
        ("eser_alpha_010", {"alpha": 0.10}),
        ("eser_alpha_020", {"alpha": 0.20}),
        ("eser_beta_010", {"beta": 0.10}),
        ("eser_beta_020", {"beta": 0.20}),
        ("eser_eta_004", {"eta": 0.04}),
        ("eser_eta_012", {"eta": 0.12}),
        ("eser_delta_0025", {"delta_hat": 0.025}),
        ("eser_delta_010", {"delta_hat": 0.10}),
    ]
    methods: list[dict[str, Any]] = []
    for name, overrides in specs:
        method = copy.deepcopy(base)
        method["name"] = name
        method.update(overrides)
        methods.append(method)
    return methods


def _secondary_activity(config: dict[str, Any], dataset: str) -> dict[str, Any]:
    cfg = copy.deepcopy(config)
    cfg["data"]["group_column"] = "activity_group"
    if dataset in {"lastfm1k", "bookcrossing"}:
        cfg["data"]["group_value_map"] = {
            "high": "high",
            "medium": "not_high",
            "low": "not_high",
        }
    return cfg


def build_configs() -> list[str]:
    paths: list[str] = []

    # RQ2: CPU cross-scorer rows with a conventional popularity scorer.
    for dataset in ["movielens1m", "lastfm1k", "bookcrossing"]:
        base = _load_base(dataset)
        cfg = _with_common(base, f"rq2_{dataset}_popularity")
        cfg["scorer"] = {"kind": "popularity"}
        cfg["methods"] = _base_methods(base)
        _write_config(cfg, paths)

    amazon = _load_base("amazon_books")
    cfg = _with_common(amazon, "rq2_amazon_books_popularity_capped")
    cfg["scorer"] = {"kind": "popularity"}
    cfg["data"]["group_column"] = "activity_group"
    cfg["data"]["user_sample_fraction"] = 0.12
    cfg["methods"] = _base_methods(amazon)
    _write_config(cfg, paths)

    # RQ3: stability correction comparison on representative datasets.
    for dataset, caps in {
        "movielens1m": {"25": 1500, "100": None},
        "bookcrossing": {"25": 5000, "100": None},
    }.items():
        base = _load_base(dataset)
        for frac, cap in caps.items():
            cfg = _with_common(base, f"rq3_{dataset}_stability_{frac}pct")
            if cap is not None:
                cfg["data"]["max_eval_users"] = cap
            cfg["methods"] = _stability_methods(base)
            _write_config(cfg, paths)

    # RQ4: component ablations on two representative datasets.
    for dataset in ["movielens1m", "lastfm1k"]:
        base = _load_base(dataset)
        cfg = _with_common(base, f"rq4_{dataset}_ablation")
        cfg["methods"] = _ablation_methods(base)
        _write_config(cfg, paths)

    # RQ2 robustness: secondary group definitions.
    for dataset in ["movielens1m", "lastfm1k", "bookcrossing", "amazon_books"]:
        base = _load_base(dataset)
        cfg = _with_common(_secondary_activity(base, dataset), f"rq5_{dataset}_activity_group")
        if dataset == "amazon_books":
            cfg["data"]["user_sample_fraction"] = 0.12
            cfg["data"]["numeric_group"] = {
                "column": "interaction_count",
                "threshold": 20,
                "lower_label": "lower_activity",
                "upper_label": "high_activity",
            }
        cfg["methods"] = _base_methods(base)
        _write_config(cfg, paths)

    # RQ5: parameter sensitivity/efficiency on compact representative CPU runs.
    for dataset, cap in {"movielens1m": 3000, "lastfm1k": None}.items():
        base = _load_base(dataset)
        cfg = _with_common(base, f"rq5_{dataset}_sensitivity")
        if cap is not None:
            cfg["data"]["max_eval_users"] = cap
        cfg["methods"] = _rq5_sensitivity_methods(base)
        _write_config(cfg, paths)

    manifest_path = CONFIG_DIR / "recommendation_extra_cpu_manifest.json"
    manifest_path.write_text(json.dumps({"configs": paths}, indent=2) + "\n")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Create CPU-only recommendation configs for RQ2-RQ5 extras.")
    parser.add_argument("--print", action="store_true", help="Print generated config paths.")
    args = parser.parse_args()
    paths = build_configs()
    if args.print:
        for path in paths:
            print(path)


if __name__ == "__main__":
    main()
