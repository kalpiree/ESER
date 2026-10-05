#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


ESER_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ESER_ROOT / "experiments/recommendation_eser/configs"

BASE_CONFIGS = {
    "movielens1m": "movielens1m_qwen25_3b_precomputed_50neg.json",
    "lastfm1k": "lastfm1k_qwen25_3b_precomputed_50neg.json",
    "bookcrossing": "bookcrossing_qwen25_3b_precomputed_50neg.json",
    "amazon_books": "amazon_books_qwen25_3b_precomputed_50neg.json",
}


def _load(dataset: str) -> dict[str, Any]:
    return json.loads((CONFIG_DIR / BASE_CONFIGS[dataset]).read_text(encoding="utf-8"))


def _method(base: dict[str, Any], name: str) -> dict[str, Any]:
    for method in base["methods"]:
        if method["name"] == name:
            return copy.deepcopy(method)
    raise KeyError(name)


def _fair_methods(base: dict[str, Any]) -> list[dict[str, Any]]:
    eser = _method(base, "eser")
    eser_s = _method(base, "eser_s")
    group_thr = _method(base, "group_thr")
    global_thr = _method(base, "global_thr")
    coverage_set = _method(base, "coverage_set")
    topk = _method(base, "topk")
    topk_matched = _method(base, "topk_matched_eser")

    user_fair = copy.deepcopy(group_thr)
    user_fair["name"] = "user_fair"
    user_fair["kind"] = "user_fair"
    user_fair["eta"] = 0.08
    user_fair["size_penalty"] = 0.01

    item_fair = {
        "name": "item_fair",
        "kind": "item_fair",
        "alpha": 0.15,
        "beta": 0.15,
        "eta": 0.08,
        "omega": 1.0,
        "utility_metric": "ndcg",
        "top_k_grid": [3, 5, 10, 15, 20, 30],
        "item_penalty_grid": [0.0, 0.25, 0.5, 1.0, 2.0],
        "utility_penalty": 0.5,
    }

    two_sided_fair = copy.deepcopy(item_fair)
    two_sided_fair["name"] = "two_sided_fair"
    two_sided_fair["kind"] = "two_sided_fair"

    ltt = {
        "name": "ltt",
        "kind": "ltt",
        "alpha": 0.15,
        "beta": 0.15,
        "eta": 0.08,
        "omega": 1.0,
        "utility_metric": "ndcg",
        "lambda_grid": [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0],
        "min_items": 1,
        "size_penalty": 0.01,
    }

    return [
        topk,
        global_thr,
        group_thr,
        coverage_set,
        user_fair,
        item_fair,
        two_sided_fair,
        ltt,
        eser_s,
        eser,
        topk_matched,
    ]


def build_configs() -> list[str]:
    paths = []
    for dataset in ["movielens1m", "lastfm1k", "bookcrossing", "amazon_books"]:
        base = _load(dataset)
        config = copy.deepcopy(base)
        config["name"] = f"rq_mainfair_{dataset}_qwen25_3b_50neg"
        config["data"]["case_seed_name"] = base["name"]
        config["methods"] = _fair_methods(base)
        path = CONFIG_DIR / f"{config['name']}.json"
        path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        paths.append(str(path.relative_to(ESER_ROOT)))
    manifest = CONFIG_DIR / "main_fairness_baselines_manifest.json"
    manifest.write_text(json.dumps({"configs": paths}, indent=2) + "\n", encoding="utf-8")
    return paths


def main() -> None:
    for path in build_configs():
        print(path)


if __name__ == "__main__":
    main()
