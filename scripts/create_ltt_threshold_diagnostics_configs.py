#!/usr/bin/env python3
from __future__ import annotations

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


def _load_config(dataset: str) -> dict[str, Any]:
    return json.loads((CONFIG_DIR / BASE_CONFIGS[dataset]).read_text(encoding="utf-8"))


def _common_method(name: str, kind: str, alpha: float, beta: float) -> dict[str, Any]:
    method: dict[str, Any] = {
        "name": name,
        "kind": kind,
        "alpha": alpha,
        "beta": beta,
        "omega": 1.0,
        "utility_metric": "ndcg",
    }
    if kind == "global_threshold":
        method["max_thresholds_global"] = 128
    else:
        method["max_thresholds_per_group"] = 64
    return method


def _methods() -> list[dict[str, Any]]:
    return [
        _common_method("ltt_group_balanced", "group_threshold", 0.15, 0.15),
        _common_method("ltt_global_balanced", "global_threshold", 0.15, 0.15),
        _common_method("ltt_group_lenient", "group_threshold", 0.20, 0.20),
        _common_method("ltt_global_lenient", "global_threshold", 0.20, 0.20),
        {
            "name": "topk10_reference",
            "kind": "top_k",
            "k": 10,
            "omega": 1.0,
            "utility_metric": "ndcg",
        },
    ]


def build_configs() -> list[str]:
    paths: list[str] = []
    for dataset in ["movielens1m", "lastfm1k", "bookcrossing", "amazon_books"]:
        base_config = _load_config(dataset)
        config = copy.deepcopy(base_config)
        config["name"] = f"rq_ltt_{dataset}_threshold_diagnostics"
        config["data"]["case_seed_name"] = base_config["name"]
        config["split"]["negative_sample_size"] = 50
        config["methods"] = _methods()
        if dataset == "amazon_books":
            config["data"]["group_column"] = "activity_group"
            config["data"]["user_sample_fraction"] = 0.12
        path = CONFIG_DIR / f"{config['name']}.json"
        path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        paths.append(str(path.relative_to(ESER_ROOT)))
    manifest_path = CONFIG_DIR / "ltt_threshold_diagnostics_manifest.json"
    manifest_path.write_text(json.dumps({"configs": paths}, indent=2) + "\n", encoding="utf-8")
    return paths


def main() -> None:
    for path in build_configs():
        print(path)


if __name__ == "__main__":
    main()
