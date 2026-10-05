#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


ESER_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ESER_ROOT / "experiments/recommendation_eser/configs"
BASE_CONFIG = CONFIG_DIR / "movielens1m_qwen25_3b_precomputed_50neg.json"

QWEN_SCORE_PATHS = {
    "qwen25_0_5b": (
        "../outputs/scores/movielens1m/qwen25_0_5b_scores_50neg_multillm.jsonl",
        "movielens1m_qwen25_0_5b_precomputed_50neg",
    ),
    "qwen25_1_5b": (
        "../outputs/scores/movielens1m/qwen25_1_5b_scores_50neg_multillm.jsonl",
        "movielens1m_qwen25_1_5b_precomputed_50neg",
    ),
    "qwen25_3b": (
        "../outputs/scores/movielens1m/qwen25_3b_scores_50neg_multillm.jsonl",
        "movielens1m_qwen25_3b_precomputed_50neg",
    ),
}


def _methods() -> list[dict[str, Any]]:
    return [
        {
            "name": "topk",
            "kind": "top_k",
            "k": 10,
            "omega": 1.0,
            "utility_metric": "ndcg",
        },
        {
            "name": "global_thr",
            "kind": "global_threshold",
            "alpha": 0.15,
            "beta": 0.15,
            "omega": 1.0,
            "utility_metric": "ndcg",
            "max_thresholds_global": 96,
        },
        {
            "name": "group_thr",
            "kind": "group_threshold",
            "alpha": 0.15,
            "beta": 0.15,
            "omega": 1.0,
            "utility_metric": "ndcg",
            "max_thresholds_per_group": 48,
        },
        {
            "name": "coverage_set",
            "kind": "coverage_set",
            "beta": 0.15,
            "omega": 1.0,
            "utility_metric": "ndcg",
            "max_thresholds_per_group": 48,
        },
        {
            "name": "user_fair",
            "kind": "user_fair",
            "alpha": 0.15,
            "beta": 0.15,
            "eta": 0.08,
            "omega": 1.0,
            "utility_metric": "ndcg",
            "max_thresholds_per_group": 48,
            "size_penalty": 0.01,
        },
        {
            "name": "far",
            "kind": "item_fair",
            "alpha": 0.15,
            "beta": 0.15,
            "eta": 0.08,
            "omega": 1.0,
            "utility_metric": "ndcg",
            "top_k_grid": [5, 10, 15, 20, 30],
            "item_penalty_grid": [0.0, 0.25, 0.5, 1.0, 2.0],
            "utility_penalty": 0.5,
        },
        {
            "name": "fairrec_style",
            "kind": "two_sided_fair",
            "alpha": 0.15,
            "beta": 0.15,
            "eta": 0.08,
            "omega": 1.0,
            "utility_metric": "ndcg",
            "top_k_grid": [5, 10, 15, 20, 30],
            "item_penalty_grid": [0.0, 0.25, 0.5, 1.0, 2.0],
            "utility_penalty": 0.5,
        },
        {
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
        },
    ]


def _base_config() -> dict[str, Any]:
    config = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    config["data"]["case_seed_name"] = config["name"]
    config["methods"] = _methods()
    return config


def _write_config(config: dict[str, Any]) -> str:
    path = CONFIG_DIR / f"{config['name']}.json"
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return str(path.relative_to(ESER_ROOT))


def build_configs() -> list[str]:
    base = _base_config()
    configs: list[dict[str, Any]] = []

    conventional = {
        "popularity": {"kind": "popularity"},
        "bpr_mf": {
            "kind": "bpr_mf",
            "device": "auto",
            "embedding_dim": 64,
            "epochs": 5,
            "batch_size": 4096,
            "max_train_pairs": 350000,
            "learning_rate": 0.01,
            "seed": 17,
        },
        "neumf": {
            "kind": "neumf",
            "device": "auto",
            "embedding_dim": 48,
            "hidden_dim": 96,
            "epochs": 5,
            "batch_size": 4096,
            "max_train_pairs": 300000,
            "negatives_per_positive": 1,
            "learning_rate": 0.001,
            "seed": 17,
        },
        "sasrec_lite": {
            "kind": "sasrec_lite",
            "device": "auto",
            "embedding_dim": 64,
            "max_sequence_length": 50,
            "heads": 2,
            "layers": 2,
            "dropout": 0.1,
            "epochs": 3,
            "batch_size": 512,
            "max_train_sequences": 140000,
            "learning_rate": 0.001,
            "seed": 17,
        },
    }
    for scorer_name, scorer_cfg in conventional.items():
        config = copy.deepcopy(base)
        config["name"] = f"rq_motivation_movielens1m_{scorer_name}_50neg"
        config["scorer"] = scorer_cfg
        configs.append(config)

    for scorer_name, (score_path, case_seed_name) in QWEN_SCORE_PATHS.items():
        config = copy.deepcopy(base)
        config["name"] = f"rq_motivation_movielens1m_{scorer_name}_50neg"
        config["data"]["case_seed_name"] = case_seed_name
        config["scorer"] = {"kind": "precomputed_jsonl", "scores_path": score_path}
        configs.append(config)

    paths = [_write_config(config) for config in configs]
    manifest = CONFIG_DIR / "motivation_analysis_manifest.json"
    manifest.write_text(json.dumps({"configs": paths}, indent=2) + "\n", encoding="utf-8")
    return paths


def main() -> None:
    for path in build_configs():
        print(path)


if __name__ == "__main__":
    main()
