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
    "amazon_books": "amazon_books_token_overlap.json",
}

MODELS = {
    "qwen25_0_5b": "qwen25_0_5b_scores_50neg_multillm.jsonl",
    "qwen25_1_5b": "qwen25_1_5b_scores_50neg_multillm.jsonl",
    "qwen25_3b": "qwen25_3b_scores_50neg_multillm.jsonl",
}

DATASET_SCORE_DIRS = {
    "movielens1m": "movielens1m",
    "lastfm1k": "lastfm1k",
    "bookcrossing": "bookcrossing",
    "amazon_books": "amazon_books",
}


def _load_config(name: str) -> dict[str, Any]:
    return json.loads((CONFIG_DIR / name).read_text(encoding="utf-8"))


def _normalise_methods(config: dict[str, Any]) -> None:
    for method in config.get("methods", []):
        if method.get("kind") == "eser":
            method["stability_mode"] = "none"


def _precomputed_config(
    base: dict[str, Any],
    *,
    dataset: str,
    model_key: str,
    score_filename: str,
    max_eval_users: int | None = 2450,
) -> dict[str, Any]:
    config = copy.deepcopy(base)
    config["name"] = f"{dataset}_{model_key}_precomputed_50neg"
    config["split"]["negative_sample_size"] = 50
    if max_eval_users is not None:
        config.setdefault("data", {})["max_eval_users"] = max_eval_users
    config["scorer"] = {
        "kind": "precomputed_jsonl",
        "scores_path": f"../outputs/scores/{DATASET_SCORE_DIRS[dataset]}/{score_filename}",
    }
    _normalise_methods(config)
    return config


def main() -> None:
    written: list[str] = []

    for dataset in ["movielens1m", "lastfm1k", "bookcrossing", "amazon_books"]:
        base = _load_config(BASE_CONFIGS[dataset])
        for model_key, score_filename in MODELS.items():
            config = _precomputed_config(
                base,
                dataset=dataset,
                model_key=model_key,
                score_filename=score_filename,
            )
            path = CONFIG_DIR / f"{config['name']}.json"
            path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
            written.append(str(path.relative_to(ESER_ROOT)))

    manifest = CONFIG_DIR / "small_llm_multidataset_manifest.json"
    manifest.write_text(json.dumps({"configs": written}, indent=2) + "\n", encoding="utf-8")
    for item in written:
        print(item)


if __name__ == "__main__":
    main()
