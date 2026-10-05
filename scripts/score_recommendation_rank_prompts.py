#!/usr/bin/env python3
"""Score listwise recommendation prompts with a local Hugging Face causal LM."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


CANDIDATE_KEY_RE = re.compile(r"C\d{3}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=1536)
    parser.add_argument("--max-input-tokens", type=int, default=8192)
    parser.add_argument("--use-chat-template", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--torch-dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
    )
    return parser.parse_args()


def _resolve_torch_dtype(dtype_name: str) -> torch.dtype:
    import torch

    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float32":
        return torch.float32
    if not torch.cuda.is_available():
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _completed_cases(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            case_id = payload.get("case_id")
            if case_id:
                completed.add(str(case_id))
    return completed


def _iter_rows(
    path: Path,
    *,
    limit: int | None,
    num_shards: int,
    shard_index: int,
    completed: set[str],
):
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if limit is not None and index >= limit:
                break
            if index % num_shards != shard_index:
                continue
            payload = json.loads(line)
            if payload["case_id"] in completed:
                continue
            yield payload


def _format_prompt(tokenizer: AutoTokenizer, prompt: str, use_chat_template: bool) -> str:
    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt


def _extract_json_payload(text: str) -> object | None:
    stripped = text.strip()
    candidates = [stripped]
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        candidates.append(stripped[start : end + 1])
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def _parse_ranking(completion: str, candidate_keys: list[str]) -> list[str]:
    allowed = set(candidate_keys)
    parsed = _extract_json_payload(completion)
    ranking: list[str] = []

    if isinstance(parsed, dict):
        raw_ranking = parsed.get("ranking")
        if isinstance(raw_ranking, list):
            for value in raw_ranking:
                key = str(value).strip()
                if key in allowed and key not in ranking:
                    ranking.append(key)

    if not ranking:
        for match in CANDIDATE_KEY_RE.finditer(completion):
            key = match.group(0)
            if key in allowed and key not in ranking:
                ranking.append(key)

    for key in candidate_keys:
        if key not in ranking:
            ranking.append(key)
    return ranking


def _score_rows(row: dict, completion: str) -> list[dict]:
    candidates = row["candidates"]
    candidate_keys = [candidate["candidate_key"] for candidate in candidates]
    ranking = _parse_ranking(completion, candidate_keys)
    denom = max(1, len(ranking) - 1)
    rank_score = {key: 1.0 - (rank / denom) for rank, key in enumerate(ranking)}
    rank_by_key = {key: rank + 1 for rank, key in enumerate(ranking)}

    rows: list[dict] = []
    for candidate in candidates:
        key = candidate["candidate_key"]
        rank = rank_by_key[key]
        rows.append(
            {
                "case_id": row["case_id"],
                "split": row["split"],
                "user_id": row["user_id"],
                "item_id": candidate["item_id"],
                "score": float(rank_score.get(key, 0.0)),
                "rank": rank,
                "candidate_key": key,
                "raw_completion": completion if rank == 1 else "",
            }
        )

    # Keep output order aligned to the model's ranking for easier inspection.
    rows.sort(key=lambda payload: payload["rank"])
    return rows


def _generate_batch(
    *,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    max_new_tokens: int,
    max_input_tokens: int,
) -> list[str]:
    import torch

    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_tokens,
    )
    encoded = {key: value.to(model.device) for key, value in encoded.items()}
    with torch.no_grad():
        generated = model.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )
    prompt_width = encoded["input_ids"].shape[1]
    return tokenizer.batch_decode(generated[:, prompt_width:], skip_special_tokens=True)


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive.")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must be in [0, num_shards).")

    input_path = Path(args.input_jsonl).resolve()
    output_path = Path(args.output_jsonl).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = _completed_cases(output_path) if args.resume else set()
    if completed:
        print(f"[rec:rank-score] resuming completed_cases={len(completed)}", flush=True)

    model_path = Path(args.model_path).resolve()
    model_ref = str(model_path) if model_path.exists() else args.model_path
    dtype = _resolve_torch_dtype(args.torch_dtype)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_ref, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_ref,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    print(
        f"[rec:rank-score] shard={args.shard_index + 1}/{args.num_shards} "
        f"batch_size={args.batch_size} dtype={dtype} max_input_tokens={args.max_input_tokens}",
        flush=True,
    )

    mode = "a" if args.resume and output_path.exists() else "w"
    completed_now = 0
    with output_path.open(mode, encoding="utf-8") as handle:
        batch: list[dict] = []
        for row in _iter_rows(
            input_path,
            limit=args.limit,
            num_shards=args.num_shards,
            shard_index=args.shard_index,
            completed=completed,
        ):
            batch.append(row)
            if len(batch) < args.batch_size:
                continue
            prompts = [_format_prompt(tokenizer, item["prompt"], args.use_chat_template) for item in batch]
            completions = _generate_batch(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                max_new_tokens=args.max_new_tokens,
                max_input_tokens=args.max_input_tokens,
            )
            for item, completion in zip(batch, completions):
                for score_row in _score_rows(item, completion):
                    handle.write(json.dumps(score_row, ensure_ascii=True) + "\n")
            handle.flush()
            completed_now += len(batch)
            if completed_now % 50 == 0:
                print(f"[rec:rank-score] cases={completed_now}", flush=True)
            batch = []

        if batch:
            prompts = [_format_prompt(tokenizer, item["prompt"], args.use_chat_template) for item in batch]
            completions = _generate_batch(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                max_new_tokens=args.max_new_tokens,
                max_input_tokens=args.max_input_tokens,
            )
            for item, completion in zip(batch, completions):
                for score_row in _score_rows(item, completion):
                    handle.write(json.dumps(score_row, ensure_ascii=True) + "\n")
            handle.flush()
            completed_now += len(batch)

    print(f"[rec:rank-score] completed_cases={completed_now}", flush=True)


if __name__ == "__main__":
    main()
