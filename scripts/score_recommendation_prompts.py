#!/usr/bin/env python3
"""Score exported recommendation prompts with a local Hugging Face causal LM."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


FLOAT_RE = re.compile(r"([01](?:\.\d+)?)")
TERMINAL = -1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=8192,
        help="Cap prompt tokens before generation to avoid rare long-context OOM spikes.",
    )
    parser.add_argument("--use-chat-template", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true", help="Append to an existing score JSONL and skip rows already written.")
    parser.add_argument("--num-shards", type=int, default=1, help="Split the input rows into this many deterministic shards.")
    parser.add_argument("--shard-index", type=int, default=0, help="0-based shard index when num_shards > 1.")
    parser.add_argument(
        "--mode",
        choices=["constrained_generation", "free_generation"],
        default="constrained_generation",
        help="Use constrained numeric decoding by default to avoid free-form explanations.",
    )
    parser.add_argument(
        "--score-step",
        type=float,
        default=0.05,
        help="Discrete score grid spacing used in constrained generation mode.",
    )
    parser.add_argument(
        "--torch-dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
        help="Override model dtype selection.",
    )
    return parser.parse_args()


def _count_existing_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def _iter_rows(
    path: Path,
    *,
    limit: int | None,
    num_shards: int,
    shard_index: int,
    skip_rows: int,
):
    yielded = 0
    seen_in_shard = 0
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if limit is not None and index >= limit:
                break
            if index % num_shards != shard_index:
                continue
            if seen_in_shard < skip_rows:
                seen_in_shard += 1
                continue
            yielded += 1
            yield json.loads(line)


def _count_rows(
    path: Path,
    *,
    limit: int | None,
    num_shards: int,
    shard_index: int,
    skip_rows: int,
) -> int:
    total = 0
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if limit is not None and index >= limit:
                break
            if index % num_shards != shard_index:
                continue
            total += 1
    return max(0, total - skip_rows)


def _format_prompt(tokenizer: AutoTokenizer, prompt: str, use_chat_template: bool) -> str:
    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt


def _build_score_options(step: float) -> list[str]:
    if step <= 0 or step > 1:
        raise ValueError(f"score_step must be in (0, 1], got {step}.")
    count = int(round(1.0 / step))
    if not math.isclose(count * step, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"score_step must divide 1.0 exactly, got {step}.")
    return [f"{index * step:.2f}" for index in range(count + 1)]


def _build_prefix_trie(option_token_ids: list[list[int]]) -> dict[int, dict]:
    root: dict[int, dict] = {}
    for token_ids in option_token_ids:
        node = root
        for token_id in token_ids:
            node = node.setdefault(token_id, {})
        node[TERMINAL] = {}
    return root


def _allowed_next_tokens(trie: dict[int, dict], generated_ids: list[int], eos_token_id: int) -> list[int]:
    node = trie
    for token_id in generated_ids:
        if token_id not in node:
            return [eos_token_id]
        node = node[token_id]

    allowed = [token_id for token_id in node if token_id != TERMINAL]
    if TERMINAL in node:
        allowed.append(eos_token_id)
    return allowed or [eos_token_id]


def _parse_score(text: str) -> float:
    cleaned = text.strip()
    if cleaned.startswith("{") and cleaned.endswith("}"):
        try:
            payload = json.loads(cleaned)
            return _clamp_score(float(payload["score"]))
        except Exception:
            pass
    match = FLOAT_RE.search(cleaned)
    if not match:
        raise ValueError(f"Could not parse score from model output: {text!r}")
    return _clamp_score(float(match.group(1)))


def _clamp_score(score: float) -> float:
    return max(0.0, min(1.0, score))


def _resolve_torch_dtype(dtype_name: str) -> torch.dtype:
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


def _score_batch_free_generation(
    *,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    max_new_tokens: int,
    max_input_tokens: int | None,
) -> tuple[list[float], list[str]]:
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
    decoded = tokenizer.batch_decode(generated[:, prompt_width:], skip_special_tokens=True)
    scores = [_parse_score(completion) for completion in decoded]
    return scores, decoded


def _score_batch_constrained_generation(
    *,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    score_options: list[str],
    max_new_tokens: int,
    max_input_tokens: int | None,
) -> tuple[list[float], list[str]]:
    option_token_ids = [tokenizer.encode(option, add_special_tokens=False) for option in score_options]
    trie = _build_prefix_trie(option_token_ids)

    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_tokens,
    )
    encoded = {key: value.to(model.device) for key, value in encoded.items()}
    prompt_width = encoded["input_ids"].shape[1]

    def prefix_allowed_tokens_fn(batch_id: int, input_ids: torch.Tensor) -> list[int]:
        generated_prefix = input_ids[prompt_width:].tolist()
        return _allowed_next_tokens(trie, generated_prefix, tokenizer.eos_token_id)

    with torch.no_grad():
        generated = model.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        )

    decoded = tokenizer.batch_decode(generated[:, prompt_width:], skip_special_tokens=True)
    cleaned = [text.strip() for text in decoded]
    scores = [_parse_score(text) for text in cleaned]
    return scores, cleaned


def main() -> None:
    args = parse_args()
    if args.num_shards <= 0:
        raise ValueError(f"num_shards must be positive, got {args.num_shards}.")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError(
            f"shard_index must be in [0, num_shards), got shard_index={args.shard_index}, num_shards={args.num_shards}."
        )
    input_path = Path(args.input_jsonl).resolve()
    output_path = Path(args.output_jsonl).resolve()
    model_path = Path(args.model_path).resolve()
    model_ref = str(model_path) if model_path.exists() else args.model_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing_rows = _count_existing_rows(output_path) if args.resume else 0
    if args.resume and existing_rows:
        print(f"[rec:qwen] resuming existing_rows={existing_rows}", flush=True)

    dtype = _resolve_torch_dtype(args.torch_dtype)
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

    score_options = _build_score_options(args.score_step)
    constrained_max_new_tokens = max(
        len(tokenizer.encode(option, add_special_tokens=False)) for option in score_options
    ) + 1

    print(
        f"[rec:qwen] shard={args.shard_index + 1}/{args.num_shards} "
        f"batch_size={args.batch_size} dtype={dtype} "
        f"max_input_tokens={args.max_input_tokens}",
        flush=True,
    )

    mode = "a" if args.resume and output_path.exists() else "w"
    completed = 0
    with output_path.open(mode, encoding="utf-8") as handle:
        batch: list[dict] = []
        for row in _iter_rows(
            input_path,
            limit=args.limit,
            num_shards=args.num_shards,
            shard_index=args.shard_index,
            skip_rows=existing_rows,
        ):
            batch.append(row)
            if len(batch) < args.batch_size:
                continue
            prompts = [_format_prompt(tokenizer, row["prompt"], args.use_chat_template) for row in batch]
            if args.mode == "constrained_generation":
                scores, completions = _score_batch_constrained_generation(
                    model=model,
                    tokenizer=tokenizer,
                    prompts=prompts,
                    score_options=score_options,
                    max_new_tokens=max(args.max_new_tokens, constrained_max_new_tokens),
                    max_input_tokens=args.max_input_tokens,
                )
            else:
                scores, completions = _score_batch_free_generation(
                    model=model,
                    tokenizer=tokenizer,
                    prompts=prompts,
                    max_new_tokens=args.max_new_tokens,
                    max_input_tokens=args.max_input_tokens,
                )

            for row, score, completion in zip(batch, scores, completions):
                handle.write(
                    json.dumps(
                        {
                            "split": row["split"],
                            "user_id": row["user_id"],
                            "item_id": row["item_id"],
                            "score": score,
                            "raw_completion": completion,
                        },
                        ensure_ascii=True,
                    )
                    + "\n"
                )

            completed += len(batch)
            handle.flush()
            if completed % max(args.batch_size * 50, 200) == 0:
                print(f"scored {completed} prompts", flush=True)
            batch = []

        if batch:
            prompts = [_format_prompt(tokenizer, row["prompt"], args.use_chat_template) for row in batch]
            if args.mode == "constrained_generation":
                scores, completions = _score_batch_constrained_generation(
                    model=model,
                    tokenizer=tokenizer,
                    prompts=prompts,
                    score_options=score_options,
                    max_new_tokens=max(args.max_new_tokens, constrained_max_new_tokens),
                    max_input_tokens=args.max_input_tokens,
                )
            else:
                scores, completions = _score_batch_free_generation(
                    model=model,
                    tokenizer=tokenizer,
                    prompts=prompts,
                    max_new_tokens=args.max_new_tokens,
                    max_input_tokens=args.max_input_tokens,
                )

            for row, score, completion in zip(batch, scores, completions):
                handle.write(
                    json.dumps(
                        {
                            "split": row["split"],
                            "user_id": row["user_id"],
                            "item_id": row["item_id"],
                            "score": score,
                            "raw_completion": completion,
                        },
                        ensure_ascii=True,
                    )
                    + "\n"
                )

            completed += len(batch)
            handle.flush()
            print(f"scored {completed} prompts", flush=True)

    if completed == 0:
        print("[rec:qwen] nothing to score for this shard", flush=True)


if __name__ == "__main__":
    main()
