# Black-Box LLM Scoring Runbook

This runbook describes the paper-facing path for using open-weight LLMs as
black-box recommendation scorers. ESER does not train or fine-tune these models;
it consumes their fixed user-item scores and calibrates the final exposure set.

## Scope

The current manuscript uses Qwen2.5 scorers:

- Qwen2.5-0.5B-Instruct,
- Qwen2.5-1.5B-Instruct,
- Qwen2.5-3B-Instruct.

The main result uses Qwen2.5-3B-Instruct. Smaller Qwen2.5 variants are used for
cross-scorer robustness. Candidate sets follow the paper setup: one held-out
positive plus 50 sampled negatives per user.

## Cache Contract

Prompt batches are JSONL files:

```text
experiments/recommendation_eser/outputs/prompts/<dataset>/<scorer>/<lane>.jsonl
```

Score batches are JSONL files:

```text
experiments/recommendation_eser/outputs/scores/<dataset>/<scorer>/<lane>.jsonl
```

The experiment runner expects the `precomputed_jsonl` format:

```json
{"split":"test","user_id":"...","item_id":"...","score":0.73}
```

Additional audit fields can be retained in the score JSONL; the runner ignores
unknown fields.

## Local Smoke Check

From the ESER repository root:

```bash
python scripts/run_recommendation_experiment.py \
  --config experiments/recommendation_eser/configs/demo_token_overlap.json
```

This verifies the calibration and evaluation pipeline without loading an LLM.

## IHPC Dataset And Model Setup

Bootstrap a dataset:

```bash
IHPC_EXEC_HOST=mimas2 bash scripts/ihpc_run_recommendation_dataset_bootstrap_detached.sh movielens1m
```

Download the main Qwen2.5 scorer:

```bash
IHPC_EXEC_HOST=dsmi1 bash scripts/ihpc_run_recommendation_model_download_detached.sh qwen25_3b_instruct
```

Download smaller robustness scorers when needed:

```bash
IHPC_EXEC_HOST=dsmi1 bash scripts/ihpc_run_recommendation_model_download_detached.sh qwen25_0_5b_instruct
IHPC_EXEC_HOST=dsmi1 bash scripts/ihpc_run_recommendation_model_download_detached.sh qwen25_1_5b_instruct
```

## Export Prompts

```bash
python scripts/export_recommendation_prompts.py \
  --config experiments/recommendation_eser/configs/movielens1m_qwen25_3b_precomputed_50neg.json \
  --output experiments/recommendation_eser/outputs/prompts/movielens1m/qwen25_3b/test.jsonl
```

Group labels are not included in the prompt. Group information is used only for
calibration and evaluation after scores have been produced.

## Score Prompts

```bash
IHPC_EXEC_HOST=dsmi1 bash scripts/ihpc_run_recommendation_scoring_detached.sh \
  models/recommendation/qwen25_3b_instruct \
  experiments/recommendation_eser/outputs/prompts/movielens1m/qwen25_3b/test.jsonl \
  experiments/recommendation_eser/outputs/scores/movielens1m/qwen25_3b/test.jsonl \
  --batch-size 8 \
  --max-new-tokens 6 \
  --use-chat-template \
  --resume
```

Use the same pattern for Qwen2.5-0.5B and Qwen2.5-1.5B by changing the model
directory and output path.

## Replay Scores Through ESER

```bash
python scripts/run_recommendation_experiment.py \
  --config experiments/recommendation_eser/configs/movielens1m_qwen25_3b_precomputed_50neg.json
```

The config points to the score JSONL. The experiment runner then fits each
baseline on the calibration split and reports metrics on the held-out test
split.

## Operational Notes

- Use 50 negatives for paper-facing runs unless a config explicitly documents a
  separate diagnostic setting.
- Keep prompt files, score files, model checkpoints, and logs out of Git.
- If a scoring job is interrupted, rerun with `--resume` so completed score rows
  are reused.
