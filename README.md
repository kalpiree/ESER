# Beyond Ranking: Risk-Controlled Selective Exposure

This repository contains the ESER experiment code for **Beyond Ranking:
Risk-Controlled Selective Exposure for LLM-Based Recommenders**.

ESER is a post-scoring calibration method for recommendation exposure. It does
not retrain or modify the underlying recommender. Instead, it takes fixed
user-item relevance scores from an LLM-based or conventional scorer and selects
the final displayed recommendation set under reliability and fairness
constraints.

## What ESER Controls

ESER calibrates group-specific exposure thresholds to control:

- `BER`: bad-exposure rate, or the rate at which irrelevant items are displayed;
- `MOR`: missed-opportunity rate, or the rate at which relevant items are not
  displayed;
- `Delta_E`: combined exposure disparity from utility and burden differences.

The paper default is:

```text
alpha = 0.15
beta = 0.15
eta = 0.08
delta_hat = 0.05
```

## Repository Layout

```text
src/recommendation_exposure/        Core data, metrics, scorer, and method code
scripts/                            Dataset, scoring, experiment, and IHPC helpers
experiments/recommendation_eser/    Configs and paper-facing experiment notes
```

Large data, model checkpoints, prompt dumps, score dumps, logs, and generated
outputs are intentionally not versioned.

## Install

From this directory:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

The core calibration code uses the Python standard library. Optional LLM scoring
requires the model-specific dependencies used on IHPC, such as `torch` and
`transformers`.

## Quick Demo

```bash
python scripts/run_recommendation_experiment.py \
  --config experiments/recommendation_eser/configs/demo_token_overlap.json
```

The demo uses a tiny synthetic dataset and a lightweight token-overlap scorer,
so it is only a plumbing check. Paper runs use prepared benchmark datasets and
precomputed scorer outputs.

## Paper Experiment Setup

The paper-facing setup uses:

- datasets: MovieLens-1M, Amazon Books, LastFM, and Book-Crossing;
- candidate sets: one held-out positive plus 50 sampled negatives;
- main scorer: Qwen2.5-3B-Instruct;
- scorer robustness: Qwen2.5-0.5B, Qwen2.5-1.5B, Qwen2.5-3B, and SASRec;
- baselines: Top-k, Top-k-Match, Global-Thr., Group-Thr., Coverage-Set,
  User-Fair, Item-Fair, Two-Sided-Fair, LTT, ESER w/o SC, and ESER.

See `experiments/recommendation_eser/DESIGN.md` for the research-question
mapping and `experiments/recommendation_eser/PAPER_ALIGNMENT.md` for the local
result files used by each table or figure.

## LLM Score Replay

The typical LLM workflow is:

```bash
python scripts/export_recommendation_prompts.py \
  --config experiments/recommendation_eser/configs/movielens1m_qwen25_3b_precomputed_50neg.json \
  --output experiments/recommendation_eser/outputs/movielens1m_prompts.jsonl
```

Score the prompts on IHPC, then replay the resulting score JSONL:

```bash
python scripts/run_recommendation_experiment.py \
  --config experiments/recommendation_eser/configs/movielens1m_qwen25_3b_precomputed_50neg.json
```

The prompt template excludes group labels. Group information is used only during
calibration and evaluation of exposure constraints.

## Notes For Release

Before publishing, keep only code, configs, documentation, and small demo data.
Do not include:

- `/models/`,
- `/experiments/recommendation_eser/data/raw/`,
- `/experiments/recommendation_eser/data/prepared/`,
- `/experiments/recommendation_eser/outputs/`,
- `/logs/`,
- credential files,
- hand-edited result drafts.

## IHPC Notes

For cluster runs, copy `configs/ihpc.env.example` to `configs/ihpc.env` and set
the remote ESER directory. The local `configs/ihpc.env` file is ignored so
account-specific paths and host settings are not committed.

Example detached run:

```bash
IHPC_EXEC_HOST=dsmi1 bash scripts/ihpc_run_recommendation_experiment_detached.sh \
  experiments/recommendation_eser/configs/demo_token_overlap.json
```
