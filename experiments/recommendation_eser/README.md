# Recommendation ESER Experiments

This directory contains the experiment workspace for **Beyond Ranking:
Risk-Controlled Selective Exposure for LLM-Based Recommenders**.

ESER is a post-scoring exposure method: it consumes fixed candidate scores from
an LLM or conventional recommender and calibrates the final displayed set with
group-specific exposure thresholds.

## What Is Included

- `configs/`: runnable JSON configs for dataset lanes, ablations, robustness,
  sensitivity, and timing.
- `data/demo/`: tiny synthetic data for plumbing checks.
- `outputs/`: default local output location; large outputs are ignored by Git.
- `DESIGN.md`: paper-facing experiment design and RQ mapping.
- `PAPER_ALIGNMENT.md`: mapping from paper tables/figures to local result files.

## Data Contract

Each prepared dataset uses three CSV files:

1. `users.csv`
   - required columns: `user_id`, the configured user-group column
2. `items.csv`
   - required columns: `item_id`
   - recommended metadata columns: `title`, `category`, `description`, `tags`
3. `interactions.csv`
   - required columns: `user_id`, `item_id`, `timestamp`

The splitter follows the paper setup:

- chronological leave-one-out evaluation per user,
- a shared calibration split and test split across methods,
- one held-out positive plus 50 sampled negatives per user,
- five random seeds for paper-facing aggregation.

## Scorers

The paper-facing scorer families are:

- Qwen2.5-0.5B-Instruct,
- Qwen2.5-1.5B-Instruct,
- Qwen2.5-3B-Instruct,
- SASRec as the conventional scorer reference.

Local synthetic and lightweight scorers are still available for fast checks:

- `token_overlap`,
- `popularity`,
- `random`,
- `precomputed_jsonl`.

The normal LLM path is:

1. export recommendation prompts,
2. score prompts on IHPC,
3. merge score shards if needed,
4. rerun the experiment with `precomputed_jsonl`.

## Methods

The implemented method set matches the paper comparisons:

- `Top-k`,
- `Top-k-Match`,
- `Global-Thr.`,
- `Group-Thr.`,
- `Coverage-Set`,
- `User-Fair`,
- `Item-Fair`,
- `Two-Sided-Fair`,
- `LTT`,
- `ESER w/o SC`,
- `ESER`.

The ablation configs additionally cover `ESER w/o Opp.`, `ESER w/o Fair.`, and
`ESER-Global` where those variants are needed for the component analysis.

## Local Quick Start

Run the demo experiment:

```bash
python scripts/run_recommendation_experiment.py \
  --config experiments/recommendation_eser/configs/demo_token_overlap.json
```

Export prompt JSONL for an LLM scorer:

```bash
python scripts/export_recommendation_prompts.py \
  --config experiments/recommendation_eser/configs/demo_token_overlap.json \
  --output experiments/recommendation_eser/outputs/demo_prompts.jsonl
```

Run from precomputed scores:

```bash
python scripts/run_recommendation_experiment.py \
  --config experiments/recommendation_eser/configs/movielens1m_qwen25_3b_precomputed_50neg.json
```

## IHPC Examples

Detached dataset bootstrap:

```bash
IHPC_EXEC_HOST=mimas2 bash scripts/ihpc_run_recommendation_dataset_bootstrap_detached.sh movielens1m
IHPC_EXEC_HOST=mimas2 bash scripts/ihpc_run_recommendation_dataset_bootstrap_detached.sh lastfm1k
```

Detached model download:

```bash
IHPC_EXEC_HOST=dsmi1 bash scripts/ihpc_run_recommendation_model_download_detached.sh qwen25_3b_instruct
```

Detached LLM scoring:

```bash
IHPC_EXEC_HOST=dsmi1 bash scripts/ihpc_run_recommendation_scoring_detached.sh \
  models/recommendation/qwen25_3b_instruct \
  experiments/recommendation_eser/outputs/movielens_prompts.jsonl \
  experiments/recommendation_eser/outputs/scores/movielens1m/qwen25_3b_scores.jsonl \
  --batch-size 8 \
  --use-chat-template
```

## Dataset Assumptions

- MovieLens-1M uses gender as the primary group and activity as the secondary
  group.
- Amazon Books uses cold-start status as the primary group and activity as the
  secondary group.
- LastFM uses region as the primary group and activity as the secondary group.
- Book-Crossing uses age as the primary group and activity as the secondary
  group.

Book-Crossing does not provide reliable timestamps in the raw dump, so the
current pipeline falls back to observed row order for chronological splitting.
