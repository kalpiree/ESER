# Reproducing Experiments

Run the following commands from the repository root after completing the setup
in the [README](../README.md).

## Prepare a Dataset

```bash
python scripts/download_recommendation_assets.py download-dataset movielens1m
python scripts/prepare_recommendation_dataset.py --dataset movielens1m
```

Supported dataset identifiers are `movielens1m`, `amazon_books`, `lastfm1k`,
and `bookcrossing`. Raw files are stored in
`experiments/recommendation_eser/data/raw/`; preparation writes `users.csv`,
`items.csv`, and `interactions.csv` under the corresponding `data/prepared/`
directory.

## Download a Scorer

```bash
python scripts/download_recommendation_assets.py download-model qwen25_3b_instruct
```

Use `qwen25_0_5b_instruct` or `qwen25_1_5b_instruct` for the smaller models.
The [asset manifest](../experiments/recommendation_eser/assets_manifest.json)
specifies download sources and model directories.

## Export Candidate Prompts

```bash
python scripts/export_recommendation_prompts.py \
  --config experiments/recommendation_eser/configs/movielens1m_qwen25_3b_precomputed_50neg.json \
  --split both \
  --output experiments/recommendation_eser/outputs/prompts/movielens1m_qwen25_3b.jsonl
```

The export includes calibration and test candidates. The prompt contains the
user's item history and candidate metadata; group labels are used only during
calibration and evaluation.

## Score Candidates

```bash
python scripts/score_recommendation_prompts.py \
  --model-path models/recommendation/qwen25_3b_instruct \
  --input-jsonl experiments/recommendation_eser/outputs/prompts/movielens1m_qwen25_3b.jsonl \
  --output-jsonl experiments/recommendation_eser/outputs/scores/movielens1m/qwen25_3b_scores_50neg_multillm.jsonl \
  --batch-size 4 \
  --max-new-tokens 6 \
  --use-chat-template
```

Adjust `--batch-size` to available GPU memory. To continue an interrupted run,
use `--resume` with the same input ordering, model, output file, and shard
settings. For parallel scoring, `--num-shards` and `--shard-index` partition
the prompt file; each process needs a separate output file.

Score files use JSONL rows with `split`, `user_id`, `item_id`, and `score`:

```json
{"split":"test","user_id":"1","item_id":"42","score":0.75}
```

## Calibrate and Evaluate

```bash
python scripts/run_recommendation_experiment.py \
  --config experiments/recommendation_eser/configs/movielens1m_qwen25_3b_precomputed_50neg.json
```

For the full fairness comparison on the same scores:

```bash
python scripts/run_recommendation_experiment.py \
  --config experiments/recommendation_eser/configs/rq_mainfair_movielens1m_qwen25_3b_50neg.json
```

The runner fits policies using calibration cases and evaluates them on the test
cases. Output directories default to
`experiments/recommendation_eser/outputs/<configuration-name>/`.

To use a score file at a different location:

```bash
python scripts/run_recommendation_with_scores.py \
  --config experiments/recommendation_eser/configs/movielens1m_qwen25_3b_precomputed_50neg.json \
  --scores-jsonl path/to/scores.jsonl
```

## Other Datasets and Scorers

The `small_llm_multidataset_manifest.json` file lists the four-dataset,
three-model score configurations. For each configuration, export its candidates,
score them with the corresponding model, and use the configured `scores_path`.

Conventional scorer configurations are listed in
`motivation_analysis_manifest.json`. BPR-MF, NeuMF, and `sasrec_lite` use PyTorch
and train as part of the experiment run. Popularity and token-overlap scorers
do not require model downloads.

## Collect Results

After running all four configurations in `main_fairness_baselines_manifest.json`:

```bash
python scripts/collect_main_fairness_table.py
```

This writes CSV, Markdown, and LaTeX tables under
`experiments/recommendation_eser/outputs/tables/`. Each row comes from the
corresponding run's `summary.json`.

Collect aggregate rows from a selected run:

```bash
python scripts/collect_rq1_results.py \
  --experiments movielens1m_qwen25_3b_precomputed_50neg
```

The [experiment reference](../experiments/recommendation_eser/README.md)
describes the configuration fields and metrics.
