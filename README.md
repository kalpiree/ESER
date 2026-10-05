# ESER

Code for **Beyond Ranking: Risk-Controlled Selective Exposure for LLM-Based
Recommenders**.

ESER calibrates group-specific exposure thresholds from fixed recommendation
scores. It selects displayed items subject to bad-exposure rate (BER),
missed-opportunity rate (MOR), and exposure-disparity constraints.

## Repository Structure

- `src/recommendation_exposure/`: data processing, scorers, calibration, and metrics.
- `scripts/`: dataset preparation, LLM scoring, experiments, and result collection.
- `experiments/recommendation_eser/configs/`: experiment configurations.
- `experiments/recommendation_eser/data/demo/`: synthetic example data.
- `docs/reproduce.md`: dataset and experiment instructions.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

For LLM scoring and model downloads:

```bash
pip install torch transformers accelerate huggingface_hub
```

## Example

```bash
python scripts/run_recommendation_experiment.py \
  --config experiments/recommendation_eser/configs/demo_token_overlap.json
```

This example uses synthetic data and token-overlap scores. It writes fitted
thresholds and evaluation metrics to
`experiments/recommendation_eser/outputs/demo_token_overlap/`.

## Prepare Data and Models

```bash
python scripts/download_recommendation_assets.py download-dataset movielens1m
python scripts/prepare_recommendation_dataset.py --dataset movielens1m
python scripts/download_recommendation_assets.py download-model qwen25_3b_instruct
```

Dataset preparation supports MovieLens-1M, Amazon Books, LastFM-1K, and
Book-Crossing. Model download configurations are provided for
Qwen2.5-0.5B-Instruct, Qwen2.5-1.5B-Instruct, and Qwen2.5-3B-Instruct.

## Run Experiments

After generating scores using the [reproduction guide](docs/reproduce.md):

```bash
python scripts/run_recommendation_experiment.py \
  --config experiments/recommendation_eser/configs/movielens1m_qwen25_3b_precomputed_50neg.json
```

Each configuration specifies the scorer, candidate sampling, group definitions,
and exposure methods. The main control parameters are `alpha` (BER tolerance),
`beta` (MOR tolerance), `eta` (disparity tolerance), and `delta_hat` (confidence
parameter).

## Results

Each run produces:

- `summary.json`: fitted thresholds, group metrics, and constraint checks.
- `test_metrics.csv`: aggregate test metrics for each method.

Collect completed main-comparison runs into CSV, Markdown, and LaTeX tables:

```bash
python scripts/collect_main_fairness_table.py
```

See the [experiment reference](experiments/recommendation_eser/README.md) for
configuration fields, dataset groups, and metrics.
