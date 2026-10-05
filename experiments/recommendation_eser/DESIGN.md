# Recommendation Experiment Design

## Goal

Reproduce the recommendation experiments for **Beyond Ranking:
Risk-Controlled Selective Exposure for LLM-Based Recommenders** in a clean,
standalone codebase.

ESER is evaluated as a calibration layer. The scorer produces a pointwise score
for each user-item candidate pair, and ESER decides which scored items should be
shown in the final exposure set.

## Paper-Facing Setup

- Datasets:
  - MovieLens-1M,
  - Amazon Books,
  - LastFM,
  - Book-Crossing.
- Primary group definitions:
  - MovieLens-1M: gender,
  - Amazon Books: cold-start status,
  - LastFM: region,
  - Book-Crossing: age.
- Secondary group definitions:
  - activity groups for robustness checks.
- Candidate construction:
  - chronological leave-one-out evaluation,
  - one positive item plus 50 sampled negatives per user,
  - five seeds for paper-facing aggregation.
- Main scorer:
  - Qwen2.5-3B-Instruct.
- Cross-scorer robustness:
  - Qwen2.5-0.5B-Instruct,
  - Qwen2.5-1.5B-Instruct,
  - Qwen2.5-3B-Instruct,
  - SASRec as a conventional scorer.
- Motivation analysis scorers:
  - BPR-MF,
  - NeuMF,
  - LightGCN,
  - SASRec,
  - Qwen2.5 model variants.

## Methods

The comparison includes:

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

The component analysis also uses:

- `ESER w/o Opp.`,
- `ESER w/o Fair.`,
- `ESER-Global`.

## Metrics

- `BER`: bad-exposure rate,
- `MOR`: missed-opportunity rate,
- `Size`: average number of displayed items,
- `HR`: hit rate of the exposed set,
- `NDCG`: ranking-aware utility over the exposed set,
- `U-Diff`: group utility disparity,
- `B-Diff`: group burden disparity,
- `Delta_E`: combined exposure disparity.

The default control parameters are:

```text
alpha = 0.15
beta = 0.15
eta = 0.08
delta_hat = 0.05
```

## Research Question Mapping

- RQ1: main performance and exposure-level comparison across datasets.
- RQ2: scorer robustness and secondary group robustness.
- RQ3: calibration-size and stability-correction behavior.
- RQ4: component ablation.
- RQ5: parameter sensitivity and efficiency.

## Implementation Status

- Dataset preparation helpers are implemented for the target datasets.
- LLM prompt export, scoring, shard merge, and score replay are implemented.
- Threshold-based, local-threshold, and fairness baseline policies are
  implemented in `src/recommendation_exposure/methods.py`.
- Table collection scripts are in `scripts/`.
- Large data, model checkpoints, prompt dumps, score dumps, logs, and transient
  cache files are intentionally excluded from the clean repository.
