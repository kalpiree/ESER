# Experiments

Run commands from the repository root. The [reproduction guide](../../docs/reproduce.md)
covers data preparation, scoring, calibration, and result collection.

## Dataset Format

Prepared datasets contain three CSV files:

| File | Required columns | Additional columns |
| --- | --- | --- |
| `users.csv` | `user_id`, configured group column | Secondary group column |
| `items.csv` | `item_id` | `title`, `category`, `description`, `tags` |
| `interactions.csv` | `user_id`, `item_id`, `timestamp` | Dataset-specific metadata |

| Dataset | Domain | Primary group | Secondary group |
| --- | --- | --- | --- |
| MovieLens-1M | Movies | Gender | Activity |
| Amazon Books | Books | Cold-start status | Activity |
| LastFM-1K | Music | Region | Activity |
| Book-Crossing | Books | Age | Activity |

Book-Crossing has no interaction timestamps in the source files. Its preparation
script uses row order for splitting.

## Configuration

Data and score paths in a configuration are resolved relative to its JSON file.

| Field | Purpose |
| --- | --- |
| `data.group_column` | User attribute used for group-specific calibration |
| `data.secondary_group_column` | Alternative user grouping |
| `data.min_history` | Minimum history length for evaluation |
| `data.max_eval_users` | Optional evaluation-user cap |
| `data.case_seed_name` | Shared sampling identity across configurations |
| `split.calibration_positive_count` | Held-out calibration positives per user |
| `split.test_positive_count` | Held-out test positives per user |
| `split.negative_sample_size` | Sampled negatives per candidate set |
| `split.seed` | Sampling seed |
| `scorer` | Scoring model or precomputed score file |
| `methods` | Exposure policies and their parameters |

The `*_precomputed_50neg.json` configurations use one held-out positive and
50 sampled negatives per set. Each configuration runs one seed. To evaluate
another seed, regenerate its candidate sets and corresponding scores.

## Exposure Methods

| Method kind | Selection rule |
| --- | --- |
| `top_k` | Highest-scoring `k` items |
| `top_k_matched` | Top-k with size matched to a reference method |
| `global_threshold` | One calibrated threshold across groups |
| `group_threshold` | Separate calibrated thresholds per group |
| `coverage_set` | Thresholds selected for relevant-item inclusion |
| `user_fair` | Group thresholds selected using utility disparity |
| `item_fair` | Reranking with an item-exposure penalty |
| `two_sided_fair` | Reranking with item-exposure and group-utility penalties |
| `ltt` | Per-user threshold based on score mean and standard deviation |
| `eser` | Joint reliability, opportunity, and disparity constraints |

The fairness policies operate on scorer outputs. `stability_mode` controls the
ESER stability correction; `none` disables it. The supplied configurations
specify this setting explicitly.

## Metrics

The experiment runner reports BER, MOR, mean displayed-set size, hit rate,
NDCG, utility disparity, burden disparity, and combined exposure disparity
(`delta_e`). `summary.json` includes both aggregate and group-level metrics,
fit metadata, and constraint satisfaction for calibration and test splits.
