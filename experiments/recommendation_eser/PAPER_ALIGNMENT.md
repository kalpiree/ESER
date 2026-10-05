# Paper Alignment Notes

This file maps the current manuscript to the local experiment artifacts. The
PDF manuscript is treated as the source of truth for names, research questions,
and default settings.

## Tables And Figures

| Paper item | Purpose | Local artifact |
| --- | --- | --- |
| Table 1 | Motivation scorer comparison | `results/eser_tables/motivation_analysis_*` and generated motivation CSVs |
| Table 2 | Motivation policy comparison | `results/eser_tables/motivation_analysis_*` and generated motivation CSVs |
| Table 3 | Dataset statistics | `results/eser_tables/dataset_statistics.csv`, `results/eser_tables/dataset_statistics_table.tex` |
| Figure 4 | Exposure-level motivation plots | Motivation plot CSVs under `results/eser_tables/` |
| Tables 4-5 | RQ1 main results | `results/eser_tables/main_fairness_table_rows.csv`, `results/eser_tables/rq1_main_results.csv` |
| Table 6 | RQ2 scorer robustness | `results/eser_tables/rq1_tables/multillm_recommendation_rows.csv` plus conventional-scorer rows |
| Table 7 | RQ2 secondary groups | `results/eser_tables/rq5_robustness_results.csv` |
| Table 8 | RQ3 calibration fractions | `results/eser_tables/rq3_stability_results.csv` |
| Figure 5 | RQ3 calibration-size curves | `results/eser_tables/rq3_stability_results.csv` |
| Table 9 | RQ4 ablation | `results/eser_tables/rq4_ablation_results.csv` |
| Figure 6 | RQ5 sensitivity | sensitivity sweep CSVs in the external result archive |
| Table 10 | RQ5 efficiency | `results/eser_tables/efficiency_timing_groupthr_quick.csv` |

## Naming Notes

- The current paper has RQ1-RQ5.
- `ESER w/o SC` refers to ESER without the stability correction.
- `Delta_E` is the paper's combined exposure-disparity metric.

## Result Hygiene

Planning or hand-edited result drafts are not raw measurement outputs and should
not be treated as empirical results when preparing a public release.
