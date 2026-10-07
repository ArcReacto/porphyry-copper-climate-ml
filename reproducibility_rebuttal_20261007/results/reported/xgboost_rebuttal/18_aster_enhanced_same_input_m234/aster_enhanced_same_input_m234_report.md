# Enhanced ASTER same-input M2/M3/M4 experiment

## 1. Purpose

This experiment tests climate residualization on the same 132-dimensional enhanced ASTER input. The sample pool, fixed state folds, downstream XGBoost, and evaluation are identical across M2/M3/M4; only the fold-local observation adjustment differs.

## 2. Data and boundary

- Samples: 1738 (158 positive, 1580 negative; exact 1:10 benchmark);
- ASTER: 132 enhanced A1-A3 features (56 ratios, 52 normalized differences, 24 alteration indices);
- Climate: 24 adjusters from the main benchmark, used only by the residualizer;
- Validation: five fixed state-grouped folds;
- LULC is intentionally excluded because the available 1,735-row table belongs to a different coordinate cohort.

## 3. Configurations

- **M2_A132_RAW_XGB**: all 132 enhanced ASTER features in their original form;
- **M3_A132_BROAD_XGB**: all 132 ASTER features residualized against climate within each training fold;
- **M4_A132_SPEARMAN_XGB**: only ASTER features whose maximum absolute training-fold Spearman correlation with the 24 climate adjusters is at least 0.30 are residualized; all other ASTER features remain raw.

Median handling, climate scaling, Spearman selection, Ridge(alpha=10) fitting, and XGBoost fitting are all training-fold-only. Climate variables never enter the downstream classifier.

## 4. Fixed folds and M4 selection

| fold | split_id | test_group | n_train | n_test | positive_train | positive_test | m4_selected_features |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | state_fold_0 | Arizona | 1100 | 638 | 100 | 58 | 60 |
| 1 | state_fold_1 | Washington | 1331 | 407 | 121 | 37 | 68 |
| 2 | state_fold_2 | Nevada | 1430 | 308 | 130 | 28 | 72 |
| 3 | state_fold_3 | California;Idaho;New Mexico;Wyoming | 1540 | 198 | 140 | 18 | 80 |
| 4 | state_fold_4 | Colorado;Montana;Oregon;Utah | 1551 | 187 | 141 | 17 | 72 |

| fold | split_id | test_group | selected | total |
| --- | --- | --- | --- | --- |
| 0 | state_fold_0 | Arizona | 60 | 132 |
| 1 | state_fold_1 | Washington | 68 | 132 |
| 2 | state_fold_2 | Nevada | 72 | 132 |
| 3 | state_fold_3 | California;Idaho;New Mexico;Wyoming | 80 | 132 |
| 4 | state_fold_4 | Colorado;Montana;Oregon;Utah | 72 | 132 |

## 5. Mean fold metrics

| model | roc_auc_mean | roc_auc_std | average_precision_mean | average_precision_std | f1_mean | top05_f1_mean | top05_ndcg_mean | top10_f1_mean | top10_ndcg_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M2_A132_RAW_XGB | 0.5952 | 0.0895 | 0.2000 | 0.0922 | 0.1899 | 0.1875 | 0.3169 | 0.2024 | 0.2642 |
| M3_A132_BROAD_XGB | 0.6257 | 0.0716 | 0.2253 | 0.0968 | 0.2308 | 0.2157 | 0.3554 | 0.2179 | 0.2869 |
| M4_A132_SPEARMAN_XGB | 0.6346 | 0.0716 | 0.2277 | 0.0937 | 0.1950 | 0.2037 | 0.3503 | 0.2025 | 0.2795 |

## 6. Pooled OOF metrics

| model | roc_auc | average_precision | f1 | top05_precision | top05_recall | top05_f1 | top05_ndcg | top10_f1 | top10_ndcg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M2_A132_RAW_XGB | 0.6005 | 0.1633 | 0.1895 | 0.2644 | 0.1456 | 0.1878 | 0.3050 | 0.1928 | 0.2419 |
| M3_A132_BROAD_XGB | 0.6310 | 0.1917 | 0.2294 | 0.3103 | 0.1709 | 0.2204 | 0.3335 | 0.2048 | 0.2517 |
| M4_A132_SPEARMAN_XGB | 0.6252 | 0.1790 | 0.1949 | 0.2874 | 0.1582 | 0.2041 | 0.3094 | 0.1867 | 0.2302 |

## 7. Paired fold differences relative to M2

| comparison | metric | m2_mean | candidate_mean | mean_difference | paired_bootstrap_ci95_low | paired_bootstrap_ci95_high | better_folds | equal_folds | worse_folds | fold_differences |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M3_A132_BROAD_XGB minus M2_A132_RAW_XGB | average_precision | 0.2000 | 0.2253 | 0.0254 | 0.0068 | 0.0564 | 5 | 0 | 0 | [0.085477, 0.012869, 0.017347, 0.002754, 0.008359] |
| M3_A132_BROAD_XGB minus M2_A132_RAW_XGB | top05_f1 | 0.1875 | 0.2157 | 0.0283 | -0.0567 | 0.1133 | 3 | 0 | 2 | [0.111111, -0.103448, 0.136364, 0.071429, -0.074074] |
| M3_A132_BROAD_XGB minus M2_A132_RAW_XGB | top05_ndcg | 0.3169 | 0.3554 | 0.0385 | -0.0462 | 0.1232 | 3 | 0 | 2 | [0.193849, -0.077984, 0.082406, 0.063621, -0.069431] |
| M4_A132_SPEARMAN_XGB minus M2_A132_RAW_XGB | average_precision | 0.2000 | 0.2277 | 0.0278 | 0.0068 | 0.0502 | 4 | 0 | 1 | [0.064532, -0.006932, 0.012295, 0.023477, 0.045476] |
| M4_A132_SPEARMAN_XGB minus M2_A132_RAW_XGB | top05_f1 | 0.1875 | 0.2037 | 0.0162 | -0.0478 | 0.0694 | 3 | 1 | 1 | [0.022222, -0.103448, 0.090909, 0.071429, 0.0] |
| M4_A132_SPEARMAN_XGB minus M2_A132_RAW_XGB | top05_ndcg | 0.3169 | 0.3503 | 0.0334 | -0.0208 | 0.0826 | 3 | 0 | 2 | [0.103476, -0.066964, 0.070088, 0.065972, -0.00581] |

The intervals above are descriptive 10,000-repeat bootstrap intervals over only five held-out folds and are not high-power significance tests.

## 8. Result summary

- M3 minus M2 mean-fold AP: +0.0254; Top-5% F1: +0.0283.
- M4 minus M2 mean-fold AP: +0.0278; Top-5% F1: +0.0162.
- These results apply to the enhanced ASTER-only observation configuration and should not be described as universal across feature modalities or spatial transfer settings.
