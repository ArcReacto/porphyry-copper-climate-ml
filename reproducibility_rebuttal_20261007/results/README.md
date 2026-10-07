# Reported rebuttal results

This directory contains immutable copies of the machine-readable outputs referenced by the rebuttal. New script runs write to `../outputs/` and do not overwrite these files.

## Contents

| Directory | Contents |
|---|---|
| `reported/main_m1_m4/` | Original M1-M4 metrics, OOF predictions, summaries, M4 selected features, and workflow manifest. M3 is the partial-feature residualization/CausalAI comparison described in the rebuttal. |
| `reported/fold_local_graphs/` | State-fold and 2-degree/3-degree spatial-fold graph manifests, selected targets, edge files, and associated OOF records used by the XGBoost replay. |
| `reported/xgboost_rebuttal/` | Fixed-XGBoost results for GraphUnion, spatial blocks, negative sources, bootstrap intervals, ratio sensitivity, buffer audits, matched-random controls, controlled distortion, matched feature baselines, Top-K positive clusters, and both ASTER experiments. |
| `reported/paleoclimate_xgboost/` | Historical-climate fold metrics, OOF predictions, selected features, contrasts, report, and manifest. |
| `reported/idann_reference/` | Three-seed IDANN-inspired implementation-level comparison, including OOF predictions and training log. |

The package preserves both fold-level and OOF-level files so that mean-fold and pooled-OOF estimands can be checked separately. Result files are evidence records, not additional model inputs.

Local absolute paths in copied JSON manifests were replaced with `PACKAGE_ROOT` or `EXTERNAL_REFERENCE`. No metric, label, prediction, selected-feature value, or random seed was changed during packaging.
