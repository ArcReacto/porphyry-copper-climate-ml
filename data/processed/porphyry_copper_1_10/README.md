# Porphyry Copper 1:10 Processed Benchmark Dataset

This directory contains the processed porphyry copper benchmark dataset used by the main CD-MPM experiments.

## Sample Design

- Positive samples: known porphyry copper occurrences in the western-core United States study area.
- Hard negative samples: nearby non-target mining-area samples derived from MRDS-style mining records.
- Neutral samples: unlabeled background candidates retained for optional semi-supervised or robustness analyses.

The main supervised training file uses only positive and hard negative samples.

## Files

| File | Description |
| --- | --- |
| `samples_known_mining_neutral_ratio_1_10.csv` | Sample metadata for positive, hard negative, and neutral samples. |
| `samples_known_mining_neutral_ratio_1_10.parquet` | Parquet version of the sample metadata table. |
| `model_dataset_known_mining_neutral_ratio_1_10_supervised_all_features_v1.csv` | Main supervised feature table, excluding neutral samples. |
| `model_dataset_known_mining_neutral_ratio_1_10_supervised_all_features_v1.parquet` | Parquet version of the main supervised feature table. |
| `model_dataset_known_mining_neutral_ratio_1_10_supervised_all_features_v1_features.txt` | Feature list for the supervised feature table. |
| `model_dataset_known_mining_neutral_ratio_1_10_with_neutral_all_features_v1.csv` | Feature table retaining neutral background samples. |
| `model_dataset_known_mining_neutral_ratio_1_10_with_neutral_all_features_v1.parquet` | Parquet version of the feature table retaining neutral samples. |
| `model_dataset_known_mining_neutral_ratio_1_10_with_neutral_all_features_v1_features.txt` | Feature list for the feature table retaining neutral samples. |
| `metadata.json` | Clean public metadata for this released dataset package. |

## Label Encoding

- `Y_label = 1`: known porphyry copper positive sample.
- `Y_label = 0`: hard negative sample.
- `Y_label = -1`: neutral unlabeled background sample.

## Dataset Size

- Supervised table: 1,738 rows and 682 columns.
- With-neutral table: 3,318 rows and 682 columns.
- Feature list: 656 model feature columns.

The released benchmark is derived from aligned geochemical, geophysical, geological, structural, terrain, and climate variables. Raw source data are not included in this repository.
