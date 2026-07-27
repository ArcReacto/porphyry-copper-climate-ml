from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.io_utils import ensure_project_dirs, load_config, output_path, read_table, write_dataframe, write_json


SUBSETS = ["western_core", "southwest_core"]
FEATURE_SETS = {
    "all_features": [
        "geochem1_usgs_",
        "geochem2_nure_",
        "gravity_na_",
        "gravity_cmmi_",
        "terrain_",
        "fault_",
        "geology_",
        "climate_",
    ],
}
META_COLUMNS = [
    "sample_id",
    "Y_label",
    "sample_type",
    "negative_type",
    "state",
    "latitude",
    "longitude",
    "dep_id",
    "mrds_id",
    "site_name",
    "env_aridity_index_ppt_pet",
    "env_climate_ppt_annual_mm",
    "env_climate_pet_annual_mm",
    "env_climate_deficit_annual_mm",
    "env_climate_tmean_annual_c",
    "env_climate_swe_annual_mean_mm",
    "env_aridity_class_unep",
    "env_aridity_class_3",
    "env_snow_influence",
    "env_relief_class",
    "env_water_deficit_class",
    "env_causal_group",
    "env_causal_group_id",
    "env_weathering_regime",
]
MODEL_DATASET_DIR = "model_datasets"


def is_coverage_or_distance_column(column: str) -> bool:
    return (
        column in {"geochem1_usgs_nearest_distance_km", "geochem2_nure_nearest_distance_km", "gravity_na_nearest_distance_km"}
        or "_count_" in column
        or "_n_" in column
        or column in {"distance_to_nearest_positive_km", "distance_to_nearest_mrds_km"}
    )


def select_feature_columns(df: pd.DataFrame, prefixes: list[str], missing_threshold: float) -> tuple[list[str], dict]:
    numeric_cols = set(df.select_dtypes(include="number").columns)
    raw_feature_cols = [
        c
        for c in df.columns
        if c in numeric_cols and any(c.startswith(prefix) for prefix in prefixes) and not is_coverage_or_distance_column(c)
    ]

    missing_rate = df[raw_feature_cols].isna().mean() if raw_feature_cols else pd.Series(dtype=float)
    after_missing = [c for c in raw_feature_cols if missing_rate[c] <= missing_threshold]
    non_constant = []
    dropped_constant = []
    for col in after_missing:
        nunique = df[col].nunique(dropna=True)
        if nunique > 1:
            non_constant.append(col)
        else:
            dropped_constant.append(col)

    info = {
        "raw_feature_columns": len(raw_feature_cols),
        "dropped_by_missing_threshold": int(len(raw_feature_cols) - len(after_missing)),
        "dropped_constant": len(dropped_constant),
        "selected_feature_columns": len(non_constant),
        "missing_threshold": missing_threshold,
        "dropped_constant_columns": dropped_constant,
        "top_dropped_missing_columns": missing_rate[missing_rate > missing_threshold]
        .sort_values(ascending=False)
        .head(30)
        .to_dict(),
    }
    return non_constant, info


def build_dataset(source: pd.DataFrame, subset_name: str, feature_set_name: str, missing_threshold: float, config: dict) -> dict:
    prefixes = FEATURE_SETS[feature_set_name]
    feature_cols, info = select_feature_columns(source, prefixes, missing_threshold)
    meta_cols = [c for c in META_COLUMNS if c in source.columns]
    out = source[meta_cols + feature_cols].copy()

    parquet_path = output_path(config, "outputs", f"{MODEL_DATASET_DIR}/model_dataset_{subset_name}_{feature_set_name}_v1.parquet")
    csv_path = output_path(config, "outputs", f"{MODEL_DATASET_DIR}/model_dataset_{subset_name}_{feature_set_name}_v1.csv")
    features_path = output_path(
        config,
        "outputs",
        f"{MODEL_DATASET_DIR}/model_dataset_{subset_name}_{feature_set_name}_v1_features.txt",
    )

    write_dataframe(out, parquet_path)
    write_dataframe(out, csv_path)
    features_path.write_text("\n".join(feature_cols) + "\n", encoding="utf-8")

    return {
        "subset": subset_name,
        "feature_set": feature_set_name,
        "rows": int(len(out)),
        "columns": int(len(out.columns)),
        "meta_columns": meta_cols,
        "feature_columns": feature_cols,
        "feature_selection": info,
        "label_counts": out["Y_label"].value_counts(dropna=False).sort_index().to_dict(),
        "negative_type_counts": out["negative_type"].value_counts(dropna=False).to_dict()
        if "negative_type" in out.columns
        else {},
        "outputs": {"parquet": str(parquet_path), "csv": str(csv_path), "features_txt": str(features_path)},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Create leakage-aware baseline modeling datasets.")
    parser.add_argument(
        "--subsets",
        nargs="*",
        default=["western_core"],
        choices=SUBSETS,
        help="Analysis subsets to process. Default builds the main western_core dataset only.",
    )
    parser.add_argument(
        "--feature-sets",
        nargs="*",
        default=list(FEATURE_SETS),
        choices=list(FEATURE_SETS),
        help="Feature sets to create.",
    )
    parser.add_argument(
        "--missing-threshold",
        type=float,
        default=0.70,
        help="Drop candidate feature columns with missing rate above this value.",
    )
    args = parser.parse_args()

    config = load_config()
    ensure_project_dirs(config)

    summary = {"missing_threshold": args.missing_threshold, "datasets": []}
    for subset_name in args.subsets:
        source_path = output_path(config, "outputs", f"model_features_{subset_name}.parquet")
        if not source_path.exists():
            raise FileNotFoundError(f"Missing {source_path}. Run scripts/stage_02_sampling_and_region/08_make_analysis_subsets.py first.")
        source = read_table(source_path)
        for feature_set_name in args.feature_sets:
            item = build_dataset(source, subset_name, feature_set_name, args.missing_threshold, config)
            summary["datasets"].append(item)
            print(
                f"Wrote model dataset {subset_name}/{feature_set_name}: "
                f"{item['rows']} rows, {len(item['feature_columns'])} features"
            )
            print(item["outputs"]["parquet"])

    write_json(output_path(config, "logs", "09_make_model_dataset_summary.json"), summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
