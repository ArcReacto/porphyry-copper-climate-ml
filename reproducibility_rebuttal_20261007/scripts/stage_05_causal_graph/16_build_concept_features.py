from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.io_utils import ensure_project_dirs, load_config, output_path, read_table, write_dataframe, write_json


MODEL_DATASET = "model_datasets/model_dataset_western_core_all_features_v1.parquet"
MAPPING_RESOLVED = "causal_graph/feature_to_concept_resolved.csv"
CAUSAL_GRAPH_DIR = "causal_graph"
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
BINARY_CONCEPTS = {
    "carbonate_geology",
    "felsic_intermediate_geology",
    "igneous_geology",
    "intrusive_geology",
    "mafic_geology",
    "metamorphic_geology",
    "sedimentary_geology",
    "unconsolidated_cover",
}
ROBUST_Z_CLIP = 8.0
MAX_SCORE_ROLES = {
    "mineralization_signal",
    "pathfinder_signal",
    "geochemical_background",
    "structural_control",
    "geophysical_source_density",
    "geophysical_source_strength",
}


def robust_z(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").astype(float)
    median = numeric.median(skipna=True)
    q25 = numeric.quantile(0.25)
    q75 = numeric.quantile(0.75)
    iqr = q75 - q25
    if pd.isna(iqr) or abs(iqr) < 1e-12:
        std = numeric.std(skipna=True)
        if pd.isna(std) or abs(std) < 1e-12:
            z = numeric * 0.0
        else:
            z = (numeric - numeric.mean(skipna=True)) / std
    else:
        z = (numeric - median) / iqr
    return z.clip(lower=-ROBUST_Z_CLIP, upper=ROBUST_Z_CLIP)


def should_reverse_distance(concept: str, feature_name: str) -> bool:
    return concept.endswith("_proximity") or feature_name.endswith("_nearest_distance_km")


def aggregate_concept(df: pd.DataFrame, rows: pd.DataFrame) -> tuple[pd.Series, dict]:
    concept = rows["concept"].iloc[0]
    concept_group = rows["concept_group"].iloc[0]
    role = rows["role"].iloc[0]
    features = [c for c in rows["feature_name"].tolist() if c in df.columns]
    if not features:
        return pd.Series(np.nan, index=df.index), {
            "concept": concept,
            "concept_group": concept_group,
            "role": role,
            "feature_count": 0,
            "aggregation_method": "missing",
            "missing_rate": 1.0,
        }

    values = df[features].apply(pd.to_numeric, errors="coerce")

    if concept in BINARY_CONCEPTS:
        score = values.max(axis=1, skipna=True)
        method = "binary_max"
    else:
        transformed_cols = {}
        for feature in features:
            s = values[feature]
            if should_reverse_distance(concept, feature):
                s = -s
            transformed_cols[feature] = robust_z(s)
        transformed = pd.DataFrame(transformed_cols, index=df.index)

        if role in MAX_SCORE_ROLES:
            score = transformed.max(axis=1, skipna=True)
            method = "robust_z_row_max"
        else:
            score = transformed.median(axis=1, skipna=True)
            method = "robust_z_row_median"

    return score.rename(f"concept_{concept}"), {
        "concept": concept,
        "concept_group": concept_group,
        "role": role,
        "feature_count": len(features),
        "aggregation_method": method,
        "missing_rate": float(score.isna().mean()),
        "source_features": ";".join(features),
    }


def main() -> int:
    config = load_config()
    ensure_project_dirs(config)

    dataset_path = output_path(config, "outputs", MODEL_DATASET)
    mapping_path = output_path(config, "outputs", MAPPING_RESOLVED)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Missing model dataset: {dataset_path}")
    if not mapping_path.exists():
        raise FileNotFoundError(f"Missing concept mapping: {mapping_path}. Run scripts/stage_05_causal_graph/15_create_feature_concept_mapping.py first.")

    df = read_table(dataset_path)
    mapping = read_table(mapping_path)
    mapping = mapping[(mapping["mapped"] == True) & (mapping["include_in_concept_features"] == True)].copy()  # noqa: E712

    meta_cols = [c for c in META_COLUMNS if c in df.columns]
    out = df[meta_cols].copy()
    dictionary_rows = []
    for concept, rows in mapping.groupby("concept", sort=True):
        score, item = aggregate_concept(df, rows)
        out[f"concept_{concept}"] = score
        dictionary_rows.append(item)

    concept_cols = [c for c in out.columns if c.startswith("concept_")]
    missing = (
        out[concept_cols]
        .isna()
        .mean()
        .rename("missing_rate")
        .reset_index()
        .rename(columns={"index": "concept_feature"})
        .sort_values("missing_rate", ascending=False)
    )
    dictionary = pd.DataFrame(dictionary_rows).sort_values(["concept_group", "concept"])

    output_dir = output_path(config, "outputs", CAUSAL_GRAPH_DIR)
    parquet_path = output_dir / "concept_features_western_core.parquet"
    csv_path = output_dir / "concept_features_western_core.csv"
    dictionary_path = output_dir / "concept_feature_dictionary.csv"
    missing_path = output_dir / "concept_feature_missing_rates.csv"

    write_dataframe(out, parquet_path)
    write_dataframe(out, csv_path)
    write_dataframe(dictionary, dictionary_path)
    write_dataframe(missing, missing_path)

    summary = {
        "input_dataset": str(dataset_path),
        "input_mapping": str(mapping_path),
        "rows": int(len(out)),
        "columns": int(len(out.columns)),
        "meta_columns": int(len(meta_cols)),
        "concept_features": int(len(concept_cols)),
        "concept_groups": dictionary["concept_group"].value_counts().to_dict(),
        "max_missing_rate": float(missing["missing_rate"].max()) if len(missing) else 0.0,
        "outputs": {
            "parquet": str(parquet_path),
            "csv": str(csv_path),
            "dictionary": str(dictionary_path),
            "missing_rates": str(missing_path),
        },
    }
    if "Y_label" in out.columns:
        summary["label_counts"] = out["Y_label"].value_counts(dropna=False).sort_index().to_dict()
    if "env_causal_group" in out.columns:
        summary["environment_counts"] = out["env_causal_group"].value_counts(dropna=False).to_dict()

    write_json(output_path(config, "logs", "16_build_concept_features_summary.json"), summary)
    print(f"Wrote concept feature table: {len(out)} rows, {len(concept_cols)} concept features")
    print(parquet_path)
    print(csv_path)
    print(dictionary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
