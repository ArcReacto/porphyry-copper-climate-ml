from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.io_utils import ensure_project_dirs, load_config, output_path, read_table, write_dataframe, write_json


FEATURE_TABLES = {
    "samples": "model_features_samples.parquet",
    "western_core": "model_features_western_core.parquet",
    "southwest_core": "model_features_southwest_core.parquet",
}
ENV_PREFIX = "env_"
GROUP_ORDER = {
    "arid_basin_or_range": 1,
    "semi_arid_transition": 2,
    "subhumid_humid_mountain": 3,
    "snow_influenced_mountain": 4,
    "unknown": 99,
}


def require_columns(df: pd.DataFrame, columns: list[str], table_name: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"{table_name} is missing required columns: {missing}")


def aridity_unep(ai: float) -> str:
    if pd.isna(ai):
        return "unknown"
    if ai < 0.05:
        return "hyper_arid"
    if ai < 0.20:
        return "arid"
    if ai < 0.50:
        return "semi_arid"
    if ai < 0.65:
        return "dry_subhumid"
    return "humid"


def aridity_three_class(ai: float) -> str:
    if pd.isna(ai):
        return "unknown"
    if ai < 0.20:
        return "arid"
    if ai < 0.50:
        return "semi_arid"
    return "subhumid_humid"


def add_environment_columns(df: pd.DataFrame, thresholds: dict[str, float]) -> pd.DataFrame:
    out = df.copy()
    out = out.drop(columns=[c for c in out.columns if c.startswith(ENV_PREFIX)], errors="ignore")

    ai = pd.to_numeric(out["climate_ppt_pet_ratio_annual"], errors="coerce")
    ppt = pd.to_numeric(out["climate_ppt_annual_sum"], errors="coerce")
    pet = pd.to_numeric(out["climate_pet_annual_sum"], errors="coerce")
    deficit = pd.to_numeric(out["climate_def_annual_sum"], errors="coerce")
    tmean = pd.to_numeric(out["climate_tmean_annual_mean"], errors="coerce")
    swe = pd.to_numeric(out["climate_swe_annual_mean"], errors="coerce")
    relief = pd.to_numeric(out["terrain_relief_1000m_m"], errors="coerce")
    slope = pd.to_numeric(out["terrain_slope_deg"], errors="coerce")

    out["env_aridity_index_ppt_pet"] = ai
    out["env_climate_ppt_annual_mm"] = ppt
    out["env_climate_pet_annual_mm"] = pet
    out["env_climate_deficit_annual_mm"] = deficit
    out["env_climate_tmean_annual_c"] = tmean
    out["env_climate_swe_annual_mean_mm"] = swe
    out["env_aridity_class_unep"] = ai.map(aridity_unep)
    out["env_aridity_class_3"] = ai.map(aridity_three_class)
    out["env_snow_influence"] = np.where(
        (swe >= thresholds["snow_swe_mm"]) | (tmean <= thresholds["cold_tmean_c"]),
        "snow_influenced",
        "low_snow",
    )

    high_relief = (relief >= thresholds["relief_1000m_q75"]) | (slope >= thresholds["slope_q75"])
    low_relief = (relief <= thresholds["relief_1000m_q25"]) & (slope <= thresholds["slope_q25"])
    out["env_relief_class"] = np.select(
        [high_relief, low_relief],
        ["high_relief", "low_relief"],
        default="moderate_relief",
    )

    moisture_stress = np.select(
        [
            (ai < 0.20) | (deficit >= thresholds["deficit_q75"]),
            (ai < 0.50) | (deficit >= thresholds["deficit_q50"]),
        ],
        ["high_water_deficit", "moderate_water_deficit"],
        default="low_water_deficit",
    )
    out["env_water_deficit_class"] = moisture_stress

    causal_group = []
    for _, row in out.iterrows():
        if row["env_snow_influence"] == "snow_influenced" and row["env_relief_class"] == "high_relief":
            causal_group.append("snow_influenced_mountain")
        elif row["env_aridity_class_3"] == "arid":
            causal_group.append("arid_basin_or_range")
        elif row["env_aridity_class_3"] == "semi_arid":
            causal_group.append("semi_arid_transition")
        elif row["env_aridity_class_3"] == "subhumid_humid":
            causal_group.append("subhumid_humid_mountain")
        else:
            causal_group.append("unknown")

    out["env_causal_group"] = causal_group
    out["env_causal_group_id"] = out["env_causal_group"].map(GROUP_ORDER).fillna(GROUP_ORDER["unknown"]).astype(int)
    out["env_weathering_regime"] = (
        out["env_aridity_class_3"].astype(str)
        + "__"
        + out["env_water_deficit_class"].astype(str)
        + "__"
        + out["env_relief_class"].astype(str)
    )
    return out


def summarize_groups(df: pd.DataFrame, name: str) -> dict:
    summary: dict[str, object] = {"table": name, "rows": int(len(df))}
    for col in [
        "env_aridity_class_unep",
        "env_aridity_class_3",
        "env_snow_influence",
        "env_relief_class",
        "env_water_deficit_class",
        "env_causal_group",
    ]:
        summary[col] = df[col].value_counts(dropna=False).to_dict()

    if "Y_label" in df.columns:
        cross = pd.crosstab(df["env_causal_group"], df["Y_label"], dropna=False)
        summary["env_causal_group_by_label"] = {
            str(index): {str(col): int(value) for col, value in row.items()} for index, row in cross.iterrows()
        }
    return summary


def main() -> int:
    config = load_config()
    ensure_project_dirs(config)

    western = read_table(output_path(config, "outputs", FEATURE_TABLES["western_core"]))
    required = [
        "climate_ppt_annual_sum",
        "climate_pet_annual_sum",
        "climate_def_annual_sum",
        "climate_ppt_pet_ratio_annual",
        "climate_tmean_annual_mean",
        "climate_swe_annual_mean",
        "terrain_relief_1000m_m",
        "terrain_slope_deg",
    ]
    require_columns(western, required, "western_core")

    thresholds = {
        "snow_swe_mm": 20.0,
        "cold_tmean_c": 5.0,
        "relief_1000m_q25": float(western["terrain_relief_1000m_m"].quantile(0.25)),
        "relief_1000m_q75": float(western["terrain_relief_1000m_m"].quantile(0.75)),
        "slope_q25": float(western["terrain_slope_deg"].quantile(0.25)),
        "slope_q75": float(western["terrain_slope_deg"].quantile(0.75)),
        "deficit_q50": float(western["climate_def_annual_sum"].quantile(0.50)),
        "deficit_q75": float(western["climate_def_annual_sum"].quantile(0.75)),
    }

    summary = {"thresholds_source": "western_core", "thresholds": thresholds, "tables": []}
    for name, filename in FEATURE_TABLES.items():
        path = output_path(config, "outputs", filename)
        df = read_table(path)
        require_columns(df, required, name)
        out = add_environment_columns(df, thresholds)
        write_dataframe(out, path)
        write_dataframe(out, path.with_suffix(".csv"))
        item = summarize_groups(out, name)
        summary["tables"].append(item)
        print(f"Wrote environment groups for {name}: {len(out)} rows")
        print(f"  causal groups: {item['env_causal_group']}")

    write_json(output_path(config, "logs", "14_make_environment_groups_summary.json"), summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
