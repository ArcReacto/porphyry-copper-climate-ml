from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.io_utils import ensure_project_dirs, load_config, output_path, read_table, write_dataframe
from src.spatial_utils import clean_lat_lon

from decoupling_utils import define_feature_roles, evaluate_splitter, make_default_splitters, make_leave_one_group_splitter, summarize_metrics


OUT_DIR = PROJECT_ROOT / "outputs" / "negative_ratio_sensitivity"
DATASET_ROOT = PROJECT_ROOT / "outputs" / "model_datasets" / "by_sample_scheme"
RATIOS = [2, 5, 10, 20]
RATIO_LABELS = {2: "ratio_1_2", 5: "ratio_1_5", 10: "ratio_1_10", 20: "ratio_1_20"}
RANDOM_STATE = 20260622


def load_script_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


stage02_samples = load_script_module(
    "stage02_samples",
    PROJECT_ROOT / "scripts" / "stage_02_sampling_and_region" / "07_build_sample_table.py",
)
stage01_align = load_script_module(
    "stage01_align",
    PROJECT_ROOT / "scripts" / "stage_01_data_alignment" / "05_spatial_align_features.py",
)
stage03_terrain = load_script_module(
    "stage03_terrain",
    PROJECT_ROOT / "scripts" / "stage_03_feature_expansion" / "11_align_terrain_geology.py",
)
stage03_climate = load_script_module(
    "stage03_climate",
    PROJECT_ROOT / "scripts" / "stage_03_feature_expansion" / "12_align_climate.py",
)
stage03_gravity = load_script_module(
    "stage03_gravity",
    PROJECT_ROOT / "scripts" / "stage_03_feature_expansion" / "13_align_cmmi_gravity_derivatives.py",
)
stage03_env = load_script_module(
    "stage03_env",
    PROJECT_ROOT / "scripts" / "stage_03_feature_expansion" / "14_make_environment_groups.py",
)
stage04_dataset = load_script_module(
    "stage04_dataset",
    PROJECT_ROOT / "scripts" / "stage_04_model_baseline" / "09_make_model_dataset.py",
)


def mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def dataset_scheme_dir(label: str) -> Path:
    return mkdir(DATASET_ROOT / label)


def normalize_object_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == "object":
            out[col] = out[col].astype("string")
    return out


def base_positive_samples(config: dict) -> tuple[pd.DataFrame, list[str]]:
    western = read_table(output_path(config, "outputs", "model_features_western_core.parquet"))
    western_states = sorted(western.loc[western["Y_label"] == 1, "state"].dropna().unique().tolist())
    samples = read_table(output_path(config, "data_intermediate", "samples_master.parquet"))
    positives = samples[(samples["Y_label"] == 1) & (samples["state"].isin(western_states))].copy()
    positives = positives.drop_duplicates(subset=["dep_id"], keep="first").reset_index(drop=True)
    return positives, western_states


def sample_hard_negatives(candidate_pool: pd.DataFrame, positives: pd.DataFrame, max_hard_ratio: int) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    chunks = []
    pos_counts = positives["state"].value_counts()
    for state, pos_count in pos_counts.items():
        n = int(pos_count * max_hard_ratio)
        state_candidates = candidate_pool[candidate_pool["state"].eq(state)].copy()
        if len(state_candidates) < n:
            print(f"Warning: hard negatives for {state}: requested {n}, available {len(state_candidates)}")
            n = len(state_candidates)
        state_candidates["_sample_order"] = rng.random(len(state_candidates))
        state_candidates = state_candidates.sort_values("_sample_order").head(n).copy()
        state_candidates["negative_rank_within_state_type"] = np.arange(1, len(state_candidates) + 1)
        chunks.append(state_candidates.drop(columns=["_sample_order"]))
    return pd.concat(chunks, ignore_index=True)


def build_union_samples(config: dict) -> tuple[pd.DataFrame, dict]:
    positives, western_states = base_positive_samples(config)
    max_ratio = max(RATIOS)
    max_hard_ratio = max_ratio // 2
    max_background_ratio = max_ratio - max_hard_ratio

    candidate_pool = read_table(output_path(config, "data_intermediate", "mrds_negative_candidate_pool.parquet"))
    candidate_pool = clean_lat_lon(candidate_pool, "latitude", "longitude")
    candidate_pool = candidate_pool[candidate_pool["state"].isin(western_states)].copy()
    hard = sample_hard_negatives(candidate_pool, positives, max_hard_ratio=max_hard_ratio)
    hard["distance_to_nearest_mrds_km"] = 0.0

    mrds = pd.read_csv(config["mines"]["mrds_full"], low_memory=False)
    mrds = clean_lat_lon(mrds, "latitude", "longitude")
    background = stage02_samples.build_background_negatives(
        positives=positives,
        mrds_points=mrds,
        shapefile_path=stage02_samples.default_state_shapefile(config),
        ratio=max_background_ratio,
        min_positive_distance_km=30.0,
        min_mrds_distance_km=10.0,
        seed=RANDOM_STATE + 1,
    )
    background["negative_rank_within_state_type"] = (
        background.groupby("state").cumcount() + 1
    )

    positives = positives.copy()
    positives["distance_to_nearest_positive_km"] = 0.0
    positives["distance_to_nearest_mrds_km"] = 0.0
    positives["negative_rank_within_state_type"] = 0

    positives_samples = positives
    hard_samples = stage02_samples.add_sample_fields(hard, "negative", 0, "mrds_hard_negative")
    background_samples = stage02_samples.add_sample_fields(
        background,
        "negative",
        0,
        "background_candidate_negative",
    )

    all_columns = list(
        dict.fromkeys(list(positives_samples.columns) + list(hard_samples.columns) + list(background_samples.columns))
    )
    samples = pd.concat(
        [
            positives_samples.reindex(columns=all_columns),
            hard_samples.reindex(columns=all_columns),
            background_samples.reindex(columns=all_columns),
        ],
        ignore_index=True,
    )
    samples = normalize_object_columns(samples)

    summary = {
        "western_states": western_states,
        "positive_rows": int(len(positives_samples)),
        "max_ratio": max_ratio,
        "max_hard_ratio": max_hard_ratio,
        "max_background_ratio": max_background_ratio,
        "union_rows": int(len(samples)),
        "negative_type_counts": samples["negative_type"].value_counts(dropna=False).to_dict(),
    }
    return samples, summary


def select_ratio_samples(union_samples: pd.DataFrame, ratio: int) -> pd.DataFrame:
    hard_ratio = ratio // 2
    background_ratio = ratio - hard_ratio
    positives = union_samples[union_samples["Y_label"] == 1].copy()
    selected = [positives]

    pos_counts = positives["state"].value_counts()
    for negative_type, per_positive in [
        ("mrds_hard_negative", hard_ratio),
        ("background_candidate_negative", background_ratio),
    ]:
        neg = union_samples[union_samples["negative_type"].eq(negative_type)].copy()
        chunks = []
        for state, pos_count in pos_counts.items():
            n = int(pos_count * per_positive)
            part = neg[neg["state"].eq(state)].sort_values("negative_rank_within_state_type").head(n)
            chunks.append(part)
        selected.append(pd.concat(chunks, ignore_index=True))

    out = pd.concat(selected, ignore_index=True)
    out["ratio_label"] = RATIO_LABELS[ratio]
    return normalize_object_columns(out)


def align_base_geochem_gravity(samples: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, list[dict]]:
    radii_km = [float(x) for x in config["targets"]["radii_km"]]
    elements = config["targets"]["elements"]
    features = clean_lat_lon(samples, "latitude", "longitude")
    logs: list[dict] = []

    geochem2_cols = []
    for element in elements:
        geochem2_cols.extend([f"{element}_ppm", f"{element}_pct", f"{element}_sq_ppm"])
    features, log = stage01_align.attach_source(
        features,
        output_path(config, "data_intermediate", "geochem2_nure_clean.parquet"),
        "geochem2_nure",
        geochem2_cols,
        radii_km,
        "scripts/stage_01_data_alignment/02_prepare_geochem2_nure.py",
        False,
    )
    logs.append(log)

    geochem1_path = output_path(config, "data_intermediate", "geochem1_selected_elements_wide.parquet")
    geochem1 = read_table(geochem1_path)
    rename = {
        f"geochem1_{element}_value": f"{element}_value"
        for element in elements
        if f"geochem1_{element}_value" in geochem1.columns
    }
    geochem1 = geochem1.rename(columns=rename)
    temp_path = output_path(config, "data_intermediate", "_geochem1_selected_for_align.parquet")
    write_dataframe(geochem1, temp_path)
    features, log = stage01_align.attach_source(
        features,
        temp_path,
        "geochem1_usgs",
        list(rename.values()),
        radii_km,
        "scripts/stage_01_data_alignment/03_prepare_geochem1_usgs.py",
        False,
    )
    logs.append(log)

    features, log = stage01_align.attach_source(
        features,
        output_path(config, "data_intermediate", "gravity_north_america.parquet"),
        "gravity_na",
        ["grav_anom"],
        radii_km,
        "scripts/stage_01_data_alignment/04_prepare_gravity.py",
        False,
    )
    logs.append(log)
    return features, logs


def add_terrain_geology(features: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, list[dict]]:
    radii_km = [float(x) for x in config["targets"]["radii_km"]]
    logs = []
    features, log = stage03_terrain.add_terrain_features(features, Path(config["dem"]["hgt_root"]))
    logs.append(log)
    features, log = stage03_terrain.add_fault_features(features, Path(config["geology"]["faults_uscanada_shp"]), radii_km)
    logs.append(log)
    features, log = stage03_terrain.add_geology_features(features, Path(config["geology"]["geology_conus_shp"]))
    logs.append(log)
    return features, logs


def add_climate(features: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, list[dict]]:
    climate_dir = Path(config["climate"]["terraclimate_climatology_dir"])
    files = stage03_climate.climate_files(climate_dir)
    variables = sorted(files)
    targets = features[["latitude", "longitude"]].copy()
    monthly_parts = []
    logs = []
    for var_name in variables:
        monthly, log = stage03_climate.extract_monthly_values(files[var_name], var_name, targets)
        monthly_parts.append(monthly)
        logs.append(log)
    climate = pd.concat(monthly_parts, axis=1)
    enhanced = pd.concat([features, climate], axis=1)
    for var_name in variables:
        mode = "sum" if var_name in stage03_climate.SUM_VARS else "mean"
        stage03_climate.add_summary_features(enhanced, var_name, mode)
    stage03_climate.add_derived_features(enhanced)
    enhanced["climate_has_terraclimate"] = 1
    return enhanced, logs


def add_cmmi_gravity(features: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, list[dict]]:
    cmmi = config["gravity"]["cmmi_derivatives"]
    radii_km = [float(x) for x in config["targets"]["radii_km"]]
    targets = features[["latitude", "longitude"]].copy()
    parts = []
    logs = []
    rasters = [
        ("hgm", Path(cmmi["hgm_tif"]), Path(cmmi["hgm_tfw"])),
        ("up30km", Path(cmmi["up30km_tif"]), Path(cmmi["up30km_tfw"])),
        ("up30km_hgm", Path(cmmi["up30km_hgm_tif"]), Path(cmmi["up30km_hgm_tfw"])),
    ]
    for name, tif_path, tfw_path in rasters:
        part, log = stage03_gravity.raster_values_at_points(tif_path, tfw_path, targets, name)
        parts.append(part)
        logs.append(log)
    for kind, shp_key in [("shallow", "shallow_worms_shp"), ("deep", "deep_worms_shp")]:
        worms = stage03_gravity.read_worm_points(Path(cmmi[shp_key]), kind)
        worms = stage03_gravity.bbox_filter(worms, targets)
        part, log = stage03_gravity.add_worm_point_features(targets, worms, kind, radii_km)
        parts.append(part)
        logs.append(log)
    return pd.concat([features, *parts], axis=1), logs


def add_environment(features: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, dict]:
    western = read_table(output_path(config, "outputs", "model_features_western_core.parquet"))
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
    stage03_env.require_columns(western, required, "western_core")
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
    return stage03_env.add_environment_columns(features, thresholds), thresholds


def build_union_aligned_features(config: dict) -> tuple[pd.DataFrame, dict]:
    samples, sample_summary = build_union_samples(config)
    union_dir = mkdir(OUT_DIR / "union_ratio_1_20")
    union_dataset_dir = dataset_scheme_dir("union_ratio_1_20")
    write_dataframe(samples, union_dir / "samples_union_ratio_1_20.parquet")
    write_dataframe(samples, union_dir / "samples_union_ratio_1_20.csv")
    write_dataframe(samples, union_dataset_dir / "samples_union_ratio_1_20.parquet")
    write_dataframe(samples, union_dataset_dir / "samples_union_ratio_1_20.csv")

    logs: dict[str, object] = {"samples": sample_summary}
    features, logs_base = align_base_geochem_gravity(samples, config)
    logs["base_geochem_gravity"] = logs_base
    features, logs_terrain = add_terrain_geology(features, config)
    logs["terrain_geology"] = logs_terrain
    features, logs_climate = add_climate(features, config)
    logs["climate"] = logs_climate
    features, logs_cmmi = add_cmmi_gravity(features, config)
    logs["cmmi_gravity"] = logs_cmmi
    features, thresholds = add_environment(features, config)
    logs["environment_thresholds"] = thresholds

    write_dataframe(features, union_dir / "model_features_union_ratio_1_20.parquet")
    write_dataframe(features, union_dir / "model_features_union_ratio_1_20.csv")
    write_dataframe(features, union_dataset_dir / "model_features_union_ratio_1_20.parquet")
    write_dataframe(features, union_dataset_dir / "model_features_union_ratio_1_20.csv")
    (union_dir / "alignment_summary.json").write_text(json.dumps(logs, ensure_ascii=False, indent=2), encoding="utf-8")
    (union_dataset_dir / "alignment_summary.json").write_text(
        json.dumps(logs, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return features, logs


def subset_aligned_features(union_features: pd.DataFrame, ratio: int) -> pd.DataFrame:
    sample_ids = set(select_ratio_samples(union_features, ratio)["sample_id"].astype(str))
    out = union_features[union_features["sample_id"].astype(str).isin(sample_ids)].copy()
    out["ratio_label"] = RATIO_LABELS[ratio]
    return out.reset_index(drop=True)


def build_model_dataset(features: pd.DataFrame, ratio: int) -> tuple[pd.DataFrame, dict]:
    feature_cols, info = stage04_dataset.select_feature_columns(
        features,
        stage04_dataset.FEATURE_SETS["all_features"],
        missing_threshold=0.70,
    )
    meta_cols = [c for c in stage04_dataset.META_COLUMNS + ["ratio_label"] if c in features.columns]
    out = features[meta_cols + feature_cols].copy()
    label = RATIO_LABELS[ratio]
    ratio_dir = mkdir(OUT_DIR / label)
    scheme_dir = dataset_scheme_dir(label)
    dataset_base = f"model_dataset_western_core_{label}_all_features_v1"
    for target_dir in [ratio_dir, scheme_dir]:
        write_dataframe(out, target_dir / f"{dataset_base}.parquet")
        write_dataframe(out, target_dir / f"{dataset_base}.csv")
        (target_dir / f"{dataset_base}_features.txt").write_text(
            "\n".join(feature_cols) + "\n",
            encoding="utf-8",
        )
    summary = {
        "ratio": f"1:{ratio}",
        "rows": int(len(out)),
        "columns": int(out.shape[1]),
        "feature_columns": int(len(feature_cols)),
        "label_counts": out["Y_label"].value_counts(dropna=False).sort_index().to_dict(),
        "negative_type_counts": out["negative_type"].value_counts(dropna=False).to_dict(),
        "feature_selection": info,
        "standard_dataset_dir": str(scheme_dir),
    }
    (ratio_dir / "dataset_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (scheme_dir / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return out, summary


def run_model_comparison(dataset: pd.DataFrame, ratio: int) -> dict:
    ratio_dir = mkdir(OUT_DIR / RATIO_LABELS[ratio])
    role_table = define_feature_roles(dataset)
    all_metrics = []
    all_predictions = []
    for cv_name, splitter, groups in make_default_splitters(dataset):
        print(f"Ratio 1:{ratio} - running {cv_name}")
        metrics, predictions = evaluate_splitter(
            df=dataset,
            role_table=role_table,
            splitter=splitter,
            cv_name=cv_name,
            groups=groups,
        )
        all_metrics.append(metrics)
        all_predictions.append(predictions)

    for group_col, cv_name in [
        ("state", "leave_one_state_out"),
        ("env_causal_group_id", "leave_one_environment_group_out"),
    ]:
        if group_col not in dataset.columns:
            continue
        splitter, groups = make_leave_one_group_splitter(dataset[group_col])
        print(f"Ratio 1:{ratio} - running {cv_name}")
        metrics, predictions = evaluate_splitter(
            df=dataset,
            role_table=role_table,
            splitter=splitter,
            cv_name=cv_name,
            groups=groups,
        )
        all_metrics.append(metrics)
        all_predictions.append(predictions)

    metrics = pd.concat(all_metrics, ignore_index=True)
    predictions = pd.concat(all_predictions, ignore_index=True)
    summary = summarize_metrics(metrics)
    group_summary = summarize_metrics(metrics, group_cols=["cv", "test_group", "model"])

    metrics.to_csv(ratio_dir / "negative_ratio_model_metrics.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(ratio_dir / "negative_ratio_model_predictions.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(ratio_dir / "negative_ratio_model_summary.csv", index=False, encoding="utf-8-sig")
    group_summary.to_csv(ratio_dir / "negative_ratio_group_summary.csv", index=False, encoding="utf-8-sig")
    return {"summary": summary, "group_summary": group_summary}


def compact(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        c
        for c in [
            "ratio",
            "cv",
            "model",
            "roc_auc_mean",
            "average_precision_mean",
            "balanced_accuracy_mean",
            "precision_mean",
            "recall_mean",
            "f1_mean",
        ]
        if c in df.columns
    ]
    out = df[cols].copy()
    for col in out.columns:
        if col.endswith("_mean"):
            out[col] = out[col].round(4)
    return out


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No data._"
    headers = [str(c) for c in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]).replace("|", "\\|") for c in df.columns) + " |")
    return "\n".join(lines)


def write_overall_report(dataset_summaries: list[dict], summaries: list[pd.DataFrame]) -> None:
    combined = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    combined.to_csv(OUT_DIR / "negative_ratio_sensitivity_summary.csv", index=False, encoding="utf-8-sig")
    DATASET_ROOT.mkdir(parents=True, exist_ok=True)
    combined.to_csv(DATASET_ROOT / "negative_ratio_sensitivity_summary.csv", index=False, encoding="utf-8-sig")

    dataset_rows = pd.DataFrame(
        [
            {
                "ratio": item["ratio"],
                "rows": item["rows"],
                "positive": item["label_counts"].get(1, item["label_counts"].get("1", 0)),
                "negative": item["label_counts"].get(0, item["label_counts"].get("0", 0)),
                "features": item["feature_columns"],
            }
            for item in dataset_summaries
        ]
    )
    report = f"""# 负样本比例敏感性实验

## 1. 实验目的

比较当前气候解耦模型在不同负样本比例下的稳定性。本实验不覆盖原始 1:2 主数据集，所有输出均保存在独立目录：

```text
outputs/negative_ratio_sensitivity
```

标准化归档数据集同时保存到：

```text
outputs/model_datasets/by_sample_scheme
```

比较比例：

```text
1:5
1:10
1:20
```

其中 1:5 作为从当前 1:2 到更高负样本比例之间的过渡比例。

## 2. 数据集规模

{markdown_table(dataset_rows)}

## 3. 模型结果汇总

{markdown_table(compact(combined))}

## 4. 输出目录

```text
outputs/negative_ratio_sensitivity/ratio_1_5
outputs/negative_ratio_sensitivity/ratio_1_10
outputs/negative_ratio_sensitivity/ratio_1_20

outputs/model_datasets/by_sample_scheme/ratio_1_5
outputs/model_datasets/by_sample_scheme/ratio_1_10
outputs/model_datasets/by_sample_scheme/ratio_1_20
```
"""
    (OUT_DIR / "negative_ratio_sensitivity_report.md").write_text(report, encoding="utf-8")
    (DATASET_ROOT / "negative_ratio_sensitivity_report.md").write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build sample-scheme datasets and optionally run negative-ratio model sensitivity experiments."
    )
    parser.add_argument(
        "--ratios",
        nargs="+",
        type=int,
        choices=sorted(RATIO_LABELS),
        default=RATIOS,
        help="Negative ratios to build. Defaults to 2 5 10 20.",
    )
    parser.add_argument(
        "--build-only",
        action="store_true",
        help="Only build and archive datasets; skip the legacy M1-M4 model comparison.",
    )
    return parser.parse_args()


def main() -> None:
    global RATIOS
    args = parse_args()
    RATIOS = sorted(args.ratios)

    config = load_config()
    ensure_project_dirs(config)
    mkdir(OUT_DIR)
    mkdir(DATASET_ROOT)

    union_features, logs = build_union_aligned_features(config)
    dataset_summaries = []
    model_summaries = []
    for ratio in RATIOS:
        print(f"Building ratio 1:{ratio} dataset")
        features = subset_aligned_features(union_features, ratio)
        ratio_dir = mkdir(OUT_DIR / RATIO_LABELS[ratio])
        scheme_dir = dataset_scheme_dir(RATIO_LABELS[ratio])
        write_dataframe(features, ratio_dir / f"model_features_western_core_{RATIO_LABELS[ratio]}.parquet")
        write_dataframe(features, ratio_dir / f"model_features_western_core_{RATIO_LABELS[ratio]}.csv")
        write_dataframe(features, scheme_dir / f"model_features_western_core_{RATIO_LABELS[ratio]}.parquet")
        write_dataframe(features, scheme_dir / f"model_features_western_core_{RATIO_LABELS[ratio]}.csv")
        dataset, dataset_summary = build_model_dataset(features, ratio)
        dataset_summaries.append(dataset_summary)
        if not args.build_only:
            model_result = run_model_comparison(dataset, ratio)
            summary = model_result["summary"].copy()
            summary.insert(0, "ratio", f"1:{ratio}")
            model_summaries.append(summary)

    (OUT_DIR / "build_alignment_summary.json").write_text(json.dumps(logs, ensure_ascii=False, indent=2), encoding="utf-8")
    (DATASET_ROOT / "build_alignment_summary.json").write_text(
        json.dumps(logs, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_overall_report(dataset_summaries, model_summaries)
    if args.build_only:
        print("Built and archived sample-scheme datasets")
        print(DATASET_ROOT)
    else:
        print("Wrote negative ratio sensitivity report")
    print(OUT_DIR / "negative_ratio_sensitivity_report.md")


if __name__ == "__main__":
    main()
