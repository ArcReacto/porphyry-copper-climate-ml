from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "generalization_audit"
TARGET = "Y_label"
RANDOM_STATE = 20260622
DECISION_THRESHOLD = 0.5
DEFAULT_RF_N_ESTIMATORS = 120
DEFAULT_HGB_MAX_ITER = 80

DEFAULT_DATASETS = [
    PROJECT_ROOT
    / "outputs"
    / "model_datasets"
    / "by_sample_scheme"
    / "ratio_1_2"
    / "model_dataset_western_core_ratio_1_2_all_features_v1.parquet",
    PROJECT_ROOT
    / "outputs"
    / "model_datasets"
    / "by_sample_scheme"
    / "ratio_1_5"
    / "model_dataset_western_core_ratio_1_5_all_features_v1.parquet",
    PROJECT_ROOT
    / "outputs"
    / "model_datasets"
    / "by_sample_scheme"
    / "ratio_1_10"
    / "model_dataset_western_core_ratio_1_10_all_features_v1.parquet",
    PROJECT_ROOT
    / "outputs"
    / "model_datasets"
    / "by_sample_scheme"
    / "ratio_1_20"
    / "model_dataset_western_core_ratio_1_20_all_features_v1.parquet",
]

META_COLUMNS = {
    "sample_id",
    "Y_label",
    "sample_type",
    "negative_type",
    "state",
    "dep_id",
    "mrds_id",
    "site_name",
}

CLIMATE_PATTERNS = (
    "climate_",
    "env_climate_",
    "env_aridity",
    "env_water_deficit",
    "env_snow_influence",
    "env_weathering_regime",
    "env_causal_group",
)

LOCATION_PATTERNS = (
    "latitude",
    "longitude",
    "distance",
    "_dist",
    "dist_",
    "nearest_mrds",
    "state",
)

CORE_PREFIXES = (
    "geochem1_",
    "geochem2_",
    "gravity_",
    "terrain_",
    "fault_",
    "geology_",
)


def load_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False, encoding="utf-8-sig")


def dataset_name_from_path(path: Path) -> str:
    stem = path.stem
    if stem.startswith("model_dataset_"):
        stem = stem.replace("model_dataset_", "", 1)
    return stem


def ratio_from_dataset_name(name: str) -> str:
    match = re.search(r"ratio_1_(\d+)", name)
    return f"1:{match.group(1)}" if match else name


def numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for col in df.columns:
        if col in META_COLUMNS:
            continue
        if col == TARGET:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


def is_climate_column(col: str) -> bool:
    return col.startswith(CLIMATE_PATTERNS)


def is_location_column(col: str) -> bool:
    low = col.lower()
    return any(pattern in low for pattern in LOCATION_PATTERNS)


def is_core_column(col: str) -> bool:
    return col.startswith(CORE_PREFIXES) and not is_location_column(col) and not is_climate_column(col)


def feature_sets_for_dataset(df: pd.DataFrame) -> dict[str, list[str]]:
    numeric_cols = numeric_feature_columns(df)
    all_set = list(numeric_cols)
    no_climate = [c for c in numeric_cols if not is_climate_column(c)]
    no_location_distance = [c for c in numeric_cols if not is_location_column(c)]
    no_climate_location = [c for c in numeric_cols if not is_climate_column(c) and not is_location_column(c)]
    core = [c for c in numeric_cols if is_core_column(c)]

    out = {
        "all_features": all_set,
        "no_climate": no_climate,
        "no_location_distance": no_location_distance,
        "no_climate_no_location": no_climate_location,
        "core_geo_geochem_geophysics": core,
    }
    return {k: v for k, v in out.items() if len(v) > 0}


def concept_path_for_dataset(name: str) -> Path:
    return PROJECT_ROOT / "outputs" / "standardized_runs" / name / "03_causal_graph" / "04_concept_features.parquet"


def concept_feature_sets(name: str) -> tuple[pd.DataFrame | None, dict[str, list[str]]]:
    path = concept_path_for_dataset(name)
    if not path.exists():
        return None, {}
    df = load_table(path).reset_index(drop=True)
    concept_cols = [c for c in df.columns if c.startswith("concept_") and pd.api.types.is_numeric_dtype(df[c])]
    climate_concepts = {
        "concept_actual_evapotranspiration",
        "concept_aridity",
        "concept_diurnal_temperature_range",
        "concept_evapotranspiration_ratio",
        "concept_max_temperature",
        "concept_mean_temperature",
        "concept_min_temperature",
        "concept_potential_evapotranspiration",
        "concept_precipitation",
        "concept_runoff",
        "concept_snow_influence",
        "concept_soil_moisture",
        "concept_solar_radiation",
        "concept_vapor_pressure",
        "concept_vapor_pressure_deficit",
        "concept_water_balance",
        "concept_water_deficit",
        "concept_wind_speed",
    }
    sets = {
        "concept_all": concept_cols,
        "concept_no_climate": [c for c in concept_cols if c not in climate_concepts],
    }
    return df, {k: v for k, v in sets.items() if len(v) > 0}


def make_model(model_name: str, pos_weight: float, rf_n_estimators: int, hgb_max_iter: int) -> Pipeline:
    if model_name == "hist_gradient_boosting":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        learning_rate=0.05,
                        max_iter=hgb_max_iter,
                        max_leaf_nodes=15,
                        l2_regularization=0.1,
                        class_weight={0: 1.0, 1: pos_weight},
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )
    if model_name == "random_forest":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=rf_n_estimators,
                        min_samples_leaf=3,
                        max_features="sqrt",
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
    if model_name == "logistic_regression":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=5000,
                        class_weight="balanced",
                        solver="liblinear",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )
    raise ValueError(f"Unknown model: {model_name}")


def predict_prob(model: Pipeline, x: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    if hasattr(model[-1], "decision_function"):
        score = model.decision_function(x)
        return 1.0 / (1.0 + np.exp(-score))
    return model.predict(x).astype(float)


def safe_metric(name: str, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> float:
    if name in {"roc_auc", "average_precision"} and len(np.unique(y_true)) < 2:
        return math.nan
    if name == "roc_auc":
        return float(roc_auc_score(y_true, y_prob))
    if name == "average_precision":
        return float(average_precision_score(y_true, y_prob))
    if name == "balanced_accuracy":
        return float(balanced_accuracy_score(y_true, y_pred))
    if name == "precision":
        return float(precision_score(y_true, y_pred, zero_division=0))
    if name == "recall":
        return float(recall_score(y_true, y_pred, zero_division=0))
    if name == "f1":
        return float(f1_score(y_true, y_pred, zero_division=0))
    if name == "mcc":
        return float(matthews_corrcoef(y_true, y_pred))
    raise ValueError(name)


def spatial_block_groups(df: pd.DataFrame, block_degrees: float) -> pd.Series:
    lat_bin = np.floor(df["latitude"].astype(float) / block_degrees).astype("Int64").astype(str)
    lon_bin = np.floor(df["longitude"].astype(float) / block_degrees).astype("Int64").astype(str)
    return lat_bin + "_" + lon_bin


def splitters_for_dataset(
    df: pd.DataFrame,
    include_loso: bool,
    include_spatial: bool,
    block_degrees: float,
    max_splits: int,
) -> list[tuple[str, object, pd.Series | None]]:
    y = df[TARGET].astype(int)
    positives = int(y.sum())
    negatives = int((y == 0).sum())
    n_splits = min(max_splits, positives, negatives)
    if n_splits < 2:
        raise ValueError("Need at least two positive and negative samples.")

    splitters: list[tuple[str, object, pd.Series | None]] = [
        ("stratified_kfold", StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE), None)
    ]
    if "state" in df.columns and df["state"].nunique(dropna=True) >= 2:
        state_groups = df["state"].fillna("unknown").astype(str)
        splitters.append(("groupkfold_state", GroupKFold(n_splits=min(max_splits, state_groups.nunique())), state_groups))
        if include_loso:
            splitters.append(("leave_one_state_out", LeaveOneGroupOut(), state_groups))

    if include_spatial and {"latitude", "longitude"}.issubset(df.columns):
        blocks = spatial_block_groups(df, block_degrees)
        if blocks.nunique(dropna=True) >= 2:
            splitters.append(("spatial_block_kfold", GroupKFold(n_splits=min(max_splits, blocks.nunique())), blocks))
    return splitters


def evaluate_feature_set(
    df: pd.DataFrame,
    dataset_name: str,
    feature_set: str,
    feature_cols: list[str],
    models: list[str],
    include_loso: bool,
    include_spatial: bool,
    block_degrees: float,
    max_splits: int,
    rf_n_estimators: int,
    hgb_max_iter: int,
) -> tuple[list[dict], list[dict]]:
    y = df[TARGET].astype(int).reset_index(drop=True)
    x = df[feature_cols].reset_index(drop=True)
    pos = int(y.sum())
    neg = int((y == 0).sum())
    pos_weight = neg / max(pos, 1)
    rows: list[dict] = []
    pred_rows: list[dict] = []

    for cv_name, splitter, groups in splitters_for_dataset(
        df, include_loso, include_spatial, block_degrees, max_splits
    ):
        split_iter = splitter.split(x, y, groups) if groups is not None else splitter.split(x, y)
        for fold, (train_idx, test_idx) in enumerate(split_iter, start=1):
            train_idx = np.asarray(train_idx)
            test_idx = np.asarray(test_idx)
            if y.iloc[train_idx].nunique() < 2:
                continue
            test_group = None
            if groups is not None:
                test_group = ";".join(sorted(pd.Series(groups).iloc[test_idx].astype(str).unique().tolist()))
            for model_name in models:
                model = make_model(model_name, pos_weight, rf_n_estimators, hgb_max_iter)
                model.fit(x.iloc[train_idx], y.iloc[train_idx])
                y_prob = predict_prob(model, x.iloc[test_idx])
                y_pred = (y_prob >= DECISION_THRESHOLD).astype(int)
                tn, fp, fn, tp = confusion_matrix(y.iloc[test_idx], y_pred, labels=[0, 1]).ravel()
                metric_row = {
                    "dataset": dataset_name,
                    "ratio": ratio_from_dataset_name(dataset_name),
                    "feature_set": feature_set,
                    "model": model_name,
                    "cv": cv_name,
                    "fold": int(fold),
                    "test_group": test_group,
                    "n_features": int(len(feature_cols)),
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                    "positive_test": int(y.iloc[test_idx].sum()),
                    "negative_test": int((y.iloc[test_idx] == 0).sum()),
                    "tn": int(tn),
                    "fp": int(fp),
                    "fn": int(fn),
                    "tp": int(tp),
                }
                for metric in [
                    "roc_auc",
                    "average_precision",
                    "balanced_accuracy",
                    "precision",
                    "recall",
                    "f1",
                    "mcc",
                ]:
                    metric_row[metric] = safe_metric(metric, y.iloc[test_idx].to_numpy(), y_pred, y_prob)
                rows.append(metric_row)
                for idx, true_value, pred_value, prob_value in zip(test_idx, y.iloc[test_idx], y_pred, y_prob):
                    pred_rows.append(
                        {
                            "dataset": dataset_name,
                            "feature_set": feature_set,
                            "model": model_name,
                            "cv": cv_name,
                            "fold": int(fold),
                            "test_group": test_group,
                            "row_index": int(idx),
                            "sample_id": df.iloc[idx].get("sample_id", ""),
                            "state": df.iloc[idx].get("state", ""),
                            "negative_type": df.iloc[idx].get("negative_type", ""),
                            "Y_label": int(true_value),
                            "y_pred": int(pred_value),
                            "y_prob": float(prob_value),
                        }
                    )
    return rows, pred_rows


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "roc_auc",
        "average_precision",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "mcc",
    ]
    summary = (
        metrics.groupby(["dataset", "ratio", "feature_set", "model", "cv"], dropna=False)[metric_cols]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary.columns = [
        "_".join([str(part) for part in col if str(part)])
        if isinstance(col, tuple)
        else str(col)
        for col in summary.columns
    ]
    return summary


def compact_summary(summary: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "ratio",
        "feature_set",
        "model",
        "cv",
        "roc_auc_mean",
        "average_precision_mean",
        "balanced_accuracy_mean",
        "precision_mean",
        "recall_mean",
        "f1_mean",
        "mcc_mean",
    ]
    out = summary[[c for c in cols if c in summary.columns]].copy()
    for col in out.columns:
        if col.endswith("_mean"):
            out[col] = out[col].round(4)
    return out


def markdown_table(df: pd.DataFrame, max_rows: int = 80) -> str:
    if df.empty:
        return "_No data._"
    display = df.head(max_rows).copy()
    lines = [
        "| " + " | ".join(str(c) for c in display.columns) + " |",
        "| " + " | ".join(["---"] * len(display.columns)) + " |",
    ]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(str(row[c]).replace("|", "\\|") for c in display.columns) + " |")
    return "\n".join(lines)


def plot_metric(summary: pd.DataFrame, out_dir: Path, metric: str, cv_name: str) -> None:
    sub = summary[summary["cv"] == cv_name].copy()
    if sub.empty:
        return
    sub["ratio_order"] = sub["ratio"].str.extract(r"1:(\d+)").astype(float)
    sub = sub.sort_values(["ratio_order", "feature_set", "model"])
    plt.figure(figsize=(13, 6))
    sns.barplot(
        data=sub,
        x="ratio",
        y=f"{metric}_mean",
        hue="feature_set",
        errorbar=None,
    )
    plt.title(f"{metric} by feature set ({cv_name})")
    plt.xlabel("Sample ratio")
    plt.ylabel(metric)
    plt.xticks(rotation=0)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / f"plot_{metric}_{cv_name}.png", dpi=200)
    plt.close()


def write_report(
    out_dir: Path,
    datasets: list[Path],
    summary: pd.DataFrame,
    feature_manifest: pd.DataFrame,
    models: list[str],
    include_loso: bool,
    include_spatial: bool,
) -> None:
    group = summary[summary["cv"] == "groupkfold_state"].copy()
    spatial = summary[summary["cv"] == "spatial_block_kfold"].copy()

    best_group = pd.DataFrame()
    if not group.empty:
        best_group = group.loc[group.groupby(["dataset", "feature_set"])["f1_mean"].idxmax()].copy()
        best_group = compact_summary(best_group.sort_values(["ratio", "feature_set"]))

    all_features = group[group["feature_set"] == "all_features"].copy()
    core_compare = group[group["feature_set"].isin(["all_features", "no_climate_no_location", "core_geo_geochem_geophysics"])].copy()
    core_compare = compact_summary(core_compare.sort_values(["ratio", "feature_set", "model"]))

    report = f"""# Generalization Audit Report

## Purpose

This audit checks whether the current models mainly learn stable exploration signals or rely on region-specific background information.

It compares:

- strict validation protocols: stratified K-fold, state-grouped K-fold, spatial-block K-fold{", leave-one-state-out" if include_loso else ""}
- feature ablations: all features, no climate, no location/distance, no climate plus no location, core geology/geochemistry/geophysics features, and concept features when available
- models: {", ".join(models)}

## Input Datasets

```text
{chr(10).join(str(p) for p in datasets)}
```

## Feature Set Sizes

{markdown_table(feature_manifest)}

## Best groupkfold_state Result Per Dataset And Feature Set

{markdown_table(best_group, max_rows=120)}

## Core Comparison Under groupkfold_state

{markdown_table(core_compare, max_rows=160)}

## How To Read This Report

- `stratified_kfold` is the optimistic same-distribution reference.
- `groupkfold_state` checks whether the model generalizes across state groups.
- `spatial_block_kfold` checks whether performance survives spatially grouped validation.
- If all-features performance is much higher than no-location or core-feature performance, the model may rely on regional memory.
- If no-climate or no-climate-no-location stays competitive, the signal is more likely to be geologically stable.

## Output Files

```text
generalization_audit_metrics.csv
generalization_audit_predictions.csv
generalization_audit_summary.csv
generalization_feature_sets.csv
plot_f1_groupkfold_state.png
plot_average_precision_groupkfold_state.png
```
"""
    (out_dir / "generalization_audit_report.md").write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run strict generalization and de-regionalization audit.")
    parser.add_argument("--datasets", nargs="+", default=[str(p) for p in DEFAULT_DATASETS])
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--models",
        nargs="+",
        default=["hist_gradient_boosting", "random_forest"],
        choices=["hist_gradient_boosting", "random_forest", "logistic_regression"],
    )
    parser.add_argument("--include-loso", action="store_true", help="Also run leave-one-state-out validation.")
    parser.add_argument("--no-spatial", action="store_true", help="Disable spatial block validation.")
    parser.add_argument("--spatial-block-degrees", type=float, default=4.0)
    parser.add_argument("--max-splits", type=int, default=3, help="Maximum folds for K-fold style validators.")
    parser.add_argument("--rf-n-estimators", type=int, default=DEFAULT_RF_N_ESTIMATORS)
    parser.add_argument("--hgb-max-iter", type=int, default=DEFAULT_HGB_MAX_ITER)
    parser.add_argument(
        "--feature-sets",
        nargs="+",
        default=[
            "all_features",
            "no_climate",
            "no_location_distance",
            "no_climate_no_location",
            "core_geo_geochem_geophysics",
            "concept_all",
            "concept_no_climate",
        ],
        help="Feature sets to evaluate.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_rows: list[dict] = []
    pred_rows: list[dict] = []
    feature_manifest_rows: list[dict] = []
    dataset_paths = [Path(p) for p in args.datasets]

    for dataset_path in dataset_paths:
        dataset_name = dataset_name_from_path(dataset_path)
        df = load_table(dataset_path).reset_index(drop=True)
        if TARGET not in df.columns:
            raise ValueError(f"{dataset_path} does not contain {TARGET}.")

        raw_sets = feature_sets_for_dataset(df)
        for feature_set, cols in raw_sets.items():
            if feature_set not in args.feature_sets:
                continue
            feature_manifest_rows.append(
                {
                    "dataset": dataset_name,
                    "ratio": ratio_from_dataset_name(dataset_name),
                    "feature_set": feature_set,
                    "source_table": "model_dataset",
                    "n_features": len(cols),
                }
            )
            rows, preds = evaluate_feature_set(
                df=df,
                dataset_name=dataset_name,
                feature_set=feature_set,
                feature_cols=cols,
                models=args.models,
                include_loso=args.include_loso,
                include_spatial=not args.no_spatial,
                block_degrees=args.spatial_block_degrees,
                max_splits=args.max_splits,
                rf_n_estimators=args.rf_n_estimators,
                hgb_max_iter=args.hgb_max_iter,
            )
            metrics_rows.extend(rows)
            pred_rows.extend(preds)

        concept_df, concept_sets = concept_feature_sets(dataset_name)
        if concept_df is not None:
            for feature_set, cols in concept_sets.items():
                if feature_set not in args.feature_sets:
                    continue
                feature_manifest_rows.append(
                    {
                        "dataset": dataset_name,
                        "ratio": ratio_from_dataset_name(dataset_name),
                        "feature_set": feature_set,
                        "source_table": "concept_features",
                        "n_features": len(cols),
                    }
                )
                rows, preds = evaluate_feature_set(
                    df=concept_df,
                    dataset_name=dataset_name,
                    feature_set=feature_set,
                    feature_cols=cols,
                    models=args.models,
                    include_loso=args.include_loso,
                    include_spatial=not args.no_spatial,
                    block_degrees=args.spatial_block_degrees,
                    max_splits=args.max_splits,
                    rf_n_estimators=args.rf_n_estimators,
                    hgb_max_iter=args.hgb_max_iter,
                )
                metrics_rows.extend(rows)
                pred_rows.extend(preds)

        print(f"Finished {dataset_name}")

    metrics = pd.DataFrame(metrics_rows)
    predictions = pd.DataFrame(pred_rows)
    feature_manifest = pd.DataFrame(feature_manifest_rows)
    summary = summarize_metrics(metrics)
    write_table(metrics, out_dir / "generalization_audit_metrics.csv")
    write_table(predictions, out_dir / "generalization_audit_predictions.csv")
    write_table(summary, out_dir / "generalization_audit_summary.csv")
    write_table(feature_manifest, out_dir / "generalization_feature_sets.csv")

    plot_metric(summary, out_dir, "f1", "groupkfold_state")
    plot_metric(summary, out_dir, "average_precision", "groupkfold_state")
    if not args.no_spatial:
        plot_metric(summary, out_dir, "f1", "spatial_block_kfold")
        plot_metric(summary, out_dir, "average_precision", "spatial_block_kfold")

    manifest = {
        "datasets": [str(p) for p in dataset_paths],
        "output_dir": str(out_dir),
        "models": args.models,
        "feature_sets": args.feature_sets,
        "include_loso": bool(args.include_loso),
        "include_spatial": not bool(args.no_spatial),
        "spatial_block_degrees": float(args.spatial_block_degrees),
        "max_splits": int(args.max_splits),
        "rf_n_estimators": int(args.rf_n_estimators),
        "hgb_max_iter": int(args.hgb_max_iter),
        "metric_rows": int(len(metrics)),
    }
    (out_dir / "generalization_audit_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(
        out_dir=out_dir,
        datasets=dataset_paths,
        summary=summary,
        feature_manifest=feature_manifest,
        models=args.models,
        include_loso=args.include_loso,
        include_spatial=not args.no_spatial,
    )
    print(f"Wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
