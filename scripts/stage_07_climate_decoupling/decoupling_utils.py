from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_CSV = PROJECT_ROOT / "outputs" / "model_datasets" / "model_dataset_western_core_all_features_v1.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "decoupled_climate"
TARGET = "Y_label"
RANDOM_STATE = 20260622
DECISION_THRESHOLD = 0.5
CLIMATE_SENSITIVE_THRESHOLD = 0.30

BASE_META_COLUMNS = {
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
}

CLIMATE_PREFIXES = (
    "climate_",
    "env_climate_",
    "env_aridity",
    "env_water_deficit",
    "env_snow_influence",
)

ENV_GROUP_PREFIXES = (
    "env_causal_group",
    "env_weathering_regime",
    "env_relief_class",
)

GEO_STRUCTURE_PREFIXES = (
    "gravity_",
    "terrain_",
    "fault_",
    "geology_",
)


def ensure_output_dir() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def load_main_dataset() -> pd.DataFrame:
    return pd.read_csv(INPUT_CSV, low_memory=False)


def is_numeric(series: pd.Series) -> bool:
    return series.dtype.kind in "fiub"


def role_for_column(column: str, series: pd.Series) -> str:
    if column == TARGET:
        return "target"
    if column in BASE_META_COLUMNS:
        return "metadata"
    if column.startswith("geochem1_") or column.startswith("geochem2_"):
        return "geochemistry"
    if column.startswith(CLIMATE_PREFIXES):
        return "climate"
    if column.startswith(ENV_GROUP_PREFIXES):
        return "environment_group"
    if column.startswith(GEO_STRUCTURE_PREFIXES):
        return "geo_structure"
    if column.startswith("env_"):
        return "environment_group"
    if is_numeric(series):
        return "other_numeric"
    return "other_non_numeric"


def is_climate_adjuster_column(column: str, role: str, series: pd.Series) -> bool:
    if role != "climate" or not is_numeric(series):
        return False
    if column.startswith("env_climate_") or column.startswith("env_aridity"):
        return True
    if "_annual_" in column or column.endswith("_annual_sum") or column.endswith("_annual_mean"):
        return True
    if column in {
        "climate_water_balance_annual_mm",
        "climate_ppt_pet_ratio_annual",
        "climate_aet_pet_ratio_annual",
    }:
        return True
    return False


def define_feature_roles(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in df.columns:
        series = df[column]
        role = role_for_column(column, series)
        climate_adjuster = is_climate_adjuster_column(column, role, series)
        rows.append(
            {
                "column": column,
                "role": role,
                "dtype": str(series.dtype),
                "is_numeric": bool(is_numeric(series)),
                "missing_rate": float(series.isna().mean()),
                "n_unique": int(series.nunique(dropna=True)),
                "used_in_m1_full_climate": bool(
                    is_numeric(series) and role not in {"target", "metadata", "environment_group"}
                ),
                "used_in_m2_no_climate": bool(
                    is_numeric(series)
                    and role not in {"target", "metadata", "climate", "environment_group"}
                ),
                "used_in_m3_climate_normalized": bool(
                    is_numeric(series)
                    and role in {"geochemistry", "geo_structure", "other_numeric"}
                ),
                "used_as_climate_adjuster": bool(climate_adjuster),
            }
        )
    return pd.DataFrame(rows)


def columns_by_role(role_table: pd.DataFrame, role: str) -> list[str]:
    return role_table.loc[(role_table["role"] == role) & (role_table["is_numeric"]), "column"].tolist()


def feature_columns_for_model(role_table: pd.DataFrame, model_key: str) -> list[str]:
    if model_key.startswith("M4_Sensitive_Residualized"):
        flag = "used_in_m3_climate_normalized"
        return role_table.loc[role_table[flag], "column"].tolist()

    flag = {
        "M1_Full_Climate": "used_in_m1_full_climate",
        "M2_No_Climate": "used_in_m2_no_climate",
        "M3_Climate_Normalized": "used_in_m3_climate_normalized",
    }[model_key]
    return role_table.loc[role_table[flag], "column"].tolist()


def climate_adjuster_columns(role_table: pd.DataFrame) -> list[str]:
    return role_table.loc[role_table["used_as_climate_adjuster"], "column"].tolist()


def climate_sensitive_element_columns(
    role_table: pd.DataFrame,
    threshold: float = CLIMATE_SENSITIVE_THRESHOLD,
) -> list[str]:
    all_elements = set(columns_by_role(role_table, "geochemistry"))
    path = OUTPUT_DIR / "top_climate_sensitive_elements.csv"
    if not path.exists():
        return []
    sensitive = pd.read_csv(path)
    if "spearman_abs_max" not in sensitive.columns or "element_feature" not in sensitive.columns:
        return []
    selected = sensitive.loc[
        sensitive["spearman_abs_max"] >= threshold,
        "element_feature",
    ].tolist()
    return [column for column in selected if column in all_elements]


def m4_threshold_from_model_key(model_key: str) -> float:
    if model_key == "M4_Sensitive_Residualized":
        return CLIMATE_SENSITIVE_THRESHOLD
    if "_t0_" in model_key:
        suffix = model_key.rsplit("_t0_", 1)[1]
        return float(f"0.{suffix}")
    return CLIMATE_SENSITIVE_THRESHOLD


def make_rf_model() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=500,
                    min_samples_leaf=3,
                    max_features="sqrt",
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


@dataclass
class ClimateResidualizer:
    element_cols: list[str]
    climate_cols: list[str]
    alpha: float = 10.0

    def fit(self, x: pd.DataFrame) -> "ClimateResidualizer":
        self.element_medians_ = x[self.element_cols].median(numeric_only=True)
        y = x[self.element_cols].fillna(self.element_medians_).to_numpy()
        self.model_ = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=self.alpha)),
            ]
        )
        self.model_.fit(x[self.climate_cols], y)
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        predicted = self.model_.predict(x[self.climate_cols])
        observed = x[self.element_cols].fillna(self.element_medians_).to_numpy()
        residuals = observed - predicted
        return pd.DataFrame(
            residuals,
            index=x.index,
            columns=[f"{column}_climate_resid" for column in self.element_cols],
        )


def build_train_test_features(
    df: pd.DataFrame,
    role_table: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    model_key: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()

    if model_key in {"M1_Full_Climate", "M2_No_Climate"}:
        columns = feature_columns_for_model(role_table, model_key)
        return train_df[columns], test_df[columns], columns

    element_cols = columns_by_role(role_table, "geochemistry")
    climate_cols = climate_adjuster_columns(role_table)
    direct_cols = [
        c
        for c in feature_columns_for_model(role_table, "M3_Climate_Normalized")
        if c not in element_cols
    ]

    if model_key == "M3_Climate_Normalized":
        residual_cols = element_cols
        raw_element_cols: list[str] = []
    elif model_key.startswith("M4_Sensitive_Residualized"):
        threshold = m4_threshold_from_model_key(model_key)
        residual_cols = climate_sensitive_element_columns(role_table, threshold=threshold)
        raw_element_cols = [c for c in element_cols if c not in set(residual_cols)]
    else:
        raise ValueError(f"Unknown model_key: {model_key}")

    residualizer = ClimateResidualizer(element_cols=residual_cols, climate_cols=climate_cols)
    residualizer.fit(train_df)
    train_resid = residualizer.transform(train_df)
    test_resid = residualizer.transform(test_df)
    x_train = pd.concat([train_resid, train_df[raw_element_cols + direct_cols]], axis=1)
    x_test = pd.concat([test_resid, test_df[raw_element_cols + direct_cols]], axis=1)
    columns = list(train_resid.columns) + raw_element_cols + direct_cols
    return x_train, x_test, columns


def safe_metric(metric_name: str, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> float:
    if metric_name in {"roc_auc", "average_precision"} and len(np.unique(y_true)) < 2:
        return math.nan
    if metric_name == "roc_auc":
        return float(roc_auc_score(y_true, y_prob))
    if metric_name == "average_precision":
        return float(average_precision_score(y_true, y_prob))
    if metric_name == "balanced_accuracy":
        return float(balanced_accuracy_score(y_true, y_pred))
    if metric_name == "precision":
        return float(precision_score(y_true, y_pred, zero_division=0))
    if metric_name == "recall":
        return float(recall_score(y_true, y_pred, zero_division=0))
    if metric_name == "f1":
        return float(f1_score(y_true, y_pred, zero_division=0))
    raise ValueError(metric_name)


def evaluate_splitter(
    df: pd.DataFrame,
    role_table: pd.DataFrame,
    splitter,
    cv_name: str,
    groups: pd.Series | None = None,
    model_keys: Iterable[str] = (
        "M1_Full_Climate",
        "M2_No_Climate",
        "M3_Climate_Normalized",
        "M4_Sensitive_Residualized",
    ),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = df[TARGET].astype(int).reset_index(drop=True)
    metrics_rows = []
    prediction_rows = []
    split_iter = splitter.split(df, y, groups) if groups is not None else splitter.split(df, y)

    for fold, (train_idx, test_idx) in enumerate(split_iter, start=1):
        train_idx = np.asarray(train_idx)
        test_idx = np.asarray(test_idx)
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]
        group_label = None
        if groups is not None:
            test_groups = pd.Series(groups).iloc[test_idx].astype(str).unique().tolist()
            group_label = ";".join(sorted(test_groups))

        for model_key in model_keys:
            x_train, x_test, feature_columns = build_train_test_features(
                df=df,
                role_table=role_table,
                train_idx=train_idx,
                test_idx=test_idx,
                model_key=model_key,
            )
            model = make_rf_model()
            model.fit(x_train, y_train)
            y_prob = model.predict_proba(x_test)[:, 1]
            y_pred = (y_prob >= DECISION_THRESHOLD).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
            metrics_rows.append(
                {
                    "cv": cv_name,
                    "fold": fold,
                    "test_group": group_label,
                    "model": model_key,
                    "n_features": len(feature_columns),
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                    "positive_test": int(y_test.sum()),
                    "negative_test": int((y_test == 0).sum()),
                    "roc_auc": safe_metric("roc_auc", y_test.to_numpy(), y_pred, y_prob),
                    "average_precision": safe_metric("average_precision", y_test.to_numpy(), y_pred, y_prob),
                    "balanced_accuracy": safe_metric("balanced_accuracy", y_test.to_numpy(), y_pred, y_prob),
                    "precision": safe_metric("precision", y_test.to_numpy(), y_pred, y_prob),
                    "recall": safe_metric("recall", y_test.to_numpy(), y_pred, y_prob),
                    "f1": safe_metric("f1", y_test.to_numpy(), y_pred, y_prob),
                    "tn": int(tn),
                    "fp": int(fp),
                    "fn": int(fn),
                    "tp": int(tp),
                }
            )
            for row_index, true_value, pred_value, prob_value in zip(
                test_idx, y_test.to_numpy(), y_pred, y_prob
            ):
                prediction_rows.append(
                    {
                        "row_index": int(row_index),
                        "sample_id": df.iloc[row_index].get("sample_id", ""),
                        "cv": cv_name,
                        "fold": fold,
                        "test_group": group_label,
                        "model": model_key,
                        "Y_label": int(true_value),
                        "y_pred": int(pred_value),
                        "y_prob": float(prob_value),
                        "state": df.iloc[row_index].get("state", ""),
                        "env_causal_group_id": df.iloc[row_index].get("env_causal_group_id", ""),
                        "env_weathering_regime": df.iloc[row_index].get("env_weathering_regime", ""),
                    }
                )
    return pd.DataFrame(metrics_rows), pd.DataFrame(prediction_rows)


def summarize_metrics(metrics: pd.DataFrame, group_cols: list[str] | None = None) -> pd.DataFrame:
    if group_cols is None:
        group_cols = ["cv", "model"]
    metric_cols = [
        "roc_auc",
        "average_precision",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
    ]
    summary = (
        metrics.groupby(group_cols, dropna=False)[metric_cols]
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


def make_default_splitters(df: pd.DataFrame):
    y = df[TARGET].astype(int)
    yield "stratified_kfold", StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE), None
    if "state" in df.columns and df["state"].nunique(dropna=True) >= 5:
        groups = df["state"].fillna("unknown")
        yield "groupkfold_state", GroupKFold(n_splits=5), groups


def make_leave_one_group_splitter(groups: pd.Series):
    return LeaveOneGroupOut(), groups.fillna("unknown").astype(str)
