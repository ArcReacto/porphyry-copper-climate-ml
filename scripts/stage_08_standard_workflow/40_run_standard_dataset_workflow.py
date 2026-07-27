from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
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
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "stage_07_climate_decoupling"))

from decoupling_utils import (  # noqa: E402
    TARGET,
    climate_adjuster_columns,
    columns_by_role,
    define_feature_roles,
    feature_columns_for_model,
)


RANDOM_STATE = 20260622
DECISION_THRESHOLD = 0.5
DEFAULT_M4_THRESHOLD = 0.30
MIN_PAIR_COUNT = 30
RF_N_ESTIMATORS = 300

BASELINE_MODELS = [
    "dummy_stratified",
    "logistic_regression",
    "random_forest",
    "hist_gradient_boosting",
]

DECOUPLING_MODELS = [
    "M1_Full_Climate",
    "M2_No_Climate",
    "M3_Climate_Normalized",
    "M4_Sensitive_Residualized",
]


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


def summarize_metrics(metrics: pd.DataFrame, group_cols: list[str] | None = None) -> pd.DataFrame:
    if group_cols is None:
        group_cols = ["dataset", "experiment", "cv", "model"]
    metric_cols = [
        "roc_auc",
        "average_precision",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
    ]
    summary = metrics.groupby(group_cols, dropna=False)[metric_cols].agg(["mean", "std", "count"]).reset_index()
    summary.columns = [
        "_".join([str(part) for part in col if str(part)])
        if isinstance(col, tuple)
        else str(col)
        for col in summary.columns
    ]
    return summary


def compact_summary(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        c
        for c in [
            "dataset",
            "experiment",
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


def markdown_table(df: pd.DataFrame, max_rows: int = 80) -> str:
    if df.empty:
        return "_No data._"
    display = df.head(max_rows).copy()
    headers = [str(c) for c in display.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(str(row[c]).replace("|", "\\|") for c in display.columns) + " |")
    return "\n".join(lines)


def dataset_name_from_path(path: Path, explicit_name: str | None) -> str:
    if explicit_name:
        return explicit_name
    stem = path.stem
    if stem.startswith("model_dataset_"):
        stem = stem.replace("model_dataset_", "", 1)
    return stem


def make_splitters(df: pd.DataFrame, validation_mode: str) -> list[tuple[str, object, pd.Series | None]]:
    y = df[TARGET].astype(int)
    positives = int(y.sum())
    negatives = int((y == 0).sum())
    n_splits = min(5, positives, negatives)
    if n_splits < 2:
        raise ValueError("Need at least two positive and two negative samples for cross-validation.")

    splitters: list[tuple[str, object, pd.Series | None]] = [
        ("stratified_kfold", StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE), None)
    ]
    if "state" in df.columns and df["state"].nunique(dropna=True) >= 2:
        groups = df["state"].fillna("unknown").astype(str)
        splitters.append(("groupkfold_state", GroupKFold(n_splits=min(5, groups.nunique())), groups))
        if validation_mode == "exhaustive":
            splitters.append(("leave_one_state_out", LeaveOneGroupOut(), groups))
    if (
        validation_mode == "exhaustive"
        and "env_causal_group_id" in df.columns
        and df["env_causal_group_id"].nunique(dropna=True) >= 2
    ):
        groups = df["env_causal_group_id"].fillna("unknown").astype(str)
        splitters.append(("leave_one_environment_group_out", LeaveOneGroupOut(), groups))
    return splitters


def make_baseline_models(pos_weight: float) -> dict[str, Pipeline]:
    return {
        "dummy_stratified": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", DummyClassifier(strategy="stratified", random_state=RANDOM_STATE)),
            ]
        ),
        "logistic_regression": Pipeline(
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
        ),
        "random_forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=RF_N_ESTIMATORS,
                        min_samples_leaf=3,
                        max_features="sqrt",
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        learning_rate=0.05,
                        max_iter=200,
                        max_leaf_nodes=15,
                        l2_regularization=0.1,
                        class_weight={0: 1.0, 1: pos_weight},
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def make_rf_model() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=RF_N_ESTIMATORS,
                    min_samples_leaf=3,
                    max_features="sqrt",
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def predict_prob(model: Pipeline, x: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    if hasattr(model[-1], "decision_function"):
        score = model.decision_function(x)
        return 1.0 / (1.0 + np.exp(-score))
    return model.predict(x).astype(float)


def add_metric_rows(
    rows: list[dict],
    predictions: list[dict],
    *,
    dataset_name: str,
    experiment: str,
    cv_name: str,
    fold: int,
    test_group: str | None,
    model_name: str,
    n_features: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    df: pd.DataFrame,
    y_test: pd.Series,
    y_prob: np.ndarray,
) -> None:
    y_pred = (y_prob >= DECISION_THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    rows.append(
        {
            "dataset": dataset_name,
            "experiment": experiment,
            "cv": cv_name,
            "fold": int(fold),
            "test_group": test_group,
            "model": model_name,
            "n_features": int(n_features),
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
    for row_index, true_value, pred_value, prob_value in zip(test_idx, y_test.to_numpy(), y_pred, y_prob):
        predictions.append(
            {
                "dataset": dataset_name,
                "experiment": experiment,
                "row_index": int(row_index),
                "sample_id": df.iloc[row_index].get("sample_id", ""),
                "cv": cv_name,
                "fold": int(fold),
                "test_group": test_group,
                "model": model_name,
                "Y_label": int(true_value),
                "y_pred": int(pred_value),
                "y_prob": float(prob_value),
                "state": df.iloc[row_index].get("state", ""),
                "negative_type": df.iloc[row_index].get("negative_type", ""),
                "env_causal_group_id": df.iloc[row_index].get("env_causal_group_id", ""),
            }
        )


def baseline_feature_columns(role_table: pd.DataFrame) -> list[str]:
    return role_table.loc[role_table["used_in_m1_full_climate"], "column"].tolist()


def run_baselines(
    df: pd.DataFrame,
    role_table: pd.DataFrame,
    dataset_name: str,
    validation_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = df[TARGET].astype(int).reset_index(drop=True)
    pos = int(y.sum())
    neg = int((y == 0).sum())
    pos_weight = neg / max(pos, 1)
    feature_cols = baseline_feature_columns(role_table)
    x = df[feature_cols]
    models = make_baseline_models(pos_weight)
    rows: list[dict] = []
    predictions: list[dict] = []

    for cv_name, splitter, groups in make_splitters(df, validation_mode):
        split_iter = splitter.split(x, y, groups) if groups is not None else splitter.split(x, y)
        for fold, (train_idx, test_idx) in enumerate(split_iter, start=1):
            train_idx = np.asarray(train_idx)
            test_idx = np.asarray(test_idx)
            if y.iloc[train_idx].nunique() < 2:
                continue
            test_group = None
            if groups is not None:
                test_group = ";".join(sorted(pd.Series(groups).iloc[test_idx].astype(str).unique().tolist()))
            for model_name in BASELINE_MODELS:
                model = models[model_name]
                model.fit(x.iloc[train_idx], y.iloc[train_idx])
                y_prob = predict_prob(model, x.iloc[test_idx])
                add_metric_rows(
                    rows,
                    predictions,
                    dataset_name=dataset_name,
                    experiment="baseline_models",
                    cv_name=cv_name,
                    fold=fold,
                    test_group=test_group,
                    model_name=model_name,
                    n_features=len(feature_cols),
                    train_idx=train_idx,
                    test_idx=test_idx,
                    df=df,
                    y_test=y.iloc[test_idx],
                    y_prob=y_prob,
                )
    return pd.DataFrame(rows), pd.DataFrame(predictions)


def local_climate_sensitive_columns(
    train_df: pd.DataFrame,
    role_table: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    element_cols = columns_by_role(role_table, "geochemistry")
    climate_cols = climate_adjuster_columns(role_table)
    if not element_cols or not climate_cols:
        return pd.DataFrame(columns=["element_feature", "spearman_abs_max", "is_climate_sensitive"])

    selected = train_df[element_cols + climate_cols]
    pair_counts = selected[element_cols].notna().astype(int).T.dot(selected[climate_cols].notna().astype(int))
    spearman = selected.corr(method="spearman", min_periods=MIN_PAIR_COUNT).loc[element_cols, climate_cols]

    rows = []
    for element in element_cols:
        best_feature = ""
        best_value = math.nan
        best_n = 0
        for climate in climate_cols:
            value = spearman.loc[element, climate]
            if pd.isna(value):
                continue
            abs_value = abs(float(value))
            if pd.isna(best_value) or abs_value > best_value:
                best_value = abs_value
                best_feature = climate
                best_n = int(pair_counts.loc[element, climate])
        rows.append(
            {
                "element_feature": element,
                "best_climate_feature_spearman": best_feature,
                "spearman_abs_max": best_value,
                "n_pair_at_best": best_n,
                "is_climate_sensitive": bool(pd.notna(best_value) and best_value >= threshold),
            }
        )
    return pd.DataFrame(rows).sort_values("spearman_abs_max", ascending=False)


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


def build_decoupled_features(
    df: pd.DataFrame,
    role_table: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    model_key: str,
    m4_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], pd.DataFrame]:
    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()
    empty_sensitive = pd.DataFrame()

    if model_key in {"M1_Full_Climate", "M2_No_Climate"}:
        columns = feature_columns_for_model(role_table, model_key)
        return train_df[columns], test_df[columns], columns, empty_sensitive

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
        sensitive = empty_sensitive
    elif model_key == "M4_Sensitive_Residualized":
        sensitive = local_climate_sensitive_columns(train_df, role_table, threshold=m4_threshold)
        residual_cols = sensitive.loc[sensitive["is_climate_sensitive"], "element_feature"].tolist()
        raw_element_cols = [c for c in element_cols if c not in set(residual_cols)]
    else:
        raise ValueError(model_key)

    if residual_cols:
        residualizer = ClimateResidualizer(element_cols=residual_cols, climate_cols=climate_cols)
        residualizer.fit(train_df)
        train_resid = residualizer.transform(train_df)
        test_resid = residualizer.transform(test_df)
        x_train = pd.concat([train_resid, train_df[raw_element_cols + direct_cols]], axis=1)
        x_test = pd.concat([test_resid, test_df[raw_element_cols + direct_cols]], axis=1)
        columns = list(train_resid.columns) + raw_element_cols + direct_cols
    else:
        x_train = train_df[raw_element_cols + direct_cols]
        x_test = test_df[raw_element_cols + direct_cols]
        columns = raw_element_cols + direct_cols
    return x_train, x_test, columns, sensitive


def run_climate_decoupling(
    df: pd.DataFrame,
    role_table: pd.DataFrame,
    dataset_name: str,
    m4_threshold: float,
    validation_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y = df[TARGET].astype(int).reset_index(drop=True)
    rows: list[dict] = []
    predictions: list[dict] = []
    sensitive_rows: list[pd.DataFrame] = []

    for cv_name, splitter, groups in make_splitters(df, validation_mode):
        split_iter = splitter.split(df, y, groups) if groups is not None else splitter.split(df, y)
        for fold, (train_idx, test_idx) in enumerate(split_iter, start=1):
            train_idx = np.asarray(train_idx)
            test_idx = np.asarray(test_idx)
            if y.iloc[train_idx].nunique() < 2:
                continue
            test_group = None
            if groups is not None:
                test_group = ";".join(sorted(pd.Series(groups).iloc[test_idx].astype(str).unique().tolist()))
            for model_key in DECOUPLING_MODELS:
                x_train, x_test, feature_columns, sensitive = build_decoupled_features(
                    df=df,
                    role_table=role_table,
                    train_idx=train_idx,
                    test_idx=test_idx,
                    model_key=model_key,
                    m4_threshold=m4_threshold,
                )
                if model_key == "M4_Sensitive_Residualized" and not sensitive.empty:
                    temp = sensitive.copy()
                    temp.insert(0, "dataset", dataset_name)
                    temp.insert(1, "cv", cv_name)
                    temp.insert(2, "fold", fold)
                    temp.insert(3, "test_group", test_group)
                    sensitive_rows.append(temp)
                model = make_rf_model()
                model.fit(x_train, y.iloc[train_idx])
                y_prob = model.predict_proba(x_test)[:, 1]
                add_metric_rows(
                    rows,
                    predictions,
                    dataset_name=dataset_name,
                    experiment="climate_decoupling",
                    cv_name=cv_name,
                    fold=fold,
                    test_group=test_group,
                    model_name=model_key,
                    n_features=len(feature_columns),
                    train_idx=train_idx,
                    test_idx=test_idx,
                    df=df,
                    y_test=y.iloc[test_idx],
                    y_prob=y_prob,
                )
    sensitive_all = pd.concat(sensitive_rows, ignore_index=True) if sensitive_rows else pd.DataFrame()
    return pd.DataFrame(rows), pd.DataFrame(predictions), sensitive_all


def write_dataset_profile(
    df: pd.DataFrame,
    role_table: pd.DataFrame,
    dataset_name: str,
    dataset_path: Path,
    out_dir: Path,
) -> dict:
    profile_dir = out_dir / "00_dataset_profile"
    profile_dir.mkdir(parents=True, exist_ok=True)
    label_counts = df[TARGET].value_counts(dropna=False).sort_index().to_dict()
    negative_counts = df["negative_type"].value_counts(dropna=False).to_dict() if "negative_type" in df.columns else {}
    state_counts = df["state"].value_counts(dropna=False).to_dict() if "state" in df.columns else {}
    role_counts = role_table["role"].value_counts(dropna=False).to_dict()
    profile = {
        "dataset": dataset_name,
        "input_path": str(dataset_path),
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "label_counts": {str(k): int(v) for k, v in label_counts.items()},
        "negative_type_counts": {str(k): int(v) for k, v in negative_counts.items()},
        "state_counts": {str(k): int(v) for k, v in state_counts.items()},
        "role_counts": {str(k): int(v) for k, v in role_counts.items()},
        "m4_sensitive_selection": "Fold-local Spearman screening on training split only.",
    }
    (profile_dir / "dataset_profile.json").write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
    write_table(role_table, profile_dir / "feature_roles.csv")
    return profile


def write_report(out_dir: Path, profile: dict, summaries: list[pd.DataFrame], validation_mode: str) -> None:
    combined = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    report = f"""# 标准化数据集实验报告：{profile["dataset"]}

## 1. 数据集概况

| 项目 | 数值 |
|---|---:|
| 行数 | {profile["rows"]} |
| 列数 | {profile["columns"]} |
| 正样本 | {profile["label_counts"].get("1", 0)} |
| 负样本 | {profile["label_counts"].get("0", 0)} |

负样本类型：

```json
{json.dumps(profile["negative_type_counts"], ensure_ascii=False, indent=2)}
```

## 2. 本次标准流程

| 实验 | 比较对象 | 目的 |
|---|---|---|
| baseline_models | Dummy、Logistic Regression、Random Forest、HistGradientBoosting | 判断当前数据集的基础可预测性 |
| climate_decoupling | M1 Full Climate、M2 No Climate、M3 Full Residualized、M4 Sensitive Residualized | 判断气候变量直接输入/气候残差化/部分残差化的影响 |

M4 的气候敏感特征不再复用旧数据集结果，而是在每个交叉验证 fold 的训练集内部用 Spearman 相关重新筛选，默认阈值为 `0.3`。

本次验证模式：`{validation_mode}`。

- `core`：运行 `stratified_kfold` 和 `groupkfold_state`，适合多个采样比例批量比较。
- `exhaustive`：额外运行 `leave_one_state_out` 和 `leave_one_environment_group_out`，适合最终严格验证。

## 3. 结果摘要

{markdown_table(compact_summary(combined), max_rows=120)}

## 4. 输出目录

```text
{out_dir}
```
"""
    (out_dir / "standard_workflow_report.md").write_text(report, encoding="utf-8")


def run_workflow(
    dataset_path: Path,
    dataset_name: str | None,
    output_root: Path,
    m4_threshold: float,
    validation_mode: str,
) -> dict:
    df = load_table(dataset_path)
    name = dataset_name_from_path(dataset_path, dataset_name)
    out_dir = output_root / name
    out_dir.mkdir(parents=True, exist_ok=True)

    role_table = define_feature_roles(df)
    profile = write_dataset_profile(df, role_table, name, dataset_path, out_dir)

    baseline_metrics, baseline_predictions = run_baselines(df, role_table, name, validation_mode)
    baseline_summary = summarize_metrics(baseline_metrics)
    baseline_dir = out_dir / "01_baseline_models"
    write_table(baseline_metrics, baseline_dir / "baseline_metrics.csv")
    write_table(baseline_predictions, baseline_dir / "baseline_predictions.csv")
    write_table(baseline_summary, baseline_dir / "baseline_summary.csv")

    climate_metrics, climate_predictions, sensitive = run_climate_decoupling(
        df,
        role_table,
        name,
        m4_threshold,
        validation_mode,
    )
    climate_summary = summarize_metrics(climate_metrics)
    climate_dir = out_dir / "02_climate_decoupling"
    write_table(climate_metrics, climate_dir / "climate_decoupling_metrics.csv")
    write_table(climate_predictions, climate_dir / "climate_decoupling_predictions.csv")
    write_table(climate_summary, climate_dir / "climate_decoupling_summary.csv")
    write_table(sensitive, climate_dir / "m4_fold_local_sensitive_features.csv")

    combined_summary = pd.concat([baseline_summary, climate_summary], ignore_index=True)
    write_table(combined_summary, out_dir / "standard_workflow_summary.csv")
    write_report(out_dir, profile, [baseline_summary, climate_summary], validation_mode)

    manifest = {
        "dataset": name,
        "input_path": str(dataset_path),
        "output_dir": str(out_dir),
        "m4_threshold": float(m4_threshold),
        "validation_mode": validation_mode,
        "rf_n_estimators": RF_N_ESTIMATORS,
        "outputs": {
            "profile": str(out_dir / "00_dataset_profile" / "dataset_profile.json"),
            "baseline_summary": str(baseline_dir / "baseline_summary.csv"),
            "climate_decoupling_summary": str(climate_dir / "climate_decoupling_summary.csv"),
            "combined_summary": str(out_dir / "standard_workflow_summary.csv"),
            "report": str(out_dir / "standard_workflow_report.md"),
        },
    }
    (out_dir / "standard_workflow_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the standardized experiment workflow for one or more model datasets.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="Model dataset CSV/parquet paths. Each dataset is processed independently.",
    )
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="Optional name for a single dataset run. Do not use with multiple datasets.",
    )
    parser.add_argument(
        "--output-root",
        default=str(PROJECT_ROOT / "outputs" / "standardized_runs"),
        help="Root directory for standardized outputs.",
    )
    parser.add_argument(
        "--m4-threshold",
        type=float,
        default=DEFAULT_M4_THRESHOLD,
        help="Spearman absolute-correlation threshold for fold-local M4 sensitive feature screening.",
    )
    parser.add_argument(
        "--validation-mode",
        choices=["core", "exhaustive"],
        default="core",
        help="core runs stratified_kfold and groupkfold_state. exhaustive also runs leave-one-state/environment validation.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_paths = [Path(p) for p in args.datasets]
    if args.dataset_name and len(dataset_paths) > 1:
        raise ValueError("--dataset-name can only be used with one dataset.")
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    manifests = []
    for path in dataset_paths:
        manifest = run_workflow(path, args.dataset_name, output_root, args.m4_threshold, args.validation_mode)
        manifests.append(manifest)
        print(f"Wrote standardized workflow for {manifest['dataset']}")
        print(manifest["outputs"]["report"])

    (output_root / "standardized_workflow_index.json").write_text(
        json.dumps({"runs": manifests}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
