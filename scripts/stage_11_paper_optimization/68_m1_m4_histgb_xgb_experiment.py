from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
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
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TARGET = "Y_label"
RANDOM_STATE = 20260622
DECISION_THRESHOLD = 0.5
MIN_PAIR_COUNT = 30
DEFAULT_M4_THRESHOLD = 0.30

MODEL_KEYS = [
    "M1_Full_Climate",
    "M2_No_Climate",
    "M3_Climate_Normalized",
    "M4_Sensitive_Residualized",
]
CLASSIFIER_FAMILIES = ["hist_gradient_boosting", "xgboost"]


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False, encoding="utf-8-sig")


def load_dataset_from_run(run_dir: Path) -> tuple[pd.DataFrame, Path]:
    manifest_path = run_dir / "standard_workflow_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        input_path = Path(manifest["input_path"])
        if input_path.exists():
            return read_table(input_path), input_path
        # Some old manifests were written under a mojibake path. Fall back to the dataset folder.
    fallback = PROJECT_ROOT / "outputs" / "model_datasets" / "by_sample_scheme" / "known_mining_neutral" / "known_mining_neutral_ratio_1_10" / "model_dataset_known_mining_neutral_ratio_1_10_supervised_all_features_v1.parquet"
    if fallback.exists() and run_dir.name == "known_mining_neutral_ratio_1_10_supervised_all_features_v1":
        return read_table(fallback), fallback
    found = next((run_dir.glob("**/model_dataset*.parquet")), None)
    if found is None:
        raise FileNotFoundError(f"Cannot find input dataset for run: {run_dir}")
    return read_table(found), found


def make_splitters(df: pd.DataFrame, validation_mode: str = "core") -> list[tuple[str, object, pd.Series | None]]:
    y = df[TARGET].astype(int)
    positives = int(y.sum())
    negatives = int((y == 0).sum())
    n_splits = min(5, positives, negatives)
    splitters: list[tuple[str, object, pd.Series | None]] = []
    if validation_mode in {"core", "all", "stratified"}:
        splitters.append(
            ("stratified_kfold", StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE), None)
        )
    if validation_mode in {"core", "all", "group"} and "state" in df.columns and df["state"].nunique(dropna=True) >= 2:
        groups = df["state"].fillna("unknown").astype(str)
        splitters.append(("groupkfold_state", GroupKFold(n_splits=min(5, groups.nunique())), groups))
    return splitters


def feature_columns_for_model(role_table: pd.DataFrame, model_key: str) -> list[str]:
    if model_key == "M1_Full_Climate":
        return role_table.loc[role_table["used_in_m1_full_climate"], "column"].tolist()
    if model_key == "M2_No_Climate":
        return role_table.loc[role_table["used_in_m2_no_climate"], "column"].tolist()
    if model_key in {"M3_Climate_Normalized", "M4_Sensitive_Residualized"}:
        return role_table.loc[role_table["used_in_m2_no_climate"], "column"].tolist()
    raise ValueError(model_key)


def columns_by_role(role_table: pd.DataFrame, role: str) -> list[str]:
    return role_table.loc[(role_table["is_numeric"]) & (role_table["role"] == role), "column"].tolist()


def climate_adjuster_columns(role_table: pd.DataFrame) -> list[str]:
    return role_table.loc[(role_table["is_numeric"]) & (role_table["used_as_climate_adjuster"]), "column"].tolist()


def local_climate_sensitive_columns(train_df: pd.DataFrame, role_table: pd.DataFrame, threshold: float) -> pd.DataFrame:
    element_cols = columns_by_role(role_table, "geochemistry")
    climate_cols = climate_adjuster_columns(role_table)
    if not element_cols or not climate_cols:
        return pd.DataFrame()
    selected = train_df[element_cols + climate_cols]
    pair_counts = selected[element_cols].notna().astype(int).T.dot(selected[climate_cols].notna().astype(int))
    spearman = selected.corr(method="spearman", min_periods=MIN_PAIR_COUNT).loc[element_cols, climate_cols]
    rows = []
    for element in element_cols:
        best_climate = ""
        best_abs = math.nan
        best_n = 0
        for climate in climate_cols:
            value = spearman.loc[element, climate]
            if pd.isna(value):
                continue
            abs_value = abs(float(value))
            if pd.isna(best_abs) or abs_value > best_abs:
                best_abs = abs_value
                best_climate = climate
                best_n = int(pair_counts.loc[element, climate])
        rows.append(
            {
                "element_feature": element,
                "best_climate_feature_spearman": best_climate,
                "spearman_abs_max": best_abs,
                "n_pair_at_best": best_n,
                "is_climate_sensitive": bool(pd.notna(best_abs) and best_abs >= threshold),
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
    direct_cols = [c for c in feature_columns_for_model(role_table, "M3_Climate_Normalized") if c not in element_cols]

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


def make_classifier(family: str, pos_weight: float) -> Pipeline:
    if family == "hist_gradient_boosting":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        learning_rate=0.05,
                        max_iter=300,
                        max_leaf_nodes=15,
                        l2_regularization=0.1,
                        class_weight={0: 1.0, 1: pos_weight},
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )
    if family == "xgboost":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=400,
                        learning_rate=0.03,
                        max_depth=3,
                        min_child_weight=3,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        reg_lambda=2.0,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        scale_pos_weight=pos_weight,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
    raise ValueError(family)


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


def top_k_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    order = np.argsort(-y_prob)
    ranked_true = y_true[order]
    total_pos = max(int(y_true.sum()), 1)
    base_rate = float(y_true.mean()) if len(y_true) else math.nan
    out: dict[str, float] = {}
    for frac in [0.01, 0.05, 0.10, 0.20]:
        k = max(1, int(math.ceil(len(y_true) * frac)))
        selected = ranked_true[:k]
        precision_at_k = float(selected.mean())
        recall_at_k = float(selected.sum() / total_pos)
        f1_at_k = (
            2 * precision_at_k * recall_at_k / (precision_at_k + recall_at_k)
            if precision_at_k + recall_at_k > 0
            else 0.0
        )
        discounts = 1.0 / np.log2(np.arange(2, len(selected) + 2))
        dcg = float(np.sum(selected.astype(float) * discounts))
        ideal = np.sort(ranked_true)[::-1][:k].astype(float)
        ideal_dcg = float(np.sum(ideal * discounts))
        label = str(int(frac * 100)).zfill(2)
        out[f"top{label}_precision"] = precision_at_k
        out[f"top{label}_recall"] = recall_at_k
        out[f"top{label}_f1"] = f1_at_k
        out[f"top{label}_lift"] = precision_at_k / base_rate if base_rate else math.nan
        out[f"top{label}_ndcg"] = dcg / ideal_dcg if ideal_dcg > 0 else math.nan
    return out


def add_metric_row(
    rows: list[dict],
    predictions: list[dict],
    *,
    dataset_name: str,
    cv_name: str,
    fold: int,
    test_group: str | None,
    model_key: str,
    classifier_family: str,
    n_features: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    df: pd.DataFrame,
    y_test: pd.Series,
    y_prob: np.ndarray,
) -> None:
    y_true = y_test.to_numpy()
    y_pred = (y_prob >= DECISION_THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    model_name = f"{model_key}_{classifier_family}"
    row = {
        "dataset": dataset_name,
        "experiment": "m1_m4_classifier_replacement",
        "cv": cv_name,
        "fold": int(fold),
        "test_group": test_group,
        "model": model_key,
        "classifier_family": classifier_family,
        "model_name": model_name,
        "n_features": int(n_features),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "positive_test": int(y_true.sum()),
        "negative_test": int((y_true == 0).sum()),
        "roc_auc": safe_metric("roc_auc", y_true, y_pred, y_prob),
        "average_precision": safe_metric("average_precision", y_true, y_pred, y_prob),
        "balanced_accuracy": safe_metric("balanced_accuracy", y_true, y_pred, y_prob),
        "precision": safe_metric("precision", y_true, y_pred, y_prob),
        "recall": safe_metric("recall", y_true, y_pred, y_prob),
        "f1": safe_metric("f1", y_true, y_pred, y_prob),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    row.update(top_k_metrics(y_true, y_prob))
    rows.append(row)
    for row_index, true_value, pred_value, prob_value in zip(test_idx, y_true, y_pred, y_prob):
        predictions.append(
            {
                "dataset": dataset_name,
                "experiment": "m1_m4_classifier_replacement",
                "row_index": int(row_index),
                "sample_id": df.iloc[row_index].get("sample_id", ""),
                "cv": cv_name,
                "fold": int(fold),
                "test_group": test_group,
                "model": model_key,
                "classifier_family": classifier_family,
                "model_name": model_name,
                "Y_label": int(true_value),
                "y_pred": int(pred_value),
                "y_prob": float(prob_value),
                "state": df.iloc[row_index].get("state", ""),
                "negative_type": df.iloc[row_index].get("negative_type", ""),
                "env_causal_group_id": df.iloc[row_index].get("env_causal_group_id", ""),
            }
        )


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "roc_auc",
        "average_precision",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "top05_precision",
        "top05_recall",
        "top05_f1",
        "top05_lift",
        "top05_ndcg",
        "top10_precision",
        "top10_recall",
        "top10_f1",
        "top10_lift",
        "top10_ndcg",
    ]
    summary = metrics.groupby(["dataset", "cv", "model", "classifier_family", "model_name"], dropna=False)[
        metric_cols
    ].agg(["mean", "std", "count"]).reset_index()
    summary.columns = [
        "_".join(str(part) for part in col if str(part)) if isinstance(col, tuple) else str(col)
        for col in summary.columns
    ]
    return summary


def markdown_table(df: pd.DataFrame, max_rows: int = 100) -> str:
    if df.empty:
        return "_No data._"
    show = df.head(max_rows).copy()
    for col in show.columns:
        if pd.api.types.is_float_dtype(show[col]):
            show[col] = show[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
    lines = [
        "| " + " | ".join(map(str, show.columns)) + " |",
        "| " + " | ".join(["---"] * len(show.columns)) + " |",
    ]
    for _, row in show.iterrows():
        lines.append("| " + " | ".join(str(row[c]).replace("|", "/") for c in show.columns) + " |")
    return "\n".join(lines)


def write_report(out_dir: Path, dataset_name: str, summary: pd.DataFrame) -> None:
    cols = [
        "cv",
        "model",
        "classifier_family",
        "roc_auc_mean",
        "average_precision_mean",
        "f1_mean",
        "top05_precision_mean",
        "top05_recall_mean",
        "top05_f1_mean",
        "top05_lift_mean",
        "top05_ndcg_mean",
        "top10_precision_mean",
        "top10_recall_mean",
        "top10_f1_mean",
        "top10_lift_mean",
        "top10_ndcg_mean",
    ]
    group = summary[summary["cv"] == "groupkfold_state"].copy()
    report = [
        f"# M1-M4 Classifier Replacement Experiment: {dataset_name}",
        "",
        "## Experiment Setting",
        "",
        "- Dataset: `known_mining_neutral_ratio_1_10_supervised_all_features_v1`.",
        "- Samples used for training and validation: positive + negative only; neutral samples are not included.",
        "- Validation: StratifiedKFold and GroupKFold by state.",
        "- Replaced classifiers: HistGradientBoosting and XGBoost.",
        "- Feature/decoupling logic remains the same as the standard M1-M4 climate-decoupling workflow.",
        "",
        "## Model Definitions",
        "",
        "| Model | Meaning |",
        "|---|---|",
        "| M1_Full_Climate | Full feature set including climate variables. |",
        "| M2_No_Climate | Non-climate feature set. |",
        "| M3_Climate_Normalized | All geochemical variables residualized against climate variables. |",
        "| M4_Sensitive_Residualized | Only fold-local climate-sensitive geochemical variables residualized. |",
        "",
        "## GroupKFold Main Results",
        "",
        markdown_table(group[[c for c in cols if c in group.columns]].sort_values("top05_f1_mean", ascending=False)),
        "",
        "## All CV Results",
        "",
        markdown_table(summary[[c for c in cols if c in summary.columns]].sort_values(["cv", "top05_f1_mean"], ascending=[True, False]), max_rows=200),
    ]
    (out_dir / "m1_m4_histgb_xgb_report.md").write_text("\n".join(report), encoding="utf-8")


def run_experiment(run_dir: Path, output_dir: Path, m4_threshold: float, validation_mode: str) -> pd.DataFrame:
    dataset_name = run_dir.name
    df, dataset_path = load_dataset_from_run(run_dir)
    df = df[df[TARGET].isin([0, 1])].reset_index(drop=True)
    role_table = pd.read_csv(run_dir / "00_dataset_profile" / "feature_roles.csv")
    y = df[TARGET].astype(int).reset_index(drop=True)
    pos_weight = int((y == 0).sum()) / max(int(y.sum()), 1)

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
            for model_key in MODEL_KEYS:
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
                for family in CLASSIFIER_FAMILIES:
                    model = make_classifier(family, pos_weight)
                    model.fit(x_train, y.iloc[train_idx])
                    y_prob = model.predict_proba(x_test)[:, 1]
                    add_metric_row(
                        rows,
                        predictions,
                        dataset_name=dataset_name,
                        cv_name=cv_name,
                        fold=fold,
                        test_group=test_group,
                        model_key=model_key,
                        classifier_family=family,
                        n_features=len(feature_columns),
                        train_idx=train_idx,
                        test_idx=test_idx,
                        df=df,
                        y_test=y.iloc[test_idx],
                        y_prob=y_prob,
                    )

    metrics = pd.DataFrame(rows)
    predictions_df = pd.DataFrame(predictions)
    sensitive_all = pd.concat(sensitive_rows, ignore_index=True) if sensitive_rows else pd.DataFrame()
    summary = summarize_metrics(metrics)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_table(metrics, output_dir / "m1_m4_histgb_xgb_fold_metrics.csv")
    write_table(predictions_df, output_dir / "m1_m4_histgb_xgb_predictions.csv")
    write_table(summary, output_dir / "m1_m4_histgb_xgb_summary.csv")
    write_table(sensitive_all, output_dir / "m1_m4_histgb_xgb_m4_sensitive_features.csv")
    write_report(output_dir, dataset_name, summary)
    manifest = {
        "dataset": dataset_name,
        "input_dataset_path": str(dataset_path),
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "m4_threshold": m4_threshold,
        "validation_mode": validation_mode,
        "samples_used": "positive_negative_only",
        "neutral_samples_used": False,
        "model_keys": MODEL_KEYS,
        "classifier_families": CLASSIFIER_FAMILIES,
    }
    (output_dir / "m1_m4_histgb_xgb_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run M1-M4 with HistGradientBoosting and XGBoost classifiers.")
    parser.add_argument("--run-dir", required=True, help="Standardized run directory.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--m4-threshold", type=float, default=DEFAULT_M4_THRESHOLD)
    parser.add_argument("--validation-mode", choices=["core", "all", "stratified", "group"], default="core")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else PROJECT_ROOT / "outputs" / "paper_optimization" / "model_family_replacement" / run_dir.name
    )
    run_experiment(run_dir, output_dir, args.m4_threshold, args.validation_mode)
    print(f"Wrote M1-M4 HistGB/XGB experiment outputs to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
