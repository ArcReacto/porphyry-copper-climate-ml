from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

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
TARGET = "Y_label"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "standardized_runs"
RANDOM_STATE = 20260622
DECISION_THRESHOLD = 0.5
DEFAULT_M4_THRESHOLD = 0.30
MIN_PAIR_COUNT = 30
RF_N_ESTIMATORS = 300

CONCEPT_MODELS = [
    "Concept_M1_All_Concepts",
    "Concept_M2_No_Climate_Concepts",
    "Concept_M3_All_NonClimate_Residualized",
    "Concept_M4_Sensitive_Residualized",
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


def dataset_name_from_path(path: Path, explicit_name: str | None = None) -> str:
    if explicit_name:
        return explicit_name
    stem = path.stem
    if stem.startswith("model_dataset_"):
        stem = stem.replace("model_dataset_", "", 1)
    if stem == "04_concept_features":
        # .../<dataset>/03_causal_graph/04_concept_features.parquet
        return path.parents[1].name
    return stem


def concept_path_from_dataset(path: Path, output_root: Path, dataset_name: str) -> Path:
    if path.name.startswith("04_concept_features"):
        return path
    return output_root / dataset_name / "03_causal_graph" / "04_concept_features.parquet"


def dictionary_path_from_concept_path(concept_path: Path) -> Path:
    return concept_path.parent / "05_concept_feature_dictionary.csv"


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


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    metric_cols = ["roc_auc", "average_precision", "balanced_accuracy", "precision", "recall", "f1"]
    summary = metrics.groupby(["dataset", "experiment", "cv", "model"], dropna=False)[metric_cols].agg(
        ["mean", "std", "count"]
    ).reset_index()
    summary.columns = [
        "_".join([str(part) for part in col if str(part)])
        if isinstance(col, tuple)
        else str(col)
        for col in summary.columns
    ]
    return summary


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


def concept_role_table(df: pd.DataFrame, dictionary_path: Path) -> pd.DataFrame:
    concept_cols = sorted([c for c in df.columns if c.startswith("concept_") and pd.api.types.is_numeric_dtype(df[c])])
    group_by_column: dict[str, str] = {}
    concept_by_column: dict[str, str] = {}
    if dictionary_path.exists():
        dictionary = pd.read_csv(dictionary_path)
        if {"concept", "concept_group"}.issubset(dictionary.columns):
            for _, row in dictionary.iterrows():
                col = f"concept_{row['concept']}"
                group_by_column[col] = str(row["concept_group"])
                concept_by_column[col] = str(row["concept"])

    rows = []
    for col in concept_cols:
        concept = concept_by_column.get(col, col.replace("concept_", "", 1))
        group = group_by_column.get(col, "unknown")
        rows.append(
            {
                "column": col,
                "concept": concept,
                "concept_group": group,
                "is_climate_concept": bool(group == "climate"),
                "missing_rate": float(df[col].isna().mean()),
                "n_unique": int(df[col].nunique(dropna=True)),
                "used_in_concept_m1_all": True,
                "used_in_concept_m2_no_climate": bool(group != "climate"),
                "used_as_climate_adjuster": bool(group == "climate"),
                "used_as_residual_target": bool(group != "climate"),
            }
        )
    return pd.DataFrame(rows)


def feature_columns(role_table: pd.DataFrame, model_key: str) -> list[str]:
    if model_key == "Concept_M1_All_Concepts":
        return role_table["column"].tolist()
    if model_key == "Concept_M2_No_Climate_Concepts":
        return role_table.loc[~role_table["is_climate_concept"], "column"].tolist()
    if model_key in {"Concept_M3_All_NonClimate_Residualized", "Concept_M4_Sensitive_Residualized"}:
        return role_table.loc[~role_table["is_climate_concept"], "column"].tolist()
    raise ValueError(model_key)


def climate_columns(role_table: pd.DataFrame) -> list[str]:
    return role_table.loc[role_table["is_climate_concept"], "column"].tolist()


def local_climate_sensitive_concepts(
    train_df: pd.DataFrame,
    role_table: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    non_climate_cols = role_table.loc[~role_table["is_climate_concept"], "column"].tolist()
    climate_cols = climate_columns(role_table)
    if not non_climate_cols or not climate_cols:
        return pd.DataFrame(
            columns=[
                "concept_feature",
                "concept_group",
                "best_climate_concept_spearman",
                "spearman_abs_max",
                "n_pair_at_best",
                "is_climate_sensitive",
            ]
        )

    selected = train_df[non_climate_cols + climate_cols]
    pair_counts = selected[non_climate_cols].notna().astype(int).T.dot(selected[climate_cols].notna().astype(int))
    spearman = selected.corr(method="spearman", min_periods=MIN_PAIR_COUNT).loc[non_climate_cols, climate_cols]
    group_lookup = role_table.set_index("column")["concept_group"].to_dict()

    rows = []
    for concept_col in non_climate_cols:
        best_feature = ""
        best_value = math.nan
        best_n = 0
        for climate_col in climate_cols:
            value = spearman.loc[concept_col, climate_col]
            if pd.isna(value):
                continue
            abs_value = abs(float(value))
            if pd.isna(best_value) or abs_value > best_value:
                best_value = abs_value
                best_feature = climate_col
                best_n = int(pair_counts.loc[concept_col, climate_col])
        rows.append(
            {
                "concept_feature": concept_col,
                "concept_group": group_lookup.get(concept_col, "unknown"),
                "best_climate_concept_spearman": best_feature,
                "spearman_abs_max": best_value,
                "n_pair_at_best": best_n,
                "is_climate_sensitive": bool(pd.notna(best_value) and best_value >= threshold),
            }
        )
    return pd.DataFrame(rows).sort_values("spearman_abs_max", ascending=False)


@dataclass
class ClimateResidualizer:
    target_cols: list[str]
    climate_cols: list[str]
    alpha: float = 10.0

    def fit(self, x: pd.DataFrame) -> "ClimateResidualizer":
        self.target_medians_ = x[self.target_cols].median(numeric_only=True)
        y = x[self.target_cols].fillna(self.target_medians_).to_numpy()
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
        observed = x[self.target_cols].fillna(self.target_medians_).to_numpy()
        residuals = observed - predicted
        return pd.DataFrame(
            residuals,
            index=x.index,
            columns=[f"{column}_climate_resid" for column in self.target_cols],
        )


def build_features(
    df: pd.DataFrame,
    role_table: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    model_key: str,
    threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], pd.DataFrame]:
    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()
    empty_sensitive = pd.DataFrame()

    if model_key in {"Concept_M1_All_Concepts", "Concept_M2_No_Climate_Concepts"}:
        cols = feature_columns(role_table, model_key)
        return train_df[cols], test_df[cols], cols, empty_sensitive

    climate_cols = climate_columns(role_table)
    non_climate_cols = feature_columns(role_table, model_key)
    if not climate_cols:
        return train_df[non_climate_cols], test_df[non_climate_cols], non_climate_cols, empty_sensitive

    if model_key == "Concept_M3_All_NonClimate_Residualized":
        residual_cols = non_climate_cols
        raw_cols: list[str] = []
        sensitive = empty_sensitive
    elif model_key == "Concept_M4_Sensitive_Residualized":
        sensitive = local_climate_sensitive_concepts(train_df, role_table, threshold)
        residual_cols = sensitive.loc[sensitive["is_climate_sensitive"], "concept_feature"].tolist()
        residual_set = set(residual_cols)
        raw_cols = [c for c in non_climate_cols if c not in residual_set]
    else:
        raise ValueError(model_key)

    if not residual_cols:
        return train_df[raw_cols], test_df[raw_cols], raw_cols, sensitive

    residualizer = ClimateResidualizer(target_cols=residual_cols, climate_cols=climate_cols)
    residualizer.fit(train_df)
    train_resid = residualizer.transform(train_df)
    test_resid = residualizer.transform(test_df)
    x_train = pd.concat([train_resid, train_df[raw_cols]], axis=1)
    x_test = pd.concat([test_resid, test_df[raw_cols]], axis=1)
    cols = list(train_resid.columns) + raw_cols
    return x_train, x_test, cols, sensitive


def add_result_rows(
    rows: list[dict],
    predictions: list[dict],
    *,
    dataset_name: str,
    cv_name: str,
    fold: int,
    test_group: str | None,
    model_name: str,
    feature_columns_used: list[str],
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
            "experiment": "concept_climate_decoupling",
            "cv": cv_name,
            "fold": int(fold),
            "test_group": test_group,
            "model": model_name,
            "n_features": int(len(feature_columns_used)),
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
                "experiment": "concept_climate_decoupling",
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


def run_concept_decoupling(
    df: pd.DataFrame,
    role_table: pd.DataFrame,
    dataset_name: str,
    threshold: float,
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
            for model_key in CONCEPT_MODELS:
                x_train, x_test, used_cols, sensitive = build_features(
                    df=df,
                    role_table=role_table,
                    train_idx=train_idx,
                    test_idx=test_idx,
                    model_key=model_key,
                    threshold=threshold,
                )
                if x_train.shape[1] == 0:
                    continue
                if model_key == "Concept_M4_Sensitive_Residualized" and not sensitive.empty:
                    temp = sensitive.copy()
                    temp.insert(0, "dataset", dataset_name)
                    temp.insert(1, "cv", cv_name)
                    temp.insert(2, "fold", fold)
                    temp.insert(3, "test_group", test_group)
                    sensitive_rows.append(temp)
                model = make_rf_model()
                model.fit(x_train, y.iloc[train_idx])
                y_prob = model.predict_proba(x_test)[:, 1]
                add_result_rows(
                    rows,
                    predictions,
                    dataset_name=dataset_name,
                    cv_name=cv_name,
                    fold=fold,
                    test_group=test_group,
                    model_name=model_key,
                    feature_columns_used=used_cols,
                    train_idx=train_idx,
                    test_idx=test_idx,
                    df=df,
                    y_test=y.iloc[test_idx],
                    y_prob=y_prob,
                )

    sensitive_all = pd.concat(sensitive_rows, ignore_index=True) if sensitive_rows else pd.DataFrame()
    return pd.DataFrame(rows), pd.DataFrame(predictions), sensitive_all


def compact_summary(summary: pd.DataFrame) -> pd.DataFrame:
    cols = [
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
    out = summary[[c for c in cols if c in summary.columns]].copy()
    for col in out.columns:
        if col.endswith("_mean"):
            out[col] = out[col].round(4)
    return out


def write_report(
    out_dir: Path,
    dataset_name: str,
    concept_path: Path,
    df: pd.DataFrame,
    role_table: pd.DataFrame,
    summary: pd.DataFrame,
    threshold: float,
    validation_mode: str,
) -> None:
    concept_count = int(role_table.shape[0])
    climate_count = int(role_table["is_climate_concept"].sum())
    non_climate_count = concept_count - climate_count
    label_counts = df[TARGET].value_counts(dropna=False).sort_index().to_dict()
    report = f"""# Concept Climate Decoupling Report: {dataset_name}

## Dataset

| Item | Value |
|---|---:|
| Rows | {len(df)} |
| Concept features | {concept_count} |
| Climate concepts | {climate_count} |
| Non-climate concepts | {non_climate_count} |
| Positive samples | {int(label_counts.get(1, 0))} |
| Negative samples | {int(label_counts.get(0, 0))} |

Input concept table:

```text
{concept_path}
```

Validation mode: `{validation_mode}`

M4 fold-local climate sensitivity threshold:

```text
|Spearman r| >= {threshold}
```

## Model Definitions

| Model | Meaning |
|---|---|
| Concept_M1_All_Concepts | Use all concept features, including climate concepts. |
| Concept_M2_No_Climate_Concepts | Remove climate concepts and use only non-climate concepts. |
| Concept_M3_All_NonClimate_Residualized | Residualize all non-climate concepts against climate concepts. |
| Concept_M4_Sensitive_Residualized | Residualize only fold-local climate-sensitive non-climate concepts. |

## Metric Summary

{markdown_table(compact_summary(summary))}

## Notes

- This experiment works at the concept-feature level, not the raw high-dimensional feature level.
- M3 and M4 use only training-fold information when fitting residual models and selecting climate-sensitive concepts.
- Use `groupkfold_state` as the main spatial generalization reference when comparing with full-feature climate decoupling.
"""
    (out_dir / "concept_climate_decoupling_report.md").write_text(report, encoding="utf-8")


def run_one_dataset(
    dataset_path: Path,
    dataset_name: str | None,
    output_root: Path,
    threshold: float,
    validation_mode: str,
) -> pd.DataFrame:
    name = dataset_name_from_path(dataset_path, dataset_name)
    concept_path = concept_path_from_dataset(dataset_path, output_root, name)
    if not concept_path.exists():
        raise FileNotFoundError(
            f"Concept feature table not found: {concept_path}. "
            "Run run_standard_causal_graph_workflow.ps1 for this dataset first."
        )

    out_dir = output_root / name / "04_concept_climate_decoupling"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_table(concept_path).reset_index(drop=True)
    if TARGET not in df.columns:
        raise ValueError(f"{concept_path} does not contain {TARGET}.")

    role_table = concept_role_table(df, dictionary_path_from_concept_path(concept_path))
    metrics, predictions, sensitive = run_concept_decoupling(
        df=df,
        role_table=role_table,
        dataset_name=name,
        threshold=threshold,
        validation_mode=validation_mode,
    )
    summary = summarize_metrics(metrics)

    write_table(role_table, out_dir / "00_concept_feature_roles.csv")
    write_table(metrics, out_dir / "01_concept_climate_decoupling_metrics.csv")
    write_table(predictions, out_dir / "02_concept_climate_decoupling_predictions.csv")
    write_table(summary, out_dir / "03_concept_climate_decoupling_summary.csv")
    if not sensitive.empty:
        write_table(sensitive, out_dir / "04_m4_fold_local_sensitive_concepts.csv")

    manifest = {
        "dataset": name,
        "input_dataset_path": str(dataset_path),
        "input_concept_path": str(concept_path),
        "output_dir": str(out_dir),
        "validation_mode": validation_mode,
        "m4_threshold": threshold,
        "rows": int(len(df)),
        "concept_features": int(role_table.shape[0]),
        "climate_concepts": int(role_table["is_climate_concept"].sum()),
        "models": CONCEPT_MODELS,
    }
    (out_dir / "concept_climate_decoupling_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(out_dir, name, concept_path, df, role_table, summary, threshold, validation_mode)
    print(f"Wrote concept climate decoupling outputs: {out_dir}")
    return summary.assign(run_dir=name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run concept-level climate decoupling experiments.")
    parser.add_argument("--datasets", nargs="+", required=True, help="Model dataset paths or concept feature table paths.")
    parser.add_argument("--dataset-name", default=None, help="Optional dataset name for a single input dataset.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Standardized run output root.")
    parser.add_argument("--m4-threshold", type=float, default=DEFAULT_M4_THRESHOLD)
    parser.add_argument("--validation-mode", choices=["core", "exhaustive"], default="core")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    summaries = []
    for raw_path in args.datasets:
        dataset_path = Path(raw_path)
        summary = run_one_dataset(
            dataset_path=dataset_path,
            dataset_name=args.dataset_name if len(args.datasets) == 1 else None,
            output_root=output_root,
            threshold=args.m4_threshold,
            validation_mode=args.validation_mode,
        )
        summaries.append(summary)

    if summaries:
        combined = pd.concat(summaries, ignore_index=True)
        cols = ["run_dir"] + [c for c in combined.columns if c != "run_dir"]
        combined = combined[cols]
        write_table(combined, output_root / "concept_climate_decoupling_all_ratio_summary.csv")
        print(f"Wrote combined summary: {output_root / 'concept_climate_decoupling_all_ratio_summary.csv'}")


if __name__ == "__main__":
    main()
