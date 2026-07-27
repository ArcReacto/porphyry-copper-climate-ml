from __future__ import annotations

import argparse
import json
import math
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
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TARGET = "Y_label"
RANDOM_STATE = 20260622
DECISION_THRESHOLD = 0.5
MIN_PAIR_COUNT = 30
RF_N_ESTIMATORS = 300

MODELS = [
    "Concept_M1_All",
    "Concept_M2_No_Climate",
    "Concept_M4_Spearman",
    "Concept_M4_GraphGuided",
    "Concept_M4_GraphUnion",
]


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


def concept_name(column: str) -> str:
    return column.replace("concept_", "", 1)


def concept_column(name: str) -> str:
    return name if name.startswith("concept_") else f"concept_{name}"


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


def load_role_table(dictionary_path: Path, concept_df: pd.DataFrame) -> pd.DataFrame:
    dictionary = pd.read_csv(dictionary_path)
    group_lookup = dictionary.set_index("concept")["concept_group"].astype(str).to_dict()
    rows = []
    for col in sorted(c for c in concept_df.columns if c.startswith("concept_")):
        concept = concept_name(col)
        group = group_lookup.get(concept, "unknown")
        rows.append(
            {
                "column": col,
                "concept": concept,
                "concept_group": group,
                "is_climate": group == "climate",
            }
        )
    return pd.DataFrame(rows)


def make_splitters(df: pd.DataFrame) -> list[tuple[str, object, pd.Series | None]]:
    y = df[TARGET].astype(int)
    positives = int(y.sum())
    negatives = int((y == 0).sum())
    n_splits = min(5, positives, negatives)
    splitters: list[tuple[str, object, pd.Series | None]] = [
        ("stratified_kfold", StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE), None)
    ]
    if "state" in df.columns and df["state"].nunique(dropna=True) >= 2:
        groups = df["state"].fillna("unknown").astype(str)
        splitters.append(("groupkfold_state", GroupKFold(n_splits=min(5, groups.nunique())), groups))
    return splitters


def local_spearman_sensitive(
    train_df: pd.DataFrame,
    non_climate_cols: list[str],
    climate_cols: list[str],
    threshold: float,
) -> pd.DataFrame:
    if not non_climate_cols or not climate_cols:
        return pd.DataFrame()
    selected = train_df[non_climate_cols + climate_cols]
    pair_counts = selected[non_climate_cols].notna().astype(int).T.dot(selected[climate_cols].notna().astype(int))
    spearman = selected.corr(method="spearman", min_periods=MIN_PAIR_COUNT).loc[non_climate_cols, climate_cols]
    rows = []
    for target in non_climate_cols:
        best_climate = ""
        best_abs = math.nan
        best_value = math.nan
        best_n = 0
        for climate in climate_cols:
            value = spearman.loc[target, climate]
            if pd.isna(value):
                continue
            abs_value = abs(float(value))
            if pd.isna(best_abs) or abs_value > best_abs:
                best_abs = abs_value
                best_value = float(value)
                best_climate = climate
                best_n = int(pair_counts.loc[target, climate])
        rows.append(
            {
                "target_column": target,
                "target_concept": concept_name(target),
                "best_climate_column": best_climate,
                "best_climate_concept": concept_name(best_climate) if best_climate else "",
                "spearman": best_value,
                "spearman_abs_max": best_abs,
                "n_pair_at_best": best_n,
                "is_spearman_sensitive": bool(pd.notna(best_abs) and best_abs >= threshold),
            }
        )
    return pd.DataFrame(rows).sort_values("spearman_abs_max", ascending=False)


@dataclass
class ClimateResidualizer:
    target_cols: list[str]
    climate_cols: list[str]
    alpha: float = 10.0

    def fit(self, df: pd.DataFrame) -> "ClimateResidualizer":
        self.target_medians_ = df[self.target_cols].median(numeric_only=True)
        y = df[self.target_cols].fillna(self.target_medians_).to_numpy()
        self.model_ = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=self.alpha)),
            ]
        )
        self.model_.fit(df[self.climate_cols], y)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        predicted = self.model_.predict(df[self.climate_cols])
        observed = df[self.target_cols].fillna(self.target_medians_).to_numpy()
        residuals = observed - predicted
        return pd.DataFrame(
            residuals,
            index=df.index,
            columns=[f"{column}_climate_resid" for column in self.target_cols],
        )


def build_features(
    df: pd.DataFrame,
    role_table: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    model_key: str,
    graph_targets: set[str],
    spearman_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], pd.DataFrame]:
    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()
    climate_cols = role_table.loc[role_table["is_climate"], "column"].tolist()
    non_climate_cols = role_table.loc[~role_table["is_climate"], "column"].tolist()

    if model_key == "Concept_M1_All":
        cols = climate_cols + non_climate_cols
        return train_df[cols], test_df[cols], cols, pd.DataFrame()
    if model_key == "Concept_M2_No_Climate":
        return train_df[non_climate_cols], test_df[non_climate_cols], non_climate_cols, pd.DataFrame()

    sensitive = local_spearman_sensitive(train_df, non_climate_cols, climate_cols, spearman_threshold)
    spearman_targets = set(sensitive.loc[sensitive["is_spearman_sensitive"], "target_column"].tolist())
    if model_key == "Concept_M4_Spearman":
        residual_targets = spearman_targets
    elif model_key == "Concept_M4_GraphGuided":
        residual_targets = graph_targets
    elif model_key == "Concept_M4_GraphUnion":
        residual_targets = graph_targets | spearman_targets
    else:
        raise ValueError(model_key)

    residual_cols = [c for c in non_climate_cols if c in residual_targets]
    raw_cols = [c for c in non_climate_cols if c not in set(residual_cols)]
    sensitive = sensitive.copy()
    sensitive["selected_by_graph"] = sensitive["target_column"].isin(graph_targets)
    sensitive["selected_by_model"] = sensitive["target_column"].isin(residual_cols)
    sensitive["model"] = model_key

    if not residual_cols or not climate_cols:
        return train_df[raw_cols], test_df[raw_cols], raw_cols, sensitive

    residualizer = ClimateResidualizer(target_cols=residual_cols, climate_cols=climate_cols)
    residualizer.fit(train_df)
    train_resid = residualizer.transform(train_df)
    test_resid = residualizer.transform(test_df)
    x_train = pd.concat([train_resid, train_df[raw_cols]], axis=1)
    x_test = pd.concat([test_resid, test_df[raw_cols]], axis=1)
    columns = list(train_resid.columns) + raw_cols
    return x_train, x_test, columns, sensitive


def evaluate_model(
    rows: list[dict],
    preds: list[dict],
    *,
    dataset: str,
    cv_name: str,
    fold: int,
    test_group: str | None,
    model_key: str,
    used_cols: list[str],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    df: pd.DataFrame,
    y_prob: np.ndarray,
) -> None:
    y_true = df.iloc[test_idx][TARGET].astype(int).to_numpy()
    y_pred = (y_prob >= DECISION_THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    row = {
        "dataset": dataset,
        "experiment": "graph_guided_concept_decoupling",
        "cv": cv_name,
        "fold": int(fold),
        "test_group": test_group,
        "model": model_key,
        "n_features": int(len(used_cols)),
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
        preds.append(
            {
                "dataset": dataset,
                "experiment": "graph_guided_concept_decoupling",
                "row_index": int(row_index),
                "sample_id": df.iloc[row_index].get("sample_id", ""),
                "cv": cv_name,
                "fold": int(fold),
                "test_group": test_group,
                "model": model_key,
                "Y_label": int(true_value),
                "y_pred": int(pred_value),
                "y_prob": float(prob_value),
                "state": df.iloc[row_index].get("state", ""),
                "negative_type": df.iloc[row_index].get("negative_type", ""),
                "env_causal_group_id": df.iloc[row_index].get("env_causal_group_id", ""),
                "env_weathering_regime": df.iloc[row_index].get("env_weathering_regime", ""),
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
    summary = metrics.groupby(["dataset", "experiment", "cv", "model"], dropna=False)[metric_cols].agg(
        ["mean", "std", "count"]
    ).reset_index()
    summary.columns = [
        "_".join(str(part) for part in col if str(part))
        if isinstance(col, tuple)
        else str(col)
        for col in summary.columns
    ]
    return summary


def markdown_table(df: pd.DataFrame, max_rows: int = 80) -> str:
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


def write_report(
    out_dir: Path,
    dataset: str,
    summary: pd.DataFrame,
    graph_targets: set[str],
    spearman_threshold: float,
) -> None:
    display_cols = [
        "dataset",
        "cv",
        "model",
        "roc_auc_mean",
        "average_precision_mean",
        "balanced_accuracy_mean",
        "precision_mean",
        "recall_mean",
        "f1_mean",
        "top05_recall_mean",
        "top05_precision_mean",
        "top05_f1_mean",
        "top05_lift_mean",
        "top05_ndcg_mean",
        "top10_recall_mean",
        "top10_precision_mean",
        "top10_f1_mean",
        "top10_lift_mean",
        "top10_ndcg_mean",
    ]
    report = [
        f"# Graph-Guided Concept Climate Decoupling: {dataset}",
        "",
        "## Compared Models",
        "",
        "| Model | Meaning |",
        "|---|---|",
        "| Concept_M1_All | All concept features, including climate. |",
        "| Concept_M2_No_Climate | Climate concepts removed. |",
        "| Concept_M4_Spearman | Residualize fold-local Spearman climate-sensitive concepts. |",
        "| Concept_M4_GraphGuided | Residualize targets selected by the climate sensitivity graph. |",
        "| Concept_M4_GraphUnion | Residualize concepts selected by either graph or fold-local Spearman. |",
        "",
        f"Spearman threshold: `{spearman_threshold}`",
        "",
        f"Graph-guided residual targets: `{len(graph_targets)}`",
        "",
        "## Summary",
        "",
        markdown_table(summary[[c for c in display_cols if c in summary.columns]]),
    ]
    (out_dir / "graph_guided_concept_decoupling_report.md").write_text("\n".join(report), encoding="utf-8")


def run_experiment(
    run_dir: Path,
    graph_dir: Path,
    output_dir: Path,
    spearman_threshold: float,
) -> pd.DataFrame:
    dataset = run_dir.name
    concept_path = run_dir / "03_causal_graph" / "04_concept_features.parquet"
    dictionary_path = run_dir / "03_causal_graph" / "05_concept_feature_dictionary.csv"
    graph_targets_path = graph_dir / "climate_sensitive_target_concepts.csv"

    df = read_table(concept_path).reset_index(drop=True)
    df = df[df[TARGET].isin([0, 1])].reset_index(drop=True)
    role_table = load_role_table(dictionary_path, df)
    target_summary = pd.read_csv(graph_targets_path)
    graph_targets = set(
        concept_column(name)
        for name in target_summary.loc[
            target_summary["selected_for_graph_guided_m4"].astype(bool),
            "target_concept",
        ].astype(str)
    )

    y = df[TARGET].astype(int).reset_index(drop=True)
    metrics_rows: list[dict] = []
    pred_rows: list[dict] = []
    selection_rows: list[pd.DataFrame] = []

    for cv_name, splitter, groups in make_splitters(df):
        split_iter = splitter.split(df, y, groups) if groups is not None else splitter.split(df, y)
        for fold, (train_idx, test_idx) in enumerate(split_iter, start=1):
            train_idx = np.asarray(train_idx)
            test_idx = np.asarray(test_idx)
            test_group = None
            if groups is not None:
                test_group = ";".join(sorted(pd.Series(groups).iloc[test_idx].astype(str).unique().tolist()))
            for model_key in MODELS:
                x_train, x_test, used_cols, sensitive = build_features(
                    df=df,
                    role_table=role_table,
                    train_idx=train_idx,
                    test_idx=test_idx,
                    model_key=model_key,
                    graph_targets=graph_targets,
                    spearman_threshold=spearman_threshold,
                )
                if not sensitive.empty:
                    temp = sensitive.copy()
                    temp.insert(0, "dataset", dataset)
                    temp.insert(1, "cv", cv_name)
                    temp.insert(2, "fold", fold)
                    temp.insert(3, "test_group", test_group)
                    selection_rows.append(temp)
                model = make_rf_model()
                model.fit(x_train, y.iloc[train_idx])
                y_prob = model.predict_proba(x_test)[:, 1]
                evaluate_model(
                    metrics_rows,
                    pred_rows,
                    dataset=dataset,
                    cv_name=cv_name,
                    fold=fold,
                    test_group=test_group,
                    model_key=model_key,
                    used_cols=used_cols,
                    train_idx=train_idx,
                    test_idx=test_idx,
                    df=df,
                    y_prob=y_prob,
                )

    metrics = pd.DataFrame(metrics_rows)
    predictions = pd.DataFrame(pred_rows)
    selections = pd.concat(selection_rows, ignore_index=True) if selection_rows else pd.DataFrame()
    summary = summarize_metrics(metrics)

    write_table(metrics, output_dir / "graph_guided_concept_metrics.csv")
    write_table(predictions, output_dir / "graph_guided_concept_predictions.csv")
    write_table(summary, output_dir / "graph_guided_concept_summary.csv")
    write_table(selections, output_dir / "graph_guided_concept_selection_by_fold.csv")
    write_table(
        pd.DataFrame({"target_column": sorted(graph_targets), "target_concept": [concept_name(c) for c in sorted(graph_targets)]}),
        output_dir / "graph_guided_residual_targets.csv",
    )
    write_report(output_dir, dataset, summary, graph_targets, spearman_threshold)

    manifest = {
        "run_dir": str(run_dir),
        "graph_dir": str(graph_dir),
        "output_dir": str(output_dir),
        "spearman_threshold": spearman_threshold,
        "graph_target_count": len(graph_targets),
        "models": MODELS,
    }
    (output_dir / "graph_guided_concept_decoupling_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run graph-guided concept climate decoupling.")
    parser.add_argument("--run-dir", required=True, help="Standardized run directory.")
    parser.add_argument("--graph-dir", required=True, help="Output directory from 61_build_climate_sensitivity_graph.py.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--spearman-threshold", type=float, default=0.30)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir)
    graph_dir = Path(args.graph_dir)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else PROJECT_ROOT / "outputs" / "paper_optimization" / "graph_guided_concept_decoupling" / run_dir.name
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    run_experiment(run_dir, graph_dir, output_dir, args.spearman_threshold)
    print(f"Wrote graph-guided concept decoupling outputs to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
