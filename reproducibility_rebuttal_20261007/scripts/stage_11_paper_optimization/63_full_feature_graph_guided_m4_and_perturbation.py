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

MODEL_KEYS = [
    "Full_RF",
    "CoreGeo_RF",
    "NoClimate_RF",
    "M4_Spearman_RF",
    "M4_GraphGuided_RF",
    "M4_GraphUnion_RF",
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


def summarize_perturbation(perturb: pd.DataFrame) -> pd.DataFrame:
    value_cols = [
        "mean_abs_delta",
        "median_abs_delta",
        "p95_abs_delta",
        "max_abs_delta",
        "frac_abs_delta_gt_0_05",
        "roc_auc",
        "average_precision",
        "balanced_accuracy",
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
    summary = perturb.groupby(["dataset", "cv", "model", "scenario"], dropna=False)[value_cols].agg(
        ["mean", "std", "count"]
    ).reset_index()
    summary.columns = [
        "_".join(str(part) for part in col if str(part))
        if isinstance(col, tuple)
        else str(col)
        for col in summary.columns
    ]
    return summary


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


def load_dataset_from_run(run_dir: Path) -> tuple[pd.DataFrame, Path]:
    manifest_path = run_dir / "standard_workflow_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        input_path = Path(manifest["input_path"])
        if not input_path.is_absolute():
            input_path = (run_dir / input_path).resolve()
        if input_path.exists():
            return read_table(input_path), input_path
    fallback = next((run_dir.glob("**/model_dataset*.parquet")), None)
    if fallback is None:
        fallback = next((run_dir.glob("**/model_dataset*.csv")), None)
    if fallback is None:
        raise FileNotFoundError(f"Cannot find input dataset for run: {run_dir}")
    return read_table(fallback), fallback


def feature_sets(role_table: pd.DataFrame) -> dict[str, list[str]]:
    numeric = role_table[role_table["is_numeric"]].copy()
    full = numeric.loc[numeric["used_in_m1_full_climate"], "column"].tolist()
    no_climate = numeric.loc[numeric["used_in_m2_no_climate"], "column"].tolist()
    core_geo = numeric.loc[numeric["role"].isin(["geochemistry", "geo_structure"]), "column"].tolist()
    climate = numeric.loc[numeric["used_as_climate_adjuster"], "column"].tolist()
    geochem = numeric.loc[numeric["role"] == "geochemistry", "column"].tolist()
    return {
        "full": full,
        "no_climate": no_climate,
        "core_geo": core_geo,
        "climate_adjusters": climate,
        "geochemistry": geochem,
    }


def selected_graph_target_columns(
    mapping_path: Path,
    graph_targets_path: Path,
    role_table: pd.DataFrame,
) -> pd.DataFrame:
    mapping = pd.read_csv(mapping_path)
    targets = pd.read_csv(graph_targets_path)
    selected_concepts = set(
        targets.loc[targets["selected_for_graph_guided_m4"].astype(bool), "target_concept"].astype(str)
    )
    role_lookup = role_table.set_index("column").to_dict(orient="index")
    rows = []
    mapped = mapping[
        mapping["mapped"].astype(bool)
        & mapping["include_in_concept_features"].astype(bool)
        & mapping["concept"].astype(str).isin(selected_concepts)
    ].copy()
    for _, row in mapped.iterrows():
        column = str(row["feature_name"])
        role = role_lookup.get(column)
        if not role:
            continue
        if not bool(role.get("is_numeric")):
            continue
        n_unique = int(role.get("n_unique", 0))
        # Binary geology flags remain raw; residualization is reserved for continuous observations.
        residualizable = n_unique > 2 and str(role.get("role")) in {"geochemistry", "geo_structure"}
        rows.append(
            {
                "column": column,
                "concept": str(row["concept"]),
                "concept_group": str(row["concept_group"]),
                "raw_role": str(role.get("role")),
                "n_unique": n_unique,
                "residualizable": residualizable,
            }
        )
    if not rows:
        return pd.DataFrame(columns=["column", "concept", "concept_group", "raw_role", "n_unique", "residualizable"])
    return pd.DataFrame(rows).drop_duplicates("column").sort_values(["concept_group", "concept", "column"])


def local_spearman_sensitive(
    train_df: pd.DataFrame,
    target_cols: list[str],
    climate_cols: list[str],
    threshold: float,
) -> pd.DataFrame:
    if not target_cols or not climate_cols:
        return pd.DataFrame()
    selected = train_df[target_cols + climate_cols]
    pair_counts = selected[target_cols].notna().astype(int).T.dot(selected[climate_cols].notna().astype(int))
    spearman = selected.corr(method="spearman", min_periods=MIN_PAIR_COUNT).loc[target_cols, climate_cols]
    rows = []
    for target in target_cols:
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
                "best_climate_column": best_climate,
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


@dataclass
class FeatureRecipe:
    model_key: str
    raw_cols: list[str]
    residual_cols: list[str]
    climate_cols: list[str]
    residualizer: ClimateResidualizer | None = None

    def fit(self, train_df: pd.DataFrame) -> "FeatureRecipe":
        if self.residual_cols and self.climate_cols:
            self.residualizer = ClimateResidualizer(self.residual_cols, self.climate_cols)
            self.residualizer.fit(train_df)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        parts = []
        if self.residualizer is not None:
            parts.append(self.residualizer.transform(df))
        if self.raw_cols:
            parts.append(df[self.raw_cols])
        if not parts:
            return pd.DataFrame(index=df.index)
        return pd.concat(parts, axis=1)

    def output_columns(self) -> list[str]:
        residual_names = [f"{c}_climate_resid" for c in self.residual_cols] if self.residualizer is not None else []
        return residual_names + self.raw_cols


def make_recipe(
    model_key: str,
    train_df: pd.DataFrame,
    fsets: dict[str, list[str]],
    graph_residual_cols: set[str],
    spearman_threshold: float,
) -> tuple[FeatureRecipe, pd.DataFrame]:
    full = fsets["full"]
    no_climate = fsets["no_climate"]
    core_geo = fsets["core_geo"]
    climate = fsets["climate_adjusters"]
    geochem = fsets["geochemistry"]
    selection = pd.DataFrame()

    if model_key == "Full_RF":
        return FeatureRecipe(model_key, full, [], climate).fit(train_df), selection
    if model_key == "CoreGeo_RF":
        return FeatureRecipe(model_key, core_geo, [], climate).fit(train_df), selection
    if model_key == "NoClimate_RF":
        return FeatureRecipe(model_key, no_climate, [], climate).fit(train_df), selection

    sensitive = local_spearman_sensitive(train_df, geochem, climate, spearman_threshold)
    spearman_cols = set(sensitive.loc[sensitive["is_spearman_sensitive"], "target_column"].tolist())
    if model_key == "M4_Spearman_RF":
        residual_cols = spearman_cols
    elif model_key == "M4_GraphGuided_RF":
        residual_cols = graph_residual_cols
    elif model_key == "M4_GraphUnion_RF":
        residual_cols = graph_residual_cols | spearman_cols
    else:
        raise ValueError(model_key)

    residual_cols_ordered = [c for c in no_climate if c in residual_cols]
    raw_cols = [c for c in no_climate if c not in set(residual_cols_ordered)]
    selection = sensitive.copy()
    selection["selected_by_spearman"] = selection["target_column"].isin(spearman_cols)
    selection["selected_by_graph"] = selection["target_column"].isin(graph_residual_cols)
    selection["selected_by_model"] = selection["target_column"].isin(residual_cols_ordered)
    selection["model"] = model_key
    return FeatureRecipe(model_key, raw_cols, residual_cols_ordered, climate).fit(train_df), selection


def perturb_climate(df: pd.DataFrame, climate_cols: list[str], scenario: str, train_medians: pd.Series) -> pd.DataFrame:
    out = df.copy()
    temp_tokens = ("tmean", "tmax", "tmin", "dtr", "temperature")
    if scenario == "original":
        return out
    if scenario == "climate_plus":
        for col in climate_cols:
            if any(token in col.lower() for token in temp_tokens):
                out[col] = out[col] + 1.0
            else:
                out[col] = out[col] * 1.10
        return out
    if scenario == "climate_minus":
        for col in climate_cols:
            if any(token in col.lower() for token in temp_tokens):
                out[col] = out[col] - 1.0
            else:
                out[col] = out[col] * 0.90
        return out
    if scenario == "climate_train_median":
        for col in climate_cols:
            out[col] = train_medians.get(col, out[col].median())
        return out
    raise ValueError(scenario)


def add_metric_row(
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
        "experiment": "full_feature_graph_guided_m4",
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
                "experiment": "full_feature_graph_guided_m4",
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


def add_perturbation_row(
    rows: list[dict],
    *,
    dataset: str,
    cv_name: str,
    fold: int,
    test_group: str | None,
    model_key: str,
    scenario: str,
    y_true: np.ndarray,
    original_prob: np.ndarray,
    perturbed_prob: np.ndarray,
) -> None:
    delta = perturbed_prob - original_prob
    y_pred = (perturbed_prob >= DECISION_THRESHOLD).astype(int)
    row = {
        "dataset": dataset,
        "cv": cv_name,
        "fold": int(fold),
        "test_group": test_group,
        "model": model_key,
        "scenario": scenario,
        "n_test": int(len(y_true)),
        "mean_delta": float(np.mean(delta)),
        "mean_abs_delta": float(np.mean(np.abs(delta))),
        "median_abs_delta": float(np.median(np.abs(delta))),
        "p95_abs_delta": float(np.quantile(np.abs(delta), 0.95)),
        "max_abs_delta": float(np.max(np.abs(delta))),
        "frac_abs_delta_gt_0_05": float(np.mean(np.abs(delta) > 0.05)),
        "roc_auc": safe_metric("roc_auc", y_true, y_pred, perturbed_prob),
        "average_precision": safe_metric("average_precision", y_true, y_pred, perturbed_prob),
        "balanced_accuracy": safe_metric("balanced_accuracy", y_true, y_pred, perturbed_prob),
        "f1": safe_metric("f1", y_true, y_pred, perturbed_prob),
    }
    row.update(top_k_metrics(y_true, perturbed_prob))
    rows.append(row)


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


def write_report(out_dir: Path, dataset: str, summary: pd.DataFrame, perturb_summary: pd.DataFrame) -> None:
    metric_cols = [
        "dataset",
        "cv",
        "model",
        "roc_auc_mean",
        "average_precision_mean",
        "balanced_accuracy_mean",
        "precision_mean",
        "recall_mean",
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
    pert_cols = [
        "dataset",
        "cv",
        "model",
        "scenario",
        "mean_abs_delta_mean",
        "p95_abs_delta_mean",
        "frac_abs_delta_gt_0_05_mean",
        "average_precision_mean",
        "top05_precision_mean",
        "top05_recall_mean",
        "top05_f1_mean",
        "top05_lift_mean",
        "top05_ndcg_mean",
    ]
    report = [
        f"# Full-Feature Graph-Guided M4 and Climate Perturbation: {dataset}",
        "",
        "## Models",
        "",
        "| Model | Meaning |",
        "|---|---|",
        "| Full_RF | Full raw feature set including climate. |",
        "| CoreGeo_RF | Geochemistry + geophysics/geology/terrain/structure feature roles. |",
        "| NoClimate_RF | All numeric non-climate features. |",
        "| M4_Spearman_RF | Fold-local Spearman climate-sensitive geochemical features are residualized. |",
        "| M4_GraphGuided_RF | Raw features mapped to selected Climate Sensitivity Graph target concepts are residualized. |",
        "| M4_GraphUnion_RF | Union of Spearman-selected and graph-selected residual targets. |",
        "",
        "## Original Prediction Metrics",
        "",
        markdown_table(summary[[c for c in metric_cols if c in summary.columns]]),
        "",
        "## Climate Perturbation Stability",
        "",
        markdown_table(perturb_summary[[c for c in pert_cols if c in perturb_summary.columns]]),
        "",
        "## Perturbation Scenarios",
        "",
        "- `climate_plus`: non-temperature climate adjusters +10%; temperature-like variables +1 degree.",
        "- `climate_minus`: non-temperature climate adjusters -10%; temperature-like variables -1 degree.",
        "- `climate_train_median`: climate adjusters replaced by training-fold medians.",
    ]
    (out_dir / "full_feature_graph_guided_m4_report.md").write_text("\n".join(report), encoding="utf-8")


def run_experiment(
    run_dir: Path,
    graph_dir: Path,
    output_dir: Path,
    spearman_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset = run_dir.name
    df, dataset_path = load_dataset_from_run(run_dir)
    df = df[df[TARGET].isin([0, 1])].reset_index(drop=True)
    role_table = pd.read_csv(run_dir / "00_dataset_profile" / "feature_roles.csv")
    mapping_path = run_dir / "03_causal_graph" / "02_feature_to_concept_resolved.csv"
    graph_targets_path = graph_dir / "climate_sensitive_target_concepts.csv"
    graph_cols = selected_graph_target_columns(mapping_path, graph_targets_path, role_table)
    graph_residual_cols = set(graph_cols.loc[graph_cols["residualizable"], "column"].tolist())
    fsets = feature_sets(role_table)

    metrics_rows: list[dict] = []
    pred_rows: list[dict] = []
    perturb_rows: list[dict] = []
    selection_rows: list[pd.DataFrame] = []
    y = df[TARGET].astype(int).reset_index(drop=True)

    for cv_name, splitter, groups in make_splitters(df):
        split_iter = splitter.split(df, y, groups) if groups is not None else splitter.split(df, y)
        for fold, (train_idx, test_idx) in enumerate(split_iter, start=1):
            train_idx = np.asarray(train_idx)
            test_idx = np.asarray(test_idx)
            train_df = df.iloc[train_idx].copy()
            test_df = df.iloc[test_idx].copy()
            y_train = y.iloc[train_idx]
            y_true = y.iloc[test_idx].to_numpy()
            test_group = None
            if groups is not None:
                test_group = ";".join(sorted(pd.Series(groups).iloc[test_idx].astype(str).unique().tolist()))
            climate_cols = fsets["climate_adjusters"]
            train_medians = train_df[climate_cols].median(numeric_only=True) if climate_cols else pd.Series(dtype=float)

            for model_key in MODEL_KEYS:
                recipe, selection = make_recipe(
                    model_key,
                    train_df,
                    fsets,
                    graph_residual_cols,
                    spearman_threshold,
                )
                if not selection.empty:
                    temp = selection.copy()
                    temp.insert(0, "dataset", dataset)
                    temp.insert(1, "cv", cv_name)
                    temp.insert(2, "fold", fold)
                    temp.insert(3, "test_group", test_group)
                    selection_rows.append(temp)
                x_train = recipe.transform(train_df)
                x_test = recipe.transform(test_df)
                model = make_rf_model()
                model.fit(x_train, y_train)
                y_prob = model.predict_proba(x_test)[:, 1]
                add_metric_row(
                    metrics_rows,
                    pred_rows,
                    dataset=dataset,
                    cv_name=cv_name,
                    fold=fold,
                    test_group=test_group,
                    model_key=model_key,
                    used_cols=recipe.output_columns(),
                    train_idx=train_idx,
                    test_idx=test_idx,
                    df=df,
                    y_prob=y_prob,
                )
                for scenario in ["climate_plus", "climate_minus", "climate_train_median"]:
                    perturbed_test = perturb_climate(test_df, climate_cols, scenario, train_medians)
                    x_perturbed = recipe.transform(perturbed_test)
                    perturbed_prob = model.predict_proba(x_perturbed)[:, 1]
                    add_perturbation_row(
                        perturb_rows,
                        dataset=dataset,
                        cv_name=cv_name,
                        fold=fold,
                        test_group=test_group,
                        model_key=model_key,
                        scenario=scenario,
                        y_true=y_true,
                        original_prob=y_prob,
                        perturbed_prob=perturbed_prob,
                    )

    metrics = pd.DataFrame(metrics_rows)
    predictions = pd.DataFrame(pred_rows)
    perturb = pd.DataFrame(perturb_rows)
    selections = pd.concat(selection_rows, ignore_index=True) if selection_rows else pd.DataFrame()
    summary = summarize_metrics(metrics)
    perturb_summary = summarize_perturbation(perturb)

    write_table(metrics, output_dir / "full_feature_graph_guided_m4_metrics.csv")
    write_table(predictions, output_dir / "full_feature_graph_guided_m4_predictions.csv")
    write_table(summary, output_dir / "full_feature_graph_guided_m4_summary.csv")
    write_table(perturb, output_dir / "climate_perturbation_metrics.csv")
    write_table(perturb_summary, output_dir / "climate_perturbation_summary.csv")
    write_table(graph_cols, output_dir / "graph_guided_raw_feature_targets.csv")
    write_table(selections, output_dir / "m4_spearman_selection_by_fold.csv")
    write_report(output_dir, dataset, summary, perturb_summary)

    manifest = {
        "dataset": dataset,
        "input_dataset_path": str(dataset_path),
        "run_dir": str(run_dir),
        "graph_dir": str(graph_dir),
        "output_dir": str(output_dir),
        "spearman_threshold": spearman_threshold,
        "graph_mapped_feature_count": int(len(graph_cols)),
        "graph_residual_feature_count": int(len(graph_residual_cols)),
        "models": MODEL_KEYS,
        "climate_perturbation_scenarios": ["climate_plus", "climate_minus", "climate_train_median"],
    }
    (output_dir / "full_feature_graph_guided_m4_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary, perturb_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full-feature Graph-guided M4 and climate perturbation evaluation.")
    parser.add_argument("--run-dir", required=True, help="Standardized run directory.")
    parser.add_argument("--graph-dir", required=True, help="Climate Sensitivity Graph output directory.")
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
        else PROJECT_ROOT / "outputs" / "paper_optimization" / "full_feature_graph_guided_m4" / run_dir.name
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    run_experiment(run_dir, graph_dir, output_dir, args.spearman_threshold)
    print(f"Wrote full-feature Graph-guided M4 outputs to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
