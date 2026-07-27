from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
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
sys.path.insert(0, str(PROJECT_ROOT))

RANDOM_STATE = 20260622
TARGET = "Y_label"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "standardized_runs"
RF_N_ESTIMATORS = 300


def load_stage_module(name: str, filename: str):
    path = PROJECT_ROOT / "scripts" / "stage_05_causal_graph" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


stage15 = load_stage_module("stage15_feature_mapping", "15_create_feature_concept_mapping.py")
stage16 = load_stage_module("stage16_concept_features", "16_build_concept_features.py")
stage18 = load_stage_module("stage18_environment_stability", "18_environment_stability_analysis.py")
stage19 = load_stage_module("stage19_causal_discovery", "19_discover_environment_causal_graphs.py")
stage20 = load_stage_module("stage20_cross_environment", "20_compare_cross_environment_edges.py")


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


def dataset_name_from_path(path: Path, explicit_name: str | None) -> str:
    if explicit_name:
        return explicit_name
    stem = path.stem
    if stem.startswith("model_dataset_"):
        stem = stem.replace("model_dataset_", "", 1)
    return stem


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
    summary = metrics.groupby(["dataset", "cv", "model"], dropna=False)[metric_cols].agg(["mean", "std", "count"]).reset_index()
    summary.columns = [
        "_".join([str(part) for part in col if str(part)])
        if isinstance(col, tuple)
        else str(col)
        for col in summary.columns
    ]
    return summary


def make_models(pos_weight: float) -> dict[str, Pipeline]:
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


def predict_prob(model: Pipeline, x: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    if hasattr(model[-1], "decision_function"):
        score = model.decision_function(x)
        return 1.0 / (1.0 + np.exp(-score))
    return model.predict(x).astype(float)


def build_mapping(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    feature_cols = [c for c in df.columns if c not in META_COLUMNS]
    mappings = pd.DataFrame(stage15.mapping_rows())
    resolved = stage15.resolve_features(feature_cols, mappings)
    summary = (
        resolved.groupby(["concept_group", "concept"], as_index=False)
        .agg(feature_count=("feature_name", "count"), included_count=("include_in_concept_features", "sum"))
        .sort_values(["concept_group", "concept"])
    )
    write_table(mappings, out_dir / "01_feature_to_concept_mapping.csv")
    write_table(resolved, out_dir / "02_feature_to_concept_resolved.csv")
    write_table(summary, out_dir / "03_feature_to_concept_summary.csv")
    return resolved


def build_concept_features(df: pd.DataFrame, resolved: pd.DataFrame, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    mapping = resolved[(resolved["mapped"] == True) & (resolved["include_in_concept_features"] == True)].copy()  # noqa: E712
    meta_cols = [c for c in META_COLUMNS if c in df.columns]
    concept_df = df[meta_cols].copy()
    dictionary_rows = []
    for _, rows in mapping.groupby("concept", sort=True):
        score, item = stage16.aggregate_concept(df, rows)
        concept_df[f"concept_{item['concept']}"] = score
        dictionary_rows.append(item)

    dictionary = pd.DataFrame(dictionary_rows).sort_values(["concept_group", "concept"])
    concept_cols = [c for c in concept_df.columns if c.startswith("concept_")]
    missing = (
        concept_df[concept_cols]
        .isna()
        .mean()
        .rename("missing_rate")
        .reset_index()
        .rename(columns={"index": "concept_feature"})
        .sort_values("missing_rate", ascending=False)
    )
    write_table(concept_df, out_dir / "04_concept_features.parquet")
    write_table(concept_df, out_dir / "04_concept_features.csv")
    write_table(dictionary, out_dir / "05_concept_feature_dictionary.csv")
    write_table(missing, out_dir / "06_concept_feature_missing_rates.csv")
    return concept_df, dictionary


def train_concept_baselines(concept_df: pd.DataFrame, dataset_name: str, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    concept_cols = sorted([c for c in concept_df.columns if c.startswith("concept_")])
    y = concept_df[TARGET].astype(int).reset_index(drop=True)
    x = concept_df[concept_cols]
    pos = int(y.sum())
    neg = int((y == 0).sum())
    pos_weight = neg / max(pos, 1)
    n_splits = min(5, pos, neg)
    splitters: list[tuple[str, object, pd.Series | None]] = [
        ("stratified_kfold", StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE), None)
    ]
    if "state" in concept_df.columns and concept_df["state"].nunique(dropna=True) >= 2:
        groups = concept_df["state"].fillna("unknown").astype(str)
        splitters.append(("groupkfold_state", GroupKFold(n_splits=min(5, groups.nunique())), groups))

    rows = []
    predictions = []
    for model_name, model in make_models(pos_weight).items():
        for cv_name, splitter, groups in splitters:
            split_iter = splitter.split(x, y, groups) if groups is not None else splitter.split(x, y)
            for fold, (train_idx, test_idx) in enumerate(split_iter, start=1):
                train_idx = np.asarray(train_idx)
                test_idx = np.asarray(test_idx)
                model.fit(x.iloc[train_idx], y.iloc[train_idx])
                y_prob = predict_prob(model, x.iloc[test_idx])
                y_pred = (y_prob >= 0.5).astype(int)
                tn, fp, fn, tp = confusion_matrix(y.iloc[test_idx], y_pred, labels=[0, 1]).ravel()
                rows.append(
                    {
                        "dataset": dataset_name,
                        "cv": cv_name,
                        "model": model_name,
                        "fold": fold,
                        "n_train": int(len(train_idx)),
                        "n_test": int(len(test_idx)),
                        "positive_test": int(y.iloc[test_idx].sum()),
                        "negative_test": int((y.iloc[test_idx] == 0).sum()),
                        "roc_auc": safe_metric("roc_auc", y.iloc[test_idx].to_numpy(), y_pred, y_prob),
                        "average_precision": safe_metric("average_precision", y.iloc[test_idx].to_numpy(), y_pred, y_prob),
                        "balanced_accuracy": safe_metric("balanced_accuracy", y.iloc[test_idx].to_numpy(), y_pred, y_prob),
                        "precision": safe_metric("precision", y.iloc[test_idx].to_numpy(), y_pred, y_prob),
                        "recall": safe_metric("recall", y.iloc[test_idx].to_numpy(), y_pred, y_prob),
                        "f1": safe_metric("f1", y.iloc[test_idx].to_numpy(), y_pred, y_prob),
                        "tn": int(tn),
                        "fp": int(fp),
                        "fn": int(fn),
                        "tp": int(tp),
                    }
                )
                for row_index, true_value, pred_value, prob_value in zip(test_idx, y.iloc[test_idx], y_pred, y_prob):
                    predictions.append(
                        {
                            "dataset": dataset_name,
                            "row_index": int(row_index),
                            "sample_id": concept_df.iloc[row_index].get("sample_id", ""),
                            "cv": cv_name,
                            "fold": fold,
                            "model": model_name,
                            "Y_label": int(true_value),
                            "y_pred": int(pred_value),
                            "y_prob": float(prob_value),
                            "state": concept_df.iloc[row_index].get("state", ""),
                            "negative_type": concept_df.iloc[row_index].get("negative_type", ""),
                            "env_causal_group": concept_df.iloc[row_index].get("env_causal_group", ""),
                        }
                    )

    metrics = pd.DataFrame(rows)
    preds = pd.DataFrame(predictions)
    summary = summarize_metrics(metrics)
    write_table(metrics, out_dir / "07_concept_baseline_metrics.csv")
    write_table(preds, out_dir / "08_concept_baseline_predictions.csv")
    write_table(summary, out_dir / "09_concept_baseline_summary.csv")
    return metrics, summary


def concept_metadata(dictionary: pd.DataFrame) -> pd.DataFrame:
    return dictionary[["concept", "concept_group", "role", "feature_count", "aggregation_method"]].drop_duplicates("concept")


def run_environment_stability(
    concept_df: pd.DataFrame,
    dictionary: pd.DataFrame,
    out_dir: Path,
    bootstrap: int,
    top_k: int,
    selection_rate_threshold: float,
    auc_threshold: float,
    rf_top_k: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    concept_cols = sorted([c for c in concept_df.columns if c.startswith("concept_")])
    meta = concept_metadata(dictionary)
    rng = np.random.default_rng(RANDOM_STATE)
    effects = stage18.feature_effects(concept_df, concept_cols, meta)
    boot = stage18.bootstrap_selection(concept_df, concept_cols, bootstrap, top_k, rng)
    rf = stage18.environment_rf_importance(concept_df, concept_cols)
    logistic = stage18.environment_sparse_logistic(concept_df, concept_cols)
    stable = stage18.stable_signals(effects, boot, rf, logistic, meta, selection_rate_threshold, auc_threshold, rf_top_k)
    candidates = stable[stable["selected_environment_count"] >= 2].copy()
    candidates["recommended_role_for_step5"] = np.where(
        candidates["direction_consistent"],
        "candidate_predictor_node",
        "candidate_environment_sensitive_node",
    )

    write_table(effects, out_dir / "10_environment_feature_effects.csv")
    write_table(boot, out_dir / "11_environment_bootstrap_selection.csv")
    write_table(rf, out_dir / "12_environment_rf_importance.csv")
    write_table(logistic, out_dir / "13_environment_sparse_logistic.csv")
    write_table(stable, out_dir / "14_environment_stable_signals.csv")
    write_table(candidates, out_dir / "15_causal_graph_candidate_concepts.csv")
    return stable, candidates


def run_causal_discovery(
    concept_df: pd.DataFrame,
    candidates: pd.DataFrame,
    out_dir: Path,
    partial_threshold: float,
    p_threshold: float,
    max_edges: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected = candidates[candidates["selected_environment_count"] >= 2].copy()
    selected = selected.sort_values(["stable_score", "direction_consistent", "mean_auc_effect_strength"], ascending=False)
    all_edges = []
    summaries = []
    for env in stage19.MAIN_ENVIRONMENTS:
        env_df = concept_df[concept_df["env_causal_group"] == env].copy()
        n_pos = int(env_df[TARGET].sum()) if TARGET in env_df.columns else 0
        n_neg = int((env_df[TARGET] == 0).sum()) if TARGET in env_df.columns else 0
        if n_pos < stage19.MIN_POSITIVE or n_neg < stage19.MIN_NEGATIVE or selected.empty:
            summaries.append(
                {
                    "environment": env,
                    "rows": int(len(env_df)),
                    "positive": n_pos,
                    "negative": n_neg,
                    "candidate_concepts": int(len(selected)),
                    "edges": 0,
                    "status": "skipped_too_few_samples_or_no_candidates",
                }
            )
            continue
        edges = stage19.discover_environment_edges(env, env_df, selected, partial_threshold, p_threshold, max_edges)
        write_table(edges, out_dir / f"16_causal_edges_{env}.csv")
        all_edges.append(edges)
        summaries.append(
            {
                "environment": env,
                "rows": int(len(env_df)),
                "positive": n_pos,
                "negative": n_neg,
                "candidate_concepts": int(len(selected)),
                "edges": int(len(edges)),
                "status": "ok",
            }
        )

    all_edges_df = pd.concat(all_edges, ignore_index=True) if all_edges else pd.DataFrame()
    preliminary_overlap = stage19.summarize_overlap(all_edges_df) if not all_edges_df.empty else pd.DataFrame()
    write_table(all_edges_df, out_dir / "17_causal_edges_all_environments.csv")
    write_table(preliminary_overlap, out_dir / "18_causal_edges_preliminary_overlap.csv")
    write_table(pd.DataFrame(summaries), out_dir / "19_causal_discovery_environment_summary.csv")

    if all_edges_df.empty:
        cross_summary = pd.DataFrame()
        stable_edges = pd.DataFrame()
        kg = pd.DataFrame()
    else:
        cross_summary = stage20.build_cross_environment_summary(all_edges_df)
        stable_edges = cross_summary[cross_summary["environment_count"] >= 2].copy()
        kg = stage20.build_kg_candidates(stable_edges)
    write_table(cross_summary, out_dir / "20_cross_environment_edge_all_summary.csv")
    write_table(stable_edges, out_dir / "21_cross_environment_stable_edges.csv")
    write_table(kg, out_dir / "22_causal_interpretation_edge_candidates.csv")
    return all_edges_df, stable_edges, kg


def compact_summary(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        c
        for c in [
            "dataset",
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


def markdown_table(df: pd.DataFrame, max_rows: int = 40) -> str:
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


def write_report(
    out_dir: Path,
    dataset_name: str,
    concept_df: pd.DataFrame,
    dictionary: pd.DataFrame,
    baseline_summary: pd.DataFrame,
    stable: pd.DataFrame,
    edges: pd.DataFrame,
    kg: pd.DataFrame,
) -> None:
    concept_cols = [c for c in concept_df.columns if c.startswith("concept_")]
    top_stable_cols = [
        c
        for c in ["concept", "concept_group", "role", "stable_score", "selected_environments", "interpretation"]
        if c in stable.columns
    ]
    edge_cols = [
        c
        for c in ["source", "target", "directed", "edge_class", "environments", "stable_edge_score", "mean_edge_weight"]
        if c in edges.columns
    ]
    report = f"""# 标准化因果图实验报告：{dataset_name}

## 1. 输入与输出

本流程读取一个已经构建好的 `model_dataset`，在标准目录下生成概念特征、概念模型训练结果和候选因果图结果。

```text
{out_dir}
```

## 2. 概念特征规模

| 项目 | 数值 |
|---|---:|
| 样本数 | {len(concept_df)} |
| 概念特征数 | {len(concept_cols)} |
| 概念组数 | {dictionary["concept_group"].nunique() if not dictionary.empty else 0} |

## 3. 概念 Baseline 结果

{markdown_table(compact_summary(baseline_summary))}

## 4. 稳定概念信号

{markdown_table(stable[top_stable_cols] if top_stable_cols else stable, max_rows=30)}

## 5. 跨环境稳定候选边

{markdown_table(edges[edge_cols] if edge_cols else edges, max_rows=30)}

## 6. 因果解释边候选

{markdown_table(kg.head(30), max_rows=30)}

## 7. 说明

- 概念特征由原始地球化学、地球物理、地形、地质、断层和气候变量按地学含义聚合得到。
- 候选因果边是统计发现结果，不等同于已证实的地质因果关系。
- 这个标准入口保留旧 stage_05 的方法逻辑，但输出按数据集独立存放。
"""
    (out_dir / "causal_graph_standard_report.md").write_text(report, encoding="utf-8")


def run_one_dataset(args: argparse.Namespace, dataset_path: Path, dataset_name: str | None) -> dict:
    df = load_table(dataset_path)
    name = dataset_name_from_path(dataset_path, dataset_name)
    out_dir = Path(args.output_root) / name / "03_causal_graph"
    out_dir.mkdir(parents=True, exist_ok=True)

    resolved = build_mapping(df, out_dir)
    concept_df, dictionary = build_concept_features(df, resolved, out_dir)
    _, baseline_summary = train_concept_baselines(concept_df, name, out_dir)
    stable, candidates = run_environment_stability(
        concept_df,
        dictionary,
        out_dir,
        bootstrap=args.bootstrap,
        top_k=args.top_k,
        selection_rate_threshold=args.selection_rate_threshold,
        auc_threshold=args.auc_threshold,
        rf_top_k=args.rf_top_k,
    )
    _, stable_edges, kg = run_causal_discovery(
        concept_df,
        candidates,
        out_dir,
        partial_threshold=args.partial_threshold,
        p_threshold=args.p_threshold,
        max_edges=args.max_edges_per_environment,
    )
    write_report(out_dir, name, concept_df, dictionary, baseline_summary, stable, stable_edges, kg)

    manifest = {
        "dataset": name,
        "input_dataset": str(dataset_path),
        "output_dir": str(out_dir),
        "concept_features": int(sum(c.startswith("concept_") for c in concept_df.columns)),
        "rows": int(len(concept_df)),
        "outputs": {
            "concept_features": str(out_dir / "04_concept_features.parquet"),
            "concept_baseline_summary": str(out_dir / "09_concept_baseline_summary.csv"),
            "stable_signals": str(out_dir / "14_environment_stable_signals.csv"),
            "candidate_concepts": str(out_dir / "15_causal_graph_candidate_concepts.csv"),
            "all_edges": str(out_dir / "17_causal_edges_all_environments.csv"),
            "stable_edges": str(out_dir / "21_cross_environment_stable_edges.csv"),
            "kg_candidates": str(out_dir / "22_causal_interpretation_edge_candidates.csv"),
            "report": str(out_dir / "causal_graph_standard_report.md"),
        },
    }
    (out_dir / "causal_graph_standard_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run standardized causal graph workflow for model datasets.")
    parser.add_argument("--datasets", nargs="+", required=True, help="Input model_dataset CSV/parquet paths.")
    parser.add_argument("--dataset-name", default=None, help="Optional name for a single dataset.")
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Root output directory. Defaults to outputs/standardized_runs.",
    )
    parser.add_argument("--bootstrap", type=int, default=100, help="Bootstrap runs for environment stability.")
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--selection-rate-threshold", type=float, default=0.50)
    parser.add_argument("--auc-threshold", type=float, default=0.30)
    parser.add_argument("--rf-top-k", type=int, default=12)
    parser.add_argument("--partial-threshold", type=float, default=0.18)
    parser.add_argument("--p-threshold", type=float, default=0.20)
    parser.add_argument("--max-edges-per-environment", type=int, default=45)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_paths = [Path(p) for p in args.datasets]
    if args.dataset_name and len(dataset_paths) > 1:
        raise ValueError("--dataset-name can only be used with one dataset.")
    manifests = []
    for path in dataset_paths:
        manifest = run_one_dataset(args, path, args.dataset_name)
        manifests.append(manifest)
        print(f"Wrote standardized causal graph workflow for {manifest['dataset']}")
        print(manifest["outputs"]["report"])
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "standardized_causal_graph_index.json").write_text(
        json.dumps({"runs": manifests}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
