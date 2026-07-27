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
STAGE11_DIR = Path(__file__).resolve().parent
BASE_SCRIPT = STAGE11_DIR / "63_full_feature_graph_guided_m4_and_perturbation.py"
RANDOM_STATE = 20260622
TARGET = "Y_label"

MODEL_KEYS = [
    "Full_RF",
    "NoClimate_RF",
    "M4_Spearman_RF",
    "M4_GraphUnion_RF",
]

TRAINING_MODES = [
    "no_aug",
    "counterfactual_aug",
]

AUGMENT_SCENARIOS = [
    "climate_plus",
    "climate_minus",
    "climate_train_median",
]


def load_base_module():
    spec = importlib.util.spec_from_file_location("stage11_full_feature_m4", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load base script: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


base = load_base_module()


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False, encoding="utf-8-sig")


def make_augmented_train_df(
    train_df: pd.DataFrame,
    climate_cols: list[str],
    train_medians: pd.Series,
    scenarios: list[str],
) -> pd.DataFrame:
    frames = [train_df.copy()]
    for scenario in scenarios:
        perturbed = base.perturb_climate(train_df, climate_cols, scenario, train_medians)
        perturbed = perturbed.copy()
        if "sample_id" in perturbed.columns:
            perturbed["sample_id"] = perturbed["sample_id"].astype(str) + f"__cf_{scenario}"
        frames.append(perturbed)
    return pd.concat(frames, ignore_index=True)


def make_augmented_y(y_train: pd.Series, repeat_count: int) -> pd.Series:
    return pd.concat([y_train.reset_index(drop=True)] * repeat_count, ignore_index=True)


def add_train_metric_row(
    rows: list[dict],
    *,
    dataset: str,
    cv_name: str,
    fold: int,
    test_group: str | None,
    model_key: str,
    training_mode: str,
    n_features: int,
    n_train_original: int,
    n_train_effective: int,
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> None:
    y_pred = (y_prob >= base.DECISION_THRESHOLD).astype(int)
    tn, fp, fn, tp = base.confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    row = {
        "dataset": dataset,
        "experiment": "counterfactual_climate_augmentation",
        "cv": cv_name,
        "fold": int(fold),
        "test_group": test_group,
        "model": model_key,
        "training_mode": training_mode,
        "n_features": int(n_features),
        "n_train_original": int(n_train_original),
        "n_train_effective": int(n_train_effective),
        "n_test": int(len(y_true)),
        "positive_test": int(y_true.sum()),
        "negative_test": int((y_true == 0).sum()),
        "roc_auc": base.safe_metric("roc_auc", y_true, y_pred, y_prob),
        "average_precision": base.safe_metric("average_precision", y_true, y_pred, y_prob),
        "balanced_accuracy": base.safe_metric("balanced_accuracy", y_true, y_pred, y_prob),
        "precision": base.safe_metric("precision", y_true, y_pred, y_prob),
        "recall": base.safe_metric("recall", y_true, y_pred, y_prob),
        "f1": base.safe_metric("f1", y_true, y_pred, y_prob),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    row.update(base.top_k_metrics(y_true, y_prob))
    rows.append(row)


def add_perturbation_row(
    rows: list[dict],
    *,
    dataset: str,
    cv_name: str,
    fold: int,
    test_group: str | None,
    model_key: str,
    training_mode: str,
    scenario: str,
    y_true: np.ndarray,
    original_prob: np.ndarray,
    perturbed_prob: np.ndarray,
) -> None:
    delta = perturbed_prob - original_prob
    y_pred = (perturbed_prob >= base.DECISION_THRESHOLD).astype(int)
    row = {
        "dataset": dataset,
        "cv": cv_name,
        "fold": int(fold),
        "test_group": test_group,
        "model": model_key,
        "training_mode": training_mode,
        "scenario": scenario,
        "n_test": int(len(y_true)),
        "mean_delta": float(np.mean(delta)),
        "mean_abs_delta": float(np.mean(np.abs(delta))),
        "median_abs_delta": float(np.median(np.abs(delta))),
        "p95_abs_delta": float(np.quantile(np.abs(delta), 0.95)),
        "max_abs_delta": float(np.max(np.abs(delta))),
        "frac_abs_delta_gt_0_05": float(np.mean(np.abs(delta) > 0.05)),
        "roc_auc": base.safe_metric("roc_auc", y_true, y_pred, perturbed_prob),
        "average_precision": base.safe_metric("average_precision", y_true, y_pred, perturbed_prob),
        "balanced_accuracy": base.safe_metric("balanced_accuracy", y_true, y_pred, perturbed_prob),
        "f1": base.safe_metric("f1", y_true, y_pred, perturbed_prob),
    }
    row.update(base.top_k_metrics(y_true, perturbed_prob))
    rows.append(row)


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
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
    out = metrics.groupby(
        ["dataset", "experiment", "cv", "model", "training_mode"],
        dropna=False,
    )[metric_cols].agg(["mean", "std", "count"]).reset_index()
    out.columns = [
        "_".join(str(part) for part in col if str(part))
        if isinstance(col, tuple)
        else str(col)
        for col in out.columns
    ]
    return out


def summarize_perturbation(perturb: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "mean_abs_delta",
        "median_abs_delta",
        "p95_abs_delta",
        "max_abs_delta",
        "frac_abs_delta_gt_0_05",
        "roc_auc",
        "average_precision",
        "balanced_accuracy",
        "f1",
        "top05_recall",
        "top05_precision",
        "top05_f1",
        "top05_lift",
        "top05_ndcg",
        "top10_recall",
        "top10_precision",
        "top10_f1",
        "top10_lift",
        "top10_ndcg",
    ]
    out = perturb.groupby(
        ["dataset", "cv", "model", "training_mode", "scenario"],
        dropna=False,
    )[metric_cols].agg(["mean", "std", "count"]).reset_index()
    out.columns = [
        "_".join(str(part) for part in col if str(part))
        if isinstance(col, tuple)
        else str(col)
        for col in out.columns
    ]
    return out


def add_delta_vs_no_aug(summary: pd.DataFrame) -> pd.DataFrame:
    key_cols = ["dataset", "experiment", "cv", "model"]
    metric_cols = [c for c in summary.columns if c.endswith("_mean")]
    base_rows = summary[summary["training_mode"] == "no_aug"][key_cols + metric_cols].copy()
    aug_rows = summary[summary["training_mode"] == "counterfactual_aug"][key_cols + metric_cols].copy()
    merged = aug_rows.merge(base_rows, on=key_cols, suffixes=("_aug", "_no_aug"))
    for metric in metric_cols:
        merged[f"delta_{metric}"] = merged[f"{metric}_aug"] - merged[f"{metric}_no_aug"]
    return merged


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


def write_report(
    out_dir: Path,
    dataset: str,
    summary: pd.DataFrame,
    deltas: pd.DataFrame,
    perturb_summary: pd.DataFrame,
) -> None:
    main_cols = [
        "dataset",
        "cv",
        "model",
        "training_mode",
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
    delta_cols = [
        "dataset",
        "cv",
        "model",
        "delta_roc_auc_mean",
        "delta_average_precision_mean",
        "delta_balanced_accuracy_mean",
        "delta_f1_mean",
        "delta_top05_precision_mean",
        "delta_top05_recall_mean",
        "delta_top05_f1_mean",
        "delta_top05_lift_mean",
        "delta_top05_ndcg_mean",
        "delta_top10_recall_mean",
        "delta_top10_ndcg_mean",
    ]
    pert_cols = [
        "dataset",
        "cv",
        "model",
        "training_mode",
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
        f"# Counterfactual Climate Augmentation: {dataset}",
        "",
        "## Design",
        "",
        "For each training fold, climate-perturbed copies are generated from the training samples only. Labels are unchanged. The validation fold is never augmented.",
        "",
        "Training perturbation scenarios:",
        "",
        "- `climate_plus`: non-temperature climate adjusters +10%; temperature-like variables +1 degree.",
        "- `climate_minus`: non-temperature climate adjusters -10%; temperature-like variables -1 degree.",
        "- `climate_train_median`: climate adjusters replaced by training-fold medians.",
        "",
        "## Original Test Metrics",
        "",
        markdown_table(summary[[c for c in main_cols if c in summary.columns]]),
        "",
        "## Counterfactual Augmentation Deltas",
        "",
        "Positive values mean the augmented model is better than the no-augmentation counterpart under the same CV/model.",
        "",
        markdown_table(deltas[[c for c in delta_cols if c in deltas.columns]]),
        "",
        "## Perturbation Stability",
        "",
        markdown_table(perturb_summary[[c for c in pert_cols if c in perturb_summary.columns]]),
    ]
    (out_dir / "counterfactual_climate_augmentation_report.md").write_text("\n".join(report), encoding="utf-8")


def run_experiment(
    run_dir: Path,
    graph_dir: Path,
    output_dir: Path,
    spearman_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dataset = run_dir.name
    df, dataset_path = base.load_dataset_from_run(run_dir)
    df = df[df[TARGET].isin([0, 1])].reset_index(drop=True)
    role_table = pd.read_csv(run_dir / "00_dataset_profile" / "feature_roles.csv")
    mapping_path = run_dir / "03_causal_graph" / "02_feature_to_concept_resolved.csv"
    graph_targets_path = graph_dir / "climate_sensitive_target_concepts.csv"
    graph_cols = base.selected_graph_target_columns(mapping_path, graph_targets_path, role_table)
    graph_residual_cols = set(graph_cols.loc[graph_cols["residualizable"], "column"].tolist())
    fsets = base.feature_sets(role_table)
    climate_cols = fsets["climate_adjusters"]
    y = df[TARGET].astype(int).reset_index(drop=True)

    metric_rows: list[dict] = []
    perturb_rows: list[dict] = []
    pred_rows: list[dict] = []

    for cv_name, splitter, groups in base.make_splitters(df):
        split_iter = splitter.split(df, y, groups) if groups is not None else splitter.split(df, y)
        for fold, (train_idx, test_idx) in enumerate(split_iter, start=1):
            train_idx = np.asarray(train_idx)
            test_idx = np.asarray(test_idx)
            train_df = df.iloc[train_idx].copy()
            test_df = df.iloc[test_idx].copy()
            y_train = y.iloc[train_idx].reset_index(drop=True)
            y_true = y.iloc[test_idx].to_numpy()
            test_group = None
            if groups is not None:
                test_group = ";".join(sorted(pd.Series(groups).iloc[test_idx].astype(str).unique().tolist()))
            train_medians = train_df[climate_cols].median(numeric_only=True) if climate_cols else pd.Series(dtype=float)
            augmented_train_df = make_augmented_train_df(
                train_df,
                climate_cols,
                train_medians,
                AUGMENT_SCENARIOS,
            )
            augmented_y = make_augmented_y(y_train, repeat_count=1 + len(AUGMENT_SCENARIOS))

            for model_key in MODEL_KEYS:
                recipe, _ = base.make_recipe(
                    model_key,
                    train_df,
                    fsets,
                    graph_residual_cols,
                    spearman_threshold,
                )
                x_train_original = recipe.transform(train_df)
                x_train_augmented = recipe.transform(augmented_train_df)
                x_test = recipe.transform(test_df)

                train_options = {
                    "no_aug": (x_train_original, y_train),
                    "counterfactual_aug": (x_train_augmented, augmented_y),
                }
                for training_mode, (x_train, y_fit) in train_options.items():
                    model = base.make_rf_model()
                    model.fit(x_train, y_fit)
                    y_prob = model.predict_proba(x_test)[:, 1]
                    add_train_metric_row(
                        metric_rows,
                        dataset=dataset,
                        cv_name=cv_name,
                        fold=fold,
                        test_group=test_group,
                        model_key=model_key,
                        training_mode=training_mode,
                        n_features=x_train.shape[1],
                        n_train_original=len(train_df),
                        n_train_effective=len(x_train),
                        y_true=y_true,
                        y_prob=y_prob,
                    )
                    for row_index, true_value, prob_value in zip(test_idx, y_true, y_prob):
                        pred_rows.append(
                            {
                                "dataset": dataset,
                                "cv": cv_name,
                                "fold": fold,
                                "test_group": test_group,
                                "model": model_key,
                                "training_mode": training_mode,
                                "row_index": int(row_index),
                                "sample_id": df.iloc[row_index].get("sample_id", ""),
                                "Y_label": int(true_value),
                                "y_prob": float(prob_value),
                                "state": df.iloc[row_index].get("state", ""),
                                "env_causal_group_id": df.iloc[row_index].get("env_causal_group_id", ""),
                            }
                        )
                    for scenario in AUGMENT_SCENARIOS:
                        perturbed_test = base.perturb_climate(test_df, climate_cols, scenario, train_medians)
                        x_perturbed = recipe.transform(perturbed_test)
                        perturbed_prob = model.predict_proba(x_perturbed)[:, 1]
                        add_perturbation_row(
                            perturb_rows,
                            dataset=dataset,
                            cv_name=cv_name,
                            fold=fold,
                            test_group=test_group,
                            model_key=model_key,
                            training_mode=training_mode,
                            scenario=scenario,
                            y_true=y_true,
                            original_prob=y_prob,
                            perturbed_prob=perturbed_prob,
                        )

    metrics = pd.DataFrame(metric_rows)
    perturb = pd.DataFrame(perturb_rows)
    predictions = pd.DataFrame(pred_rows)
    summary = summarize(metrics)
    perturb_summary = summarize_perturbation(perturb)
    deltas = add_delta_vs_no_aug(summary)

    write_table(metrics, output_dir / "counterfactual_aug_metrics.csv")
    write_table(summary, output_dir / "counterfactual_aug_summary.csv")
    write_table(deltas, output_dir / "counterfactual_aug_delta_vs_no_aug.csv")
    write_table(perturb, output_dir / "counterfactual_aug_perturbation_metrics.csv")
    write_table(perturb_summary, output_dir / "counterfactual_aug_perturbation_summary.csv")
    write_table(predictions, output_dir / "counterfactual_aug_predictions.csv")
    write_report(output_dir, dataset, summary, deltas, perturb_summary)

    manifest = {
        "dataset": dataset,
        "input_dataset_path": str(dataset_path),
        "run_dir": str(run_dir),
        "graph_dir": str(graph_dir),
        "output_dir": str(output_dir),
        "models": MODEL_KEYS,
        "training_modes": TRAINING_MODES,
        "augmentation_scenarios": AUGMENT_SCENARIOS,
        "spearman_threshold": spearman_threshold,
        "graph_residual_feature_count": int(len(graph_residual_cols)),
    }
    (output_dir / "counterfactual_climate_augmentation_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary, deltas, perturb_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run lightweight counterfactual climate augmentation.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--graph-dir", required=True)
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
        else PROJECT_ROOT / "outputs" / "paper_optimization" / "counterfactual_climate_augmentation" / run_dir.name
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    run_experiment(run_dir, graph_dir, output_dir, args.spearman_threshold)
    print(f"Wrote counterfactual climate augmentation outputs to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
