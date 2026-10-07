from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from xgb_model import PARAMETERS, make_xgb_model


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = (
    PROJECT_ROOT / "data" / "sample_schemes"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "rebuttal_experiments" / "xgboost" / "05_final_ratio_sensitivity"
MODEL_KEYS = ["NoClimate_RF", "M4_Spearman_RF"]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


base = load_module(
    "ratio_base",
    PROJECT_ROOT
    / "scripts"
    / "stage_11_paper_optimization"
    / "63_full_feature_graph_guided_m4_and_perturbation.py",
)
workflow = load_module(
    "ratio_workflow",
    PROJECT_ROOT / "scripts" / "stage_08_standard_workflow" / "40_run_standard_dataset_workflow.py",
)


def ratio_label(path: Path) -> str:
    match = re.search(r"ratio_1_(\d+)", path.name)
    if not match:
        raise ValueError(f"Cannot parse ratio from {path}")
    return f"1:{match.group(1)}"


def dataset_paths(data_root: Path, ratios: list[int]) -> list[Path]:
    paths = []
    for ratio in ratios:
        path = (
            data_root
            / f"ratio_1_{ratio}"
            / f"model_dataset_known_mining_neutral_ratio_1_{ratio}_supervised_all_features_v1.csv"
        )
        if ratio == 10 and not path.exists():
            path = (
                data_root.parent
                / "run_inputs"
                / "known_mining_neutral_ratio_1_10_supervised_all_features_v1"
                / "model_dataset_known_mining_neutral_ratio_1_10_supervised_all_features_v1.csv"
            )
        if not path.exists():
            raise FileNotFoundError(path)
        paths.append(path)
    return paths


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
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
    out = metrics.groupby(["negative_ratio", "model"], dropna=False)[metric_cols].agg(["mean", "std", "count"])
    out = out.reset_index()
    out.columns = [
        "_".join(str(part) for part in column if str(part)) if isinstance(column, tuple) else str(column)
        for column in out.columns
    ]
    return out


def paired(metrics: pd.DataFrame, suffix: str) -> pd.DataFrame:
    rows = []
    metric_cols = ["average_precision", "f1", "top05_f1", "top05_ndcg", "top10_f1", "top10_ndcg"]
    for ratio, current in metrics.groupby("negative_ratio"):
        baseline = current[current["model"] == f"NoClimate_{suffix}"].set_index(["fold", "test_group"])
        candidate = current[current["model"] == f"M4_Spearman_{suffix}"].set_index(["fold", "test_group"])
        joined = baseline[metric_cols].join(candidate[metric_cols], lsuffix="_base", rsuffix="_candidate")
        for metric in metric_cols:
            diff = joined[f"{metric}_candidate"] - joined[f"{metric}_base"]
            rows.append(
                {
                    "negative_ratio": ratio,
                    "metric": metric,
                    "baseline_mean": float(joined[f"{metric}_base"].mean()),
                    "candidate_mean": float(joined[f"{metric}_candidate"].mean()),
                    "mean_difference": float(diff.mean()),
                    "better_folds": int((diff > 0).sum()),
                    "equal_folds": int(np.isclose(diff, 0).sum()),
                    "worse_folds": int((diff < 0).sum()),
                }
            )
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame) -> str:
    show = df.copy()
    for column in show.columns:
        if pd.api.types.is_float_dtype(show[column]):
            show[column] = show[column].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
    lines = [
        "| " + " | ".join(map(str, show.columns)) + " |",
        "| " + " | ".join(["---"] * len(show.columns)) + " |",
    ]
    for _, row in show.iterrows():
        lines.append("| " + " | ".join(str(row[column]).replace("|", "\\|") for column in show.columns) + " |")
    return "\n".join(lines)


def run(args: argparse.Namespace) -> None:
    suffix = "XGB" if args.model_family == "xgboost" else "RF"
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = dataset_paths(data_root, args.ratios)
    metric_rows: list[dict] = []
    prediction_rows: list[dict] = []
    selection_rows: list[pd.DataFrame] = []
    dataset_rows = []

    for path in paths:
        ratio = ratio_label(path)
        df = base.read_table(path)
        df = df[df[base.TARGET].isin([0, 1])].reset_index(drop=True)
        role_table = workflow.define_feature_roles(df)
        fsets = base.feature_sets(role_table)
        y = df[base.TARGET].astype(int).reset_index(drop=True)
        groups = df["state"].fillna("unknown").astype(str)
        dataset_rows.append(
            {
                "negative_ratio": ratio,
                "dataset_path": str(path),
                "rows": len(df),
                "positive": int(y.sum()),
                "negative": int((y == 0).sum()),
                "states": int(groups.nunique()),
                "numeric_features": int(role_table["is_numeric"].sum()),
                "no_climate_features": len(fsets["no_climate"]),
                "climate_adjusters": len(fsets["climate_adjusters"]),
                "geochemistry_features": len(fsets["geochemistry"]),
            }
        )
        splitter = GroupKFold(n_splits=5)
        for fold, (train_idx, test_idx) in enumerate(splitter.split(df, y, groups), start=1):
            train_idx = np.asarray(train_idx)
            test_idx = np.asarray(test_idx)
            train_df = df.iloc[train_idx].copy()
            test_df = df.iloc[test_idx].copy()
            test_group = ";".join(sorted(groups.iloc[test_idx].unique().tolist()))
            for model_key in MODEL_KEYS:
                recipe, selection = base.make_recipe(
                    model_key,
                    train_df,
                    fsets,
                    set(),
                    args.spearman_threshold,
                )
                if not selection.empty:
                    temp = selection.copy()
                    temp.insert(0, "negative_ratio", ratio)
                    temp.insert(1, "fold", fold)
                    temp.insert(2, "test_group", test_group)
                    temp["model"] = temp["model"].str.replace("_RF", f"_{suffix}", regex=False)
                    selection_rows.append(temp)
                model = make_xgb_model(y.iloc[train_idx]) if args.model_family == "xgboost" else base.make_rf_model()
                model.fit(recipe.transform(train_df), y.iloc[train_idx])
                y_prob = model.predict_proba(recipe.transform(test_df))[:, 1]
                metric_start = len(metric_rows)
                pred_start = len(prediction_rows)
                base.add_metric_row(
                    metric_rows,
                    prediction_rows,
                    dataset=path.stem,
                    cv_name="groupkfold_state",
                    fold=fold,
                    test_group=test_group,
                    model_key=model_key.replace("_RF", f"_{suffix}"),
                    used_cols=recipe.output_columns(),
                    train_idx=train_idx,
                    test_idx=test_idx,
                    df=df,
                    y_prob=y_prob,
                )
                metric_rows[metric_start]["negative_ratio"] = ratio
                for row in prediction_rows[pred_start:]:
                    row["negative_ratio"] = ratio

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.DataFrame(prediction_rows)
    selections = pd.concat(selection_rows, ignore_index=True) if selection_rows else pd.DataFrame()
    datasets = pd.DataFrame(dataset_rows)
    summary = summarize(metrics)
    paired_table = paired(metrics, suffix)
    base.write_table(datasets, output_dir / "ratio_dataset_audit.csv")
    base.write_table(metrics, output_dir / "ratio_fold_metrics.csv")
    base.write_table(predictions, output_dir / "ratio_oof_predictions.csv")
    base.write_table(selections, output_dir / "ratio_spearman_selection.csv")
    base.write_table(summary, output_dir / "ratio_model_summary.csv")
    base.write_table(paired_table, output_dir / "ratio_paired_differences.csv")

    columns = [
        "negative_ratio",
        "model",
        "roc_auc_mean",
        "average_precision_mean",
        "f1_mean",
        "top05_f1_mean",
        "top05_lift_mean",
        "top05_ndcg_mean",
        "top10_f1_mean",
        "top10_lift_mean",
        "top10_ndcg_mean",
    ]
    report = [
        "# Rebuttal 实验 05：最终方法负样本比例敏感性",
        "",
        "本实验使用 known-mining hard-negative 数据集和同一 5 折按州协议。根据实验 01，GraphUnion 不再作为主方法，本表只比较 M2 与完全折内的 M4 Spearman。AP 会随正例率变化，因此跨比例解释同时参考 Lift 和 Top-K 指标。",
        "",
        "## 数据审计",
        "",
        markdown_table(datasets),
        "",
        "## 主结果",
        "",
        markdown_table(summary[columns]),
        "",
        "## M4 相对 M2",
        "",
        markdown_table(paired_table),
    ]
    (output_dir / "final_ratio_sensitivity_report.md").write_text("\n".join(report), encoding="utf-8")
    manifest = {
        "experiment": "final_spearman_negative_ratio_sensitivity",
        "ratios": [f"1:{ratio}" for ratio in args.ratios],
        "models": [key.replace("_RF", f"_{suffix}") for key in MODEL_KEYS],
        "model_family": args.model_family,
        "xgboost_parameters": PARAMETERS if args.model_family == "xgboost" else None,
        "validation": "5-fold GroupKFold by state",
        "spearman_threshold": args.spearman_threshold,
        "graphunion_exclusion_reason": "Experiment 01 showed that fold-local GraphUnion did not preserve the original gain.",
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run final M2/M4 Spearman negative-ratio sensitivity.")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--ratios", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--spearman-threshold", type=float, default=0.30)
    parser.add_argument("--model-family", choices=["xgboost", "rf"], default="xgboost")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run(args)
    print(f"Wrote rebuttal experiment 05 outputs to: {Path(args.output_dir).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
