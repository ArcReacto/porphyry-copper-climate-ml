from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from xgb_model import PARAMETERS, make_xgb_model


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_DIR = (
    PROJECT_ROOT
    / "data"
    / "run_inputs"
    / "known_mining_neutral_ratio_1_10_supervised_all_features_v1"
)
DEFAULT_DATASET = (
    PROJECT_ROOT
    / "data"
    / "sample_schemes"
    / "ratio_1_10"
    / "model_dataset_known_mining_neutral_ratio_1_10_with_neutral_all_features_v1.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "rebuttal_experiments" / "xgboost" / "03_negative_source_sensitivity"
MODEL_KEYS = ["NoClimate_RF", "M4_Spearman_RF"]
RANDOM_STATE = 20260622


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


base = load_module(
    "negative_source_base",
    PROJECT_ROOT
    / "scripts"
    / "stage_11_paper_optimization"
    / "63_full_feature_graph_guided_m4_and_perturbation.py",
)


def stratified_half(df: pd.DataFrame, seed_offset: int) -> pd.DataFrame:
    selected = []
    for state_index, (state, group) in enumerate(sorted(df.groupby("state"), key=lambda item: str(item[0]))):
        n = len(group) // 2
        selected.append(group.sample(n=n, random_state=RANDOM_STATE + seed_offset + state_index))
    return pd.concat(selected, ignore_index=False).sort_index()


def build_schemes(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    positives = df[df["Y_label"] == 1].copy()
    hard = df[df["Y_label"] == 0].copy()
    background = df[df["Y_label"] == -1].copy()
    if not (len(positives) == 158 and len(hard) == 1580 and len(background) == 1580):
        raise ValueError(
            f"Unexpected sample counts: positive={len(positives)}, hard={len(hard)}, background={len(background)}"
        )
    hard["negative_source_scheme"] = "hard_mrds"
    background["negative_source_scheme"] = "background_proxy"
    background["Y_label"] = 0
    mixed_hard = stratified_half(hard, 0)
    mixed_background = stratified_half(background, 1000)
    schemes = {
        "hard_only": pd.concat([positives, hard], ignore_index=True),
        "background_only": pd.concat([positives, background], ignore_index=True),
        "mixed_50_50": pd.concat([positives, mixed_hard, mixed_background], ignore_index=True),
    }
    for name, scheme in schemes.items():
        scheme["evaluation_scheme"] = name
        if int((scheme["Y_label"] == 1).sum()) != 158 or int((scheme["Y_label"] == 0).sum()) != 1580:
            raise ValueError(f"Scheme {name} does not preserve the 1:10 class ratio.")
    return schemes


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
        "top05_ndcg",
        "top10_precision",
        "top10_recall",
        "top10_f1",
        "top10_ndcg",
    ]
    out = metrics.groupby(["negative_scheme", "model"], dropna=False)[metric_cols].agg(["mean", "std", "count"])
    out = out.reset_index()
    out.columns = [
        "_".join(str(part) for part in column if str(part)) if isinstance(column, tuple) else str(column)
        for column in out.columns
    ]
    return out


def paired(metrics: pd.DataFrame, suffix: str) -> pd.DataFrame:
    rows: list[dict] = []
    metric_cols = ["average_precision", "f1", "top05_f1", "top05_ndcg", "top10_f1", "top10_ndcg"]
    for scheme, current in metrics.groupby("negative_scheme"):
        baseline = current[current["model"] == f"NoClimate_{suffix}"].set_index(["fold", "test_group"])
        candidate = current[current["model"] == f"M4_Spearman_{suffix}"].set_index(["fold", "test_group"])
        joined = baseline[metric_cols].join(candidate[metric_cols], lsuffix="_base", rsuffix="_candidate")
        for metric in metric_cols:
            diff = joined[f"{metric}_candidate"] - joined[f"{metric}_base"]
            rows.append(
                {
                    "negative_scheme": scheme,
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


def run(args: argparse.Namespace) -> None:
    suffix = "XGB" if args.model_family == "xgboost" else "RF"
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source = base.read_table(Path(args.dataset))
    role_table = pd.read_csv(run_dir / "00_dataset_profile" / "feature_roles.csv")
    fsets = base.feature_sets(role_table)
    schemes = build_schemes(source)

    sample_manifest_rows: list[pd.DataFrame] = []
    metric_rows: list[dict] = []
    prediction_rows: list[dict] = []
    selection_rows: list[pd.DataFrame] = []

    for scheme_name, df in schemes.items():
        df = df.reset_index(drop=False).rename(columns={"index": "source_row_index"})
        audit = df[
            [
                "source_row_index",
                "sample_id",
                "state",
                "longitude",
                "latitude",
                "sample_type",
                "negative_type",
                "negative_source_scheme",
                "Y_label",
            ]
        ].copy()
        audit.insert(0, "negative_scheme", scheme_name)
        sample_manifest_rows.append(audit)
        y = df["Y_label"].astype(int).reset_index(drop=True)
        groups = df["state"].fillna("unknown").astype(str)
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
                    temp.insert(0, "negative_scheme", scheme_name)
                    temp.insert(1, "fold", fold)
                    temp.insert(2, "test_group", test_group)
                    temp["model"] = temp["model"].str.replace("_RF", f"_{suffix}", regex=False)
                    selection_rows.append(temp)
                x_train = recipe.transform(train_df)
                x_test = recipe.transform(test_df)
                model = make_xgb_model(y.iloc[train_idx]) if args.model_family == "xgboost" else base.make_rf_model()
                model.fit(x_train, y.iloc[train_idx])
                y_prob = model.predict_proba(x_test)[:, 1]
                metric_start = len(metric_rows)
                pred_start = len(prediction_rows)
                base.add_metric_row(
                    metric_rows,
                    prediction_rows,
                    dataset=scheme_name,
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
                metric_rows[metric_start]["negative_scheme"] = scheme_name
                for row in prediction_rows[pred_start:]:
                    row["negative_scheme"] = scheme_name
                    source_row = int(df.iloc[row["row_index"]]["source_row_index"])
                    row["source_row_index"] = source_row
                    row["negative_source_scheme"] = str(
                        df.iloc[row["row_index"]].get("negative_source_scheme", "positive")
                    )

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.DataFrame(prediction_rows)
    sample_manifest = pd.concat(sample_manifest_rows, ignore_index=True)
    selections = pd.concat(selection_rows, ignore_index=True) if selection_rows else pd.DataFrame()
    summary = summarize(metrics)
    paired_table = paired(metrics, suffix)
    source_counts = (
        sample_manifest.groupby(["negative_scheme", "negative_source_scheme", "Y_label"], dropna=False)
        .size()
        .rename("rows")
        .reset_index()
    )

    base.write_table(sample_manifest, output_dir / "sample_scheme_manifest.csv")
    base.write_table(source_counts, output_dir / "sample_source_counts.csv")
    base.write_table(metrics, output_dir / "negative_source_fold_metrics.csv")
    base.write_table(predictions, output_dir / "negative_source_oof_predictions.csv")
    base.write_table(selections, output_dir / "negative_source_spearman_selection.csv")
    base.write_table(summary, output_dir / "negative_source_model_summary.csv")
    base.write_table(paired_table, output_dir / "negative_source_paired_comparison.csv")

    columns = [
        "negative_scheme",
        "model",
        "roc_auc_mean",
        "average_precision_mean",
        "f1_mean",
        "top05_f1_mean",
        "top05_ndcg_mean",
        "top10_f1_mean",
        "top10_ndcg_mean",
    ]
    report = [
        "# Rebuttal 实验 03：负样本来源敏感性",
        "",
        "`background_only` 使用未标注背景点作为负类代理，仅用于标签方案敏感性分析，不把这些点解释为已证实无矿。三个方案均保持 158:1580 的 1:10 类别比例。",
        "",
        "## 样本来源",
        "",
        _markdown_table(source_counts),
        "",
        "## 模型结果",
        "",
        _markdown_table(summary[columns]),
        "",
        "## M4 Spearman 相对 M2",
        "",
        _markdown_table(paired_table),
    ]
    (output_dir / "negative_source_sensitivity_report.md").write_text("\n".join(report), encoding="utf-8")
    manifest = {
        "experiment": "negative_source_sensitivity",
        "input_dataset": str(Path(args.dataset)),
        "schemes": {
            "hard_only": "158 positives + 1580 MRDS hard negatives",
            "background_only": "158 positives + 1580 unlabeled background proxies relabeled for sensitivity only",
            "mixed_50_50": "158 positives + 790 hard negatives + 790 background proxies",
        },
        "validation": "5-fold GroupKFold by state",
        "models": [key.replace("_RF", f"_{suffix}") for key in MODEL_KEYS],
        "model_family": args.model_family,
        "xgboost_parameters": PARAMETERS if args.model_family == "xgboost" else None,
        "random_state": RANDOM_STATE,
        "interpretation_boundary": "Background proxies are unlabeled, not verified negatives.",
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _markdown_table(df: pd.DataFrame) -> str:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate sensitivity to negative sample source.")
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--model-family", choices=["xgboost", "rf"], default="xgboost")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--spearman-threshold", type=float, default=0.30)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run(args)
    print(f"Wrote rebuttal experiment 03 outputs to: {Path(args.output_dir).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
