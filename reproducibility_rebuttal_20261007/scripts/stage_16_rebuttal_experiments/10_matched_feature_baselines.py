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
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "rebuttal_experiments" / "xgboost" / "10_matched_feature_baselines"
METRICS = ["roc_auc", "average_precision", "f1", "top05_f1", "top05_ndcg", "top10_f1", "top10_ndcg"]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


exp01 = load_module(
    "rebuttal_exp01_for_matched_baseline",
    PROJECT_ROOT / "scripts" / "stage_16_rebuttal_experiments" / "01_fold_local_graphunion.py",
)
base = exp01.base63


def recipe(train_df, model_key: str, raw_universe: list[str], climate: list[str], residual_cols: list[str]):
    residual_set = set(residual_cols)
    ordered = [column for column in raw_universe if column in residual_set]
    raw = [column for column in raw_universe if column not in residual_set]
    return base.FeatureRecipe(model_key, raw, ordered, climate).fit(train_df)


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    out = metrics.groupby(["feature_view", "adjustment"])[METRICS].agg(["mean", "std", "count"]).reset_index()
    out.columns = [
        "_".join(str(piece) for piece in column if str(piece)) if isinstance(column, tuple) else str(column)
        for column in out.columns
    ]
    return out


def paired(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []

    def compare(left_filter: dict, right_filter: dict, contrast: str) -> None:
        left = metrics.copy()
        right = metrics.copy()
        for key, value in left_filter.items():
            left = left[left[key].eq(value)]
        for key, value in right_filter.items():
            right = right[right[key].eq(value)]
        left = left.set_index(["fold", "test_group"])
        right = right.set_index(["fold", "test_group"])
        joined = left[METRICS].join(right[METRICS], lsuffix="_left", rsuffix="_right")
        for metric in METRICS:
            delta = joined[f"{metric}_right"] - joined[f"{metric}_left"]
            rows.append(
                {
                    "contrast": contrast,
                    "metric": metric,
                    "left_mean": joined[f"{metric}_left"].mean(),
                    "right_mean": joined[f"{metric}_right"].mean(),
                    "right_minus_left": delta.mean(),
                    "right_better_folds": int((delta > 0).sum()),
                    "equal_folds": int(np.isclose(delta, 0).sum()),
                    "right_worse_folds": int((delta < 0).sum()),
                }
            )

    compare(
        {"feature_view": "geochemistry_only", "adjustment": "raw"},
        {"feature_view": "geochemistry_only", "adjustment": "m4_spearman"},
        "geochemistry: M4 minus raw",
    )
    compare(
        {"feature_view": "multimodal_no_climate", "adjustment": "raw"},
        {"feature_view": "multimodal_no_climate", "adjustment": "m4_spearman"},
        "multimodal: M4 minus raw",
    )
    compare(
        {"feature_view": "geochemistry_only", "adjustment": "raw"},
        {"feature_view": "multimodal_no_climate", "adjustment": "raw"},
        "raw: multimodal minus geochemistry",
    )
    compare(
        {"feature_view": "geochemistry_only", "adjustment": "m4_spearman"},
        {"feature_view": "multimodal_no_climate", "adjustment": "m4_spearman"},
        "M4: multimodal minus geochemistry",
    )
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame, digits: int = 4) -> str:
    show = frame.copy()
    for column in show.select_dtypes(include=["float"]).columns:
        show[column] = show[column].map(lambda value: "" if pd.isna(value) else f"{value:.{digits}f}")
    lines = [
        "| " + " | ".join(show.columns.astype(str)) + " |",
        "| " + " | ".join(["---"] * len(show.columns)) + " |",
    ]
    for row in show.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |")
    return "\n".join(lines)


def run(args: argparse.Namespace) -> None:
    suffix = "XGB" if args.model_family == "xgboost" else "RF"
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df, dataset_path = base.load_dataset_from_run(run_dir)
    df = df[df[base.TARGET].isin([0, 1])].reset_index(drop=True)
    y = df[base.TARGET].astype(int).reset_index(drop=True)
    groups = df["state"].fillna("unknown").astype(str)
    role_table = pd.read_csv(run_dir / "00_dataset_profile" / "feature_roles.csv")
    fsets = base.feature_sets(role_table)
    geochemistry = fsets["geochemistry"]
    multimodal = fsets["no_climate"]
    climate = fsets["climate_adjusters"]

    metric_rows: list[dict] = []
    prediction_rows: list[dict] = []
    selection_rows: list[pd.DataFrame] = []
    splitter = GroupKFold(n_splits=5)

    for fold, (train_idx, test_idx) in enumerate(splitter.split(df, y, groups), start=1):
        train_idx = np.asarray(train_idx)
        test_idx = np.asarray(test_idx)
        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()
        test_group = ";".join(sorted(groups.iloc[test_idx].unique().tolist()))
        sensitive = base.local_spearman_sensitive(train_df, geochemistry, climate, args.spearman_threshold)
        selected = sensitive.loc[sensitive["is_spearman_sensitive"], "target_column"].tolist()
        sensitive = sensitive.copy()
        sensitive.insert(0, "fold", fold)
        sensitive.insert(1, "test_group", test_group)
        selection_rows.append(sensitive)

        configurations = [
            ("geochemistry_only", "raw", recipe(train_df, f"GeoRaw_{suffix}", geochemistry, climate, [])),
            ("geochemistry_only", "m3_broad", recipe(train_df, f"GeoM3_{suffix}", geochemistry, climate, geochemistry)),
            ("geochemistry_only", "m4_spearman", recipe(train_df, f"GeoM4_{suffix}", geochemistry, climate, selected)),
            ("multimodal_no_climate", "raw", recipe(train_df, f"MultiRaw_{suffix}", multimodal, climate, [])),
            ("multimodal_no_climate", "m3_broad", recipe(train_df, f"MultiM3_{suffix}", multimodal, climate, geochemistry)),
            ("multimodal_no_climate", "m4_spearman", recipe(train_df, f"MultiM4_{suffix}", multimodal, climate, selected)),
        ]

        for feature_view, adjustment, current_recipe in configurations:
            model = make_xgb_model(y.iloc[train_idx]) if args.model_family == "xgboost" else base.make_rf_model()
            model.fit(current_recipe.transform(train_df), y.iloc[train_idx])
            y_prob = model.predict_proba(current_recipe.transform(test_df))[:, 1]
            metric_start = len(metric_rows)
            prediction_start = len(prediction_rows)
            base.add_metric_row(
                metric_rows,
                prediction_rows,
                dataset="known_mining_neutral_ratio_1_10",
                cv_name="groupkfold_state_matched_features",
                fold=fold,
                test_group=test_group,
                model_key=f"{feature_view}__{adjustment}__{suffix}",
                used_cols=current_recipe.output_columns(),
                train_idx=train_idx,
                test_idx=test_idx,
                df=df,
                y_prob=y_prob,
            )
            metric_rows[metric_start]["feature_view"] = feature_view
            metric_rows[metric_start]["adjustment"] = adjustment
            for row in prediction_rows[prediction_start:]:
                row["feature_view"] = feature_view
                row["adjustment"] = adjustment

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.DataFrame(prediction_rows)
    selections = pd.concat(selection_rows, ignore_index=True)
    summary = summarize(metrics)
    contrasts = paired(metrics)
    metrics.to_csv(output_dir / "matched_feature_fold_metrics.csv", index=False)
    predictions.to_csv(output_dir / "matched_feature_oof_predictions.csv", index=False)
    selections.to_csv(output_dir / "matched_feature_spearman_selection.csv", index=False)
    summary.to_csv(output_dir / "matched_feature_summary.csv", index=False)
    contrasts.to_csv(output_dir / "matched_feature_contrasts.csv", index=False)

    summary_columns = ["feature_view", "adjustment"] + [f"{metric}_mean" for metric in METRICS]
    report = f"""# Rebuttal 新实验 N3：同特征公平基线

## 协议

- 数据：1:10 hard-negative，{len(df)} 条；
- 验证：5 折 GroupKFold by state；
- 分类器、随机种子、缺失值处理与调参预算完全一致；
- 下游分类器：{args.model_family}；XGBoost 正类权重按训练折计算，未在测试折调参；
- Geochemistry-only：{len(geochemistry)} 个特征；
- Multimodal-no-climate：{len(multimodal)} 个特征；
- 气候调节变量：{len(climate)} 个，仅供残差器使用，不进入 M2/M4 下游分类器。

## 主结果

{markdown_table(summary[summary_columns])}

## 成对差值

{markdown_table(contrasts)}

## 解释边界

本实验隔离了“特征模态增加”和“气候残差化”两个因素，但不等同于 Zhang et al. 或文献 [15] 的完整复现。外部方法只有在取得准确算法、代码和参数后才能以其论文名称报告。
"""
    (output_dir / "matched_feature_baseline_report.md").write_text(report, encoding="utf-8")
    manifest = {
        "dataset_path": str(dataset_path),
        "run_dir": str(run_dir),
        "rows": len(df),
        "geochemistry_features": len(geochemistry),
        "multimodal_no_climate_features": len(multimodal),
        "climate_adjusters": len(climate),
        "spearman_threshold": args.spearman_threshold,
        "note": "This is a matched feature-view ablation, not a named-paper reimplementation.",
        "model_family": args.model_family,
        "xgboost_parameters": PARAMETERS if args.model_family == "xgboost" else None,
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote N3 outputs to: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Matched feature-view baseline experiment for rebuttal.")
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--spearman-threshold", type=float, default=0.30)
    parser.add_argument("--model-family", choices=["xgboost", "rf"], default="xgboost")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
