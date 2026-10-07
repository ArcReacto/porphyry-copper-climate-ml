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
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "rebuttal_experiments" / "xgboost" / "08_random_matched_residualization"
METRICS = ["roc_auc", "average_precision", "f1", "top05_f1", "top05_ndcg", "top10_f1", "top10_ndcg"]
DETERMINISTIC_PREFIXES = ["NoClimate", "M4_Spearman", "DropSelected", "SelectedRawOnly"]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


exp01 = load_module(
    "rebuttal_exp01_for_random_control",
    PROJECT_ROOT / "scripts" / "stage_16_rebuttal_experiments" / "01_fold_local_graphunion.py",
)
base = exp01.base63


def matched_random_columns(
    geochemistry: list[str],
    selected: list[str],
    role_table: pd.DataFrame,
    rng: np.random.Generator,
) -> list[str]:
    role = role_table.set_index("column").loc[geochemistry].copy()
    role["missing_stratum"] = pd.qcut(role["missing_rate"], q=4, labels=False, duplicates="drop").fillna(0).astype(int)
    selected_frame = role.loc[selected]
    sampled: list[str] = []
    for stratum, count in selected_frame["missing_stratum"].value_counts().sort_index().items():
        candidates = role.index[role["missing_stratum"].eq(stratum)].to_numpy()
        sampled.extend(rng.choice(candidates, size=int(count), replace=False).tolist())
    if len(sampled) != len(selected):
        raise RuntimeError("Matched random feature count does not equal the M4 selected count.")
    return sorted(sampled)


def make_custom_recipe(
    model_key: str,
    train_df: pd.DataFrame,
    no_climate: list[str],
    climate: list[str],
    residual_cols: list[str],
    raw_cols: list[str] | None = None,
):
    residual_set = set(residual_cols)
    if raw_cols is None:
        raw_cols = [column for column in no_climate if column not in residual_set]
    ordered_residuals = [column for column in no_climate if column in residual_set]
    return base.FeatureRecipe(model_key, raw_cols, ordered_residuals, climate).fit(train_df)


def add_model_result(
    model_key: str,
    recipe,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    df: pd.DataFrame,
    y: pd.Series,
    fold: int,
    test_group: str,
    metric_rows: list[dict],
    prediction_rows: list[dict],
    repeat: int | None,
    family: str,
) -> None:
    model = make_xgb_model(y.iloc[train_idx]) if family == "xgboost" else base.make_rf_model()
    model.fit(recipe.transform(train_df), y.iloc[train_idx])
    y_prob = model.predict_proba(recipe.transform(test_df))[:, 1]
    metric_start = len(metric_rows)
    prediction_start = len(prediction_rows)
    base.add_metric_row(
        metric_rows,
        prediction_rows,
        dataset="known_mining_neutral_ratio_1_10",
        cv_name="groupkfold_state_random_matched",
        fold=fold,
        test_group=test_group,
        model_key=model_key,
        used_cols=recipe.output_columns(),
        train_idx=train_idx,
        test_idx=test_idx,
        df=df,
        y_prob=y_prob,
    )
    metric_rows[metric_start]["random_repeat"] = repeat
    for row in prediction_rows[prediction_start:]:
        row["random_repeat"] = repeat


def summarize_deterministic(metrics: pd.DataFrame, suffix: str) -> pd.DataFrame:
    part = metrics[metrics["model"].isin([f"{prefix}_{suffix}" for prefix in DETERMINISTIC_PREFIXES])]
    out = part.groupby("model")[METRICS].agg(["mean", "std", "count"]).reset_index()
    out.columns = [
        "_".join(str(piece) for piece in column if str(piece)) if isinstance(column, tuple) else str(column)
        for column in out.columns
    ]
    return out


def summarize_random_control(metrics: pd.DataFrame, deterministic: pd.DataFrame, suffix: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    random_metrics = metrics[metrics["model"].eq(f"RandomMatched_{suffix}")].copy()
    by_repeat = random_metrics.groupby("random_repeat")[METRICS].mean().reset_index()
    m4 = deterministic.set_index("model").loc[f"M4_Spearman_{suffix}"]
    rows: list[dict] = []
    for metric in METRICS:
        values = by_repeat[metric].dropna().to_numpy()
        observed = float(m4[f"{metric}_mean"])
        rows.append(
            {
                "metric": metric,
                "m4_observed": observed,
                "random_mean": float(np.mean(values)),
                "random_std": float(np.std(values, ddof=1)),
                "random_p05": float(np.quantile(values, 0.05)),
                "random_p50": float(np.quantile(values, 0.50)),
                "random_p95": float(np.quantile(values, 0.95)),
                "m4_percentile": float(100.0 * np.mean(values <= observed)),
                "empirical_p_random_ge_m4": float((1 + np.sum(values >= observed)) / (1 + len(values))),
                "random_repetitions": int(len(values)),
            }
        )
    return by_repeat, pd.DataFrame(rows)


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
    no_climate = fsets["no_climate"]
    climate = fsets["climate_adjusters"]

    metric_rows: list[dict] = []
    prediction_rows: list[dict] = []
    selection_rows: list[pd.DataFrame] = []
    random_set_rows: list[dict] = []

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

        recipes = {
            f"NoClimate_{suffix}": make_custom_recipe(f"NoClimate_{suffix}", train_df, no_climate, climate, []),
            f"M4_Spearman_{suffix}": make_custom_recipe(f"M4_Spearman_{suffix}", train_df, no_climate, climate, selected),
            f"DropSelected_{suffix}": make_custom_recipe(
                f"DropSelected_{suffix}",
                train_df,
                no_climate,
                climate,
                [],
                raw_cols=[column for column in no_climate if column not in set(selected)],
            ),
            f"SelectedRawOnly_{suffix}": make_custom_recipe(
                f"SelectedRawOnly_{suffix}",
                train_df,
                no_climate,
                climate,
                [],
                raw_cols=selected,
            ),
        }
        for model_key, recipe in recipes.items():
            add_model_result(
                model_key,
                recipe,
                train_df,
                test_df,
                train_idx,
                test_idx,
                df,
                y,
                fold,
                test_group,
                metric_rows,
                prediction_rows,
                None,
                args.model_family,
            )

        for repeat in range(args.random_repetitions):
            rng = np.random.default_rng(args.random_seed + fold * 100000 + repeat)
            random_cols = matched_random_columns(geochemistry, selected, role_table, rng)
            random_recipe = make_custom_recipe(
                f"RandomMatched_{suffix}",
                train_df,
                no_climate,
                climate,
                random_cols,
            )
            add_model_result(
                f"RandomMatched_{suffix}",
                random_recipe,
                train_df,
                test_df,
                train_idx,
                test_idx,
                df,
                y,
                fold,
                test_group,
                metric_rows,
                prediction_rows,
                repeat,
                args.model_family,
            )
            random_set_rows.extend(
                {
                    "fold": fold,
                    "test_group": test_group,
                    "random_repeat": repeat,
                    "feature": column,
                }
                for column in random_cols
            )

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.DataFrame(prediction_rows)
    selections = pd.concat(selection_rows, ignore_index=True)
    random_sets = pd.DataFrame(random_set_rows)
    deterministic = summarize_deterministic(metrics, suffix)
    random_by_repeat, random_comparison = summarize_random_control(metrics, deterministic, suffix)

    metrics.to_csv(output_dir / "random_control_fold_metrics.csv", index=False)
    predictions.to_csv(output_dir / "random_control_oof_predictions.csv", index=False)
    selections.to_csv(output_dir / "m4_spearman_selection_by_fold.csv", index=False)
    random_sets.to_csv(output_dir / "random_feature_sets.csv", index=False)
    deterministic.to_csv(output_dir / "deterministic_model_summary.csv", index=False)
    random_by_repeat.to_csv(output_dir / "random_repeat_summary.csv", index=False)
    random_comparison.to_csv(output_dir / "m4_vs_random_control.csv", index=False)

    summary_columns = ["model"] + [f"{metric}_mean" for metric in METRICS]
    report = f"""# Rebuttal 新实验 N2：随机匹配残差化与特征筛选对照

## 协议

- 数据：known-mining hard-negative 1:10，{len(df)} 条样本；
- 验证：5 折 GroupKFold by state；
- Spearman 阈值：{args.spearman_threshold:.2f}；
- 随机对照：每折按 M4 实际选择数量，并在相同 geochemistry 特征族和缺失率四分位层内抽样；
- 重复次数：{args.random_repetitions}；
- 所有选择、抽样和残差器仅使用训练折。
- 下游分类器：{args.model_family}；XGBoost 参数与原 M1-M4 替换实验一致，正类权重按训练折计算。

## 确定性模型结果

{markdown_table(deterministic[summary_columns])}

## M4 相对随机匹配残差化

{markdown_table(random_comparison)}

## 解释

- `DropSelected_{suffix}` 删除 M4 选中的气候敏感特征，用于判断 M4 是否只是排除这些特征；
- `SelectedRawOnly_{suffix}` 只使用被选中特征的原始值，用于检查这些特征本身的预测信息；
- `RandomMatched_{suffix}` 与 M4 使用相同数量、相同大类和相近缺失率的残差化特征；
- `empirical_p_random_ge_m4` 越小，说明随机集合达到或超过 M4 的概率越低。

本实验只检验 Spearman 选择相对随机选择的增益，不证明被去除成分具有真实因果含义。
"""
    (output_dir / "random_matched_residualization_report.md").write_text(report, encoding="utf-8")
    manifest = {
        "dataset_path": str(dataset_path),
        "run_dir": str(run_dir),
        "rows": len(df),
        "positive": int(y.sum()),
        "negative": int((y == 0).sum()),
        "random_repetitions": args.random_repetitions,
        "random_seed": args.random_seed,
        "spearman_threshold": args.spearman_threshold,
        "matching": "geochemistry role plus feature missing-rate quartile",
        "model_family": args.model_family,
        "xgboost_parameters": PARAMETERS if args.model_family == "xgboost" else None,
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote N2 outputs to: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Matched random residualization control for rebuttal.")
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--spearman-threshold", type=float, default=0.30)
    parser.add_argument("--random-repetitions", type=int, default=100)
    parser.add_argument("--random-seed", type=int, default=20261001)
    parser.add_argument("--model-family", choices=["xgboost", "rf"], default="xgboost")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
