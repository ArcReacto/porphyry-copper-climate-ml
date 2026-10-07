"""Compare paleoclimate and modern-climate uses under the rebuttal state folds."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold


PROJECT_ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from xgb_model import PARAMETERS, make_xgb_model
RUN_DIR = (
    PROJECT_ROOT
    / "data/run_inputs/known_mining_neutral_ratio_1_10_supervised_all_features_v1"
)
BASE_PATH = (
    PROJECT_ROOT
    / "scripts/stage_11_paper_optimization/63_full_feature_graph_guided_m4_and_perturbation.py"
)
SOURCE_TMAX = "wnata_jja_tmax_anomaly_mean_degC_1700_1850"
SOURCE_PRECIP = "naspa_cool_precip_mean_mm_1700_1850"
PALEO_TMAX = "paleo_wnata_jja_tmax_anomaly_1700_1850"
PALEO_PRECIP = "paleo_naspa_cool_precip_mm_1700_1850"
PALEO_COLUMNS = [PALEO_TMAX, PALEO_PRECIP]
MODERN_TWO = ["climate_tmax_annual_mean", "climate_ppt_annual_sum"]
CONFIGS = (
    "m2_no_climate",
    "m4_modern_24",
    "m4_modern_2",
    "m4_paleo_2",
    "m4_modern_plus_paleo_26",
    "m1_modern_full",
    "m1_modern_plus_paleo",
)
METRICS = (
    "roc_auc",
    "average_precision",
    "f1",
    "top05_precision",
    "top05_recall",
    "top05_f1",
    "top05_ndcg",
)
CONTRASTS = (
    ("m4_modern_2", "m4_paleo_2"),
    ("m4_modern_24", "m4_paleo_2"),
    ("m4_modern_24", "m4_modern_plus_paleo_26"),
    ("m2_no_climate", "m4_modern_24"),
    ("m2_no_climate", "m4_paleo_2"),
    ("m1_modern_full", "m1_modern_plus_paleo"),
)


def load_base():
    spec = importlib.util.spec_from_file_location("paleo_rebuttal_base63", BASE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(BASE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def markdown_table(frame: pd.DataFrame, digits: int = 4) -> str:
    show = frame.copy()
    for column in show.select_dtypes(include=["float"]).columns:
        show[column] = show[column].map(
            lambda value: "" if pd.isna(value) else f"{value:.{digits}f}"
        )
    lines = [
        "| " + " | ".join(show.columns) + " |",
        "| " + " | ".join(["---"] * len(show.columns)) + " |",
    ]
    lines.extend(
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in show.itertuples(index=False, name=None)
    )
    return "\n".join(lines)


def load_common_samples(base, alignment_path: Path) -> tuple[pd.DataFrame, Path]:
    df, source = base.load_dataset_from_run(RUN_DIR)
    df = df[df[base.TARGET].isin([0, 1])].reset_index(drop=True)
    df["global_sample_id"] = df["state"].astype(str) + "|" + df["sample_id"].astype(str)
    if not df["global_sample_id"].is_unique:
        raise ValueError("The state|sample_id key must be unique in the model dataset")

    alignment = pd.read_csv(alignment_path, low_memory=False)
    if not alignment["global_sample_id"].is_unique:
        raise ValueError("The state|sample_id key must be unique in paleoclimate alignment")
    alignment = alignment[
        ["global_sample_id", "label", "latitude", "longitude", SOURCE_TMAX, SOURCE_PRECIP]
    ].rename(
        columns={
            "latitude": "aligned_latitude",
            "longitude": "aligned_longitude",
            "label": "aligned_label",
        }
    )
    df = df.merge(alignment, on="global_sample_id", how="left", validate="one_to_one")
    if df["aligned_label"].isna().any():
        raise ValueError("Some model samples have no paleoclimate alignment record")
    if not np.array_equal(df[base.TARGET].to_numpy(), df["aligned_label"].to_numpy()):
        raise ValueError("Labels differ between the model dataset and alignment")
    for column in ("latitude", "longitude"):
        if not np.allclose(df[column], df[f"aligned_{column}"], atol=1e-5):
            raise ValueError(f"Coordinates differ for {column}")

    df[PALEO_TMAX] = pd.to_numeric(df[SOURCE_TMAX], errors="coerce")
    df[PALEO_PRECIP] = pd.to_numeric(df[SOURCE_PRECIP], errors="coerce")
    df = df.dropna(subset=PALEO_COLUMNS).reset_index(drop=True)
    if len(df) != 1729:
        raise ValueError(f"Expected 1,729 complete paleoclimate records, found {len(df)}")
    return df, source


def recipe_for(base, config: str, train: pd.DataFrame, fsets: dict, threshold: float):
    current = dict(fsets)
    if config == "m2_no_climate":
        return base.make_recipe("NoClimate_RF", train, current, set(), threshold)
    if config == "m1_modern_full":
        return base.make_recipe("Full_RF", train, current, set(), threshold)
    if config == "m1_modern_plus_paleo":
        current["full"] = fsets["full"] + PALEO_COLUMNS
        return base.make_recipe("Full_RF", train, current, set(), threshold)

    if config == "m4_modern_24":
        current["climate_adjusters"] = fsets["climate_adjusters"]
    elif config == "m4_modern_2":
        current["climate_adjusters"] = MODERN_TWO
    elif config == "m4_paleo_2":
        current["climate_adjusters"] = PALEO_COLUMNS
    elif config == "m4_modern_plus_paleo_26":
        current["climate_adjusters"] = fsets["climate_adjusters"] + PALEO_COLUMNS
    else:
        raise ValueError(config)
    return base.make_recipe("M4_Spearman_RF", train, current, set(), threshold)


def summarize(metrics: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for config in CONFIGS:
        frame = metrics.loc[metrics["config"] == config]
        oof = predictions.loc[predictions["config"] == config]
        if len(frame) != 5 or len(oof) != len(predictions) // len(CONFIGS):
            raise ValueError(f"Incomplete fold or OOF result: {config}")
        rows.append(
            {
                "config": config,
                "n_features_mean": float(frame["n_features"].mean()),
                "n_residual_targets_mean": float(frame["n_residual_targets"].mean()),
                **{f"{metric}_mean": float(frame[metric].mean()) for metric in METRICS},
                **{f"{metric}_std": float(frame[metric].std(ddof=1)) for metric in METRICS},
                "oof_pooled_roc_auc": float(roc_auc_score(oof["Y_label"], oof["y_prob"])),
                "oof_pooled_average_precision": float(
                    average_precision_score(oof["Y_label"], oof["y_prob"])
                ),
            }
        )
    return pd.DataFrame(rows)


def paired_differences(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for baseline, candidate in CONTRASTS:
        left = metrics.loc[metrics["config"] == baseline].set_index(["fold", "test_group"])
        right = metrics.loc[metrics["config"] == candidate].set_index(["fold", "test_group"])
        joined = left[list(METRICS)].join(right[list(METRICS)], lsuffix="_base", rsuffix="_candidate")
        if len(joined) != 5:
            raise ValueError(f"Fold mismatch: {baseline} vs {candidate}")
        for metric in METRICS:
            delta = joined[f"{metric}_candidate"] - joined[f"{metric}_base"]
            rows.append(
                {
                    "baseline": baseline,
                    "candidate": candidate,
                    "metric": metric,
                    "mean_delta": float(delta.mean()),
                    "wins": int((delta > 1e-12).sum()),
                    "ties": int((delta.abs() <= 1e-12).sum()),
                    "losses": int((delta < -1e-12).sum()),
                    "min_delta": float(delta.min()),
                    "max_delta": float(delta.max()),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--alignment",
        type=Path,
        default=PROJECT_ROOT / "data/paleoclimate/paleoclimate_aligned_1738.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/rebuttal_experiments/xgboost/12_paleoclimate_validation",
    )
    parser.add_argument("--spearman-threshold", type=float, default=0.30)
    parser.add_argument("--model-family", choices=["xgboost", "rf"], default="xgboost")
    args = parser.parse_args()

    base = load_base()
    df, dataset_path = load_common_samples(base, args.alignment)
    role_path = RUN_DIR / "00_dataset_profile/feature_roles.csv"
    fsets = base.feature_sets(pd.read_csv(role_path))
    needed = set(fsets["full"] + fsets["no_climate"] + fsets["climate_adjusters"] + MODERN_TWO)
    absent = sorted(needed - set(df.columns))
    if absent:
        raise ValueError(f"Missing model feature columns: {absent[:10]}")
    if not set(MODERN_TWO).issubset(fsets["climate_adjusters"]):
        raise ValueError("The modern two-variable contrast is not in the official adjuster list")

    y = df[base.TARGET].astype(int)
    groups = df["state"].astype(str)
    metrics_rows: list[dict] = []
    prediction_rows: list[dict] = []
    selected_rows: list[dict] = []
    for fold, (train_idx, test_idx) in enumerate(GroupKFold(n_splits=5).split(df, y, groups), start=1):
        train = df.iloc[train_idx].copy()
        test = df.iloc[test_idx].copy()
        test_group = ";".join(sorted(groups.iloc[test_idx].unique()))
        for config in CONFIGS:
            recipe, selection = recipe_for(base, config, train, fsets, args.spearman_threshold)
            model = make_xgb_model(y.iloc[train_idx]) if args.model_family == "xgboost" else base.make_rf_model()
            model.fit(recipe.transform(train), y.iloc[train_idx])
            scores = model.predict_proba(recipe.transform(test))[:, 1]
            metric_start, pred_start = len(metrics_rows), len(prediction_rows)
            base.add_metric_row(
                metrics_rows,
                prediction_rows,
                dataset="known_mining_neutral_ratio_1_10_common_paleo",
                cv_name="groupkfold_state_paleo_sensitivity",
                fold=fold,
                test_group=test_group,
                model_key=config,
                used_cols=recipe.output_columns(),
                train_idx=train_idx,
                test_idx=test_idx,
                df=df,
                y_prob=scores,
            )
            metrics_rows[metric_start]["config"] = config
            metrics_rows[metric_start]["n_residual_targets"] = len(recipe.residual_cols)
            for row in prediction_rows[pred_start:]:
                row["config"] = config
                row["global_sample_id"] = df.iloc[row["row_index"]]["global_sample_id"]
            if not selection.empty:
                selected = selection.loc[selection["selected_by_model"]]
                for item in selected.to_dict(orient="records"):
                    selected_rows.append(
                        {
                            "config": config,
                            "fold": fold,
                            "test_group": test_group,
                            "target_column": item["target_column"],
                            "best_climate_column": item["best_climate_column"],
                            "spearman": float(item["spearman"]),
                        }
                    )
            print(f"fold {fold}/5  {config}  AP={metrics_rows[metric_start]['average_precision']:.4f}", flush=True)

    metrics = pd.DataFrame(metrics_rows)
    predictions = pd.DataFrame(prediction_rows)
    summary = summarize(metrics, predictions)
    contrasts = paired_differences(metrics)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "model_summary.json", summary.to_dict(orient="records"))
    write_json(args.output_dir / "model_fold_metrics.json", metrics.to_dict(orient="records"))
    write_json(args.output_dir / "model_paired_differences.json", contrasts.to_dict(orient="records"))
    with (args.output_dir / "model_oof_predictions.jsonl").open("w", encoding="utf-8") as stream:
        for item in predictions.to_dict(orient="records"):
            stream.write(json.dumps(item, ensure_ascii=False, default=str, allow_nan=False) + "\n")
    with (args.output_dir / "model_selected_features.jsonl").open("w", encoding="utf-8") as stream:
        for item in selected_rows:
            stream.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n")

    shown = summary[
        [
            "config",
            "roc_auc_mean",
            "average_precision_mean",
            "f1_mean",
            "top05_f1_mean",
            "top05_ndcg_mean",
        ]
    ]
    paired_show = contrasts.loc[
        contrasts["metric"].isin(["roc_auc", "average_precision", "top05_f1", "top05_ndcg"]),
        ["baseline", "candidate", "metric", "mean_delta", "wins", "ties", "losses"],
    ]
    fold_ap = metrics.pivot(index=["fold", "test_group"], columns="config", values="average_precision")
    fold_ap = fold_ap.reset_index()[
        [
            "fold",
            "test_group",
            "m2_no_climate",
            "m4_modern_24",
            "m4_modern_2",
            "m4_paleo_2",
            "m4_modern_plus_paleo_26",
        ]
    ]
    paired_index = contrasts.set_index(["baseline", "candidate", "metric"])

    def paired_ap(baseline: str, candidate: str) -> tuple[float, int]:
        row = paired_index.loc[(baseline, candidate, "average_precision")]
        return float(row["mean_delta"]), int(row["wins"])

    paleo_vs_m2, paleo_vs_m2_wins = paired_ap("m2_no_climate", "m4_paleo_2")
    paleo_vs_modern2, paleo_vs_modern2_wins = paired_ap("m4_modern_2", "m4_paleo_2")
    combined_vs_modern, combined_vs_modern_wins = paired_ap(
        "m4_modern_24", "m4_modern_plus_paleo_26"
    )
    direct_vs_modern, direct_vs_modern_wins = paired_ap(
        "m1_modern_full", "m1_modern_plus_paleo"
    )
    combined_selected = [
        item for item in selected_rows if item["config"] == "m4_modern_plus_paleo_26"
    ]
    combined_paleo_best = sum(
        item["best_climate_column"] in PALEO_COLUMNS for item in combined_selected
    )
    report = f"""# 古气候模型敏感性验证

## 固定协议

- 数据：1:10 hard-negative 论文主数据与古气候共同覆盖的 {len(df)} 个点；正样本 {int(y.sum())} 个。9 个 WNATA 范围外的负样本对所有配置统一移除。
- 交叉验证：11 州 GroupKFold 5 折；每折测试州、训练/测试样本对全部配置完全相同。
- 模型：{args.model_family}；XGBoost 沿用主实验固定超参数，正类权重按训练折计算。Spearman 目标选择与 Ridge 残差器均只在训练折拟合。
- 古气候试算窗口：1700–1850 年；WNATA 为夏季最高气温距平均值，NASPA 为冷季降水均值。现代两变量是**年**最高气温均值和**年**降水总量，不是季节严格匹配对照。
- M2/M4 下游特征维持原有 385 个非气候字段。M4 的气候字段仅供敏感目标选择与残差化，不直接作为模型输入。

## 配置说明

| 配置 | 用法 |
|---|---|
| m2_no_climate | 385 个非气候特征，不做气候残差化 |
| m4_modern_24 | 原 24 个现代气候调节变量 |
| m4_modern_2 | 仅现代年最高气温与年降水 2 个调节变量 |
| m4_paleo_2 | 仅 2 个古气候调节变量 |
| m4_modern_plus_paleo_26 | 24 个现代变量加 2 个古气候变量共同调节 |
| m1_modern_full | 原现代气候入模的 662 特征集合 |
| m1_modern_plus_paleo | 662 特征基础上再加入 2 个古气候特征 |

## 逐折均值

{markdown_table(shown)}

## 成对差值（候选配置减基线，按同折比较）

{markdown_table(paired_show)}

## 逐折 AP

{markdown_table(fold_ap)}

## 结果判断

- 古气候作为 M4 调节变量，相对 M2 的平均 AP 差为 {paleo_vs_m2:+.4f}（{paleo_vs_m2_wins}/5 折改善）；相对现代两变量 M4 的平均 AP 差为 {paleo_vs_modern2:+.4f}（{paleo_vs_modern2_wins}/5 折改善）。本窗口未显示稳定优势。
- 在原 24 个现代变量上叠加两个古气候变量，平均 AP 差为 {combined_vs_modern:+.4f}（{combined_vs_modern_wins}/5 折改善）。不能称为叠加提分。
- 将古气候特征直接加入 M1 全特征模型，平均 AP 差为 {direct_vs_modern:+.4f}（{direct_vs_modern_wins}/5 折改善）。这两个新增特征没有带来稳定排序收益。
- M4 的残差化目标数跨折均值：现代 24 变量 {summary.set_index('config').loc['m4_modern_24', 'n_residual_targets_mean']:.1f}，现代 2 变量 {summary.set_index('config').loc['m4_modern_2', 'n_residual_targets_mean']:.1f}，古气候 2 变量 {summary.set_index('config').loc['m4_paleo_2', 'n_residual_targets_mean']:.1f}。在现代+古气候配置的 {len(combined_selected)} 个折内被选目标记录里，{combined_paleo_best} 个以古气候变量为最强相关调节变量；这是选择机制描述，不是因果归因。
- 初步结论限定为：在 1700–1850 年均值、两种不同季节的重建变量、当前样本与固定模型下，未观察到稳健提升。不能外推为古气候无用，也不能据此解决审稿人的地质时间尺度质疑。

## 解释边界

1. 五折均值是主口径；全体 OOF 合并指标另存于 `model_summary.json`，跨折分数未统一校准，不能直接替代逐折均值。
2. 只有 5 折，且各折州/正例数不均衡，胜折数和均值不能自动解释为统计显著。
3. 古气候与现代气候季节、时间尺度和变量定义不同。替换对照只是时间尺度敏感性检查，不能直接证明哪套气候更接近成矿时环境。
4. 古气候格点重建基于树轮和现代校准期；本实验没有真实古风化或矿床形成年代的监督标签，也不是成矿因果验证。
5. 源特征角色表沿用原标准化运行文件；新引入的古气候特征没有用全样本标签做筛选，折内仅做原方法的 Spearman 选择和训练。
"""
    (args.output_dir / "model_validation_report.md").write_text(report, encoding="utf-8")
    manifest = {
        "run_dir": str(RUN_DIR),
        "source_dataset": str(dataset_path),
        "source_dataset_sha256": sha256(dataset_path),
        "feature_roles_sha256": sha256(role_path),
        "alignment_sha256": sha256(args.alignment),
        "sample_count": len(df),
        "positive_count": int(y.sum()),
        "n_folds": 5,
        "spearman_threshold": args.spearman_threshold,
        "random_state": base.RANDOM_STATE,
        "model_family": args.model_family,
        "xgboost_parameters": PARAMETERS if args.model_family == "xgboost" else None,
        "configs": list(CONFIGS),
    }
    write_json(args.output_dir / "model_run_manifest.json", manifest)
    print(f"Completed {len(CONFIGS)} configs x 5 state folds; outputs: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
