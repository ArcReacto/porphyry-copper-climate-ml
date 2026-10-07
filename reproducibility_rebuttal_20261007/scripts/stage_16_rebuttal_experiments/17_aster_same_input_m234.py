from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from xgb_model import PARAMETERS, make_xgb_model


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "aster_basic"
DEFAULT_RUN_DIR = (
    PROJECT_ROOT
    / "data"
    / "run_inputs"
    / "known_mining_neutral_ratio_1_10_supervised_all_features_v1"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "rebuttal_experiments"
    / "xgboost"
    / "17_aster_same_input_m234"
)
METRICS = [
    "roc_auc",
    "average_precision",
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


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


base = load_module(
    "rebuttal_aster_base63",
    PROJECT_ROOT
    / "scripts"
    / "stage_11_paper_optimization"
    / "63_full_feature_graph_guided_m4_and_perturbation.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_unique(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    if "sample_id" not in frame.columns:
        raise ValueError(f"Missing sample_id: {path}")
    if frame["sample_id"].duplicated().any():
        raise ValueError(f"Duplicate sample_id values: {path}")
    return frame


def assert_metadata_match(left: pd.DataFrame, right: pd.DataFrame, name: str) -> None:
    common = ["sample_id", "Y_label", "latitude", "longitude", "state"]
    merged = left[common].merge(right[common], on="sample_id", suffixes=("_left", "_right"))
    if len(merged) != len(left) or len(merged) != len(right):
        raise ValueError(f"Sample coverage mismatch for {name}")
    if not np.array_equal(merged["Y_label_left"].to_numpy(), merged["Y_label_right"].to_numpy()):
        raise ValueError(f"Label mismatch for {name}")
    if not np.array_equal(merged["state_left"].astype(str).to_numpy(), merged["state_right"].astype(str).to_numpy()):
        raise ValueError(f"State mismatch for {name}")
    for coordinate in ["latitude", "longitude"]:
        if not np.allclose(
            merged[f"{coordinate}_left"].to_numpy(float),
            merged[f"{coordinate}_right"].to_numpy(float),
            atol=1e-9,
            equal_nan=True,
        ):
            raise ValueError(f"Coordinate mismatch for {name}: {coordinate}")


def load_dataset(data_dir: Path, run_dir: Path) -> tuple[pd.DataFrame, dict[str, list[str]], dict]:
    paths = {
        "aster_climate": data_dir / "aster_climate_features.csv",
        "vnir": data_dir / "features_VNIR.csv",
        "swir": data_dir / "features_SWIR.csv",
        "tir": data_dir / "features_TIR.csv",
        "lulc": data_dir / "lulc_features.csv",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing input files: {missing}")

    climate_aster = read_unique(paths["aster_climate"])
    vnir = read_unique(paths["vnir"])
    swir = read_unique(paths["swir"])
    tir = read_unique(paths["tir"])
    lulc = read_unique(paths["lulc"])

    assert_metadata_match(vnir, swir, "VNIR/SWIR")
    assert_metadata_match(vnir, tir, "VNIR/TIR")
    target_ids = set(vnir["sample_id"].astype(str))
    if target_ids != set(swir["sample_id"].astype(str)) or target_ids != set(tir["sample_id"].astype(str)):
        raise ValueError("VNIR, SWIR, and TIR sample sets differ")

    climate_aster = climate_aster[climate_aster["sample_id"].astype(str).isin(target_ids)].copy()
    if len(climate_aster) != len(target_ids):
        raise ValueError("Climate/ASTER table does not cover the 1,735 spectral samples")
    climate_aster = climate_aster.sort_values("sample_id").reset_index(drop=True)
    vnir = vnir.sort_values("sample_id").reset_index(drop=True)
    assert_metadata_match(vnir, climate_aster, "spectral/climate")

    lulc = lulc[lulc["sample_id"].astype(str).isin(target_ids)].copy()
    if len(lulc) != len(target_ids):
        raise ValueError("LULC table does not cover the 1,735 spectral samples")
    lulc = lulc.sort_values("sample_id").reset_index(drop=True)

    aster_cols = [
        column
        for column in climate_aster.columns
        if column.startswith("aster_") or column.startswith("tir_")
    ]
    if len(aster_cols) != 31:
        raise ValueError(f"Expected 31 basic ASTER features, found {len(aster_cols)}")

    role_table = pd.read_csv(run_dir / "00_dataset_profile" / "feature_roles.csv")
    climate_cols = role_table.loc[
        role_table["is_numeric"].astype(bool)
        & role_table["used_as_climate_adjuster"].astype(bool),
        "column",
    ].tolist()
    absent_climate = sorted(set(climate_cols) - set(climate_aster.columns))
    if absent_climate:
        raise ValueError(f"Climate adjusters absent from ASTER table: {absent_climate}")
    if len(climate_cols) != 24:
        raise ValueError(f"Expected 24 climate adjusters, found {len(climate_cols)}")

    lulc_cols = [column for column in lulc.columns if column not in {"sample_id", "n_lulc_scenes"}]
    if len(lulc_cols) != 48:
        raise ValueError(f"Expected 48 LULC features, found {len(lulc_cols)}")
    non_numeric_lulc = [column for column in lulc_cols if not pd.api.types.is_numeric_dtype(lulc[column])]
    if non_numeric_lulc:
        raise ValueError(f"Non-numeric LULC features: {non_numeric_lulc}")

    frame = climate_aster.merge(lulc[["sample_id"] + lulc_cols], on="sample_id", how="inner", validate="one_to_one")
    frame = frame.sort_values("sample_id").reset_index(drop=True)
    if len(frame) != 1735:
        raise ValueError(f"Expected 1,735 aligned rows, found {len(frame)}")
    if frame["Y_label"].value_counts().to_dict() != {0: 1590, 1: 145}:
        raise ValueError(f"Unexpected label distribution: {frame['Y_label'].value_counts().to_dict()}")

    feature_sets = {"aster": aster_cols, "lulc": lulc_cols, "climate": climate_cols}
    audit = {
        "rows": len(frame),
        "positive": int(frame["Y_label"].sum()),
        "negative": int((frame["Y_label"] == 0).sum()),
        "states": int(frame["state"].nunique()),
        "aster_features": len(aster_cols),
        "lulc_features": len(lulc_cols),
        "climate_adjusters": len(climate_cols),
        "aster_missing_fraction": float(frame[aster_cols].isna().mean().mean()),
        "lulc_missing_fraction": float(frame[lulc_cols].isna().mean().mean()),
        "climate_missing_fraction": float(frame[climate_cols].isna().mean().mean()),
        "input_sha256": {key: sha256(path) for key, path in paths.items()},
    }
    return frame, feature_sets, audit


def recipe(
    train_df: pd.DataFrame,
    model_key: str,
    aster_cols: list[str],
    lulc_cols: list[str],
    climate_cols: list[str],
    residual_cols: list[str],
):
    residual_set = set(residual_cols)
    ordered_residual = [column for column in aster_cols if column in residual_set]
    raw = [column for column in aster_cols if column not in residual_set] + lulc_cols
    return base.FeatureRecipe(model_key, raw, ordered_residual, climate_cols).fit(train_df)


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    out = metrics.groupby("model")[METRICS].agg(["mean", "std", "count"]).reset_index()
    out.columns = [
        "_".join(str(piece) for piece in column if str(piece)) if isinstance(column, tuple) else str(column)
        for column in out.columns
    ]
    return out


def paired(metrics: pd.DataFrame, bootstrap_repeats: int = 10000, seed: int = 20261006) -> pd.DataFrame:
    baseline = metrics[metrics["model"].eq("M2_ASTER_RAW_XGB")].set_index(["fold", "test_group"])
    rows: list[dict] = []
    rng = np.random.default_rng(seed)
    for model in ["M3_ASTER_BROAD_XGB", "M4_ASTER_SPEARMAN_XGB"]:
        candidate = metrics[metrics["model"].eq(model)].set_index(["fold", "test_group"])
        joined = baseline[METRICS].join(candidate[METRICS], lsuffix="_m2", rsuffix="_candidate")
        for metric in METRICS:
            delta = joined[f"{metric}_candidate"] - joined[f"{metric}_m2"]
            delta_values = delta.to_numpy(float)
            resampled = rng.choice(
                delta_values,
                size=(bootstrap_repeats, len(delta_values)),
                replace=True,
            ).mean(axis=1)
            rows.append(
                {
                    "comparison": f"{model} minus M2_ASTER_RAW_XGB",
                    "metric": metric,
                    "m2_mean": float(joined[f"{metric}_m2"].mean()),
                    "candidate_mean": float(joined[f"{metric}_candidate"].mean()),
                    "mean_difference": float(delta.mean()),
                    "paired_bootstrap_ci95_low": float(np.quantile(resampled, 0.025)),
                    "paired_bootstrap_ci95_high": float(np.quantile(resampled, 0.975)),
                    "better_folds": int((delta > 0).sum()),
                    "equal_folds": int(np.isclose(delta, 0).sum()),
                    "worse_folds": int((delta < 0).sum()),
                    "fold_differences": json.dumps(delta.round(6).tolist()),
                }
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


def write_report(
    output_dir: Path,
    summary: pd.DataFrame,
    comparisons: pd.DataFrame,
    fold_manifest: pd.DataFrame,
    audit: dict,
    selection_summary: pd.DataFrame,
) -> None:
    headline = summary[
        [
            "model",
            "roc_auc_mean",
            "roc_auc_std",
            "average_precision_mean",
            "average_precision_std",
            "f1_mean",
            "top05_f1_mean",
            "top05_ndcg_mean",
            "top10_f1_mean",
            "top10_ndcg_mean",
        ]
    ]
    contrast = comparisons[comparisons["metric"].isin(["average_precision", "top05_f1", "top05_ndcg"])]
    summary_by_model = summary.set_index("model")
    m2 = summary_by_model.loc["M2_ASTER_RAW_XGB"]
    m3 = summary_by_model.loc["M3_ASTER_BROAD_XGB"]
    m4 = summary_by_model.loc["M4_ASTER_SPEARMAN_XGB"]
    m3_ap_delta = float(m3["average_precision_mean"] - m2["average_precision_mean"])
    m3_top5_delta = float(m3["top05_f1_mean"] - m2["top05_f1_mean"])
    m4_ap_delta = float(m4["average_precision_mean"] - m2["average_precision_mean"])
    m4_top5_delta = float(m4["top05_f1_mean"] - m2["top05_f1_mean"])
    report = f"""# Same-input ASTER M2/M3/M4 对照实验

## 1. 实验目的

在完全相同的样本、ASTER/LULC输入、XGBoost分类器和按州五折划分下，只改变气候调整方式，检验选择性气候残差化能否迁移到ASTER观测配置。

## 2. 数据

- 样本：{audit['rows']}（正样本 {audit['positive']}，负样本 {audit['negative']}，11州）；
- ASTER：31个基础特征（VNIR 7、SWIR 11、TIR 13）；
- LULC：48个多尺度土地覆盖特征；
- 气候调节变量：24个，仅供残差器使用，不进入下游XGBoost；
- ASTER/LULC/气候平均缺失率：{audit['aster_missing_fraction']:.4f}/{audit['lulc_missing_fraction']:.4f}/{audit['climate_missing_fraction']:.4f}。

## 3. 实验配置

- 验证：5折 GroupKFold by state；
- 分类器：XGBoost，400棵树，learning_rate=0.03，max_depth=3，min_child_weight=3，subsample=0.85，colsample_bytree=0.85，reg_lambda=2.0；
- 类别权重：每个训练折内根据正负样本数计算；
- M4选择阈值：训练折内 ASTER 特征与24个气候变量的最大绝对 Spearman 相关系数 >= 0.30；
- 残差器：训练折内中位数填补、气候标准化、Ridge(alpha=10)；
- 测试折仅用于变换和评价，未用于填补、选择、残差器或模型拟合。

## 4. 方法

- **M2_ASTER_RAW_XGB**：31个原始ASTER特征 + 48个LULC特征；
- **M3_ASTER_BROAD_XGB**：对全部31个ASTER特征做气候残差化，LULC保持原值；
- **M4_ASTER_SPEARMAN_XGB**：仅残差化训练折内选中的ASTER特征，未选中的ASTER与全部LULC保持原值。

三种配置均向XGBoost提供79个下游特征，区别仅为ASTER列使用原值还是气候残差值。

## 5. 逐折划分与选择数量

{markdown_table(fold_manifest)}

M4逐折选中特征数量：

{markdown_table(selection_summary)}

## 6. 主结果

{markdown_table(headline)}

## 7. 相对M2的成对结果

{markdown_table(contrast)}

区间为对5个测试折的配对差值进行10,000次 bootstrap 得到的描述性95%区间。折数仅为5，区间只用于展示不确定性，不作高功效显著性检验。

## 8. 结果分析与结论

1. **M3全量残差化在平均指标上略优于M2，但不是所有地区都改善。** ROC-AUC由{m2['roc_auc_mean']:.4f}升至{m3['roc_auc_mean']:.4f}，AP由{m2['average_precision_mean']:.4f}升至{m3['average_precision_mean']:.4f}（差值{m3_ap_delta:+.4f}），Top-5% F1由{m2['top05_f1_mean']:.4f}升至{m3['top05_f1_mean']:.4f}（差值{m3_top5_delta:+.4f}）。AP只在2/5折提高，主要收益来自Washington和Nevada，说明效果具有明显地区异质性。
2. **M4选择性残差化未在这套基础ASTER输入上复现主实验优势。** 相对M2，M4的AP差值为{m4_ap_delta:+.4f}，Top-5% F1差值为{m4_top5_delta:+.4f}；AP仅1/5折提高。选择性调整不是跨输入模态必然增益的方法。
3. **该实验支持“气候调整效果依赖观测配置”，而不支持“所有ASTER配置上M4稳定最佳”。** 可在rebuttal中把它作为同输入、同模型、同折分的公平性补充，并如实报告M3的小幅平均收益和M4的负结果。
4. **对方法主张的影响是收缩适用范围，而非否定训练折内调整框架。** 结合既有地球化学/多模态结果，更稳妥的结论是：选择性调整在部分地学特征体系和指标上有效，但需要按观测模态验证，不能把一个选择规则直接视为普适最优。

## 9. 解释边界

本实验验证的是31维基础ASTER观测配置，不是此前611维 `ASTER_enhanced` 配置。因此，它可以回答“气候残差化是否能在另一类ASTER输入上工作”，但不能直接替代611维增强特征的同输入对照。SWIR和LULC存在缺失；处理均在训练折内完成，三种模型使用相同样本和相同输入字段。
"""
    (output_dir / "aster_same_input_m234_report.md").write_text(report, encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df, feature_sets, audit = load_dataset(data_dir, run_dir)
    y = df["Y_label"].astype(int).reset_index(drop=True)
    groups = df["state"].astype(str).reset_index(drop=True)
    aster_cols = feature_sets["aster"]
    lulc_cols = feature_sets["lulc"]
    climate_cols = feature_sets["climate"]

    metric_rows: list[dict] = []
    prediction_rows: list[dict] = []
    selection_rows: list[pd.DataFrame] = []
    fold_rows: list[dict] = []
    assignment_rows: list[dict] = []
    splitter = GroupKFold(n_splits=5)

    for fold, (train_idx, test_idx) in enumerate(splitter.split(df, y, groups), start=1):
        train_idx = np.asarray(train_idx)
        test_idx = np.asarray(test_idx)
        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()
        test_group = ";".join(sorted(groups.iloc[test_idx].unique().tolist()))

        sensitive = base.local_spearman_sensitive(train_df, aster_cols, climate_cols, args.spearman_threshold)
        selected = sensitive.loc[sensitive["is_spearman_sensitive"], "target_column"].tolist()
        sensitive = sensitive.copy()
        sensitive.insert(0, "fold", fold)
        sensitive.insert(1, "test_group", test_group)
        selection_rows.append(sensitive)

        fold_rows.append(
            {
                "fold": fold,
                "test_group": test_group,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "positive_train": int(y.iloc[train_idx].sum()),
                "positive_test": int(y.iloc[test_idx].sum()),
                "m4_selected_aster_features": len(selected),
            }
        )
        for row_index in test_idx:
            assignment_rows.append(
                {
                    "sample_id": df.iloc[row_index]["sample_id"],
                    "state": df.iloc[row_index]["state"],
                    "Y_label": int(df.iloc[row_index]["Y_label"]),
                    "fold": fold,
                    "test_group": test_group,
                    "row_index": int(row_index),
                }
            )

        configurations = [
            (
                "M2_ASTER_RAW_XGB",
                recipe(train_df, "M2_ASTER_RAW_XGB", aster_cols, lulc_cols, climate_cols, []),
            ),
            (
                "M3_ASTER_BROAD_XGB",
                recipe(train_df, "M3_ASTER_BROAD_XGB", aster_cols, lulc_cols, climate_cols, aster_cols),
            ),
            (
                "M4_ASTER_SPEARMAN_XGB",
                recipe(train_df, "M4_ASTER_SPEARMAN_XGB", aster_cols, lulc_cols, climate_cols, selected),
            ),
        ]

        for model_key, current_recipe in configurations:
            model = make_xgb_model(y.iloc[train_idx])
            train_matrix = current_recipe.transform(train_df)
            test_matrix = current_recipe.transform(test_df)
            if train_matrix.shape[1] != len(aster_cols) + len(lulc_cols):
                raise ValueError(f"Unexpected feature count for {model_key}: {train_matrix.shape[1]}")
            model.fit(train_matrix, y.iloc[train_idx])
            probability = model.predict_proba(test_matrix)[:, 1]
            base.add_metric_row(
                metric_rows,
                prediction_rows,
                dataset="aster_basic_lulc_1735",
                cv_name="groupkfold_state_same_input_aster",
                fold=fold,
                test_group=test_group,
                model_key=model_key,
                used_cols=current_recipe.output_columns(),
                train_idx=train_idx,
                test_idx=test_idx,
                df=df,
                y_prob=probability,
            )
            print(
                f"fold={fold} test={test_group} model={model_key} "
                f"AP={metric_rows[-1]['average_precision']:.4f} "
                f"F1@5={metric_rows[-1]['top05_f1']:.4f}",
                flush=True,
            )

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.DataFrame(prediction_rows)
    selections = pd.concat(selection_rows, ignore_index=True)
    fold_manifest = pd.DataFrame(fold_rows)
    assignments = pd.DataFrame(assignment_rows)
    summary = summarize(metrics)
    comparisons = paired(metrics)
    selection_summary = (
        selections.groupby(["fold", "test_group"])["is_spearman_sensitive"]
        .agg(selected="sum", total="count")
        .reset_index()
    )

    metrics.to_csv(output_dir / "aster_same_input_fold_metrics.csv", index=False)
    predictions.to_csv(output_dir / "aster_same_input_oof_predictions.csv", index=False)
    selections.to_csv(output_dir / "aster_same_input_spearman_selection.csv", index=False)
    summary.to_csv(output_dir / "aster_same_input_summary.csv", index=False)
    comparisons.to_csv(output_dir / "aster_same_input_paired_comparison.csv", index=False)
    fold_manifest.to_csv(output_dir / "aster_same_input_fold_manifest.csv", index=False)
    assignments.to_csv(output_dir / "aster_same_input_fold_assignments.csv", index=False)
    pd.DataFrame([audit]).drop(columns=["input_sha256"]).to_csv(
        output_dir / "aster_same_input_data_audit.csv", index=False
    )

    manifest = {
        "experiment": "same-input ASTER M2/M3/M4 comparison",
        "data_dir": str(data_dir),
        "run_dir_for_climate_roles": str(run_dir),
        "output_dir": str(output_dir),
        **audit,
        "validation": "5-fold GroupKFold by state",
        "models": ["M2_ASTER_RAW_XGB", "M3_ASTER_BROAD_XGB", "M4_ASTER_SPEARMAN_XGB"],
        "spearman_threshold": args.spearman_threshold,
        "paired_bootstrap": {"unit": "held-out fold", "repeats": 10000, "seed": 20261006},
        "residualizer": {"model": "Ridge", "alpha": 10.0, "fit_scope": "training fold only"},
        "classifier": "XGBoost",
        "xgboost_parameters": PARAMETERS,
        "feature_boundary": "31 basic ASTER + 48 LULC; no 611-dimensional ASTER_enhanced features",
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_report(output_dir, summary, comparisons, fold_manifest, audit, selection_summary)
    print(f"Wrote ASTER M2/M3/M4 outputs to: {output_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Same-input ASTER M2/M3/M4 rebuttal experiment.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--spearman-threshold", type=float, default=0.30)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
