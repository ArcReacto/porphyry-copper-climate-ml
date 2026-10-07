from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from xgb_model import PARAMETERS, make_xgb_model
from sklearn.cluster import DBSCAN
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import BallTree


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_DIR = (
    PROJECT_ROOT
    / "data"
    / "run_inputs"
    / "known_mining_neutral_ratio_1_10_supervised_all_features_v1"
)
DEFAULT_SAMPLE_PATH = (
    PROJECT_ROOT
    / "data"
    / "sample_schemes"
    / "ratio_1_10"
    / "samples_known_mining_neutral_ratio_1_10.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "rebuttal_experiments" / "xgboost" / "06_deposit_dedup_buffer_audit"
EARTH_RADIUS_KM = 6371.0088
MODEL_KEYS = ["NoClimate_RF", "M4_Spearman_RF"]
REPORT_METRICS = [
    "roc_auc",
    "average_precision",
    "f1",
    "top05_f1",
    "top05_ndcg",
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


exp01 = load_module(
    "rebuttal_exp01_for_audit",
    PROJECT_ROOT / "scripts" / "stage_16_rebuttal_experiments" / "01_fold_local_graphunion.py",
)


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def normalized_name(value: object) -> str:
    text = "" if pd.isna(value) else str(value).lower().strip()
    return re.sub(r"[^a-z0-9]+", "", text)


def coordinate_radians(df: pd.DataFrame) -> np.ndarray:
    coords = df[["latitude", "longitude"]].apply(pd.to_numeric, errors="coerce")
    if coords.isna().any().any():
        raise ValueError("Latitude/longitude contain missing or non-numeric values.")
    return np.radians(coords.to_numpy(dtype=float))


def make_global_ids(samples: pd.DataFrame) -> pd.Series:
    candidate = samples["state"].fillna("UNKNOWN").astype(str).str.strip() + "|" + samples["sample_id"].astype(str)
    if candidate.duplicated().any():
        candidate = (
            candidate
            + "|"
            + samples["latitude"].round(7).astype(str)
            + "|"
            + samples["longitude"].round(7).astype(str)
        )
    if candidate.duplicated().any():
        candidate = candidate + "|row=" + samples.index.astype(str)
    if candidate.duplicated().any():
        raise ValueError("Could not generate a globally unique sample key.")
    return candidate.rename("global_sample_id")


def align_samples(model_df: pd.DataFrame, sample_df: pd.DataFrame) -> pd.DataFrame:
    samples = sample_df[sample_df["Y_label"].isin([0, 1])].copy().reset_index(drop=True)
    if len(samples) != len(model_df):
        raise ValueError(f"Sample/model row mismatch: {len(samples)} vs {len(model_df)}")
    keys = ["sample_id", "state", "latitude", "longitude"]
    left = model_df[keys].reset_index(drop=True).copy()
    right = samples[keys].reset_index(drop=True).copy()
    for column in ["latitude", "longitude"]:
        if not np.allclose(
            pd.to_numeric(left[column], errors="coerce"),
            pd.to_numeric(right[column], errors="coerce"),
            equal_nan=True,
        ):
            raise ValueError(f"Model and sample tables are not aligned on {column}.")
    for column in ["sample_id", "state"]:
        if not left[column].fillna("").astype(str).equals(right[column].fillna("").astype(str)):
            raise ValueError(f"Model and sample tables are not aligned on {column}.")
    samples["row_index"] = np.arange(len(samples), dtype=int)
    samples["global_sample_id"] = make_global_ids(samples)
    return samples


def duplicate_audit(samples: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[pd.DataFrame] = []

    def add_groups(frame: pd.DataFrame, group_type: str, group_key: pd.Series) -> None:
        work = frame.copy()
        work["group_key"] = group_key.astype(str)
        work = work[work["group_key"].ne("") & work["group_key"].ne("nan")]
        counts = work["group_key"].value_counts()
        repeated = counts[counts > 1].index
        if len(repeated) == 0:
            return
        work = work[work["group_key"].isin(repeated)].copy()
        work.insert(0, "group_type", group_type)
        rows.append(work)

    add_groups(samples, "sample_id", samples["sample_id"])
    coordinate_key = samples["latitude"].round(7).astype(str) + "," + samples["longitude"].round(7).astype(str)
    add_groups(samples, "exact_coordinate", coordinate_key)

    deposit_key = pd.Series("", index=samples.index, dtype=object)
    positive = samples["Y_label"].eq(1)
    negative = samples["Y_label"].eq(0)
    deposit_key.loc[positive] = "positive_dep_id=" + samples.loc[positive, "dep_id"].astype(str)
    deposit_key.loc[negative] = "negative_source_dep_id=" + samples.loc[negative, "source_mrds_dep_id"].astype(str)
    add_groups(samples, "deposit_or_source_id", deposit_key)

    normalized = samples["site_name"].map(normalized_name)
    add_groups(samples[samples["Y_label"].eq(1)], "positive_site_name", normalized[samples["Y_label"].eq(1)])

    detail_columns = [
        "group_type",
        "group_key",
        "row_index",
        "global_sample_id",
        "sample_id",
        "state",
        "Y_label",
        "sample_type",
        "dep_id",
        "source_mrds_dep_id",
        "site_name",
        "latitude",
        "longitude",
    ]
    detail = pd.concat(rows, ignore_index=True)[detail_columns] if rows else pd.DataFrame(columns=detail_columns)
    if detail.empty:
        summary = pd.DataFrame(columns=["group_type", "duplicate_groups", "duplicate_rows", "label_conflict_groups", "cross_state_groups"])
    else:
        grouped = detail.groupby(["group_type", "group_key"], dropna=False)
        per_group = grouped.agg(rows=("row_index", "size"), labels=("Y_label", "nunique"), states=("state", "nunique")).reset_index()
        summary = (
            per_group.groupby("group_type")
            .agg(
                duplicate_groups=("group_key", "size"),
                duplicate_rows=("rows", "sum"),
                label_conflict_groups=("labels", lambda x: int((x > 1).sum())),
                cross_state_groups=("states", lambda x: int((x > 1).sum())),
            )
            .reset_index()
        )
    return detail, summary


def spatial_clusters(samples: pd.DataFrame, thresholds_km: list[float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    coords = coordinate_radians(samples)
    membership_rows: list[pd.DataFrame] = []
    summary_rows: list[dict] = []
    for threshold in thresholds_km:
        labels = DBSCAN(
            eps=threshold / EARTH_RADIUS_KM,
            min_samples=1,
            algorithm="ball_tree",
            metric="haversine",
        ).fit_predict(coords)
        work = samples[["row_index", "global_sample_id", "sample_id", "state", "Y_label", "latitude", "longitude"]].copy()
        work.insert(0, "threshold_km", threshold)
        work.insert(1, "cluster_id", labels.astype(int))
        cluster_stats = (
            work.groupby("cluster_id")
            .agg(cluster_size=("row_index", "size"), states=("state", "nunique"), labels=("Y_label", "nunique"))
            .reset_index()
        )
        work = work.merge(cluster_stats, on="cluster_id", how="left", validate="many_to_one")
        membership_rows.append(work)
        summary_rows.append(
            {
                "threshold_km": threshold,
                "clusters": int(cluster_stats.shape[0]),
                "multi_sample_clusters": int((cluster_stats["cluster_size"] > 1).sum()),
                "samples_in_multi_sample_clusters": int(
                    cluster_stats.loc[cluster_stats["cluster_size"] > 1, "cluster_size"].sum()
                ),
                "cross_state_clusters": int((cluster_stats["states"] > 1).sum()),
                "label_mixed_clusters": int((cluster_stats["labels"] > 1).sum()),
                "largest_cluster": int(cluster_stats["cluster_size"].max()),
            }
        )
    return pd.concat(membership_rows, ignore_index=True), pd.DataFrame(summary_rows)


def nearest_distance(points: np.ndarray, reference: np.ndarray) -> np.ndarray:
    if len(reference) == 0:
        return np.full(len(points), np.nan)
    tree = BallTree(reference, metric="haversine")
    distance, _ = tree.query(points, k=1)
    return distance[:, 0] * EARTH_RADIUS_KM


def cross_fold_neighbors(samples: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    groups = samples["state"].fillna("UNKNOWN").astype(str)
    coords = coordinate_radians(samples)
    rows: list[pd.DataFrame] = []
    splitter = GroupKFold(n_splits=5)
    for fold, (train_idx, test_idx) in enumerate(splitter.split(samples, y, groups), start=1):
        train_idx = np.asarray(train_idx)
        test_idx = np.asarray(test_idx)
        test_coords = coords[test_idx]
        train_coords = coords[train_idx]
        current = samples.loc[test_idx, ["row_index", "global_sample_id", "sample_id", "state", "Y_label", "latitude", "longitude"]].copy()
        current.insert(0, "fold", fold)
        current.insert(1, "test_group", ";".join(sorted(groups.iloc[test_idx].unique().tolist())))
        current["nearest_train_any_km"] = nearest_distance(test_coords, train_coords)
        current["nearest_train_positive_km"] = nearest_distance(test_coords, train_coords[y.iloc[train_idx].to_numpy() == 1])
        current["nearest_train_negative_km"] = nearest_distance(test_coords, train_coords[y.iloc[train_idx].to_numpy() == 0])
        same = np.where(
            y.iloc[test_idx].to_numpy() == 1,
            current["nearest_train_positive_km"],
            current["nearest_train_negative_km"],
        )
        opposite = np.where(
            y.iloc[test_idx].to_numpy() == 1,
            current["nearest_train_negative_km"],
            current["nearest_train_positive_km"],
        )
        current["nearest_train_same_label_km"] = same
        current["nearest_train_opposite_label_km"] = opposite
        rows.append(current)
    return pd.concat(rows, ignore_index=True)


def buffer_training_indices(
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    coords: np.ndarray,
    buffer_km: float,
) -> tuple[np.ndarray, np.ndarray]:
    if buffer_km <= 0:
        return train_idx.copy(), np.array([], dtype=int)
    tree = BallTree(coords[train_idx], metric="haversine")
    neighbor_positions = tree.query_radius(coords[test_idx], r=buffer_km / EARTH_RADIUS_KM)
    removed_positions = np.unique(np.concatenate(neighbor_positions)) if len(neighbor_positions) else np.array([], dtype=int)
    keep_mask = np.ones(len(train_idx), dtype=bool)
    keep_mask[removed_positions] = False
    return train_idx[keep_mask], train_idx[removed_positions]


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    summary = metrics.groupby(["buffer_km", "model"])[REPORT_METRICS].agg(["mean", "std", "count"]).reset_index()
    summary.columns = [
        "_".join(str(part) for part in column if str(part)) if isinstance(column, tuple) else str(column)
        for column in summary.columns
    ]
    return summary


def paired_differences(metrics: pd.DataFrame, suffix: str) -> pd.DataFrame:
    rows: list[dict] = []
    for buffer_km, part in metrics.groupby("buffer_km"):
        base = part[part["model"] == f"NoClimate_{suffix}"].set_index(["fold", "test_group"])
        candidate = part[part["model"] == f"M4_Spearman_{suffix}"].set_index(["fold", "test_group"])
        joined = base[REPORT_METRICS].join(candidate[REPORT_METRICS], lsuffix="_base", rsuffix="_candidate")
        for metric in REPORT_METRICS:
            diff = joined[f"{metric}_candidate"] - joined[f"{metric}_base"]
            rows.append(
                {
                    "buffer_km": buffer_km,
                    "metric": metric,
                    "baseline_mean": joined[f"{metric}_base"].mean(),
                    "candidate_mean": joined[f"{metric}_candidate"].mean(),
                    "mean_difference": diff.mean(),
                    "better_folds": int((diff > 0).sum()),
                    "equal_folds": int(np.isclose(diff, 0).sum()),
                    "worse_folds": int((diff < 0).sum()),
                }
            )
    return pd.DataFrame(rows)


def run_buffered_models(
    model_df: pd.DataFrame,
    samples: pd.DataFrame,
    y: pd.Series,
    role_table: pd.DataFrame,
    buffers_km: list[float],
    spearman_threshold: float,
    model_family: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    suffix = "XGB" if model_family == "xgboost" else "RF"
    groups = samples["state"].fillna("UNKNOWN").astype(str)
    coords = coordinate_radians(samples)
    fsets = exp01.base63.feature_sets(role_table)
    metric_rows: list[dict] = []
    prediction_rows: list[dict] = []
    manifest_rows: list[dict] = []
    splitter = GroupKFold(n_splits=5)

    splits = list(splitter.split(model_df, y, groups))
    for buffer_km in buffers_km:
        for fold, (base_train_idx, test_idx) in enumerate(splits, start=1):
            base_train_idx = np.asarray(base_train_idx)
            test_idx = np.asarray(test_idx)
            train_idx, removed_idx = buffer_training_indices(base_train_idx, test_idx, coords, buffer_km)
            train_y = y.iloc[train_idx]
            if train_y.nunique() < 2:
                raise ValueError(f"Buffer {buffer_km:g} km fold {fold} removed an entire class.")
            train_df = model_df.iloc[train_idx].copy()
            test_df = model_df.iloc[test_idx].copy()
            test_group = ";".join(sorted(groups.iloc[test_idx].unique().tolist()))

            manifest_rows.append(
                {
                    "buffer_km": buffer_km,
                    "fold": fold,
                    "test_group": test_group,
                    "n_train_before": len(base_train_idx),
                    "n_train_after": len(train_idx),
                    "n_train_removed": len(removed_idx),
                    "positive_train_before": int(y.iloc[base_train_idx].sum()),
                    "positive_train_after": int(train_y.sum()),
                    "positive_removed": int(y.iloc[removed_idx].sum()) if len(removed_idx) else 0,
                    "n_test": len(test_idx),
                    "positive_test": int(y.iloc[test_idx].sum()),
                }
            )

            for model_key in MODEL_KEYS:
                recipe, _ = exp01.base63.make_recipe(
                    model_key,
                    train_df,
                    fsets,
                    set(),
                    spearman_threshold,
                )
                x_train = recipe.transform(train_df)
                x_test = recipe.transform(test_df)
                model = make_xgb_model(train_y) if model_family == "xgboost" else exp01.base63.make_rf_model()
                model.fit(x_train, train_y)
                y_prob = model.predict_proba(x_test)[:, 1]
                metric_start = len(metric_rows)
                prediction_start = len(prediction_rows)
                exp01.base63.add_metric_row(
                    metric_rows,
                    prediction_rows,
                    dataset="known_mining_neutral_ratio_1_10",
                    cv_name="state_groupkfold_buffered",
                    fold=fold,
                    test_group=test_group,
                    model_key=model_key.replace("_RF", f"_{suffix}"),
                    used_cols=recipe.output_columns(),
                    train_idx=train_idx,
                    test_idx=test_idx,
                    df=model_df,
                    y_prob=y_prob,
                )
                metric_rows[metric_start]["buffer_km"] = buffer_km
                metric_rows[metric_start]["n_train_removed"] = len(removed_idx)
                for row in prediction_rows[prediction_start:]:
                    row["buffer_km"] = buffer_km
                    row["global_sample_id"] = samples.loc[row["row_index"], "global_sample_id"]

    return pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows), pd.DataFrame(manifest_rows)


def markdown_table(frame: pd.DataFrame, digits: int = 4) -> str:
    if frame.empty:
        return "（无记录）"
    display = frame.copy()
    for column in display.select_dtypes(include=["float"]).columns:
        display[column] = display[column].map(lambda value: "" if pd.isna(value) else f"{value:.{digits}f}")
    headers = [str(column) for column in display.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in display.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |")
    return "\n".join(lines)


def write_report(
    output_dir: Path,
    samples: pd.DataFrame,
    duplicate_summary: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    neighbors: pd.DataFrame,
    manifest: pd.DataFrame,
    model_summary: pd.DataFrame,
    paired: pd.DataFrame,
) -> None:
    nearest_summary = (
        neighbors.groupby(["fold", "test_group"])[
            ["nearest_train_any_km", "nearest_train_same_label_km", "nearest_train_opposite_label_km"]
        ]
        .agg(["min", "median", lambda values: values.quantile(0.05)])
        .reset_index()
    )
    nearest_summary.columns = [
        "_".join(str(part) for part in column if str(part)).replace("<lambda_0>", "p05")
        if isinstance(column, tuple)
        else str(column)
        for column in nearest_summary.columns
    ]
    result_columns = [
        "buffer_km",
        "model",
        "roc_auc_mean",
        "average_precision_mean",
        "f1_mean",
        "top05_f1_mean",
        "top05_ndcg_mean",
        "top10_f1_mean",
        "top10_ndcg_mean",
    ]
    paired_show = paired[paired["metric"].isin(["average_precision", "f1", "top05_f1", "top10_f1"])].copy()
    report = f"""# Rebuttal 实验 06：矿区去重、跨折近邻与空间缓冲审计

## 数据与唯一键

- 监督样本：{len(samples)} 条，其中正样本 {int(samples['Y_label'].sum())}、hard negative {int((samples['Y_label'] == 0).sum())}。
- 原 `sample_id` 唯一值：{samples['sample_id'].nunique()}；跨州重复编号涉及 {int(samples['sample_id'].duplicated(keep=False).sum())} 行。
- 新 `global_sample_id = state | sample_id` 唯一值：{samples['global_sample_id'].nunique()}，重复 0 行。
- 精确坐标唯一值：{samples[['latitude', 'longitude']].drop_duplicates().shape[0]}。

## 重复记录审计

{markdown_table(duplicate_summary)}

`duplicate_groups.csv` 保留了每个重复组的逐行明细；重复并不自动等同于泄漏，需结合是否跨折判断。

## 空间邻近簇

{markdown_table(cluster_summary)}

DBSCAN 使用球面距离且 `min_samples=1`。`cross_state_clusters` 是州分组验证中最需要关注的跨折邻近簇。

## 州分组折的训练—测试最近距离

{markdown_table(nearest_summary)}

## 缓冲后训练集规模

{markdown_table(manifest)}

缓冲操作仅移除距任一测试点小于阈值的训练样本，测试集不变。每个缓冲阈值均重新进行折内 Spearman 特征选择与残差化。

## 缓冲验证结果

{markdown_table(model_summary[result_columns])}

## M4 Spearman 相对 M2

{markdown_table(paired_show)}

## 解释边界

1. 该实验排查近邻记忆和跨州边界泄漏，不证明样本标签本身无噪声。
2. 距离缓冲会同时改变训练规模和空间代表性，性能下降不能全部归因于泄漏。
3. 名称和 deposit/source ID 的重复只用于审计；本实验没有擅自删除同矿床的多记录，避免改变论文数据定义。
4. 若缓冲后结论改变，应在 rebuttal 中报告敏感性，不应只保留无缓冲结果。
"""
    (output_dir / "deposit_dedup_buffer_audit_report.md").write_text(report, encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_df, dataset_path = exp01.base63.load_dataset_from_run(run_dir)
    model_df = model_df[model_df[exp01.base63.TARGET].isin([0, 1])].reset_index(drop=True)
    y = model_df[exp01.base63.TARGET].astype(int).reset_index(drop=True)
    samples = align_samples(model_df, read_table(Path(args.sample_path)))
    role_table = pd.read_csv(run_dir / "00_dataset_profile" / "feature_roles.csv")

    duplicate_detail, duplicate_summary = duplicate_audit(samples)
    cluster_membership, cluster_summary = spatial_clusters(samples, args.cluster_thresholds_km)
    neighbors = cross_fold_neighbors(samples, y)
    metrics, predictions, manifest = run_buffered_models(
        model_df,
        samples,
        y,
        role_table,
        args.buffers_km,
        args.spearman_threshold,
        args.model_family,
    )
    model_summary = summarize_metrics(metrics)
    suffix = "XGB" if args.model_family == "xgboost" else "RF"
    paired = paired_differences(metrics, suffix)

    samples.to_csv(output_dir / "sample_global_id_audit.csv", index=False)
    duplicate_detail.to_csv(output_dir / "duplicate_groups.csv", index=False)
    duplicate_summary.to_csv(output_dir / "duplicate_summary.csv", index=False)
    cluster_membership.to_csv(output_dir / "deposit_clusters.csv", index=False)
    cluster_summary.to_csv(output_dir / "deposit_cluster_summary.csv", index=False)
    neighbors.to_csv(output_dir / "cross_fold_neighbor_audit.csv", index=False)
    manifest.to_csv(output_dir / "buffered_dataset_manifest.csv", index=False)
    metrics.to_csv(output_dir / "buffered_fold_metrics.csv", index=False)
    predictions.to_csv(output_dir / "buffered_oof_predictions.csv", index=False)
    model_summary.to_csv(output_dir / "buffered_model_summary.csv", index=False)
    paired.to_csv(output_dir / "buffered_paired_differences.csv", index=False)

    write_report(
        output_dir,
        samples,
        duplicate_summary,
        cluster_summary,
        neighbors,
        manifest,
        model_summary,
        paired,
    )
    manifest_json = {
        "dataset_path": str(dataset_path),
        "sample_path": str(Path(args.sample_path)),
        "run_dir": str(run_dir),
        "rows": len(samples),
        "positive": int(y.sum()),
        "negative": int((y == 0).sum()),
        "cluster_thresholds_km": args.cluster_thresholds_km,
        "buffers_km": args.buffers_km,
        "models": [key.replace("_RF", f"_{suffix}") for key in MODEL_KEYS],
        "model_family": args.model_family,
        "xgboost_parameters": PARAMETERS if args.model_family == "xgboost" else None,
        "global_sample_id_definition": "state|sample_id",
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest_json, indent=2), encoding="utf-8")
    print(f"Wrote rebuttal experiment 06 outputs to: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit duplicate deposits, cross-fold neighbors, and buffered validation.")
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--sample-path", default=str(DEFAULT_SAMPLE_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--cluster-thresholds-km", nargs="+", type=float, default=[1.0, 5.0, 10.0, 20.0])
    parser.add_argument("--buffers-km", nargs="+", type=float, default=[0.0, 10.0, 20.0, 30.0])
    parser.add_argument("--spearman-threshold", type=float, default=0.30)
    parser.add_argument("--model-family", choices=["xgboost", "rf"], default="xgboost")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
