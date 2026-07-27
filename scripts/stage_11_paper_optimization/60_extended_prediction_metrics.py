from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "paper_optimization"
TOP_K_FRACTIONS = (0.01, 0.05, 0.10, 0.20)


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


def safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return math.nan
    return float(roc_auc_score(y_true, y_prob))


def safe_ap(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return math.nan
    return float(average_precision_score(y_true, y_prob))


def ndcg_at_k(y_true_sorted: np.ndarray, k: int) -> float:
    selected = y_true_sorted[:k].astype(float)
    if len(selected) == 0:
        return math.nan
    discounts = 1.0 / np.log2(np.arange(2, len(selected) + 2))
    dcg = float(np.sum(selected * discounts))
    ideal = np.sort(y_true_sorted)[::-1][:k].astype(float)
    ideal_dcg = float(np.sum(ideal * discounts))
    return dcg / ideal_dcg if ideal_dcg > 0 else math.nan


def top_k_metrics(y_true: np.ndarray, y_prob: np.ndarray, fractions: tuple[float, ...]) -> dict[str, float]:
    order = np.argsort(-y_prob)
    ranked_true = y_true[order]
    total_pos = max(int(y_true.sum()), 1)
    base_rate = float(y_true.mean()) if len(y_true) else math.nan
    out: dict[str, float] = {}
    for frac in fractions:
        k = max(1, int(math.ceil(len(y_true) * frac)))
        selected = ranked_true[:k]
        precision_at_k = float(selected.mean()) if k else math.nan
        recall_at_k = float(selected.sum() / total_pos)
        f1_at_k = (
            2 * precision_at_k * recall_at_k / (precision_at_k + recall_at_k)
            if precision_at_k + recall_at_k > 0
            else 0.0
        )
        label = str(int(frac * 100)).zfill(2)
        out[f"top{label}_k"] = int(k)
        out[f"top{label}_precision"] = precision_at_k
        out[f"top{label}_recall"] = recall_at_k
        out[f"top{label}_f1"] = f1_at_k
        out[f"top{label}_lift"] = precision_at_k / base_rate if base_rate and not pd.isna(base_rate) else math.nan
        out[f"top{label}_ndcg"] = ndcg_at_k(ranked_true, k)
    return out


def binary_metrics(group: pd.DataFrame) -> dict[str, float]:
    y_true = group["Y_label"].astype(int).to_numpy()
    y_prob = group["y_prob"].astype(float).to_numpy()
    y_pred = (y_prob >= 0.5).astype(int)
    out = {
        "n": int(len(group)),
        "n_pos": int(y_true.sum()),
        "n_neg": int((y_true == 0).sum()),
        "roc_auc": safe_auc(y_true, y_prob),
        "average_precision": safe_ap(y_true, y_prob),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)) if len(y_true) else math.nan,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)) if len(y_true) else math.nan,
        "recall": float(recall_score(y_true, y_pred, zero_division=0)) if len(y_true) else math.nan,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)) if len(y_true) else math.nan,
    }
    out.update(top_k_metrics(y_true, y_prob, TOP_K_FRACTIONS))
    return out


def base_group_columns(df: pd.DataFrame) -> list[str]:
    preferred = ["dataset", "experiment", "cv", "model"]
    return [c for c in preferred if c in df.columns]


def fold_group_columns(df: pd.DataFrame) -> list[str]:
    cols = base_group_columns(df)
    if "fold" in df.columns:
        cols.append("fold")
    return cols


def summarize_predictions(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required = {"Y_label", "y_prob"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Prediction table is missing columns: {sorted(missing)}")
    df = df[df["Y_label"].isin([0, 1])].copy()
    df["Y_label"] = df["Y_label"].astype(int)

    fold_rows = []
    for keys, group in df.groupby(fold_group_columns(df), dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(fold_group_columns(df), keys))
        row.update(binary_metrics(group))
        fold_rows.append(row)
    fold_metrics = pd.DataFrame(fold_rows)

    model_rows = []
    for keys, group in df.groupby(base_group_columns(df), dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(base_group_columns(df), keys))
        row.update(binary_metrics(group))
        model_rows.append(row)
    model_metrics = pd.DataFrame(model_rows)

    group_rows = []
    for group_col in ["state", "env_causal_group_id", "env_weathering_regime", "zone"]:
        if group_col not in df.columns:
            continue
        for keys, group in df.groupby(base_group_columns(df) + [group_col], dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            row = dict(zip(base_group_columns(df) + [group_col], keys))
            row["group_column"] = group_col
            row["group_value"] = str(row.pop(group_col))
            row.update(binary_metrics(group))
            group_rows.append(row)
    group_metrics = pd.DataFrame(group_rows)

    return fold_metrics, model_metrics, group_metrics


def worst_group_summary(group_metrics: pd.DataFrame) -> pd.DataFrame:
    if group_metrics.empty:
        return group_metrics
    metric_cols = ["roc_auc", "average_precision", "balanced_accuracy", "recall", "f1"]
    rows = []
    for keys, group in group_metrics.groupby(base_group_columns(group_metrics) + ["group_column"], dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(base_group_columns(group_metrics) + ["group_column"], keys))
        valid = group[group["n_pos"] > 0].copy()
        for metric in metric_cols:
            metric_valid = valid.dropna(subset=[metric])
            if metric_valid.empty:
                row[f"worst_{metric}"] = math.nan
                row[f"worst_{metric}_group"] = ""
            else:
                worst = metric_valid.sort_values(metric, ascending=True).iloc[0]
                row[f"worst_{metric}"] = float(worst[metric])
                row[f"worst_{metric}_group"] = str(worst["group_value"])
        rows.append(row)
    return pd.DataFrame(rows)


def compact_markdown_table(df: pd.DataFrame, max_rows: int = 80) -> str:
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


def write_report(out_dir: Path, model_metrics: pd.DataFrame, worst: pd.DataFrame, input_paths: list[Path]) -> None:
    key_cols = [c for c in ["dataset", "experiment", "cv", "model"] if c in model_metrics.columns]
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
    display_cols = key_cols + [c for c in metric_cols if c in model_metrics.columns]
    report = [
        "# Extended Prediction Metrics",
        "",
        "## Inputs",
        "",
        *[f"- `{path}`" for path in input_paths],
        "",
        "## Model-Level Metrics",
        "",
        compact_markdown_table(model_metrics[display_cols].sort_values(key_cols) if key_cols else model_metrics),
        "",
        "## Worst-Group Metrics",
        "",
        compact_markdown_table(worst.head(120)),
        "",
        "## Notes",
        "",
        "- `average_precision` is PR-AUC / AUPRC.",
        "- Top-K metrics rank samples by predicted probability and evaluate the highest-scored locations.",
        "- `topXX_f1` is computed from Precision@K and Recall@K; `topXX_ndcg` measures whether positives are concentrated at the very top of the ranked candidate list.",
        "- Worst-group metrics report the weakest state or environment subgroup where labels contain positives.",
    ]
    (out_dir / "extended_metrics_report.md").write_text("\n".join(report), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute AP, Top-K, lift, and worst-group metrics from prediction files.")
    parser.add_argument("--predictions", nargs="+", required=True, help="Prediction CSV/parquet files.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT / "extended_metrics"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    input_paths = [Path(p) for p in args.predictions]
    frames = []
    for path in input_paths:
        frame = read_table(path)
        frame["prediction_file"] = path.name
        frames.append(frame)
    predictions = pd.concat(frames, ignore_index=True)

    fold_metrics, model_metrics, group_metrics = summarize_predictions(predictions)
    worst = worst_group_summary(group_metrics)

    write_table(fold_metrics, out_dir / "extended_fold_metrics.csv")
    write_table(model_metrics, out_dir / "extended_model_metrics.csv")
    write_table(group_metrics, out_dir / "extended_group_metrics.csv")
    write_table(worst, out_dir / "extended_worst_group_metrics.csv")
    write_report(out_dir, model_metrics, worst, input_paths)

    manifest = {
        "inputs": [str(path) for path in input_paths],
        "outputs": {
            "fold_metrics": str(out_dir / "extended_fold_metrics.csv"),
            "model_metrics": str(out_dir / "extended_model_metrics.csv"),
            "group_metrics": str(out_dir / "extended_group_metrics.csv"),
            "worst_group_metrics": str(out_dir / "extended_worst_group_metrics.csv"),
            "report": str(out_dir / "extended_metrics_report.md"),
        },
        "top_k_fractions": list(TOP_K_FRACTIONS),
    }
    (out_dir / "extended_metrics_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote extended metrics to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
