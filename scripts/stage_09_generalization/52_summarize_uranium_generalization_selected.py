from __future__ import annotations

from pathlib import Path
import math
import json

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_NAME = "uranium_11state_generalization_ratio_1_10_supervised_all_features_v1"
RUN_DIR = PROJECT_ROOT / "outputs" / "standardized_runs" / RUN_NAME
OUT_DIR = PROJECT_ROOT / "outputs" / "paper_generalization" / "uranium_11state_generalization_ratio_1_10"

BASELINE_MODELS = ["dummy_stratified", "logistic_regression", "random_forest", "hist_gradient_boosting"]
CLIMATE_MODELS = ["M2_No_Climate", "M4_Sensitive_Residualized"]


def top_k_metrics(y_true: np.ndarray, y_prob: np.ndarray, frac: float) -> dict[str, float]:
    order = np.argsort(-y_prob)
    ranked_true = y_true[order]
    total_pos = max(int(y_true.sum()), 1)
    base_rate = float(y_true.mean()) if len(y_true) else math.nan
    k = max(1, int(math.ceil(len(y_true) * frac)))
    selected = ranked_true[:k]
    precision_at_k = float(selected.mean())
    recall_at_k = float(selected.sum() / total_pos)
    f1_at_k = (
        2 * precision_at_k * recall_at_k / (precision_at_k + recall_at_k)
        if precision_at_k + recall_at_k > 0
        else 0.0
    )
    discounts = 1.0 / np.log2(np.arange(2, len(selected) + 2))
    dcg = float(np.sum(selected.astype(float) * discounts))
    ideal = np.sort(ranked_true)[::-1][:k].astype(float)
    ideal_dcg = float(np.sum(ideal * discounts))
    return {
        f"top{int(frac*100):02d}_precision": precision_at_k,
        f"top{int(frac*100):02d}_recall": recall_at_k,
        f"top{int(frac*100):02d}_f1": f1_at_k,
        f"top{int(frac*100):02d}_lift": precision_at_k / base_rate if base_rate else math.nan,
        f"top{int(frac*100):02d}_ndcg": dcg / ideal_dcg if ideal_dcg > 0 else math.nan,
    }


def model_group_for(model: str, default: str) -> str:
    if model == "M2_No_Climate":
        return "m2_no_climate"
    if model == "M4_Sensitive_Residualized":
        return "m4_residualized"
    return default


def summarize_predictions(path: Path, keep_models: list[str], model_group: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    pred = pd.read_csv(path)
    pred = pred[pred["model"].isin(keep_models)].copy()
    rows = []
    for (dataset, experiment, cv, fold, test_group, model), sub in pred.groupby(
        ["dataset", "experiment", "cv", "fold", "test_group", "model"],
        dropna=False,
    ):
        y_true = sub["Y_label"].astype(int).to_numpy()
        y_prob = sub["y_prob"].astype(float).to_numpy()
        row = {
            "dataset": dataset,
            "experiment": experiment,
            "cv": cv,
            "fold": int(fold),
            "test_group": "" if pd.isna(test_group) else str(test_group),
            "model": model,
            "model_group": model_group_for(str(model), model_group),
            "n_test": int(len(sub)),
            "positive_test": int(y_true.sum()),
        }
        row.update(top_k_metrics(y_true, y_prob, 0.05))
        row.update(top_k_metrics(y_true, y_prob, 0.10))
        rows.append(row)
    fold_metrics = pd.DataFrame(rows)
    metric_cols = [c for c in fold_metrics.columns if c.startswith("top")]
    summary = (
        fold_metrics.groupby(["dataset", "experiment", "cv", "model", "model_group"], dropna=False)[metric_cols]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary.columns = [
        "_".join(str(part) for part in col if str(part)) if isinstance(col, tuple) else str(col)
        for col in summary.columns
    ]
    return fold_metrics, summary


def read_standard_summary() -> pd.DataFrame:
    baseline = pd.read_csv(RUN_DIR / "01_baseline_models" / "baseline_summary.csv")
    baseline = baseline[baseline["model"].isin(BASELINE_MODELS)].copy()
    baseline["model_group"] = "baseline"

    climate = pd.read_csv(RUN_DIR / "02_climate_decoupling" / "climate_decoupling_summary.csv")
    climate = climate[climate["model"].isin(CLIMATE_MODELS)].copy()
    climate["model_group"] = climate["model"].map(
        {
            "M2_No_Climate": "m2_no_climate",
            "M4_Sensitive_Residualized": "m4_residualized",
        }
    )
    return pd.concat([baseline, climate], ignore_index=True)


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No data._"
    lines = [
        "| " + " | ".join(df.columns) + " |",
        "| " + " | ".join(["---"] * len(df.columns)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in df.columns) + " |")
    return "\n".join(lines)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    baseline_fold, baseline_topk = summarize_predictions(
        RUN_DIR / "01_baseline_models" / "baseline_predictions.csv",
        BASELINE_MODELS,
        "baseline",
    )
    climate_fold, climate_topk = summarize_predictions(
        RUN_DIR / "02_climate_decoupling" / "climate_decoupling_predictions.csv",
        CLIMATE_MODELS,
        "climate_decoupling",
    )
    topk_fold = pd.concat([baseline_fold, climate_fold], ignore_index=True)
    topk_summary = pd.concat([baseline_topk, climate_topk], ignore_index=True)

    standard = read_standard_summary()
    merged = standard.merge(
        topk_summary,
        on=["dataset", "experiment", "cv", "model", "model_group"],
        how="left",
    )

    selected_cols = [
        "cv",
        "model_group",
        "model",
        "roc_auc_mean",
        "average_precision_mean",
        "f1_mean",
        "top05_precision_mean",
        "top05_recall_mean",
        "top05_f1_mean",
        "top05_ndcg_mean",
        "top10_f1_mean",
        "top10_ndcg_mean",
    ]
    table = merged[selected_cols].copy()
    for col in table.columns:
        if col.endswith("_mean"):
            table[col] = table[col].astype(float).round(4)
    table = table.sort_values(["cv", "model_group", "model"]).reset_index(drop=True)

    topk_fold.to_csv(OUT_DIR / "uranium_selected_generalization_topk_by_fold.csv", index=False, encoding="utf-8-sig")
    merged.to_csv(OUT_DIR / "uranium_selected_generalization_summary_full.csv", index=False, encoding="utf-8-sig")
    table.to_csv(OUT_DIR / "uranium_selected_generalization_summary_compact.csv", index=False, encoding="utf-8-sig")

    dataset_summary_path = (
        PROJECT_ROOT
        / "outputs"
        / "model_datasets"
        / "by_sample_scheme"
        / "by_target"
        / "uranium"
        / "known_mining_neutral"
        / "uranium_11state_known_mining_neutral_ratio_1_10"
        / "dataset_summary.json"
    )
    dataset_summary = json.loads(dataset_summary_path.read_text(encoding="utf-8")) if dataset_summary_path.exists() else {}

    report = [
        "# Uranium 11-State Generalization Experiment",
        "",
        "This experiment uses the uranium 11-state positive CSV in the data folder and the known-mining-nearby 1:10 sampling rule.",
        "",
        "## Data Check",
        "",
        "- Positive uranium samples: 173.",
        "- Positive states: Wyoming, Colorado, New Mexico, Nevada.",
        "- All positive samples are inside the 11-state western-core study area and pass state-boundary geometry checks.",
        "- Supervised training uses positives and hard negatives only; neutral samples are retained in the dataset package but not used for supervised metrics.",
        "",
        "## Sampling Summary",
        "",
        f"- Positive rows: {dataset_summary.get('samples', {}).get('positive_rows', 'NA')}",
        f"- Hard negative rows: {dataset_summary.get('samples', {}).get('negative_rows', 'NA')}",
        f"- Neutral rows: {dataset_summary.get('samples', {}).get('neutral_rows', 'NA')}",
        "",
        "## Selected Results",
        "",
        markdown_table(table),
        "",
        "## Interpretation",
        "",
        "- Stratified validation shows high within-distribution predictability for uranium points, especially for tree-based baselines.",
        "- State-grouped validation is much harder because uranium positives are concentrated in only four states. The low GroupKFold scores should be interpreted as cross-state spatial generalization difficulty rather than ordinary random-split performance.",
        "- M4 residualization does not clearly dominate M2 on uranium under this quick generalization setting, so uranium is better used as a robustness/generalization audit rather than the main evidence for the climate-decoupling claim.",
    ]
    (OUT_DIR / "uranium_11state_generalization_report.md").write_text("\n".join(report), encoding="utf-8")
    print(OUT_DIR / "uranium_selected_generalization_summary_compact.csv")
    print(OUT_DIR / "uranium_11state_generalization_report.md")


if __name__ == "__main__":
    main()
