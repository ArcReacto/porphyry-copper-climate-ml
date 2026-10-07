from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from xgb_model import PARAMETERS, make_xgb_model


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


base = load_module(
    "rebuttal_aster_enhanced_base63",
    PROJECT_ROOT
    / "scripts"
    / "stage_11_paper_optimization"
    / "63_full_feature_graph_guided_m4_and_perturbation.py",
)


DEFAULT_INPUT_ROOT = (
    PROJECT_ROOT / "data" / "aster_enhanced"
)
DEFAULT_MAIN_DATA = (
    PROJECT_ROOT
    / "data"
    / "run_inputs"
    / "known_mining_neutral_ratio_1_10_supervised_all_features_v1"
    / "model_dataset_known_mining_neutral_ratio_1_10_supervised_all_features_v1.csv"
)
DEFAULT_ROLE_TABLE = (
    PROJECT_ROOT
    / "data"
    / "run_inputs"
    / "known_mining_neutral_ratio_1_10_supervised_all_features_v1"
    / "00_dataset_profile"
    / "feature_roles.csv"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "rebuttal_experiments"
    / "xgboost"
    / "18_aster_enhanced_same_input_m234"
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_dataset(
    input_root: Path,
    main_data_path: Path,
    role_table_path: Path,
) -> tuple[pd.DataFrame, list[str], list[str], dict]:
    aster_path = input_root / "A132_enhanced_features_binary_1738.csv"
    feature_manifest_path = input_root / "A132_feature_manifest.csv"
    split_path = input_root / "fixed_splits_stage1.csv"
    required = [aster_path, feature_manifest_path, split_path, main_data_path, role_table_path]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required inputs: {missing}")

    aster = pd.read_csv(aster_path, low_memory=False)
    feature_manifest = pd.read_csv(feature_manifest_path)
    features = feature_manifest["feature"].astype(str).tolist()
    if len(features) != 132 or len(set(features)) != 132:
        raise ValueError("Feature manifest must contain 132 unique features")
    absent = sorted(set(features) - set(aster.columns))
    if absent:
        raise ValueError(f"ASTER table is missing manifest features: {absent}")
    if aster["global_sample_id"].duplicated().any():
        raise ValueError("global_sample_id is not unique")
    if aster["global_sample_id"].isna().any():
        raise ValueError("global_sample_id contains missing values")
    if aster[features].isna().any().any():
        raise ValueError("ASTER enhanced features contain missing values")
    values = aster[features].to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("ASTER enhanced features contain non-finite values")
    if aster["Y_label"].value_counts().to_dict() != {0: 1580, 1: 158}:
        raise ValueError(f"Unexpected label distribution: {aster['Y_label'].value_counts().to_dict()}")

    roles = pd.read_csv(role_table_path)
    climate_cols = roles.loc[
        roles["is_numeric"].astype(bool)
        & roles["used_as_climate_adjuster"].astype(bool),
        "column",
    ].astype(str).tolist()
    if len(climate_cols) != 24 or len(set(climate_cols)) != 24:
        raise ValueError(f"Expected 24 unique climate adjusters, found {len(climate_cols)}")

    main = pd.read_csv(main_data_path, low_memory=False)
    required_main = {"sample_id", "state", "Y_label", "latitude", "longitude", *climate_cols}
    absent_main = sorted(required_main - set(main.columns))
    if absent_main:
        raise ValueError(f"Main benchmark is missing columns: {absent_main}")
    if main.duplicated(["sample_id", "state"]).any():
        raise ValueError("Main benchmark is not unique by (sample_id, state)")

    main_subset = main[["sample_id", "state", "Y_label", "latitude", "longitude", *climate_cols]].copy()
    frame = aster.merge(
        main_subset,
        on=["sample_id", "state"],
        how="left",
        validate="one_to_one",
        suffixes=("_aster", "_main"),
        indicator=True,
    )
    if not frame["_merge"].eq("both").all():
        raise ValueError("ASTER rows do not fully match the main benchmark")
    if not np.array_equal(frame["Y_label_aster"].to_numpy(), frame["Y_label_main"].to_numpy()):
        raise ValueError("Label mismatch between ASTER and the main benchmark")
    for coordinate in ["latitude", "longitude"]:
        delta = (frame[f"{coordinate}_aster"] - frame[f"{coordinate}_main"]).abs()
        if float(delta.max()) > 1e-9:
            raise ValueError(f"Coordinate mismatch for {coordinate}: max delta={delta.max()}")
    if frame[climate_cols].isna().any().any():
        raise ValueError("Climate adjusters contain missing values")

    splits = pd.read_csv(split_path, low_memory=False)
    split_subset = splits.loc[
        splits["label"].isin([0, 1]),
        ["sample_id", "region_id", "split_id", "label"],
    ].rename(columns={"region_id": "state", "label": "split_label"})
    if split_subset.duplicated(["sample_id", "state"]).any():
        raise ValueError("Binary split table is not unique by (sample_id, state)")
    frame = frame.merge(
        split_subset,
        on=["sample_id", "state"],
        how="left",
        validate="one_to_one",
        indicator="split_merge",
    )
    if not frame["split_merge"].eq("both").all():
        raise ValueError("Fixed split table does not cover every ASTER row")
    if not np.array_equal(frame["Y_label_aster"].to_numpy(), frame["split_label"].to_numpy()):
        raise ValueError("Label mismatch between ASTER and fixed splits")

    frame = frame.rename(
        columns={
            "Y_label_aster": "Y_label",
            "latitude_aster": "latitude",
            "longitude_aster": "longitude",
        }
    ).drop(
        columns=[
            "Y_label_main",
            "latitude_main",
            "longitude_main",
            "_merge",
            "split_merge",
            "split_label",
        ]
    )
    frame = frame.sort_values("global_sample_id").reset_index(drop=True)

    audit = {
        "rows": int(len(frame)),
        "positive": int(frame["Y_label"].sum()),
        "negative": int((frame["Y_label"] == 0).sum()),
        "states": int(frame["state"].nunique()),
        "aster_features": len(features),
        "climate_adjusters": len(climate_cols),
        "aster_missing_fraction": float(frame[features].isna().mean().mean()),
        "climate_missing_fraction": float(frame[climate_cols].isna().mean().mean()),
        "folds": int(frame["split_id"].nunique()),
        "input_sha256": {
            "aster": sha256(aster_path),
            "feature_manifest": sha256(feature_manifest_path),
            "fixed_splits": sha256(split_path),
            "main_benchmark": sha256(main_data_path),
            "feature_roles": sha256(role_table_path),
        },
    }
    return frame, features, climate_cols, audit


def make_recipe(
    train_df: pd.DataFrame,
    model_key: str,
    feature_cols: list[str],
    climate_cols: list[str],
    residual_cols: list[str],
):
    residual_set = set(residual_cols)
    ordered_residual = [column for column in feature_cols if column in residual_set]
    raw_cols = [column for column in feature_cols if column not in residual_set]
    return base.FeatureRecipe(model_key, raw_cols, ordered_residual, climate_cols).fit(train_df)


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    out = metrics.groupby("model")[METRICS].agg(["mean", "std", "count"]).reset_index()
    out.columns = [
        "_".join(str(piece) for piece in column if str(piece))
        if isinstance(column, tuple)
        else str(column)
        for column in out.columns
    ]
    return out


def paired(metrics: pd.DataFrame, repeats: int = 10000, seed: int = 20261006) -> pd.DataFrame:
    baseline = metrics[metrics["model"].eq("M2_A132_RAW_XGB")].set_index(["fold", "test_group"])
    rng = np.random.default_rng(seed)
    rows = []
    for model in ["M3_A132_BROAD_XGB", "M4_A132_SPEARMAN_XGB"]:
        candidate = metrics[metrics["model"].eq(model)].set_index(["fold", "test_group"])
        joined = baseline[METRICS].join(candidate[METRICS], lsuffix="_m2", rsuffix="_candidate")
        for metric in METRICS:
            delta = joined[f"{metric}_candidate"] - joined[f"{metric}_m2"]
            values = delta.to_numpy(float)
            boot = rng.choice(values, size=(repeats, len(values)), replace=True).mean(axis=1)
            rows.append(
                {
                    "comparison": f"{model} minus M2_A132_RAW_XGB",
                    "metric": metric,
                    "m2_mean": float(joined[f"{metric}_m2"].mean()),
                    "candidate_mean": float(joined[f"{metric}_candidate"].mean()),
                    "mean_difference": float(delta.mean()),
                    "paired_bootstrap_ci95_low": float(np.quantile(boot, 0.025)),
                    "paired_bootstrap_ci95_high": float(np.quantile(boot, 0.975)),
                    "better_folds": int((delta > 0).sum()),
                    "equal_folds": int(np.isclose(delta, 0).sum()),
                    "worse_folds": int((delta < 0).sum()),
                    "fold_differences": json.dumps(delta.round(6).tolist()),
                }
            )
    return pd.DataFrame(rows)


def pooled(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, group in predictions.groupby("model"):
        y_true = group["Y_label"].to_numpy(int)
        y_prob = group["y_prob"].to_numpy(float)
        y_pred = (y_prob >= 0.5).astype(int)
        row = {
            "model": model,
            "n": int(len(group)),
            "positive": int(y_true.sum()),
            "roc_auc": base.safe_metric("roc_auc", y_true, y_pred, y_prob),
            "average_precision": base.safe_metric("average_precision", y_true, y_pred, y_prob),
            "f1": base.safe_metric("f1", y_true, y_pred, y_prob),
        }
        row.update(base.top_k_metrics(y_true, y_prob))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("model")


def markdown_table(frame: pd.DataFrame, digits: int = 4) -> str:
    show = frame.copy()
    for column in show.select_dtypes(include=["float"]).columns:
        show[column] = show[column].map(
            lambda value: "" if pd.isna(value) else f"{value:.{digits}f}"
        )
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
    pooled_metrics: pd.DataFrame,
    comparisons: pd.DataFrame,
    fold_manifest: pd.DataFrame,
    selection_summary: pd.DataFrame,
    audit: dict,
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
    pooled_show = pooled_metrics[
        [
            "model",
            "roc_auc",
            "average_precision",
            "f1",
            "top05_precision",
            "top05_recall",
            "top05_f1",
            "top05_ndcg",
            "top10_f1",
            "top10_ndcg",
        ]
    ]
    contrast = comparisons[comparisons["metric"].isin(["average_precision", "top05_f1", "top05_ndcg"])]
    by_model = summary.set_index("model")
    m2 = by_model.loc["M2_A132_RAW_XGB"]
    m3 = by_model.loc["M3_A132_BROAD_XGB"]
    m4 = by_model.loc["M4_A132_SPEARMAN_XGB"]
    report = f"""# Enhanced ASTER same-input M2/M3/M4 experiment

## 1. Purpose

This experiment tests climate residualization on the same 132-dimensional enhanced ASTER input. The sample pool, fixed state folds, downstream XGBoost, and evaluation are identical across M2/M3/M4; only the fold-local observation adjustment differs.

## 2. Data and boundary

- Samples: {audit['rows']} ({audit['positive']} positive, {audit['negative']} negative; exact 1:10 benchmark);
- ASTER: 132 enhanced A1-A3 features (56 ratios, 52 normalized differences, 24 alteration indices);
- Climate: 24 adjusters from the main benchmark, used only by the residualizer;
- Validation: five fixed state-grouped folds;
- LULC is intentionally excluded because the available 1,735-row table belongs to a different coordinate cohort.

## 3. Configurations

- **M2_A132_RAW_XGB**: all 132 enhanced ASTER features in their original form;
- **M3_A132_BROAD_XGB**: all 132 ASTER features residualized against climate within each training fold;
- **M4_A132_SPEARMAN_XGB**: only ASTER features whose maximum absolute training-fold Spearman correlation with the 24 climate adjusters is at least 0.30 are residualized; all other ASTER features remain raw.

Median handling, climate scaling, Spearman selection, Ridge(alpha=10) fitting, and XGBoost fitting are all training-fold-only. Climate variables never enter the downstream classifier.

## 4. Fixed folds and M4 selection

{markdown_table(fold_manifest)}

{markdown_table(selection_summary)}

## 5. Mean fold metrics

{markdown_table(headline)}

## 6. Pooled OOF metrics

{markdown_table(pooled_show)}

## 7. Paired fold differences relative to M2

{markdown_table(contrast)}

The intervals above are descriptive 10,000-repeat bootstrap intervals over only five held-out folds and are not high-power significance tests.

## 8. Result summary

- M3 minus M2 mean-fold AP: {m3['average_precision_mean'] - m2['average_precision_mean']:+.4f}; Top-5% F1: {m3['top05_f1_mean'] - m2['top05_f1_mean']:+.4f}.
- M4 minus M2 mean-fold AP: {m4['average_precision_mean'] - m2['average_precision_mean']:+.4f}; Top-5% F1: {m4['top05_f1_mean'] - m2['top05_f1_mean']:+.4f}.
- These results apply to the enhanced ASTER-only observation configuration and should not be described as universal across feature modalities or spatial transfer settings.
"""
    (output_dir / "aster_enhanced_same_input_m234_report.md").write_text(report, encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame, feature_cols, climate_cols, audit = load_dataset(
        input_root,
        Path(args.main_data),
        Path(args.role_table),
    )

    y = frame["Y_label"].astype(int).reset_index(drop=True)
    metric_rows: list[dict] = []
    prediction_rows: list[dict] = []
    selection_rows: list[pd.DataFrame] = []
    fold_rows: list[dict] = []
    assignment_rows: list[dict] = []

    fold_ids = sorted(frame["split_id"].unique().tolist())
    if len(fold_ids) != 5:
        raise ValueError(f"Expected five fixed folds, found {fold_ids}")

    for fold, split_id in enumerate(fold_ids):
        test_mask = frame["split_id"].eq(split_id).to_numpy()
        test_idx = np.flatnonzero(test_mask)
        train_idx = np.flatnonzero(~test_mask)
        train_df = frame.iloc[train_idx].copy()
        test_df = frame.iloc[test_idx].copy()
        test_group = ";".join(sorted(test_df["state"].unique().tolist()))

        sensitive = base.local_spearman_sensitive(
            train_df,
            feature_cols,
            climate_cols,
            args.spearman_threshold,
        )
        selected = sensitive.loc[sensitive["is_spearman_sensitive"], "target_column"].tolist()
        sensitive = sensitive.copy()
        sensitive.insert(0, "fold", fold)
        sensitive.insert(1, "split_id", split_id)
        sensitive.insert(2, "test_group", test_group)
        selection_rows.append(sensitive)

        fold_rows.append(
            {
                "fold": fold,
                "split_id": split_id,
                "test_group": test_group,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "positive_train": int(y.iloc[train_idx].sum()),
                "positive_test": int(y.iloc[test_idx].sum()),
                "m4_selected_features": len(selected),
            }
        )
        for row_index in test_idx:
            assignment_rows.append(
                {
                    "global_sample_id": frame.iloc[row_index]["global_sample_id"],
                    "sample_id": frame.iloc[row_index]["sample_id"],
                    "state": frame.iloc[row_index]["state"],
                    "Y_label": int(frame.iloc[row_index]["Y_label"]),
                    "fold": fold,
                    "split_id": split_id,
                    "test_group": test_group,
                    "row_index": int(row_index),
                }
            )

        configurations = [
            (
                "M2_A132_RAW_XGB",
                make_recipe(train_df, "M2_A132_RAW_XGB", feature_cols, climate_cols, []),
            ),
            (
                "M3_A132_BROAD_XGB",
                make_recipe(train_df, "M3_A132_BROAD_XGB", feature_cols, climate_cols, feature_cols),
            ),
            (
                "M4_A132_SPEARMAN_XGB",
                make_recipe(train_df, "M4_A132_SPEARMAN_XGB", feature_cols, climate_cols, selected),
            ),
        ]

        for model_key, recipe in configurations:
            train_matrix = recipe.transform(train_df)
            test_matrix = recipe.transform(test_df)
            if train_matrix.shape[1] != 132 or test_matrix.shape[1] != 132:
                raise ValueError(f"Unexpected feature count for {model_key}")
            model = make_xgb_model(y.iloc[train_idx])
            model.fit(train_matrix, y.iloc[train_idx])
            probability = model.predict_proba(test_matrix)[:, 1]
            base.add_metric_row(
                metric_rows,
                prediction_rows,
                dataset="aster_enhanced_a132_1738",
                cv_name="fixed_state_grouped_5fold",
                fold=fold,
                test_group=test_group,
                model_key=model_key,
                used_cols=recipe.output_columns(),
                train_idx=train_idx,
                test_idx=test_idx,
                df=frame,
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
    pooled_metrics = pooled(predictions)
    selection_summary = (
        selections.groupby(["fold", "split_id", "test_group"])["is_spearman_sensitive"]
        .agg(selected="sum", total="count")
        .reset_index()
    )

    metrics.to_csv(output_dir / "aster_enhanced_fold_metrics.csv", index=False)
    predictions.to_csv(output_dir / "aster_enhanced_oof_predictions.csv", index=False)
    selections.to_csv(output_dir / "aster_enhanced_spearman_selection.csv", index=False)
    summary.to_csv(output_dir / "aster_enhanced_summary.csv", index=False)
    pooled_metrics.to_csv(output_dir / "aster_enhanced_pooled_metrics.csv", index=False)
    comparisons.to_csv(output_dir / "aster_enhanced_paired_comparison.csv", index=False)
    fold_manifest.to_csv(output_dir / "aster_enhanced_fold_manifest.csv", index=False)
    assignments.to_csv(output_dir / "aster_enhanced_fold_assignments.csv", index=False)
    pd.DataFrame([{k: v for k, v in audit.items() if k != "input_sha256"}]).to_csv(
        output_dir / "aster_enhanced_data_audit.csv",
        index=False,
    )

    manifest = {
        "experiment": "enhanced ASTER same-input M2/M3/M4 comparison",
        "input_root": str(input_root),
        "main_data": str(Path(args.main_data)),
        "role_table": str(Path(args.role_table)),
        "output_dir": str(output_dir),
        **audit,
        "validation": "five fixed state-grouped folds from fixed_splits_stage1.csv",
        "models": ["M2_A132_RAW_XGB", "M3_A132_BROAD_XGB", "M4_A132_SPEARMAN_XGB"],
        "spearman_threshold": args.spearman_threshold,
        "paired_bootstrap": {"unit": "held-out fold", "repeats": 10000, "seed": 20261006},
        "residualizer": {"model": "Ridge", "alpha": 10.0, "fit_scope": "training fold only"},
        "classifier": "XGBoost",
        "xgboost_parameters": PARAMETERS,
        "feature_boundary": "132 enhanced ASTER A1-A3 features only; no LULC and no climate in classifier",
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_report(
        output_dir,
        summary,
        pooled_metrics,
        comparisons,
        fold_manifest,
        selection_summary,
        audit,
    )
    print(f"Wrote enhanced ASTER M2/M3/M4 outputs to: {output_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enhanced ASTER same-input M2/M3/M4 experiment")
    parser.add_argument("--input-root", default=str(DEFAULT_INPUT_ROOT))
    parser.add_argument("--main-data", default=str(DEFAULT_MAIN_DATA))
    parser.add_argument("--role-table", default=str(DEFAULT_ROLE_TABLE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--spearman-threshold", type=float, default=0.30)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
