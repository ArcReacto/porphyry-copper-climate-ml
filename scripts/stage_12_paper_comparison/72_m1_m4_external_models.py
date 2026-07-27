from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = PROJECT_ROOT / "outputs" / "standardized_runs" / "known_mining_neutral_ratio_1_10_supervised_all_features_v1"
OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "paper_comparison_methods"
    / "m1_m4_external_models"
    / "known_mining_neutral_ratio_1_10_supervised_all_features_v1"
)
BASE_SCRIPT = PROJECT_ROOT / "scripts" / "stage_11_paper_optimization" / "68_m1_m4_histgb_xgb_experiment.py"
METHOD_SCRIPT = PROJECT_ROOT / "scripts" / "stage_12_paper_comparison" / "71_reproduce_recent_mpm_methods.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


base = load_module(BASE_SCRIPT, "m1_m4_base")
methods = load_module(METHOD_SCRIPT, "recent_mpm_methods")


CLASSIFIER_FAMILIES = ["paper_stacking_ensemble", "supervised_catboost"]


def make_external_classifier(family: str, pos_weight: float):
    if family == "paper_stacking_ensemble":
        return methods.make_stacking_model(pos_weight)
    if family == "supervised_catboost":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", methods.make_catboost(pos_weight)),
            ]
        )
    raise ValueError(family)


def write_report(output_dir: Path, dataset_name: str, summary: pd.DataFrame, validation_mode: str) -> None:
    cols = [
        "cv",
        "model",
        "classifier_family",
        "roc_auc_mean",
        "average_precision_mean",
        "balanced_accuracy_mean",
        "f1_mean",
        "top05_precision_mean",
        "top05_recall_mean",
        "top05_f1_mean",
        "top05_lift_mean",
        "top05_ndcg_mean",
        "top10_precision_mean",
        "top10_recall_mean",
        "top10_f1_mean",
        "top10_lift_mean",
        "top10_ndcg_mean",
    ]
    display = summary[[c for c in cols if c in summary.columns]].sort_values(
        ["cv", "classifier_family", "top05_f1_mean"], ascending=[True, True, False]
    )
    group = display[display["cv"].eq("groupkfold_state")].copy()
    report = [
        "# M1-M4 on External MPM Models",
        "",
        "## Experiment Setting",
        "",
        f"- Dataset: `{dataset_name}`",
        "- Samples: positive + negative only; neutral samples are not used.",
        "- Feature workflow: latest standard M1-M4 workflow.",
        "- M1: full features including climate.",
        "- M2: no climate features.",
        "- M3: all geochemistry residualized against climate.",
        "- M4: fold-local climate-sensitive geochemistry residualized against climate.",
        f"- Validation mode: `{validation_mode}`.",
        "",
        "## External Models",
        "",
        "| Classifier | Source idea |",
        "|---|---|",
        "| `paper_stacking_ensemble` | RF + XGBoost + AdaBoost as primary learners, Logistic Regression as meta learner. |",
        "| `supervised_catboost` | CatBoost with paper-side parameters: iterations=300, depth=8, l2_leaf_reg=0.2, learning_rate=0.03. |",
        "",
        "## GroupKFold Main Results",
        "",
        base.markdown_table(group, max_rows=100),
        "",
        "## All Results",
        "",
        base.markdown_table(display, max_rows=200),
        "",
        "## Notes",
        "",
        "- This experiment tests whether the M1-M4 climate-decoupling workflow transfers to external MPM model families.",
        "- `groupkfold_state` remains the strict spatial generalization setting.",
    ]
    (output_dir / "m1_m4_external_models_report.md").write_text("\n".join(report), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run latest M1-M4 workflow with external MPM comparison models.")
    parser.add_argument("--run-dir", default=str(RUN_DIR))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--m4-threshold", type=float, default=base.DEFAULT_M4_THRESHOLD)
    parser.add_argument(
        "--validation-mode",
        choices=["group", "stratified", "all"],
        default="group",
        help="Default is group for faster, stricter spatial comparison.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_name = run_dir.name
    df, dataset_path = base.load_dataset_from_run(run_dir)
    df = df[df[base.TARGET].isin([0, 1])].reset_index(drop=True)
    role_table = pd.read_csv(run_dir / "00_dataset_profile" / "feature_roles.csv")
    y = df[base.TARGET].astype(int).reset_index(drop=True)
    pos_weight = int((y == 0).sum()) / max(int(y.sum()), 1)

    metrics_rows: list[dict] = []
    prediction_rows: list[dict] = []
    sensitive_rows: list[pd.DataFrame] = []

    for cv_name, splitter, groups in base.make_splitters(df, args.validation_mode):
        split_iter = splitter.split(df, y, groups) if groups is not None else splitter.split(df, y)
        for fold, (train_idx, test_idx) in enumerate(split_iter, start=1):
            train_idx = np.asarray(train_idx)
            test_idx = np.asarray(test_idx)
            test_group = None
            if groups is not None:
                test_group = ";".join(sorted(pd.Series(groups).iloc[test_idx].astype(str).unique().tolist()))

            for model_key in base.MODEL_KEYS:
                x_train, x_test, used_columns, sensitive = base.build_decoupled_features(
                    df=df,
                    role_table=role_table,
                    train_idx=train_idx,
                    test_idx=test_idx,
                    model_key=model_key,
                    m4_threshold=args.m4_threshold,
                )
                if model_key == "M4_Sensitive_Residualized" and not sensitive.empty:
                    temp = sensitive.copy()
                    temp.insert(0, "dataset", dataset_name)
                    temp.insert(1, "cv", cv_name)
                    temp.insert(2, "fold", fold)
                    temp.insert(3, "test_group", test_group)
                    sensitive_rows.append(temp)

                for family in CLASSIFIER_FAMILIES:
                    clf = make_external_classifier(family, pos_weight)
                    clf.fit(x_train, y.iloc[train_idx])
                    y_prob = clf.predict_proba(x_test)[:, 1]
                    base.add_metric_row(
                        metrics_rows,
                        prediction_rows,
                        dataset_name=dataset_name,
                        cv_name=cv_name,
                        fold=fold,
                        test_group=test_group,
                        model_key=model_key,
                        classifier_family=family,
                        n_features=len(used_columns),
                        train_idx=train_idx,
                        test_idx=test_idx,
                        df=df,
                        y_test=y.iloc[test_idx],
                        y_prob=y_prob,
                    )

    metrics = pd.DataFrame(metrics_rows)
    predictions = pd.DataFrame(prediction_rows)
    sensitive = pd.concat(sensitive_rows, ignore_index=True) if sensitive_rows else pd.DataFrame()
    summary = base.summarize_metrics(metrics)

    base.write_table(metrics, output_dir / "m1_m4_external_models_fold_metrics.csv")
    base.write_table(predictions, output_dir / "m1_m4_external_models_predictions.csv")
    base.write_table(summary, output_dir / "m1_m4_external_models_summary.csv")
    base.write_table(sensitive, output_dir / "m1_m4_external_models_m4_sensitive_features.csv")
    manifest = {
        "dataset": dataset_name,
        "dataset_path": str(dataset_path),
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "samples_used": "positive_negative_only",
        "neutral_samples_used": False,
        "m4_threshold": args.m4_threshold,
        "validation_mode": args.validation_mode,
        "m1_m4_models": base.MODEL_KEYS,
        "external_classifiers": CLASSIFIER_FAMILIES,
    }
    (output_dir / "m1_m4_external_models_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(output_dir, dataset_name, summary, args.validation_mode)

    print(f"Wrote M1-M4 external model results to: {output_dir}")
    print((output_dir / "m1_m4_external_models_report.md").resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
