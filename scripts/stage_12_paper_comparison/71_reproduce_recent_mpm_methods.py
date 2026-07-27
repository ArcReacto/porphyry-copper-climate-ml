from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import AdaBoostClassifier, HistGradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TARGET = "Y_label"
RANDOM_STATE = 20260622
TOP_K_FRACTIONS = (0.01, 0.05, 0.10, 0.20)

DATASET_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "model_datasets"
    / "by_sample_scheme"
    / "known_mining_neutral"
    / "known_mining_neutral_ratio_1_10"
)
DEFAULT_SUPERVISED_DATASET = DATASET_DIR / "model_dataset_known_mining_neutral_ratio_1_10_supervised_all_features_v1.parquet"
DEFAULT_WITH_NEUTRAL_DATASET = DATASET_DIR / "model_dataset_known_mining_neutral_ratio_1_10_with_neutral_all_features_v1.parquet"
DEFAULT_ROLE_TABLE = (
    PROJECT_ROOT
    / "outputs"
    / "standardized_runs"
    / "known_mining_neutral_ratio_1_10_supervised_all_features_v1"
    / "00_dataset_profile"
    / "feature_roles.csv"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "paper_comparison_methods"
    / "known_mining_neutral_ratio_1_10_supervised_all_features_v1"
)


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


def feature_columns(df: pd.DataFrame, role_table_path: Path | None) -> list[str]:
    if role_table_path and role_table_path.exists():
        role_table = pd.read_csv(role_table_path)
        if "used_in_m1_full_climate" in role_table.columns:
            cols = role_table.loc[role_table["used_in_m1_full_climate"], "column"].tolist()
            return [c for c in cols if c in df.columns]
        if {"column", "is_numeric", "role"}.issubset(role_table.columns):
            excluded_roles = {"metadata", "target", "environment_group", "other_non_numeric"}
            cols = role_table.loc[
                (role_table["is_numeric"]) & (~role_table["role"].isin(excluded_roles)), "column"
            ].tolist()
            return [c for c in cols if c in df.columns]
    return [c for c in df.select_dtypes(include=[np.number]).columns if c != TARGET]


def make_xgb(pos_weight: float, n_estimators: int = 300, paper_params: bool = False) -> XGBClassifier:
    if paper_params:
        return XGBClassifier(
            n_estimators=100,
            learning_rate=0.3,
            max_depth=6,
            min_child_weight=1,
            subsample=1.0,
            colsample_bytree=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=pos_weight,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    return XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=0.03,
        max_depth=3,
        min_child_weight=3,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=2.0,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def make_stacking_model(pos_weight: float) -> StackingClassifier:
    estimators = [
        (
            "rf",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=50,
                            max_depth=10,
                            criterion="gini",
                            class_weight="balanced",
                            random_state=RANDOM_STATE,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
        ),
        (
            "xgb",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("model", make_xgb(pos_weight, n_estimators=100, paper_params=True)),
                ]
            ),
        ),
        (
            "ada",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        AdaBoostClassifier(
                            n_estimators=50,
                            learning_rate=1.0,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
        ),
    ]
    return StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear"),
        cv=3,
        stack_method="predict_proba",
        n_jobs=None,
        passthrough=False,
    )


def make_catboost(pos_weight: float) -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=300,
        learning_rate=0.03,
        depth=8,
        l2_leaf_reg=0.2,
        loss_function="Logloss",
        eval_metric="AUC",
        class_weights=[1.0, pos_weight],
        random_seed=RANDOM_STATE,
        verbose=False,
        allow_writing_files=False,
    )


def make_hist_gradient_boosting(pos_weight: float) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                HistGradientBoostingClassifier(
                    learning_rate=0.05,
                    max_iter=300,
                    max_leaf_nodes=15,
                    l2_regularization=0.1,
                    class_weight={0: 1.0, 1: pos_weight},
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


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


def top_k_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    order = np.argsort(-y_prob)
    ranked_true = y_true[order]
    total_pos = max(int(y_true.sum()), 1)
    base_rate = float(y_true.mean()) if len(y_true) else math.nan
    out: dict[str, float] = {}
    for frac in TOP_K_FRACTIONS:
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


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(int)
    out = {
        "n": int(len(y_true)),
        "n_pos": int(y_true.sum()),
        "n_neg": int((y_true == 0).sum()),
        "roc_auc": safe_auc(y_true, y_prob),
        "average_precision": safe_ap(y_true, y_prob),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)) if len(y_true) else math.nan,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)) if len(y_true) else math.nan,
        "recall": float(recall_score(y_true, y_pred, zero_division=0)) if len(y_true) else math.nan,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)) if len(y_true) else math.nan,
    }
    out.update(top_k_metrics(y_true, y_prob))
    return out


def make_splitters(df: pd.DataFrame, y: pd.Series, cv_mode: str) -> list[tuple[str, object, pd.Series | None]]:
    positives = int(y.sum())
    negatives = int((y == 0).sum())
    n_splits = min(5, positives, negatives)
    if n_splits < 2:
        raise ValueError("Not enough positive/negative samples for cross validation.")
    splitters: list[tuple[str, object, pd.Series | None]] = []
    if cv_mode in {"all", "stratified"}:
        splitters.append(
            ("stratified_kfold", StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE), None)
        )
    if cv_mode in {"all", "group"} and "state" in df.columns and df["state"].nunique(dropna=True) >= 2:
        groups = df["state"].fillna("unknown").astype(str)
        splitters.append(("groupkfold_state", GroupKFold(n_splits=min(5, groups.nunique())), groups))
    return splitters


def summarize_predictions(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_rows = []
    for keys, group in predictions.groupby(["dataset", "cv", "method", "fold"], dropna=False):
        row = dict(zip(["dataset", "cv", "method", "fold"], keys))
        row.update(binary_metrics(group[TARGET].astype(int).to_numpy(), group["y_prob"].astype(float).to_numpy()))
        fold_rows.append(row)
    fold_metrics = pd.DataFrame(fold_rows)

    model_rows = []
    for keys, group in predictions.groupby(["dataset", "cv", "method"], dropna=False):
        row = dict(zip(["dataset", "cv", "method"], keys))
        row.update(binary_metrics(group[TARGET].astype(int).to_numpy(), group["y_prob"].astype(float).to_numpy()))
        model_rows.append(row)
    method_metrics = pd.DataFrame(model_rows)
    return fold_metrics, method_metrics


def neutral_training_pool(with_neutral_df: pd.DataFrame, train_states: set[str] | None) -> pd.DataFrame:
    neutral = with_neutral_df[with_neutral_df[TARGET].eq(-1)].copy()
    if train_states is not None and "state" in neutral.columns:
        neutral = neutral[neutral["state"].fillna("unknown").astype(str).isin(train_states)].copy()
    return neutral


def fit_gmm_pseudo_catboost(
    train_df: pd.DataFrame,
    neutral_pool: pd.DataFrame,
    feature_cols: list[str],
    pos_weight: float,
    max_pseudo_per_class: int | None,
) -> tuple[Pipeline, dict]:
    if neutral_pool.empty:
        model = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", make_catboost(pos_weight))])
        model.fit(train_df[feature_cols], train_df[TARGET].astype(int))
        return model, {"pseudo_positive": 0, "pseudo_negative": 0, "pseudo_total": 0}

    pre = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    combined_x = pd.concat([train_df[feature_cols], neutral_pool[feature_cols]], axis=0)
    z = pre.fit_transform(combined_x)
    n_train = len(train_df)
    z_train = z[:n_train]
    z_neutral = z[n_train:]

    gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=RANDOM_STATE, reg_covar=1e-5)
    gmm.fit(z_neutral)
    train_post = gmm.predict_proba(z_train)
    y_train = train_df[TARGET].astype(int).to_numpy()
    component_pos_rate: list[float] = []
    for component in range(2):
        weights = train_post[:, component]
        denom = max(float(weights.sum()), 1e-12)
        component_pos_rate.append(float(np.sum(weights * y_train) / denom))
    component_pos_rate = np.asarray(component_pos_rate)
    neutral_post = gmm.predict_proba(z_neutral)
    positive_component = int(np.argmax(component_pos_rate))
    neutral_component = np.argmax(neutral_post, axis=1)
    neutral_confidence = np.max(neutral_post, axis=1)
    pseudo_y_all = (neutral_component == positive_component).astype(int)

    selected_indices: list[int] = []
    rng = np.random.default_rng(RANDOM_STATE)
    for label in [0, 1]:
        label_indices = np.where(pseudo_y_all == label)[0]
        if len(label_indices) == 0:
            continue
        order = label_indices[np.argsort(-neutral_confidence[label_indices])]
        if max_pseudo_per_class is not None:
            order = order[:max_pseudo_per_class]
        rng.shuffle(order)
        selected_indices.extend(order.tolist())
    selected_indices = sorted(selected_indices)
    selected = neutral_pool.iloc[selected_indices].copy()
    pseudo_y = pseudo_y_all[selected_indices]
    pseudo_weight = np.full(len(selected), 0.5)

    augmented_x = pd.concat([train_df[feature_cols], selected[feature_cols]], axis=0)
    augmented_y = np.concatenate([y_train, pseudo_y])
    sample_weight = np.concatenate([np.ones(len(train_df)), pseudo_weight])

    model = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", make_catboost(pos_weight))])
    model.fit(augmented_x, augmented_y, model__sample_weight=sample_weight)
    info = {
        "pseudo_positive": int(np.sum(pseudo_y == 1)),
        "pseudo_negative": int(np.sum(pseudo_y == 0)),
        "pseudo_total": int(len(selected)),
        "neutral_pool": int(len(neutral_pool)),
        "component_pos_rate_0": float(component_pos_rate[0]),
        "component_pos_rate_1": float(component_pos_rate[1]),
        "positive_component": positive_component,
        "mean_pseudo_confidence": float(np.mean(neutral_confidence[selected_indices])) if selected_indices else math.nan,
    }
    return model, info


def compact_markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No data._"
    show = df.copy()
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


def write_report(output_dir: Path, method_metrics: pd.DataFrame, manifest: dict) -> None:
    metric_cols = [
        "dataset",
        "cv",
        "method",
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
    display = method_metrics[[c for c in metric_cols if c in method_metrics.columns]].sort_values(
        ["cv", "top05_f1"], ascending=[True, False]
    )
    report = [
        "# Recent MPM Method Reproduction Results",
        "",
        "## Experiment Setting",
        "",
        f"- Supervised dataset: `{manifest['supervised_dataset']}`",
        f"- With-neutral dataset: `{manifest['with_neutral_dataset']}`",
        "- Evaluation samples: positive and negative samples only.",
        "- Feature set: full numeric M1 feature set; no climate residualization or graph-guided processing.",
        f"- Feature count: `{manifest['feature_count']}`",
        f"- CV mode: `{manifest['cv_mode']}`",
        "",
        "## Reproduced Method Analogues",
        "",
        "| Method | Paper-side idea | Current implementation |",
        "|---|---|---|",
        "| `paper_stacking_ensemble` | Interpretable/ensemble MPM using multiple learners | RF + XGBoost + AdaBoost stacking with LR meta learner, using paper-side key hyperparameters |",
        "| `supervised_catboost` | Supervised CatBoost MPM benchmark | CatBoost with paper-side hyperparameters trained on positive/negative samples |",
        "| `gmm_pseudo_catboost` | GMM + CatBoost semi-supervised MPM | GMM pseudo-labels neutral samples, then trains weighted CatBoost |",
        "",
        "## Results",
        "",
        compact_markdown_table(display),
        "",
        "## Notes",
        "",
        "- These are adapted reproductions on the current point-based MPM table, not pixel-buffer reproductions from the original study areas.",
        "- The CatBoost parameters follow the CatBoost + GMM paper: iterations=300, depth=8, l2_leaf_reg=0.2, learning_rate=0.03.",
        "- `groupkfold_state` should be treated as the stricter spatial generalization result.",
    ]
    (output_dir / "recent_mpm_method_reproduction_report.md").write_text("\n".join(report), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce selected recent MPM comparison methods on the current dataset.")
    parser.add_argument("--supervised-dataset", default=str(DEFAULT_SUPERVISED_DATASET))
    parser.add_argument("--with-neutral-dataset", default=str(DEFAULT_WITH_NEUTRAL_DATASET))
    parser.add_argument("--role-table", default=str(DEFAULT_ROLE_TABLE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--cv-mode", choices=["all", "group", "stratified"], default="all")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["paper_stacking_ensemble", "supervised_catboost", "gmm_pseudo_catboost"],
    )
    parser.add_argument(
        "--max-pseudo-per-class",
        type=int,
        default=800,
        help="Maximum neutral pseudo-labeled samples per GMM class per fold. Use -1 for all.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    supervised_path = Path(args.supervised_dataset)
    with_neutral_path = Path(args.with_neutral_dataset)
    role_table_path = Path(args.role_table) if args.role_table else None

    df = read_table(supervised_path)
    df = df[df[TARGET].isin([0, 1])].reset_index(drop=True)
    with_neutral_df = read_table(with_neutral_path)
    dataset_name = supervised_path.stem.replace("model_dataset_", "")
    feature_cols = feature_columns(df, role_table_path)
    y = df[TARGET].astype(int).reset_index(drop=True)
    pos_weight = int((y == 0).sum()) / max(int(y.sum()), 1)

    pred_parts: list[pd.DataFrame] = []
    pseudo_rows: list[dict] = []

    for cv_name, splitter, groups in make_splitters(df, y, args.cv_mode):
        split_iter = splitter.split(df[feature_cols], y, groups) if groups is not None else splitter.split(df[feature_cols], y)
        for fold, (train_idx, test_idx) in enumerate(split_iter, start=1):
            train_idx = np.asarray(train_idx)
            test_idx = np.asarray(test_idx)
            train_df = df.iloc[train_idx].copy()
            test_df = df.iloc[test_idx].copy()
            x_test = test_df[feature_cols]
            y_test = test_df[TARGET].astype(int)
            test_group = ""
            train_states = None
            if groups is not None:
                group_series = pd.Series(groups)
                test_group = ";".join(sorted(group_series.iloc[test_idx].astype(str).unique().tolist()))
                train_states = set(group_series.iloc[train_idx].astype(str).unique().tolist())

            if "paper_stacking_ensemble" in args.methods or "stacking_ensemble" in args.methods:
                model = make_stacking_model(pos_weight)
                model.fit(train_df[feature_cols], train_df[TARGET].astype(int))
                pred_parts.append(
                    pd.DataFrame(
                        {
                            "dataset": dataset_name,
                            "cv": cv_name,
                            "fold": fold,
                            "test_group": test_group,
                            "method": "paper_stacking_ensemble",
                            "row_index": test_idx,
                            TARGET: y_test.to_numpy(),
                            "y_prob": model.predict_proba(x_test)[:, 1],
                            "sample_id": test_df.get("sample_id", pd.Series([""] * len(test_df))).to_numpy(),
                            "state": test_df.get("state", pd.Series([""] * len(test_df))).to_numpy(),
                        }
                    )
                )

            if "supervised_catboost" in args.methods:
                model = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", make_catboost(pos_weight))])
                model.fit(train_df[feature_cols], train_df[TARGET].astype(int))
                pred_parts.append(
                    pd.DataFrame(
                        {
                            "dataset": dataset_name,
                            "cv": cv_name,
                            "fold": fold,
                            "test_group": test_group,
                            "method": "supervised_catboost",
                            "row_index": test_idx,
                            TARGET: y_test.to_numpy(),
                            "y_prob": model.predict_proba(x_test)[:, 1],
                            "sample_id": test_df.get("sample_id", pd.Series([""] * len(test_df))).to_numpy(),
                            "state": test_df.get("state", pd.Series([""] * len(test_df))).to_numpy(),
                        }
                    )
                )

            if "gmm_pseudo_catboost" in args.methods:
                neutral_pool = neutral_training_pool(with_neutral_df, train_states)
                max_pseudo = None if args.max_pseudo_per_class < 0 else args.max_pseudo_per_class
                model, pseudo_info = fit_gmm_pseudo_catboost(
                    train_df,
                    neutral_pool,
                    feature_cols,
                    pos_weight=pos_weight,
                    max_pseudo_per_class=max_pseudo,
                )
                pseudo_info.update({"dataset": dataset_name, "cv": cv_name, "fold": fold, "test_group": test_group})
                pseudo_rows.append(pseudo_info)
                pred_parts.append(
                    pd.DataFrame(
                        {
                            "dataset": dataset_name,
                            "cv": cv_name,
                            "fold": fold,
                            "test_group": test_group,
                            "method": "gmm_pseudo_catboost",
                            "row_index": test_idx,
                            TARGET: y_test.to_numpy(),
                            "y_prob": model.predict_proba(x_test)[:, 1],
                            "sample_id": test_df.get("sample_id", pd.Series([""] * len(test_df))).to_numpy(),
                            "state": test_df.get("state", pd.Series([""] * len(test_df))).to_numpy(),
                        }
                    )
                )

    predictions = pd.concat(pred_parts, ignore_index=True)
    fold_metrics, method_metrics = summarize_predictions(predictions)
    pseudo_summary = pd.DataFrame(pseudo_rows)

    write_table(predictions, output_dir / "recent_mpm_method_predictions.csv")
    write_table(fold_metrics, output_dir / "recent_mpm_method_fold_metrics.csv")
    write_table(method_metrics, output_dir / "recent_mpm_method_metrics.csv")
    write_table(pseudo_summary, output_dir / "recent_mpm_method_gmm_pseudo_summary.csv")

    manifest = {
        "supervised_dataset": str(supervised_path),
        "with_neutral_dataset": str(with_neutral_path),
        "role_table": str(role_table_path) if role_table_path else "",
        "output_dir": str(output_dir),
        "dataset": dataset_name,
        "feature_count": len(feature_cols),
        "methods": args.methods,
        "cv_mode": args.cv_mode,
        "samples_evaluated": "positive_negative_only",
        "neutral_used_by_gmm_pseudo_catboost": "training_only_as_unlabeled_pseudo_labeled_samples",
        "max_pseudo_per_class": args.max_pseudo_per_class,
        "random_state": RANDOM_STATE,
    }
    (output_dir / "recent_mpm_method_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(output_dir, method_metrics, manifest)
    print(f"Wrote recent MPM method comparison results to: {output_dir}")
    print((output_dir / "recent_mpm_method_reproduction_report.md").resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
