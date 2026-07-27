"""
38_branch_fusion_negative_ratio_experiment.py

目标：
对已构建的数据集运行“成矿过程分支融合”实验：
- 按 Source / Transport / Deposition / Preservation 四类概念分支训练单分支模型；
- 比较完整特征、去气候特征、早期融合、后期 stacking、气候残差化融合；
- 可一次运行多个负样本比例数据集，并生成跨数据集汇总报告。

输入：
- outputs/model_datasets/by_sample_scheme/**/model_dataset_*_all_features_v1.csv
- config/feature_branches/feature_branch_assignment.json

输出：
- outputs/branch_fusion_experiments/<dataset_name>/*.csv
- outputs/branch_fusion_experiments/branch_fusion_summary.csv
- outputs/branch_fusion_experiments/branch_fusion_report.md
"""

import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BRANCH_JSON = PROJECT_ROOT / "config" / "feature_branches" / "feature_branch_assignment.json"
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "outputs" / "model_datasets" / "by_sample_scheme"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "branch_fusion_experiments"

RANDOM_STATE = 2026
N_ESTIMATORS = 500
MIN_SAMPLES_LEAF = 5
MAX_DEPTH = 10
N_SPLITS = 5
RESIDUAL_R_THRESHOLD = 0.2
POSITIVE_LABEL = 1
BASE_MODEL = "rf"

CLIMATE_PREFIXES = [
    "climate_", "env_climate_", "env_aridity", "env_water_deficit",
    "env_snow_influence", "env_relief_class", "env_weathering_regime", "env_causal_group",
]
CLIMATE_CORE_COLS = [
    "env_aridity_index_ppt_pet",
    "env_climate_ppt_annual_mm",
    "env_climate_pet_annual_mm",
    "env_climate_deficit_annual_mm",
    "env_climate_tmean_annual_c",
]
META_COLS = [
    "sample_id", "Y_label", "sample_type", "negative_type",
    "state", "latitude", "longitude", "dep_id", "mrds_id",
    "site_name", "env_causal_group",
]


def is_climate_col(col):
    return any(col.startswith(p) for p in CLIMATE_PREFIXES) or col in CLIMATE_CORE_COLS


def get_aridity_zone(ai):
    if pd.isna(ai):
        return "unknown"
    if ai < 0.2:
        return "hyper_arid"
    if ai < 0.5:
        return "arid"
    if ai < 0.65:
        return "semi_arid"
    if ai < 1.0:
        return "dry_subhumid"
    return "humid"


def build_base_model(class_weight=None):
    if BASE_MODEL == "rf":
        return RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            max_features="sqrt",
            bootstrap=True,
            oob_score=True,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight=class_weight,
        )
    if BASE_MODEL == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            max_iter=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            learning_rate=0.05,
            random_state=RANDOM_STATE,
        )
    if BASE_MODEL == "logistic_regression":
        return LogisticRegression(
            max_iter=3000,
            random_state=RANDOM_STATE,
            class_weight=class_weight,
            n_jobs=-1,
        )
    raise ValueError(f"Unsupported base model: {BASE_MODEL}")


def base_model_display_name():
    return {
        "rf": f"RandomForestClassifier(n_estimators={N_ESTIMATORS}, max_depth={MAX_DEPTH}, min_samples_leaf={MIN_SAMPLES_LEAF})",
        "hist_gradient_boosting": f"HistGradientBoostingClassifier(max_iter={N_ESTIMATORS}, max_depth={MAX_DEPTH}, min_samples_leaf={MIN_SAMPLES_LEAF})",
        "logistic_regression": "LogisticRegression(max_iter=3000)",
    }[BASE_MODEL]


def read_dataset(path):
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def unique_keep_order(values):
    seen = set()
    out = []
    for value in values:
        if value not in seen:
            out.append(value)
            seen.add(value)
    return out


def format_cell(value):
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value).replace("|", "/")


def to_markdown_table(df, include_index=False):
    table = df.reset_index() if include_index else df.copy()
    table = table.astype(object)
    headers = [str(c) for c in table.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(format_cell(row[c]) for c in table.columns) + " |")
    return "\n".join(lines)


def positive_binary_labels(y):
    return (np.asarray(y) == POSITIVE_LABEL).astype(int)


def predict_positive_proba(model_or_pipeline, X):
    proba = model_or_pipeline.predict_proba(X)
    model = (
        model_or_pipeline.named_steps["model"]
        if hasattr(model_or_pipeline, "named_steps") and "model" in model_or_pipeline.named_steps
        else model_or_pipeline
    )
    classes = list(model.classes_)
    if POSITIVE_LABEL not in classes:
        return np.zeros(len(X), dtype=float)
    return proba[:, classes.index(POSITIVE_LABEL)]


def make_pipeline(features, model):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
        ("model", model),
    ])


def evaluate(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    has_two_classes = len(np.unique(y_true)) > 1
    return {
        "roc_auc": roc_auc_score(y_true, y_prob) if has_two_classes else np.nan,
        "average_precision": average_precision_score(y_true, y_prob) if has_two_classes else np.nan,
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def cv_model(X, y, groups, features, model_type=None, class_weight=None):
    if len(features) == 0:
        raise ValueError("No usable features supplied to cv_model.")
    model = build_base_model(class_weight) if model_type is None else model_type
    pipe = make_pipeline(features, model)
    y_eval = positive_binary_labels(y)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof_prob_skf = np.zeros(len(y))
    skf_metrics = []
    for tr_idx, val_idx in skf.split(X[features], y):
        pipe.fit(X.iloc[tr_idx][features], y.iloc[tr_idx])
        prob = predict_positive_proba(pipe, X.iloc[val_idx][features])
        oof_prob_skf[val_idx] = prob
        skf_metrics.append(evaluate(y_eval[val_idx], prob))

    gkf = GroupKFold(n_splits=N_SPLITS)
    oof_prob_gkf = np.zeros(len(y))
    gkf_metrics = []
    for tr_idx, val_idx in gkf.split(X[features], y, groups):
        pipe.fit(X.iloc[tr_idx][features], y.iloc[tr_idx])
        prob = predict_positive_proba(pipe, X.iloc[val_idx][features])
        oof_prob_gkf[val_idx] = prob
        gkf_metrics.append(evaluate(y_eval[val_idx], prob))

    def avg_metrics(m_list):
        df = pd.DataFrame(m_list)
        return df.mean().to_dict(), df.std().to_dict()

    skf_mean, skf_std = avg_metrics(skf_metrics)
    gkf_mean, gkf_std = avg_metrics(gkf_metrics)

    return {
        "skf_mean": skf_mean,
        "skf_std": skf_std,
        "gkf_mean": gkf_mean,
        "gkf_std": gkf_std,
        "oof_prob_skf": oof_prob_skf,
        "oof_prob_gkf": oof_prob_gkf,
    }


def compute_climate_residuals_both(train_df, test_df, element_cols, climate_cols, r_threshold=0.2):
    if not climate_cols:
        return train_df.copy(), test_df.copy()

    train = train_df.copy()
    test = test_df.copy()

    sensitive = {}
    for elem in element_cols:
        cols = [elem] + climate_cols
        corr = train[cols].corr(method="spearman")[elem].drop(elem).abs().max()
        if pd.isna(corr):
            corr = 0.0
        sensitive[elem] = corr

    imp = SimpleImputer(strategy="median")
    Xc_train = imp.fit_transform(train[climate_cols])
    Xc_test = imp.transform(test[climate_cols])

    for elem in element_cols:
        if sensitive[elem] < r_threshold:
            continue
        y_train = train[elem].fillna(train[elem].median()).values
        model = Ridge(alpha=1.0)
        model.fit(Xc_train, y_train)
        pred_train = model.predict(Xc_train)
        pred_test = model.predict(Xc_test)
        train[f"{elem}_resid"] = train[elem].values - pred_train
        test[f"{elem}_resid"] = test[elem].values - pred_test

    return train, test


def fit_and_importance(X, y, features, class_weight=None):
    pipe = make_pipeline(features, build_base_model(class_weight))
    pipe.fit(X[features], y)
    model = pipe.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        scores = model.feature_importances_
        score_name = "importance"
    elif hasattr(model, "coef_"):
        scores = np.abs(model.coef_).max(axis=0) if model.coef_.ndim > 1 else np.abs(model.coef_)
        score_name = "abs_coefficient"
    else:
        return pd.DataFrame(columns=["feature", "importance", "importance_type"])

    feature_names = features if len(scores) == len(features) else [f"feature_{i}" for i in range(len(scores))]
    imp = pd.DataFrame({
        "feature": feature_names,
        "importance": scores,
        "importance_type": score_name,
    }).sort_values("importance", ascending=False)
    return imp


def state_zone_metrics(df, prob_col, label_col="Y_label", state_col="state"):
    df = df.copy()
    df["pred"] = (df[prob_col] >= 0.5).astype(int)
    df["_positive_label"] = positive_binary_labels(df[label_col])
    if "zone" not in df.columns:
        df["zone"] = df["env_aridity_index_ppt_pet"].apply(get_aridity_zone)

    def group_perf(g, name_col):
        y_true = g["_positive_label"].values
        if y_true.sum() == 0 or y_true.sum() == len(g):
            return None
        return {
            name_col: g[name_col].iloc[0],
            "n_total": len(g),
            "n_pos": int(y_true.sum()),
            "n_neg": int((g[label_col] == 0).sum()),
            "n_neutral": int((g[label_col] == -1).sum()),
            "recall": recall_score(y_true, g["pred"], zero_division=0),
            "precision": precision_score(y_true, g["pred"], zero_division=0),
            "f1": f1_score(y_true, g["pred"], zero_division=0),
            "auc": roc_auc_score(y_true, g[prob_col]) if len(set(y_true)) > 1 else np.nan,
        }

    state_list = [group_perf(g, state_col) for _, g in df.groupby(state_col)]
    zone_list = [group_perf(g, "zone") for _, g in df.groupby("zone")]
    return pd.DataFrame([x for x in state_list if x is not None]), pd.DataFrame([x for x in zone_list if x is not None])


def run_one_ratio(data_path, out_dir, branch_info, include_neutral=False):
    data_path = Path(data_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = read_dataset(data_path)
    if "Y_label" not in df.columns:
        raise ValueError(f"{data_path} does not contain Y_label.")
    if "state" not in df.columns:
        raise ValueError(f"{data_path} does not contain state.")

    before_rows = len(df)
    allowed_labels = [-1, 0, 1] if include_neutral else [0, 1]
    df = df[df["Y_label"].isin(allowed_labels)].copy()
    if len(df) < before_rows:
        print(f"  Filtered rows outside labels {allowed_labels}: {before_rows} -> {len(df)}")
    df["Y_label"] = df["Y_label"].astype(int)

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    numeric_set = set(numeric_cols)

    branches = branch_info["branch_features"]
    required_branches = ["Source", "Transport", "Deposition", "Preservation"]
    missing_branches = [b for b in required_branches if b not in branches]
    if missing_branches:
        raise ValueError(f"Branch JSON missing branches: {missing_branches}")
    branch_sets = {
        k: unique_keep_order([c for c in v if c in df.columns and c in numeric_set])
        for k, v in branches.items()
    }

    y = df["Y_label"]
    y_eval = positive_binary_labels(y)
    groups = df["state"].values
    if "env_aridity_index_ppt_pet" in df.columns:
        df["zone"] = df["env_aridity_index_ppt_pet"].apply(get_aridity_zone)
    else:
        df["zone"] = "unknown"

    all_feature_cols = unique_keep_order([c for c in numeric_cols if c not in META_COLS and c != "zone"])
    full_features = all_feature_cols
    no_climate_features = [c for c in full_features if not is_climate_col(c)]

    deposition_features = branch_sets["Deposition"]
    element_cols = [c for c in deposition_features if df[c].dtype.kind in "fi"]
    climate_core_for_residual = [c for c in CLIMATE_CORE_COLS if c in df.columns]

    n_pos = int(y.sum())
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    n_neutral = int((y == -1).sum())
    ratio_str = f"1:{n_neg // n_pos}" if n_pos > 0 else "unknown"
    if n_neutral:
        ratio_str = f"{ratio_str}; neutral={n_neutral}"

    print(f"\n{'='*70}")
    print(f"Dataset: {data_path.name}")
    print(f"  samples={len(df)}, pos={n_pos}, neg={n_neg}, neutral={n_neutral}, ratio={ratio_str}")
    print(f"  training_mode={'three-class with neutral' if include_neutral else 'binary supervised'}")
    print(f"  features={len(full_features)}, deposition={len(deposition_features)}, elements={len(element_cols)}")
    for branch in required_branches:
        print(f"  branch {branch}: {len(branch_sets[branch])} usable numeric features")

    experiments = {}
    experiments["M1_Full_RF"] = {
        "features": full_features,
        "residualize": False,
        "desc": "全部原始特征，包括气候变量。",
    }
    experiments["M2_NoClimate_RF"] = {
        "features": no_climate_features,
        "residualize": False,
        "desc": "剔除所有气候变量后的基线。",
    }
    for branch in ["Source", "Transport", "Deposition", "Preservation"]:
        experiments[f"M{3 + ['Source', 'Transport', 'Deposition', 'Preservation'].index(branch)}_{branch}_Only"] = {
            "features": branch_sets[branch],
            "residualize": False,
            "desc": f"仅使用 {branch} 分支特征。",
        }
    fusion_features = unique_keep_order(
        branch_sets["Source"] + branch_sets["Transport"] + branch_sets["Deposition"] + branch_sets["Preservation"]
    )
    experiments["M7_Early_Fusion"] = {
        "features": fusion_features,
        "residualize": False,
        "desc": "Source + Transport + Deposition + Preservation 四分支特征拼接。",
    }
    no_climate_fusion_features = [
        c for c in fusion_features if c in no_climate_features
    ]
    experiments["M11_NoClimate_EarlyFusion"] = {
        "features": no_climate_fusion_features,
        "residualize": False,
        "desc": "非气候版 Source + Transport + Deposition + Preservation 四分支特征拼接。",
    }
    experiments["M9_ClimateResidual_EarlyFusion"] = {
        "features": ["RESID_FUSED"],
        "residualize": True,
        "desc": "Deposition 元素做气候残差化，Source + Transport 保留，不输入任何原始气候变量。",
    }

    results_summary = []
    oof_probabilities = pd.DataFrame({
        "sample_id": df["sample_id"].values,
        "Y_label": y.values,
        "state": df["state"].values,
        "zone": df["zone"].values,
    })
    if "env_aridity_index_ppt_pet" in df.columns:
        oof_probabilities["env_aridity_index_ppt_pet"] = df["env_aridity_index_ppt_pet"].values

    for exp_name, cfg in experiments.items():
        print(f"  Running {exp_name} ...")

        if cfg.get("residualize"):
            skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
            oof_prob = np.zeros(len(df))
            fold_metrics = []

            for tr_idx, val_idx in skf.split(df, y):
                train_fold = df.iloc[tr_idx].copy()
                val_fold = df.iloc[val_idx].copy()
                train_resid, val_resid = compute_climate_residuals_both(
                    train_fold, val_fold, element_cols, climate_core_for_residual,
                    r_threshold=RESIDUAL_R_THRESHOLD,
                )
                resid_cols = [c for c in train_resid.columns if c.endswith("_resid")]
                feat = unique_keep_order(branch_sets["Source"] + branch_sets["Transport"] + resid_cols)
                feat = [c for c in feat if c in train_resid.columns]
                if len(feat) == 0:
                    raise ValueError(f"No residualized features generated for {exp_name}")
                pipe = make_pipeline(feat, build_base_model())
                pipe.fit(train_resid[feat], y.iloc[tr_idx])
                prob = predict_positive_proba(pipe, val_resid[feat])
                oof_prob[val_idx] = prob
                fold_metrics.append(evaluate(y_eval[val_idx], prob))

            metrics_mean = pd.DataFrame(fold_metrics).mean().to_dict()
            metrics_std = pd.DataFrame(fold_metrics).std().to_dict()

            gkf = GroupKFold(n_splits=N_SPLITS)
            oof_prob_gkf = np.zeros(len(df))
            gkf_metrics = []
            for tr_idx, val_idx in gkf.split(df, y, groups):
                train_fold = df.iloc[tr_idx].copy()
                val_fold = df.iloc[val_idx].copy()
                train_resid, val_resid = compute_climate_residuals_both(
                    train_fold, val_fold, element_cols, climate_core_for_residual,
                    r_threshold=RESIDUAL_R_THRESHOLD,
                )
                resid_cols = [c for c in train_resid.columns if c.endswith("_resid")]
                feat = unique_keep_order(branch_sets["Source"] + branch_sets["Transport"] + resid_cols)
                feat = [c for c in feat if c in train_resid.columns]
                pipe = make_pipeline(feat, build_base_model())
                pipe.fit(train_resid[feat], y.iloc[tr_idx])
                prob = predict_positive_proba(pipe, val_resid[feat])
                oof_prob_gkf[val_idx] = prob
                gkf_metrics.append(evaluate(y_eval[val_idx], prob))
            gkf_mean = pd.DataFrame(gkf_metrics).mean().to_dict()
            gkf_std = pd.DataFrame(gkf_metrics).std().to_dict()

            results_summary.append({
                "experiment": exp_name,
                "n_features": len(feat),
                "skf_auc_mean": metrics_mean["roc_auc"],
                "skf_auc_std": metrics_std["roc_auc"],
                "skf_f1_mean": metrics_mean["f1"],
                "skf_f1_std": metrics_std["f1"],
                "gkf_auc_mean": gkf_mean["roc_auc"],
                "gkf_auc_std": gkf_std["roc_auc"],
                "gkf_f1_mean": gkf_mean["f1"],
                "gkf_f1_std": gkf_std["f1"],
            })
            oof_probabilities[f"{exp_name}_skf_prob"] = oof_prob
            oof_probabilities[f"{exp_name}_gkf_prob"] = oof_prob_gkf

        else:
            feats = cfg["features"]
            res = cv_model(df, y, groups, feats, class_weight=None)
            results_summary.append({
                "experiment": exp_name,
                "n_features": len(feats),
                "skf_auc_mean": res["skf_mean"]["roc_auc"],
                "skf_auc_std": res["skf_std"]["roc_auc"],
                "skf_f1_mean": res["skf_mean"]["f1"],
                "skf_f1_std": res["skf_std"]["f1"],
                "gkf_auc_mean": res["gkf_mean"]["roc_auc"],
                "gkf_auc_std": res["gkf_std"]["roc_auc"],
                "gkf_f1_mean": res["gkf_mean"]["f1"],
                "gkf_f1_std": res["gkf_std"]["f1"],
            })
            oof_probabilities[f"{exp_name}_skf_prob"] = res["oof_prob_skf"]
            oof_probabilities[f"{exp_name}_gkf_prob"] = res["oof_prob_gkf"]

    # M8 stacking
    print("  Running M8_Late_Fusion_Stacking ...")
    branch_oof_cols_skf = [
        f"M{3 + i}_{b}_Only_skf_prob"
        for i, b in enumerate(["Source", "Transport", "Deposition", "Preservation"])
    ]
    branch_oof_cols_gkf = [
        f"M{3 + i}_{b}_Only_gkf_prob"
        for i, b in enumerate(["Source", "Transport", "Deposition", "Preservation"])
    ]
    X_stack = oof_probabilities[branch_oof_cols_skf].values
    y_stack = oof_probabilities["Y_label"].values
    y_stack_eval = positive_binary_labels(y_stack)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    stack_oof = np.zeros(len(y_stack))
    stack_metrics = []
    for tr_idx, val_idx in skf.split(X_stack, y_stack):
        model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        model.fit(X_stack[tr_idx], y_stack[tr_idx])
        prob = predict_positive_proba(model, X_stack[val_idx])
        stack_oof[val_idx] = prob
        stack_metrics.append(evaluate(y_stack_eval[val_idx], prob))
    stack_mean = pd.DataFrame(stack_metrics).mean().to_dict()
    stack_std = pd.DataFrame(stack_metrics).std().to_dict()

    gkf = GroupKFold(n_splits=N_SPLITS)
    X_stack_gkf = oof_probabilities[branch_oof_cols_gkf].values
    stack_oof_gkf = np.zeros(len(y_stack))
    stack_metrics_gkf = []
    for tr_idx, val_idx in gkf.split(X_stack_gkf, y_stack, groups):
        model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        model.fit(X_stack_gkf[tr_idx], y_stack[tr_idx])
        prob = predict_positive_proba(model, X_stack_gkf[val_idx])
        stack_oof_gkf[val_idx] = prob
        stack_metrics_gkf.append(evaluate(y_stack_eval[val_idx], prob))
    stack_gkf_mean = pd.DataFrame(stack_metrics_gkf).mean().to_dict()
    stack_gkf_std = pd.DataFrame(stack_metrics_gkf).std().to_dict()

    results_summary.append({
        "experiment": "M8_Late_Fusion_Stacking",
        "n_features": 4,
        "skf_auc_mean": stack_mean["roc_auc"],
        "skf_auc_std": stack_std["roc_auc"],
        "skf_f1_mean": stack_mean["f1"],
        "skf_f1_std": stack_std["f1"],
        "gkf_auc_mean": stack_gkf_mean["roc_auc"],
        "gkf_auc_std": stack_gkf_std["roc_auc"],
        "gkf_f1_mean": stack_gkf_mean["f1"],
        "gkf_f1_std": stack_gkf_std["f1"],
    })
    oof_probabilities["M8_Late_Fusion_Stacking_skf_prob"] = stack_oof
    oof_probabilities["M8_Late_Fusion_Stacking_gkf_prob"] = stack_oof_gkf

    # M12 NoClimate Late Fusion Stacking：使用非气候分支 OOF 概率
    # 注意：Preservation 分支在当前分支定义中全为气候/环境变量，
    # 因此“非气候四分支”实际可用的分支为 Source + Transport + Deposition（3 个）。
    print("  Running M12_NoClimate_LateFusion_Stacking ...")

    def get_branch_no_climate_features(branch):
        return [c for c in branch_sets[branch] if not is_climate_col(c)]

    no_clim_branches = []
    no_clim_branch_oof = {}
    for b in ["Source", "Transport", "Deposition", "Preservation"]:
        feats_no_clim = get_branch_no_climate_features(b)
        if len(feats_no_clim) == 0:
            print(f"    Branch {b} has no non-climate features, skipping in M12.")
            continue
        no_clim_branches.append(b)
        res_no_clim = cv_model(df, y, groups, feats_no_clim, class_weight=None)
        no_clim_branch_oof[f"{b}_NoClimate_skf_prob"] = res_no_clim["oof_prob_skf"]
        no_clim_branch_oof[f"{b}_NoClimate_gkf_prob"] = res_no_clim["oof_prob_gkf"]
    for k, v in no_clim_branch_oof.items():
        oof_probabilities[k] = v

    stack_cols_m12_skf = [f"{b}_NoClimate_skf_prob" for b in no_clim_branches]
    X_m12_skf = oof_probabilities[stack_cols_m12_skf].values
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    stack_oof_m12 = np.zeros(len(y))
    stack_metrics_m12 = []
    for tr_idx, val_idx in skf.split(X_m12_skf, y):
        model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        model.fit(X_m12_skf[tr_idx], y.iloc[tr_idx])
        prob = predict_positive_proba(model, X_m12_skf[val_idx])
        stack_oof_m12[val_idx] = prob
        stack_metrics_m12.append(evaluate(y_eval[val_idx], prob))
    stack_m12_mean = pd.DataFrame(stack_metrics_m12).mean().to_dict()
    stack_m12_std = pd.DataFrame(stack_metrics_m12).std().to_dict()

    stack_cols_m12_gkf = [f"{b}_NoClimate_gkf_prob" for b in no_clim_branches]
    X_m12_gkf = oof_probabilities[stack_cols_m12_gkf].values
    gkf = GroupKFold(n_splits=N_SPLITS)
    stack_oof_m12_gkf = np.zeros(len(y))
    stack_metrics_m12_gkf = []
    for tr_idx, val_idx in gkf.split(X_m12_gkf, y, groups):
        model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        model.fit(X_m12_gkf[tr_idx], y.iloc[tr_idx])
        prob = predict_positive_proba(model, X_m12_gkf[val_idx])
        stack_oof_m12_gkf[val_idx] = prob
        stack_metrics_m12_gkf.append(evaluate(y_eval[val_idx], prob))
    stack_m12_gkf_mean = pd.DataFrame(stack_metrics_m12_gkf).mean().to_dict()
    stack_m12_gkf_std = pd.DataFrame(stack_metrics_m12_gkf).std().to_dict()

    results_summary.append({
        "experiment": "M12_NoClimate_LateFusion_Stacking",
        "n_features": len(no_clim_branches),
        "skf_auc_mean": stack_m12_mean["roc_auc"],
        "skf_auc_std": stack_m12_std["roc_auc"],
        "skf_f1_mean": stack_m12_mean["f1"],
        "skf_f1_std": stack_m12_std["f1"],
        "gkf_auc_mean": stack_m12_gkf_mean["roc_auc"],
        "gkf_auc_std": stack_m12_gkf_std["roc_auc"],
        "gkf_f1_mean": stack_m12_gkf_mean["f1"],
        "gkf_f1_std": stack_m12_gkf_std["f1"],
    })
    oof_probabilities["M12_NoClimate_LateFusion_Stacking_skf_prob"] = stack_oof_m12
    oof_probabilities["M12_NoClimate_LateFusion_Stacking_gkf_prob"] = stack_oof_m12_gkf

    # M10 stacking
    print("  Running M10_ClimateResidual_Stacking ...")
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    depos_resid_skf_oof = np.zeros(len(df))
    for tr_idx, val_idx in skf.split(df, y):
        train_resid, val_resid = compute_climate_residuals_both(
            df.iloc[tr_idx].copy(), df.iloc[val_idx].copy(),
            element_cols, climate_core_for_residual, r_threshold=RESIDUAL_R_THRESHOLD,
        )
        resid_cols = [c for c in train_resid.columns if c.endswith("_resid")]
        feat = [c for c in resid_cols if c in train_resid.columns]
        if len(feat) == 0:
            continue
        pipe = make_pipeline(feat, build_base_model())
        pipe.fit(train_resid[feat], y.iloc[tr_idx])
        prob = predict_positive_proba(pipe, val_resid[feat])
        depos_resid_skf_oof[val_idx] = prob
    oof_probabilities["Deposition_resid_skf_prob"] = depos_resid_skf_oof

    stack_cols_m10 = [
        "M3_Source_Only_skf_prob",
        "M4_Transport_Only_skf_prob",
        "Deposition_resid_skf_prob",
        "M6_Preservation_Only_skf_prob",
    ]
    X_m10 = oof_probabilities[stack_cols_m10].values
    stack_oof_m10 = np.zeros(len(y))
    stack_metrics_m10 = []
    for tr_idx, val_idx in skf.split(X_m10, y):
        model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        model.fit(X_m10[tr_idx], y.iloc[tr_idx])
        prob = predict_positive_proba(model, X_m10[val_idx])
        stack_oof_m10[val_idx] = prob
        stack_metrics_m10.append(evaluate(y_eval[val_idx], prob))
    stack_m10_mean = pd.DataFrame(stack_metrics_m10).mean().to_dict()
    stack_m10_std = pd.DataFrame(stack_metrics_m10).std().to_dict()

    # M10 GKF：需要独立生成 Deposition_resid 的 GKF OOF
    gkf = GroupKFold(n_splits=N_SPLITS)
    depos_resid_gkf_oof = np.zeros(len(df))
    for tr_idx, val_idx in gkf.split(df, y, groups):
        train_resid, val_resid = compute_climate_residuals_both(
            df.iloc[tr_idx].copy(), df.iloc[val_idx].copy(),
            element_cols, climate_core_for_residual, r_threshold=RESIDUAL_R_THRESHOLD,
        )
        resid_cols = [c for c in train_resid.columns if c.endswith("_resid")]
        feat = [c for c in resid_cols if c in train_resid.columns]
        if len(feat) == 0:
            continue
        pipe = make_pipeline(feat, build_base_model())
        pipe.fit(train_resid[feat], y.iloc[tr_idx])
        prob = predict_positive_proba(pipe, val_resid[feat])
        depos_resid_gkf_oof[val_idx] = prob
    oof_probabilities["Deposition_resid_gkf_prob"] = depos_resid_gkf_oof

    stack_cols_m10_gkf = [
        "M3_Source_Only_gkf_prob",
        "M4_Transport_Only_gkf_prob",
        "Deposition_resid_gkf_prob",
        "M6_Preservation_Only_gkf_prob",
    ]
    X_m10_gkf = oof_probabilities[stack_cols_m10_gkf].values
    stack_oof_m10_gkf = np.zeros(len(y))
    stack_metrics_m10_gkf = []
    for tr_idx, val_idx in gkf.split(X_m10_gkf, y, groups):
        model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        model.fit(X_m10_gkf[tr_idx], y.iloc[tr_idx])
        prob = predict_positive_proba(model, X_m10_gkf[val_idx])
        stack_oof_m10_gkf[val_idx] = prob
        stack_metrics_m10_gkf.append(evaluate(y_eval[val_idx], prob))
    stack_m10_gkf_mean = pd.DataFrame(stack_metrics_m10_gkf).mean().to_dict()
    stack_m10_gkf_std = pd.DataFrame(stack_metrics_m10_gkf).std().to_dict()

    results_summary.append({
        "experiment": "M10_ClimateResidual_Stacking",
        "n_features": 4,
        "skf_auc_mean": stack_m10_mean["roc_auc"],
        "skf_auc_std": stack_m10_std["roc_auc"],
        "skf_f1_mean": stack_m10_mean["f1"],
        "skf_f1_std": stack_m10_std["f1"],
        "gkf_auc_mean": stack_m10_gkf_mean["roc_auc"],
        "gkf_auc_std": stack_m10_gkf_std["roc_auc"],
        "gkf_f1_mean": stack_m10_gkf_mean["f1"],
        "gkf_f1_std": stack_m10_gkf_std["f1"],
    })
    oof_probabilities["M10_ClimateResidual_Stacking_skf_prob"] = stack_oof_m10
    oof_probabilities["M10_ClimateResidual_Stacking_gkf_prob"] = stack_oof_m10_gkf

    # Feature importance for non-stacking experiments
    print("  Computing feature importances ...")
    importance_records = []
    for exp_name in ["M1_Full_RF", "M2_NoClimate_RF", "M7_Early_Fusion", "M11_NoClimate_EarlyFusion"]:
        cfg = experiments[exp_name]
        feats = cfg["features"]
        imp = fit_and_importance(df, y, feats)
        imp["experiment"] = exp_name
        imp["rank"] = np.arange(1, len(imp) + 1)
        importance_records.append(imp.head(30))
    for branch in ["Source", "Transport", "Deposition", "Preservation"]:
        feats = branch_sets[branch]
        if len(feats) == 0:
            continue
        imp = fit_and_importance(df, y, feats)
        imp["experiment"] = f"{branch}_Only"
        imp["rank"] = np.arange(1, len(imp) + 1)
        importance_records.append(imp.head(20))
    for branch in ["Source", "Transport", "Deposition"]:
        feats = [c for c in branch_sets[branch] if not is_climate_col(c)]
        if len(feats) == 0:
            continue
        imp = fit_and_importance(df, y, feats)
        imp["experiment"] = f"{branch}_NoClimate_Only"
        imp["rank"] = np.arange(1, len(imp) + 1)
        importance_records.append(imp.head(20))
    importance_df = pd.concat(importance_records, ignore_index=True)

    # State / zone metrics
    print("  Computing state / zone metrics ...")
    key_models = [
        "M1_Full_RF_skf_prob", "M2_NoClimate_RF_skf_prob",
        "M7_Early_Fusion_skf_prob", "M8_Late_Fusion_Stacking_skf_prob",
        "M9_ClimateResidual_EarlyFusion_skf_prob",
        "M10_ClimateResidual_Stacking_skf_prob",
        "M11_NoClimate_EarlyFusion_skf_prob",
        "M12_NoClimate_LateFusion_Stacking_skf_prob",
        "M12_NoClimate_LateFusion_Stacking_gkf_prob",
    ]
    state_metrics_list, zone_metrics_list = [], []
    for col in key_models:
        if col not in oof_probabilities.columns:
            continue
        tmp = oof_probabilities[["Y_label", "state", "zone", col]].rename(columns={col: "prob"})
        sm, zm = state_zone_metrics(tmp, "prob")
        sm["model"] = col
        zm["model"] = col
        state_metrics_list.append(sm)
        zone_metrics_list.append(zm)
    state_metrics = pd.concat(state_metrics_list, ignore_index=True) if state_metrics_list else pd.DataFrame()
    zone_metrics = pd.concat(zone_metrics_list, ignore_index=True) if zone_metrics_list else pd.DataFrame()

    # Save
    results_df = pd.DataFrame(results_summary)
    results_df.insert(0, "base_model", BASE_MODEL)
    results_df.to_csv(out_dir / "experiment_metrics_summary.csv", index=False)
    importance_df.to_csv(out_dir / "feature_importance_top.csv", index=False)
    oof_probabilities.to_csv(out_dir / "oof_probabilities.csv", index=False)
    if not state_metrics.empty:
        state_metrics.to_csv(out_dir / "state_metrics.csv", index=False)
    if not zone_metrics.empty:
        zone_metrics.to_csv(out_dir / "zone_metrics.csv", index=False)

    # Markdown report
    report_path = out_dir / "experiment_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# 负样本比例 {ratio_str} 的四分支 RF 与融合实验报告\n\n")
        f.write(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## 实验设置\n\n")
        f.write(f"- 数据集：{data_path.name}\n")
        f.write(f"- 样本数：{len(df)}（正例 {n_pos}，负例 {n_neg}，中性 {n_neutral}，比例约 {ratio_str}）\n")
        f.write(f"- 训练标签：{'三类训练（正样本/负样本/中性样本），指标按正样本 vs 其他样本计算' if include_neutral else '二分类训练（正样本/负样本）'}\n")
        f.write(f"- 分支基础模型：{base_model_display_name()}\n")
        f.write(f"- 验证：{N_SPLITS} 折 StratifiedKFold + {N_SPLITS} 折 GroupKFold by state\n")
        f.write(f"- 新增模型：M11（非气候四分支 early fusion）、M12（非气候四分支 late fusion stacking）\n\n")

        f.write("## 全局性能对比\n\n")
        f.write(to_markdown_table(results_df, include_index=False))
        f.write("\n\n")

        f.write("## 非气候融合 vs 完整融合\n\n")
        m7_skf = results_df[results_df["experiment"] == "M7_Early_Fusion"]["skf_auc_mean"].values
        m11_skf = results_df[results_df["experiment"] == "M11_NoClimate_EarlyFusion"]["skf_auc_mean"].values
        m8_skf = results_df[results_df["experiment"] == "M8_Late_Fusion_Stacking"]["skf_auc_mean"].values
        m12_skf = results_df[results_df["experiment"] == "M12_NoClimate_LateFusion_Stacking"]["skf_auc_mean"].values
        m8_gkf = results_df[results_df["experiment"] == "M8_Late_Fusion_Stacking"]["gkf_auc_mean"].values
        m12_gkf = results_df[results_df["experiment"] == "M12_NoClimate_LateFusion_Stacking"]["gkf_auc_mean"].values
        m7_gkf = results_df[results_df["experiment"] == "M7_Early_Fusion"]["gkf_auc_mean"].values
        m11_gkf = results_df[results_df["experiment"] == "M11_NoClimate_EarlyFusion"]["gkf_auc_mean"].values
        if len(m7_skf) and len(m11_skf) and len(m8_skf) and len(m12_skf) and len(m7_gkf) and len(m12_gkf):
            f.write(f"- **M11 vs M7 SKF AUC**: {m7_skf[0]:.4f} vs {m11_skf[0]:.4f} (Δ = {m11_skf[0] - m7_skf[0]:+.4f})\n")
            f.write(f"- **M12 vs M8 SKF AUC**: {m8_skf[0]:.4f} vs {m12_skf[0]:.4f} (Δ = {m12_skf[0] - m8_skf[0]:+.4f})\n")
            f.write(f"- **M11 vs M7 GKF AUC**: {m7_gkf[0]:.4f} vs {m11_gkf[0]:.4f} (Δ = {m11_gkf[0] - m7_gkf[0]:+.4f})\n")
            f.write(f"- **M12 vs M8 GKF AUC**: {m8_gkf[0]:.4f} vs {m12_gkf[0]:.4f} (Δ = {m12_gkf[0] - m8_gkf[0]:+.4f})\n")

        f.write("\n## 关键发现\n\n")
        best_skf = results_df.loc[results_df["skf_auc_mean"].idxmax()]
        best_gkf = results_df.loc[results_df["gkf_auc_mean"].idxmax()]
        f.write(f"- **SKF AUC 最高**：{best_skf['experiment']}（{best_skf['skf_auc_mean']:.4f} ± {best_skf['skf_auc_std']:.4f}）\n")
        f.write(f"- **GKF AUC 最高**：{best_gkf['experiment']}（{best_gkf['gkf_auc_mean']:.4f} ± {best_gkf['gkf_auc_std']:.4f}）\n")
        m9_row = results_df[results_df["experiment"] == "M9_ClimateResidual_EarlyFusion"]
        m7_row = results_df[results_df["experiment"] == "M7_Early_Fusion"]
        if not m9_row.empty and not m7_row.empty:
            diff_skf = m9_row.iloc[0]["skf_auc_mean"] - m7_row.iloc[0]["skf_auc_mean"]
            diff_gkf = m9_row.iloc[0]["gkf_auc_mean"] - m7_row.iloc[0]["gkf_auc_mean"]
            f.write(f"- **M9 vs M7 SKF AUC 差异**：{diff_skf:+.4f}\n")
            f.write(f"- **M9 vs M7 GKF AUC 差异**：{diff_gkf:+.4f}\n")

        m11_row = results_df[results_df["experiment"] == "M11_NoClimate_EarlyFusion"]
        m2_row = results_df[results_df["experiment"] == "M2_NoClimate_RF"]
        if not m11_row.empty and not m2_row.empty:
            f.write(f"- **M11 vs M2 SKF AUC**: {m2_row.iloc[0]['skf_auc_mean']:.4f} vs {m11_row.iloc[0]['skf_auc_mean']:.4f}\n")
            f.write(f"- **M11 vs M2 GKF AUC**: {m2_row.iloc[0]['gkf_auc_mean']:.4f} vs {m11_row.iloc[0]['gkf_auc_mean']:.4f}\n")

        m5_row = results_df[results_df["experiment"] == "M5_Deposition_Only"]
        if not m5_row.empty:
            f.write(f"- **Deposition_Only**：SKF AUC = {m5_row.iloc[0]['skf_auc_mean']:.4f}，GKF AUC = {m5_row.iloc[0]['gkf_auc_mean']:.4f}\n")

        f.write("\n## 结论\n\n")
        f.write(f"对于 {ratio_str} 正负样本比例：\n")
        if m9_row.empty or m7_row.empty:
            f.write("- 请参考全局性能表。\n")
        else:
            m9_skf = m9_row.iloc[0]["skf_auc_mean"]
            m7_skf = m7_row.iloc[0]["skf_auc_mean"]
            m9_gkf = m9_row.iloc[0]["gkf_auc_mean"]
            m7_gkf = m7_row.iloc[0]["gkf_auc_mean"]
            if m9_skf <= m7_skf and m9_gkf <= m7_gkf:
                f.write("- M9 climate-residualization 未带来性能提升，跨州泛化能力低于 M7。\n")
            elif m9_gkf > m7_gkf:
                f.write("- M9 在跨州留出验证中略优于 M7，可考虑作为候选模型。\n")
            else:
                f.write("- M9 与 M7 性能接近。\n")
        f.write(f"- {best_gkf['experiment']} 在跨州留出（GKF）中表现最佳，推荐优先使用。\n")

    print(f"  Saved to {out_dir}")
    return results_df


def dataset_label(path):
    path = Path(path)
    parent = path.parent.name
    stem = path.stem
    if "with_neutral" in stem:
        return f"{parent}_with_neutral_training"
    for prefix in ["model_dataset_western_core_", "model_dataset_"]:
        stem = stem.replace(prefix, "")
    if parent.startswith("ratio_") or parent.startswith("known_"):
        return parent
    return stem


def discover_datasets(dataset_root, include_neutral=False):
    dataset_root = Path(dataset_root)
    candidates = []

    for ratio in ["ratio_1_2", "ratio_1_5", "ratio_1_10", "ratio_1_20"]:
        candidates.extend(sorted((dataset_root / ratio).glob("model_dataset_*_all_features_v1.csv")))
        candidates.extend(sorted((dataset_root / ratio).glob("model_dataset_*_all_features_v1.parquet")))

    known_root = dataset_root / "known_mining_neutral"
    known_pattern = "*with_neutral_all_features_v1" if include_neutral else "*supervised_all_features_v1"
    candidates.extend(sorted(known_root.glob(f"**/model_dataset_{known_pattern}.csv")))
    candidates.extend(sorted(known_root.glob(f"**/model_dataset_{known_pattern}.parquet")))

    unique = []
    seen = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved not in seen and (include_neutral or "with_neutral" not in path.name):
            unique.append(path)
            seen.add(resolved)
    return unique


def expand_dataset_args(dataset_args, include_neutral=False):
    paths = []
    for raw in dataset_args:
        path = Path(raw)
        if path.is_dir():
            paths.extend(sorted(path.glob("model_dataset_*_all_features_v1.csv")))
            paths.extend(sorted(path.glob("model_dataset_*_all_features_v1.parquet")))
            paths.extend(sorted(path.glob("model_dataset_*_supervised_all_features_v1.csv")))
            paths.extend(sorted(path.glob("model_dataset_*_supervised_all_features_v1.parquet")))
        else:
            paths.append(path)
    return [p for p in paths if p.exists() and (include_neutral or "with_neutral" not in p.name)]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run branch-fusion experiments for porphyry copper datasets."
    )
    parser.add_argument(
        "--branch-json",
        default=str(DEFAULT_BRANCH_JSON),
        help="Feature branch assignment JSON.",
    )
    parser.add_argument(
        "--dataset-root",
        default=str(DEFAULT_DATASET_ROOT),
        help="Root directory used when --datasets is not supplied.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Dataset files or directories. If omitted, the standard ratio datasets are discovered.",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Output directory for branch-fusion experiment results.",
    )
    parser.add_argument("--n-estimators", type=int, default=N_ESTIMATORS)
    parser.add_argument("--max-depth", type=int, default=MAX_DEPTH)
    parser.add_argument("--min-samples-leaf", type=int, default=MIN_SAMPLES_LEAF)
    parser.add_argument("--n-splits", type=int, default=N_SPLITS)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument("--residual-threshold", type=float, default=RESIDUAL_R_THRESHOLD)
    parser.add_argument(
        "--base-model",
        choices=["rf", "hist_gradient_boosting", "logistic_regression"],
        default=BASE_MODEL,
        help="Base learner used inside branch and early-fusion models.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use lighter parameters for a smoke test.",
    )
    parser.add_argument(
        "--include-neutral",
        action="store_true",
        help="Keep Y_label=-1 samples as a third training class; metrics still evaluate Y_label=1 vs rest.",
    )
    return parser.parse_args()


def main():
    global RANDOM_STATE, N_ESTIMATORS, MIN_SAMPLES_LEAF, MAX_DEPTH, N_SPLITS, RESIDUAL_R_THRESHOLD, BASE_MODEL

    args = parse_args()
    RANDOM_STATE = args.random_state
    BASE_MODEL = args.base_model
    N_ESTIMATORS = 120 if args.quick else args.n_estimators
    MAX_DEPTH = 8 if args.quick else args.max_depth
    MIN_SAMPLES_LEAF = args.min_samples_leaf
    N_SPLITS = 3 if args.quick else args.n_splits
    RESIDUAL_R_THRESHOLD = args.residual_threshold

    branch_json = Path(args.branch_json)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    with open(branch_json, "r", encoding="utf-8") as f:
        branch_info = json.load(f)

    datasets = (
        expand_dataset_args(args.datasets, include_neutral=args.include_neutral)
        if args.datasets
        else discover_datasets(args.dataset_root, include_neutral=args.include_neutral)
    )
    if not datasets:
        raise FileNotFoundError("No model datasets were found for the branch-fusion experiment.")

    all_results = []
    print("Branch-fusion experiment")
    print(f"  branch_json={branch_json}")
    print(f"  output_root={output_root}")
    print(f"  base_model={BASE_MODEL}")
    print(f"  n_estimators={N_ESTIMATORS}, max_depth={MAX_DEPTH}, n_splits={N_SPLITS}")
    print("  datasets:")
    for path in datasets:
        print(f"    - {path}")

    for data_path in datasets:
        label = dataset_label(data_path)
        out_dir = output_root / f"{label}_{BASE_MODEL}"
        results = run_one_ratio(data_path, out_dir, branch_info, include_neutral=args.include_neutral)
        results.insert(0, "dataset", label)
        results.insert(1, "dataset_file", str(Path(data_path).resolve()))
        all_results.append(results)

    summary = pd.concat(all_results, ignore_index=True)
    summary_cols = [
        "dataset", "dataset_file", "base_model", "experiment", "n_features",
        "skf_auc_mean", "skf_auc_std", "skf_f1_mean", "skf_f1_std",
        "gkf_auc_mean", "gkf_auc_std", "gkf_f1_mean", "gkf_f1_std",
    ]
    summary = summary[[c for c in summary_cols if c in summary.columns]]
    summary_path = output_root / "branch_fusion_summary.csv"
    summary.to_csv(summary_path, index=False)

    skf_pivot = summary.pivot(index="experiment", columns="dataset", values="skf_auc_mean")
    gkf_pivot = summary.pivot(index="experiment", columns="dataset", values="gkf_auc_mean")
    gkf_f1_pivot = summary.pivot(index="experiment", columns="dataset", values="gkf_f1_mean")

    report_path = output_root / "branch_fusion_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# 分支融合新方法实验报告\n\n")
        f.write(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## 实验设置\n\n")
        f.write(f"- 脚本：`scripts/stage_10_branch_fusion/38_branch_fusion_negative_ratio_experiment.py`\n")
        f.write(f"- 分支映射：`{branch_json}`\n")
        f.write(f"- 分支基础模型：{base_model_display_name()}\n")
        f.write(f"- 验证：{N_SPLITS} 折 StratifiedKFold + {N_SPLITS} 折 GroupKFold(by state)\n")
        f.write(f"- 气候残差敏感阈值：Spearman |r| >= {RESIDUAL_R_THRESHOLD}\n\n")
        f.write(f"- 中性样本训练：{'是，Y_label=-1 作为第三类参与训练' if args.include_neutral else '否，仅使用 Y_label=0/1'}\n\n")

        f.write("## 数据集\n\n")
        for path in datasets:
            f.write(f"- `{dataset_label(path)}`: `{Path(path).name}`\n")
        f.write("\n")

        f.write("## SKF AUC 跨数据集对比\n\n")
        f.write(to_markdown_table(skf_pivot, include_index=True))
        f.write("\n\n")

        f.write("## GKF AUC 跨数据集对比\n\n")
        f.write(to_markdown_table(gkf_pivot, include_index=True))
        f.write("\n\n")

        f.write("## GKF F1 跨数据集对比\n\n")
        f.write(to_markdown_table(gkf_f1_pivot, include_index=True))
        f.write("\n\n")

        f.write("## 每个数据集的最佳 GKF AUC 模型\n\n")
        for dataset, group in summary.groupby("dataset"):
            best = group.loc[group["gkf_auc_mean"].idxmax()]
            f.write(
                f"- `{dataset}`: {best['experiment']}，"
                f"GKF AUC={best['gkf_auc_mean']:.4f}，GKF F1={best['gkf_f1_mean']:.4f}\n"
            )

        f.write("\n## 解释要点\n\n")
        f.write("- M3-M6 用于观察单个成矿概念分支是否足以形成预测信号。\n")
        f.write("- M7/M8 分别代表四分支早期融合和后期 stacking，检验分支化建模是否优于直接拼接。\n")
        f.write("- M11/M12 是去气候版本，用于判断模型是否主要依赖非气候探矿信号。\n")
        f.write("- M9/M10 对 Deposition 分支做气候残差化，用于检验气候影响解耦后是否提升跨州泛化。\n")

    print(f"\nAll branch-fusion results saved to: {output_root}")
    print(f"Summary: {summary_path}")
    print(f"Report:  {report_path}")


if __name__ == "__main__":
    main()
