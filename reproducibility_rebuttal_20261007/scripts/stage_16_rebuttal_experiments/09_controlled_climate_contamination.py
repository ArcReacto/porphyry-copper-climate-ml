from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from xgb_model import PARAMETERS, make_xgb_model


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_DIR = (
    PROJECT_ROOT
    / "data"
    / "run_inputs"
    / "known_mining_neutral_ratio_1_10_supervised_all_features_v1"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "rebuttal_experiments" / "xgboost" / "09_controlled_climate_contamination"
METRICS = ["roc_auc", "average_precision", "f1", "top05_f1", "top05_ndcg", "top10_f1", "top10_ndcg"]
MODEL_KEYS = ["M2_raw", "M3_broad", "M4_spearman", "Selection_drop"]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


exp01 = load_module(
    "rebuttal_exp01_for_controlled_contamination",
    PROJECT_ROOT / "scripts" / "stage_16_rebuttal_experiments" / "01_fold_local_graphunion.py",
)
base = exp01.base63


def make_recipe(
    train_df: pd.DataFrame,
    model_key: str,
    no_climate: list[str],
    climate: list[str],
    geochemistry: list[str],
    selected: list[str],
):
    selected_set = set(selected)
    if model_key == "M2_raw":
        raw_cols, residual_cols = no_climate, []
    elif model_key == "M3_broad":
        residual_cols = geochemistry
        raw_cols = [column for column in no_climate if column not in set(geochemistry)]
    elif model_key == "M4_spearman":
        residual_cols = [column for column in no_climate if column in selected_set]
        raw_cols = [column for column in no_climate if column not in selected_set]
    elif model_key == "Selection_drop":
        residual_cols = []
        raw_cols = [column for column in no_climate if column not in selected_set]
    else:
        raise ValueError(model_key)
    return base.FeatureRecipe(model_key, raw_cols, residual_cols, climate).fit(train_df)


def climate_score(train_df: pd.DataFrame, test_df: pd.DataFrame, climate: list[str]) -> tuple[np.ndarray, np.ndarray]:
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=1, random_state=20261001)),
        ]
    )
    train_score = model.fit_transform(train_df[climate]).ravel()
    test_score = model.transform(test_df[climate]).ravel()
    mean = float(np.mean(train_score))
    std = float(np.std(train_score, ddof=0))
    std = std if std > 0 else 1.0
    return (train_score - mean) / std, (test_score - mean) / std


def inject_contamination(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    columns: list[str],
    train_climate_z: np.ndarray,
    test_climate_z: np.ndarray,
    gamma: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    train_out = train_df.copy()
    test_out = test_df.copy()
    scales = train_df[columns].std(ddof=0).replace(0, 1.0).fillna(1.0)
    for column in columns:
        train_mask = train_out[column].notna()
        test_mask = test_out[column].notna()
        train_out.loc[train_mask, column] = (
            train_out.loc[train_mask, column].astype(float)
            + gamma * float(scales[column]) * train_climate_z[train_mask.to_numpy()]
        )
        test_out.loc[test_mask, column] = (
            test_out.loc[test_mask, column].astype(float)
            + gamma * float(scales[column]) * test_climate_z[test_mask.to_numpy()]
        )
    return train_out, test_out, scales


def representation(recipe, frame: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, float]:
    transformed = recipe.transform(frame)
    values: dict[str, pd.Series] = {}
    for column in columns:
        residual_name = f"{column}_climate_resid"
        if residual_name in transformed.columns:
            values[column] = transformed[residual_name]
        elif column in transformed.columns:
            values[column] = transformed[column]
    available_fraction = len(values) / max(len(columns), 1)
    return pd.DataFrame(values, index=frame.index), available_fraction


def standardized_rmse(
    estimated: pd.DataFrame,
    reference: pd.DataFrame,
    scales: pd.Series,
) -> float:
    common = [column for column in estimated.columns if column in reference.columns]
    if not common:
        return float("nan")
    values: list[np.ndarray] = []
    for column in common:
        mask = estimated[column].notna() & reference[column].notna()
        if mask.any():
            denom = max(float(scales.get(column, 1.0)), 1e-12)
            values.append(((estimated.loc[mask, column] - reference.loc[mask, column]) / denom).to_numpy())
    if not values:
        return float("nan")
    errors = np.concatenate(values)
    return float(np.sqrt(np.mean(np.square(errors))))


def mean_feature_correlation(estimated: pd.DataFrame, reference: pd.DataFrame) -> float:
    values = []
    for column in estimated.columns.intersection(reference.columns):
        pair = pd.concat([estimated[column], reference[column]], axis=1).dropna()
        if len(pair) >= 30 and pair.iloc[:, 0].nunique() > 1 and pair.iloc[:, 1].nunique() > 1:
            values.append(pair.iloc[:, 0].corr(pair.iloc[:, 1], method="spearman"))
    return float(np.nanmean(values)) if values else float("nan")


def held_out_predictability(
    train_source: pd.DataFrame,
    test_source: pd.DataFrame,
    train_target: pd.DataFrame,
    test_target: pd.DataFrame,
) -> tuple[float, float]:
    if train_target.shape[1] == 0 or train_source.shape[1] == 0:
        return float("nan"), float("nan")
    target_medians = train_target.median(numeric_only=True)
    y_train = train_target.fillna(target_medians).to_numpy()
    y_test = test_target.fillna(target_medians).to_numpy()
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=10.0)),
        ]
    )
    model.fit(train_source, y_train)
    prediction = np.asarray(model.predict(test_source))
    if prediction.ndim == 1:
        prediction = prediction.reshape(-1, 1)
    if y_test.ndim == 1:
        y_test = y_test.reshape(-1, 1)
    correlations = []
    for column_index in range(y_test.shape[1]):
        observed = pd.Series(y_test[:, column_index])
        predicted = pd.Series(prediction[:, column_index])
        if observed.nunique() > 1 and predicted.nunique() > 1:
            correlations.append(observed.corr(predicted, method="spearman"))
    return (
        float(r2_score(y_test, prediction, multioutput="variance_weighted")),
        float(np.nanmean(correlations)) if correlations else float("nan"),
    )


def summarize(metrics: pd.DataFrame, diagnostics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_summary = metrics.groupby(["gamma", "model"])[METRICS].agg(["mean", "std", "count"]).reset_index()
    metric_summary.columns = [
        "_".join(str(piece) for piece in column if str(piece)) if isinstance(column, tuple) else str(column)
        for column in metric_summary.columns
    ]
    diag_cols = [
        "contaminated_recovery_srmse",
        "contaminated_recovery_spearman",
        "uncontaminated_damage_srmse",
        "contaminated_available_fraction",
        "heldout_climate_r2",
        "heldout_climate_predictive_spearman",
        "heldout_geostructure_r2",
        "heldout_geostructure_predictive_spearman",
        "n_selected",
    ]
    diag_summary = diagnostics.groupby(["gamma", "model"])[diag_cols].agg(["mean", "std", "count"]).reset_index()
    diag_summary.columns = [
        "_".join(str(piece) for piece in column if str(piece)) if isinstance(column, tuple) else str(column)
        for column in diag_summary.columns
    ]
    return metric_summary, diag_summary


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
    climate = fsets["climate_adjusters"]
    geochemistry = fsets["geochemistry"]
    no_climate = fsets["no_climate"]
    geostructure = role_table.loc[
        role_table["is_numeric"] & role_table["role"].eq("geo_structure"), "column"
    ].tolist()

    role_lookup = role_table.set_index("column")
    eligible = role_lookup.loc[geochemistry].sort_values(["missing_rate", "column"]).index.tolist()
    contamination_cols = eligible[: min(args.n_contaminated, len(eligible))]
    unaffected_cols = [column for column in geochemistry if column not in set(contamination_cols)]

    metric_rows: list[dict] = []
    prediction_rows: list[dict] = []
    diagnostic_rows: list[dict] = []
    selection_rows: list[pd.DataFrame] = []

    splitter = GroupKFold(n_splits=5)
    for fold, (train_idx, test_idx) in enumerate(splitter.split(df, y, groups), start=1):
        train_idx = np.asarray(train_idx)
        test_idx = np.asarray(test_idx)
        clean_train = df.iloc[train_idx].copy()
        clean_test = df.iloc[test_idx].copy()
        test_group = ";".join(sorted(groups.iloc[test_idx].unique().tolist()))
        train_climate_z, test_climate_z = climate_score(clean_train, clean_test, climate)

        for gamma in args.gammas:
            train_df, test_df, contamination_scales = inject_contamination(
                clean_train,
                clean_test,
                contamination_cols,
                train_climate_z,
                test_climate_z,
                gamma,
            )
            sensitive = base.local_spearman_sensitive(train_df, geochemistry, climate, args.spearman_threshold)
            selected = sensitive.loc[sensitive["is_spearman_sensitive"], "target_column"].tolist()
            selected_frame = sensitive.copy()
            selected_frame.insert(0, "gamma", gamma)
            selected_frame.insert(0, "test_group", test_group)
            selected_frame.insert(0, "fold", fold)
            selection_rows.append(selected_frame)

            unaffected_scales = clean_train[unaffected_cols].std(ddof=0).replace(0, 1.0).fillna(1.0)
            for model_key in MODEL_KEYS:
                current_recipe = make_recipe(
                    train_df,
                    model_key,
                    no_climate,
                    climate,
                    geochemistry,
                    selected,
                )
                model = make_xgb_model(y.iloc[train_idx]) if args.model_family == "xgboost" else base.make_rf_model()
                transformed_train = current_recipe.transform(train_df)
                transformed_test = current_recipe.transform(test_df)
                model.fit(transformed_train, y.iloc[train_idx])
                y_prob = model.predict_proba(transformed_test)[:, 1]
                metric_start = len(metric_rows)
                prediction_start = len(prediction_rows)
                base.add_metric_row(
                    metric_rows,
                    prediction_rows,
                    dataset="known_mining_neutral_ratio_1_10",
                    cv_name="groupkfold_state_controlled_contamination",
                    fold=fold,
                    test_group=test_group,
                    model_key=f"{model_key}_{suffix}",
                    used_cols=current_recipe.output_columns(),
                    train_idx=train_idx,
                    test_idx=test_idx,
                    df=df,
                    y_prob=y_prob,
                )
                metric_rows[metric_start]["gamma"] = gamma
                for row in prediction_rows[prediction_start:]:
                    row["gamma"] = gamma

                train_contam_rep, available_fraction = representation(
                    current_recipe, train_df, contamination_cols
                )
                test_contam_rep, _ = representation(current_recipe, test_df, contamination_cols)
                train_unaffected_rep, unaffected_available = representation(
                    current_recipe, train_df, unaffected_cols
                )
                test_unaffected_rep, _ = representation(current_recipe, test_df, unaffected_cols)

                common_contam = train_contam_rep.columns.intersection(test_contam_rep.columns).tolist()
                climate_r2, climate_spearman = held_out_predictability(
                    clean_train[climate],
                    clean_test[climate],
                    train_contam_rep[common_contam],
                    test_contam_rep[common_contam],
                )
                geology_r2, geology_spearman = held_out_predictability(
                    clean_train[geostructure],
                    clean_test[geostructure],
                    train_contam_rep[common_contam],
                    test_contam_rep[common_contam],
                )
                diagnostic_rows.append(
                    {
                        "fold": fold,
                        "test_group": test_group,
                        "gamma": gamma,
                        "model": f"{model_key}_{suffix}",
                        "n_selected": len(selected),
                        "contaminated_recovery_srmse": standardized_rmse(
                            test_contam_rep,
                            clean_test[contamination_cols],
                            contamination_scales,
                        ),
                        "contaminated_recovery_spearman": mean_feature_correlation(
                            test_contam_rep,
                            clean_test[contamination_cols],
                        ),
                        "uncontaminated_damage_srmse": standardized_rmse(
                            test_unaffected_rep,
                            clean_test[unaffected_cols],
                            unaffected_scales,
                        ),
                        "contaminated_available_fraction": available_fraction,
                        "uncontaminated_available_fraction": unaffected_available,
                        "heldout_climate_r2": climate_r2,
                        "heldout_climate_predictive_spearman": climate_spearman,
                        "heldout_geostructure_r2": geology_r2,
                        "heldout_geostructure_predictive_spearman": geology_spearman,
                    }
                )

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.DataFrame(prediction_rows)
    diagnostics = pd.DataFrame(diagnostic_rows)
    selections = pd.concat(selection_rows, ignore_index=True)
    metric_summary, diagnostic_summary = summarize(metrics, diagnostics)

    metrics.to_csv(output_dir / "controlled_contamination_fold_metrics.csv", index=False)
    predictions.to_csv(output_dir / "controlled_contamination_oof_predictions.csv", index=False)
    diagnostics.to_csv(output_dir / "controlled_contamination_diagnostics.csv", index=False)
    selections.to_csv(output_dir / "controlled_contamination_selection.csv", index=False)
    metric_summary.to_csv(output_dir / "controlled_contamination_metric_summary.csv", index=False)
    diagnostic_summary.to_csv(output_dir / "controlled_contamination_diagnostic_summary.csv", index=False)
    pd.DataFrame({"feature": contamination_cols}).to_csv(output_dir / "contaminated_features.csv", index=False)

    metric_show = metric_summary[
        ["gamma", "model"] + [f"{metric}_mean" for metric in METRICS]
    ]
    diag_show = diagnostic_summary[
        [
            "gamma",
            "model",
            "contaminated_recovery_srmse_mean",
            "contaminated_recovery_spearman_mean",
            "uncontaminated_damage_srmse_mean",
            "contaminated_available_fraction_mean",
            "heldout_climate_r2_mean",
            "heldout_climate_predictive_spearman_mean",
            "heldout_geostructure_r2_mean",
            "heldout_geostructure_predictive_spearman_mean",
            "n_selected_mean",
        ]
    ]
    report = f"""# Rebuttal 新实验 N1：受控气候污染恢复与信息保留

## 协议

- 数据：1:10 hard-negative，{len(df)} 条样本；
- 外层验证：5 折 GroupKFold by state；
- 预注册污染特征：按缺失率从低到高选取 {len(contamination_cols)} 个地球化学特征，不使用标签；
- 污染强度：{', '.join(str(value) for value in args.gammas)}；
- 污染式：`x_contaminated = x_clean + gamma * train_std(x) * climate_PC1_z`；
- 气候标准化、Spearman 选择和残差器全部仅使用训练折拟合。

## 预测指标

{markdown_table(metric_show)}

## 机制诊断

{markdown_table(diag_show)}

## 指标方向

- `contaminated_recovery_srmse`：恢复值相对原始干净值的标准化 RMSE，越小越好；
- `contaminated_recovery_spearman`：恢复值与干净值的平均 Spearman 相关，越大越好；
- `uncontaminated_damage_srmse`：未注入污染特征被改变的程度，越小越好；
- `heldout_climate_r2`：测试折中气候变量对特征表示的可预测性，去污染后应下降；
- `heldout_*_predictive_spearman`：辅助模型在测试折的预测值与实际特征的平均 Spearman 相关，用于补充跨州 R² 可能为负时的解读；
- `heldout_geostructure_r2`：测试折中固定地质/构造变量对特征表示的可预测性代理，用于监测信息保留，不是因果量。R² 为负表示辅助线性模型在该跨州折不如训练集均值基线，不能解释为“负的地质信息”。

## 结论边界

该实验只能验证方法对“人工注入且强度已知的气候项”的恢复能力，不能证明真实数据中所有气候相关变化都是噪声。`Selection_drop` 是删除选中特征的筛选对照，其可恢复特征比例需与 RMSE 一起解读。
"""
    (output_dir / "controlled_climate_contamination_report.md").write_text(report, encoding="utf-8")
    manifest = {
        "model_family": args.model_family,
        "xgboost_parameters": PARAMETERS if args.model_family == "xgboost" else None,
        "dataset_path": str(dataset_path),
        "run_dir": str(run_dir),
        "rows": len(df),
        "positive": int(y.sum()),
        "negative": int((y == 0).sum()),
        "n_contaminated": len(contamination_cols),
        "gammas": args.gammas,
        "spearman_threshold": args.spearman_threshold,
        "contamination_feature_selection": "lowest missing rate geochemistry columns; labels not used",
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote N1 outputs to: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Controlled climate contamination recovery experiment.")
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--n-contaminated", type=int, default=48)
    parser.add_argument("--gammas", nargs="+", type=float, default=[0.0, 0.5, 1.0, 2.0])
    parser.add_argument("--spearman-threshold", type=float, default=0.30)
    parser.add_argument("--model-family", choices=["xgboost", "rf"], default="xgboost")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
