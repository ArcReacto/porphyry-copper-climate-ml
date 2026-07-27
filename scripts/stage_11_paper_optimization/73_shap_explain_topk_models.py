from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_SCRIPT = PROJECT_ROOT / "scripts" / "stage_11_paper_optimization" / "63_full_feature_graph_guided_m4_and_perturbation.py"
RANDOM_STATE = 20260622
TARGET = "Y_label"

DEFAULT_DATASET = (
    PROJECT_ROOT
    / "outputs"
    / "model_datasets"
    / "by_sample_scheme"
    / "known_mining_neutral"
    / "known_mining_neutral_ratio_1_10"
    / "model_dataset_known_mining_neutral_ratio_1_10_supervised_all_features_v1.parquet"
)
DEFAULT_RUN_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "standardized_runs"
    / "known_mining_neutral_ratio_1_10_supervised_all_features_v1"
)
DEFAULT_GRAPH_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "paper_optimization"
    / "climate_sensitivity_graph"
    / "known_mining_neutral_ratio_1_10_supervised_all_features_v1"
)
DEFAULT_MAPPING_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "paper_optimization"
    / "full_feature_graph_guided_m4"
    / "known_mining_neutral_ratio_1_10_supervised_all_features_v1"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "paper_optimization"
    / "shap_explanations"
    / "known_mining_neutral_ratio_1_10_supervised_all_features_v1"
)


def load_base_module():
    spec = importlib.util.spec_from_file_location("full_feature_graph_guided_m4", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import base script: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


base = load_base_module()


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def make_xgboost(pos_weight: float) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                XGBClassifier(
                    n_estimators=400,
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
                ),
            ),
        ]
    )


def build_recipe(
    model_key: str,
    df: pd.DataFrame,
    role_table: pd.DataFrame,
    graph_dir: Path,
    mapping_dir: Path,
    spearman_threshold: float,
):
    mapping_path = mapping_dir / "graph_guided_raw_feature_targets.csv"
    graph_targets_path = graph_dir / "climate_sensitive_target_concepts.csv"
    mapping = pd.read_csv(mapping_path)
    if {"column", "residualizable"}.issubset(mapping.columns):
        graph_cols = mapping.copy()
    else:
        graph_cols = base.selected_graph_target_columns(mapping_path, graph_targets_path, role_table)
    graph_residual_cols = set(graph_cols.loc[graph_cols["residualizable"], "column"].tolist())
    fsets = base.feature_sets(role_table)
    return base.make_recipe(
        model_key,
        df,
        fsets,
        graph_residual_cols,
        spearman_threshold,
    )


def summarize_feature_groups(
    model_label: str,
    feature_names: list[str],
    shap_values: np.ndarray,
    role_table: pd.DataFrame,
) -> pd.DataFrame:
    role_lookup = role_table.set_index("column")["role"].to_dict() if "role" in role_table.columns else {}
    rows = []
    mean_abs = np.abs(shap_values).mean(axis=0)
    for name, value in zip(feature_names, mean_abs):
        raw_name = name.removesuffix("_climate_resid")
        rows.append(
            {
                "model": model_label,
                "feature": name,
                "raw_feature": raw_name,
                "role": role_lookup.get(raw_name, "derived_or_unknown"),
                "mean_abs_shap": float(value),
                "is_climate_residual": bool(name.endswith("_climate_resid")),
            }
        )
    detail = pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False)
    group = (
        detail.groupby(["model", "role"], dropna=False)["mean_abs_shap"]
        .sum()
        .reset_index()
        .sort_values(["model", "mean_abs_shap"], ascending=[True, False])
    )
    total = group.groupby("model")["mean_abs_shap"].transform("sum")
    group["share"] = group["mean_abs_shap"] / total.replace(0, np.nan)
    return detail, group


def save_shap_plots(
    *,
    model_label: str,
    pipeline: Pipeline,
    x_model: pd.DataFrame,
    y: pd.Series,
    role_table: pd.DataFrame,
    output_dir: Path,
    max_explain_rows: int,
    max_display: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline.fit(x_model, y)
    probabilities = pipeline.predict_proba(x_model)[:, 1]

    explain_idx = np.linspace(0, len(x_model) - 1, min(max_explain_rows, len(x_model)), dtype=int)
    x_explain = x_model.iloc[explain_idx].copy()
    x_imputed = pd.DataFrame(
        pipeline.named_steps["imputer"].transform(x_explain),
        columns=x_model.columns,
        index=x_explain.index,
    )
    booster = pipeline.named_steps["model"]
    explainer = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(x_imputed)
    if isinstance(shap_values, list):
        shap_values = shap_values[-1]
    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = np.asarray(base_value).reshape(-1)[-1]

    explanation = shap.Explanation(
        values=shap_values,
        base_values=np.repeat(float(base_value), len(x_imputed)),
        data=x_imputed.to_numpy(),
        feature_names=list(x_model.columns),
    )

    plt.figure()
    shap.plots.beeswarm(explanation, max_display=max_display, show=False)
    plt.title(f"{model_label} SHAP summary")
    plt.tight_layout()
    beeswarm_path = output_dir / f"{model_label}_shap_beeswarm.png"
    plt.savefig(beeswarm_path, dpi=220, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.plots.bar(explanation, max_display=max_display, show=False)
    plt.title(f"{model_label} mean |SHAP|")
    plt.tight_layout()
    bar_path = output_dir / f"{model_label}_shap_bar.png"
    plt.savefig(bar_path, dpi=220, bbox_inches="tight")
    plt.close()

    top_index = int(np.argmax(probabilities))
    if top_index in x_imputed.index:
        local_pos = list(x_imputed.index).index(top_index)
    else:
        local_pos = int(np.argmin(np.abs(explain_idx - top_index)))
    plt.figure()
    shap.plots.waterfall(explanation[local_pos], max_display=min(max_display, 18), show=False)
    plt.title(f"{model_label} top-ranked sample")
    plt.tight_layout()
    waterfall_path = output_dir / f"{model_label}_top_ranked_waterfall.png"
    plt.savefig(waterfall_path, dpi=220, bbox_inches="tight")
    plt.close()

    feature_detail, group_summary = summarize_feature_groups(
        model_label,
        list(x_model.columns),
        shap_values,
        role_table,
    )
    info = {
        "model": model_label,
        "n_train_rows": int(len(x_model)),
        "n_explain_rows": int(len(x_imputed)),
        "n_features": int(x_model.shape[1]),
        "top_ranked_row_index": top_index,
        "top_ranked_probability": float(probabilities[top_index]),
        "beeswarm": str(beeswarm_path),
        "bar": str(bar_path),
        "waterfall": str(waterfall_path),
    }
    return feature_detail, group_summary, info


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SHAP explanations for latest MPM models.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--graph-dir", type=Path, default=DEFAULT_GRAPH_DIR)
    parser.add_argument("--mapping-dir", type=Path, default=DEFAULT_MAPPING_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--spearman-threshold", type=float, default=0.3)
    parser.add_argument("--max-explain-rows", type=int, default=600)
    parser.add_argument("--max-display", type=int, default=20)
    args = parser.parse_args()

    df = read_table(args.dataset)
    df = df[df[TARGET].isin([0, 1])].reset_index(drop=True)
    y = df[TARGET].astype(int)
    pos_weight = float((y == 0).sum() / max((y == 1).sum(), 1))
    role_table = pd.read_csv(args.run_dir / "00_dataset_profile" / "feature_roles.csv")

    recipes = {
        "M2_NoClimate_XGB": build_recipe(
            "NoClimate_RF",
            df,
            role_table,
            args.graph_dir,
            args.mapping_dir,
            args.spearman_threshold,
        )[0],
        "M4_GraphUnion_XGB": build_recipe(
            "M4_GraphUnion_RF",
            df,
            role_table,
            args.graph_dir,
            args.mapping_dir,
            args.spearman_threshold,
        )[0],
    }

    all_feature_details = []
    all_group_summaries = []
    infos = []
    for model_label, recipe in recipes.items():
        x_model = recipe.transform(df)
        pipe = make_xgboost(pos_weight)
        feature_detail, group_summary, info = save_shap_plots(
            model_label=model_label,
            pipeline=pipe,
            x_model=x_model,
            y=y,
            role_table=role_table,
            output_dir=args.output_dir,
            max_explain_rows=args.max_explain_rows,
            max_display=args.max_display,
        )
        all_feature_details.append(feature_detail)
        all_group_summaries.append(group_summary)
        infos.append(info)

    feature_detail_df = pd.concat(all_feature_details, ignore_index=True)
    group_summary_df = pd.concat(all_group_summaries, ignore_index=True)
    feature_detail_df.to_csv(args.output_dir / "shap_feature_importance_detail.csv", index=False, encoding="utf-8-sig")
    group_summary_df.to_csv(args.output_dir / "shap_feature_group_importance.csv", index=False, encoding="utf-8-sig")

    feature_rows = []
    for info in infos:
        feature_rows.append(
            {
                "model": info["model"],
                "n_train_rows": info["n_train_rows"],
                "n_explain_rows": info["n_explain_rows"],
                "n_features": info["n_features"],
                "top_ranked_row_index": info["top_ranked_row_index"],
                "top_ranked_probability": info["top_ranked_probability"],
            }
        )
    pd.DataFrame(feature_rows).to_csv(args.output_dir / "shap_model_overview.csv", index=False, encoding="utf-8-sig")

    manifest = {
        "dataset": str(args.dataset),
        "run_dir": str(args.run_dir),
        "graph_dir": str(args.graph_dir),
        "mapping_dir": str(args.mapping_dir),
        "output_dir": str(args.output_dir),
        "spearman_threshold": args.spearman_threshold,
        "max_explain_rows": args.max_explain_rows,
        "models": infos,
        "note": "Models are refit on the full supervised 1:10 positive/negative dataset for post-hoc visualization.",
    }
    (args.output_dir / "shap_explanations_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote SHAP explanations to: {args.output_dir}")


if __name__ == "__main__":
    main()
