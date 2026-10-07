from __future__ import annotations

import argparse
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.io_utils import ensure_project_dirs, load_config, output_path, read_table, write_dataframe, write_json


warnings.filterwarnings("ignore", message="'penalty' was deprecated.*")
warnings.filterwarnings("ignore", message="Inconsistent values: penalty=l1.*")

INPUT_DATASET = "causal_graph/concept_features_western_core.parquet"
CONCEPT_DICTIONARY = "causal_graph/concept_feature_dictionary.csv"
OUTPUT_DIR = "causal_graph"
RANDOM_STATE = 20260622
MAIN_ENVIRONMENTS = [
    "arid_basin_or_range",
    "semi_arid_transition",
    "snow_influenced_mountain",
]
REFERENCE_ENVIRONMENTS = ["subhumid_humid_mountain"]


def concept_name(feature: str) -> str:
    return feature.removeprefix("concept_")


def safe_auc(y: pd.Series, values: pd.Series) -> float:
    mask = y.notna() & values.notna()
    if mask.sum() < 2 or y[mask].nunique() < 2:
        return math.nan
    return float(roc_auc_score(y[mask].astype(int), values[mask].astype(float)))


def cohen_d(pos_values: pd.Series, neg_values: pd.Series) -> float:
    pos = pd.to_numeric(pos_values, errors="coerce").dropna().astype(float)
    neg = pd.to_numeric(neg_values, errors="coerce").dropna().astype(float)
    if len(pos) < 2 or len(neg) < 2:
        return math.nan
    pooled_var = ((len(pos) - 1) * pos.var(ddof=1) + (len(neg) - 1) * neg.var(ddof=1)) / (len(pos) + len(neg) - 2)
    if pd.isna(pooled_var) or pooled_var <= 1e-12:
        return 0.0
    return float((pos.mean() - neg.mean()) / math.sqrt(pooled_var))


def load_concept_metadata(config: dict, concept_cols: list[str]) -> pd.DataFrame:
    path = output_path(config, "outputs", CONCEPT_DICTIONARY)
    if path.exists():
        meta = read_table(path)
        meta = meta[["concept", "concept_group", "role", "feature_count", "aggregation_method"]].drop_duplicates("concept")
    else:
        meta = pd.DataFrame({"concept": [concept_name(c) for c in concept_cols]})
        meta["concept_group"] = "unknown"
        meta["role"] = "unknown"
        meta["feature_count"] = np.nan
        meta["aggregation_method"] = "unknown"
    return meta


def feature_effects(df: pd.DataFrame, concept_cols: list[str], concept_meta: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for env, env_df in df.groupby("env_causal_group", dropna=False):
        y = env_df["Y_label"].astype(int)
        n_pos = int(y.sum())
        n_neg = int((y == 0).sum())
        for col in concept_cols:
            values = pd.to_numeric(env_df[col], errors="coerce")
            pos_values = values[y == 1]
            neg_values = values[y == 0]
            auc = safe_auc(y, values)
            direction = "positive_higher" if pos_values.mean(skipna=True) >= neg_values.mean(skipna=True) else "negative_higher"
            rows.append(
                {
                    "environment": env,
                    "concept": concept_name(col),
                    "concept_feature": col,
                    "n_rows": int(len(env_df)),
                    "n_positive": n_pos,
                    "n_negative": n_neg,
                    "positive_mean": float(pos_values.mean(skipna=True)),
                    "negative_mean": float(neg_values.mean(skipna=True)),
                    "mean_difference_pos_minus_neg": float(pos_values.mean(skipna=True) - neg_values.mean(skipna=True)),
                    "positive_median": float(pos_values.median(skipna=True)),
                    "negative_median": float(neg_values.median(skipna=True)),
                    "median_difference_pos_minus_neg": float(pos_values.median(skipna=True) - neg_values.median(skipna=True)),
                    "cohen_d": cohen_d(pos_values, neg_values),
                    "univariate_auc": auc,
                    "auc_effect_strength": float(abs(auc - 0.5) * 2) if not math.isnan(auc) else math.nan,
                    "direction": direction,
                    "missing_rate": float(values.isna().mean()),
                }
            )
    out = pd.DataFrame(rows)
    out = out.merge(concept_meta, on="concept", how="left")
    front = [
        "environment",
        "concept",
        "concept_group",
        "role",
        "concept_feature",
        "n_rows",
        "n_positive",
        "n_negative",
    ]
    out = out[front + [c for c in out.columns if c not in front]]
    return out.sort_values(["environment", "auc_effect_strength", "concept"], ascending=[True, False, True])


def bootstrap_selection(
    df: pd.DataFrame,
    concept_cols: list[str],
    n_bootstrap: int,
    top_k: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows = []
    for env, env_df in df.groupby("env_causal_group", dropna=False):
        pos_idx = env_df.index[env_df["Y_label"].astype(int) == 1].to_numpy()
        neg_idx = env_df.index[env_df["Y_label"].astype(int) == 0].to_numpy()
        if len(pos_idx) < 10 or len(neg_idx) < 10:
            for col in concept_cols:
                rows.append(
                    {
                        "environment": env,
                        "concept": concept_name(col),
                        "bootstrap_runs": 0,
                        "bootstrap_selected_count": 0,
                        "bootstrap_selection_rate": math.nan,
                        "reason": "too_few_positive_or_negative_samples",
                    }
                )
            continue

        selected_counts = {concept_name(col): 0 for col in concept_cols}
        for _ in range(n_bootstrap):
            sampled_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
            sampled_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
            sample = df.loc[np.concatenate([sampled_pos, sampled_neg])]
            y = sample["Y_label"].astype(int)
            strengths = []
            for col in concept_cols:
                auc = safe_auc(y, pd.to_numeric(sample[col], errors="coerce"))
                strength = abs(auc - 0.5) * 2 if not math.isnan(auc) else -1.0
                strengths.append((concept_name(col), strength))
            selected = sorted(strengths, key=lambda item: item[1], reverse=True)[:top_k]
            for concept, _ in selected:
                selected_counts[concept] += 1

        for col in concept_cols:
            concept = concept_name(col)
            count = selected_counts[concept]
            rows.append(
                {
                    "environment": env,
                    "concept": concept,
                    "bootstrap_runs": int(n_bootstrap),
                    "bootstrap_selected_count": int(count),
                    "bootstrap_selection_rate": float(count / n_bootstrap),
                    "reason": "ok",
                }
            )
    return pd.DataFrame(rows)


def environment_rf_importance(df: pd.DataFrame, concept_cols: list[str]) -> pd.DataFrame:
    rows = []
    for env, env_df in df.groupby("env_causal_group", dropna=False):
        y = env_df["Y_label"].astype(int)
        n_pos = int(y.sum())
        n_neg = int((y == 0).sum())
        if n_pos < 10 or n_neg < 10:
            for col in concept_cols:
                rows.append(
                    {
                        "environment": env,
                        "concept": concept_name(col),
                        "rf_gini_importance": math.nan,
                        "rf_rank": math.nan,
                        "reason": "too_few_positive_or_negative_samples",
                    }
                )
            continue

        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=500,
                        min_samples_leaf=3,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        x = env_df[concept_cols]
        model.fit(x, y)
        importances = model.named_steps["model"].feature_importances_
        ranked = pd.DataFrame({"concept": [concept_name(c) for c in concept_cols], "rf_gini_importance": importances})
        ranked["rf_rank"] = ranked["rf_gini_importance"].rank(ascending=False, method="min").astype(int)
        ranked["environment"] = env
        ranked["reason"] = "ok"
        rows.extend(ranked.to_dict("records"))
    return pd.DataFrame(rows)


def environment_sparse_logistic(df: pd.DataFrame, concept_cols: list[str]) -> pd.DataFrame:
    rows = []
    for env, env_df in df.groupby("env_causal_group", dropna=False):
        y = env_df["Y_label"].astype(int)
        n_pos = int(y.sum())
        n_neg = int((y == 0).sum())
        if n_pos < 10 or n_neg < 10:
            for col in concept_cols:
                rows.append(
                    {
                        "environment": env,
                        "concept": concept_name(col),
                        "logistic_l1_coefficient": math.nan,
                        "logistic_abs_rank": math.nan,
                        "logistic_selected": False,
                        "reason": "too_few_positive_or_negative_samples",
                    }
                )
            continue

        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        penalty="l1",
                        C=0.25,
                        solver="liblinear",
                        class_weight="balanced",
                        max_iter=5000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )
        x = env_df[concept_cols]
        model.fit(x, y)
        coefficients = model.named_steps["model"].coef_[0]
        ranked = pd.DataFrame(
            {
                "concept": [concept_name(c) for c in concept_cols],
                "logistic_l1_coefficient": coefficients,
            }
        )
        ranked["logistic_abs_coefficient"] = ranked["logistic_l1_coefficient"].abs()
        ranked["logistic_abs_rank"] = ranked["logistic_abs_coefficient"].rank(ascending=False, method="min").astype(int)
        ranked["logistic_selected"] = ranked["logistic_abs_coefficient"] > 1e-9
        ranked["environment"] = env
        ranked["reason"] = "ok"
        rows.extend(ranked.to_dict("records"))
    return pd.DataFrame(rows)


def interpret_signal(stable_score: float, direction_consistent: bool, groups: str) -> str:
    if stable_score >= 1.0 and direction_consistent:
        return "cross_environment_stable_signal"
    if stable_score >= 2 / 3 and direction_consistent:
        return "mostly_stable_signal"
    if stable_score > 0 and not direction_consistent:
        return "environment_sensitive_or_direction_conflict"
    if stable_score > 0:
        return "environment_specific_signal"
    return "not_selected_in_main_environments"


def stable_signals(
    effects: pd.DataFrame,
    boot: pd.DataFrame,
    rf: pd.DataFrame,
    logistic: pd.DataFrame,
    concept_meta: pd.DataFrame,
    selection_rate_threshold: float,
    auc_threshold: float,
    rf_top_k: int,
) -> pd.DataFrame:
    merged = effects.merge(boot, on=["environment", "concept"], how="left").merge(
        rf[["environment", "concept", "rf_gini_importance", "rf_rank", "reason"]],
        on=["environment", "concept"],
        how="left",
        suffixes=("", "_rf"),
    )
    merged = merged.merge(
        logistic[["environment", "concept", "logistic_l1_coefficient", "logistic_abs_rank", "logistic_selected"]],
        on=["environment", "concept"],
        how="left",
    )
    merged["selected_by_bootstrap"] = merged["bootstrap_selection_rate"] >= selection_rate_threshold
    merged["selected_by_auc"] = merged["auc_effect_strength"] >= auc_threshold
    merged["selected_by_rf"] = merged["rf_rank"] <= rf_top_k
    merged["selected_by_sparse_logistic"] = merged["logistic_selected"] == True  # noqa: E712
    merged["environment_selected"] = (
        merged["selected_by_bootstrap"].fillna(False)
        | merged["selected_by_auc"].fillna(False)
        | merged["selected_by_rf"].fillna(False)
        | merged["selected_by_sparse_logistic"].fillna(False)
    )

    rows = []
    main = merged[merged["environment"].isin(MAIN_ENVIRONMENTS)].copy()
    for concept, part in main.groupby("concept", sort=True):
        selected_envs = part.loc[part["environment_selected"], "environment"].tolist()
        selected_count = len(selected_envs)
        stable_score = selected_count / len(MAIN_ENVIRONMENTS)
        selected_part = part[part["environment_selected"]]
        directions = selected_part["direction"].dropna().unique().tolist()
        direction_consistent = len(directions) <= 1 if selected_count else False
        mean_auc_strength = float(part["auc_effect_strength"].mean(skipna=True))
        mean_bootstrap_rate = float(part["bootstrap_selection_rate"].mean(skipna=True))
        mean_rf_rank = float(part["rf_rank"].mean(skipna=True))
        mean_logistic_abs_rank = float(part["logistic_abs_rank"].mean(skipna=True))
        rows.append(
            {
                "concept": concept,
                "selected_environments": ";".join(selected_envs),
                "stable_score": float(stable_score),
                "selected_environment_count": int(selected_count),
                "direction_consistent": bool(direction_consistent),
                "dominant_direction": directions[0] if direction_consistent and directions else "mixed_or_not_selected",
                "mean_auc_effect_strength": mean_auc_strength,
                "mean_bootstrap_selection_rate": mean_bootstrap_rate,
                "mean_rf_rank": mean_rf_rank,
                "mean_logistic_abs_rank": mean_logistic_abs_rank,
                "interpretation": interpret_signal(stable_score, direction_consistent, ";".join(selected_envs)),
            }
        )

    out = pd.DataFrame(rows).merge(concept_meta, on="concept", how="left")
    for env in MAIN_ENVIRONMENTS:
        env_part = merged[merged["environment"] == env][
            [
                "concept",
                "environment_selected",
                "auc_effect_strength",
                "bootstrap_selection_rate",
                "rf_rank",
                "logistic_l1_coefficient",
                "logistic_abs_rank",
                "direction",
            ]
        ].rename(
            columns={
                "environment_selected": f"{env}_selected",
                "auc_effect_strength": f"{env}_auc_effect_strength",
                "bootstrap_selection_rate": f"{env}_bootstrap_selection_rate",
                "rf_rank": f"{env}_rf_rank",
                "logistic_l1_coefficient": f"{env}_logistic_l1_coefficient",
                "logistic_abs_rank": f"{env}_logistic_abs_rank",
                "direction": f"{env}_direction",
            }
        )
        out = out.merge(env_part, on="concept", how="left")

    front = [
        "concept",
        "concept_group",
        "role",
        "stable_score",
        "selected_environment_count",
        "selected_environments",
        "direction_consistent",
        "dominant_direction",
        "interpretation",
    ]
    out = out[front + [c for c in out.columns if c not in front]]
    return out.sort_values(
        ["stable_score", "direction_consistent", "mean_auc_effect_strength", "mean_bootstrap_selection_rate"],
        ascending=[False, False, False, False],
    )


def write_report(
    path: Path,
    df: pd.DataFrame,
    effects: pd.DataFrame,
    stable: pd.DataFrame,
    candidates: pd.DataFrame,
    selection_rate_threshold: float,
    auc_threshold: float,
    rf_top_k: int,
) -> None:
    env_counts = df.groupby(["env_causal_group", "Y_label"]).size().unstack(fill_value=0)
    top_cols = [
        "concept",
        "concept_group",
        "stable_score",
        "selected_environments",
        "direction_consistent",
        "interpretation",
    ]
    lines = [
        "Environment stability analysis report",
        f"Rows: {len(df)}",
        f"Concept features: {sum(c.startswith('concept_') for c in df.columns)}",
        "",
        "Environment label counts:",
        env_counts.to_string(),
        "",
        "Selection rules:",
        f"- selected_by_bootstrap: bootstrap_selection_rate >= {selection_rate_threshold}",
        f"- selected_by_auc: auc_effect_strength >= {auc_threshold}",
        f"- selected_by_rf: environment RF rank <= {rf_top_k}",
        "- selected_by_sparse_logistic: nonzero L1 logistic coefficient",
        "- environment_selected: any of the above is true",
        "- stable_score: selected main environments / 3",
        "",
        "Top stable signals:",
        stable[top_cols].head(30).to_string(index=False),
        "",
        "Recommended candidate concepts for causal graph discovery:",
        candidates[top_cols].to_string(index=False),
        "",
        "Top per-environment univariate effects:",
    ]
    for env in MAIN_ENVIRONMENTS + REFERENCE_ENVIRONMENTS:
        part = effects[effects["environment"] == env]
        lines.append("")
        lines.append(f"[{env}]")
        cols = ["concept", "concept_group", "auc_effect_strength", "univariate_auc", "cohen_d", "direction"]
        lines.append(part[cols].head(15).round(4).to_string(index=False))
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze concept signal stability across climate/weathering environments.")
    parser.add_argument("--bootstrap", type=int, default=200, help="Bootstrap runs per eligible environment.")
    parser.add_argument("--top-k", type=int, default=12, help="Top concepts selected in each bootstrap run.")
    parser.add_argument("--selection-rate-threshold", type=float, default=0.50)
    parser.add_argument("--auc-threshold", type=float, default=0.30, help="AUC effect strength threshold. 0.30 means AUC >= 0.65 or <= 0.35.")
    parser.add_argument("--rf-top-k", type=int, default=12)
    args = parser.parse_args()

    config = load_config()
    ensure_project_dirs(config)
    dataset_path = output_path(config, "outputs", INPUT_DATASET)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Missing concept feature dataset: {dataset_path}. Run scripts/stage_05_causal_graph/16_build_concept_features.py first.")

    df = read_table(dataset_path)
    concept_cols = sorted([c for c in df.columns if c.startswith("concept_")])
    if not concept_cols:
        raise ValueError("No concept_* feature columns found.")
    if "env_causal_group" not in df.columns:
        raise ValueError("Missing env_causal_group column.")

    concept_meta = load_concept_metadata(config, concept_cols)
    rng = np.random.default_rng(RANDOM_STATE)

    effects = feature_effects(df, concept_cols, concept_meta)
    boot = bootstrap_selection(df, concept_cols, args.bootstrap, args.top_k, rng)
    rf = environment_rf_importance(df, concept_cols)
    logistic = environment_sparse_logistic(df, concept_cols)
    stable = stable_signals(
        effects,
        boot,
        rf,
        logistic,
        concept_meta,
        args.selection_rate_threshold,
        args.auc_threshold,
        args.rf_top_k,
    )

    effects_path = output_path(config, "outputs", f"{OUTPUT_DIR}/environment_feature_effects.csv")
    boot_path = output_path(config, "outputs", f"{OUTPUT_DIR}/environment_bootstrap_selection.csv")
    rf_path = output_path(config, "outputs", f"{OUTPUT_DIR}/environment_rf_importance.csv")
    logistic_path = output_path(config, "outputs", f"{OUTPUT_DIR}/environment_sparse_logistic.csv")
    stable_path = output_path(config, "outputs", f"{OUTPUT_DIR}/environment_stable_signals.csv")
    candidates_path = output_path(config, "outputs", f"{OUTPUT_DIR}/causal_graph_candidate_concepts.csv")
    report_path = output_path(config, "outputs", f"{OUTPUT_DIR}/environment_stability_report.txt")

    candidates = stable[stable["selected_environment_count"] >= 2].copy()
    candidates["recommended_role_for_step5"] = np.where(
        candidates["direction_consistent"],
        "candidate_predictor_node",
        "candidate_environment_sensitive_node",
    )

    write_dataframe(effects, effects_path)
    write_dataframe(boot, boot_path)
    write_dataframe(rf, rf_path)
    write_dataframe(logistic, logistic_path)
    write_dataframe(stable, stable_path)
    write_dataframe(candidates, candidates_path)
    write_report(report_path, df, effects, stable, candidates, args.selection_rate_threshold, args.auc_threshold, args.rf_top_k)

    summary = {
        "input_dataset": str(dataset_path),
        "rows": int(len(df)),
        "concept_features": int(len(concept_cols)),
        "main_environments": MAIN_ENVIRONMENTS,
        "reference_environments": REFERENCE_ENVIRONMENTS,
        "bootstrap_runs": int(args.bootstrap),
        "bootstrap_top_k": int(args.top_k),
        "selection_rate_threshold": float(args.selection_rate_threshold),
        "auc_threshold": float(args.auc_threshold),
        "rf_top_k": int(args.rf_top_k),
        "environment_counts": df.groupby(["env_causal_group", "Y_label"]).size().unstack(fill_value=0).to_dict(),
        "stable_signal_counts": stable["interpretation"].value_counts().to_dict(),
        "outputs": {
            "effects": str(effects_path),
            "bootstrap_selection": str(boot_path),
            "rf_importance": str(rf_path),
            "sparse_logistic": str(logistic_path),
            "stable_signals": str(stable_path),
            "candidate_concepts": str(candidates_path),
            "report": str(report_path),
        },
    }
    write_json(output_path(config, "logs", "18_environment_stability_analysis_summary.json"), summary)

    print(f"Wrote environment stability analysis for {len(df)} rows and {len(concept_cols)} concept features")
    print(stable_path)
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
