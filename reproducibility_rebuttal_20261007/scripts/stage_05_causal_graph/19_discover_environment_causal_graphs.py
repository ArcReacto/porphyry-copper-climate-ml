from __future__ import annotations

import argparse
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.io_utils import ensure_project_dirs, load_config, output_path, read_table, write_dataframe, write_json


warnings.filterwarnings("ignore", message="'penalty' was deprecated.*")
warnings.filterwarnings("ignore", message="Inconsistent values: penalty=l1.*")

INPUT_DATASET = "causal_graph/concept_features_western_core.parquet"
CANDIDATE_CONCEPTS = "causal_graph/causal_graph_candidate_concepts.csv"
OUTPUT_DIR = "causal_graph"
RANDOM_STATE = 20260622
MAIN_ENVIRONMENTS = [
    "arid_basin_or_range",
    "semi_arid_transition",
    "snow_influenced_mountain",
]
MIN_POSITIVE = 10
MIN_NEGATIVE = 10
TARGET_NODE = "Y_label"

GROUP_TIER = {
    "climate": 0,
    "terrain": 0,
    "geology": 0,
    "geophysics": 0,
    "structure": 0,
    "geochemistry": 1,
    "target": 2,
    "unknown": 1,
}


def concept_to_feature(concept: str) -> str:
    return f"concept_{concept}"


def feature_to_concept(feature: str) -> str:
    return feature.removeprefix("concept_")


def pearson_safe(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    if len(a) < 3 or np.nanstd(a) <= 1e-12 or np.nanstd(b) <= 1e-12:
        return math.nan, math.nan
    corr, p_value = stats.pearsonr(a, b)
    return float(corr), float(p_value)


def ridge_partial_corr(x: np.ndarray, y: np.ndarray, controls: np.ndarray | None) -> tuple[float, float]:
    if controls is None or controls.shape[1] == 0:
        return pearson_safe(x, y)
    if len(x) <= controls.shape[1] + 3:
        return math.nan, math.nan

    ridge_x = Ridge(alpha=1.0)
    ridge_y = Ridge(alpha=1.0)
    ridge_x.fit(controls, x)
    ridge_y.fit(controls, y)
    x_resid = x - ridge_x.predict(controls)
    y_resid = y - ridge_y.predict(controls)
    corr, _ = pearson_safe(x_resid, y_resid)
    if math.isnan(corr):
        return math.nan, math.nan

    # Approximate p-value using effective degrees of freedom. The ridge step makes
    # this a screening statistic rather than a formal hypothesis test.
    df = max(len(x) - controls.shape[1] - 2, 1)
    denom = max(1.0 - corr**2, 1e-12)
    t_stat = corr * math.sqrt(df / denom)
    p_value = 2 * stats.t.sf(abs(t_stat), df)
    return float(corr), float(p_value)


def node_group(node: str, metadata: dict[str, dict]) -> str:
    if node == TARGET_NODE:
        return "target"
    return metadata.get(node, {}).get("concept_group", "unknown")


def orient_edge(node_a: str, node_b: str, metadata: dict[str, dict]) -> tuple[str, str, str, bool]:
    if node_a == TARGET_NODE and node_b != TARGET_NODE:
        return node_b, node_a, "target_sink", True
    if node_b == TARGET_NODE and node_a != TARGET_NODE:
        return node_a, node_b, "target_sink", True

    group_a = node_group(node_a, metadata)
    group_b = node_group(node_b, metadata)
    tier_a = GROUP_TIER.get(group_a, GROUP_TIER["unknown"])
    tier_b = GROUP_TIER.get(group_b, GROUP_TIER["unknown"])
    if tier_a < tier_b:
        return node_a, node_b, "domain_tier", True
    if tier_b < tier_a:
        return node_b, node_a, "domain_tier", True
    return node_a, node_b, "same_tier_undirected_association", False


def interpretation_for_edge(source: str, target: str, source_group: str, target_group: str, directed: bool) -> str:
    if target == TARGET_NODE:
        return "candidate_direct_signal_to_porphyry_label"
    if source_group in {"climate", "terrain"} and target_group == "geochemistry":
        return "candidate_environment_or_exposure_effect_on_geochemical_expression"
    if source_group in {"geology", "geophysics", "structure"} and target_group == "geochemistry":
        return "candidate_geologic_control_on_geochemical_expression"
    if not directed:
        return "same_tier_statistical_association"
    return "candidate_directed_association_by_domain_tier"


def fit_sparse_logistic(env_df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    y = env_df[TARGET_NODE].astype(int)
    if int(y.sum()) < MIN_POSITIVE or int((y == 0).sum()) < MIN_NEGATIVE:
        return pd.DataFrame()
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
    model.fit(env_df[feature_cols], y)
    coefs = model.named_steps["model"].coef_[0]
    out = pd.DataFrame(
        {
            "concept": [feature_to_concept(c) for c in feature_cols],
            "logistic_l1_coefficient": coefs,
            "logistic_abs_coefficient": np.abs(coefs),
        }
    )
    out["logistic_selected"] = out["logistic_abs_coefficient"] > 1e-9
    return out


def discover_environment_edges(
    env: str,
    env_df: pd.DataFrame,
    candidate_table: pd.DataFrame,
    partial_threshold: float,
    p_threshold: float,
    max_edges: int,
) -> pd.DataFrame:
    concepts = candidate_table["concept"].tolist()
    feature_cols = [concept_to_feature(c) for c in concepts if concept_to_feature(c) in env_df.columns]
    nodes = [feature_to_concept(c) for c in feature_cols] + [TARGET_NODE]
    x_raw = env_df[feature_cols + [TARGET_NODE]].copy()
    x_raw[TARGET_NODE] = env_df[TARGET_NODE].astype(float)

    imputed = SimpleImputer(strategy="median").fit_transform(x_raw)
    scaled = StandardScaler().fit_transform(imputed)
    data = pd.DataFrame(scaled, columns=nodes, index=env_df.index)
    raw_target = env_df[TARGET_NODE].astype(int).to_numpy()
    n_pos = int(raw_target.sum())
    n_neg = int((raw_target == 0).sum())

    metadata = candidate_table.set_index("concept").to_dict("index")
    logistic = fit_sparse_logistic(env_df, feature_cols)
    logistic_map = logistic.set_index("concept").to_dict("index") if not logistic.empty else {}

    rows = []
    for i, node_a in enumerate(nodes):
        for node_b in nodes[i + 1 :]:
            a = data[node_a].to_numpy()
            b = data[node_b].to_numpy()
            controls = data[[n for n in nodes if n not in {node_a, node_b}]].to_numpy()
            corr, corr_p = pearson_safe(a, b)
            partial_corr, partial_p = ridge_partial_corr(a, b, controls)

            involves_target = node_a == TARGET_NODE or node_b == TARGET_NODE
            feature_node = node_b if node_a == TARGET_NODE else node_a if node_b == TARGET_NODE else None
            logistic_item = logistic_map.get(feature_node, {}) if feature_node else {}
            logistic_selected = bool(logistic_item.get("logistic_selected", False))
            logistic_coef = float(logistic_item.get("logistic_l1_coefficient", math.nan))

            selected_by_partial = (
                not math.isnan(partial_corr)
                and abs(partial_corr) >= partial_threshold
                and (math.isnan(partial_p) or partial_p <= p_threshold)
            )
            selected_by_target_logistic = involves_target and logistic_selected
            if not (selected_by_partial or selected_by_target_logistic):
                continue

            source, target, direction_rule, directed = orient_edge(node_a, node_b, metadata)
            source_group = node_group(source, metadata)
            target_group = node_group(target, metadata)
            source_stability = metadata.get(source, {}).get("stable_score", math.nan)
            target_stability = metadata.get(target, {}).get("stable_score", math.nan)
            if source == TARGET_NODE:
                source_stability = math.nan
            if target == TARGET_NODE:
                target_stability = 1.0

            logistic_bonus = 0.10 if selected_by_target_logistic else 0.0
            p_bonus = 0.05 if not math.isnan(partial_p) and partial_p <= p_threshold else 0.0
            edge_weight = min((abs(partial_corr) if not math.isnan(partial_corr) else 0.0) + logistic_bonus + p_bonus, 1.0)
            rows.append(
                {
                    "environment": env,
                    "source": source,
                    "target": target,
                    "directed": directed,
                    "direction_rule": direction_rule,
                    "source_group": source_group,
                    "target_group": target_group,
                    "method": "ridge_partial_corr_plus_sparse_logistic",
                    "selected_by_partial_corr": bool(selected_by_partial),
                    "selected_by_target_logistic": bool(selected_by_target_logistic),
                    "association_corr": corr,
                    "association_p_value": corr_p,
                    "partial_corr": partial_corr,
                    "partial_p_value_approx": partial_p,
                    "abs_partial_corr": abs(partial_corr) if not math.isnan(partial_corr) else math.nan,
                    "target_logistic_l1_coefficient": logistic_coef,
                    "edge_weight": float(edge_weight),
                    "source_stable_score": source_stability,
                    "target_stable_score": target_stability,
                    "n_rows": int(len(env_df)),
                    "n_positive": n_pos,
                    "n_negative": n_neg,
                    "interpretation": interpretation_for_edge(source, target, source_group, target_group, directed),
                }
            )

    edges = pd.DataFrame(rows)
    if edges.empty:
        return edges
    edges = edges.sort_values(["edge_weight", "abs_partial_corr"], ascending=False).head(max_edges).reset_index(drop=True)
    edges["rank_in_environment"] = np.arange(1, len(edges) + 1)
    return edges


def summarize_overlap(all_edges: pd.DataFrame) -> pd.DataFrame:
    if all_edges.empty:
        return pd.DataFrame()
    tmp = all_edges.copy()
    tmp["edge_key"] = tmp.apply(
        lambda r: f"{r['source']} -> {r['target']}" if r["directed"] else " -- ".join(sorted([r["source"], r["target"]])),
        axis=1,
    )
    grouped = (
        tmp.groupby("edge_key")
        .agg(
            source=("source", "first"),
            target=("target", "first"),
            directed=("directed", "first"),
            environments=("environment", lambda s: ";".join(sorted(s.unique()))),
            environment_count=("environment", "nunique"),
            mean_edge_weight=("edge_weight", "mean"),
            max_edge_weight=("edge_weight", "max"),
            mean_abs_partial_corr=("abs_partial_corr", "mean"),
            interpretation=("interpretation", "first"),
        )
        .reset_index(drop=True)
    )
    return grouped.sort_values(["environment_count", "mean_edge_weight"], ascending=False)


def write_report(path: Path, summaries: list[dict], all_edges: pd.DataFrame, overlap: pd.DataFrame) -> None:
    lines = [
        "Environment causal graph discovery report",
        "",
        "Important interpretation:",
        "- These are candidate causal graphs, not confirmed geological causality.",
        "- Direction is based on data-driven conditional association plus domain-tier constraints.",
        "- Y_label is treated as a sink/result node.",
        "- Same-tier edges are retained as undirected statistical associations.",
        "",
        "Environment summaries:",
    ]
    lines.append(pd.DataFrame(summaries).to_string(index=False))
    lines.append("")
    lines.append("Top candidate edges by environment:")
    for env in MAIN_ENVIRONMENTS:
        part = all_edges[all_edges["environment"] == env].head(20)
        lines.append("")
        lines.append(f"[{env}]")
        if part.empty:
            lines.append("No edges selected.")
        else:
            cols = ["source", "target", "directed", "edge_weight", "partial_corr", "selected_by_target_logistic", "interpretation"]
            lines.append(part[cols].round(4).to_string(index=False))
    lines.append("")
    lines.append("Preliminary cross-environment repeated edges:")
    if overlap.empty:
        lines.append("No repeated edges.")
    else:
        cols = ["source", "target", "directed", "environments", "environment_count", "mean_edge_weight", "interpretation"]
        lines.append(overlap[cols].head(30).round(4).to_string(index=False))
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Discover per-environment candidate causal graphs from concept features.")
    parser.add_argument("--partial-threshold", type=float, default=0.18)
    parser.add_argument("--p-threshold", type=float, default=0.20)
    parser.add_argument("--max-edges-per-environment", type=int, default=45)
    args = parser.parse_args()

    config = load_config()
    ensure_project_dirs(config)
    dataset_path = output_path(config, "outputs", INPUT_DATASET)
    candidate_path = output_path(config, "outputs", CANDIDATE_CONCEPTS)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Missing concept feature dataset: {dataset_path}. Run scripts/stage_05_causal_graph/16_build_concept_features.py first.")
    if not candidate_path.exists():
        raise FileNotFoundError(f"Missing candidate concepts: {candidate_path}. Run scripts/stage_05_causal_graph/18_environment_stability_analysis.py first.")

    df = read_table(dataset_path)
    candidates = read_table(candidate_path)
    candidates = candidates[candidates["selected_environment_count"] >= 2].copy()
    candidates = candidates.sort_values(["stable_score", "direction_consistent", "mean_auc_effect_strength"], ascending=False)

    all_edges = []
    summaries = []
    for env in MAIN_ENVIRONMENTS:
        env_df = df[df["env_causal_group"] == env].copy()
        n_pos = int(env_df[TARGET_NODE].sum())
        n_neg = int((env_df[TARGET_NODE] == 0).sum())
        if n_pos < MIN_POSITIVE or n_neg < MIN_NEGATIVE:
            summaries.append(
                {
                    "environment": env,
                    "rows": int(len(env_df)),
                    "positive": n_pos,
                    "negative": n_neg,
                    "candidate_concepts": int(len(candidates)),
                    "edges": 0,
                    "status": "skipped_too_few_positive_or_negative_samples",
                }
            )
            continue
        edges = discover_environment_edges(
            env,
            env_df,
            candidates,
            args.partial_threshold,
            args.p_threshold,
            args.max_edges_per_environment,
        )
        all_edges.append(edges)
        env_path = output_path(config, "outputs", f"{OUTPUT_DIR}/causal_edges_{env}.csv")
        write_dataframe(edges, env_path)
        summaries.append(
            {
                "environment": env,
                "rows": int(len(env_df)),
                "positive": n_pos,
                "negative": n_neg,
                "candidate_concepts": int(len(candidates)),
                "edges": int(len(edges)),
                "status": "ok",
            }
        )

    all_edges_df = pd.concat(all_edges, ignore_index=True) if all_edges else pd.DataFrame()
    overlap = summarize_overlap(all_edges_df)

    all_edges_path = output_path(config, "outputs", f"{OUTPUT_DIR}/causal_edges_all_environments.csv")
    overlap_path = output_path(config, "outputs", f"{OUTPUT_DIR}/causal_edges_preliminary_overlap.csv")
    report_path = output_path(config, "outputs", f"{OUTPUT_DIR}/causal_graph_discovery_report.txt")
    write_dataframe(all_edges_df, all_edges_path)
    write_dataframe(overlap, overlap_path)
    write_report(report_path, summaries, all_edges_df, overlap)

    summary = {
        "input_dataset": str(dataset_path),
        "candidate_concepts": str(candidate_path),
        "main_environments": MAIN_ENVIRONMENTS,
        "partial_threshold": float(args.partial_threshold),
        "p_threshold": float(args.p_threshold),
        "max_edges_per_environment": int(args.max_edges_per_environment),
        "environment_summaries": summaries,
        "outputs": {
            "all_edges": str(all_edges_path),
            "preliminary_overlap": str(overlap_path),
            "report": str(report_path),
        },
    }
    write_json(output_path(config, "logs", "19_discover_environment_causal_graphs_summary.json"), summary)

    print("Wrote per-environment candidate causal graphs")
    print(all_edges_path)
    print(overlap_path)
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
