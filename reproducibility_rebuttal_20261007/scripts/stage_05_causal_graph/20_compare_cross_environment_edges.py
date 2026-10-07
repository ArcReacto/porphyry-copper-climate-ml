from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.io_utils import ensure_project_dirs, load_config, output_path, read_table, write_dataframe, write_json


INPUT_EDGES = "causal_graph/causal_edges_all_environments.csv"
OUTPUT_DIR = "causal_graph"
MAIN_ENVIRONMENTS = [
    "arid_basin_or_range",
    "semi_arid_transition",
    "snow_influenced_mountain",
]
TARGET_NODE = "Y_label"


def edge_key(row: pd.Series) -> str:
    if bool(row["directed"]):
        return f"{row['source']} -> {row['target']}"
    a, b = sorted([str(row["source"]), str(row["target"])])
    return f"{a} -- {b}"


def corr_sign(series: pd.Series) -> str:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return "unknown"
    signs = set(np.sign(values).astype(int).tolist())
    signs.discard(0)
    if len(signs) == 1:
        return "positive" if 1 in signs else "negative"
    if len(signs) == 0:
        return "zero_or_unknown"
    return "mixed"


def edge_class(row: pd.Series) -> str:
    source_group = str(row.get("source_group", "unknown"))
    target_group = str(row.get("target_group", "unknown"))
    target = str(row["target"])
    env_count = int(row["environment_count"])
    sign_consistent = bool(row["partial_corr_sign_consistent"])

    if target == TARGET_NODE and source_group == "geochemistry" and env_count == 3 and sign_consistent:
        return "stable_mineralization_signal_edge"
    if target == TARGET_NODE and env_count >= 2 and sign_consistent:
        return "mostly_stable_direct_signal_edge"
    if target == TARGET_NODE and source_group in {"climate", "terrain"}:
        return "environment_or_exposure_sensitive_target_edge"
    if source_group == target_group:
        return "within_variable_family_edge"
    if target_group == "geochemistry" and source_group in {"structure", "geophysics", "geology", "terrain", "climate"}:
        return "candidate_control_on_geochemical_expression"
    if source_group == "geochemistry" and target_group == "geochemistry":
        return "geochemical_pathfinder_cluster_edge"
    if not bool(row["directed"]):
        return "undirected_cross_environment_association"
    return "other_candidate_edge"


def interpretation(row: pd.Series) -> str:
    cls = row["edge_class"]
    if cls == "stable_mineralization_signal_edge":
        return "跨三个主要环境重复出现、方向一致，是当前最稳定的矿化相关候选边。"
    if cls == "mostly_stable_direct_signal_edge":
        return "在两个以上主要环境中指向 Y_label，可作为较稳定的直接候选信号。"
    if cls == "environment_or_exposure_sensitive_target_edge":
        return "指向 Y_label 但属于气候或地形变量，更可能反映暴露、风化、可观测性或环境调制。"
    if cls == "within_variable_family_edge":
        return "同类变量之间的稳定关联，主要用于理解变量簇结构，不直接解释为成矿因果。"
    if cls == "candidate_control_on_geochemical_expression":
        return "低层级地质或环境变量指向地球化学异常，可作为地表异常表达或地质控制的候选关系。"
    if cls == "geochemical_pathfinder_cluster_edge":
        return "地球化学异常之间的关联，可视为找矿指示元素组合关系。"
    if cls == "undirected_cross_environment_association":
        return "跨环境重复出现的无向统计关联，需要结合领域知识再解释方向。"
    return "候选边，需要后续结合地质知识和稳健性检验解释。"


def relation_for_kg(row: pd.Series) -> str:
    cls = row["edge_class"]
    if cls == "stable_mineralization_signal_edge":
        return "stable_signal_for"
    if cls == "mostly_stable_direct_signal_edge":
        return "mostly_stable_signal_for"
    if cls == "environment_or_exposure_sensitive_target_edge":
        return "environment_sensitive_signal_for"
    if cls == "candidate_control_on_geochemical_expression":
        return "candidate_modulates"
    if cls == "geochemical_pathfinder_cluster_edge":
        return "co_occurs_with"
    if cls == "within_variable_family_edge":
        return "covaries_with"
    return "associated_with"


def build_cross_environment_summary(edges: pd.DataFrame) -> pd.DataFrame:
    work = edges.copy()
    work["edge_key"] = work.apply(edge_key, axis=1)

    grouped = []
    for key, part in work.groupby("edge_key", sort=False):
        first = part.iloc[0]
        envs = sorted(part["environment"].unique().tolist())
        row = {
            "edge_key": key,
            "source": first["source"],
            "target": first["target"],
            "directed": bool(first["directed"]),
            "source_group": first["source_group"],
            "target_group": first["target_group"],
            "direction_rule": first["direction_rule"],
            "environments": ";".join(envs),
            "environment_count": int(len(envs)),
            "stable_edge_score": float(len(envs) / len(MAIN_ENVIRONMENTS)),
            "mean_edge_weight": float(part["edge_weight"].mean()),
            "max_edge_weight": float(part["edge_weight"].max()),
            "mean_abs_partial_corr": float(part["abs_partial_corr"].mean()),
            "partial_corr_sign": corr_sign(part["partial_corr"]),
            "partial_corr_sign_consistent": corr_sign(part["partial_corr"]) in {"positive", "negative"},
            "selected_by_target_logistic_any": bool(part["selected_by_target_logistic"].fillna(False).any()),
            "selected_by_target_logistic_count": int(part["selected_by_target_logistic"].fillna(False).sum()),
            "mean_rank_in_environment": float(part["rank_in_environment"].mean()),
            "min_rank_in_environment": int(part["rank_in_environment"].min()),
            "base_interpretation": first["interpretation"],
        }
        for env in MAIN_ENVIRONMENTS:
            env_part = part[part["environment"] == env]
            row[f"appears_in_{env}"] = not env_part.empty
            row[f"{env}_edge_weight"] = float(env_part["edge_weight"].iloc[0]) if not env_part.empty else np.nan
            row[f"{env}_partial_corr"] = float(env_part["partial_corr"].iloc[0]) if not env_part.empty else np.nan
            row[f"{env}_rank"] = int(env_part["rank_in_environment"].iloc[0]) if not env_part.empty else np.nan
        grouped.append(row)

    out = pd.DataFrame(grouped)
    out["edge_class"] = out.apply(edge_class, axis=1)
    out["interpretation"] = out.apply(interpretation, axis=1)
    out = out.sort_values(
        ["environment_count", "edge_class", "mean_edge_weight", "mean_abs_partial_corr"],
        ascending=[False, True, False, False],
    ).reset_index(drop=True)
    return out


def build_kg_candidates(stable_edges: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in stable_edges.iterrows():
        confidence = "high" if row["stable_edge_score"] >= 1.0 and row["partial_corr_sign_consistent"] else "medium"
        if row["edge_class"] in {"within_variable_family_edge", "undirected_cross_environment_association"}:
            confidence = "context"
        rows.append(
            {
                "source": row["source"],
                "relation": relation_for_kg(row),
                "target": row["target"],
                "edge_class": row["edge_class"],
                "evidence_type": "cross_environment_causal_graph_comparison",
                "environments": row["environments"],
                "stable_edge_score": row["stable_edge_score"],
                "mean_edge_weight": row["mean_edge_weight"],
                "confidence": confidence,
                "note": row["interpretation"],
            }
        )
    return pd.DataFrame(rows)


def write_report(path: Path, all_summary: pd.DataFrame, stable_edges: pd.DataFrame, kg: pd.DataFrame) -> None:
    class_summary = (
        stable_edges.groupby("edge_class", as_index=False)
        .agg(edge_count=("edge_key", "count"), mean_stable_score=("stable_edge_score", "mean"), mean_edge_weight=("mean_edge_weight", "mean"))
        .sort_values(["edge_count", "mean_edge_weight"], ascending=False)
    )

    lines = [
        "Cross-environment stable edge comparison report",
        "",
        "Important interpretation:",
        "- This step compares candidate graph edges across environments.",
        "- Repeated edges are more stable, but still not confirmed geological causality.",
        "- within_variable_family_edge is useful for structure understanding, not direct mineralization interpretation.",
        "",
        f"All candidate edge keys: {len(all_summary)}",
        f"Repeated edges in at least two main environments: {len(stable_edges)}",
        "",
        "Repeated edge classes:",
        class_summary.round(4).to_string(index=False),
        "",
        "Top repeated edges:",
        stable_edges[
            [
                "source",
                "target",
                "directed",
                "edge_class",
                "environments",
                "stable_edge_score",
                "mean_edge_weight",
                "partial_corr_sign",
                "interpretation",
            ]
        ]
        .head(40)
        .round(4)
        .to_string(index=False),
        "",
        "Knowledge-graph-style candidate relations:",
        kg[["source", "relation", "target", "confidence", "environments", "note"]].head(40).round(4).to_string(index=False),
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    config = load_config()
    ensure_project_dirs(config)

    edges_path = output_path(config, "outputs", INPUT_EDGES)
    if not edges_path.exists():
        raise FileNotFoundError(f"Missing causal edges: {edges_path}. Run scripts/stage_05_causal_graph/19_discover_environment_causal_graphs.py first.")

    edges = read_table(edges_path)
    all_summary = build_cross_environment_summary(edges)
    stable_edges = all_summary[all_summary["environment_count"] >= 2].copy()
    kg_candidates = build_kg_candidates(stable_edges)
    class_summary = (
        stable_edges.groupby("edge_class", as_index=False)
        .agg(edge_count=("edge_key", "count"), mean_stable_score=("stable_edge_score", "mean"), mean_edge_weight=("mean_edge_weight", "mean"))
        .sort_values(["edge_count", "mean_edge_weight"], ascending=False)
    )

    all_summary_path = output_path(config, "outputs", f"{OUTPUT_DIR}/cross_environment_edge_all_summary.csv")
    stable_path = output_path(config, "outputs", f"{OUTPUT_DIR}/cross_environment_stable_edges.csv")
    class_summary_path = output_path(config, "outputs", f"{OUTPUT_DIR}/cross_environment_edge_class_summary.csv")
    kg_path = output_path(config, "outputs", f"{OUTPUT_DIR}/causal_interpretation_edge_candidates.csv")
    report_path = output_path(config, "outputs", f"{OUTPUT_DIR}/cross_environment_edge_report.txt")

    write_dataframe(all_summary, all_summary_path)
    write_dataframe(stable_edges, stable_path)
    write_dataframe(class_summary, class_summary_path)
    write_dataframe(kg_candidates, kg_path)
    write_report(report_path, all_summary, stable_edges, kg_candidates)

    summary = {
        "input_edges": str(edges_path),
        "all_edge_keys": int(len(all_summary)),
        "stable_edges_environment_count_ge_2": int(len(stable_edges)),
        "edge_class_counts": stable_edges["edge_class"].value_counts().to_dict(),
        "outputs": {
            "all_summary": str(all_summary_path),
            "stable_edges": str(stable_path),
            "class_summary": str(class_summary_path),
            "kg_candidates": str(kg_path),
            "report": str(report_path),
        },
    }
    write_json(output_path(config, "logs", "20_compare_cross_environment_edges_summary.json"), summary)

    print("Wrote cross-environment stable edge comparison")
    print(stable_path)
    print(kg_path)
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
