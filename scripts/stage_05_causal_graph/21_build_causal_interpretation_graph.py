from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.io_utils import ensure_project_dirs, load_config, output_path, read_table, write_dataframe, write_json


INPUT_RELATIONS = "causal_graph/causal_interpretation_edge_candidates.csv"
INPUT_STABLE_EDGES = "causal_graph/cross_environment_stable_edges.csv"
OUTPUT_DIR = "causal_graph"
TARGET_LABEL = "Porphyry_Cu_presence"

NODE_LABELS = {
    "Y_label": TARGET_LABEL,
    "Cu_anomaly": "Cu anomaly",
    "Mo_anomaly": "Mo anomaly",
    "Ag_anomaly": "Ag anomaly",
    "Pb_Zn_background": "Pb-Zn background",
    "As_Sb_Bi_pathfinder": "As-Sb-Bi pathfinder",
    "W_Re_pathfinder": "W-Re pathfinder",
    "fault_density": "Fault density",
    "fault_proximity": "Fault proximity",
    "terrain_relief": "Terrain relief",
    "terrain_roughness": "Terrain roughness",
    "terrain_slope": "Terrain slope",
    "runoff": "Runoff",
    "aridity": "Aridity",
    "precipitation": "Precipitation",
    "water_balance": "Water balance",
    "water_deficit": "Water deficit",
    "potential_evapotranspiration": "Potential evapotranspiration",
    "actual_evapotranspiration": "Actual evapotranspiration",
    "evapotranspiration_ratio": "Evapotranspiration ratio",
    "snow_influence": "Snow influence",
    "max_temperature": "Max temperature",
    "solar_radiation": "Solar radiation",
    "vapor_pressure_deficit": "Vapor pressure deficit",
}

NODE_TYPES = {
    "Y_label": "target",
    "Cu_anomaly": "geochemistry",
    "Mo_anomaly": "geochemistry",
    "Ag_anomaly": "geochemistry",
    "Pb_Zn_background": "geochemistry",
    "As_Sb_Bi_pathfinder": "geochemistry",
    "W_Re_pathfinder": "geochemistry",
    "fault_density": "structure",
    "fault_proximity": "structure",
    "terrain_relief": "terrain",
    "terrain_roughness": "terrain",
    "terrain_slope": "terrain",
    "runoff": "climate",
    "aridity": "climate",
    "precipitation": "climate",
    "water_balance": "climate",
    "water_deficit": "climate",
    "potential_evapotranspiration": "climate",
    "actual_evapotranspiration": "climate",
    "evapotranspiration_ratio": "climate",
    "snow_influence": "climate",
    "max_temperature": "climate",
    "solar_radiation": "climate",
    "vapor_pressure_deficit": "climate",
}

TYPE_DESCRIPTIONS = {
    "target": "目标节点，表示当前样本是否为斑岩铜正样本。",
    "geochemistry": "地球化学异常或找矿指示元素组合。",
    "structure": "断层接近性、断层密度等构造背景。",
    "terrain": "地形起伏、坡度、粗糙度等暴露或剥蚀条件。",
    "climate": "气候、水分、雪影响等环境背景。",
    "unknown": "暂未归类节点。",
}

RELATION_LABELS = {
    "stable_signal_for": "稳定指示",
    "mostly_stable_signal_for": "较稳定指示",
    "candidate_modulates": "候选调制",
    "covaries_with": "稳定共变",
    "associated_with": "跨环境关联",
}

MERMAID_RELATION_LABELS = {
    "stable_signal_for": "stable signal",
    "mostly_stable_signal_for": "mostly stable",
    "candidate_modulates": "candidate modulates",
    "covaries_with": "covaries",
    "associated_with": "associated",
}


def normalize_node(node: str) -> str:
    return TARGET_LABEL if node == "Y_label" else node


def node_label(node: str) -> str:
    original = "Y_label" if node == TARGET_LABEL else node
    return NODE_LABELS.get(original, node.replace("_", " "))


def node_type(node: str) -> str:
    original = "Y_label" if node == TARGET_LABEL else node
    return NODE_TYPES.get(original, "unknown")


def build_edges(relations: pd.DataFrame, stable_edges: pd.DataFrame) -> pd.DataFrame:
    edges = relations.copy()
    edges["source"] = edges["source"].map(normalize_node)
    edges["target"] = edges["target"].map(normalize_node)
    edges["relation_label_cn"] = edges["relation"].map(RELATION_LABELS).fillna(edges["relation"])
    edges["source_type"] = edges["source"].map(node_type)
    edges["target_type"] = edges["target"].map(node_type)
    edges["edge_id"] = [f"E{i:03d}" for i in range(1, len(edges) + 1)]

    stable = stable_edges.copy()
    stable["source"] = stable["source"].map(normalize_node)
    stable["target"] = stable["target"].map(normalize_node)
    stable_key_cols = [
        "source",
        "target",
        "partial_corr_sign",
        "partial_corr_sign_consistent",
        "edge_class",
    ]
    stable_meta = stable[stable_key_cols].drop_duplicates(["source", "target", "edge_class"])
    edges = edges.merge(stable_meta, on=["source", "target", "edge_class"], how="left")

    front = [
        "edge_id",
        "source",
        "relation",
        "relation_label_cn",
        "target",
        "source_type",
        "target_type",
        "edge_class",
        "confidence",
        "stable_edge_score",
        "mean_edge_weight",
        "environments",
        "evidence_type",
        "partial_corr_sign",
        "partial_corr_sign_consistent",
        "note",
    ]
    return edges[front + [c for c in edges.columns if c not in front]]


def build_nodes(edges: pd.DataFrame) -> pd.DataFrame:
    nodes = sorted(set(edges["source"]).union(edges["target"]))
    rows = []
    for node in nodes:
        outgoing = int((edges["source"] == node).sum())
        incoming = int((edges["target"] == node).sum())
        rows.append(
            {
                "node_id": node,
                "label": node_label(node),
                "node_type": node_type(node),
                "description": TYPE_DESCRIPTIONS.get(node_type(node), TYPE_DESCRIPTIONS["unknown"]),
                "outgoing_edges": outgoing,
                "incoming_edges": incoming,
                "total_edges": outgoing + incoming,
                "is_target": node == TARGET_LABEL,
            }
        )
    return pd.DataFrame(rows).sort_values(["is_target", "node_type", "node_id"], ascending=[False, True, True])


def select_core_edges(edges: pd.DataFrame) -> pd.DataFrame:
    core_classes = {
        "stable_mineralization_signal_edge",
        "mostly_stable_direct_signal_edge",
        "candidate_control_on_geochemical_expression",
    }
    core = edges[edges["edge_class"].isin(core_classes)].copy()
    return core.sort_values(["confidence", "stable_edge_score", "mean_edge_weight"], ascending=[True, False, False])


def mermaid_id(node: str) -> str:
    return (
        node.replace("-", "_")
        .replace(" ", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
        .replace(".", "_")
    )


def mermaid_for_edges(edges: pd.DataFrame) -> str:
    lines = ["flowchart LR"]
    nodes = sorted(set(edges["source"]).union(edges["target"]))
    for node in nodes:
        lines.append(f'  {mermaid_id(node)}["{node_label(node)}"]')
    for _, row in edges.iterrows():
        source = mermaid_id(row["source"])
        target = mermaid_id(row["target"])
        label = MERMAID_RELATION_LABELS.get(row["relation"], row["relation"])
        if row["relation"] == "covaries_with" or row["relation"] == "associated_with":
            lines.append(f"  {source} ---|{label}| {target}")
        else:
            lines.append(f"  {source} -->|{label}| {target}")
    lines.extend(
        [
            "  classDef target fill:#fee2e2,stroke:#991b1b,color:#111827",
            "  classDef geochemistry fill:#fef3c7,stroke:#92400e,color:#111827",
            "  classDef structure fill:#e0e7ff,stroke:#3730a3,color:#111827",
            "  classDef terrain fill:#dcfce7,stroke:#166534,color:#111827",
            "  classDef climate fill:#e0f2fe,stroke:#075985,color:#111827",
        ]
    )
    for node in nodes:
        lines.append(f"  class {mermaid_id(node)} {node_type(node)}")
    return "\n".join(lines) + "\n"


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "(empty)"
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        values = [str(row[col]).replace("\n", " ") for col in cols]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def build_summary_markdown(nodes: pd.DataFrame, edges: pd.DataFrame, core_edges: pd.DataFrame, mermaid: str) -> str:
    relation_counts = edges["relation"].value_counts().rename_axis("relation").reset_index(name="count")
    type_counts = nodes["node_type"].value_counts().rename_axis("node_type").reset_index(name="count")

    lines = [
        "# 因果图步骤7：知识图谱式解释结构",
        "",
        "## 1. 说明",
        "",
        "本文件把步骤6得到的跨环境稳定边整理成知识图谱式解释结构。这里的“知识图谱式”不是最终完整知识图谱，而是把候选因果图结果转成更便于解释和展示的节点-关系-证据表。",
        "",
        "## 2. 节点统计",
        "",
        markdown_table(type_counts),
        "",
        "## 3. 关系统计",
        "",
        markdown_table(relation_counts),
        "",
        "## 4. 核心解释关系",
        "",
        markdown_table(core_edges[["source", "relation", "target", "confidence", "stable_edge_score", "environments", "note"]]),
        "",
        "## 5. 核心图",
        "",
        "```mermaid",
        mermaid.strip(),
        "```",
        "",
        "## 6. 解释边界",
        "",
        "1. `stable_signal_for` 表示跨环境稳定指示，不表示该变量导致矿床形成。",
        "2. `candidate_modulates` 表示候选调制或控制关系，需要后续验证。",
        "3. `covaries_with` 主要用于描述变量簇内部结构，不应直接解释为成矿因果链。",
        "4. `Porphyry_Cu_presence` 是当前样本标签的解释性名称，对应原始字段 `Y_label`。",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    config = load_config()
    ensure_project_dirs(config)

    relation_path = output_path(config, "outputs", INPUT_RELATIONS)
    stable_path = output_path(config, "outputs", INPUT_STABLE_EDGES)
    if not relation_path.exists():
        raise FileNotFoundError(f"Missing relation candidates: {relation_path}. Run scripts/stage_05_causal_graph/20_compare_cross_environment_edges.py first.")
    if not stable_path.exists():
        raise FileNotFoundError(f"Missing stable edges: {stable_path}. Run scripts/stage_05_causal_graph/20_compare_cross_environment_edges.py first.")

    relations = read_table(relation_path)
    stable_edges = read_table(stable_path)

    edges = build_edges(relations, stable_edges)
    nodes = build_nodes(edges)
    core_edges = select_core_edges(edges)
    mermaid = mermaid_for_edges(core_edges)
    summary_md = build_summary_markdown(nodes, edges, core_edges, mermaid)

    nodes_path = output_path(config, "outputs", f"{OUTPUT_DIR}/causal_interpretation_nodes.csv")
    edges_path = output_path(config, "outputs", f"{OUTPUT_DIR}/causal_interpretation_edges.csv")
    core_path = output_path(config, "outputs", f"{OUTPUT_DIR}/causal_interpretation_core_edges.csv")
    mermaid_path = output_path(config, "outputs", f"{OUTPUT_DIR}/causal_interpretation_core_graph.mmd")
    summary_path = output_path(config, "outputs", f"{OUTPUT_DIR}/causal_interpretation_graph_summary.md")

    write_dataframe(nodes, nodes_path)
    write_dataframe(edges, edges_path)
    write_dataframe(core_edges, core_path)
    mermaid_path.write_text(mermaid, encoding="utf-8")
    summary_path.write_text(summary_md, encoding="utf-8")

    summary = {
        "input_relations": str(relation_path),
        "input_stable_edges": str(stable_path),
        "nodes": int(len(nodes)),
        "edges": int(len(edges)),
        "core_edges": int(len(core_edges)),
        "relation_counts": edges["relation"].value_counts().to_dict(),
        "node_type_counts": nodes["node_type"].value_counts().to_dict(),
        "outputs": {
            "nodes": str(nodes_path),
            "edges": str(edges_path),
            "core_edges": str(core_path),
            "mermaid": str(mermaid_path),
            "summary_md": str(summary_path),
        },
    }
    write_json(output_path(config, "logs", "21_build_causal_interpretation_graph_summary.json"), summary)

    print("Wrote causal interpretation graph artifacts")
    print(nodes_path)
    print(edges_path)
    print(core_path)
    print(mermaid_path)
    print(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


