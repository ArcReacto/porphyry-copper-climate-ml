from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = "known_mining_neutral_ratio_1_10_supervised_all_features_v1"
DEFAULT_RUN_ROOT = PROJECT_ROOT / "outputs" / "standardized_runs"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "paper_tables" / "causal_graph"

CLIMATE_GROUP = "climate"
NON_SIGNAL_EDGE_CLASSES = {"within_variable_family_edge"}


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def md_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    view = df.head(max_rows).copy()
    if view.empty:
        return "_No rows._\n"
    columns = list(view.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in view.iterrows():
        vals = []
        for col in columns:
            val = row[col]
            if isinstance(val, float):
                vals.append(f"{val:.4f}")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def concept_label(name: str) -> str:
    return name.replace("concept_", "").replace("_", " ")


def node_id(name: str) -> str:
    cleaned = name.replace("concept_", "")
    return "".join(ch if ch.isalnum() else "_" for ch in cleaned)


def build_concept_dictionary(causal_dir: Path) -> pd.DataFrame:
    dictionary = read_csv(causal_dir / "05_concept_feature_dictionary.csv")
    cols = ["concept", "concept_group", "role", "feature_count", "aggregation_method", "missing_rate"]
    return dictionary[[c for c in cols if c in dictionary.columns]].sort_values(["concept_group", "concept"])


def build_environment_counts(causal_dir: Path) -> pd.DataFrame:
    concepts = read_csv(causal_dir / "04_concept_features.csv")
    cols = ["env_causal_group", "env_causal_group_id", "Y_label"]
    missing = [c for c in cols if c not in concepts.columns]
    if missing:
        return pd.DataFrame(columns=["env_causal_group", "samples", "positive", "negative", "positive_rate"])
    rows = []
    for group, sub in concepts.groupby("env_causal_group", dropna=False):
        rows.append(
            {
                "env_causal_group": group,
                "samples": int(len(sub)),
                "positive": int((sub["Y_label"] == 1).sum()),
                "negative": int((sub["Y_label"] == 0).sum()),
                "positive_rate": float((sub["Y_label"] == 1).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("env_causal_group")


def build_stable_concepts(causal_dir: Path, top_n: int) -> pd.DataFrame:
    stable = read_csv(causal_dir / "14_environment_stable_signals.csv")
    cols = [
        "concept",
        "concept_group",
        "role",
        "stable_score",
        "selected_environment_count",
        "selected_environments",
        "direction_consistent",
        "dominant_direction",
        "mean_auc_effect_strength",
        "mean_bootstrap_selection_rate",
        "interpretation",
    ]
    stable = stable[[c for c in cols if c in stable.columns]].copy()
    return stable.sort_values(
        ["stable_score", "selected_environment_count", "mean_auc_effect_strength"],
        ascending=[False, False, False],
    ).head(top_n)


def build_climate_related_edges(causal_dir: Path, top_n: int) -> pd.DataFrame:
    edges = read_csv(causal_dir / "21_cross_environment_stable_edges.csv")
    climate_mask = (
        (edges["source_group"].eq(CLIMATE_GROUP) & ~edges["target_group"].eq(CLIMATE_GROUP))
        | (edges["target_group"].eq(CLIMATE_GROUP) & ~edges["source_group"].eq(CLIMATE_GROUP))
    )
    edges = edges[climate_mask].copy()
    cols = [
        "source",
        "target",
        "source_group",
        "target_group",
        "edge_class",
        "environments",
        "environment_count",
        "stable_edge_score",
        "mean_edge_weight",
        "mean_abs_partial_corr",
        "partial_corr_sign",
        "interpretation",
    ]
    edges = edges[[c for c in cols if c in edges.columns]].copy()
    return edges.sort_values(
        ["stable_edge_score", "environment_count", "mean_edge_weight"],
        ascending=[False, False, False],
    ).head(top_n)


def build_m4_consistency(run_dir: Path, climate_edges: pd.DataFrame, top_n: int) -> pd.DataFrame:
    m4_path = run_dir / "04_concept_climate_decoupling" / "04_m4_fold_local_sensitive_concepts.csv"
    m4 = read_csv(m4_path)
    m4 = m4[m4["is_climate_sensitive"] == True].copy()  # noqa: E712
    m4["concept"] = m4["concept_feature"].str.replace("^concept_", "", regex=True)
    m4["best_climate_concept"] = m4["best_climate_concept_spearman"].str.replace("^concept_", "", regex=True)
    summary = (
        m4.groupby(["concept", "concept_group", "best_climate_concept"], as_index=False)
        .agg(
            mean_spearman_abs=("spearman_abs_max", "mean"),
            max_spearman_abs=("spearman_abs_max", "max"),
            selected_folds=("fold", "nunique"),
        )
        .sort_values(["selected_folds", "mean_spearman_abs"], ascending=[False, False])
    )

    edge_pairs = set()
    for _, row in climate_edges.iterrows():
        edge_pairs.add((str(row["source"]), str(row["target"])))
        edge_pairs.add((str(row["target"]), str(row["source"])))

    summary["appears_in_stable_climate_edge"] = summary.apply(
        lambda r: (r["best_climate_concept"], r["concept"]) in edge_pairs,
        axis=1,
    )
    return summary.head(top_n)


def build_mermaid(stable_concepts: pd.DataFrame, climate_edges: pd.DataFrame, m4: pd.DataFrame) -> str:
    lines = [
        "```mermaid",
        "flowchart LR",
        "  classDef climate fill:#E8F5FF,stroke:#3498DB,color:#123;",
        "  classDef geochem fill:#FFF1E6,stroke:#E67E22,color:#123;",
        "  classDef geophysics fill:#F1E8FF,stroke:#8E44AD,color:#123;",
        "  classDef terrain fill:#EAF7EA,stroke:#27AE60,color:#123;",
        "  classDef geology fill:#F7F1E3,stroke:#A67C00,color:#123;",
        "  classDef structure fill:#FDEDEC,stroke:#C0392B,color:#123;",
        "  classDef target fill:#FFFBE6,stroke:#B7950B,stroke-width:2px,color:#123;",
        "",
        "  Y[\"Porphyry copper label\"]:::target",
    ]

    group_class = {
        "climate": "climate",
        "geochemistry": "geochem",
        "geophysics": "geophysics",
        "terrain": "terrain",
        "geology": "geology",
        "structure": "structure",
    }

    nodes: dict[str, str] = {}

    def add_node(name: str, group: str | None = None) -> str:
        nid = node_id(name)
        if nid not in nodes:
            cls = group_class.get(str(group), "")
            suffix = f":::{cls}" if cls else ""
            lines.append(f"  {nid}[\"{concept_label(name)}\"]{suffix}")
            nodes[nid] = name
        return nid

    for _, row in stable_concepts.head(10).iterrows():
        nid = add_node(str(row["concept"]), str(row.get("concept_group", "")))
        if str(row.get("concept_group", "")) != CLIMATE_GROUP:
            lines.append(f"  {nid} -.->|stable signal| Y")

    for _, row in climate_edges.head(10).iterrows():
        s = add_node(str(row["source"]), str(row.get("source_group", "")))
        t = add_node(str(row["target"]), str(row.get("target_group", "")))
        weight = row.get("mean_edge_weight", "")
        label = "stable association"
        if isinstance(weight, float):
            label = f"w {weight:.2f}"
        lines.append(f"  {s} ---|{label}| {t}")

    for _, row in m4.head(10).iterrows():
        c = add_node(str(row["best_climate_concept"]), CLIMATE_GROUP)
        x = add_node(str(row["concept"]), str(row.get("concept_group", "")))
        rho = row.get("mean_spearman_abs", "")
        label = "M4 sensitive"
        if isinstance(rho, float):
            label = f"M4 rho {rho:.2f}"
        lines.append(f"  {c} -.->|{label}| {x}")

    lines.append("```")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize causal graph outputs for paper tables and structure diagrams.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--top-n", type=int, default=20)
    args = parser.parse_args()

    run_dir = args.run_root / args.dataset
    causal_dir = run_dir / "03_causal_graph"
    out_dir = args.output_root / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    concept_dictionary = build_concept_dictionary(causal_dir)
    environment_counts = build_environment_counts(causal_dir)
    stable_concepts = build_stable_concepts(causal_dir, args.top_n)
    climate_edges = build_climate_related_edges(causal_dir, args.top_n)
    m4_consistency = build_m4_consistency(run_dir, climate_edges, args.top_n)

    write_csv(concept_dictionary, out_dir / "table1_concept_feature_dictionary.csv")
    write_csv(environment_counts, out_dir / "table2_environment_group_counts.csv")
    write_csv(stable_concepts, out_dir / "table3_stable_exploration_concepts.csv")
    write_csv(climate_edges, out_dir / "table4_climate_related_edges.csv")
    write_csv(m4_consistency, out_dir / "table5_m4_causal_consistency.csv")

    mermaid = build_mermaid(stable_concepts, climate_edges, m4_consistency)
    (out_dir / "causal_graph_structure_mermaid.md").write_text(
        "# Causal Graph Structure Draft\n\n" + mermaid,
        encoding="utf-8",
    )

    summary = [
        f"# Paper Causal Graph Summary: {args.dataset}",
        "",
        "## Table 1. Concept Feature Dictionary",
        md_table(concept_dictionary[["concept", "concept_group", "role", "feature_count", "aggregation_method"]], 15),
        "## Table 2. Environment Group Counts",
        md_table(environment_counts, 10),
        "## Table 3. Stable Exploration Concepts",
        md_table(stable_concepts, 15),
        "## Table 4. Climate-Related Edges",
        md_table(climate_edges, 15),
        "## Table 5. M4-Causal Graph Consistency",
        md_table(m4_consistency, 15),
        "## Structure Diagram Draft",
        mermaid,
    ]
    (out_dir / "causal_graph_paper_summary.md").write_text("\n".join(summary), encoding="utf-8")
    print(f"Wrote paper causal graph tables: {out_dir}")


if __name__ == "__main__":
    main()
