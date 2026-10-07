from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_ROOT = PROJECT_ROOT / "outputs" / "standardized_runs"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "paper_optimization" / "climate_sensitivity_graph"
MIN_PAIR_COUNT = 30


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


def concept_name(column: str) -> str:
    return column.replace("concept_", "", 1)


def concept_column(name: str) -> str:
    return name if name.startswith("concept_") else f"concept_{name}"


def load_dictionary(path: Path) -> pd.DataFrame:
    dictionary = pd.read_csv(path)
    required = {"concept", "concept_group"}
    missing = required - set(dictionary.columns)
    if missing:
        raise ValueError(f"Concept dictionary is missing columns: {sorted(missing)}")
    return dictionary


def concept_groups(dictionary: pd.DataFrame) -> dict[str, str]:
    return dictionary.set_index("concept")["concept_group"].astype(str).to_dict()


def global_climate_correlations(concept_df: pd.DataFrame, dictionary: pd.DataFrame) -> pd.DataFrame:
    groups = concept_groups(dictionary)
    concept_cols = [c for c in concept_df.columns if c.startswith("concept_")]
    climate_cols = [concept_column(c) for c, g in groups.items() if g == "climate" and concept_column(c) in concept_cols]
    target_cols = [
        concept_column(c)
        for c, g in groups.items()
        if g != "climate" and concept_column(c) in concept_cols
    ]
    rows = []
    if not climate_cols or not target_cols:
        return pd.DataFrame()

    selected = concept_df[target_cols + climate_cols]
    pair_counts = selected[target_cols].notna().astype(int).T.dot(selected[climate_cols].notna().astype(int))
    spearman = selected.corr(method="spearman", min_periods=MIN_PAIR_COUNT).loc[target_cols, climate_cols]
    for target in target_cols:
        for climate in climate_cols:
            value = spearman.loc[target, climate]
            if pd.isna(value):
                continue
            rows.append(
                {
                    "climate_concept": concept_name(climate),
                    "target_concept": concept_name(target),
                    "target_group": groups.get(concept_name(target), "unknown"),
                    "global_spearman": float(value),
                    "global_abs_spearman": abs(float(value)),
                    "n_pair": int(pair_counts.loc[target, climate]),
                }
            )
    return pd.DataFrame(rows)


def m4_sensitive_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(
            columns=[
                "climate_concept",
                "target_concept",
                "m4_sensitive_rate",
                "m4_mean_abs_spearman",
                "m4_max_abs_spearman",
                "m4_fold_count",
            ]
        )
    df = pd.read_csv(path)
    required = {
        "concept_feature",
        "best_climate_concept_spearman",
        "spearman_abs_max",
        "is_climate_sensitive",
    }
    if not required.issubset(df.columns):
        return pd.DataFrame()
    temp = df.copy()
    temp["target_concept"] = temp["concept_feature"].map(concept_name)
    temp["climate_concept"] = temp["best_climate_concept_spearman"].map(concept_name)
    grouped = temp.groupby(["climate_concept", "target_concept"], dropna=False)
    out = grouped.agg(
        m4_sensitive_rate=("is_climate_sensitive", "mean"),
        m4_mean_abs_spearman=("spearman_abs_max", "mean"),
        m4_max_abs_spearman=("spearman_abs_max", "max"),
        m4_fold_count=("spearman_abs_max", "count"),
    ).reset_index()
    return out


def causal_edge_summary(path: Path, dictionary: pd.DataFrame) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(
            columns=[
                "climate_concept",
                "target_concept",
                "causal_edge_present",
                "causal_environment_count",
                "causal_stable_edge_score",
                "causal_mean_edge_weight",
                "causal_edge_class",
            ]
        )
    groups = concept_groups(dictionary)
    df = pd.read_csv(path)
    rows = []
    for _, row in df.iterrows():
        source = str(row.get("source", ""))
        target = str(row.get("target", ""))
        source_group = groups.get(source, str(row.get("source_group", "")))
        target_group = groups.get(target, str(row.get("target_group", "")))
        if source_group == "climate" and target_group != "climate":
            climate, non_climate = source, target
        elif target_group == "climate" and source_group != "climate":
            climate, non_climate = target, source
        else:
            continue
        rows.append(
            {
                "climate_concept": climate,
                "target_concept": non_climate,
                "causal_edge_present": True,
                "causal_environment_count": int(row.get("environment_count", 0)),
                "causal_stable_edge_score": float(row.get("stable_edge_score", 0.0)),
                "causal_mean_edge_weight": float(row.get("mean_edge_weight", 0.0)),
                "causal_edge_class": str(row.get("edge_class", "")),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).drop_duplicates(["climate_concept", "target_concept"])


def min_max(series: pd.Series) -> pd.Series:
    series = pd.to_numeric(series, errors="coerce")
    if series.dropna().empty:
        return pd.Series(np.zeros(len(series)), index=series.index)
    lo = float(series.min())
    hi = float(series.max())
    if math.isclose(lo, hi):
        return pd.Series(np.ones(len(series)), index=series.index)
    return (series - lo) / (hi - lo)


def build_graph(run_dir: Path, output_dir: Path, min_combined_score: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    causal_dir = run_dir / "03_causal_graph"
    concept_path = causal_dir / "04_concept_features.parquet"
    dictionary_path = causal_dir / "05_concept_feature_dictionary.csv"
    edge_path = causal_dir / "20_cross_environment_edge_all_summary.csv"
    m4_path = run_dir / "04_concept_climate_decoupling" / "04_m4_fold_local_sensitive_concepts.csv"

    concept_df = read_table(concept_path)
    dictionary = load_dictionary(dictionary_path)

    corr = global_climate_correlations(concept_df, dictionary)
    m4 = m4_sensitive_summary(m4_path)
    edges = causal_edge_summary(edge_path, dictionary)

    graph = corr.merge(m4, on=["climate_concept", "target_concept"], how="left")
    graph = graph.merge(edges, on=["climate_concept", "target_concept"], how="left")
    fill_zero_cols = [
        "m4_sensitive_rate",
        "m4_mean_abs_spearman",
        "m4_max_abs_spearman",
        "m4_fold_count",
        "causal_environment_count",
        "causal_stable_edge_score",
        "causal_mean_edge_weight",
    ]
    for col in fill_zero_cols:
        if col in graph.columns:
            graph[col] = graph[col].fillna(0.0)
    graph["causal_edge_present"] = graph.get("causal_edge_present", False).fillna(False).astype(bool)
    graph["causal_edge_class"] = graph.get("causal_edge_class", "").fillna("")

    graph["global_score_norm"] = min_max(graph["global_abs_spearman"])
    graph["m4_score_norm"] = min_max(graph["m4_sensitive_rate"])
    graph["causal_score_norm"] = min_max(graph["causal_stable_edge_score"])
    graph["combined_climate_sensitivity_score"] = (
        0.45 * graph["global_score_norm"]
        + 0.35 * graph["m4_score_norm"]
        + 0.20 * graph["causal_score_norm"]
    )
    graph["selected_for_graph_guided_m4"] = graph["combined_climate_sensitivity_score"] >= min_combined_score
    graph = graph.sort_values("combined_climate_sensitivity_score", ascending=False).reset_index(drop=True)

    target_summary = graph.groupby(["target_concept", "target_group"], dropna=False).agg(
        best_climate_concept=("climate_concept", lambda x: graph.loc[x.index[0], "climate_concept"]),
        max_global_abs_spearman=("global_abs_spearman", "max"),
        max_m4_sensitive_rate=("m4_sensitive_rate", "max"),
        max_causal_stable_edge_score=("causal_stable_edge_score", "max"),
        max_combined_climate_sensitivity_score=("combined_climate_sensitivity_score", "max"),
        selected_for_graph_guided_m4=("selected_for_graph_guided_m4", "max"),
    ).reset_index()
    target_summary = target_summary.sort_values(
        "max_combined_climate_sensitivity_score",
        ascending=False,
    ).reset_index(drop=True)

    write_table(graph, output_dir / "climate_sensitivity_edges.csv")
    write_table(target_summary, output_dir / "climate_sensitive_target_concepts.csv")
    write_mermaid(graph, output_dir / "climate_sensitivity_graph_mermaid.md")
    write_report(output_dir, run_dir, graph, target_summary, min_combined_score)
    return graph, target_summary


def write_mermaid(graph: pd.DataFrame, path: Path, max_edges: int = 24) -> None:
    selected = graph[graph["selected_for_graph_guided_m4"]].head(max_edges)
    if selected.empty:
        selected = graph.head(max_edges)
    lines = [
        "```mermaid",
        "flowchart LR",
        '  C["气候概念"]',
        '  T["被影响的找矿观测概念"]',
    ]
    for _, row in selected.iterrows():
        climate = str(row["climate_concept"])
        target = str(row["target_concept"])
        score = float(row["combined_climate_sensitivity_score"])
        c_id = "c_" + "".join(ch if ch.isalnum() else "_" for ch in climate)
        t_id = "t_" + "".join(ch if ch.isalnum() else "_" for ch in target)
        lines.append(f'  {c_id}["{climate}"]')
        lines.append(f'  {t_id}["{target}"]')
        lines.append(f'  {c_id} -->|score {score:.2f}| {t_id}')
    lines.append("```")
    path.write_text("\n".join(lines), encoding="utf-8")


def markdown_table(df: pd.DataFrame, max_rows: int = 30) -> str:
    if df.empty:
        return "_No data._"
    show = df.head(max_rows).copy()
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


def write_report(
    output_dir: Path,
    run_dir: Path,
    graph: pd.DataFrame,
    target_summary: pd.DataFrame,
    min_combined_score: float,
) -> None:
    selected = graph[graph["selected_for_graph_guided_m4"]].copy()
    report = [
        "# Climate Sensitivity Graph",
        "",
        "## Purpose",
        "",
        "This graph is built for climate-decoupling experiments. It is not a strict causal graph by itself; it combines climate-concept correlation, fold-local M4 sensitivity, and climate-related stable graph edges.",
        "",
        "## Input Run",
        "",
        f"`{run_dir}`",
        "",
        "## Scoring",
        "",
        "`combined score = 0.45 * global Spearman rank + 0.35 * M4 fold sensitivity + 0.20 * stable causal-edge score`",
        "",
        f"Selected targets use `combined score >= {min_combined_score}`.",
        "",
        "## Selected Climate-Sensitive Targets",
        "",
        markdown_table(
            target_summary[target_summary["selected_for_graph_guided_m4"]][
                [
                    "target_concept",
                    "target_group",
                    "best_climate_concept",
                    "max_global_abs_spearman",
                    "max_m4_sensitive_rate",
                    "max_causal_stable_edge_score",
                    "max_combined_climate_sensitivity_score",
                ]
            ],
            max_rows=60,
        ),
        "",
        "## Top Edges",
        "",
        markdown_table(
            selected[
                [
                    "climate_concept",
                    "target_concept",
                    "target_group",
                    "global_spearman",
                    "m4_sensitive_rate",
                    "causal_stable_edge_score",
                    "combined_climate_sensitivity_score",
                ]
            ],
            max_rows=60,
        ),
    ]
    (output_dir / "climate_sensitivity_graph_report.md").write_text("\n".join(report), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a climate sensitivity graph from concept and M4 outputs.")
    parser.add_argument("--run-dir", required=True, help="Standardized run directory.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--min-combined-score", type=float, default=0.50)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_ROOT / run_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)
    graph, target_summary = build_graph(run_dir, output_dir, args.min_combined_score)
    manifest = {
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "min_combined_score": args.min_combined_score,
        "n_edges": int(len(graph)),
        "n_selected_edges": int(graph["selected_for_graph_guided_m4"].sum()),
        "n_selected_targets": int(target_summary["selected_for_graph_guided_m4"].sum()),
    }
    (output_dir / "climate_sensitivity_graph_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote climate sensitivity graph to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
