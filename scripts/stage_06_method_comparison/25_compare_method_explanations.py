from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.io_utils import ensure_project_dirs, load_config, output_path, read_table, write_dataframe, write_json


OUTPUT_DIR = "method_comparison"

FULL_IMPORTANCE = "baseline_results/baseline_feature_importance_western_core_all_features_v1.csv"
FEATURE_MAPPING = "causal_graph/feature_to_concept_resolved.csv"
CONCEPT_IMPORTANCE = "causal_graph/concept_baseline_feature_importance_western_core_concept_features_v1.csv"
CORE_IMPORTANCE = "method_comparison/causal_core_baseline_feature_importance.csv"
MPM_WEIGHTS = "method_comparison/mpm_weighted_overlay_feature_weights.csv"
CORE_EDGES = "causal_graph/causal_interpretation_core_edges.csv"
STABLE_EDGES = "causal_graph/cross_environment_stable_edges.csv"

CORE_RELATIONS = [
    ("Cu_anomaly", "stable_signal_for", "Porphyry_Cu_presence"),
    ("Mo_anomaly", "mostly_stable_signal_for", "Porphyry_Cu_presence"),
    ("terrain_relief", "mostly_stable_signal_for", "Porphyry_Cu_presence"),
    ("fault_density", "candidate_modulates", "Cu_anomaly"),
]


def concept_feature_name(concept: str) -> str:
    if concept.startswith("concept_"):
        return concept
    return f"concept_{concept}"


def clean_concept_name(value: str) -> str:
    text = str(value)
    return text.removeprefix("concept_")


def rank_desc(series: pd.Series) -> pd.Series:
    return series.rank(method="min", ascending=False, na_option="bottom").astype("Int64")


def scale_0_1(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    max_value = x.max(skipna=True)
    if pd.isna(max_value) or max_value <= 0:
        return pd.Series(np.zeros(len(x)), index=x.index)
    return (x / max_value).fillna(0.0)


def load_required(config: dict, area: str, name: str) -> pd.DataFrame:
    path = output_path(config, "outputs", name) if area == "outputs" else output_path(config, area, name)
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return read_table(path)


def build_full_importance_by_concept(config: dict) -> pd.DataFrame:
    full = load_required(config, "outputs", FULL_IMPORTANCE)
    mapping = load_required(config, "outputs", FEATURE_MAPPING)
    merged = full.merge(mapping, left_on="feature", right_on="feature_name", how="left")
    mapped = merged[merged["concept"].notna()].copy()
    grouped = (
        mapped.groupby(["concept", "concept_group", "role"], dropna=False)
        .agg(
            full_rf_importance_sum=("rf_gini_importance", "sum"),
            full_rf_importance_max=("rf_gini_importance", "max"),
            full_feature_count=("feature", "count"),
            full_top_feature=("feature", lambda s: s.iloc[0]),
        )
        .reset_index()
    )
    grouped["concept_feature"] = grouped["concept"].map(concept_feature_name)
    return grouped


def load_feature_importance(config: dict, name: str, value_col: str, prefix: str) -> pd.DataFrame:
    df = load_required(config, "outputs", name)
    out = df[["feature", value_col]].copy()
    out["concept"] = out["feature"].map(clean_concept_name)
    out["concept_feature"] = out["concept"].map(concept_feature_name)
    out = out.rename(columns={value_col: f"{prefix}_importance"})
    return out[["concept", "concept_feature", f"{prefix}_importance"]]


def load_mpm_weights(config: dict) -> pd.DataFrame:
    df = load_required(config, "outputs", MPM_WEIGHTS)
    out = df[["feature", "description", "weight"]].copy()
    out["concept"] = out["feature"].map(clean_concept_name)
    out["concept_feature"] = out["concept"].map(concept_feature_name)
    out = out.rename(columns={"weight": "mpm_weight", "description": "mpm_description"})
    return out[["concept", "concept_feature", "mpm_weight", "mpm_description"]]


def load_causal_support(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    edges = load_required(config, "outputs", CORE_EDGES)
    stable = load_required(config, "outputs", STABLE_EDGES)

    causal_rows = []
    for _, row in edges.iterrows():
        for endpoint, role in [("source", "source"), ("target", "target")]:
            node = str(row[endpoint])
            if node in {"Y_label", "Porphyry_Cu_presence"}:
                continue
            causal_rows.append(
                {
                    "concept": clean_concept_name(node),
                    "concept_feature": concept_feature_name(clean_concept_name(node)),
                    "causal_role": role,
                    "causal_relation": row["relation"],
                    "causal_target_or_source": row["target"] if role == "source" else row["source"],
                    "causal_confidence": row["confidence"],
                    "causal_stable_edge_score": row["stable_edge_score"],
                    "causal_mean_edge_weight": row["mean_edge_weight"],
                    "causal_environments": row["environments"],
                    "causal_note": row["note"],
                }
            )
    causal = pd.DataFrame(causal_rows)
    if causal.empty:
        causal_summary = pd.DataFrame()
    else:
        causal_summary = (
            causal.sort_values(["causal_stable_edge_score", "causal_mean_edge_weight"], ascending=False)
            .groupby(["concept", "concept_feature"], as_index=False)
            .agg(
                causal_relation=("causal_relation", lambda s: ";".join(dict.fromkeys(map(str, s)))),
                causal_role=("causal_role", lambda s: ";".join(dict.fromkeys(map(str, s)))),
                causal_confidence=("causal_confidence", lambda s: ";".join(dict.fromkeys(map(str, s)))),
                causal_stable_edge_score=("causal_stable_edge_score", "max"),
                causal_mean_edge_weight=("causal_mean_edge_weight", "max"),
                causal_environments=("causal_environments", lambda s: ";".join(dict.fromkeys(map(str, s)))),
                causal_note=("causal_note", lambda s: " | ".join(dict.fromkeys(map(str, s)))),
            )
        )

    relation_rows = []
    for source, expected_relation, target in CORE_RELATIONS:
        matched = edges[(edges["source"].eq(source)) & (edges["relation"].eq(expected_relation)) & (edges["target"].eq(target))]
        stable_key = f"{source} -> Y_label" if target == "Porphyry_Cu_presence" else f"{source} -> {target}"
        stable_match = stable[stable["edge_key"].eq(stable_key)] if "edge_key" in stable.columns else pd.DataFrame()
        if matched.empty:
            relation_rows.append(
                {
                    "source": source,
                    "expected_relation": expected_relation,
                    "target": target,
                    "supported_by_causal_graph": False,
                    "confidence": "",
                    "stable_edge_score": np.nan,
                    "mean_edge_weight": np.nan,
                    "environments": "",
                    "note": "",
                    "stable_edge_key": stable_key,
                    "stable_edge_found": bool(not stable_match.empty),
                }
            )
        else:
            row = matched.iloc[0]
            relation_rows.append(
                {
                    "source": source,
                    "expected_relation": expected_relation,
                    "target": target,
                    "supported_by_causal_graph": True,
                    "confidence": row["confidence"],
                    "stable_edge_score": row["stable_edge_score"],
                    "mean_edge_weight": row["mean_edge_weight"],
                    "environments": row["environments"],
                    "note": row["note"],
                    "stable_edge_key": stable_key,
                    "stable_edge_found": bool(not stable_match.empty),
                }
            )
    relation_comparison = pd.DataFrame(relation_rows)
    return causal_summary, relation_comparison


def build_feature_comparison(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    full = build_full_importance_by_concept(config)
    concept = load_feature_importance(
        config,
        CONCEPT_IMPORTANCE,
        "permutation_importance_mean",
        "concept_rf_permutation",
    )
    core = load_feature_importance(
        config,
        CORE_IMPORTANCE,
        "permutation_importance_mean",
        "core_rf_permutation",
    )
    mpm = load_mpm_weights(config)
    causal, relation_comparison = load_causal_support(config)

    concepts = pd.DataFrame(
        {
            "concept_feature": sorted(
                set(full["concept_feature"])
                | set(concept["concept_feature"])
                | set(core["concept_feature"])
                | set(mpm["concept_feature"])
                | set(causal["concept_feature"] if not causal.empty else [])
            )
        }
    )
    concepts["concept"] = concepts["concept_feature"].map(clean_concept_name)
    out = concepts.merge(full, on=["concept", "concept_feature"], how="left")
    out = out.merge(concept, on=["concept", "concept_feature"], how="left")
    out = out.merge(core, on=["concept", "concept_feature"], how="left")
    out = out.merge(mpm, on=["concept", "concept_feature"], how="left")
    if not causal.empty:
        out = out.merge(causal, on=["concept", "concept_feature"], how="left")

    out["full_rf_importance_sum_scaled"] = scale_0_1(out["full_rf_importance_sum"])
    out["concept_rf_permutation_scaled"] = scale_0_1(out["concept_rf_permutation_importance"])
    out["core_rf_permutation_scaled"] = scale_0_1(out["core_rf_permutation_importance"])
    out["mpm_weight_scaled"] = scale_0_1(out["mpm_weight"])
    out["causal_support_scaled"] = scale_0_1(out["causal_stable_edge_score"])

    out["full_rf_rank"] = rank_desc(out["full_rf_importance_sum"])
    out["concept_rf_rank"] = rank_desc(out["concept_rf_permutation_importance"])
    out["core_rf_rank"] = rank_desc(out["core_rf_permutation_importance"])
    out["mpm_rank"] = rank_desc(out["mpm_weight"])
    out["causal_rank"] = rank_desc(out["causal_stable_edge_score"])

    out["supported_by_full_rf_top10"] = out["full_rf_rank"].le(10).fillna(False)
    out["supported_by_concept_rf_top10"] = out["concept_rf_rank"].le(10).fillna(False)
    out["supported_by_core_rf"] = out["core_rf_permutation_importance"].notna()
    out["supported_by_mpm"] = out["mpm_weight"].fillna(0).gt(0)
    out["supported_by_causal_graph"] = out["causal_stable_edge_score"].notna()
    support_cols = [
        "supported_by_full_rf_top10",
        "supported_by_concept_rf_top10",
        "supported_by_core_rf",
        "supported_by_mpm",
        "supported_by_causal_graph",
    ]
    out["support_count"] = out[support_cols].sum(axis=1)
    out["combined_support_score"] = (
        out["full_rf_importance_sum_scaled"] * 0.20
        + out["concept_rf_permutation_scaled"] * 0.20
        + out["core_rf_permutation_scaled"] * 0.20
        + out["mpm_weight_scaled"] * 0.20
        + out["causal_support_scaled"] * 0.20
    )
    out["agreement_level"] = np.select(
        [
            out["support_count"].ge(4),
            out["support_count"].eq(3),
            out["support_count"].eq(2),
        ],
        ["high", "medium", "low"],
        default="limited",
    )
    return out.sort_values(["support_count", "combined_support_score"], ascending=False), relation_comparison


def markdown_table(df: pd.DataFrame, float_digits: int = 4) -> str:
    if df.empty:
        return "(empty)"
    display = df.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda x: "" if pd.isna(x) else f"{x:.{float_digits}f}")
    cols = list(display.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(str(row[col]).replace("\n", " ") for col in cols) + " |")
    return "\n".join(lines)


def write_report(path: Path, feature_table: pd.DataFrame, relation_table: pd.DataFrame) -> None:
    top_cols = [
        "concept",
        "concept_group",
        "role",
        "agreement_level",
        "support_count",
        "combined_support_score",
        "full_rf_rank",
        "concept_rf_rank",
        "core_rf_rank",
        "mpm_rank",
        "causal_rank",
        "causal_relation",
    ]
    top = feature_table[[c for c in top_cols if c in feature_table.columns]].head(15)
    relation_cols = [
        "source",
        "expected_relation",
        "target",
        "supported_by_causal_graph",
        "confidence",
        "stable_edge_score",
        "mean_edge_weight",
        "environments",
        "note",
    ]
    high_agreement = feature_table[feature_table["agreement_level"].eq("high")]["concept"].tolist()
    medium_agreement = feature_table[feature_table["agreement_level"].eq("medium")]["concept"].tolist()
    lines = [
        "# 对比方案五：解释结果对比",
        "",
        "## 1. 实验目的",
        "",
        "本实验比较不同方法给出的解释是否一致：全特征随机森林、概念级随机森林、核心因果图特征随机森林、传统 MPM 权重，以及跨环境因果图稳定边。",
        "",
        "## 2. 高一致性变量",
        "",
        "、".join(high_agreement) if high_agreement else "无",
        "",
        "## 3. 中等一致性变量",
        "",
        "、".join(medium_agreement) if medium_agreement else "无",
        "",
        "## 4. 变量解释对比表 Top 15",
        "",
        markdown_table(top),
        "",
        "## 5. 核心关系对比",
        "",
        markdown_table(relation_table[[c for c in relation_cols if c in relation_table.columns]]),
        "",
        "## 6. 初步结论",
        "",
        "1. 如果一个变量同时被 ML 重要性、MPM 权重和因果图稳定边支持，说明它不仅有预测贡献，也有较强的解释一致性。",
        "2. `Cu_anomaly` 是当前最一致的核心变量，既是 MPM 主权重，也是 ML 和因果图共同支持的矿化信号。",
        "3. `Mo_anomaly` 和 `terrain_relief` 具有较强辅助解释价值；前者更接近矿化伴生信号，后者更可能反映暴露、剥蚀或可见性。",
        "4. `fault_density` 的直接预测权重不一定最高，但在因果图中作为调制 `Cu_anomaly` 的候选关系值得保留。",
        "5. 这一步仍是解释一致性比较，不是因果证明。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    config = load_config()
    ensure_project_dirs(config)
    feature_table, relation_table = build_feature_comparison(config)

    feature_path = output_path(config, "outputs", f"{OUTPUT_DIR}/method_feature_rank_comparison.csv")
    relation_path = output_path(config, "outputs", f"{OUTPUT_DIR}/method_relation_comparison.csv")
    report_path = output_path(config, "outputs", f"{OUTPUT_DIR}/method_explanation_comparison_report.md")

    write_dataframe(feature_table, feature_path)
    write_dataframe(relation_table, relation_path)
    write_report(report_path, feature_table, relation_table)

    run_summary = {
        "outputs": {
            "feature_rank_comparison": str(feature_path),
            "relation_comparison": str(relation_path),
            "report": str(report_path),
        },
        "high_agreement_concepts": feature_table[feature_table["agreement_level"].eq("high")]["concept"].tolist(),
        "medium_agreement_concepts": feature_table[feature_table["agreement_level"].eq("medium")]["concept"].tolist(),
    }
    write_json(output_path(config, "logs", "25_compare_method_explanations_summary.json"), run_summary)

    print(f"Wrote method explanation comparison: {len(feature_table)} concepts")
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
