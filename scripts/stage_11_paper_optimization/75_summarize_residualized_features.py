from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = "known_mining_neutral_ratio_1_10_supervised_all_features_v1"
DEFAULT_RUN_DIR = PROJECT_ROOT / "outputs" / "standardized_runs" / DEFAULT_DATASET
DEFAULT_FULL_M4_DIR = PROJECT_ROOT / "outputs" / "paper_optimization" / "full_feature_graph_guided_m4" / DEFAULT_DATASET
DEFAULT_GRAPH_DIR = PROJECT_ROOT / "outputs" / "paper_optimization" / "climate_sensitivity_graph" / DEFAULT_DATASET
DEFAULT_SURFACE_DIR = PROJECT_ROOT / "outputs" / "paper_optimization" / "surface_only_m4_sensitivity" / DEFAULT_DATASET
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "final_paper_results" / "residualized_feature_summary" / DEFAULT_DATASET
DEFAULT_DOC_PATH = PROJECT_ROOT / "docs" / "气候关联残差化特征透明度汇总.md"


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def bool_series(series: pd.Series) -> pd.Series:
    if series.empty:
        return series.astype(bool)
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def infer_family(column: str, role: str = "", concept_group: str = "") -> str:
    col = column.lower()
    group = str(concept_group).lower()
    role_l = str(role).lower()
    if role_l == "geochemistry" or col.startswith("geochem"):
        return "geochemistry"
    if col.startswith("terrain_") or group == "terrain":
        return "terrain"
    if col.startswith("gravity_") or "gravity" in col or group == "geophysics":
        return "geophysics"
    if col.startswith("fault_") or group == "structure":
        return "structure"
    if col.startswith("geology_") or group == "geology":
        return "geology"
    if role_l:
        return role_l
    return "other"


def interpretation_note(family: str) -> str:
    notes = {
        "geochemistry": "Surface anomaly; plausible weathering, leaching, transport, or preservation modification.",
        "terrain": "Surface exposure/erosion context; plausible climate-conditioned observation modification.",
        "geophysics": "Deep or regional physical signal; interpret climate link as spatial association, not direct modern climate effect.",
        "structure": "Structural context; interpret climate link cautiously as regional association unless linked to exposure/erosion.",
        "geology": "Stable geological background; interpret climate link as spatial association or sampling/background covariance.",
    }
    return notes.get(family, "Interpret cautiously as a statistical association under the current benchmark.")


def mode_or_blank(values: pd.Series) -> str:
    values = values.dropna().astype(str)
    values = values[values != ""]
    if values.empty:
        return ""
    return Counter(values).most_common(1)[0][0]


def summarize_spearman(selection: pd.DataFrame, role_table: pd.DataFrame) -> pd.DataFrame:
    if selection.empty:
        return pd.DataFrame()
    df = selection.copy()
    if "cv" in df.columns:
        df = df[df["cv"].astype(str).str.lower().eq("groupkfold_state")]
    if "model" in df.columns:
        df = df[df["model"].astype(str).eq("M4_Spearman_RF")]
    if df.empty:
        return pd.DataFrame()

    df["selected_by_spearman"] = bool_series(df["selected_by_spearman"])
    selected = df[df["selected_by_spearman"]].copy()
    if selected.empty:
        return pd.DataFrame()

    role_lookup = role_table.set_index("column")["role"].to_dict() if "column" in role_table.columns else {}
    fold_count = df[["cv", "fold"]].drop_duplicates().shape[0]
    rows = []
    for feature, group in selected.groupby("target_column"):
        family = infer_family(feature, role_lookup.get(feature, ""))
        rows.append(
            {
                "feature_name": feature,
                "feature_family": family,
                "concept": "",
                "concept_group": "",
                "climate_variable_or_concept": mode_or_blank(group["best_climate_column"]),
                "selection_method": "Spearman",
                "correlation_or_edge_weight": float(group["spearman_abs_max"].mean()),
                "max_abs_correlation": float(group["spearman_abs_max"].max()),
                "selected_folds": int(group[["cv", "fold"]].drop_duplicates().shape[0]),
                "total_folds": int(fold_count),
                "selection_frequency": float(group[["cv", "fold"]].drop_duplicates().shape[0] / fold_count) if fold_count else np.nan,
                "residualized_in_original_graph_m4": False,
                "residualized_in_surface_only_m4": family in {"geochemistry", "terrain"},
                "interpretation_note": interpretation_note(family),
            }
        )
    return pd.DataFrame(rows)


def summarize_graph(
    graph_features: pd.DataFrame,
    graph_edges: pd.DataFrame,
    target_concepts: pd.DataFrame,
    surface_features: pd.DataFrame,
) -> pd.DataFrame:
    if graph_features.empty:
        return pd.DataFrame()
    gf = graph_features.copy()
    gf["residualizable"] = bool_series(gf["residualizable"])
    gf = gf[gf["residualizable"]].copy()
    if gf.empty:
        return pd.DataFrame()

    surface_lookup = {}
    if not surface_features.empty and "column" in surface_features.columns:
        sf = surface_features.copy()
        sf["residualizable"] = bool_series(sf["residualizable"])
        surface_lookup = sf.set_index("column")["residualizable"].to_dict()

    concept_lookup = {}
    if not target_concepts.empty:
        concept_lookup = target_concepts.set_index("target_concept").to_dict(orient="index")

    best_edge_lookup = {}
    if not graph_edges.empty:
        ge = graph_edges.copy()
        ge["selected_for_graph_guided_m4"] = bool_series(ge["selected_for_graph_guided_m4"])
        ge = ge[ge["selected_for_graph_guided_m4"]].copy()
        if not ge.empty:
            ge = ge.sort_values("combined_climate_sensitivity_score", ascending=False)
            best_edge_lookup = ge.drop_duplicates("target_concept").set_index("target_concept").to_dict(orient="index")

    rows = []
    for _, row in gf.iterrows():
        feature = str(row["column"])
        concept = str(row["concept"])
        group = str(row["concept_group"])
        family = infer_family(feature, str(row.get("raw_role", "")), group)
        concept_info = concept_lookup.get(concept, {})
        edge_info = best_edge_lookup.get(concept, {})
        rows.append(
            {
                "feature_name": feature,
                "feature_family": family,
                "concept": concept,
                "concept_group": group,
                "climate_variable_or_concept": str(
                    edge_info.get("climate_concept", concept_info.get("best_climate_concept", ""))
                ),
                "selection_method": "GraphGuided",
                "correlation_or_edge_weight": float(
                    edge_info.get(
                        "combined_climate_sensitivity_score",
                        concept_info.get("max_combined_climate_sensitivity_score", np.nan),
                    )
                ),
                "max_abs_correlation": float(
                    edge_info.get("global_abs_spearman", concept_info.get("max_global_abs_spearman", np.nan))
                ),
                "selected_folds": float(concept_info.get("max_m4_sensitive_rate", np.nan)),
                "total_folds": "concept-rate",
                "selection_frequency": float(concept_info.get("max_m4_sensitive_rate", np.nan)),
                "residualized_in_original_graph_m4": True,
                "residualized_in_surface_only_m4": bool(surface_lookup.get(feature, family in {"geochemistry", "terrain"})),
                "interpretation_note": interpretation_note(family),
            }
        )
    return pd.DataFrame(rows)


def build_union_summary(spearman: pd.DataFrame, graph: pd.DataFrame) -> pd.DataFrame:
    frames = [df for df in [spearman, graph] if not df.empty]
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined["residualized_in_graph_union"] = True
    combined = combined.sort_values(
        ["selection_method", "feature_family", "correlation_or_edge_weight", "feature_name"],
        ascending=[True, True, False, True],
    )
    return combined


def concept_summary(graph_edges: pd.DataFrame, target_concepts: pd.DataFrame) -> pd.DataFrame:
    if target_concepts.empty:
        return pd.DataFrame()
    out = target_concepts.copy()
    if "selected_for_graph_guided_m4" in out.columns:
        out["selected_for_graph_guided_m4"] = bool_series(out["selected_for_graph_guided_m4"])
    out = out[out["selected_for_graph_guided_m4"]].copy()
    cols = [
        "target_concept",
        "target_group",
        "best_climate_concept",
        "max_global_abs_spearman",
        "max_m4_sensitive_rate",
        "max_causal_stable_edge_score",
        "max_combined_climate_sensitivity_score",
    ]
    return out[[c for c in cols if c in out.columns]].sort_values("max_combined_climate_sensitivity_score", ascending=False)


def markdown_table(df: pd.DataFrame, columns: list[str], max_rows: int = 20) -> str:
    if df.empty:
        return "_No rows._"
    rows = df[[c for c in columns if c in df.columns]].head(max_rows).copy()
    for col in rows.columns:
        if pd.api.types.is_numeric_dtype(rows[col]):
            rows[col] = rows[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.3f}")
    header = "| " + " | ".join(rows.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(rows.columns)) + " |"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in rows.to_numpy()]
    return "\n".join([header, sep] + body)


def write_report(
    doc_path: Path,
    output_dir: Path,
    feature_summary: pd.DataFrame,
    concept_table: pd.DataFrame,
    parameters: dict,
) -> None:
    family_counts = (
        feature_summary.groupby(["selection_method", "feature_family"]).size().reset_index(name="feature_count")
        if not feature_summary.empty
        else pd.DataFrame()
    )
    surface_counts = (
        feature_summary.groupby(["selection_method", "residualized_in_surface_only_m4"]).size().reset_index(name="feature_count")
        if not feature_summary.empty
        else pd.DataFrame()
    )
    lines = [
        "# 气候关联残差化特征透明度汇总",
        "",
        "本文件回答老师提出的透明度问题：Spearman、GraphGuided、GraphUnion 到底筛选了哪些特征，哪些特征最终进入残差化，以及这些关系应该怎样解释。",
        "",
        "## 当前筛选规则",
        "",
        "- `M4_Spearman`：在每个训练折内，仅对地球化学特征与气候变量计算 Spearman 相关；默认阈值为 `|rho| >= 0.30`，有效样本数至少 `30`。",
        "- `M4_GraphGuided`：使用 Climate Sensitivity Graph 选择 target concept，再映射回原始特征；原始版本允许连续的 `geochemistry` 与 `geo_structure` 特征残差化。",
        "- `M4_GraphUnion`：取 `M4_Spearman` 与 `M4_GraphGuided` 的残差化目标并集。",
        "- `Surface-only M4`：按老师建议新增的敏感性版本，只允许地球化学与 `terrain_` 地形特征残差化，重力、断层、地质背景只作为原始空间背景特征保留。",
        "",
        "## 关键参数",
        "",
        markdown_table(pd.DataFrame([parameters]).T.reset_index().rename(columns={"index": "parameter", 0: "value"}), ["parameter", "value"], 50),
        "",
        "## 筛选特征数量",
        "",
        markdown_table(family_counts, ["selection_method", "feature_family", "feature_count"], 50),
        "",
        "## Surface-only 过滤结果",
        "",
        markdown_table(surface_counts, ["selection_method", "residualized_in_surface_only_m4", "feature_count"], 50),
        "",
        "## GraphGuided 概念级别 Top 结果",
        "",
        markdown_table(
            concept_table,
            [
                "target_concept",
                "target_group",
                "best_climate_concept",
                "max_global_abs_spearman",
                "max_m4_sensitive_rate",
                "max_causal_stable_edge_score",
                "max_combined_climate_sensitivity_score",
            ],
            15,
        ),
        "",
        "## 原始特征级别 Top 结果",
        "",
        markdown_table(
            feature_summary,
            [
                "feature_name",
                "feature_family",
                "climate_variable_or_concept",
                "selection_method",
                "correlation_or_edge_weight",
                "selection_frequency",
                "residualized_in_surface_only_m4",
                "interpretation_note",
            ],
            25,
        ),
        "",
        "## 解释边界",
        "",
        "- 地球化学和地形特征可以解释为较可能受到风化、淋溶、侵蚀、搬运、保存和暴露条件修饰的表生观测。",
        "- 重力、深部地球物理、稳定地质背景不解释为现代气候直接影响；若出现在图中，只解释为气候、地形和区域地质背景共同变化下的稳定空间统计关联。",
        "- 当前指标用于比较同一构建样本协议下不同方法的相对排序能力，不代表实际野外找矿命中率。",
        "",
        "## 输出文件",
        "",
        f"- `{output_dir / 'residualized_feature_summary.csv'}`",
        f"- `{output_dir / 'graph_selected_concept_summary.csv'}`",
        f"- `{output_dir / 'selection_family_counts.csv'}`",
    ]
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize climate-associated residualization targets.")
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--full-m4-dir", default=str(DEFAULT_FULL_M4_DIR))
    parser.add_argument("--graph-dir", default=str(DEFAULT_GRAPH_DIR))
    parser.add_argument("--surface-dir", default=str(DEFAULT_SURFACE_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--doc-path", default=str(DEFAULT_DOC_PATH))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir)
    full_m4_dir = Path(args.full_m4_dir)
    graph_dir = Path(args.graph_dir)
    surface_dir = Path(args.surface_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    role_table = read_csv(run_dir / "00_dataset_profile" / "feature_roles.csv")
    spearman_selection = read_csv(full_m4_dir / "m4_spearman_selection_by_fold.csv")
    graph_features = read_csv(full_m4_dir / "graph_guided_raw_feature_targets.csv")
    surface_features = read_csv(surface_dir / "graph_guided_raw_feature_targets.csv")
    graph_edges = read_csv(graph_dir / "climate_sensitivity_edges.csv")
    target_concepts = read_csv(graph_dir / "climate_sensitive_target_concepts.csv")
    graph_manifest = {}
    manifest_path = graph_dir / "climate_sensitivity_graph_manifest.json"
    if manifest_path.exists():
        graph_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    spearman_summary = summarize_spearman(spearman_selection, role_table)
    graph_summary = summarize_graph(graph_features, graph_edges, target_concepts, surface_features)
    feature_summary = build_union_summary(spearman_summary, graph_summary)
    concept_table = concept_summary(graph_edges, target_concepts)

    feature_summary.to_csv(output_dir / "residualized_feature_summary.csv", index=False, encoding="utf-8-sig")
    spearman_summary.to_csv(output_dir / "spearman_residualized_feature_summary.csv", index=False, encoding="utf-8-sig")
    graph_summary.to_csv(output_dir / "graph_residualized_feature_summary.csv", index=False, encoding="utf-8-sig")
    concept_table.to_csv(output_dir / "graph_selected_concept_summary.csv", index=False, encoding="utf-8-sig")
    if not feature_summary.empty:
        feature_summary.groupby(["selection_method", "feature_family"]).size().reset_index(name="feature_count").to_csv(
            output_dir / "selection_family_counts.csv", index=False, encoding="utf-8-sig"
        )

    parameters = {
        "spearman_threshold": 0.30,
        "min_pair_count": 30,
        "graph_combined_score": "0.45*global_score_norm + 0.35*m4_score_norm + 0.20*causal_score_norm",
        "graph_min_combined_score": graph_manifest.get("min_combined_score", "not recorded"),
        "residual_model": "SimpleImputer(median) + StandardScaler + Ridge(alpha=10.0)",
        "validation_for_main_table": "GroupKFold by state; interpreted as state-grouped validation",
        "surface_only_rule": "Residualize geochemistry and terrain_* only; keep gravity/fault/geology raw",
    }
    (output_dir / "selection_parameters.json").write_text(
        json.dumps(parameters, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(Path(args.doc_path), output_dir, feature_summary, concept_table, parameters)
    print(f"Wrote residualized feature summary to: {output_dir}")
    print(f"Wrote report to: {args.doc_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
