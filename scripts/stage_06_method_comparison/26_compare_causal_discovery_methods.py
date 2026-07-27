from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from causallearn.score import LocalScoreFunction
from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ScoreBased import GES

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.io_utils import ensure_project_dirs, load_config, output_path, read_table, write_dataframe, write_json


OUTPUT_DIR = "method_comparison"
INPUT_DATASET = "causal_graph/concept_features_western_core.parquet"
OUR_STABLE_EDGES = "causal_graph/cross_environment_stable_edges.csv"
OUR_CORE_EDGES = "causal_graph/causal_interpretation_core_edges.csv"

TARGET = "Y_label"
PC_ALPHA = 0.05
MIN_ABS_PARTIAL_CORR = 0.12
GES_MAX_PARENTS = 4

CONCEPTS = [
    "Cu_anomaly",
    "Mo_anomaly",
    "terrain_relief",
    "fault_density",
    "fault_proximity",
    "As_Sb_Bi_pathfinder",
    "Pb_Zn_background",
    "Ag_anomaly",
    "W_Re_pathfinder",
    "snow_influence",
    "water_deficit",
]

EXPECTED_RELATIONS = [
    ("Cu_anomaly", TARGET),
    ("Mo_anomaly", TARGET),
    ("terrain_relief", TARGET),
    ("fault_density", "Cu_anomaly"),
]

CONCEPT_GROUP = {
    "Cu_anomaly": "geochemistry",
    "Mo_anomaly": "geochemistry",
    "As_Sb_Bi_pathfinder": "geochemistry",
    "Pb_Zn_background": "geochemistry",
    "Ag_anomaly": "geochemistry",
    "W_Re_pathfinder": "geochemistry",
    "fault_density": "structure",
    "fault_proximity": "structure",
    "terrain_relief": "terrain",
    "snow_influence": "climate",
    "water_deficit": "climate",
    TARGET: "target",
}

TIER = {
    "structure": 1,
    "terrain": 1,
    "climate": 1,
    "geochemistry": 2,
    "target": 3,
}


def concept_to_column(concept: str) -> str:
    if concept == TARGET:
        return TARGET
    return f"concept_{concept}"


def column_to_concept(column: str) -> str:
    return column.removeprefix("concept_")


def prepare_matrix(df: pd.DataFrame, concepts: list[str]) -> pd.DataFrame:
    columns = [TARGET] + [concept_to_column(c) for c in concepts]
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for PC-like comparison: {missing}")
    raw = df[columns].copy()
    raw = raw.rename(columns={concept_to_column(c): c for c in concepts})
    imputed = SimpleImputer(strategy="median").fit_transform(raw)
    scaled = StandardScaler().fit_transform(imputed)
    return pd.DataFrame(scaled, columns=raw.columns)


def patch_causal_learn_bic_for_numpy2() -> None:
    """causal-learn 0.1.4.7 calls float(1x1 ndarray), which fails on NumPy 2."""

    def patched_bic_score(data, i: int, pai: list[int], parameters=None) -> float:
        if isinstance(data, tuple):
            cov, n = data
        else:
            cov = np.cov(data.T, ddof=0)
            n = data.shape[0]
        lambda_value = 0.5 if parameters is None else parameters.get("lambda_value", 0.5)
        sigma = cov[i, i]
        if len(pai) > 0:
            yx = cov[np.ix_([i], pai)]
            xx = cov[np.ix_(pai, pai)]
            try:
                xx_inv = np.linalg.inv(xx)
            except np.linalg.LinAlgError:
                xx_inv = np.linalg.pinv(xx)
            sigma = (cov[i, i] - yx @ xx_inv @ yx.T).item()
        if sigma <= 0:
            sigma = np.finfo(float).eps
        return float(-0.5 * n * (1 + np.log(sigma)) - lambda_value * (len(pai) + 1) * np.log(n))

    LocalScoreFunction.local_score_BIC = patched_bic_score
    LocalScoreFunction.local_score_BIC_from_cov = patched_bic_score
    GES.local_score_BIC = patched_bic_score
    GES.local_score_BIC_from_cov = patched_bic_score


def residualize(y: np.ndarray, controls: np.ndarray | None) -> np.ndarray:
    if controls is None or controls.shape[1] == 0:
        return y - np.nanmean(y)
    model = LinearRegression()
    model.fit(controls, y)
    return y - model.predict(controls)


def partial_corr(x: np.ndarray, y: np.ndarray, controls: np.ndarray | None) -> tuple[float, float, int]:
    rx = residualize(x, controls)
    ry = residualize(y, controls)
    r, _ = stats.pearsonr(rx, ry)
    k = 0 if controls is None else controls.shape[1]
    dfree = len(x) - k - 2
    if not np.isfinite(r) or dfree <= 0 or abs(r) >= 1:
        return float(r), math.nan, int(dfree)
    t_stat = r * math.sqrt(dfree / max(1e-12, 1 - r * r))
    p_value = 2 * stats.t.sf(abs(t_stat), dfree)
    return float(r), float(p_value), int(dfree)


def orient_edge(a: str, b: str) -> tuple[str, str, bool, str]:
    group_a = CONCEPT_GROUP.get(a, "unknown")
    group_b = CONCEPT_GROUP.get(b, "unknown")
    tier_a = TIER.get(group_a, 99)
    tier_b = TIER.get(group_b, 99)
    if a == TARGET:
        return b, a, True, "target_sink"
    if b == TARGET:
        return a, b, True, "target_sink"
    if tier_a < tier_b:
        return a, b, True, "domain_tier_order"
    if tier_b < tier_a:
        return b, a, True, "domain_tier_order"
    return a, b, False, "same_tier_undirected"


def domain_normalize_edge(a: str, b: str, raw_directed: bool, raw_source: str, raw_target: str) -> tuple[str, str, bool, str]:
    if not raw_directed:
        source, target, directed, rule = orient_edge(a, b)
        return source, target, directed, f"domain_normalized_from_undirected:{rule}"
    source, target, directed, rule = orient_edge(raw_source, raw_target)
    return source, target, directed, f"domain_normalized_from_raw_directed:{rule}"


def edge_to_raw_parts(edge) -> tuple[str, str, bool, str, str, str]:
    node1 = edge.get_node1().get_name()
    node2 = edge.get_node2().get_name()
    ep1 = str(edge.get_endpoint1())
    ep2 = str(edge.get_endpoint2())
    if ep1 == "TAIL" and ep2 == "ARROW":
        return node1, node2, True, node1, node2, f"{node1} -> {node2}"
    if ep1 == "ARROW" and ep2 == "TAIL":
        return node1, node2, True, node2, node1, f"{node2} -> {node1}"
    return node1, node2, False, node1, node2, " -- ".join(sorted([node1, node2]))


def graph_edges_to_dataframe(graph, method: str) -> pd.DataFrame:
    rows = []
    for edge in graph.get_graph_edges():
        node1, node2, raw_directed, raw_source, raw_target, raw_edge_key = edge_to_raw_parts(edge)
        source, target, directed, normalization_rule = domain_normalize_edge(node1, node2, raw_directed, raw_source, raw_target)
        edge_key = f"{source} -> {target}" if directed else " -- ".join(sorted([source, target]))
        rows.append(
            {
                "method": method,
                "edge_key": edge_key,
                "source": source,
                "target": target,
                "directed": directed,
                "raw_edge": str(edge),
                "raw_edge_key": raw_edge_key,
                "raw_source": raw_source,
                "raw_target": raw_target,
                "raw_directed": raw_directed,
                "normalization_rule": normalization_rule,
                "source_group": CONCEPT_GROUP.get(source, "unknown"),
                "target_group": CONCEPT_GROUP.get(target, "unknown"),
            }
        )
    return pd.DataFrame(rows)


def run_standard_pc(matrix: pd.DataFrame) -> pd.DataFrame:
    result = pc(
        matrix.to_numpy(),
        alpha=PC_ALPHA,
        indep_test="fisherz",
        stable=True,
        show_progress=False,
        node_names=list(matrix.columns),
    )
    return graph_edges_to_dataframe(result.G, "standard_pc")


def run_standard_ges(matrix: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    patch_causal_learn_bic_for_numpy2()
    result = GES.ges(
        matrix.to_numpy(),
        score_func="local_score_BIC",
        node_names=list(matrix.columns),
        maxP=GES_MAX_PARENTS,
    )
    edges = graph_edges_to_dataframe(result["G"], "standard_ges_bic")
    return edges, float(result.get("score", np.nan))


def discover_pc_like_edges(matrix: pd.DataFrame) -> pd.DataFrame:
    rows = []
    variables = list(matrix.columns)
    values = matrix.to_numpy()
    col_index = {col: idx for idx, col in enumerate(variables)}
    for a, b in itertools.combinations(variables, 2):
        control_cols = [c for c in variables if c not in {a, b}]
        controls = values[:, [col_index[c] for c in control_cols]]
        r, p_value, dfree = partial_corr(values[:, col_index[a]], values[:, col_index[b]], controls)
        keep = bool(np.isfinite(p_value) and p_value < PC_ALPHA and abs(r) >= MIN_ABS_PARTIAL_CORR)
        source, target, directed, rule = orient_edge(a, b)
        edge_key = f"{source} -> {target}" if directed else " -- ".join(sorted([source, target]))
        rows.append(
            {
                "edge_key": edge_key,
                "source": source,
                "target": target,
                "directed": directed,
                "direction_rule": rule,
                "var_a": a,
                "var_b": b,
                "partial_corr": r,
                "abs_partial_corr": abs(r),
                "p_value": p_value,
                "degrees_of_freedom": dfree,
                "selected_by_pc_like": keep,
                "source_group": CONCEPT_GROUP.get(source, "unknown"),
                "target_group": CONCEPT_GROUP.get(target, "unknown"),
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values(["selected_by_pc_like", "abs_partial_corr"], ascending=[False, False])


def method_support_for_edge(edges: pd.DataFrame, source: str, target: str) -> dict:
    if edges.empty:
        return {"supported": False}
    edge_key = f"{source} -> {target}"
    reverse_key = f"{target} -> {source}"
    undirected_key = " -- ".join(sorted([source, target]))
    if "selected_by_pc_like" in edges.columns:
        candidates = edges[edges["selected_by_pc_like"]].copy()
    else:
        candidates = edges.copy()
    match = candidates[
        candidates["edge_key"].isin([edge_key, reverse_key, undirected_key])
        | candidates.get("raw_edge_key", pd.Series(index=candidates.index, dtype=str)).isin(
            [edge_key, reverse_key, undirected_key]
        )
        | ((candidates["source"].eq(source) & candidates["target"].eq(target)))
        | ((candidates["source"].eq(target) & candidates["target"].eq(source)))
    ]
    if match.empty:
        return {"supported": False}
    row = match.iloc[0]
    return {
        "supported": True,
        "edge_key": row.get("edge_key", ""),
        "raw_edge": row.get("raw_edge", ""),
        "directed": row.get("directed", np.nan),
        "partial_corr": row.get("partial_corr", np.nan),
        "p_value": row.get("p_value", np.nan),
    }


def expected_relation_comparison(
    pc_like_edges: pd.DataFrame,
    standard_pc_edges: pd.DataFrame,
    standard_ges_edges: pd.DataFrame,
    our_core: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for source, target in EXPECTED_RELATIONS:
        our_target = "Porphyry_Cu_presence" if target == TARGET else target
        our_match = our_core[(our_core["source"].eq(source)) & (our_core["target"].eq(our_target))]
        pc_like = method_support_for_edge(pc_like_edges, source, target)
        standard_pc = method_support_for_edge(standard_pc_edges, source, target)
        standard_ges = method_support_for_edge(standard_ges_edges, source, target)
        row = {
            "source": source,
            "target": target,
            "expected_edge_key": f"{source} -> {target}",
            "supported_by_our_method": bool(not our_match.empty),
            "our_relation": "" if our_match.empty else str(our_match.iloc[0]["relation"]),
            "our_confidence": "" if our_match.empty else str(our_match.iloc[0]["confidence"]),
            "our_stable_edge_score": np.nan if our_match.empty else float(our_match.iloc[0]["stable_edge_score"]),
            "supported_by_pc_like": pc_like["supported"],
            "pc_like_edge_key": pc_like.get("edge_key", ""),
            "pc_like_partial_corr": pc_like.get("partial_corr", np.nan),
            "pc_like_p_value": pc_like.get("p_value", np.nan),
            "supported_by_standard_pc": standard_pc["supported"],
            "standard_pc_edge_key": standard_pc.get("edge_key", ""),
            "standard_pc_raw_edge": standard_pc.get("raw_edge", ""),
            "supported_by_standard_ges": standard_ges["supported"],
            "standard_ges_edge_key": standard_ges.get("edge_key", ""),
            "standard_ges_raw_edge": standard_ges.get("raw_edge", ""),
        }
        row["support_count_across_discovery_methods"] = int(row["supported_by_pc_like"]) + int(
            row["supported_by_standard_pc"]
        ) + int(row["supported_by_standard_ges"])
        rows.append(row)
    return pd.DataFrame(rows)


def overlap_summary(
    pc_like_edges: pd.DataFrame,
    standard_pc_edges: pd.DataFrame,
    standard_ges_edges: pd.DataFrame,
    our_stable: pd.DataFrame,
) -> pd.DataFrame:
    our_keys = set(our_stable["edge_key"].astype(str)) if "edge_key" in our_stable.columns else set()
    method_edges = {
        "pc_like_partial_corr": set(pc_like_edges[pc_like_edges["selected_by_pc_like"]]["edge_key"].astype(str)),
        "standard_pc": set(standard_pc_edges["edge_key"].astype(str)),
        "standard_ges_bic": set(standard_ges_edges["edge_key"].astype(str)),
    }
    rows = []
    for method, keys in method_edges.items():
        overlap = sorted(our_keys & keys)
        rows.append(
            {
                "comparison": f"all_stable_edges_vs_{method}",
                "our_edge_count": len(our_keys),
                "method_edge_count": len(keys),
                "overlap_count": len(overlap),
                "overlap_rate_vs_our": len(overlap) / max(1, len(our_keys)),
                "overlap_rate_vs_method": len(overlap) / max(1, len(keys)),
                "overlap_edges": ";".join(overlap),
            }
        )
    return pd.DataFrame(rows)


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


def write_report(
    path: Path,
    pc_like_edges: pd.DataFrame,
    standard_pc_edges: pd.DataFrame,
    standard_ges_edges: pd.DataFrame,
    relation: pd.DataFrame,
    overlap: pd.DataFrame,
    ges_score: float,
) -> None:
    selected = pc_like_edges[pc_like_edges["selected_by_pc_like"]].copy()
    selected_table = selected[
        [
            "edge_key",
            "directed",
            "partial_corr",
            "p_value",
            "source_group",
            "target_group",
            "direction_rule",
        ]
    ].head(20)
    standard_pc_table = standard_pc_edges[["edge_key", "raw_edge", "directed", "source_group", "target_group"]].head(20)
    standard_ges_table = standard_ges_edges[["edge_key", "raw_edge", "directed", "source_group", "target_group"]].head(20)
    relation_cols = [
        "source",
        "target",
        "supported_by_our_method",
        "our_relation",
        "our_confidence",
        "supported_by_pc_like",
        "supported_by_standard_pc",
        "supported_by_standard_ges",
        "support_count_across_discovery_methods",
    ]
    lines = [
        "# 对比方案四：因果发现方法对比",
        "",
        "## 1. 实验目的",
        "",
        "本实验同时使用轻量版 PC-like partial correlation、causal-learn 标准 PC algorithm 和 causal-learn GES/BIC，与当前跨环境稳定因果图方法进行对比。",
        "",
        "## 2. 方法设置",
        "",
        f"- 变量数：{len([TARGET] + CONCEPTS)}",
        f"- PC-like：p < {PC_ALPHA} 且 |partial r| >= {MIN_ABS_PARTIAL_CORR}，每对变量控制其余候选变量。",
        f"- 标准 PC：causal-learn PC algorithm，Fisher-Z 条件独立检验，alpha={PC_ALPHA}。",
        f"- 标准 GES：causal-learn GES，BIC score，maxP={GES_MAX_PARENTS}，score={ges_score:.4f}。",
        "- 标准 PC/GES 的原始方向保留在 raw_edge 中；同时按领域层级规则生成归一化 edge_key 便于和当前方法对比。",
        "",
        "## 3. PC-like 选中边 Top 20",
        "",
        markdown_table(selected_table),
        "",
        "## 4. 标准 PC 边 Top 20",
        "",
        markdown_table(standard_pc_table),
        "",
        "## 5. 标准 GES 边 Top 20",
        "",
        markdown_table(standard_ges_table),
        "",
        "## 6. 核心关系覆盖情况",
        "",
        markdown_table(relation[[c for c in relation_cols if c in relation.columns]]),
        "",
        "## 7. 边重叠概览",
        "",
        markdown_table(overlap),
        "",
        "## 8. 初步结论",
        "",
        "1. 如果多个标准方法也支持某条核心关系，说明该关系不仅在跨环境稳定分析中出现，也在条件独立或结构搜索基线中有信号。",
        "2. 如果标准 PC/GES 不支持某条核心关系，不代表该关系无效；可能是样本量、变量集合、非线性关系、等价类方向不可辨识或环境异质性导致。",
        "3. 当前项目应把标准 PC/GES 作为方法对照，而不是替代跨环境稳定因果图主线。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    config = load_config()
    ensure_project_dirs(config)
    dataset_path = output_path(config, "outputs", INPUT_DATASET)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Missing input dataset: {dataset_path}")

    df = read_table(dataset_path)
    matrix = prepare_matrix(df, CONCEPTS)
    pc_like_edges = discover_pc_like_edges(matrix)
    standard_pc_edges = run_standard_pc(matrix)
    standard_ges_edges, ges_score = run_standard_ges(matrix)
    our_stable = read_table(output_path(config, "outputs", OUR_STABLE_EDGES))
    our_core = read_table(output_path(config, "outputs", OUR_CORE_EDGES))
    relation = expected_relation_comparison(pc_like_edges, standard_pc_edges, standard_ges_edges, our_core)
    overlap = overlap_summary(pc_like_edges, standard_pc_edges, standard_ges_edges, our_stable)

    pc_like_edges_path = output_path(config, "outputs", f"{OUTPUT_DIR}/pc_like_partial_corr_edges.csv")
    standard_pc_edges_path = output_path(config, "outputs", f"{OUTPUT_DIR}/standard_pc_edges.csv")
    standard_ges_edges_path = output_path(config, "outputs", f"{OUTPUT_DIR}/standard_ges_bic_edges.csv")
    relation_path = output_path(config, "outputs", f"{OUTPUT_DIR}/causal_discovery_relation_comparison.csv")
    overlap_path = output_path(config, "outputs", f"{OUTPUT_DIR}/causal_discovery_overlap_summary.csv")
    report_path = output_path(config, "outputs", f"{OUTPUT_DIR}/causal_discovery_method_comparison_report.md")

    write_dataframe(pc_like_edges, pc_like_edges_path)
    write_dataframe(standard_pc_edges, standard_pc_edges_path)
    write_dataframe(standard_ges_edges, standard_ges_edges_path)
    write_dataframe(relation, relation_path)
    write_dataframe(overlap, overlap_path)
    write_report(report_path, pc_like_edges, standard_pc_edges, standard_ges_edges, relation, overlap, ges_score)

    run_summary = {
        "method": "pc_like_partial_correlation_plus_standard_pc_and_ges",
        "variables": [TARGET] + CONCEPTS,
        "alpha": PC_ALPHA,
        "min_abs_partial_corr": MIN_ABS_PARTIAL_CORR,
        "pc_like_selected_edges": int(pc_like_edges["selected_by_pc_like"].sum()),
        "standard_pc_edges": int(len(standard_pc_edges)),
        "standard_ges_edges": int(len(standard_ges_edges)),
        "ges_score": ges_score,
        "core_relations_supported_by_pc_like": int(relation["supported_by_pc_like"].sum()),
        "core_relations_supported_by_standard_pc": int(relation["supported_by_standard_pc"].sum()),
        "core_relations_supported_by_standard_ges": int(relation["supported_by_standard_ges"].sum()),
        "outputs": {
            "pc_like_edges": str(pc_like_edges_path),
            "standard_pc_edges": str(standard_pc_edges_path),
            "standard_ges_edges": str(standard_ges_edges_path),
            "relation_comparison": str(relation_path),
            "overlap_summary": str(overlap_path),
            "report": str(report_path),
        },
    }
    write_json(output_path(config, "logs", "26_compare_causal_discovery_methods_summary.json"), run_summary)

    print(f"PC-like selected edges: {int(pc_like_edges['selected_by_pc_like'].sum())}")
    print(f"Standard PC edges: {len(standard_pc_edges)}")
    print(f"Standard GES/BIC edges: {len(standard_ges_edges)}")
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
