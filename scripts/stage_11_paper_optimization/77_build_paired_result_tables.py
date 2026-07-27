from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET = "known_mining_neutral_ratio_1_10_supervised_all_features_v1"
DEFAULT_MODEL_FAMILY = PROJECT_ROOT / "outputs" / "paper_optimization" / "model_family_replacement" / DATASET / "m1_m4_histgb_xgb_summary.csv"
DEFAULT_GRAPH_FAMILY = PROJECT_ROOT / "outputs" / "paper_optimization" / "graph_guided_model_family_replacement" / DATASET / "graph_guided_hgb_xgb_summary.csv"
DEFAULT_RF = PROJECT_ROOT / "outputs" / "paper_optimization" / "full_feature_graph_guided_m4" / DATASET / "full_feature_graph_guided_m4_summary.csv"
DEFAULT_SURFACE_RF = PROJECT_ROOT / "outputs" / "paper_optimization" / "surface_only_m4_sensitivity" / DATASET / "full_feature_graph_guided_m4_summary.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "final_paper_results" / "paired_result_tables" / DATASET
DEFAULT_DOC = PROJECT_ROOT / "docs" / "成对模型结果对比表.md"


METRICS = [
    ("average_precision_mean", "AP"),
    ("f1_mean", "F1"),
    ("top05_f1_mean", "F1@5%"),
    ("top05_ndcg_mean", "NDCG@5%"),
    ("top10_f1_mean", "F1@10%"),
    ("top10_ndcg_mean", "NDCG@10%"),
]


def read_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "cv" in df.columns:
        df = df[df["cv"].astype(str).str.lower().eq("groupkfold_state")].copy()
    return df


def metric_view(df: pd.DataFrame) -> pd.DataFrame:
    keep = ["model", "classifier_family", "model_name"] + [m[0] for m in METRICS if m[0] in df.columns]
    out = df[[c for c in keep if c in df.columns]].copy()
    if "classifier_family" not in out.columns:
        out["classifier_family"] = "random_forest"
    if "model_name" not in out.columns:
        out["model_name"] = out["model"].astype(str)
    return out


def build_m1_m4_table(model_family_path: Path) -> pd.DataFrame:
    df = metric_view(read_summary(model_family_path))
    variant_order = {
        "M1_Full_Climate": 1,
        "M2_No_Climate": 2,
        "M3_Climate_Normalized": 3,
        "M4_Sensitive_Residualized": 4,
    }
    df = df[df["model"].isin(variant_order)].copy()
    df["variant_order"] = df["model"].map(variant_order)
    df["method_explanation"] = df["model"].map(
        {
            "M1_Full_Climate": "Raw climate variables included.",
            "M2_No_Climate": "Direct climate variables removed; same classifier baseline.",
            "M3_Climate_Normalized": "Broad geochemical residualization.",
            "M4_Sensitive_Residualized": "Fold-local Spearman selective residualization.",
        }
    )
    df = df.sort_values(["classifier_family", "variant_order"]).drop(columns=["variant_order"])
    for raw, pretty in METRICS:
        if raw in df.columns:
            df[pretty] = df[raw].astype(float)
            df = df.drop(columns=[raw])
    return df


def build_pair_delta_table(rf_path: Path, graph_family_path: Path, surface_rf_path: Path) -> pd.DataFrame:
    rf = metric_view(read_summary(rf_path))
    rf["classifier_family"] = "random_forest"
    graph = metric_view(read_summary(graph_family_path))
    surface = metric_view(read_summary(surface_rf_path))
    surface["classifier_family"] = "random_forest"
    surface["source"] = "surface_only"
    rf["source"] = "original"
    graph["source"] = "original"

    frames = []
    frames.append(rf[rf["model"].isin(["NoClimate_RF", "M4_Spearman_RF", "M4_GraphGuided_RF", "M4_GraphUnion_RF"])])
    frames.append(graph[graph["model"].isin(["NoClimate_RF", "M4_Spearman_RF", "M4_GraphGuided_RF", "M4_GraphUnion_RF"])])
    frames.append(surface[surface["model"].isin(["M4_GraphGuided_RF", "M4_GraphUnion_RF"])])
    all_rows = pd.concat(frames, ignore_index=True)

    renamed = {
        "NoClimate_RF": "M2_NoClimate",
        "M4_Spearman_RF": "M4_Spearman",
        "M4_GraphGuided_RF": "M4_GraphGuided",
        "M4_GraphUnion_RF": "M4_GraphUnion",
    }
    all_rows["method"] = all_rows["model"].map(renamed)
    for raw, pretty in METRICS:
        if raw in all_rows.columns:
            all_rows[pretty] = all_rows[raw].astype(float)

    base_rows = all_rows[(all_rows["method"] == "M2_NoClimate") & (all_rows["source"] == "original")].copy()
    base_lookup = {
        row["classifier_family"]: {pretty: float(row[pretty]) for _, pretty in METRICS if pretty in row}
        for _, row in base_rows.iterrows()
    }
    out_rows = []
    for _, row in all_rows.iterrows():
        fam = row["classifier_family"]
        if fam not in base_lookup:
            continue
        entry = {
            "classifier_family": fam,
            "method": row["method"] + ("_SurfaceOnly" if row["source"] == "surface_only" else ""),
            "source": row["source"],
        }
        for _, pretty in METRICS:
            if pretty in row:
                value = float(row[pretty])
                entry[pretty] = value
                entry[f"Delta {pretty} vs M2"] = value - base_lookup[fam][pretty]
        out_rows.append(entry)
    out = pd.DataFrame(out_rows)
    order = {"random_forest": 1, "hist_gradient_boosting": 2, "xgboost": 3}
    method_order = {
        "M2_NoClimate": 1,
        "M4_Spearman": 2,
        "M4_GraphGuided": 3,
        "M4_GraphUnion": 4,
        "M4_GraphGuided_SurfaceOnly": 5,
        "M4_GraphUnion_SurfaceOnly": 6,
    }
    out["family_order"] = out["classifier_family"].map(order).fillna(99)
    out["method_order"] = out["method"].map(method_order).fillna(99)
    out = out.sort_values(["family_order", "method_order"]).drop(columns=["family_order", "method_order"])
    return out


def fmt(x) -> str:
    if pd.isna(x):
        return ""
    if isinstance(x, (int, float)):
        return f"{x:.4f}"
    return str(x)


def markdown_table(df: pd.DataFrame, columns: list[str], max_rows: int = 40) -> str:
    rows = df[[c for c in columns if c in df.columns]].head(max_rows)
    header = "| " + " | ".join(rows.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(rows.columns)) + " |"
    body = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in rows.to_numpy()]
    return "\n".join([header, sep] + body)


def write_doc(path: Path, m1_m4: pd.DataFrame, pair_delta: pd.DataFrame) -> None:
    lines = [
        "# 成对模型结果对比表",
        "",
        "本文件按老师意见重新组织结果：尽量在同一分类器下比较 M1--M4，或者在同一分类器下比较 M2 No-climate 与 M4。这样可以区分性能提升来自特征处理还是模型本身。",
        "",
        "## 表 A：同一模型下的 M1-M4 对比",
        "",
        "HGB 和 XGBoost 有完整的 M1/M2/M3/M4 同源结果。M2 是同模型 no-climate baseline；M4 是 fold-local Spearman 选择性残差化。",
        "",
        markdown_table(
            m1_m4,
            ["classifier_family", "model", "method_explanation", "AP", "F1", "F1@5%", "NDCG@5%", "F1@10%", "NDCG@10%"],
            20,
        ),
        "",
        "## 表 B：同一模型下 M2 与 M4 的差值",
        "",
        "差值列均为当前方法减去同一分类器的 M2 No-climate。正值表示相对 no-climate baseline 提升。",
        "",
        markdown_table(
            pair_delta,
            [
                "classifier_family",
                "method",
                "AP",
                "Delta AP vs M2",
                "F1@5%",
                "Delta F1@5% vs M2",
                "NDCG@5%",
                "Delta NDCG@5% vs M2",
                "F1@10%",
                "Delta F1@10% vs M2",
                "NDCG@10%",
                "Delta NDCG@10% vs M2",
            ],
            40,
        ),
        "",
        "## 写作建议",
        "",
        "- 主文表格优先展示同模型差值，而不是把不同分类器的最优值混在一起。",
        "- RF 可作为解释性主链：M2、M4 Spearman、M4 GraphUnion、Surface-only M4。",
        "- HGB/XGBoost 可作为模型替换验证：M4 在同一分类器下是否优于 M2。",
        "- Stacking 若继续保留，需要单独说明第一层模型、第二层元模型和 out-of-fold 训练过程。",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build paired result tables for paper revision.")
    parser.add_argument("--model-family-summary", default=str(DEFAULT_MODEL_FAMILY))
    parser.add_argument("--graph-family-summary", default=str(DEFAULT_GRAPH_FAMILY))
    parser.add_argument("--rf-summary", default=str(DEFAULT_RF))
    parser.add_argument("--surface-rf-summary", default=str(DEFAULT_SURFACE_RF))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--doc-path", default=str(DEFAULT_DOC))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    m1_m4 = build_m1_m4_table(Path(args.model_family_summary))
    pair_delta = build_pair_delta_table(Path(args.rf_summary), Path(args.graph_family_summary), Path(args.surface_rf_summary))
    m1_m4.to_csv(output_dir / "paired_m1_m4_same_classifier.csv", index=False, encoding="utf-8-sig")
    pair_delta.to_csv(output_dir / "paired_m2_vs_m4_same_classifier.csv", index=False, encoding="utf-8-sig")
    write_doc(Path(args.doc_path), m1_m4, pair_delta)
    print(f"Wrote paired result tables to: {output_dir}")
    print(f"Wrote report to: {args.doc_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
