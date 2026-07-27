from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_SCRIPT = PROJECT_ROOT / "scripts" / "stage_11_paper_optimization" / "63_full_feature_graph_guided_m4_and_perturbation.py"


def load_base_module():
    spec = importlib.util.spec_from_file_location("full_feature_graph_guided_m4", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load base script: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def is_surface_residual_target(row: pd.Series) -> bool:
    column = str(row.get("column", ""))
    role = str(row.get("raw_role", ""))
    if not bool(row.get("residualizable", False)):
        return False
    if role == "geochemistry":
        return True
    if column.startswith("terrain_"):
        return True
    return False


def patch_surface_only_selector(base):
    original_selector = base.selected_graph_target_columns

    def selected_graph_target_columns_surface_only(mapping_path: Path, graph_targets_path: Path, role_table: pd.DataFrame):
        graph_cols = original_selector(mapping_path, graph_targets_path, role_table)
        if graph_cols.empty:
            return graph_cols
        graph_cols = graph_cols.copy()
        graph_cols["original_residualizable"] = graph_cols["residualizable"].astype(bool)
        graph_cols["surface_only_residualizable"] = graph_cols.apply(is_surface_residual_target, axis=1)
        graph_cols["residualizable"] = graph_cols["surface_only_residualizable"]
        graph_cols["surface_only_reason"] = graph_cols.apply(
            lambda row: "not_residualizable_in_original_graph"
            if not bool(row.get("original_residualizable", False))
            else (
                "kept_geochemistry"
                if str(row.get("raw_role", "")) == "geochemistry" and bool(row.get("surface_only_residualizable", False))
                else (
                    "kept_terrain"
                    if str(row.get("column", "")).startswith("terrain_") and bool(row.get("surface_only_residualizable", False))
                    else "excluded_deep_or_stable_background"
                )
            ),
            axis=1,
        )
        return graph_cols

    base.selected_graph_target_columns = selected_graph_target_columns_surface_only
    return base


def write_surface_summary(output_dir: Path) -> None:
    graph_path = output_dir / "graph_guided_raw_feature_targets.csv"
    if not graph_path.exists():
        return
    graph_cols = pd.read_csv(graph_path)
    if graph_cols.empty:
        return

    summary_rows = []
    if "surface_only_reason" in graph_cols.columns:
        by_reason = graph_cols.groupby("surface_only_reason").size().reset_index(name="feature_count")
        by_reason.to_csv(output_dir / "surface_only_filter_summary.csv", index=False, encoding="utf-8-sig")
    else:
        by_reason = pd.DataFrame()

    for model_file in ["full_feature_graph_guided_m4_manifest.json"]:
        manifest_path = output_dir / model_file
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["surface_only_m4"] = True
        manifest["surface_only_rule"] = (
            "Graph-guided residualization targets are restricted to geochemistry features and terrain_* features. "
            "Gravity, fault, geology, and other deep or stable background features are retained as raw predictors."
        )
        manifest["surface_only_kept_feature_count"] = int(graph_cols["residualizable"].astype(bool).sum())
        manifest["surface_only_original_residual_feature_count"] = int(
            graph_cols.get("original_residualizable", pd.Series(dtype=bool)).astype(bool).sum()
        )
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    kept = graph_cols[graph_cols["residualizable"].astype(bool)].copy()
    excluded = graph_cols[
        graph_cols.get("original_residualizable", pd.Series([False] * len(graph_cols))).astype(bool)
        & ~graph_cols["residualizable"].astype(bool)
    ].copy()
    summary_rows.append(f"# Surface-only M4 Sensitivity Experiment")
    summary_rows.append("")
    summary_rows.append(
        "This run follows the teacher-suggested sensitivity check: only geochemical and terrain observations are allowed to be climate-residualized. "
        "Deep gravity, faults, geology, and other stable background variables are interpreted as spatial context and are not residualized."
    )
    summary_rows.append("")
    summary_rows.append(f"- Original graph residual target count: {len(kept) + len(excluded)}")
    summary_rows.append(f"- Surface-only residual target count: {len(kept)}")
    summary_rows.append(f"- Excluded deep/stable-background target count: {len(excluded)}")
    if not by_reason.empty:
        summary_rows.append("")
        summary_rows.append("## Filter Summary")
        summary_rows.append("")
        summary_rows.append("| surface_only_reason | feature_count |")
        summary_rows.append("|---|---:|")
        for _, row in by_reason.iterrows():
            summary_rows.append(f"| {row['surface_only_reason']} | {int(row['feature_count'])} |")
    summary_rows.append("")
    summary_rows.append("## Output Files")
    summary_rows.append("")
    summary_rows.append("- `full_feature_graph_guided_m4_summary.csv`: model metrics after surface-only filtering.")
    summary_rows.append("- `graph_guided_raw_feature_targets.csv`: graph-mapped targets with surface-only filter flags.")
    summary_rows.append("- `surface_only_filter_summary.csv`: counts of kept and excluded target types.")
    (output_dir / "surface_only_m4_report.md").write_text("\n".join(summary_rows), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Surface-only M4 sensitivity experiment.")
    parser.add_argument("--run-dir", required=True, help="Standardized run directory.")
    parser.add_argument("--graph-dir", required=True, help="Climate Sensitivity Graph output directory.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--spearman-threshold", type=float, default=0.30)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir)
    graph_dir = Path(args.graph_dir)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else PROJECT_ROOT / "outputs" / "paper_optimization" / "surface_only_m4_sensitivity" / run_dir.name
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    base = patch_surface_only_selector(load_base_module())
    base.run_experiment(run_dir, graph_dir, output_dir, args.spearman_threshold)
    write_surface_summary(output_dir)
    print(f"Wrote Surface-only M4 sensitivity outputs to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
