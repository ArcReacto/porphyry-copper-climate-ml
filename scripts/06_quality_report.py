from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.io_utils import ensure_project_dirs, load_config, output_path, read_table, write_dataframe, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Create quality reports for an aligned feature table.")
    parser.add_argument(
        "--features",
        choices=["auto", "samples", "porphyry"],
        default="auto",
        help="Feature table to analyze. Auto prefers model_features_samples if it exists.",
    )
    args = parser.parse_args()

    config = load_config()
    ensure_project_dirs(config)

    samples_path = output_path(config, "outputs", "model_features_samples.parquet")
    porphyry_path = output_path(config, "outputs", "model_features_porphyry.parquet")
    if args.features == "samples":
        features_path = samples_path
    elif args.features == "porphyry":
        features_path = porphyry_path
    else:
        features_path = samples_path if samples_path.exists() else porphyry_path

    if not features_path.exists():
        raise FileNotFoundError(f"Missing {features_path}. Run scripts/05_spatial_align_features.py first.")

    df = read_table(features_path)
    output_suffix = "samples" if features_path.name == "model_features_samples.parquet" else "porphyry"
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    missing = (
        df.isna()
        .mean()
        .rename("missing_rate")
        .reset_index()
        .rename(columns={"index": "column"})
        .sort_values("missing_rate", ascending=False)
    )
    numeric_summary = df[numeric_cols].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).T.reset_index()
    numeric_summary = numeric_summary.rename(columns={"index": "column"})

    missing_path = output_path(config, "outputs", f"quality_missing_rates_{output_suffix}.csv")
    summary_path = output_path(config, "outputs", f"quality_numeric_summary_{output_suffix}.csv")
    write_dataframe(missing, missing_path)
    write_dataframe(numeric_summary, summary_path)

    report_lines = [
        f"{output_suffix.title()} feature table quality report",
        f"Source file: {features_path}",
        f"Rows: {len(df)}",
        f"Columns: {len(df.columns)}",
        f"Numeric columns: {len(numeric_cols)}",
        "",
        "Top 20 columns by missing rate:",
    ]
    for _, row in missing.head(20).iterrows():
        report_lines.append(f"- {row['column']}: {row['missing_rate']:.3f}")

    if "state" in df.columns:
        report_lines.extend(["", "State counts:"])
        for state, n in df["state"].value_counts(dropna=False).head(25).items():
            report_lines.append(f"- {state}: {n}")

    if "Y_label" in df.columns:
        report_lines.extend(["", "Label counts:"])
        for label, n in df["Y_label"].value_counts(dropna=False).items():
            report_lines.append(f"- {label}: {n}")

    if "sample_type" in df.columns:
        report_lines.extend(["", "Sample type counts:"])
        for sample_type, n in df["sample_type"].value_counts(dropna=False).items():
            report_lines.append(f"- {sample_type}: {n}")

    report_txt = output_path(config, "outputs", f"quality_report_{output_suffix}.txt")
    report_txt.write_text("\n".join(report_lines), encoding="utf-8")

    plot_paths = {}
    sns.set_theme(style="whitegrid")

    if "state" in df.columns:
        plt.figure(figsize=(10, 6))
        df["state"].value_counts().head(15).sort_values().plot(kind="barh")
        plt.title("Top States in Porphyry Mine Table")
        plt.xlabel("Mine count")
        plt.tight_layout()
        path = output_path(config, "outputs", f"plot_state_counts_{output_suffix}.png")
        plt.savefig(path, dpi=160)
        plt.close()
        plot_paths["state_counts"] = str(path)

    distance_cols = [c for c in numeric_cols if c.endswith("_nearest_distance_km")]
    if distance_cols:
        sample_cols = distance_cols[:12]
        plt.figure(figsize=(11, 6))
        df[sample_cols].plot(kind="box", ax=plt.gca(), rot=45)
        plt.title("Nearest Source Distance Distributions")
        plt.ylabel("km")
        plt.tight_layout()
        path = output_path(config, "outputs", f"plot_nearest_distances_{output_suffix}.png")
        plt.savefig(path, dpi=160)
        plt.close()
        plot_paths["nearest_distances"] = str(path)

    high_signal = [c for c in numeric_cols if any(token in c for token in ["Cu", "Mo", "grav", "Bouguer", "free_air"])]
    if high_signal:
        cols = high_signal[:20]
        corr = df[cols].corr(numeric_only=True)
        plt.figure(figsize=(12, 9))
        sns.heatmap(corr, cmap="vlag", center=0, square=False)
        plt.title("Correlation Preview for Selected Feature Columns")
        plt.tight_layout()
        path = output_path(config, "outputs", f"plot_selected_feature_correlations_{output_suffix}.png")
        plt.savefig(path, dpi=160)
        plt.close()
        plot_paths["selected_feature_correlations"] = str(path)

    summary = {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "features_path": str(features_path),
        "missing_rates_csv": str(missing_path),
        "numeric_summary_csv": str(summary_path),
        "report_txt": str(report_txt),
        "plots": plot_paths,
    }
    write_json(output_path(config, "logs", "06_quality_report_summary.json"), summary)
    print("Wrote quality report")
    print(report_txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
