from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.io_utils import ensure_project_dirs, load_config, output_path, read_table, write_dataframe, write_json


SUBSETS = {
    "western_core": [
        "Arizona",
        "Nevada",
        "Utah",
        "New Mexico",
        "Colorado",
        "California",
        "Oregon",
        "Washington",
        "Idaho",
        "Montana",
        "Wyoming",
    ],
    "southwest_core": [
        "Arizona",
        "Nevada",
        "Utah",
        "New Mexico",
        "Colorado",
    ],
}


def write_quality_report(df: pd.DataFrame, subset_name: str, states: list[str], config: dict) -> dict:
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

    missing_path = output_path(config, "outputs", f"quality_missing_rates_{subset_name}.csv")
    numeric_path = output_path(config, "outputs", f"quality_numeric_summary_{subset_name}.csv")
    report_path = output_path(config, "outputs", f"quality_report_{subset_name}.txt")
    write_dataframe(missing, missing_path)
    write_dataframe(numeric_summary, numeric_path)

    lines = [
        f"{subset_name} analysis subset quality report",
        f"Rows: {len(df)}",
        f"Columns: {len(df.columns)}",
        f"Numeric columns: {len(numeric_cols)}",
        f"Included states: {', '.join(states)}",
        "",
        "Label counts:",
    ]
    if "Y_label" in df.columns:
        for label, n in df["Y_label"].value_counts(dropna=False).sort_index().items():
            lines.append(f"- {label}: {n}")

    if "negative_type" in df.columns:
        lines.extend(["", "Negative type counts:"])
        for label, n in df["negative_type"].value_counts(dropna=False).items():
            label_text = "(positive/blank)" if str(label).strip() == "" else str(label)
            lines.append(f"- {label_text}: {n}")

    if "state" in df.columns:
        lines.extend(["", "State counts:"])
        for state, n in df["state"].value_counts(dropna=False).items():
            lines.append(f"- {state}: {n}")

    lines.extend(["", "Top 20 columns by missing rate:"])
    for _, row in missing.head(20).iterrows():
        lines.append(f"- {row['column']}: {row['missing_rate']:.3f}")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return {
        "missing_rates_csv": str(missing_path),
        "numeric_summary_csv": str(numeric_path),
        "report_txt": str(report_path),
    }


def main() -> int:
    config = load_config()
    ensure_project_dirs(config)

    features_path = output_path(config, "outputs", "model_features_samples.parquet")
    if not features_path.exists():
        raise FileNotFoundError(f"Missing {features_path}. Run scripts/stage_01_data_alignment/05_spatial_align_features.py first.")

    df = read_table(features_path)
    if "state" not in df.columns:
        raise ValueError("Feature table must include a 'state' column to build analysis subsets.")

    summary = {"source": str(features_path), "subsets": {}}
    for subset_name, states in SUBSETS.items():
        subset = df[df["state"].isin(states)].copy().reset_index(drop=True)
        parquet_path = output_path(config, "outputs", f"model_features_{subset_name}.parquet")
        csv_path = output_path(config, "outputs", f"model_features_{subset_name}.csv")
        write_dataframe(subset, parquet_path)
        write_dataframe(subset, csv_path)
        quality_outputs = write_quality_report(subset, subset_name, states, config)

        subset_summary = {
            "states": states,
            "rows": int(len(subset)),
            "columns": int(len(subset.columns)),
            "label_counts": subset["Y_label"].value_counts(dropna=False).sort_index().to_dict()
            if "Y_label" in subset.columns
            else {},
            "negative_type_counts": subset["negative_type"].value_counts(dropna=False).to_dict()
            if "negative_type" in subset.columns
            else {},
            "outputs": {"parquet": str(parquet_path), "csv": str(csv_path), **quality_outputs},
        }
        summary["subsets"][subset_name] = subset_summary
        print(f"Wrote {subset_name}: {len(subset)} rows, {len(subset.columns)} columns")
        print(parquet_path)
        print(csv_path)

    write_json(output_path(config, "logs", "08_make_analysis_subsets_summary.json"), summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
