from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.io_utils import ensure_project_dirs, load_config, output_path, read_csv_rows, write_dataframe, write_json
from src.spatial_utils import clean_lat_lon


def split_models(value: str) -> list[str]:
    if not isinstance(value, str) or not value.strip():
        return []
    cleaned = value.replace("}{", "|").replace("{", "").replace("}", "")
    parts = re.split(r"(?=\d{1,3}:\s)", cleaned)
    return [p.strip("| ").strip() for p in parts if p.strip("| ").strip()]


def main() -> int:
    config = load_config()
    ensure_project_dirs(config)

    mrds_rows, _ = read_csv_rows(config["mines"]["mrds_full"])
    header = mrds_rows[0]

    mine_rows, _ = read_csv_rows(config["mines"]["porphyry_subset_no_header"])
    if mine_rows and mine_rows[0] == header:
        mine_rows = mine_rows[1:]

    records = [dict(zip(header, row)) for row in mine_rows if len(row) == len(header)]
    df = pd.DataFrame(records)
    df = clean_lat_lon(df, "latitude", "longitude")

    for col in ("dep_id", "site_name", "state", "country", "commod1", "commod2", "commod3", "dev_stat", "model", "score"):
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip()

    df["models_parsed"] = df["model"].fillna("").map(lambda x: "|".join(split_models(str(x))))
    df["is_porphyry_cu"] = df["model"].fillna("").str.contains("Porphyry Cu", case=False, regex=False)
    df["has_copper_commodity"] = (
        df[["commod1", "commod2", "commod3"]]
        .fillna("")
        .agg(" ".join, axis=1)
        .str.contains("Copper", case=False, regex=False)
    )
    df["mine_record_type"] = "porphyry_subset"

    keep_first = [
        "dep_id",
        "site_name",
        "latitude",
        "longitude",
        "country",
        "state",
        "county",
        "commod1",
        "commod2",
        "commod3",
        "dev_stat",
        "model",
        "models_parsed",
        "is_porphyry_cu",
        "has_copper_commodity",
        "score",
        "url",
    ]
    ordered_cols = [c for c in keep_first if c in df.columns] + [c for c in df.columns if c not in keep_first]
    df = df[ordered_cols].sort_values(["state", "site_name", "dep_id"], na_position="last").reset_index(drop=True)

    parquet_path = output_path(config, "data_intermediate", "mines_porphyry.parquet")
    csv_path = output_path(config, "data_intermediate", "mines_porphyry.csv")
    write_dataframe(df, parquet_path)
    write_dataframe(df, csv_path)

    summary = {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "state_counts": df["state"].value_counts(dropna=False).to_dict(),
        "outputs": {"parquet": str(parquet_path), "csv": str(csv_path)},
    }
    write_json(output_path(config, "logs", "01_prepare_mines_summary.json"), summary)
    print(f"Wrote {len(df)} mine records")
    print(parquet_path)
    print(csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
