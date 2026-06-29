from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.geochem_utils import target_element_columns
from src.io_utils import ensure_project_dirs, load_config, output_path, write_dataframe, write_json
from src.spatial_utils import clean_lat_lon


def main() -> int:
    config = load_config()
    ensure_project_dirs(config)

    source = Path(config["geochem2"]["main_csv"])
    elements = config["targets"]["elements"]
    df = pd.read_csv(source, encoding="utf-8-sig", low_memory=False)

    element_cols = target_element_columns(list(df.columns), elements)
    base_cols = [
        "Lab_ID",
        "Field_ID",
        "Job_ID",
        "Batch_ID",
        "NURE_RecNo",
        "Lat_NAD27",
        "Long_NAD27",
        "Collection_Date",
        "Location_Description",
        "State",
        "Primary_Class",
        "Sample_Source",
    ]
    keep_cols = [c for c in base_cols if c in df.columns] + element_cols
    out = df[keep_cols].copy()

    out["latitude_nad27"] = pd.to_numeric(out["Lat_NAD27"], errors="coerce")
    out["longitude_nad27"] = pd.to_numeric(out["Long_NAD27"], errors="coerce")

    transformer = Transformer.from_crs("EPSG:4267", "EPSG:4326", always_xy=True)
    lon_wgs84, lat_wgs84 = transformer.transform(
        out["longitude_nad27"].to_numpy(dtype=float),
        out["latitude_nad27"].to_numpy(dtype=float),
    )
    out["longitude"] = lon_wgs84
    out["latitude"] = lat_wgs84
    out = clean_lat_lon(out, "latitude", "longitude")

    for col in element_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
        out.loc[out[col] <= -9000, col] = np.nan
        out.loc[out[col] < 0, col] = np.nan

    rename = {
        "Lab_ID": "lab_id",
        "Field_ID": "field_id",
        "Job_ID": "job_id",
        "Batch_ID": "batch_id",
        "NURE_RecNo": "nure_rec_no",
        "Collection_Date": "collection_date",
        "Location_Description": "location_description",
        "State": "state",
        "Primary_Class": "primary_class",
        "Sample_Source": "sample_source",
    }
    out = out.rename(columns={k: v for k, v in rename.items() if k in out.columns})
    out["source_dataset"] = "geochem2_nure_reanalysis"

    first_cols = [
        "lab_id",
        "field_id",
        "nure_rec_no",
        "latitude",
        "longitude",
        "latitude_nad27",
        "longitude_nad27",
        "state",
        "sample_source",
        "primary_class",
        "collection_date",
        "source_dataset",
    ]
    out = out[[c for c in first_cols if c in out.columns] + [c for c in out.columns if c not in first_cols]]

    parquet_path = output_path(config, "data_intermediate", "geochem2_nure_clean.parquet")
    csv_path = output_path(config, "data_intermediate", "geochem2_nure_clean_preview.csv")
    write_dataframe(out, parquet_path)
    write_dataframe(out.head(5000), csv_path)

    summary = {
        "rows": int(len(out)),
        "element_columns": element_cols,
        "state_counts": out["state"].value_counts(dropna=False).to_dict() if "state" in out else {},
        "outputs": {"parquet": str(parquet_path), "preview_csv": str(csv_path)},
    }
    write_json(output_path(config, "logs", "02_prepare_geochem2_nure_summary.json"), summary)
    print(f"Wrote {len(out)} NURE geochem2 records")
    print(parquet_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
