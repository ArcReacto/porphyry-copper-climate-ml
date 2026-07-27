from __future__ import annotations

import argparse
import re
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.io_utils import ensure_project_dirs, load_config, output_path, read_table, write_dataframe, write_json


CLIMATE_PREFIX = "climate_"
FILE_RE = re.compile(r"TerraClimate_19912020_([A-Za-z0-9]+)\.nc$")
SUM_VARS = {"ppt", "aet", "def", "pet", "q"}
MEAN_VARS = {"tmin", "tmax", "vap", "vpd", "ws", "srad", "soil", "swe"}
WARM_MONTHS = [4, 5, 6, 7, 8, 9]
COOL_MONTHS = [10, 11, 12, 1, 2, 3]

warnings.filterwarnings("ignore", category=PerformanceWarning)


def climate_files(folder: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for path in sorted(folder.glob("TerraClimate_19912020_*.nc")):
        match = FILE_RE.match(path.name)
        if match:
            files[match.group(1).lower()] = path
    return files


def nearest_indices(coords: np.ndarray, values: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=float)
    values = np.asarray(values, dtype=float)
    ascending = coords[0] < coords[-1]
    search_coords = coords if ascending else coords[::-1]
    idx = np.searchsorted(search_coords, values)
    idx = np.clip(idx, 1, len(search_coords) - 1)
    left = search_coords[idx - 1]
    right = search_coords[idx]
    nearest = np.where(np.abs(values - left) <= np.abs(values - right), idx - 1, idx)
    return nearest if ascending else len(coords) - 1 - nearest


def extract_monthly_values(path: Path, var_name: str, targets: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    ds = xr.open_dataset(path, engine="h5netcdf")
    try:
        data_var = var_name if var_name in ds.data_vars else next(iter(ds.data_vars))
        da = ds[data_var]
        lat_idx = nearest_indices(ds["lat"].values, targets["latitude"].to_numpy())
        lon_idx = nearest_indices(ds["lon"].values, targets["longitude"].to_numpy())
        selected = da.isel(
            lat=xr.DataArray(lat_idx, dims="point"),
            lon=xr.DataArray(lon_idx, dims="point"),
        )
        arr = selected.transpose("point", "time").values.astype(float)
        arr = np.where(np.isfinite(arr), arr, np.nan)
        out = pd.DataFrame(
            arr,
            index=targets.index,
            columns=[f"climate_{var_name}_m{month:02d}" for month in range(1, arr.shape[1] + 1)],
        )
        nearest_lat = ds["lat"].values[lat_idx]
        nearest_lon = ds["lon"].values[lon_idx]
        log = {
            "variable": var_name,
            "path": str(path),
            "units": da.attrs.get("units", ""),
            "long_name": da.attrs.get("long_name", ""),
            "shape": list(da.shape),
            "missing_values": int(np.isnan(arr).sum()),
            "nearest_grid_lat_min": float(np.nanmin(nearest_lat)),
            "nearest_grid_lat_max": float(np.nanmax(nearest_lat)),
            "nearest_grid_lon_min": float(np.nanmin(nearest_lon)),
            "nearest_grid_lon_max": float(np.nanmax(nearest_lon)),
        }
        return out, log
    finally:
        ds.close()


def month_cols(var_name: str) -> list[str]:
    return [f"climate_{var_name}_m{month:02d}" for month in range(1, 13)]


def add_summary_features(features: pd.DataFrame, var_name: str, mode: str) -> None:
    cols = month_cols(var_name)
    values = features[cols]
    warm_cols = [f"climate_{var_name}_m{month:02d}" for month in WARM_MONTHS]
    cool_cols = [f"climate_{var_name}_m{month:02d}" for month in COOL_MONTHS]
    if mode == "sum":
        features[f"climate_{var_name}_annual_sum"] = values.sum(axis=1, min_count=1)
        features[f"climate_{var_name}_warm_sum"] = features[warm_cols].sum(axis=1, min_count=1)
        features[f"climate_{var_name}_cool_sum"] = features[cool_cols].sum(axis=1, min_count=1)
    else:
        features[f"climate_{var_name}_annual_mean"] = values.mean(axis=1)
        features[f"climate_{var_name}_warm_mean"] = features[warm_cols].mean(axis=1)
        features[f"climate_{var_name}_cool_mean"] = features[cool_cols].mean(axis=1)
    features[f"climate_{var_name}_max_month_value"] = values.max(axis=1)
    features[f"climate_{var_name}_min_month_value"] = values.min(axis=1)
    features[f"climate_{var_name}_seasonality"] = (
        features[f"climate_{var_name}_max_month_value"] - features[f"climate_{var_name}_min_month_value"]
    )


def add_derived_features(features: pd.DataFrame) -> None:
    if all(c in features.columns for c in month_cols("tmin") + month_cols("tmax")):
        for month in range(1, 13):
            tmin = features[f"climate_tmin_m{month:02d}"]
            tmax = features[f"climate_tmax_m{month:02d}"]
            features[f"climate_tmean_m{month:02d}"] = (tmin + tmax) / 2.0
            features[f"climate_dtr_m{month:02d}"] = tmax - tmin
        add_summary_features(features, "tmean", "mean")
        add_summary_features(features, "dtr", "mean")

    if "climate_ppt_annual_sum" in features.columns and "climate_pet_annual_sum" in features.columns:
        pet = features["climate_pet_annual_sum"].replace(0, np.nan)
        features["climate_water_balance_annual_mm"] = features["climate_ppt_annual_sum"] - features["climate_pet_annual_sum"]
        features["climate_ppt_pet_ratio_annual"] = features["climate_ppt_annual_sum"] / pet
    if "climate_aet_annual_sum" in features.columns and "climate_pet_annual_sum" in features.columns:
        pet = features["climate_pet_annual_sum"].replace(0, np.nan)
        features["climate_aet_pet_ratio_annual"] = features["climate_aet_annual_sum"] / pet


def drop_existing_climate_columns(df: pd.DataFrame) -> pd.DataFrame:
    keep = [c for c in df.columns if not c.startswith(CLIMATE_PREFIX)]
    return df[keep].copy()


def main() -> int:
    parser = argparse.ArgumentParser(description="Align TerraClimate 1991-2020 climatology features to sample table.")
    parser.add_argument(
        "--variables",
        nargs="*",
        default=[],
        help="Optional TerraClimate variables to align. Defaults to all files found.",
    )
    args = parser.parse_args()

    config = load_config()
    ensure_project_dirs(config)

    climate_dir = Path(config["climate"]["terraclimate_climatology_dir"])
    files = climate_files(climate_dir)
    if not files:
        raise FileNotFoundError(f"No TerraClimate_19912020_*.nc files found in {climate_dir}")

    variables = [v.lower() for v in args.variables] if args.variables else sorted(files)
    missing = [v for v in variables if v not in files]
    if missing:
        raise FileNotFoundError(f"Missing TerraClimate files for variables: {missing}")

    features_path = output_path(config, "outputs", "model_features_samples.parquet")
    if not features_path.exists():
        raise FileNotFoundError(f"Missing {features_path}. Run scripts/stage_01_data_alignment/05_spatial_align_features.py first.")
    features = drop_existing_climate_columns(read_table(features_path))
    targets = features[["latitude", "longitude"]].copy()

    logs = []
    monthly_parts = []
    for var_name in variables:
        monthly, log = extract_monthly_values(files[var_name], var_name, targets)
        monthly_parts.append(monthly)
        logs.append(log)
        print(f"Aligned climate variable {var_name}: {monthly.shape[1]} monthly columns")

    climate = pd.concat(monthly_parts, axis=1)
    enhanced = pd.concat([features, climate], axis=1)
    for var_name in variables:
        mode = "sum" if var_name in SUM_VARS else "mean"
        add_summary_features(enhanced, var_name, mode)
    add_derived_features(enhanced)
    enhanced["climate_has_terraclimate"] = 1

    out_parquet = output_path(config, "outputs", "model_features_samples.parquet")
    out_csv = output_path(config, "outputs", "model_features_samples.csv")
    write_dataframe(enhanced, out_parquet)
    write_dataframe(enhanced, out_csv)

    added_cols = [c for c in enhanced.columns if c.startswith(CLIMATE_PREFIX)]
    summary = {
        "climate_dir": str(climate_dir),
        "variables": variables,
        "rows": int(len(enhanced)),
        "columns": int(len(enhanced.columns)),
        "added_climate_columns": len(added_cols),
        "outputs": {"parquet": str(out_parquet), "csv": str(out_csv)},
        "sources": logs,
    }
    write_json(output_path(config, "logs", "12_align_climate_summary.json"), summary)
    print(f"Wrote climate-enhanced feature table: {len(enhanced)} rows, {len(enhanced.columns)} columns")
    print(f"Added climate columns: {len(added_cols)}")
    print(out_parquet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
