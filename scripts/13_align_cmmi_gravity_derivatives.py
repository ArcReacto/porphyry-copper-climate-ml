from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shapefile
from PIL import Image
from sklearn.neighbors import BallTree

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.io_utils import ensure_project_dirs, load_config, output_path, read_table, write_dataframe, write_json
from src.spatial_utils import haversine_km


PREFIX = "gravity_cmmi_"
R_EARTH_KM = 6371.0088
NODATA_LIMIT = -3.0e38


def read_world_file(path: Path) -> tuple[float, float, float, float, float, float]:
    vals = [float(x.strip()) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    if len(vals) != 6:
        raise ValueError(f"Unexpected world file format: {path}")
    return tuple(vals)  # A, D, B, E, C, F


def raster_values_at_points(tif_path: Path, tfw_path: Path, targets: pd.DataFrame, name: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    a, d, b, e, c, f = read_world_file(tfw_path)
    if abs(d) > 1e-12 or abs(b) > 1e-12:
        raise ValueError(f"Rotated world files are not supported yet: {tfw_path}")

    img = Image.open(tif_path)
    arr = np.asarray(img, dtype=float)
    height, width = arr.shape

    lon = targets["longitude"].to_numpy(dtype=float)
    lat = targets["latitude"].to_numpy(dtype=float)
    col = np.rint((lon - c) / a).astype(int)
    row = np.rint((lat - f) / e).astype(int)
    inside = (row >= 0) & (row < height) & (col >= 0) & (col < width)

    values = np.full(len(targets), np.nan, dtype=float)
    values[inside] = arr[row[inside], col[inside]]
    values[values <= NODATA_LIMIT] = np.nan

    out = pd.DataFrame({f"{PREFIX}{name}_nearest": values}, index=targets.index)
    log = {
        "source": f"raster_{name}",
        "tif": str(tif_path),
        "tfw": str(tfw_path),
        "width": int(width),
        "height": int(height),
        "rows_inside_grid": int(inside.sum()),
        "rows_with_value": int(np.isfinite(values).sum()),
        "rows_missing": int(np.isnan(values).sum()),
        "min": float(np.nanmin(values)) if np.isfinite(values).any() else math.nan,
        "max": float(np.nanmax(values)) if np.isfinite(values).any() else math.nan,
    }
    return out, log


def read_worm_points(path: Path, kind: str) -> pd.DataFrame:
    reader = shapefile.Reader(str(path))
    rows = []
    for shp, rec in zip(reader.iterShapes(), reader.iterRecords()):
        data = rec.as_dict()
        if "Lon_WGS84" in data and "Lat_WGS84" in data:
            lon = data["Lon_WGS84"]
            lat = data["Lat_WGS84"]
        elif "Lon_WGS" in data and "Lat_WGS" in data:
            lon = data["Lon_WGS"]
            lat = data["Lat_WGS"]
        else:
            lon, lat = shp.points[0]
        rows.append(
            {
                "longitude": float(lon),
                "latitude": float(lat),
                "steepness": float(data.get("Steepness", np.nan)),
                "strike": float(data.get("Strike", np.nan)),
                "kind": kind,
            }
        )
    df = pd.DataFrame(rows)
    return df[df["latitude"].between(-90, 90) & df["longitude"].between(-180, 180)].reset_index(drop=True)


def bbox_filter(source: pd.DataFrame, targets: pd.DataFrame, buffer_deg: float = 2.0) -> pd.DataFrame:
    min_lat = targets["latitude"].min() - buffer_deg
    max_lat = targets["latitude"].max() + buffer_deg
    min_lon = targets["longitude"].min() - buffer_deg
    max_lon = targets["longitude"].max() + buffer_deg
    return source[source["latitude"].between(min_lat, max_lat) & source["longitude"].between(min_lon, max_lon)].copy()


def radians_frame(df: pd.DataFrame) -> np.ndarray:
    return np.deg2rad(df[["latitude", "longitude"]].to_numpy(dtype=float))


def add_worm_point_features(
    targets: pd.DataFrame,
    source: pd.DataFrame,
    kind: str,
    radii_km: list[float],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    prefix = f"{PREFIX}{kind}_worms"
    result: dict[str, Any] = {}
    if source.empty:
        result[f"{prefix}_nearest_distance_km"] = np.nan
        result[f"{prefix}_steepness_nearest"] = np.nan
        result[f"{prefix}_strike_nearest"] = np.nan
        for radius in radii_km:
            result[f"{prefix}_points_{radius:g}km"] = 0
            result[f"{prefix}_steepness_mean_{radius:g}km"] = np.nan
            result[f"{prefix}_steepness_max_{radius:g}km"] = np.nan
        return pd.DataFrame(result, index=targets.index), {
            "source": f"{kind}_worms",
            "rows_after_bbox_filter": 0,
            "added_columns": list(result),
        }

    tree = BallTree(radians_frame(source), metric="haversine")
    target_rad = radians_frame(targets)
    dist_rad, idx = tree.query(target_rad, k=1)
    nearest_idx = idx[:, 0]
    result[f"{prefix}_nearest_distance_km"] = dist_rad[:, 0] * R_EARTH_KM
    result[f"{prefix}_steepness_nearest"] = pd.to_numeric(source["steepness"], errors="coerce").to_numpy()[nearest_idx]
    result[f"{prefix}_strike_nearest"] = pd.to_numeric(source["strike"], errors="coerce").to_numpy()[nearest_idx]

    steepness = pd.to_numeric(source["steepness"], errors="coerce").to_numpy()
    for radius in radii_km:
        neighbors = tree.query_radius(target_rad, r=float(radius) / R_EARTH_KM)
        result[f"{prefix}_points_{radius:g}km"] = [int(len(x)) for x in neighbors]
        means = []
        maxes = []
        for ids in neighbors:
            vals = steepness[ids]
            vals = vals[np.isfinite(vals)]
            means.append(float(np.mean(vals)) if vals.size else np.nan)
            maxes.append(float(np.max(vals)) if vals.size else np.nan)
        result[f"{prefix}_steepness_mean_{radius:g}km"] = means
        result[f"{prefix}_steepness_max_{radius:g}km"] = maxes

    out = pd.DataFrame(result, index=targets.index)
    return out, {
        "source": f"{kind}_worms",
        "rows_after_bbox_filter": int(len(source)),
        "added_columns": list(out.columns),
    }


def drop_existing_cmmi_gravity(df: pd.DataFrame) -> pd.DataFrame:
    return df[[c for c in df.columns if not c.startswith(PREFIX)]].copy()


def main() -> int:
    parser = argparse.ArgumentParser(description="Align CMMI gravity derivative rasters and worms to sample feature table.")
    parser.add_argument("--skip-rasters", action="store_true", help="Skip CMMI gravity GeoTIFF raster features.")
    parser.add_argument("--skip-worms", action="store_true", help="Skip CMMI gravity source worm point features.")
    args = parser.parse_args()

    config = load_config()
    ensure_project_dirs(config)
    cmmi = config["gravity"]["cmmi_derivatives"]
    radii_km = [float(x) for x in config["targets"]["radii_km"]]

    features_path = output_path(config, "outputs", "model_features_samples.parquet")
    if not features_path.exists():
        raise FileNotFoundError(f"Missing {features_path}. Run scripts/05_spatial_align_features.py first.")
    features = drop_existing_cmmi_gravity(read_table(features_path))
    targets = features[["latitude", "longitude"]].copy()

    parts = []
    logs = []
    if not args.skip_rasters:
        rasters = [
            ("hgm", Path(cmmi["hgm_tif"]), Path(cmmi["hgm_tfw"])),
            ("up30km", Path(cmmi["up30km_tif"]), Path(cmmi["up30km_tfw"])),
            ("up30km_hgm", Path(cmmi["up30km_hgm_tif"]), Path(cmmi["up30km_hgm_tfw"])),
        ]
        for name, tif_path, tfw_path in rasters:
            part, log = raster_values_at_points(tif_path, tfw_path, targets, name)
            parts.append(part)
            logs.append(log)
            print(f"Aligned CMMI gravity raster {name}: {log['rows_with_value']} rows with values")

    if not args.skip_worms:
        for kind, shp_key in [("shallow", "shallow_worms_shp"), ("deep", "deep_worms_shp")]:
            worms = read_worm_points(Path(cmmi[shp_key]), kind)
            worms = bbox_filter(worms, targets)
            part, log = add_worm_point_features(targets, worms, kind, radii_km)
            parts.append(part)
            logs.append(log)
            print(f"Aligned CMMI gravity {kind} worms: {log['rows_after_bbox_filter']} points after bbox filter")

    if parts:
        features = pd.concat([features, *parts], axis=1)

    out_parquet = output_path(config, "outputs", "model_features_samples.parquet")
    out_csv = output_path(config, "outputs", "model_features_samples.csv")
    write_dataframe(features, out_parquet)
    write_dataframe(features, out_csv)

    added_cols = [c for c in features.columns if c.startswith(PREFIX)]
    summary = {
        "rows": int(len(features)),
        "columns": int(len(features.columns)),
        "added_cmmi_gravity_columns": int(len(added_cols)),
        "outputs": {"parquet": str(out_parquet), "csv": str(out_csv)},
        "sources": logs,
    }
    write_json(output_path(config, "logs", "13_align_cmmi_gravity_derivatives_summary.json"), summary)
    print(f"Wrote CMMI gravity-enhanced feature table: {len(features)} rows, {len(features.columns)} columns")
    print(f"Added CMMI gravity columns: {len(added_cols)}")
    print(out_parquet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
