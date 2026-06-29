from __future__ import annotations

import argparse
import math
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shapefile
from pyproj import Transformer
from shapely.geometry import LineString, MultiLineString, Point, box, shape
from shapely.ops import nearest_points, transform
from shapely.strtree import STRtree

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.io_utils import ensure_project_dirs, load_config, output_path, read_table, write_dataframe, write_json
from src.spatial_utils import haversine_km


R_EARTH_KM = 6371.0088
HGT_NODATA = -32768
HGT_GRID_SIZE = 3601
HGT_SECONDS_PER_DEGREE = 3600
TERRAIN_PREFIX = "terrain_"
FAULT_PREFIX = "fault_"
GEOLOGY_PREFIX = "geology_"


def hgt_tile_name(lat: float, lon: float) -> str:
    lat0 = math.floor(lat)
    lon0 = math.floor(lon)
    ns = "N" if lat0 >= 0 else "S"
    ew = "E" if lon0 >= 0 else "W"
    return f"{ns}{abs(lat0):02d}{ew}{abs(lon0):03d}.hgt"


def find_hgt_file(hgt_root: Path, tile_name: str) -> Path | None:
    direct = hgt_root / tile_name[:3] / tile_name
    if direct.exists():
        return direct
    matches = list(hgt_root.rglob(tile_name))
    return matches[0] if matches else None


@lru_cache(maxsize=512)
def load_hgt_array(path_text: str) -> np.ndarray:
    arr = np.fromfile(path_text, dtype=">i2")
    if arr.size != HGT_GRID_SIZE * HGT_GRID_SIZE:
        raise ValueError(f"Unexpected HGT grid size for {path_text}: {arr.size}")
    return arr.reshape((HGT_GRID_SIZE, HGT_GRID_SIZE)).astype(float)


def hgt_row_col(lat: float, lon: float) -> tuple[str, int, int, int, int]:
    lat0 = math.floor(lat)
    lon0 = math.floor(lon)
    row = int(round((lat0 + 1 - lat) * HGT_SECONDS_PER_DEGREE))
    col = int(round((lon - lon0) * HGT_SECONDS_PER_DEGREE))
    row = int(np.clip(row, 0, HGT_GRID_SIZE - 1))
    col = int(np.clip(col, 0, HGT_GRID_SIZE - 1))
    return hgt_tile_name(lat, lon), row, col, lat0, lon0


def finite_window(arr: np.ndarray, row: int, col: int, radius_px: int) -> np.ndarray:
    r0 = max(0, row - radius_px)
    r1 = min(HGT_GRID_SIZE, row + radius_px + 1)
    c0 = max(0, col - radius_px)
    c1 = min(HGT_GRID_SIZE, col + radius_px + 1)
    window = arr[r0:r1, c0:c1]
    return window[np.isfinite(window) & (window != HGT_NODATA)]


def terrain_at_point(lat: float, lon: float, hgt_root: Path) -> dict[str, Any]:
    tile, row, col, _, _ = hgt_row_col(lat, lon)
    hgt_path = find_hgt_file(hgt_root, tile)
    result: dict[str, Any] = {
        "terrain_tile": tile,
        "terrain_has_dem": 0,
        "terrain_elevation_m": np.nan,
        "terrain_slope_deg": np.nan,
        "terrain_roughness_3x3_m": np.nan,
        "terrain_relief_250m_m": np.nan,
        "terrain_relief_1000m_m": np.nan,
        "terrain_relief_5000m_m": np.nan,
    }
    if hgt_path is None:
        return result

    arr = load_hgt_array(str(hgt_path))
    elev = arr[row, col]
    if not np.isfinite(elev) or elev == HGT_NODATA:
        return result

    result["terrain_has_dem"] = 1
    result["terrain_elevation_m"] = float(elev)

    local_3x3 = finite_window(arr, row, col, 1)
    if local_3x3.size:
        result["terrain_roughness_3x3_m"] = float(np.max(local_3x3) - np.min(local_3x3))

    if 0 < row < HGT_GRID_SIZE - 1 and 0 < col < HGT_GRID_SIZE - 1:
        left = arr[row, col - 1]
        right = arr[row, col + 1]
        up = arr[row - 1, col]
        down = arr[row + 1, col]
        neighbors = np.array([left, right, up, down])
        if np.all(np.isfinite(neighbors)) and np.all(neighbors != HGT_NODATA):
            pixel_y_m = 111_132.0 / HGT_SECONDS_PER_DEGREE
            pixel_x_m = max(1.0, 111_320.0 * math.cos(math.radians(lat)) / HGT_SECONDS_PER_DEGREE)
            dzdx = (right - left) / (2 * pixel_x_m)
            dzdy = (up - down) / (2 * pixel_y_m)
            result["terrain_slope_deg"] = float(math.degrees(math.atan(math.sqrt(dzdx**2 + dzdy**2))))

    for radius_m, col_name in [
        (250, "terrain_relief_250m_m"),
        (1000, "terrain_relief_1000m_m"),
        (5000, "terrain_relief_5000m_m"),
    ]:
        radius_px = max(1, int(round(radius_m / 31.0)))
        window = finite_window(arr, row, col, radius_px)
        if window.size:
            result[col_name] = float(np.max(window) - np.min(window))

    return result


def add_terrain_features(df: pd.DataFrame, hgt_root: Path) -> tuple[pd.DataFrame, dict]:
    rows = [terrain_at_point(float(row.latitude), float(row.longitude), hgt_root) for row in df.itertuples(index=False)]
    terrain = pd.DataFrame(rows, index=df.index)
    out = df.copy()
    for col in terrain.columns:
        out[col] = terrain[col]
    return out, {
        "source": "dem_hgt",
        "hgt_root": str(hgt_root),
        "rows_with_dem": int(terrain["terrain_has_dem"].sum()),
        "rows_without_dem": int((terrain["terrain_has_dem"] == 0).sum()),
        "added_columns": terrain.columns.tolist(),
    }


def read_fault_geometries(path: Path) -> list[LineString | MultiLineString]:
    reader = shapefile.Reader(str(path))
    geoms = []
    for shp in reader.iterShapes():
        geom = shape(shp.__geo_interface__)
        if geom.is_empty:
            continue
        geoms.append(geom)
    return geoms


def get_tree_geometry(tree: STRtree, geoms: list, item):
    if isinstance(item, (int, np.integer)):
        return geoms[int(item)]
    return item


def get_tree_indices(items) -> list[int]:
    return [int(x) for x in items]


def local_transformers(lat: float, lon: float) -> tuple[Transformer, Transformer]:
    proj = f"+proj=aeqd +lat_0={lat} +lon_0={lon} +datum=WGS84 +units=m +no_defs"
    to_local = Transformer.from_crs("EPSG:4326", proj, always_xy=True)
    to_wgs84 = Transformer.from_crs(proj, "EPSG:4326", always_xy=True)
    return to_local, to_wgs84


def fault_features_for_point(
    lat: float,
    lon: float,
    geoms: list[LineString | MultiLineString],
    tree: STRtree,
    radii_km: list[float],
) -> dict[str, float]:
    point_ll = Point(lon, lat)
    max_radius_km = max(radii_km)
    cos_lat = max(0.2, abs(math.cos(math.radians(lat))))
    dlat = max_radius_km / 111.0
    dlon = max_radius_km / (111.0 * cos_lat)
    candidate_items = tree.query(box(lon - dlon, lat - dlat, lon + dlon, lat + dlat))
    candidate_indices = get_tree_indices(candidate_items)

    result: dict[str, float] = {"fault_nearest_distance_km": np.nan}
    for radius in radii_km:
        result[f"fault_lines_{radius:g}km"] = 0
        result[f"fault_length_km_{radius:g}km"] = 0.0

    nearest_item = tree.nearest(point_ll)
    if nearest_item is not None:
        nearest_geom = get_tree_geometry(tree, geoms, nearest_item)
        _, near = nearest_points(point_ll, nearest_geom)
        result["fault_nearest_distance_km"] = float(haversine_km(lat, lon, near.y, near.x))

    if not candidate_indices:
        return result

    to_local, _ = local_transformers(lat, lon)
    point_local = transform(to_local.transform, point_ll)
    local_lines = []
    for idx in candidate_indices:
        try:
            local_lines.append(transform(to_local.transform, geoms[idx]))
        except Exception:  # noqa: BLE001
            continue

    for radius in radii_km:
        buffer = point_local.buffer(radius * 1000.0)
        line_count = 0
        length_km = 0.0
        for line in local_lines:
            if not line.intersects(buffer):
                continue
            line_count += 1
            length_km += line.intersection(buffer).length / 1000.0
        result[f"fault_lines_{radius:g}km"] = int(line_count)
        result[f"fault_length_km_{radius:g}km"] = float(length_km)

    return result


def add_fault_features(df: pd.DataFrame, fault_path: Path, radii_km: list[float]) -> tuple[pd.DataFrame, dict]:
    geoms = read_fault_geometries(fault_path)
    tree = STRtree(geoms)
    rows = [
        fault_features_for_point(float(row.latitude), float(row.longitude), geoms, tree, radii_km)
        for row in df.itertuples(index=False)
    ]
    features = pd.DataFrame(rows, index=df.index)
    out = df.copy()
    for col in features.columns:
        out[col] = features[col]
    return out, {
        "source": "faults_uscanada",
        "path": str(fault_path),
        "fault_geometry_count": int(len(geoms)),
        "added_columns": features.columns.tolist(),
    }


def read_geology_polygons(path: Path, target_bbox: tuple[float, float, float, float]) -> tuple[list, list[dict[str, Any]]]:
    min_lon, min_lat, max_lon, max_lat = target_bbox
    reader = shapefile.Reader(str(path), encoding="utf-8", encodingErrors="replace")
    geoms = []
    records = []
    for shp, rec in zip(reader.iterShapes(), reader.iterRecords()):
        sx0, sy0, sx1, sy1 = shp.bbox
        if sx1 < min_lon or sx0 > max_lon or sy1 < min_lat or sy0 > max_lat:
            continue
        try:
            geom = shape(shp.__geo_interface__)
        except Exception:  # noqa: BLE001
            continue
        if geom.is_empty:
            continue
        geoms.append(geom)
        records.append(rec.as_dict())
    return geoms, records


def geology_flags(class_text: str | None) -> dict[str, int]:
    text = (class_text or "").lower()
    keywords = {
        "geology_is_igneous": "igneous",
        "geology_is_intrusive": "intrusive",
        "geology_is_volcanic": "volcanic",
        "geology_is_felsic": "felsic",
        "geology_is_intermediate": "intermediate",
        "geology_is_mafic": "mafic",
        "geology_is_sedimentary": "sedimentary",
        "geology_is_carbonate": "carbonate",
        "geology_is_metamorphic": "metamorphic",
        "geology_is_unconsolidated": "unconsolidated",
    }
    return {name: int(token in text) for name, token in keywords.items()}


def geology_for_point(lat: float, lon: float, geoms: list, records: list[dict[str, Any]], tree: STRtree) -> dict[str, Any]:
    point = Point(lon, lat)
    result: dict[str, Any] = {
        "geology_class_found": 0,
        "geology_CMMI_Class": "",
        "geology_UNIT_NAME": "",
        "geology_UNIT_LINK": "",
    }
    result.update(geology_flags(None))

    for idx in get_tree_indices(tree.query(point)):
        geom = geoms[idx]
        try:
            if not geom.covers(point):
                continue
        except Exception:  # noqa: BLE001
            continue
        rec = records[idx]
        class_text = str(rec.get("CMMI_Class", "") or "")
        result["geology_class_found"] = 1
        result["geology_CMMI_Class"] = class_text
        result["geology_UNIT_NAME"] = str(rec.get("UNIT_NAME", "") or "")
        result["geology_UNIT_LINK"] = str(rec.get("UNIT_LINK", "") or "")
        result.update(geology_flags(class_text))
        return result

    return result


def add_geology_features(df: pd.DataFrame, geology_path: Path) -> tuple[pd.DataFrame, dict]:
    target_bbox = (
        float(df["longitude"].min()) - 0.1,
        float(df["latitude"].min()) - 0.1,
        float(df["longitude"].max()) + 0.1,
        float(df["latitude"].max()) + 0.1,
    )
    geoms, records = read_geology_polygons(geology_path, target_bbox)
    tree = STRtree(geoms)
    rows = [
        geology_for_point(float(row.latitude), float(row.longitude), geoms, records, tree)
        for row in df.itertuples(index=False)
    ]
    features = pd.DataFrame(rows, index=df.index)
    out = df.copy()
    for col in features.columns:
        out[col] = features[col]
    return out, {
        "source": "geology_conus",
        "path": str(geology_path),
        "polygon_count_after_bbox_filter": int(len(geoms)),
        "rows_with_geology_class": int(features["geology_class_found"].sum()),
        "rows_without_geology_class": int((features["geology_class_found"] == 0).sum()),
        "added_columns": features.columns.tolist(),
    }


def drop_existing_feature_group(df: pd.DataFrame) -> pd.DataFrame:
    prefixes = (TERRAIN_PREFIX, FAULT_PREFIX, GEOLOGY_PREFIX)
    keep_cols = [c for c in df.columns if not c.startswith(prefixes)]
    return df[keep_cols].copy()


def main() -> int:
    parser = argparse.ArgumentParser(description="Align DEM terrain, fault, and geology features to sample feature table.")
    parser.add_argument(
        "--features",
        default="samples",
        choices=["samples"],
        help="Feature table stem to update. Currently updates model_features_samples.",
    )
    parser.add_argument("--skip-terrain", action="store_true", help="Skip DEM terrain features.")
    parser.add_argument("--skip-faults", action="store_true", help="Skip fault distance/length features.")
    parser.add_argument("--skip-geology", action="store_true", help="Skip CMMI geology polygon features.")
    args = parser.parse_args()

    config = load_config()
    ensure_project_dirs(config)
    radii_km = [float(x) for x in config["targets"]["radii_km"]]
    features_path = output_path(config, "outputs", "model_features_samples.parquet")
    if not features_path.exists():
        raise FileNotFoundError(f"Missing {features_path}. Run scripts/05_spatial_align_features.py first.")

    features = read_table(features_path)
    features = drop_existing_feature_group(features)
    logs: list[dict] = []

    if not args.skip_terrain:
        features, log = add_terrain_features(features, Path(config["dem"]["hgt_root"]))
        logs.append(log)
        print(f"Added DEM terrain features: {log['rows_with_dem']} rows with DEM")

    if not args.skip_faults:
        features, log = add_fault_features(features, Path(config["geology"]["faults_uscanada_shp"]), radii_km)
        logs.append(log)
        print(f"Added fault features from {log['fault_geometry_count']} geometries")

    if not args.skip_geology:
        features, log = add_geology_features(features, Path(config["geology"]["geology_conus_shp"]))
        logs.append(log)
        print(f"Added geology features: {log['rows_with_geology_class']} rows with mapped CMMI class")

    out_parquet = output_path(config, "outputs", "model_features_samples.parquet")
    out_csv = output_path(config, "outputs", "model_features_samples.csv")
    write_dataframe(features, out_parquet)
    write_dataframe(features, out_csv)

    summary = {
        "rows": int(len(features)),
        "columns": int(len(features.columns)),
        "outputs": {"parquet": str(out_parquet), "csv": str(out_csv)},
        "sources": logs,
    }
    write_json(output_path(config, "logs", "11_align_terrain_geology_summary.json"), summary)
    print(f"Wrote terrain/geology-enhanced feature table: {len(features)} rows, {len(features.columns)} columns")
    print(out_parquet)
    print(out_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
