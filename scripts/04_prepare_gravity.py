from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.gravity_utils import read_lat_lon_xyz
from src.io_utils import ensure_project_dirs, load_config, output_path, read_table, write_dataframe, write_json
from src.spatial_utils import clean_lat_lon


def project_bounds(config: dict, buffer_deg: float = 2.0) -> tuple[float, float, float, float]:
    mines_path = output_path(config, "data_intermediate", "mines_porphyry.parquet")
    if not mines_path.exists():
        return -180.0, 180.0, -90.0, 90.0
    mines = read_table(mines_path)
    return (
        float(mines["longitude"].min()) - buffer_deg,
        float(mines["longitude"].max()) + buffer_deg,
        float(mines["latitude"].min()) - buffer_deg,
        float(mines["latitude"].max()) + buffer_deg,
    )


def filter_bounds(df: pd.DataFrame, bounds: tuple[float, float, float, float]) -> pd.DataFrame:
    min_lon, max_lon, min_lat, max_lat = bounds
    return df[df["longitude"].between(min_lon, max_lon) & df["latitude"].between(min_lat, max_lat)].copy()


def main() -> int:
    config = load_config()
    ensure_project_dirs(config)
    gravity = config["gravity"]["first_stage"]
    bounds = project_bounds(config)

    north = read_lat_lon_xyz(
        gravity["north_america_grid_xyz"],
        value_name="grav_anom",
        chunksize=500_000,
    )
    north = filter_bounds(north, bounds)
    north["source_dataset"] = "noaa_north_america_2_5min_gravity"
    north_path = output_path(config, "data_intermediate", "gravity_north_america.parquet")
    write_dataframe(north, north_path)

    alaska_freeair = read_lat_lon_xyz(
        gravity["alaska_freeair_xyz"],
        value_name="free_air_anom_terr_corr",
        chunksize=300_000,
    )
    alaska_freeair = filter_bounds(alaska_freeair, bounds)
    alaska_freeair["source_dataset"] = "noaa_alaska_terrain_corrected_free_air"
    alaska_freeair_path = output_path(config, "data_intermediate", "gravity_alaska_freeair.parquet")
    write_dataframe(alaska_freeair, alaska_freeair_path)

    alaska_station = pd.read_csv(gravity["alaska_station_xyz"], encoding="latin1")
    alaska_station = alaska_station.rename(
        columns={
            "station_name": "station_id",
            "Bouger_anom_simp_elev1": "Bouguer_anom_simp_elev1",
        }
    )
    alaska_station = clean_lat_lon(alaska_station, "latitude", "longitude")
    for col in [
        "sea_level_elev_ft",
        "alternate_elev_ft",
        "Bouguer_anom_simp_elev1",
        "Free_air_anom",
        "Bouguer_anom_267",
    ]:
        if col in alaska_station.columns:
            alaska_station[col] = pd.to_numeric(alaska_station[col], errors="coerce")
    alaska_station = filter_bounds(alaska_station, bounds)
    alaska_station["source_dataset"] = "noaa_alaska_gravity_station"
    alaska_station_path = output_path(config, "data_intermediate", "gravity_alaska_station.parquet")
    write_dataframe(alaska_station, alaska_station_path)

    summary = {
        "bounds_used": {
            "min_lon": bounds[0],
            "max_lon": bounds[1],
            "min_lat": bounds[2],
            "max_lat": bounds[3],
        },
        "outputs": {
            "north_america": str(north_path),
            "alaska_freeair": str(alaska_freeair_path),
            "alaska_station": str(alaska_station_path),
        },
        "rows": {
            "north_america": int(len(north)),
            "alaska_freeair": int(len(alaska_freeair)),
            "alaska_station": int(len(alaska_station)),
        },
        "second_stage_note": "State xindex/yindex grids are listed in config but not transformed in first-pass scripts.",
    }
    write_json(output_path(config, "logs", "04_prepare_gravity_summary.json"), summary)
    print("Wrote gravity intermediate files")
    print(north_path)
    print(alaska_freeair_path)
    print(alaska_station_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
