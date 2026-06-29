from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.io_utils import ensure_project_dirs, load_config, output_path, read_table, write_dataframe, write_json
from src.spatial_utils import add_nearest_and_radius_stats, clean_lat_lon


def require_file(path: Path, producer_script: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run {producer_script} first.")


def bbox_filter(source: pd.DataFrame, targets: pd.DataFrame, buffer_deg: float = 2.0) -> pd.DataFrame:
    min_lat = targets["latitude"].min() - buffer_deg
    max_lat = targets["latitude"].max() + buffer_deg
    min_lon = targets["longitude"].min() - buffer_deg
    max_lon = targets["longitude"].max() + buffer_deg
    return source[source["latitude"].between(min_lat, max_lat) & source["longitude"].between(min_lon, max_lon)].copy()


def attach_source(
    features: pd.DataFrame,
    source_path: Path,
    source_name: str,
    value_cols: list[str],
    radii_km: list[float],
    producer_script: str,
    skip_missing: bool,
) -> tuple[pd.DataFrame, dict]:
    if not source_path.exists():
        if skip_missing:
            return features, {"source": source_name, "status": "missing_skipped", "path": str(source_path)}
        require_file(source_path, producer_script)

    source = read_table(source_path)
    source = clean_lat_lon(source, "latitude", "longitude")
    source = bbox_filter(source, features)
    available_value_cols = [c for c in value_cols if c in source.columns]
    before_cols = set(features.columns)
    features = add_nearest_and_radius_stats(
        targets=features,
        source=source,
        value_cols=available_value_cols,
        prefix=source_name,
        radii_km=radii_km,
    )
    added_cols = [c for c in features.columns if c not in before_cols]
    return features, {
        "source": source_name,
        "status": "ok",
        "path": str(source_path),
        "source_rows_after_bbox_filter": int(len(source)),
        "value_cols": available_value_cols,
        "added_columns": added_cols,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Spatially align geochem and gravity features to fixed sample points.")
    parser.add_argument(
        "--target-points",
        choices=["samples", "mines"],
        default="samples",
        help="Use fixed positive/negative samples by default; use mines to reproduce the old positive-only table.",
    )
    parser.add_argument("--skip-missing", action="store_true", help="Skip missing intermediate files instead of failing.")
    parser.add_argument("--skip-geochem1", action="store_true", help="Skip geochem1 features.")
    parser.add_argument("--skip-gravity", action="store_true", help="Skip gravity features.")
    parser.add_argument(
        "--include-alaska-gravity",
        action="store_true",
        help="Include Alaska-specific gravity grids/stations. Default skips them for western/southwest analysis.",
    )
    args = parser.parse_args()

    config = load_config()
    ensure_project_dirs(config)
    radii_km = [float(x) for x in config["targets"]["radii_km"]]
    elements = config["targets"]["elements"]

    if args.target_points == "samples":
        target_path = output_path(config, "data_intermediate", "samples_master.parquet")
        require_file(target_path, "scripts/07_build_sample_table.py")
        output_stem = "model_features_samples"
    else:
        target_path = output_path(config, "data_intermediate", "mines_porphyry.parquet")
        require_file(target_path, "scripts/01_prepare_mines.py")
        output_stem = "model_features_porphyry"

    features = read_table(target_path)
    features = clean_lat_lon(features, "latitude", "longitude")

    logs: list[dict] = []

    geochem2_path = output_path(config, "data_intermediate", "geochem2_nure_clean.parquet")
    geochem2_cols = []
    for element in elements:
        geochem2_cols.extend([f"{element}_ppm", f"{element}_pct", f"{element}_sq_ppm"])
    features, log = attach_source(
        features,
        geochem2_path,
        "geochem2_nure",
        geochem2_cols,
        radii_km,
        "scripts/02_prepare_geochem2_nure.py",
        args.skip_missing,
    )
    logs.append(log)

    if not args.skip_geochem1:
        geochem1_path = output_path(config, "data_intermediate", "geochem1_selected_elements_wide.parquet")
        geochem1 = read_table(geochem1_path) if geochem1_path.exists() else None
        if geochem1 is not None:
            rename = {
                f"geochem1_{element}_value": f"{element}_value"
                for element in elements
                if f"geochem1_{element}_value" in geochem1.columns
            }
            geochem1 = geochem1.rename(columns=rename)
            temp_path = output_path(config, "data_intermediate", "_geochem1_selected_for_align.parquet")
            write_dataframe(geochem1, temp_path)
            geochem1_cols = list(rename.values())
        else:
            temp_path = geochem1_path
            geochem1_cols = [f"{element}_value" for element in elements]
        features, log = attach_source(
            features,
            temp_path,
            "geochem1_usgs",
            geochem1_cols,
            radii_km,
            "scripts/03_prepare_geochem1_usgs.py",
            args.skip_missing,
        )
        logs.append(log)

    if not args.skip_gravity:
        gravity_sources = [
            (
                output_path(config, "data_intermediate", "gravity_north_america.parquet"),
                "gravity_na",
                ["grav_anom"],
            )
        ]
        if args.include_alaska_gravity:
            gravity_sources.extend(
                [
                    (
                        output_path(config, "data_intermediate", "gravity_alaska_freeair.parquet"),
                        "gravity_alaska_freeair",
                        ["free_air_anom_terr_corr"],
                    ),
                    (
                        output_path(config, "data_intermediate", "gravity_alaska_station.parquet"),
                        "gravity_alaska_station",
                        ["Free_air_anom", "Bouguer_anom_267", "Bouguer_anom_simp_elev1"],
                    ),
                ]
            )
        for source_path, source_name, value_cols in gravity_sources:
            features, log = attach_source(
                features,
                source_path,
                source_name,
                value_cols,
                radii_km,
                "scripts/04_prepare_gravity.py",
                args.skip_missing,
            )
            logs.append(log)

    out_parquet = output_path(config, "outputs", f"{output_stem}.parquet")
    out_csv = output_path(config, "outputs", f"{output_stem}.csv")
    write_dataframe(features, out_parquet)
    write_dataframe(features, out_csv)

    summary = {
        "rows": int(len(features)),
        "columns": int(len(features.columns)),
        "target_points": args.target_points,
        "target_path": str(target_path),
        "include_alaska_gravity": bool(args.include_alaska_gravity),
        "outputs": {"parquet": str(out_parquet), "csv": str(out_csv)},
        "sources": logs,
    }
    write_json(output_path(config, "logs", "05_spatial_align_features_summary.json"), summary)
    print(f"Wrote aligned feature table: {len(features)} rows, {len(features.columns)} columns")
    print(out_parquet)
    print(out_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
