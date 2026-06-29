from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shapefile
from sklearn.neighbors import BallTree
from shapely.geometry import Point, shape
from shapely.ops import unary_union
from shapely.prepared import prep

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.io_utils import ensure_project_dirs, load_config, output_path, read_table, write_dataframe, write_json
from src.spatial_utils import clean_lat_lon


TARGET_TEXT_COLUMNS = ["commod1", "commod2", "commod3", "model", "dep_type", "ore", "names"]
TARGET_PATTERN = re.compile(r"\b(?:cu|copper|mo|molybdenum|au|gold)\b|porphyry", flags=re.IGNORECASE)
EARTH_RADIUS_KM = 6371.0088


def text_blob(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    available = [c for c in columns if c in df.columns]
    if not available:
        return pd.Series("", index=df.index)
    out = df[available[0]].fillna("").astype(str)
    for col in available[1:]:
        out = out + " " + df[col].fillna("").astype(str)
    return out


def nearest_positive_distance_km(points: pd.DataFrame, positives: pd.DataFrame) -> np.ndarray:
    if points.empty:
        return np.array([], dtype=float)
    tree = BallTree(np.deg2rad(positives[["latitude", "longitude"]].to_numpy(dtype=float)), metric="haversine")
    distances_rad, _ = tree.query(np.deg2rad(points[["latitude", "longitude"]].to_numpy(dtype=float)), k=1)
    return distances_rad[:, 0] * EARTH_RADIUS_KM


def nearest_distance_km(points: pd.DataFrame, sources: pd.DataFrame) -> np.ndarray:
    if points.empty:
        return np.array([], dtype=float)
    tree = BallTree(np.deg2rad(sources[["latitude", "longitude"]].to_numpy(dtype=float)), metric="haversine")
    distances_rad, _ = tree.query(np.deg2rad(points[["latitude", "longitude"]].to_numpy(dtype=float)), k=1)
    return distances_rad[:, 0] * EARTH_RADIUS_KM


def add_sample_fields(df: pd.DataFrame, sample_type: str, y_label: int, negative_type: str = "") -> pd.DataFrame:
    out = df.copy()
    dep = out["dep_id"].fillna("").astype(str) if "dep_id" in out.columns else pd.Series("", index=out.index)
    prefix = "POS" if y_label == 1 else "NEG"
    out.insert(0, "sample_id", prefix + "_" + dep.where(dep.str.len() > 0, out.index.astype(str)))
    out.insert(1, "sample_type", sample_type)
    out.insert(2, "Y_label", y_label)
    out.insert(3, "negative_type", negative_type)
    return out


def default_state_shapefile(config: dict) -> Path:
    data_root = Path(config["data_root"])
    candidates = [
        data_root / "cb_2025_us_state_500k" / "cb_2025_us_state_500k.shp",
        data_root / "边界数据" / "census_states" / "cb_2025_us_state_500k.shp",
        data_root / "边界数据" / "census_states" / "cb_2024_us_state_500k.shp",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def load_state_geometries(shapefile_path: Path, states: set[str]) -> dict[str, object]:
    if not shapefile_path.exists():
        raise FileNotFoundError(f"Missing state boundary shapefile: {shapefile_path}")

    reader = shapefile.Reader(str(shapefile_path), encoding="utf-8")
    field_names = [field[0] for field in reader.fields[1:]]
    geometries: dict[str, list] = {state: [] for state in states}

    for shape_record in reader.iterShapeRecords():
        attrs = dict(zip(field_names, shape_record.record))
        state_name = str(attrs.get("NAME", "")).strip()
        state_abbr = str(attrs.get("STUSPS", "")).strip()
        key = state_name if state_name in states else state_abbr if state_abbr in states else ""
        if not key:
            continue
        geom = shape(shape_record.shape.__geo_interface__)
        if not geom.is_empty:
            geometries[key].append(geom)

    merged = {state: unary_union(parts) for state, parts in geometries.items() if parts}
    missing = sorted(states - set(merged))
    if missing:
        raise ValueError(f"State boundary shapefile does not contain these states: {missing}")
    return merged


def sample_points_in_state(
    state: str,
    geometry,
    n: int,
    positives: pd.DataFrame,
    mrds_points: pd.DataFrame,
    min_positive_distance_km: float,
    min_mrds_distance_km: float,
    rng: np.random.Generator,
    max_batches: int = 300,
) -> pd.DataFrame:
    if n <= 0:
        return pd.DataFrame()

    prepared = prep(geometry)
    minx, miny, maxx, maxy = geometry.bounds
    accepted: list[pd.DataFrame] = []
    accepted_count = 0

    for _ in range(max_batches):
        if accepted_count >= n:
            break
        remaining = n - accepted_count
        batch_size = max(1000, remaining * 100)
        lons = rng.uniform(minx, maxx, batch_size)
        lats = rng.uniform(miny, maxy, batch_size)
        inside = [prepared.contains(Point(lon, lat)) for lon, lat in zip(lons, lats)]
        if not any(inside):
            continue

        candidates = pd.DataFrame({"longitude": lons[inside], "latitude": lats[inside]})
        candidates["distance_to_nearest_positive_km"] = nearest_distance_km(candidates, positives)
        candidates["distance_to_nearest_mrds_km"] = nearest_distance_km(candidates, mrds_points)
        candidates = candidates[
            (candidates["distance_to_nearest_positive_km"] >= min_positive_distance_km)
            & (candidates["distance_to_nearest_mrds_km"] >= min_mrds_distance_km)
        ].copy()
        if candidates.empty:
            continue

        take = candidates.head(remaining).copy()
        take["state"] = state
        accepted.append(take)
        accepted_count += len(take)

    if not accepted:
        return pd.DataFrame()
    return pd.concat(accepted, ignore_index=True).head(n)


def build_background_negatives(
    positives: pd.DataFrame,
    mrds_points: pd.DataFrame,
    shapefile_path: Path,
    ratio: float,
    min_positive_distance_km: float,
    min_mrds_distance_km: float,
    seed: int,
) -> pd.DataFrame:
    if ratio <= 0:
        return pd.DataFrame()

    positive_counts = positives["state"].fillna("__MISSING__").value_counts()
    states = set(positive_counts.index.astype(str))
    geometries = load_state_geometries(shapefile_path, states)
    rng = np.random.default_rng(seed)
    chunks = []

    for state, pos_count in positive_counts.items():
        n = int(np.ceil(pos_count * ratio))
        state_points = sample_points_in_state(
            str(state),
            geometries[str(state)],
            n,
            positives,
            mrds_points,
            min_positive_distance_km,
            min_mrds_distance_km,
            rng,
        )
        if len(state_points) < n:
            print(f"Warning: requested {n} background negatives for {state}, sampled {len(state_points)}.")
        chunks.append(state_points)

    if not chunks:
        return pd.DataFrame()

    background = pd.concat(chunks, ignore_index=True)
    background.insert(0, "dep_id", [f"BKG_{i + 1:06d}" for i in range(len(background))])
    background["site_name"] = "Random background point"
    background["country"] = "United States"
    background["mine_record_type"] = "census_state_random_background"
    background["background_source"] = str(shapefile_path)
    return background


def normalize_mixed_object_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == "object":
            out[col] = out[col].astype("string")
    return out


def stratified_sample_by_state(candidates: pd.DataFrame, positives: pd.DataFrame, ratio: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    chunks = []
    positive_counts = positives["state"].fillna("__MISSING__").value_counts()

    for state, pos_count in positive_counts.items():
        n = int(np.ceil(pos_count * ratio))
        mask = candidates["state"].fillna("__MISSING__").eq(state)
        state_candidates = candidates.loc[mask]
        if state_candidates.empty:
            continue
        if len(state_candidates) <= n:
            chunks.append(state_candidates)
            continue
        sampled_index = rng.choice(state_candidates.index.to_numpy(), size=n, replace=False)
        chunks.append(state_candidates.loc[sampled_index])

    if not chunks:
        return candidates.iloc[0:0].copy()
    return pd.concat(chunks, ignore_index=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a fixed positive/negative sample table for feature alignment.")
    parser.add_argument("--force", action="store_true", help="Rebuild samples even if outputs already exist.")
    parser.add_argument("--negative-ratio", type=float, default=1.0, help="MRDS hard negatives per positive, stratified by state.")
    parser.add_argument("--background-ratio", type=float, default=1.0, help="Random background negatives per positive, stratified by state.")
    parser.add_argument("--no-background", action="store_true", help="Do not add random Census state background negatives.")
    parser.add_argument("--min-negative-distance-km", type=float, default=20.0, help="Minimum distance from any positive sample.")
    parser.add_argument("--min-background-positive-distance-km", type=float, default=30.0, help="Minimum background distance from any positive sample.")
    parser.add_argument("--min-background-mrds-distance-km", type=float, default=10.0, help="Minimum background distance from any MRDS record.")
    parser.add_argument("--state-shapefile", type=str, default="", help="Census state boundary shapefile path.")
    parser.add_argument("--seed", type=int, default=20260622, help="Random seed for reproducible negative sampling.")
    args = parser.parse_args()

    config = load_config()
    ensure_project_dirs(config)

    out_parquet = output_path(config, "data_intermediate", "samples_master.parquet")
    out_csv = output_path(config, "data_intermediate", "samples_master.csv")
    if out_parquet.exists() and out_csv.exists() and not args.force:
        samples = read_table(out_parquet)
        print(f"Samples already exist; skipped rebuild: {len(samples)} rows")
        print(out_parquet)
        print("Use --force to rebuild from mrds.csv.")
        return 0

    positives_path = output_path(config, "data_intermediate", "mines_porphyry.parquet")
    if not positives_path.exists():
        raise FileNotFoundError(f"Missing {positives_path}. Run scripts/01_prepare_mines.py first.")

    positives = clean_lat_lon(read_table(positives_path), "latitude", "longitude")
    positives = positives.drop_duplicates(subset=["dep_id"], keep="first").reset_index(drop=True)

    mrds = pd.read_csv(config["mines"]["mrds_full"], low_memory=False)
    mrds = clean_lat_lon(mrds, "latitude", "longitude")
    mrds = mrds.drop_duplicates(subset=["dep_id"], keep="first")

    positive_dep_ids = set(positives["dep_id"].dropna().astype(str))
    not_positive = ~mrds["dep_id"].astype(str).isin(positive_dep_ids)
    target_like = text_blob(mrds, TARGET_TEXT_COLUMNS).str.contains(TARGET_PATTERN, regex=True, na=False)
    same_states = mrds["state"].fillna("__MISSING__").isin(set(positives["state"].fillna("__MISSING__")))

    candidate_pool = mrds.loc[not_positive & ~target_like & same_states].copy()
    candidate_pool["distance_to_nearest_positive_km"] = nearest_positive_distance_km(candidate_pool, positives)
    candidate_pool = candidate_pool[candidate_pool["distance_to_nearest_positive_km"] >= args.min_negative_distance_km].copy()

    negatives = stratified_sample_by_state(candidate_pool, positives, args.negative_ratio, args.seed)

    state_shapefile = Path(args.state_shapefile) if args.state_shapefile else default_state_shapefile(config)
    if args.no_background:
        background = pd.DataFrame()
    else:
        background = build_background_negatives(
            positives=positives,
            mrds_points=mrds,
            shapefile_path=state_shapefile,
            ratio=args.background_ratio,
            min_positive_distance_km=args.min_background_positive_distance_km,
            min_mrds_distance_km=args.min_background_mrds_distance_km,
            seed=args.seed + 1,
        )

    positives["distance_to_nearest_positive_km"] = 0.0
    positives["distance_to_nearest_mrds_km"] = 0.0
    negatives["distance_to_nearest_mrds_km"] = 0.0
    positives_samples = add_sample_fields(positives, "positive", 1)
    negatives_samples = add_sample_fields(negatives, "negative", 0, "mrds_hard_negative")
    background_samples = (
        add_sample_fields(background, "negative", 0, "background_candidate_negative")
        if not background.empty
        else pd.DataFrame()
    )

    all_columns = list(
        dict.fromkeys(
            list(positives_samples.columns) + list(negatives_samples.columns) + list(background_samples.columns)
        )
    )
    samples = pd.concat(
        [
            positives_samples.reindex(columns=all_columns),
            negatives_samples.reindex(columns=all_columns),
            background_samples.reindex(columns=all_columns),
        ],
        ignore_index=True,
    )

    front_cols = [
        "sample_id",
        "sample_type",
        "Y_label",
        "negative_type",
        "distance_to_nearest_positive_km",
        "distance_to_nearest_mrds_km",
        "dep_id",
        "mrds_id",
        "site_name",
        "latitude",
        "longitude",
        "country",
        "state",
        "county",
        "commod1",
        "commod2",
        "commod3",
        "model",
        "dep_type",
        "prod_size",
        "mine_record_type",
        "background_source",
    ]
    ordered = [c for c in front_cols if c in samples.columns] + [c for c in samples.columns if c not in front_cols]
    samples = samples[ordered].sort_values(["Y_label", "state", "sample_id"], ascending=[False, True, True]).reset_index(drop=True)

    samples = normalize_mixed_object_columns(samples)
    candidate_pool = normalize_mixed_object_columns(candidate_pool)

    write_dataframe(samples, out_parquet)
    write_dataframe(samples, out_csv)

    pool_path = output_path(config, "data_intermediate", "mrds_negative_candidate_pool.parquet")
    write_dataframe(candidate_pool, pool_path)

    summary = {
        "rows": int(len(samples)),
        "positive_rows": int((samples["Y_label"] == 1).sum()),
        "negative_rows": int((samples["Y_label"] == 0).sum()),
        "mrds_hard_negative_rows": int((samples["negative_type"] == "mrds_hard_negative").sum()),
        "background_candidate_negative_rows": int((samples["negative_type"] == "background_candidate_negative").sum()),
        "negative_ratio": args.negative_ratio,
        "background_ratio": args.background_ratio,
        "min_negative_distance_km": args.min_negative_distance_km,
        "min_background_positive_distance_km": args.min_background_positive_distance_km,
        "min_background_mrds_distance_km": args.min_background_mrds_distance_km,
        "state_shapefile": str(state_shapefile),
        "seed": args.seed,
        "candidate_pool_rows_after_filters": int(len(candidate_pool)),
        "state_counts_by_label": {
            str(label): samples.loc[samples["Y_label"] == label, "state"].value_counts(dropna=False).to_dict()
            for label in sorted(samples["Y_label"].dropna().unique(), reverse=True)
        },
        "outputs": {"parquet": str(out_parquet), "csv": str(out_csv), "candidate_pool": str(pool_path)},
    }
    write_json(output_path(config, "logs", "07_build_sample_table_summary.json"), summary)

    print(f"Wrote fixed sample table: {len(samples)} rows")
    print(f"Positive samples: {summary['positive_rows']}")
    print(f"MRDS hard negatives: {summary['mrds_hard_negative_rows']}")
    print(f"Background candidate negatives: {summary['background_candidate_negative_rows']}")
    print(out_parquet)
    print(out_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
