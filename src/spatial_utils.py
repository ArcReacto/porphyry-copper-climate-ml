from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree


def normalize_longitude(lon: float) -> float:
    """Normalize longitude to the [-180, 180] interval."""
    if lon > 180:
        return lon - 360
    if lon < -180:
        return lon + 360
    return lon


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0088
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def normalize_longitude_series(values: pd.Series) -> pd.Series:
    nums = pd.to_numeric(values, errors="coerce")
    nums = nums.where(nums <= 180, nums - 360)
    nums = nums.where(nums >= -180, nums + 360)
    return nums


def clean_lat_lon(df: pd.DataFrame, lat_col: str = "latitude", lon_col: str = "longitude") -> pd.DataFrame:
    out = df.copy()
    out[lat_col] = pd.to_numeric(out[lat_col], errors="coerce")
    out[lon_col] = normalize_longitude_series(out[lon_col])
    return out[out[lat_col].between(-90, 90) & out[lon_col].between(-180, 180)].copy()


def _radians_frame(df: pd.DataFrame, lat_col: str, lon_col: str) -> np.ndarray:
    return np.deg2rad(df[[lat_col, lon_col]].to_numpy(dtype=float))


def add_nearest_and_radius_stats(
    targets: pd.DataFrame,
    source: pd.DataFrame,
    value_cols: list[str],
    prefix: str,
    radii_km: list[float],
    target_lat_col: str = "latitude",
    target_lon_col: str = "longitude",
    source_lat_col: str = "latitude",
    source_lon_col: str = "longitude",
) -> pd.DataFrame:
    result = targets.copy()
    new_cols = {}
    source = source.dropna(subset=[source_lat_col, source_lon_col]).copy()
    value_cols = [c for c in value_cols if c in source.columns]

    if source.empty:
        new_cols[f"{prefix}_nearest_distance_km"] = np.nan
        for col in value_cols:
            new_cols[f"{prefix}_{col}_nearest"] = np.nan
        for radius in radii_km:
            new_cols[f"{prefix}_count_{radius:g}km"] = 0
        return pd.concat([result, pd.DataFrame(new_cols, index=result.index)], axis=1)

    tree = BallTree(_radians_frame(source, source_lat_col, source_lon_col), metric="haversine")
    target_rad = _radians_frame(result, target_lat_col, target_lon_col)
    distances_rad, indices = tree.query(target_rad, k=1)
    distances_km = distances_rad[:, 0] * 6371.0088
    nearest_idx = indices[:, 0]

    new_cols[f"{prefix}_nearest_distance_km"] = distances_km
    for col in value_cols:
        values = pd.to_numeric(source[col], errors="coerce").to_numpy()
        new_cols[f"{prefix}_{col}_nearest"] = values[nearest_idx]

    for radius in radii_km:
        neighbors = tree.query_radius(target_rad, r=float(radius) / 6371.0088)
        new_cols[f"{prefix}_count_{radius:g}km"] = [len(x) for x in neighbors]
        for col in value_cols:
            values = pd.to_numeric(source[col], errors="coerce").to_numpy()
            means = []
            medians = []
            maxes = []
            p95s = []
            valid_counts = []
            for idxs in neighbors:
                arr = values[idxs]
                arr = arr[np.isfinite(arr)]
                valid_counts.append(int(arr.size))
                if arr.size:
                    means.append(float(np.mean(arr)))
                    medians.append(float(np.median(arr)))
                    maxes.append(float(np.max(arr)))
                    p95s.append(float(np.percentile(arr, 95)))
                else:
                    means.append(np.nan)
                    medians.append(np.nan)
                    maxes.append(np.nan)
                    p95s.append(np.nan)
            new_cols[f"{prefix}_{col}_n_{radius:g}km"] = valid_counts
            new_cols[f"{prefix}_{col}_mean_{radius:g}km"] = means
            new_cols[f"{prefix}_{col}_median_{radius:g}km"] = medians
            new_cols[f"{prefix}_{col}_max_{radius:g}km"] = maxes
            new_cols[f"{prefix}_{col}_p95_{radius:g}km"] = p95s

    return pd.concat([result, pd.DataFrame(new_cols, index=result.index)], axis=1)
