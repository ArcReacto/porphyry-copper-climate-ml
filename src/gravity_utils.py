from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.spatial_utils import clean_lat_lon, normalize_longitude_series


def read_lat_lon_xyz(
    path: str | Path,
    value_name: str,
    invalid_values: tuple[float, ...] = (-9999.0, 9999.0, 99999.0, 999999.99),
    chunksize: int | None = None,
) -> pd.DataFrame:
    p = Path(path)
    kwargs = {
        "comment": None,
        "encoding": "latin1",
        "engine": "c",
    }
    if chunksize:
        chunks = []
        for chunk in pd.read_csv(p, chunksize=chunksize, **kwargs):
            chunk = _standardize_lat_lon_grid(chunk, value_name, invalid_values)
            chunks.append(chunk)
        return pd.concat(chunks, ignore_index=True)
    df = pd.read_csv(p, **kwargs)
    return _standardize_lat_lon_grid(df, value_name, invalid_values)


def _standardize_lat_lon_grid(
    df: pd.DataFrame,
    value_name: str,
    invalid_values: tuple[float, ...],
) -> pd.DataFrame:
    cols = list(df.columns)
    if len(cols) < 3:
        raise ValueError("Expected at least three columns for lat/lon/value xyz file.")
    out = df[[cols[0], cols[1], cols[2]]].copy()
    out.columns = ["latitude", "longitude", value_name]
    out["latitude"] = pd.to_numeric(out["latitude"], errors="coerce")
    out["longitude"] = normalize_longitude_series(out["longitude"])
    out[value_name] = pd.to_numeric(out[value_name], errors="coerce")
    for invalid in invalid_values:
        out.loc[np.isclose(out[value_name], invalid, equal_nan=False), value_name] = np.nan
    out = clean_lat_lon(out, "latitude", "longitude")
    return out.dropna(subset=[value_name]).reset_index(drop=True)
