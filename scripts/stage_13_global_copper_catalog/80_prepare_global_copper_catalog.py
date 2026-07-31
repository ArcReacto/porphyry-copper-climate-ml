from __future__ import annotations

import argparse
import os
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shapefile
from shapely.geometry import Point, shape
from shapely.prepared import prep


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("CDMPM_DATA_ROOT", PROJECT_ROOT.parent / "data_raw"))
GLOBAL_DATA_ROOT = DATA_ROOT / "Global-copper-deposit-dataset"
OUT_ROOT = PROJECT_ROOT / "outputs" / "global_copper_catalog"
EARTH_RADIUS_KM = 6371.0088


def read_csv_fallback(path: Path, **kwargs: Any) -> tuple[pd.DataFrame, str]:
    last_error = ""
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "cp1252", "latin1"):
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False, **kwargs), encoding
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"
    raise ValueError(f"Could not read {path}: {last_error}")


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def normalize_name(value: object) -> str:
    text = clean_text(value).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    stopwords = {
        "mine",
        "deposit",
        "deposits",
        "project",
        "prospect",
        "copper",
        "cu",
        "the",
        "area",
    }
    parts = [part for part in text.split() if part not in stopwords]
    return " ".join(parts)


def clean_lat_lon(df: pd.DataFrame, lat_col: str, lon_col: str) -> pd.DataFrame:
    out = df.copy()
    out["latitude"] = pd.to_numeric(out[lat_col], errors="coerce")
    out["longitude"] = pd.to_numeric(out[lon_col], errors="coerce")
    out = out[
        out["latitude"].between(-90, 90, inclusive="both")
        & out["longitude"].between(-180, 180, inclusive="both")
    ].copy()
    return out


def haversine_km(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    lat1r = np.radians(lat1)
    lon1r = np.radians(lon1)
    lat2r = np.radians(lat2)
    lon2r = np.radians(lon2)
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def nearest_existing(candidates: pd.DataFrame, existing: pd.DataFrame) -> pd.DataFrame:
    out = candidates.copy()
    if out.empty or existing.empty:
        out["nearest_existing_dep_id"] = ""
        out["nearest_existing_name"] = ""
        out["nearest_existing_distance_km"] = math.nan
        return out

    cand_xy = np.radians(out[["latitude", "longitude"]].to_numpy(dtype=float))
    exist_xy = np.radians(existing[["latitude", "longitude"]].to_numpy(dtype=float))

    try:
        from sklearn.neighbors import BallTree

        tree = BallTree(exist_xy, metric="haversine")
        distances_rad, indices = tree.query(cand_xy, k=1)
        idx = indices[:, 0]
        out["nearest_existing_distance_km"] = distances_rad[:, 0] * EARTH_RADIUS_KM
    except Exception:  # noqa: BLE001
        idx_values = []
        distances = []
        exist_lat = existing["latitude"].to_numpy(dtype=float)
        exist_lon = existing["longitude"].to_numpy(dtype=float)
        for _, row in out.iterrows():
            dist = haversine_km(
                np.array([row["latitude"]] * len(existing)),
                np.array([row["longitude"]] * len(existing)),
                exist_lat,
                exist_lon,
            )
            best = int(np.nanargmin(dist))
            idx_values.append(best)
            distances.append(float(dist[best]))
        idx = np.asarray(idx_values)
        out["nearest_existing_distance_km"] = distances

    matched = existing.iloc[idx].reset_index(drop=True)
    out["nearest_existing_dep_id"] = matched.get("dep_id", pd.Series([""] * len(matched))).fillna("").astype(str).to_numpy()
    out["nearest_existing_name"] = matched.get("site_name", pd.Series([""] * len(matched))).fillna("").astype(str).to_numpy()
    return out


def default_state_shapefile() -> Path:
    candidates = [
        DATA_ROOT / "cb_2025_us_state_500k" / "cb_2025_us_state_500k.shp",
        DATA_ROOT / "边界数据" / "census_states" / "cb_2025_us_state_500k.shp",
        DATA_ROOT / "边界数据" / "census_states" / "cb_2024_us_state_500k.shp",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def load_state_geometries(states: set[str]) -> dict[str, Any]:
    shp = default_state_shapefile()
    if not shp.exists():
        raise FileNotFoundError(f"Missing state shapefile: {shp}")
    reader = shapefile.Reader(str(shp), encoding="utf-8")
    field_names = [field[0] for field in reader.fields[1:]]
    geometries: dict[str, Any] = {}
    for record in reader.iterShapeRecords():
        attrs = dict(zip(field_names, record.record))
        name = str(attrs.get("NAME", "")).strip()
        abbr = str(attrs.get("STUSPS", "")).strip()
        key = name if name in states else abbr if abbr in states else ""
        if not key:
            continue
        geom = shape(record.shape.__geo_interface__)
        if not geom.is_empty:
            geometries[key] = geom
    missing = sorted(states - set(geometries))
    if missing:
        raise ValueError(f"State boundary file does not contain: {missing}")
    return geometries


def assign_us_state(points: pd.DataFrame, target_states: list[str]) -> pd.Series:
    geoms = load_state_geometries(set(target_states))
    prepared = {state: prep(geom) for state, geom in geoms.items()}
    assigned = []
    for _, row in points.iterrows():
        p = Point(float(row["longitude"]), float(row["latitude"]))
        state_name = ""
        for state, prepared_geom in prepared.items():
            if prepared_geom.contains(p):
                state_name = state
                break
        assigned.append(state_name)
    return pd.Series(assigned, index=points.index)


def load_global_main(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path)
    raw = clean_lat_lon(raw, "Latitude", "Longitude")
    out = pd.DataFrame(
        {
            "source_dataset": "ArcReacto_Global_Copper_Deposit",
            "source_file": path.name,
            "source_id": raw.get("Mindat_id", "").astype(str),
            "mrds_dep_id": "",
            "deposit_name": raw.get("Deposit_name", "").map(clean_text),
            "country": raw.get("Country", "").map(clean_text),
            "state_or_region": "",
            "latitude": raw["latitude"],
            "longitude": raw["longitude"],
            "commodity": "Cu",
            "deposit_type": raw.get("Deposit_type", "").map(clean_text),
            "model": raw.get("Deposit_type", "").map(clean_text),
            "tonnage_mt": pd.to_numeric(raw.get("Tonnage(Mt)", np.nan), errors="coerce"),
            "cu_grade_pct": pd.to_numeric(raw.get("Copper_grade(Cu; %)", np.nan), errors="coerce"),
            "mo_grade_pct": pd.to_numeric(raw.get("Molybdenum_grade(Mo; %)", np.nan), errors="coerce"),
            "au_grade_gt": pd.to_numeric(raw.get("Gold_grade(Au; g/t)", np.nan), errors="coerce"),
            "ag_grade_gt": pd.to_numeric(raw.get("Silver_grade(Ag; g/t)", np.nan), errors="coerce"),
        }
    )
    out["source_record_key"] = out["source_dataset"] + ":" + out["source_file"] + ":" + out["source_id"]
    return out


def load_mrds_csv(path: Path, default_type: str) -> tuple[pd.DataFrame, str]:
    raw, encoding = read_csv_fallback(path)
    raw = clean_lat_lon(raw, "latitude", "longitude")

    def col(name: str, default: object = "") -> pd.Series:
        if name in raw.columns:
            return raw[name]
        return pd.Series([default] * len(raw), index=raw.index)

    out = pd.DataFrame(
        {
            "source_dataset": "ArcReacto_USGS_MRDS_Derived",
            "source_file": path.name,
            "source_id": col("rec_id").astype(str),
            "mrds_dep_id": col("dep_id").astype(str),
            "deposit_name": col("depname").map(clean_text),
            "country": col("country").map(clean_text),
            "state_or_region": col("stprov").map(clean_text),
            "latitude": raw["latitude"],
            "longitude": raw["longitude"],
            "commodity": "Cu",
            "deposit_type": default_type,
            "model": default_type,
            "tonnage_mt": pd.to_numeric(col("oreton", np.nan), errors="coerce"),
            "cu_grade_pct": pd.to_numeric(col("cugrd", np.nan), errors="coerce"),
            "mo_grade_pct": pd.to_numeric(col("mogrd", np.nan), errors="coerce"),
            "au_grade_gt": pd.to_numeric(col("augrd", np.nan), errors="coerce"),
            "ag_grade_gt": pd.to_numeric(col("aggrd", np.nan), errors="coerce"),
        }
    )
    out["source_record_key"] = out["source_dataset"] + ":" + out["source_file"] + ":" + out["source_id"]
    return out, encoding


def read_existing_positives() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    samples_path = PROJECT_ROOT / "data_intermediate" / "samples_master.parquet"
    western_path = PROJECT_ROOT / "outputs" / "model_features_western_core.parquet"
    samples = pd.read_parquet(samples_path)
    western = pd.read_parquet(western_path)
    positives = samples[samples["Y_label"].eq(1)].copy()
    western_positives = western[western["Y_label"].eq(1)].copy()
    western_states = sorted(western_positives["state"].dropna().astype(str).unique().tolist())
    for frame in (positives, western_positives):
        frame["dep_id"] = frame.get("dep_id", "").astype(str)
        frame["site_name_norm"] = frame.get("site_name", "").map(normalize_name)
        names = frame.get("names", pd.Series([""] * len(frame), index=frame.index)).map(normalize_name)
        frame["name_match_key"] = (frame["site_name_norm"] + " " + names).str.strip()
    return positives, western_positives, western_states


def add_duplicate_flags(candidates: pd.DataFrame, existing: pd.DataFrame) -> pd.DataFrame:
    out = nearest_existing(candidates, existing)
    existing_ids = set(existing["dep_id"].dropna().astype(str))
    existing_names = set()
    for _, row in existing.iterrows():
        for col in ("site_name_norm", "name_match_key"):
            value = clean_text(row.get(col, ""))
            if len(value) >= 4:
                existing_names.add(value)

    out["deposit_name_norm"] = out["deposit_name"].map(normalize_name)
    out["exact_mrds_dep_id_match"] = out["mrds_dep_id"].astype(str).isin(existing_ids) & out["mrds_dep_id"].astype(str).ne("")
    out["name_match"] = out["deposit_name_norm"].isin(existing_names) & out["deposit_name_norm"].ne("")
    out["spatial_duplicate_2km"] = out["nearest_existing_distance_km"].le(2.0)
    out["spatial_possible_duplicate_5km"] = out["nearest_existing_distance_km"].le(5.0)
    out["duplicate_status"] = np.select(
        [
            out["exact_mrds_dep_id_match"],
            out["spatial_duplicate_2km"] & out["name_match"],
            out["spatial_duplicate_2km"],
            out["spatial_possible_duplicate_5km"] | out["name_match"],
        ],
        ["exact_mrds_dep_id", "name_and_spatial_2km", "spatial_2km", "possible_5km_or_name"],
        default="not_matched",
    )
    return out


def representative_rank(row: pd.Series) -> tuple[int, int, float]:
    source_file = str(row.get("source_file", ""))
    if source_file == "USGS_MRDS_Porphyry_copper_deposit.csv":
        source_rank = 0
    elif source_file == "Global_Copper_Deposit_Main.xlsx":
        source_rank = 1
    else:
        source_rank = 2
    has_source_id = 0 if clean_text(row.get("source_id", "")) else 1
    distance = float(row.get("nearest_existing_distance_km", math.inf))
    return source_rank, has_source_id, distance


def worst_duplicate_status(statuses: pd.Series) -> str:
    rank = {
        "exact_mrds_dep_id": 0,
        "name_and_spatial_2km": 1,
        "spatial_2km": 2,
        "possible_5km_or_name": 3,
        "not_matched": 4,
    }
    values = [status for status in statuses.astype(str).tolist() if status in rank]
    if not values:
        return "not_matched"
    return min(values, key=lambda item: rank[item])


def deduplicate_candidate_records(candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        out = candidates.copy()
        out["candidate_cluster_id"] = []
        out["candidate_cluster_size"] = []
        out["candidate_cluster_sources"] = []
        return out

    df = candidates.reset_index(drop=True).copy()
    parent = list(range(len(df)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri = find(i)
        rj = find(j)
        if ri != rj:
            parent[rj] = ri

    names = df["deposit_name_norm"].fillna("").astype(str).to_numpy()
    states = df["western_core_state"].fillna("").astype(str).to_numpy()
    lats = df["latitude"].to_numpy(dtype=float)
    lons = df["longitude"].to_numpy(dtype=float)

    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            if states[i] and states[j] and states[i] != states[j]:
                continue
            same_name = names[i] and names[i] == names[j]
            distance = float(haversine_km(np.array([lats[i]]), np.array([lons[i]]), np.array([lats[j]]), np.array([lons[j]]))[0])
            if distance <= 2.0 or (same_name and distance <= 15.0):
                union(i, j)

    clusters: dict[int, list[int]] = {}
    for idx in range(len(df)):
        clusters.setdefault(find(idx), []).append(idx)

    representatives = []
    for cluster_no, indices in enumerate(clusters.values(), start=1):
        cluster = df.iloc[indices].copy()
        ordered = sorted(indices, key=lambda idx: representative_rank(df.iloc[idx]))
        rep = df.iloc[ordered[0]].copy()
        rep["candidate_cluster_id"] = f"GCPC_{cluster_no:04d}"
        rep["candidate_cluster_size"] = len(indices)
        rep["candidate_cluster_sources"] = ";".join(sorted(cluster["source_file"].dropna().astype(str).unique()))
        rep["duplicate_status"] = worst_duplicate_status(cluster["duplicate_status"])
        rep["source_record_key"] = ";".join(sorted(cluster["source_record_key"].dropna().astype(str).unique()))
        representatives.append(rep)

    return pd.DataFrame(representatives).reset_index(drop=True)


def schema_summary(path: Path, file_kind: str) -> dict[str, Any]:
    info: dict[str, Any] = {
        "file": path.name,
        "kind": file_kind,
        "size_mb": round(path.stat().st_size / 1024 / 1024, 3),
    }
    if path.suffix.lower() == ".xlsx":
        xls = pd.ExcelFile(path)
        info["sheets"] = xls.sheet_names
        df = pd.read_excel(path, sheet_name=xls.sheet_names[0], nrows=5)
        info["columns"] = list(df.columns)
    elif path.suffix.lower() == ".csv":
        df, encoding = read_csv_fallback(path, nrows=5)
        info["encoding"] = encoding
        info["columns"] = list(df.columns)
    return info


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No records._"
    lines = [
        "| " + " | ".join(map(str, df.columns)) + " |",
        "| " + " | ".join(["---"] * len(df.columns)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]).replace("|", "\\|") for c in df.columns) + " |")
    return "\n".join(lines)


def write_report(
    catalog: pd.DataFrame,
    candidates: pd.DataFrame,
    review_candidates: pd.DataFrame,
    strict_new_candidates: pd.DataFrame,
    duplicate_report: pd.DataFrame,
    schema: list[dict[str, Any]],
    outputs: dict[str, str],
) -> None:
    type_counts = (
        catalog["deposit_type"].fillna("").replace("", "unknown").value_counts().head(20).rename_axis("deposit_type").reset_index(name="count")
    )
    country_counts = catalog["country"].fillna("").replace("", "unknown").value_counts().head(20).rename_axis("country").reset_index(name="count")
    duplicate_counts = duplicate_report["duplicate_status"].value_counts().rename_axis("duplicate_status").reset_index(name="count")
    candidate_state_counts = (
        candidates["western_core_state"].replace("", "unassigned").value_counts().rename_axis("state").reset_index(name="count")
        if not candidates.empty
        else pd.DataFrame(columns=["state", "count"])
    )
    review_state_counts = (
        review_candidates["western_core_state"].replace("", "unassigned").value_counts().rename_axis("state").reset_index(name="count")
        if not review_candidates.empty
        else pd.DataFrame(columns=["state", "count"])
    )
    strict_state_counts = (
        strict_new_candidates["western_core_state"].replace("", "unassigned").value_counts().rename_axis("state").reset_index(name="count")
        if not strict_new_candidates.empty
        else pd.DataFrame(columns=["state", "count"])
    )

    lines = [
        "# 全球铜矿点数据集检查与融合报告",
        "",
        "## 目的",
        "",
        "本报告用于判断 ArcReacto/Global-copper-deposit-dataset 是否可以与当前项目的 MRDS 斑岩铜矿点目录结合。当前阶段只生成矿点目录层结果，不覆盖已有训练数据集。",
        "",
        "## 输入文件",
        "",
    ]
    for item in schema:
        lines.append(f"- `{item['file']}`: {item['kind']}, {item['size_mb']} MB")
    lines.extend(
        [
            "",
            "## 标准化结果",
            "",
            f"- 标准化铜矿点记录数：{len(catalog)}",
            f"- western-core 斑岩铜候选记录数：{len(candidates)}",
            f"- western-core 待复核候选记录数：{len(review_candidates)}",
            f"- western-core 严格未匹配新增候选记录数：{len(strict_new_candidates)}",
            "",
            "## 矿床类型分布 Top 20",
            "",
            markdown_table(type_counts),
            "",
            "## 国家分布 Top 20",
            "",
            markdown_table(country_counts),
            "",
            "## 与现有正样本重复状态",
            "",
            markdown_table(duplicate_counts),
            "",
            "## western-core 候选点州分布",
            "",
            markdown_table(candidate_state_counts),
            "",
            "## western-core 待复核候选点州分布",
            "",
            markdown_table(review_state_counts),
            "",
            "## western-core 严格未匹配新增候选点州分布",
            "",
            markdown_table(strict_state_counts),
            "",
            "## 输出文件",
            "",
        ]
    )
    for label, path in outputs.items():
        lines.append(f"- {label}: `{path}`")
    lines.extend(
        [
            "",
            "## 接入建议",
            "",
            "1. 如果继续保持斑岩铜矿 MPM 任务，建议先人工复核 `western_core_porphyry_candidates_review.csv`，其中 `possible_5km_or_name` 不应直接自动并入训练正样本。",
            "2. 如果需要自动生成保守版新增正样本，只使用 `western_core_porphyry_candidates_strict_new.csv`。",
            "3. 不建议直接将全球所有铜矿点并入当前训练集，因为当前地球化学、地质、地球物理特征主要覆盖美国西部/西南部。",
            "4. 如果未来改成全球铜矿 MPM，需要另建全球可用的特征层，而不是复用当前美国西部特征表。",
        ]
    )
    report_path = OUT_ROOT / "global_copper_schema_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare and audit the ArcReacto global copper deposit catalog.")
    parser.add_argument("--global-data-root", type=Path, default=GLOBAL_DATA_ROOT)
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    args = parser.parse_args()

    global_root = args.global_data_root
    out_root = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)
    if not global_root.exists():
        raise FileNotFoundError(f"Global copper dataset folder not found: {global_root}")

    schema_files = [
        (global_root / "Global_Copper_Deposit_Main.xlsx", "main global copper workbook"),
        (global_root / "USGS_MRDS_Porphyry_copper_deposit.csv", "MRDS-derived porphyry copper CSV"),
        (global_root / "USGS_MRDS_Sed_copper_deposit.csv", "MRDS-derived sediment-hosted copper CSV"),
        (global_root / "USGS_MRDS_VMS_deposit.csv", "MRDS-derived VMS CSV"),
    ]
    schema = [schema_summary(path, kind) for path, kind in schema_files if path.exists()]

    main_catalog = load_global_main(global_root / "Global_Copper_Deposit_Main.xlsx")
    porphyry_mrds, porphyry_encoding = load_mrds_csv(global_root / "USGS_MRDS_Porphyry_copper_deposit.csv", "porphyry")
    sed_mrds, sed_encoding = load_mrds_csv(global_root / "USGS_MRDS_Sed_copper_deposit.csv", "sediment-hosted")
    vms_mrds, vms_encoding = load_mrds_csv(global_root / "USGS_MRDS_VMS_deposit.csv", "VMS")
    for item in schema:
        if item["file"] == "USGS_MRDS_Porphyry_copper_deposit.csv":
            item["encoding"] = porphyry_encoding
        if item["file"] == "USGS_MRDS_Sed_copper_deposit.csv":
            item["encoding"] = sed_encoding
        if item["file"] == "USGS_MRDS_VMS_deposit.csv":
            item["encoding"] = vms_encoding

    catalog = pd.concat([main_catalog, porphyry_mrds, sed_mrds, vms_mrds], ignore_index=True)
    catalog["deposit_name_norm"] = catalog["deposit_name"].map(normalize_name)
    catalog["is_porphyry_copper_candidate"] = (
        catalog["deposit_type"].fillna("").str.contains("porphyr", case=False, na=False)
        | catalog["source_file"].eq("USGS_MRDS_Porphyry_copper_deposit.csv")
    )
    catalog = catalog.drop_duplicates(
        subset=["source_dataset", "source_file", "source_id", "latitude", "longitude"],
        keep="first",
    ).reset_index(drop=True)

    existing_all, existing_western, western_states = read_existing_positives()
    catalog_with_duplicates = add_duplicate_flags(catalog, existing_all)

    porphyry_candidates = catalog_with_duplicates[catalog_with_duplicates["is_porphyry_copper_candidate"]].copy()
    us_candidates = porphyry_candidates[
        porphyry_candidates["country"].fillna("").str.lower().isin(["united states", "usa", "united states of america"])
    ].copy()
    us_candidates["western_core_state"] = ""
    has_state = us_candidates["state_or_region"].isin(western_states)
    us_candidates.loc[has_state, "western_core_state"] = us_candidates.loc[has_state, "state_or_region"]
    needs_state = ~has_state
    if needs_state.any():
        assigned = assign_us_state(us_candidates.loc[needs_state], western_states)
        us_candidates.loc[needs_state, "western_core_state"] = assigned

    western_candidates_raw = us_candidates[us_candidates["western_core_state"].isin(western_states)].copy()
    western_candidates = deduplicate_candidate_records(western_candidates_raw)
    review_candidates = western_candidates[
        ~western_candidates["duplicate_status"].isin(["exact_mrds_dep_id", "name_and_spatial_2km", "spatial_2km"])
    ].copy()
    strict_new_candidates = western_candidates[western_candidates["duplicate_status"].eq("not_matched")].copy()

    outputs = {
        "标准化全铜矿点目录 CSV": str(out_root / "global_copper_catalog_normalized.csv"),
        "标准化全铜矿点目录 Parquet": str(out_root / "global_copper_catalog_normalized.parquet"),
        "重复匹配报告": str(out_root / "global_copper_duplicate_report.csv"),
        "western-core 斑岩铜原始候选点": str(out_root / "western_core_porphyry_candidates_raw.csv"),
        "western-core 斑岩铜内部去重候选点": str(out_root / "western_core_porphyry_candidates_all.csv"),
        "western-core 待复核候选点": str(out_root / "western_core_porphyry_candidates_review.csv"),
        "western-core 严格未匹配新增候选点": str(out_root / "western_core_porphyry_candidates_strict_new.csv"),
        "机器可读汇总": str(out_root / "global_copper_catalog_summary.json"),
        "Markdown 报告": str(out_root / "global_copper_schema_report.md"),
    }

    catalog_with_duplicates.to_csv(out_root / "global_copper_catalog_normalized.csv", index=False, encoding="utf-8-sig")
    catalog_with_duplicates.to_parquet(out_root / "global_copper_catalog_normalized.parquet", index=False)
    duplicate_cols = [
        "source_dataset",
        "source_file",
        "source_id",
        "mrds_dep_id",
        "deposit_name",
        "country",
        "state_or_region",
        "latitude",
        "longitude",
        "deposit_type",
        "duplicate_status",
        "nearest_existing_dep_id",
        "nearest_existing_name",
        "nearest_existing_distance_km",
    ]
    catalog_with_duplicates[duplicate_cols].to_csv(
        out_root / "global_copper_duplicate_report.csv",
        index=False,
        encoding="utf-8-sig",
    )
    western_candidates_raw.to_csv(out_root / "western_core_porphyry_candidates_raw.csv", index=False, encoding="utf-8-sig")
    western_candidates.to_csv(out_root / "western_core_porphyry_candidates_all.csv", index=False, encoding="utf-8-sig")
    review_candidates.to_csv(out_root / "western_core_porphyry_candidates_review.csv", index=False, encoding="utf-8-sig")
    strict_new_candidates.to_csv(out_root / "western_core_porphyry_candidates_strict_new.csv", index=False, encoding="utf-8-sig")

    summary = {
            "global_dataset_root": str(global_root),
            "records": {
                "normalized_catalog": int(len(catalog_with_duplicates)),
                "porphyry_candidates_global": int(len(porphyry_candidates)),
                "us_porphyry_candidates": int(len(us_candidates)),
                "western_core_porphyry_candidates_raw": int(len(western_candidates_raw)),
                "western_core_porphyry_candidates_deduplicated": int(len(western_candidates)),
                "western_core_review_candidates": int(len(review_candidates)),
                "western_core_strict_new_candidates": int(len(strict_new_candidates)),
                "existing_positive_records": int(len(existing_all)),
                "existing_western_positive_records": int(len(existing_western)),
            },
        "western_core_states": western_states,
        "duplicate_status_counts": catalog_with_duplicates["duplicate_status"].value_counts().to_dict(),
        "schema": schema,
        "outputs": outputs,
    }
    (out_root / "global_copper_catalog_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(
        catalog_with_duplicates,
        western_candidates,
        review_candidates,
        strict_new_candidates,
        catalog_with_duplicates,
        schema,
        outputs,
    )

    print("Global copper catalog audit complete.")
    print(f"Normalized records: {len(catalog_with_duplicates)}")
    print(f"Global porphyry candidates: {len(porphyry_candidates)}")
    print(f"US porphyry candidates: {len(us_candidates)}")
    print(f"Western-core porphyry candidates raw: {len(western_candidates_raw)}")
    print(f"Western-core porphyry candidates deduplicated: {len(western_candidates)}")
    print(f"Western-core review candidates: {len(review_candidates)}")
    print(f"Western-core strict new candidates: {len(strict_new_candidates)}")
    print(f"Report: {out_root / 'global_copper_schema_report.md'}")


if __name__ == "__main__":
    main()
