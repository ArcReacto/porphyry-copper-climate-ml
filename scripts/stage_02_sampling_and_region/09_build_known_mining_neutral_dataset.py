from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import Point
from shapely.prepared import prep

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.io_utils import ensure_project_dirs, load_config, output_path, read_table, write_dataframe
from src.spatial_utils import clean_lat_lon


EARTH_RADIUS_KM = 6371.0088
DEFAULT_SCHEME_NAME = "known_mining_neutral_ratio_1_10"
DATASET_ROOT = PROJECT_ROOT / "outputs" / "model_datasets" / "by_sample_scheme"
OUT_ROOT = DATASET_ROOT / "known_mining_neutral"
URANIUM_POSITIVE_CSV = "矿点-铀.csv"
TARGET_TEXT_COLUMNS = [
    "commod1",
    "commod2",
    "commod3",
    "model",
    "dep_type",
    "ore",
    "names",
    "site_name",
    "other_matl",
]
URANIUM_EXCLUDE_RE = re.compile(
    r"\b(?:uranium|uraninite|carnotite|coffinite|u3o8|thorium)\b|\bU\b",
    flags=re.IGNORECASE,
)
EPITHERMAL_AU_AG_EXCLUDE_RE = re.compile(
    r"(?:epithermal|hot[- ]?spring|comstock|bonanza|adularia|alunite|quartz[- ]?alunite|"
    r"low[- ]?sulfidation|high[- ]?sulfidation|gold|silver|\bAu\b|\bAg\b)",
    flags=re.IGNORECASE,
)
POLYMETALLIC_PB_ZN_AG_CU_EXCLUDE_RE = re.compile(
    r"(?:polymetallic|replacement|vein|manto|carbonate[- ]?replacement|skarn|"
    r"lead|zinc|silver|copper|\bPb\b|\bZn\b|\bAg\b|\bCu\b)",
    flags=re.IGNORECASE,
)
W_SKARN_VEIN_EXCLUDE_RE = re.compile(
    r"(?:tungsten|scheelite|wolframite|skarn|vein|contact[- ]?metasomatic|"
    r"\bW\b|\bWO3\b)",
    flags=re.IGNORECASE,
)
HOT_SPRING_HG_AU_AG_EXCLUDE_RE = re.compile(
    r"(?:hot[- ]?spring|mercury|cinnabar|quicksilver|epithermal|silica[- ]?sinter|"
    r"gold|silver|\bHg\b|\bAu\b|\bAg\b)",
    flags=re.IGNORECASE,
)


def load_script_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


stage02_samples = load_script_module(
    "stage02_samples",
    PROJECT_ROOT / "scripts" / "stage_02_sampling_and_region" / "07_build_sample_table.py",
)
stage07_ratios = load_script_module(
    "stage07_ratios",
    PROJECT_ROOT / "scripts" / "stage_07_climate_decoupling" / "35_negative_ratio_sensitivity.py",
)
stage04_dataset = load_script_module(
    "stage04_dataset",
    PROJECT_ROOT / "scripts" / "stage_04_model_baseline" / "09_make_model_dataset.py",
)


def normalize_object_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == "object":
            out[col] = out[col].astype("string")
    return out


def destination_point(lat: float, lon: float, distance_km: float, bearing_rad: float) -> tuple[float, float]:
    angular = distance_km / EARTH_RADIUS_KM
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)
    lat2 = math.asin(
        math.sin(lat1) * math.cos(angular)
        + math.cos(lat1) * math.sin(angular) * math.cos(bearing_rad)
    )
    lon2 = lon1 + math.atan2(
        math.sin(bearing_rad) * math.sin(angular) * math.cos(lat1),
        math.cos(angular) - math.sin(lat1) * math.sin(lat2),
    )
    return math.degrees(lat2), ((math.degrees(lon2) + 540.0) % 360.0) - 180.0


def add_sample_fields(df: pd.DataFrame, sample_type: str, y_label: int, negative_type: str = "") -> pd.DataFrame:
    out = df.copy()
    for col in ["sample_id", "sample_type", "Y_label", "negative_type"]:
        if col in out.columns:
            out = out.drop(columns=[col])
    dep = out["dep_id"].fillna("").astype(str) if "dep_id" in out.columns else pd.Series("", index=out.index)
    if y_label == 1:
        prefix = "POS"
    elif y_label == 0:
        prefix = "NEG"
    else:
        prefix = "NEU"
    out.insert(0, "sample_id", prefix + "_" + dep.where(dep.str.len() > 0, out.index.astype(str)))
    out.insert(1, "sample_type", sample_type)
    out.insert(2, "Y_label", y_label)
    out.insert(3, "negative_type", negative_type)
    return out


def western_core_states(config: dict) -> list[str]:
    western = read_table(output_path(config, "outputs", "model_features_western_core.parquet"))
    western_states = sorted(western.loc[western["Y_label"] == 1, "state"].dropna().unique().tolist())
    return western_states


def base_porphyry_positive_samples(config: dict) -> tuple[pd.DataFrame, list[str], str]:
    western_states = western_core_states(config)
    samples = read_table(output_path(config, "data_intermediate", "samples_master.parquet"))
    positives = samples[(samples["Y_label"] == 1) & (samples["state"].isin(western_states))].copy()
    positives = positives.drop_duplicates(subset=["dep_id"], keep="first").reset_index(drop=True)
    return positives, western_states, str(output_path(config, "data_intermediate", "samples_master.parquet"))


def base_csv_positive_samples(config: dict, csv_name: str) -> tuple[pd.DataFrame, list[str], str]:
    western_states = western_core_states(config)
    positive_path = Path(config["data_root"]) / csv_name
    if not positive_path.exists():
        raise FileNotFoundError(f"Missing positive sample CSV: {positive_path}")

    positives = pd.read_csv(positive_path, low_memory=False)
    positives = clean_lat_lon(positives, "latitude", "longitude")
    positives = positives[positives["country"].fillna("").eq("United States")].copy()
    positives = positives[positives["state"].isin(western_states)].copy()
    positives = positives.drop_duplicates(subset=["dep_id"], keep="first").reset_index(drop=True)
    positives["mine_record_type"] = "target_positive_from_csv"
    return positives, western_states, str(positive_path)


def base_positive_samples(
    config: dict,
    target_kind: str,
    positive_csv_name: str,
) -> tuple[pd.DataFrame, list[str], str]:
    if target_kind == "porphyry_copper":
        return base_porphyry_positive_samples(config)
    if target_kind == "uranium":
        return base_csv_positive_samples(config, positive_csv_name or URANIUM_POSITIVE_CSV)
    return base_csv_positive_samples(config, positive_csv_name)


def target_text_mask(df: pd.DataFrame, target_kind: str) -> pd.Series:
    text_columns = [col for col in TARGET_TEXT_COLUMNS if col in df.columns]
    if not text_columns:
        return pd.Series(False, index=df.index)
    text = df[text_columns].fillna("").astype(str).agg(" ".join, axis=1)
    if target_kind == "uranium":
        return text.str.contains(URANIUM_EXCLUDE_RE, regex=True, na=False)
    if target_kind == "epithermal_au_ag":
        return text.str.contains(EPITHERMAL_AU_AG_EXCLUDE_RE, regex=True, na=False)
    if target_kind == "polymetallic_pb_zn_ag_cu":
        return text.str.contains(POLYMETALLIC_PB_ZN_AG_CU_EXCLUDE_RE, regex=True, na=False)
    if target_kind == "w_skarn_veins":
        return text.str.contains(W_SKARN_VEIN_EXCLUDE_RE, regex=True, na=False)
    if target_kind == "hot_spring_hg_au_ag":
        return text.str.contains(HOT_SPRING_HG_AU_AG_EXCLUDE_RE, regex=True, na=False)
    return pd.Series(False, index=df.index)


def target_candidate_pool(
    config: dict,
    positives: pd.DataFrame,
    western_states: list[str],
    target_kind: str,
    min_positive_distance_km: float,
) -> pd.DataFrame:
    if target_kind == "porphyry_copper":
        candidate_pool = read_table(output_path(config, "data_intermediate", "mrds_negative_candidate_pool.parquet"))
        candidate_pool = clean_lat_lon(candidate_pool, "latitude", "longitude")
        candidate_pool = candidate_pool[candidate_pool["state"].isin(western_states)].copy()
        return candidate_pool[
            candidate_pool["distance_to_nearest_positive_km"] >= min_positive_distance_km
        ].copy()

    mrds = pd.read_csv(config["mines"]["mrds_full"], low_memory=False)
    mrds = clean_lat_lon(mrds, "latitude", "longitude")
    mrds = mrds[mrds["country"].fillna("").eq("United States")].copy()
    mrds = mrds[mrds["state"].isin(western_states)].copy()
    if "dep_id" in mrds.columns and "dep_id" in positives.columns:
        positive_ids = set(positives["dep_id"].dropna().astype(str))
        mrds = mrds[~mrds["dep_id"].astype(str).isin(positive_ids)].copy()
    mrds = mrds[~target_text_mask(mrds, target_kind)].copy()
    mrds = mrds.drop_duplicates(subset=["dep_id"], keep="first").reset_index(drop=True)
    mrds["distance_to_nearest_positive_km"] = stage02_samples.nearest_positive_distance_km(mrds, positives)
    return mrds[mrds["distance_to_nearest_positive_km"] >= min_positive_distance_km].copy()


def sample_known_mining_nearby_negatives(
    candidate_pool: pd.DataFrame,
    positives: pd.DataFrame,
    states: list[str],
    ratio: float,
    min_source_distance_km: float,
    max_source_distance_km: float,
    min_positive_distance_km: float,
    seed: int,
    max_attempt_multiplier: int = 300,
) -> pd.DataFrame:
    if ratio <= 0:
        return pd.DataFrame()

    state_geometries = stage02_samples.load_state_geometries(
        stage02_samples.default_state_shapefile(load_config()),
        set(states),
    )
    rng = np.random.default_rng(seed)
    chunks = []
    positive_counts = positives["state"].value_counts()

    for state, pos_count in positive_counts.items():
        n_target = int(math.ceil(pos_count * ratio))
        state_sources = candidate_pool[candidate_pool["state"].eq(state)].reset_index(drop=True)
        if state_sources.empty:
            print(f"Warning: no non-target MRDS sources for {state}")
            continue

        geom = state_geometries[str(state)]
        prepared = prep(geom)
        accepted: list[dict] = []
        max_attempts = max(n_target * max_attempt_multiplier, 1000)

        for attempt in range(max_attempts):
            if len(accepted) >= n_target:
                break
            src = state_sources.iloc[int(rng.integers(0, len(state_sources)))]
            distance = float(rng.uniform(min_source_distance_km, max_source_distance_km))
            bearing = float(rng.uniform(0.0, 2.0 * math.pi))
            lat, lon = destination_point(float(src["latitude"]), float(src["longitude"]), distance, bearing)
            if not prepared.contains(Point(lon, lat)):
                continue

            candidate = pd.DataFrame({"latitude": [lat], "longitude": [lon]})
            nearest_positive = float(stage02_samples.nearest_positive_distance_km(candidate, positives)[0])
            if nearest_positive < min_positive_distance_km:
                continue

            accepted.append(
                {
                    "dep_id": f"KMN_{state[:2].upper()}_{len(accepted) + 1:05d}",
                    "site_name": "Known mining area nearby negative point",
                    "country": "United States",
                    "state": state,
                    "latitude": lat,
                    "longitude": lon,
                    "mine_record_type": "known_mining_nearby_random_point",
                    "source_mrds_dep_id": src.get("dep_id", ""),
                    "source_mrds_site_name": src.get("site_name", src.get("names", "")),
                    "source_mrds_commod1": src.get("commod1", ""),
                    "source_mrds_commod2": src.get("commod2", ""),
                    "source_mrds_commod3": src.get("commod3", ""),
                    "distance_to_source_mrds_km": distance,
                    "distance_to_nearest_positive_km": nearest_positive,
                    "negative_rank_within_state_type": len(accepted) + 1,
                }
            )

        if len(accepted) < n_target:
            print(f"Warning: requested {n_target} nearby negatives for {state}, sampled {len(accepted)}.")
        chunks.append(pd.DataFrame(accepted))

    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True)


def build_samples(
    config: dict,
    scheme_name: str,
    target_kind: str,
    positive_csv_name: str,
    negative_ratio: float,
    neutral_ratio: float,
    min_negative_source_distance_km: float,
    max_negative_source_distance_km: float,
    min_negative_positive_distance_km: float,
    min_neutral_positive_distance_km: float,
    min_neutral_mrds_distance_km: float,
    seed: int,
) -> tuple[pd.DataFrame, dict]:
    positives, western_states, positive_source = base_positive_samples(config, target_kind, positive_csv_name)
    positives = positives.copy()
    positives["distance_to_nearest_positive_km"] = 0.0
    positives["distance_to_nearest_mrds_km"] = 0.0
    positives["negative_rank_within_state_type"] = 0

    candidate_pool = target_candidate_pool(
        config=config,
        positives=positives,
        western_states=western_states,
        target_kind=target_kind,
        min_positive_distance_km=min_negative_positive_distance_km,
    )

    negatives = sample_known_mining_nearby_negatives(
        candidate_pool=candidate_pool,
        positives=positives,
        states=western_states,
        ratio=negative_ratio,
        min_source_distance_km=min_negative_source_distance_km,
        max_source_distance_km=max_negative_source_distance_km,
        min_positive_distance_km=min_negative_positive_distance_km,
        seed=seed,
    )
    negatives["distance_to_nearest_mrds_km"] = negatives["distance_to_source_mrds_km"]

    mrds = pd.read_csv(config["mines"]["mrds_full"], low_memory=False)
    mrds = clean_lat_lon(mrds, "latitude", "longitude")
    neutrals = stage02_samples.build_background_negatives(
        positives=positives,
        mrds_points=mrds,
        shapefile_path=stage02_samples.default_state_shapefile(config),
        ratio=neutral_ratio,
        min_positive_distance_km=min_neutral_positive_distance_km,
        min_mrds_distance_km=min_neutral_mrds_distance_km,
        seed=seed + 1,
    )
    neutrals["negative_rank_within_state_type"] = neutrals.groupby("state").cumcount() + 1
    neutrals["sample_role_note"] = "No known MRDS point nearby; treated as neutral/unlabeled, not hard negative."

    positive_samples = add_sample_fields(positives, "positive", 1)
    negative_samples = add_sample_fields(negatives, "negative", 0, "known_mining_area_nearby_negative")
    neutral_samples = add_sample_fields(neutrals, "neutral", -1, "neutral_unlabeled_background")

    all_columns = list(
        dict.fromkeys(
            list(positive_samples.columns) + list(negative_samples.columns) + list(neutral_samples.columns)
        )
    )
    samples = pd.concat(
        [
            positive_samples.reindex(columns=all_columns),
            negative_samples.reindex(columns=all_columns),
            neutral_samples.reindex(columns=all_columns),
        ],
        ignore_index=True,
    )
    samples = normalize_object_columns(samples)

    summary = {
        "scheme": scheme_name,
        "target_kind": target_kind,
        "positive_source": positive_source,
        "candidate_pool_rows_after_target_filter": int(len(candidate_pool)),
        "western_core_states": western_states,
        "positive_rows": int((samples["Y_label"] == 1).sum()),
        "negative_rows": int((samples["Y_label"] == 0).sum()),
        "neutral_rows": int((samples["Y_label"] == -1).sum()),
        "negative_ratio": negative_ratio,
        "neutral_ratio": neutral_ratio,
        "negative_distance_rule": {
            "source": "random point in annulus around non-target MRDS mining record",
            "min_source_distance_km": min_negative_source_distance_km,
            "max_source_distance_km": max_negative_source_distance_km,
            "min_distance_to_positive_target_km": min_negative_positive_distance_km,
        },
        "neutral_distance_rule": {
            "source": "random point inside western-core state boundaries",
            "min_distance_to_positive_target_km": min_neutral_positive_distance_km,
            "min_distance_to_any_mrds_km": min_neutral_mrds_distance_km,
        },
        "label_counts": samples["Y_label"].value_counts(dropna=False).sort_index().to_dict(),
        "sample_type_counts": samples["sample_type"].value_counts(dropna=False).to_dict(),
        "negative_type_counts": samples["negative_type"].value_counts(dropna=False).to_dict(),
    }
    return samples, summary


def align_features(samples: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, dict]:
    features, logs = stage07_ratios.align_base_geochem_gravity(samples, config)
    features, terrain_logs = stage07_ratios.add_terrain_geology(features, config)
    logs.extend(terrain_logs)
    features, climate_logs = stage07_ratios.add_climate(features, config)
    logs.extend(climate_logs)
    features, gravity_logs = stage07_ratios.add_cmmi_gravity(features, config)
    logs.extend(gravity_logs)
    features, env_summary = stage07_ratios.add_environment(features, config)
    return features, {"alignment_logs": logs, "environment_summary": env_summary}


def build_model_datasets(features: pd.DataFrame, scheme_dir: Path, scheme_name: str) -> dict:
    feature_cols, info = stage04_dataset.select_feature_columns(
        features,
        stage04_dataset.FEATURE_SETS["all_features"],
        missing_threshold=0.70,
    )
    meta_cols = [
        c
        for c in stage04_dataset.META_COLUMNS
        + ["ratio_label", "sample_role_note"]
        if c in features.columns
    ]

    all_dataset = features[meta_cols + feature_cols].copy()
    supervised_dataset = all_dataset[all_dataset["Y_label"].isin([0, 1])].copy().reset_index(drop=True)

    all_base = f"model_dataset_{scheme_name}_with_neutral_all_features_v1"
    supervised_base = f"model_dataset_{scheme_name}_supervised_all_features_v1"

    write_dataframe(all_dataset, scheme_dir / f"{all_base}.parquet")
    write_dataframe(all_dataset, scheme_dir / f"{all_base}.csv")
    write_dataframe(supervised_dataset, scheme_dir / f"{supervised_base}.parquet")
    write_dataframe(supervised_dataset, scheme_dir / f"{supervised_base}.csv")
    (scheme_dir / f"{all_base}_features.txt").write_text("\n".join(feature_cols) + "\n", encoding="utf-8")
    (scheme_dir / f"{supervised_base}_features.txt").write_text("\n".join(feature_cols) + "\n", encoding="utf-8")

    return {
        "feature_columns": int(len(feature_cols)),
        "feature_selection": info,
        "all_dataset": {
            "rows": int(len(all_dataset)),
            "path": str(scheme_dir / f"{all_base}.parquet"),
            "label_counts": all_dataset["Y_label"].value_counts(dropna=False).sort_index().to_dict(),
        },
        "supervised_dataset": {
            "rows": int(len(supervised_dataset)),
            "path": str(scheme_dir / f"{supervised_base}.parquet"),
            "label_counts": supervised_dataset["Y_label"].value_counts(dropna=False).sort_index().to_dict(),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build known-mining-nearby negative + neutral background dataset.")
    parser.add_argument("--scheme-name", default=DEFAULT_SCHEME_NAME)
    parser.add_argument(
        "--target-kind",
        default="porphyry_copper",
        choices=[
            "porphyry_copper",
            "uranium",
            "epithermal_au_ag",
            "polymetallic_pb_zn_ag_cu",
            "w_skarn_veins",
            "hot_spring_hg_au_ag",
        ],
        help="Target mineral/deposit family used for positives and target exclusion.",
    )
    parser.add_argument(
        "--positive-csv-name",
        default="",
        help="CSV file name under data_root for target positives. Uranium defaults to 矿点-铀.csv.",
    )
    parser.add_argument(
        "--output-subdir",
        default="",
        help="Optional subdirectory under outputs/model_datasets/by_sample_scheme.",
    )
    parser.add_argument("--negative-ratio", type=float, default=10.0)
    parser.add_argument("--neutral-ratio", type=float, default=10.0)
    parser.add_argument("--min-negative-source-distance-km", type=float, default=5.0)
    parser.add_argument("--max-negative-source-distance-km", type=float, default=30.0)
    parser.add_argument("--min-negative-positive-distance-km", type=float, default=20.0)
    parser.add_argument("--min-neutral-positive-distance-km", type=float, default=30.0)
    parser.add_argument("--min-neutral-mrds-distance-km", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=20260706)
    parser.add_argument("--samples-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config()
    ensure_project_dirs(config)
    if args.output_subdir:
        output_root = DATASET_ROOT / args.output_subdir
    elif args.target_kind == "porphyry_copper":
        output_root = OUT_ROOT
    else:
        output_root = DATASET_ROOT / "by_target" / args.target_kind / "known_mining_neutral"
    scheme_dir = output_root / args.scheme_name
    scheme_dir.mkdir(parents=True, exist_ok=True)

    samples, sample_summary = build_samples(
        config=config,
        scheme_name=args.scheme_name,
        target_kind=args.target_kind,
        positive_csv_name=args.positive_csv_name,
        negative_ratio=args.negative_ratio,
        neutral_ratio=args.neutral_ratio,
        min_negative_source_distance_km=args.min_negative_source_distance_km,
        max_negative_source_distance_km=args.max_negative_source_distance_km,
        min_negative_positive_distance_km=args.min_negative_positive_distance_km,
        min_neutral_positive_distance_km=args.min_neutral_positive_distance_km,
        min_neutral_mrds_distance_km=args.min_neutral_mrds_distance_km,
        seed=args.seed,
    )
    samples["ratio_label"] = args.scheme_name
    write_dataframe(samples, scheme_dir / f"samples_{args.scheme_name}.parquet")
    write_dataframe(samples, scheme_dir / f"samples_{args.scheme_name}.csv")

    summary = {"samples": sample_summary}
    if not args.samples_only:
        features, align_summary = align_features(samples, config)
        write_dataframe(features, scheme_dir / f"model_features_{args.scheme_name}.parquet")
        write_dataframe(features, scheme_dir / f"model_features_{args.scheme_name}.csv")
        summary["alignment"] = align_summary
        summary["model_datasets"] = build_model_datasets(features, scheme_dir, args.scheme_name)

    (scheme_dir / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote known-mining neutral sample scheme: {scheme_dir}")
    print(json.dumps(sample_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
