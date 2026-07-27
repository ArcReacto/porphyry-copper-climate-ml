from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.io_utils import ensure_project_dirs, load_config, read_table, write_dataframe  # noqa: E402
from src.spatial_utils import clean_lat_lon  # noqa: E402


DEFAULT_SCHEME_NAME = "global_copper_strict_new_ratio_1_10"
DEFAULT_OUTPUT_SUBDIR = "global_copper_strict_new_incremental"
STRICT_NEW_CSV = PROJECT_ROOT / "outputs" / "global_copper_catalog" / "western_core_porphyry_candidates_strict_new.csv"
DATASET_ROOT = PROJECT_ROOT / "outputs" / "model_datasets" / "by_sample_scheme"


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
stage09_known = load_script_module(
    "stage09_known",
    PROJECT_ROOT / "scripts" / "stage_02_sampling_and_region" / "09_build_known_mining_neutral_dataset.py",
)


def normalize_object_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == "object":
            out[col] = out[col].astype("string")
    return out


def strict_new_to_positive_rows(strict_new: pd.DataFrame) -> pd.DataFrame:
    rows = pd.DataFrame()
    rows["dep_id"] = strict_new["candidate_cluster_id"].fillna("").astype(str)
    rows["mrds_id"] = strict_new["mrds_dep_id"].fillna("").astype(str)
    rows["site_name"] = strict_new["deposit_name"].fillna("").astype(str)
    rows["latitude"] = pd.to_numeric(strict_new["latitude"], errors="coerce")
    rows["longitude"] = pd.to_numeric(strict_new["longitude"], errors="coerce")
    rows["country"] = "United States"
    rows["state"] = strict_new["western_core_state"].fillna("").astype(str)
    rows["county"] = ""
    rows["commod1"] = "Cu"
    rows["commod2"] = "Mo"
    rows["commod3"] = ""
    rows["model"] = "porphyry copper"
    rows["dep_type"] = strict_new["deposit_type"].fillna("porphyry").astype(str)
    rows["prod_size"] = ""
    rows["mine_record_type"] = "positive_from_global_copper_strict_new"
    rows["background_source"] = ""
    rows["dev_stat"] = ""
    rows["models_parsed"] = "porphyry copper"
    rows["is_porphyry_cu"] = True
    rows["has_copper_commodity"] = True
    rows["score"] = ""
    rows["url"] = ""
    rows["mas_id"] = ""
    rows["region"] = ""
    rows["com_type"] = ""
    rows["oper_type"] = ""
    rows["ore"] = ""
    rows["gangue"] = ""
    rows["other_matl"] = ""
    rows["orebody_fm"] = ""
    rows["work_type"] = ""
    rows["alteration"] = ""
    rows["conc_proc"] = ""
    rows["names"] = strict_new["deposit_name"].fillna("").astype(str)
    rows["source_dataset"] = strict_new["source_dataset"].fillna("").astype(str)
    rows["source_file"] = strict_new["source_file"].fillna("").astype(str)
    rows["source_id"] = strict_new["source_id"].fillna("").astype(str)
    rows["source_mrds_dep_id"] = strict_new["mrds_dep_id"].fillna("").astype(str)
    rows["source_record_key"] = strict_new["source_record_key"].fillna("").astype(str)
    rows["candidate_cluster_id"] = strict_new["candidate_cluster_id"].fillna("").astype(str)
    rows["candidate_cluster_size"] = strict_new["candidate_cluster_size"]
    rows["candidate_cluster_sources"] = strict_new["candidate_cluster_sources"].fillna("").astype(str)
    rows["nearest_existing_positive_name"] = strict_new["nearest_existing_name"].fillna("").astype(str)
    rows["nearest_existing_positive_distance_km_before_merge"] = pd.to_numeric(
        strict_new["nearest_existing_distance_km"],
        errors="coerce",
    )
    rows = clean_lat_lon(rows, "latitude", "longitude")
    rows = rows[rows["state"].ne("")].copy()
    return rows


def load_augmented_positives(config: dict, strict_new_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[str], str]:
    base_positives, western_states, base_source = stage09_known.base_porphyry_positive_samples(config)
    strict_new = pd.read_csv(strict_new_path, low_memory=False)
    strict_positive_rows = strict_new_to_positive_rows(strict_new)
    strict_positive_rows = strict_positive_rows[strict_positive_rows["state"].isin(western_states)].copy()

    base_positives = base_positives.copy()
    base_positives["positive_origin"] = "existing_western_core_porphyry"
    strict_positive_rows["positive_origin"] = "global_copper_strict_new"

    all_columns = list(dict.fromkeys(list(base_positives.columns) + list(strict_positive_rows.columns)))
    positives = pd.concat(
        [
            base_positives.reindex(columns=all_columns),
            strict_positive_rows.reindex(columns=all_columns),
        ],
        ignore_index=True,
    )
    positives = positives.drop_duplicates(subset=["dep_id"], keep="first").reset_index(drop=True)
    positive_source = f"{base_source}; {strict_new_path}"
    return positives, strict_positive_rows, western_states, positive_source


def build_incremental_samples(
    config: dict,
    strict_new_path: Path,
    scheme_name: str,
    negative_ratio: float,
    neutral_ratio: float,
    min_negative_source_distance_km: float,
    max_negative_source_distance_km: float,
    min_negative_positive_distance_km: float,
    min_neutral_positive_distance_km: float,
    min_neutral_mrds_distance_km: float,
    seed: int,
) -> tuple[pd.DataFrame, dict]:
    positives, strict_rows, western_states, positive_source = load_augmented_positives(config, strict_new_path)
    positives = positives.copy()
    positives["distance_to_nearest_positive_km"] = 0.0
    positives["distance_to_nearest_mrds_km"] = 0.0
    positives["negative_rank_within_state_type"] = 0

    candidate_pool = read_table(PROJECT_ROOT / "data_intermediate" / "mrds_negative_candidate_pool.parquet")
    candidate_pool = clean_lat_lon(candidate_pool, "latitude", "longitude")
    candidate_pool = candidate_pool[candidate_pool["state"].isin(western_states)].copy()
    strict_source_dep_ids = set(strict_rows["source_mrds_dep_id"].dropna().astype(str))
    if strict_source_dep_ids and "dep_id" in candidate_pool.columns:
        candidate_pool = candidate_pool[~candidate_pool["dep_id"].astype(str).isin(strict_source_dep_ids)].copy()
    candidate_pool["distance_to_nearest_positive_km"] = stage02_samples.nearest_positive_distance_km(
        candidate_pool,
        positives,
    )
    candidate_pool = candidate_pool[
        candidate_pool["distance_to_nearest_positive_km"] >= min_negative_positive_distance_km
    ].copy()

    negatives = stage09_known.sample_known_mining_nearby_negatives(
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

    positive_samples = stage09_known.add_sample_fields(positives, "positive", 1)
    negative_samples = stage09_known.add_sample_fields(negatives, "negative", 0, "known_mining_area_nearby_negative")
    neutral_samples = stage09_known.add_sample_fields(neutrals, "neutral", -1, "neutral_unlabeled_background")
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
    samples["ratio_label"] = scheme_name

    summary = {
        "scheme": scheme_name,
        "target_kind": "porphyry_copper_plus_global_strict_new",
        "positive_source": positive_source,
        "strict_new_positive_source": str(strict_new_path),
        "base_positive_rows": int(len(positives) - len(strict_rows)),
        "strict_new_positive_rows": int(len(strict_rows)),
        "candidate_pool_rows_after_augmented_positive_filter": int(len(candidate_pool)),
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
            "min_distance_to_augmented_positive_target_km": min_negative_positive_distance_km,
        },
        "neutral_distance_rule": {
            "source": "random point inside western-core state boundaries",
            "min_distance_to_augmented_positive_target_km": min_neutral_positive_distance_km,
            "min_distance_to_any_mrds_km": min_neutral_mrds_distance_km,
        },
        "label_counts": samples["Y_label"].value_counts(dropna=False).sort_index().to_dict(),
        "sample_type_counts": samples["sample_type"].value_counts(dropna=False).to_dict(),
        "negative_type_counts": samples["negative_type"].value_counts(dropna=False).to_dict(),
        "positive_origin_counts": samples.loc[samples["Y_label"].eq(1), "positive_origin"].value_counts(dropna=False).to_dict()
        if "positive_origin" in samples.columns
        else {},
    }
    return samples, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an augmented western-core dataset with strict-new global copper positives.")
    parser.add_argument("--scheme-name", default=DEFAULT_SCHEME_NAME)
    parser.add_argument("--output-subdir", default=DEFAULT_OUTPUT_SUBDIR)
    parser.add_argument("--strict-new-csv", type=Path, default=STRICT_NEW_CSV)
    parser.add_argument("--negative-ratio", type=float, default=10.0)
    parser.add_argument("--neutral-ratio", type=float, default=10.0)
    parser.add_argument("--min-negative-source-distance-km", type=float, default=5.0)
    parser.add_argument("--max-negative-source-distance-km", type=float, default=30.0)
    parser.add_argument("--min-negative-positive-distance-km", type=float, default=20.0)
    parser.add_argument("--min-neutral-positive-distance-km", type=float, default=30.0)
    parser.add_argument("--min-neutral-mrds-distance-km", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--samples-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config()
    ensure_project_dirs(config)
    if not args.strict_new_csv.exists():
        raise FileNotFoundError(f"Missing strict-new candidate file: {args.strict_new_csv}")

    scheme_dir = DATASET_ROOT / args.output_subdir / args.scheme_name
    scheme_dir.mkdir(parents=True, exist_ok=True)

    samples, sample_summary = build_incremental_samples(
        config=config,
        strict_new_path=args.strict_new_csv,
        scheme_name=args.scheme_name,
        negative_ratio=args.negative_ratio,
        neutral_ratio=args.neutral_ratio,
        min_negative_source_distance_km=args.min_negative_source_distance_km,
        max_negative_source_distance_km=args.max_negative_source_distance_km,
        min_negative_positive_distance_km=args.min_negative_positive_distance_km,
        min_neutral_positive_distance_km=args.min_neutral_positive_distance_km,
        min_neutral_mrds_distance_km=args.min_neutral_mrds_distance_km,
        seed=args.seed,
    )

    write_dataframe(samples, scheme_dir / f"samples_{args.scheme_name}.parquet")
    write_dataframe(samples, scheme_dir / f"samples_{args.scheme_name}.csv")

    summary = {"samples": sample_summary}
    if not args.samples_only:
        features, align_summary = stage09_known.align_features(samples, config)
        write_dataframe(features, scheme_dir / f"model_features_{args.scheme_name}.parquet")
        write_dataframe(features, scheme_dir / f"model_features_{args.scheme_name}.csv")
        summary["alignment"] = align_summary
        summary["model_datasets"] = stage09_known.build_model_datasets(features, scheme_dir, args.scheme_name)

    (scheme_dir / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote strict-new incremental dataset: {scheme_dir}")
    print(json.dumps(sample_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
