from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.io_utils import ensure_project_dirs, load_config, output_path, write_dataframe, write_json


MODEL_DATASET_DIR = "model_datasets"
CAUSAL_GRAPH_DIR = "causal_graph"
FEATURE_LIST_NAME = "model_dataset_western_core_all_features_v1_features.txt"


def mapping_rows() -> list[dict[str, str | bool]]:
    rows: list[dict[str, str | bool]] = []

    def add(
        feature_pattern: str,
        concept: str,
        concept_group: str,
        data_source: str,
        aggregation_rule: str,
        role: str,
        note: str,
        match_type: str = "prefix",
        include_in_concept_features: bool = True,
    ) -> None:
        rows.append(
            {
                "feature_pattern": feature_pattern,
                "match_type": match_type,
                "concept": concept,
                "concept_group": concept_group,
                "data_source": data_source,
                "aggregation_rule": aggregation_rule,
                "role": role,
                "include_in_concept_features": include_in_concept_features,
                "note": note,
            }
        )

    geochem_sources = [
        ("geochem1_usgs", "USGS geochem1", {
            "Cu": "Cu_value",
            "Mo": "Mo_value",
            "Au": "Au_value",
            "Ag": "Ag_value",
            "Pb": "Pb_value",
            "Zn": "Zn_value",
            "As": "As_value",
            "Sb": "Sb_value",
            "Bi": "Bi_value",
            "W": "W_value",
            "Re": "Re_value",
        }),
        ("geochem2_nure", "NURE geochem2", {
            "Cu": "Cu_ppm",
            "Mo": "Mo_ppm",
            "Au": "Au_sq_ppm",
            "Ag": "Ag_ppm",
            "Pb": "Pb_ppm",
            "Zn": "Zn_ppm",
            "As": "As_ppm",
            "Sb": "Sb_ppm",
            "Bi": "Bi_ppm",
            "W": "W_ppm",
            "Re": "Re_ppm",
        }),
    ]
    element_concepts = {
        "Cu": ("Cu_anomaly", "mineralization_signal", "main ore element for porphyry copper"),
        "Mo": ("Mo_anomaly", "mineralization_signal", "common porphyry copper companion element"),
        "Au": ("Au_anomaly", "mineralization_signal", "gold companion signal in some porphyry systems"),
        "Ag": ("Ag_anomaly", "pathfinder_signal", "silver companion/pathfinder signal"),
        "Pb": ("Pb_Zn_background", "geochemical_background", "lead-zinc background or peripheral signal"),
        "Zn": ("Pb_Zn_background", "geochemical_background", "lead-zinc background or peripheral signal"),
        "As": ("As_Sb_Bi_pathfinder", "pathfinder_signal", "arsenic pathfinder; may be weathering sensitive"),
        "Sb": ("As_Sb_Bi_pathfinder", "pathfinder_signal", "antimony pathfinder; may be weathering sensitive"),
        "Bi": ("As_Sb_Bi_pathfinder", "pathfinder_signal", "bismuth pathfinder associated with hydrothermal systems"),
        "W": ("W_Re_pathfinder", "pathfinder_signal", "tungsten-rhenium pathfinder group"),
        "Re": ("W_Re_pathfinder", "pathfinder_signal", "rhenium pathfinder group; often sparse/nondetect"),
    }
    for source_prefix, source_name, element_fields in geochem_sources:
        for element, field_token in element_fields.items():
            concept, role, note = element_concepts[element]
            add(
                f"{source_prefix}_{field_token}_",
                concept,
                "geochemistry",
                source_name,
                "nearest/mean/median/max/p95 across radii; aggregate to concept with robust max or p95",
                role,
                note,
            )

    add(
        "gravity_na_grav_anom_",
        "regional_gravity_anomaly",
        "geophysics",
        "NOAA gravity",
        "nearest/mean/median/max/p95 across radii",
        "geophysical_context",
        "regional Bouguer/free-air gravity anomaly context",
    )
    add(
        "gravity_cmmi_hgm_nearest",
        "gravity_gradient",
        "geophysics",
        "CMMI gravity derivatives",
        "direct nearest raster value",
        "geophysical_boundary",
        "horizontal gradient magnitude; proxy for geophysical boundaries",
        match_type="exact",
    )
    add(
        "gravity_cmmi_up30km_nearest",
        "deep_regional_gravity_anomaly",
        "geophysics",
        "CMMI gravity derivatives",
        "direct nearest raster value",
        "deep_geophysical_context",
        "30 km upward continued gravity anomaly",
        match_type="exact",
    )
    add(
        "gravity_cmmi_up30km_hgm_nearest",
        "deep_gravity_gradient",
        "geophysics",
        "CMMI gravity derivatives",
        "direct nearest raster value",
        "deep_geophysical_boundary",
        "horizontal gradient of 30 km upward continued gravity",
        match_type="exact",
    )
    for depth in ["shallow", "deep"]:
        source = "CMMI shallow gravity-source worms" if depth == "shallow" else "CMMI deep gravity-source worms"
        add(
            f"gravity_cmmi_{depth}_worms_nearest_distance_km",
            f"{depth}_gravity_source_proximity",
            "geophysics",
            source,
            "direct nearest distance; later invert/sign so smaller distance means stronger proximity",
            "geophysical_source_proximity",
            f"distance to nearest {depth} gravity-source worm point",
            match_type="exact",
        )
        add(
            f"gravity_cmmi_{depth}_worms_steepness_",
            f"{depth}_gravity_source_strength",
            "geophysics",
            source,
            "nearest/mean/max steepness across radii",
            "geophysical_source_strength",
            f"steepness of {depth} gravity-source worm points",
        )
        add(
            f"gravity_cmmi_{depth}_worms_points_",
            f"{depth}_gravity_source_density",
            "geophysics",
            source,
            "point counts across radii",
            "geophysical_source_density",
            f"density/count of {depth} gravity-source worm points",
        )
        add(
            f"gravity_cmmi_{depth}_worms_strike_nearest",
            f"{depth}_gravity_source_orientation",
            "geophysics",
            source,
            "nearest strike angle; usually not used in first concept table unless encoded circularly",
            "orientation_context",
            f"orientation of nearest {depth} gravity-source worm point",
            match_type="exact",
            include_in_concept_features=False,
        )

    add("terrain_elevation_m", "terrain_elevation", "terrain", "SRTM DEM", "direct", "terrain_context", "elevation", match_type="exact")
    add("terrain_slope_deg", "terrain_slope", "terrain", "SRTM DEM", "direct", "erosion_exposure", "slope angle", match_type="exact")
    add("terrain_roughness_3x3_m", "terrain_roughness", "terrain", "SRTM DEM", "direct", "erosion_exposure", "local 3x3 roughness", match_type="exact")
    add("terrain_relief_", "terrain_relief", "terrain", "SRTM DEM", "max/mean across relief scales", "erosion_exposure", "local relief across 250m/1000m/5000m windows")

    add(
        "fault_nearest_distance_km",
        "fault_proximity",
        "structure",
        "CMMI faults",
        "direct nearest distance; later invert/sign so smaller distance means stronger proximity",
        "structural_control",
        "distance to nearest mapped fault",
        match_type="exact",
    )
    add("fault_lines_", "fault_density", "structure", "CMMI faults", "line counts across radii", "structural_control", "fault line count/density")
    add("fault_length_km_", "fault_density", "structure", "CMMI faults", "fault length across radii", "structural_control", "fault length/density")

    add("geology_is_intrusive", "intrusive_geology", "geology", "CMMI geology", "direct binary", "geological_control", "intrusive lithology flag", match_type="exact")
    add("geology_is_igneous", "igneous_geology", "geology", "CMMI geology", "direct binary", "geological_control", "igneous lithology flag", match_type="exact")
    add("geology_is_felsic", "felsic_intermediate_geology", "geology", "CMMI geology", "direct binary", "geological_control", "felsic lithology flag", match_type="exact")
    add("geology_is_intermediate", "felsic_intermediate_geology", "geology", "CMMI geology", "direct binary", "geological_control", "intermediate lithology flag", match_type="exact")
    add("geology_is_mafic", "mafic_geology", "geology", "CMMI geology", "direct binary", "geological_context", "mafic lithology flag", match_type="exact")
    add("geology_is_sedimentary", "sedimentary_geology", "geology", "CMMI geology", "direct binary", "geological_context", "sedimentary lithology flag", match_type="exact")
    add("geology_is_carbonate", "carbonate_geology", "geology", "CMMI geology", "direct binary", "geological_context", "carbonate lithology flag", match_type="exact")
    add("geology_is_metamorphic", "metamorphic_geology", "geology", "CMMI geology", "direct binary", "geological_context", "metamorphic lithology flag", match_type="exact")
    add("geology_is_unconsolidated", "unconsolidated_cover", "geology", "CMMI geology", "direct binary", "cover_context", "unconsolidated cover flag", match_type="exact")

    climate_exact = [
        ("climate_water_balance_annual_mm", "water_balance", "annual water balance", "environment"),
        ("climate_ppt_pet_ratio_annual", "aridity", "precipitation / potential evapotranspiration", "environment"),
        ("climate_aet_pet_ratio_annual", "evapotranspiration_ratio", "actual / potential evapotranspiration", "environment"),
    ]
    for pattern, concept, note, role in climate_exact:
        add(pattern, concept, "climate", "TerraClimate", "direct annual derived metric", role, note, match_type="exact")

    climate_prefixes = [
        ("climate_ppt_", "precipitation", "precipitation monthly/seasonal/annual summaries", "environment"),
        ("climate_pet_", "potential_evapotranspiration", "potential evapotranspiration summaries", "environment"),
        ("climate_aet_", "actual_evapotranspiration", "actual evapotranspiration summaries", "environment"),
        ("climate_def_", "water_deficit", "climatic water deficit summaries", "environment"),
        ("climate_q_", "runoff", "runoff summaries", "environment"),
        ("climate_soil_", "soil_moisture", "soil moisture summaries", "environment"),
        ("climate_srad_", "solar_radiation", "solar radiation summaries", "environment"),
        ("climate_swe_", "snow_influence", "snow water equivalent summaries", "environment"),
        ("climate_tmean_", "mean_temperature", "mean temperature summaries", "environment"),
        ("climate_tmax_", "max_temperature", "maximum temperature summaries", "environment"),
        ("climate_tmin_", "min_temperature", "minimum temperature summaries", "environment"),
        ("climate_dtr_", "diurnal_temperature_range", "diurnal temperature range summaries", "environment"),
        ("climate_vap_", "vapor_pressure", "vapor pressure summaries", "environment"),
        ("climate_vpd_", "vapor_pressure_deficit", "vapor pressure deficit summaries", "environment"),
        ("climate_ws_", "wind_speed", "wind speed summaries", "environment"),
    ]
    for pattern, concept, note, role in climate_prefixes:
        add(pattern, concept, "climate", "TerraClimate", "monthly/seasonal/annual aggregation", role, note)

    metadata_rows = [
        ("Y_label", "Porphyry_Cu_presence", "label", "derived samples", "direct", "target_label", "binary target label"),
        ("env_causal_group", "environment_group", "environment", "derived environment groups", "direct", "grouping_variable", "main climate/weathering environment group"),
        ("env_aridity_class_3", "aridity_class", "environment", "derived environment groups", "direct", "grouping_variable", "3-class aridity grouping"),
        ("env_snow_influence", "snow_environment", "environment", "derived environment groups", "direct", "grouping_variable", "snow influence grouping"),
        ("state", "administrative_region", "metadata", "MRDS / samples", "direct", "grouping_variable", "state used for GroupKFold"),
    ]
    for pattern, concept, group, source, agg, role, note in metadata_rows:
        add(pattern, concept, group, source, agg, role, note, match_type="exact", include_in_concept_features=False)

    return rows


def matches(feature: str, row: dict[str, str | bool]) -> bool:
    pattern = str(row["feature_pattern"])
    match_type = str(row["match_type"])
    if match_type == "exact":
        return feature == pattern
    if match_type == "prefix":
        return feature.startswith(pattern)
    if match_type == "contains":
        return pattern in feature
    raise ValueError(f"Unknown match_type: {match_type}")


def resolve_features(features: list[str], mappings: pd.DataFrame) -> pd.DataFrame:
    resolved = []
    mapping_records = mappings.to_dict(orient="records")
    for feature in features:
        match = next((row for row in mapping_records if matches(feature, row)), None)
        if match is None:
            resolved.append(
                {
                    "feature_name": feature,
                    "mapped": False,
                    "feature_pattern": "",
                    "concept": "UNMAPPED",
                    "concept_group": "UNMAPPED",
                    "data_source": "",
                    "aggregation_rule": "",
                    "role": "",
                    "include_in_concept_features": False,
                    "note": "",
                }
            )
        else:
            resolved.append(
                {
                    "feature_name": feature,
                    "mapped": True,
                    "feature_pattern": match["feature_pattern"],
                    "concept": match["concept"],
                    "concept_group": match["concept_group"],
                    "data_source": match["data_source"],
                    "aggregation_rule": match["aggregation_rule"],
                    "role": match["role"],
                    "include_in_concept_features": bool(match["include_in_concept_features"]),
                    "note": match["note"],
                }
            )
    return pd.DataFrame(resolved)


def main() -> int:
    config = load_config()
    ensure_project_dirs(config)

    feature_path = output_path(config, "outputs", f"{MODEL_DATASET_DIR}/{FEATURE_LIST_NAME}")
    if not feature_path.exists():
        raise FileNotFoundError(f"Missing feature list: {feature_path}. Run scripts/stage_04_model_baseline/09_make_model_dataset.py first.")
    features = [line.strip() for line in feature_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    mappings = pd.DataFrame(mapping_rows())
    resolved = resolve_features(features, mappings)

    output_dir = output_path(config, "outputs", CAUSAL_GRAPH_DIR)
    mapping_path = output_dir / "feature_to_concept_mapping.csv"
    resolved_path = output_dir / "feature_to_concept_resolved.csv"
    summary_path = output_dir / "feature_to_concept_summary.csv"

    summary = (
        resolved.groupby(["concept_group", "concept"], as_index=False)
        .agg(feature_count=("feature_name", "count"), included_count=("include_in_concept_features", "sum"))
        .sort_values(["concept_group", "concept"])
    )

    write_dataframe(mappings, mapping_path)
    write_dataframe(resolved, resolved_path)
    write_dataframe(summary, summary_path)

    unmapped = resolved[~resolved["mapped"]]["feature_name"].tolist()
    log = {
        "input_feature_list": str(feature_path),
        "feature_count": len(features),
        "mapping_rule_count": int(len(mappings)),
        "mapped_feature_count": int(resolved["mapped"].sum()),
        "unmapped_feature_count": int(len(unmapped)),
        "unmapped_features": unmapped,
        "concept_count": int(resolved.loc[resolved["mapped"], "concept"].nunique()),
        "concept_group_counts": resolved.loc[resolved["mapped"], "concept_group"].value_counts().to_dict(),
        "outputs": {
            "mapping": str(mapping_path),
            "resolved": str(resolved_path),
            "summary": str(summary_path),
        },
    }
    write_json(output_path(config, "logs", "15_create_feature_concept_mapping_summary.json"), log)

    print(f"Wrote feature-to-concept mapping rules: {len(mappings)}")
    print(mapping_path)
    print(f"Resolved model features: {len(features)} features, {len(unmapped)} unmapped")
    print(resolved_path)
    print(summary_path)
    if unmapped:
        print("Unmapped features:")
        for feature in unmapped:
            print(f"- {feature}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


