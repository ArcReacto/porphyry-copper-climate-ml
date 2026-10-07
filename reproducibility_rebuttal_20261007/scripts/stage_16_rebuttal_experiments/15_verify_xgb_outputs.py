"""Check sample/fold alignment and shared 1:10 XGBoost results."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/rebuttal_experiments/xgboost"
EXPECTED = {"M2_NoClimate_XGB": 0.7009790953121235, "M4_Spearman_XGB": 0.78683837465064}


def check_mean(path: Path, model: str, expected: float, filters: dict | None = None) -> None:
    frame = pd.read_csv(path)
    for column, value in (filters or {}).items():
        frame = frame.loc[frame[column].eq(value)]
    frame = frame.loc[frame["model"].eq(model)]
    if len(frame) != 5:
        raise AssertionError(f"Expected five folds for {model} in {path}; got {len(frame)}")
    actual = frame["average_precision"].mean()
    if not np.isclose(actual, expected, atol=1e-10):
        raise AssertionError(f"AP mismatch for {model} in {path}: {actual}")


def check_oof(path: Path, keys: list[str], models: set[str], rows: int) -> None:
    frame = pd.read_csv(path, usecols=keys + ["model", "row_index"])
    for key, part in frame.groupby(keys + ["model"], dropna=False):
        model = key[-1] if isinstance(key, tuple) else key
        if model not in models:
            continue
        if len(part) != rows or part["row_index"].duplicated().any():
            raise AssertionError(f"OOF coverage mismatch in {path}: {key}")


def main() -> None:
    checks = [
        ("01_fold_local_graphunion/fold_metrics.csv", {}),
        ("03_negative_source_sensitivity/negative_source_fold_metrics.csv", {"negative_scheme": "hard_only"}),
        ("05_final_ratio_sensitivity/ratio_fold_metrics.csv", {"negative_ratio": "1:10"}),
        ("06_deposit_dedup_buffer_audit/buffered_fold_metrics.csv", {"buffer_km": 0.0}),
        ("08_random_matched_residualization/random_control_fold_metrics.csv", {"random_repeat": None}),
        ("09_controlled_climate_contamination/controlled_contamination_fold_metrics.csv", {"gamma": 0.0}),
    ]
    for relative, filters in checks:
        if "random_repeat" in filters:
            frame = pd.read_csv(OUT / relative)
            frame = frame.loc[frame["random_repeat"].isna()]
            for model, expected in EXPECTED.items():
                current = frame.loc[frame["model"].eq(model.replace("M2_NoClimate", "NoClimate"))]
                if len(current) != 5 or not np.isclose(current["average_precision"].mean(), expected, atol=1e-10):
                    raise AssertionError(f"Random-control deterministic result mismatch: {model}")
            continue
        if "gamma" in filters:
            frame = pd.read_csv(OUT / relative)
            frame = frame.loc[frame["gamma"].eq(0.0)]
            mapping = {"M2_NoClimate_XGB": "M2_raw_XGB", "M4_Spearman_XGB": "M4_spearman_XGB"}
            for model, expected in EXPECTED.items():
                current = frame.loc[frame["model"].eq(mapping[model])]
                if len(current) != 5 or not np.isclose(current["average_precision"].mean(), expected, atol=1e-10):
                    raise AssertionError(f"Controlled-contamination gamma=0 mismatch: {model}")
            continue
        for model, expected in EXPECTED.items():
            displayed = model.replace("M2_NoClimate", "NoClimate") if relative.startswith(("03", "05", "06")) else model
            check_mean(OUT / relative, displayed, expected, filters)

    check_oof(OUT / "01_fold_local_graphunion/oof_predictions.csv", [], set(EXPECTED), 1738)
    check_oof(OUT / "02_spatial_block_validation/spatial_oof_predictions.csv", ["grid_degrees"], set(EXPECTED), 1738)
    matched = pd.read_csv(OUT / "10_matched_feature_baselines/matched_feature_fold_metrics.csv")
    for adjustment, expected in [("raw", EXPECTED["M2_NoClimate_XGB"]),
                                 ("m4_spearman", EXPECTED["M4_Spearman_XGB"])]:
        current = matched.loc[matched["feature_view"].eq("multimodal_no_climate")
                              & matched["adjustment"].eq(adjustment)]
        if len(current) != 5 or not np.isclose(current["average_precision"].mean(), expected, atol=1e-10):
            raise AssertionError(f"Matched-feature result mismatch: {adjustment}")
    random_control = pd.read_csv(OUT / "08_random_matched_residualization/m4_vs_random_control.csv")
    if not random_control["random_repetitions"].eq(100).all():
        raise AssertionError("Expected 100 matched-random repetitions")
    proxy = pd.read_csv(OUT / "11_distinct_deposit_topk/distinct_deposit_pooled_foldwise.csv")
    if not proxy["distinct_positive_deposits"].eq(102).all():
        raise AssertionError("Positive-sample proxy-cluster count changed")
    paleometa = json.loads((OUT / "12_paleoclimate_validation/model_run_manifest.json").read_text(encoding="utf-8"))
    if paleometa["sample_count"] != 1729 or paleometa["positive_count"] != 158 or paleometa["model_family"] != "xgboost":
        raise AssertionError("Paleoclimate common-sample manifest mismatch")
    print("PASS: XGBoost fold metrics, OOF coverage, and paleoclimate sample manifest are consistent")


if __name__ == "__main__":
    main()
