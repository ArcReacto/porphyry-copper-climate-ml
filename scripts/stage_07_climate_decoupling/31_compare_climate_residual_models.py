from __future__ import annotations

import json

import pandas as pd

from decoupling_utils import (
    ensure_output_dir,
    define_feature_roles,
    evaluate_splitter,
    load_main_dataset,
    make_default_splitters,
    summarize_metrics,
)


def main() -> None:
    out_dir = ensure_output_dir()
    df = load_main_dataset()
    role_table = define_feature_roles(df)

    all_metrics = []
    all_predictions = []
    for cv_name, splitter, groups in make_default_splitters(df):
        print(f"Running {cv_name}")
        metrics, predictions = evaluate_splitter(
            df=df,
            role_table=role_table,
            splitter=splitter,
            cv_name=cv_name,
            groups=groups,
        )
        all_metrics.append(metrics)
        all_predictions.append(predictions)

    metrics = pd.concat(all_metrics, ignore_index=True)
    predictions = pd.concat(all_predictions, ignore_index=True)
    summary = summarize_metrics(metrics)

    metrics.to_csv(out_dir / "climate_decoupling_cv_metrics.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(
        out_dir / "climate_decoupling_sample_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary.to_csv(out_dir / "climate_decoupling_cv_summary.csv", index=False, encoding="utf-8-sig")

    report = {
        "model_meaning": {
            "M1_Full_Climate": "raw geochemistry + geo/structure/background + climate variables",
            "M2_No_Climate": "raw geochemistry + geo/structure/background, climate variables removed",
            "M3_Climate_Normalized": "climate-residualized geochemistry + geo/structure/background, no raw climate input",
            "M4_Sensitive_Residualized": "only climate-sensitive geochemistry residualized; other geochemistry kept raw; no raw climate input",
        },
        "outputs": [
            "climate_decoupling_cv_metrics.csv",
            "climate_decoupling_cv_summary.csv",
            "climate_decoupling_sample_predictions.csv",
        ],
        "notes": [
            "M3 residualization is fitted inside each fold using only the training split.",
            "M4 residualizes only elements with |Spearman r| >= 0.30 against annual/summary climate variables.",
            "The classifier is a class-weighted RandomForestClassifier shared by M1/M2/M3/M4.",
        ],
    }
    (out_dir / "climate_decoupling_model_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("Wrote climate decoupling model comparison")
    print(out_dir / "climate_decoupling_cv_summary.csv")


if __name__ == "__main__":
    main()
