from __future__ import annotations

import pandas as pd

from decoupling_utils import (
    ensure_output_dir,
    define_feature_roles,
    evaluate_splitter,
    load_main_dataset,
    make_leave_one_group_splitter,
    summarize_metrics,
)


def run_leave_one(df: pd.DataFrame, role_table: pd.DataFrame, group_col: str, cv_name: str):
    groups = df[group_col].fillna("unknown").astype(str)
    splitter, groups = make_leave_one_group_splitter(groups)
    return evaluate_splitter(
        df=df,
        role_table=role_table,
        splitter=splitter,
        cv_name=cv_name,
        groups=groups,
    )


def main() -> None:
    out_dir = ensure_output_dir()
    df = load_main_dataset()
    role_table = define_feature_roles(df)

    all_metrics = []
    all_predictions = []

    if "state" in df.columns:
        print("Running leave_one_state_out")
        metrics, predictions = run_leave_one(df, role_table, "state", "leave_one_state_out")
        all_metrics.append(metrics)
        all_predictions.append(predictions)

    if "env_causal_group_id" in df.columns:
        print("Running leave_one_environment_group_out")
        metrics, predictions = run_leave_one(
            df,
            role_table,
            "env_causal_group_id",
            "leave_one_environment_group_out",
        )
        all_metrics.append(metrics)
        all_predictions.append(predictions)

    metrics = pd.concat(all_metrics, ignore_index=True)
    predictions = pd.concat(all_predictions, ignore_index=True)
    summary = summarize_metrics(metrics, group_cols=["cv", "test_group", "model"])
    overall = summarize_metrics(metrics, group_cols=["cv", "model"])

    metrics.to_csv(out_dir / "leave_region_out_metrics.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(out_dir / "leave_region_out_predictions.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out_dir / "leave_region_out_group_summary.csv", index=False, encoding="utf-8-sig")
    overall.to_csv(out_dir / "leave_region_out_overall_summary.csv", index=False, encoding="utf-8-sig")

    print("Wrote leave-region-out validation")
    print(out_dir / "leave_region_out_overall_summary.csv")


if __name__ == "__main__":
    main()
