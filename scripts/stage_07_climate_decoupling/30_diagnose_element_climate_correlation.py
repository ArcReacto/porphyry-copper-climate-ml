from __future__ import annotations

import pandas as pd

from decoupling_utils import (
    ensure_output_dir,
    climate_adjuster_columns,
    columns_by_role,
    define_feature_roles,
    load_main_dataset,
)


MIN_PAIR_COUNT = 30
SENSITIVE_THRESHOLD = 0.30


def main() -> None:
    out_dir = ensure_output_dir()
    df = load_main_dataset()
    role_table = define_feature_roles(df)
    element_cols = columns_by_role(role_table, "geochemistry")
    climate_cols = climate_adjuster_columns(role_table)

    selected = df[element_cols + climate_cols]
    pair_counts = selected[element_cols].notna().astype(int).T.dot(
        selected[climate_cols].notna().astype(int)
    )
    pearson = selected.corr(method="pearson", min_periods=MIN_PAIR_COUNT).loc[element_cols, climate_cols]
    spearman = selected.corr(method="spearman", min_periods=MIN_PAIR_COUNT).loc[element_cols, climate_cols]

    rows = []
    for element in element_cols:
        for climate in climate_cols:
            pearson_r = pearson.loc[element, climate]
            spearman_r = spearman.loc[element, climate]
            n_pair = int(pair_counts.loc[element, climate])
            rows.append(
                {
                    "element_feature": element,
                    "climate_feature": climate,
                    "n_pair": n_pair,
                    "pearson_r": pearson_r,
                    "pearson_abs": abs(pearson_r) if pd.notna(pearson_r) else pd.NA,
                    "spearman_r": spearman_r,
                    "spearman_abs": abs(spearman_r) if pd.notna(spearman_r) else pd.NA,
                }
            )

    corr = pd.DataFrame(rows)
    corr.to_csv(out_dir / "element_climate_correlation.csv", index=False, encoding="utf-8-sig")

    summary_rows = []
    for element, part in corr.groupby("element_feature"):
        best_spearman = part.sort_values("spearman_abs", ascending=False).head(1)
        best_pearson = part.sort_values("pearson_abs", ascending=False).head(1)
        if best_spearman.empty:
            continue
        bs = best_spearman.iloc[0]
        bp = best_pearson.iloc[0]
        summary_rows.append(
            {
                "element_feature": element,
                "best_climate_feature_spearman": bs["climate_feature"],
                "spearman_r_at_best": bs["spearman_r"],
                "spearman_abs_max": bs["spearman_abs"],
                "best_climate_feature_pearson": bp["climate_feature"],
                "pearson_r_at_best": bp["pearson_r"],
                "pearson_abs_max": bp["pearson_abs"],
                "is_climate_sensitive": bool(bs["spearman_abs"] >= SENSITIVE_THRESHOLD),
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values("spearman_abs_max", ascending=False)
    summary.to_csv(out_dir / "top_climate_sensitive_elements.csv", index=False, encoding="utf-8-sig")

    print("Wrote element-climate correlation diagnostics")
    print(out_dir / "element_climate_correlation.csv")
    print(out_dir / "top_climate_sensitive_elements.csv")


if __name__ == "__main__":
    main()
