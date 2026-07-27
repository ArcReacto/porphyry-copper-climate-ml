from __future__ import annotations

import json

from decoupling_utils import INPUT_CSV, ensure_output_dir, define_feature_roles, load_main_dataset


def main() -> None:
    out_dir = ensure_output_dir()
    df = load_main_dataset()
    role_table = define_feature_roles(df)

    role_table.to_csv(out_dir / "feature_role_table.csv", index=False, encoding="utf-8-sig")

    profile = {
        "input_csv": str(INPUT_CSV),
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "n_positive": int(df["Y_label"].sum()),
        "n_negative": int((df["Y_label"] == 0).sum()),
        "roles": role_table["role"].value_counts().to_dict(),
        "numeric_roles": role_table.loc[role_table["is_numeric"], "role"].value_counts().to_dict(),
        "m1_full_climate_feature_count": int(role_table["used_in_m1_full_climate"].sum()),
        "m2_no_climate_feature_count": int(role_table["used_in_m2_no_climate"].sum()),
        "m3_climate_normalized_source_feature_count": int(
            role_table["used_in_m3_climate_normalized"].sum()
        ),
        "climate_adjuster_count": int(role_table["used_as_climate_adjuster"].sum()),
    }
    (out_dir / "input_profile.json").write_text(
        json.dumps(profile, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Wrote feature role table")
    print(out_dir / "feature_role_table.csv")
    print(out_dir / "input_profile.json")


if __name__ == "__main__":
    main()
