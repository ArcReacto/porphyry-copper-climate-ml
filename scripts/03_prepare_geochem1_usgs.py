from __future__ import annotations

import sys
from pathlib import Path

import duckdb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.io_utils import ensure_project_dirs, load_config, output_path, write_json


def sql_list(values: list[str]) -> str:
    return ", ".join("'" + v.replace("'", "''") + "'" for v in values)


def main() -> int:
    config = load_config()
    ensure_project_dirs(config)
    elements = config["targets"]["elements"]

    geol_path = config["geochem1"]["tables"]["geol"]
    chem_path = config["geochem1"]["tables"]["chem"]
    rank_path = config["geochem1"]["tables"]["parameter_rank"]
    parameter_path = config["geochem1"]["tables"]["parameter"]
    out_wide = output_path(config, "data_intermediate", "geochem1_selected_elements_wide.parquet")
    out_params = output_path(config, "data_intermediate", "geochem1_selected_parameters.parquet")

    con = duckdb.connect(database=":memory:")
    con.execute("PRAGMA threads=4")
    con.execute("PRAGMA memory_limit='8GB'")

    elements_sql = sql_list(elements)
    con.execute(
        f"""
        CREATE TEMP TABLE selected_parameters AS
        SELECT
            pr.species,
            pr.species_name,
            pr.parameter,
            pr.analytic_method,
            pr.bestvalue_rank,
            p.parameter_desc,
            p.param_count,
            p.lld_rge,
            p.lld_era
        FROM read_parquet(?) pr
        LEFT JOIN read_parquet(?) p
            ON pr.parameter = p.parameter
        WHERE pr.species IN ({elements_sql})
        """,
        [rank_path, parameter_path],
    )
    con.execute(f"COPY selected_parameters TO '{str(out_params).replace(chr(92), '/')}' (FORMAT PARQUET)")

    value_expr = (
        "CASE WHEN TRY_CAST(c.qualified_value AS DOUBLE) >= 0 "
        "THEN TRY_CAST(c.qualified_value AS DOUBLE) ELSE NULL END"
    )
    pivot_exprs = []
    for element in elements:
        safe = element.replace("'", "''")
        pivot_exprs.append(
            f"MAX(CASE WHEN species = '{safe}' THEN value_numeric END) AS geochem1_{element}_value"
        )
        pivot_exprs.append(
            f"MAX(CASE WHEN species = '{safe}' THEN parameter END) AS geochem1_{element}_parameter"
        )
    pivot_sql = ",\n            ".join(pivot_exprs)

    out_sql_path = str(out_wide).replace("\\", "/")
    con.execute(
        f"""
        COPY (
            WITH chem_selected AS (
                SELECT
                    c.lab_id,
                    sp.species,
                    sp.parameter,
                    sp.bestvalue_rank,
                    {value_expr} AS value_numeric,
                    c.qualifier,
                    ROW_NUMBER() OVER (
                        PARTITION BY c.lab_id, sp.species
                        ORDER BY sp.bestvalue_rank NULLS LAST, sp.parameter
                    ) AS rn
                FROM read_parquet(?) c
                INNER JOIN selected_parameters sp
                    ON c.parameter = sp.parameter
                WHERE c.qualified_value IS NOT NULL
                  AND TRY_CAST(c.qualified_value AS DOUBLE) IS NOT NULL
                  AND TRY_CAST(c.qualified_value AS DOUBLE) >= 0
            ),
            best_by_lab_species AS (
                SELECT lab_id, species, parameter, bestvalue_rank, value_numeric, qualifier
                FROM chem_selected
                WHERE rn = 1 AND value_numeric IS NOT NULL
            ),
            wide AS (
                SELECT
                    lab_id,
                    {pivot_sql}
                FROM best_by_lab_species
                GROUP BY lab_id
            )
            SELECT
                g.lab_id,
                g.group_id,
                g.project_name,
                g.date_collect,
                g.country,
                g.state,
                g.latitude,
                g.longitude,
                g.sample_source,
                g.primary_class,
                g.specific_name,
                g.deposit_type,
                g.chem_anom,
                g.mineralization,
                g.alteration,
                g.prep,
                g.mesh_pore_size,
                w.*
                    EXCLUDE (lab_id)
            FROM wide w
            INNER JOIN read_parquet(?) g
                ON CAST(w.lab_id AS VARCHAR) = CAST(g.lab_id AS VARCHAR)
            WHERE g.latitude BETWEEN -90 AND 90
              AND g.longitude BETWEEN -180 AND 180
        ) TO '{out_sql_path}' (FORMAT PARQUET)
        """,
        [chem_path, geol_path],
    )

    row_count = con.execute("SELECT COUNT(*) FROM read_parquet(?)", [str(out_wide)]).fetchone()[0]
    param_count = con.execute("SELECT COUNT(*) FROM selected_parameters").fetchone()[0]
    con.close()

    summary = {
        "rows": int(row_count),
        "selected_parameter_rows": int(param_count),
        "elements": elements,
        "outputs": {
            "wide_parquet": str(out_wide),
            "selected_parameters_parquet": str(out_params),
        },
    }
    write_json(output_path(config, "logs", "03_prepare_geochem1_usgs_summary.json"), summary)
    print(f"Wrote {row_count} geochem1 records with selected elements")
    print(out_wide)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
