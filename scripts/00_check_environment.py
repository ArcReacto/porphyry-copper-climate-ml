from __future__ import annotations

import csv
import sys
from pathlib import Path

import duckdb
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.io_utils import count_csv_rows, file_info, flatten_paths, load_config, read_csv_rows, sniff_csv, write_json


def parquet_summary(path: str | Path) -> dict:
    p = Path(path)
    summary = file_info(p)
    if not p.exists():
        return summary
    con = duckdb.connect(database=":memory:")
    try:
        row_count = con.execute("SELECT COUNT(*) FROM read_parquet(?)", [str(p)]).fetchone()[0]
        columns = con.execute("DESCRIBE SELECT * FROM read_parquet(?)", [str(p)]).fetchdf()
        summary["row_count"] = int(row_count)
        summary["columns"] = columns["column_name"].tolist()
    finally:
        con.close()
    return summary


def csv_summary(path: str | Path, has_header: bool = True) -> dict:
    p = Path(path)
    summary = file_info(p)
    if not p.exists():
        return summary
    sniffed = sniff_csv(p)
    summary["csv_sniff"] = sniffed
    if "encoding" not in sniffed:
        return summary
    delimiter = sniffed["delimiter"]
    encoding = sniffed["encoding"]
    summary["row_count_including_header"] = count_csv_rows(p, encoding=encoding, delimiter=delimiter)
    with p.open("r", encoding=encoding, newline="", errors="replace") as f:
        reader = csv.reader(f, delimiter=delimiter)
        first = next(reader, [])
        if has_header:
            summary["columns"] = first
            summary["column_count"] = len(first)
        else:
            summary["first_record_column_count"] = len(first)
    return summary


def xyz_summary(path: str | Path, max_lines: int = 200_000) -> dict:
    p = Path(path)
    summary = file_info(p)
    if not p.exists():
        return summary
    sample_lines: list[str] = []
    numeric_rows = 0
    non_numeric_rows = 0
    minmax: list[list[float | None]] = []
    with p.open("r", encoding="latin1", errors="replace") as f:
        for line_number, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            if len(sample_lines) < 6:
                sample_lines.append(text[:240])
            parts = text.replace(",", " ").split()
            vals = []
            try:
                vals = [float(x) for x in parts[:8]]
            except ValueError:
                non_numeric_rows += 1
            if vals:
                numeric_rows += 1
                while len(minmax) < len(vals):
                    minmax.append([None, None])
                for idx, value in enumerate(vals):
                    current_min, current_max = minmax[idx]
                    minmax[idx][0] = value if current_min is None else min(current_min, value)
                    minmax[idx][1] = value if current_max is None else max(current_max, value)
            if line_number >= max_lines:
                break
    summary["sample_lines"] = sample_lines
    summary["sampled_numeric_rows"] = numeric_rows
    summary["sampled_non_numeric_rows"] = non_numeric_rows
    summary["first_8_numeric_columns_minmax_sample"] = minmax
    return summary


def validate_mine_header(config: dict) -> dict:
    mrds_path = Path(config["mines"]["mrds_full"])
    mine_path = Path(config["mines"]["porphyry_subset_no_header"])
    result = {
        "mrds_path": str(mrds_path),
        "mine_path": str(mine_path),
        "valid": False,
    }
    if not mrds_path.exists() or not mine_path.exists():
        result["error"] = "MRDS or mine subset file is missing."
        return result

    mrds_rows, _ = read_csv_rows(mrds_path)
    header = mrds_rows[0]
    mrds_ids = {row[0] for row in mrds_rows[1:] if row}

    mine_rows, mine_sniffed = read_csv_rows(mine_path)

    mine_has_header = bool(mine_rows and mine_rows[0] == header)
    mine_data_rows = mine_rows[1:] if mine_has_header else mine_rows
    missing_ids = [row[0] for row in mine_data_rows if row[0] not in mrds_ids]
    column_counts = sorted({len(row) for row in mine_data_rows})
    result.update(
        {
            "valid": not missing_ids and column_counts == [len(header)],
            "mrds_column_count": len(header),
            "mine_has_header": mine_has_header,
            "mine_row_count_including_header_if_present": len(mine_rows),
            "mine_data_row_count": len(mine_data_rows),
            "mine_column_counts": column_counts,
            "missing_dep_id_count": len(missing_ids),
            "header": header,
            "mine_csv_sniff": mine_sniffed,
        }
    )
    return result


def main() -> int:
    config = load_config()
    logs_dir = Path(config["project"]["logs"])
    logs_dir.mkdir(parents=True, exist_ok=True)

    report: dict = {
        "project_root": str(PROJECT_ROOT),
        "path_checks": {},
        "mine_header_validation": validate_mine_header(config),
        "datasets": {},
        "python": sys.version,
        "pandas": pd.__version__,
        "duckdb": duckdb.__version__,
    }

    for key, path in flatten_paths(config):
        if key.startswith("targets."):
            continue
        report["path_checks"][key] = file_info(path)

    report["datasets"]["mrds_full"] = csv_summary(config["mines"]["mrds_full"], has_header=True)
    report["datasets"]["mine_subset_no_header"] = csv_summary(
        config["mines"]["porphyry_subset_no_header"], has_header=False
    )
    report["datasets"]["geochem2_main"] = csv_summary(config["geochem2"]["main_csv"], has_header=True)

    for table_name, table_path in config["geochem1"]["tables"].items():
        report["datasets"][f"geochem1_{table_name}"] = parquet_summary(table_path)

    gravity = config["gravity"]
    for stage in ("first_stage", "second_stage"):
        for name, path in gravity[stage].items():
            if str(path).lower().endswith(".xyz"):
                report["datasets"][f"gravity_{stage}_{name}"] = xyz_summary(path)

    json_path = logs_dir / "environment_check.json"
    txt_path = logs_dir / "environment_check.txt"
    write_json(json_path, report)

    missing = [
        f"{key}: {item['path']}"
        for key, item in report["path_checks"].items()
        if not item.get("exists")
    ]
    lines = [
        "Environment check complete.",
        f"Project root: {PROJECT_ROOT}",
        f"Checked paths: {len(report['path_checks'])}",
        f"Missing paths: {len(missing)}",
        f"Mine header validation: {report['mine_header_validation']['valid']}",
        "",
        "Outputs:",
        str(json_path),
        str(txt_path),
    ]
    if missing:
        lines.extend(["", "Missing:"])
        lines.extend(missing)
    txt_path.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
