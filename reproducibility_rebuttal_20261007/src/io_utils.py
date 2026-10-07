from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(config_path) if config_path else project_root() / "config" / "paths.yaml"
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def flatten_paths(obj: Any, prefix: str = "") -> list[tuple[str, Path]]:
    paths: list[tuple[str, Path]] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            paths.extend(flatten_paths(value, child_prefix))
    elif isinstance(obj, list):
        return paths
    elif isinstance(obj, str):
        candidate = Path(obj)
        if candidate.drive or "\\" in obj or "/" in obj:
            paths.append((prefix, candidate))
    return paths


def file_info(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    info: dict[str, Any] = {
        "path": str(p),
        "exists": p.exists(),
        "is_file": p.is_file(),
        "is_dir": p.is_dir(),
    }
    if p.exists() and p.is_file():
        stat = p.stat()
        info["size_mb"] = round(stat.st_size / 1024 / 1024, 3)
        info["modified_time"] = stat.st_mtime
    return info


def sniff_csv(path: str | Path, sample_bytes: int = 8192) -> dict[str, Any]:
    p = Path(path)
    encodings = ["utf-8-sig", "utf-8", "gb18030", "latin1"]
    last_error = ""
    for encoding in encodings:
        try:
            with p.open("r", encoding=encoding, newline="", errors="strict") as f:
                sample = f.read(sample_bytes)
                f.seek(0)
                dialect = csv.Sniffer().sniff(sample) if sample else csv.excel
                reader = csv.reader(f, dialect)
                first_rows = []
                for _, row in zip(range(6), reader):
                    first_rows.append(row)
            return {
                "encoding": encoding,
                "delimiter": dialect.delimiter,
                "first_rows": first_rows,
                "first_row_column_count": len(first_rows[0]) if first_rows else 0,
            }
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
    return {"error": last_error}


def read_csv_rows(path: str | Path) -> tuple[list[list[str]], dict[str, Any]]:
    p = Path(path)
    sniffed = sniff_csv(p)
    if "encoding" not in sniffed:
        raise ValueError(f"Unable to read CSV file: {p}; {sniffed}")

    with p.open("r", encoding=sniffed["encoding"], newline="", errors="replace") as f:
        reader = csv.reader(f, delimiter=sniffed["delimiter"])
        return [row for row in reader if row], sniffed


def count_csv_rows(path: str | Path, encoding: str = "utf-8-sig", delimiter: str = ",") -> int:
    p = Path(path)
    count = 0
    with p.open("r", encoding=encoding, newline="", errors="replace") as f:
        reader = csv.reader(f, delimiter=delimiter)
        for _ in reader:
            count += 1
    return count


def write_json(path: str | Path, data: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def ensure_project_dirs(config: dict[str, Any]) -> None:
    for key in ("data_intermediate", "outputs", "logs"):
        Path(config["project"][key]).mkdir(parents=True, exist_ok=True)


def output_path(config: dict[str, Any], area: str, name: str) -> Path:
    base = Path(config["project"][area])
    base.mkdir(parents=True, exist_ok=True)
    return base / name


def write_dataframe(df: pd.DataFrame, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() == ".parquet":
        df.to_parquet(p, index=False)
    elif p.suffix.lower() == ".csv":
        df.to_csv(p, index=False, encoding="utf-8-sig")
    else:
        raise ValueError(f"Unsupported dataframe output extension: {p}")


def read_table(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    raise ValueError(f"Unsupported table extension: {p}")
