from __future__ import annotations

import csv
import hashlib
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "DATA_MANIFEST.csv"
CHECKSUMS = ROOT / "CHECKSUMS.sha256"
FORBIDDEN_PATH_PATTERNS = (
    re.compile(r"[A-Za-z]:\\Users\\"),
    re.compile(r"[A-Za-z]:/Users/"),
    re.compile(r"/(?:workspace|tmp)/"),
    re.compile(r"Desktop|Downloads"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def csv_shape(path: Path) -> tuple[int, int, int, int]:
    rows = 0
    positives = 0
    negatives = 0
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        label = "Y_label" if "Y_label" in fields else "label" if "label" in fields else None
        for row in reader:
            rows += 1
            if label and row.get(label) == "1":
                positives += 1
            elif label and row.get(label) == "0":
                negatives += 1
    return rows, len(fields), positives, negatives


def fail(message: str, errors: list[str]) -> None:
    errors.append(message)
    print(f"[FAIL] {message}")


def main() -> int:
    errors: list[str] = []
    print(f"Validating: {ROOT}")

    with MANIFEST.open("r", encoding="utf-8-sig", newline="") as stream:
        records = list(csv.DictReader(stream))
    for record in records:
        path = ROOT / record["path"]
        if not path.is_file():
            fail(f"missing manifest file: {record['path']}", errors)
            continue
        rows, columns, positives, negatives = csv_shape(path)
        expected_rows = int(record["rows"])
        expected_columns = int(record["columns"])
        if (rows, columns) != (expected_rows, expected_columns):
            fail(
                f"shape mismatch for {record['path']}: {(rows, columns)} != "
                f"{(expected_rows, expected_columns)}",
                errors,
            )
        if record["positive_rows"] and positives != int(record["positive_rows"]):
            fail(f"positive-label mismatch for {record['path']}", errors)
        if record["negative_rows"] and negatives != int(record["negative_rows"]):
            fail(f"negative-label mismatch for {record['path']}", errors)

    main_input = ROOT / records[0]["path"]
    rows, _, positives, negatives = csv_shape(main_input)
    if (rows, positives, negatives) != (1738, 158, 1580):
        fail("main benchmark is not 1,738 = 158 positive + 1,580 negative", errors)

    a132_manifest = ROOT / "data/aster_enhanced/A132_feature_manifest.csv"
    if csv_shape(a132_manifest)[0] != 132:
        fail("A132 manifest does not contain 132 feature rows", errors)

    paleo = ROOT / "data/paleoclimate/paleoclimate_aligned_1738.csv"
    complete = 0
    with paleo.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if (
                row.get("wnata_jja_tmax_anomaly_mean_degC_1700_1850", "")
                and row.get("naspa_cool_precip_mean_mm_1700_1850", "")
            ):
                complete += 1
    if complete != 1729:
        fail(f"expected 1,729 complete paleoclimate rows, found {complete}", errors)

    for script in ROOT.rglob("*.py"):
        if script.resolve() == Path(__file__).resolve():
            continue
        text = script.read_text(encoding="utf-8")
        for pattern in FORBIDDEN_PATH_PATTERNS:
            if pattern.search(text):
                fail(f"absolute/local path pattern in {script.relative_to(ROOT)}", errors)
                break

    required_results = (
        "results/reported/main_m1_m4/02_climate_decoupling/climate_decoupling_predictions.csv",
        "results/reported/fold_local_graphs/02_spatial_block_validation/spatial_fold_manifest.csv",
        "results/reported/xgboost_rebuttal/01_fold_local_graphunion/oof_predictions.csv",
        "results/reported/xgboost_rebuttal/08_random_matched_residualization/random_control_oof_predictions.csv",
        "results/reported/paleoclimate_xgboost/model_oof_predictions.jsonl",
        "results/reported/idann_reference/oof_predictions.csv",
    )
    for relative in required_results:
        if not (ROOT / relative).is_file():
            fail(f"missing reported result: {relative}", errors)

    result_extensions = {".json", ".jsonl", ".csv", ".md"}
    for result in (ROOT / "results").rglob("*"):
        if not result.is_file() or result.suffix.lower() not in result_extensions:
            continue
        text = result.read_text(encoding="utf-8-sig", errors="replace")
        for pattern in FORBIDDEN_PATH_PATTERNS:
            if pattern.search(text):
                fail(f"absolute/local path pattern in {result.relative_to(ROOT)}", errors)
                break

    unexpected_outputs = [
        path for path in (ROOT / "outputs").rglob("*") if path.is_file() and path.name != ".gitkeep"
    ]
    if unexpected_outputs:
        fail(f"generated outputs are present: {unexpected_outputs[:3]}", errors)

    oversized = [path for path in ROOT.rglob("*") if path.is_file() and path.stat().st_size >= 100_000_000]
    if oversized:
        fail(f"files at or above GitHub's 100 MB limit: {oversized}", errors)

    if CHECKSUMS.exists():
        for line in CHECKSUMS.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            expected, relative = line.split("  ", 1)
            path = ROOT / relative
            if not path.is_file() or sha256(path) != expected:
                fail(f"checksum mismatch: {relative}", errors)
    else:
        fail("CHECKSUMS.sha256 is missing", errors)

    if errors:
        print(f"Validation failed with {len(errors)} issue(s).")
        return 1
    print(
        f"Validation passed: {len(records)} CSV inputs checked; reported results are present; "
        "no uncommitted runtime outputs included."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
