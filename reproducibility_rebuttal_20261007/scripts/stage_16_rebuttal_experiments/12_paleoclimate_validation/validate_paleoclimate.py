"""Audit NASPA/WNATA coverage and align preindustrial summaries to paper samples."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SAMPLES = (
    PROJECT_ROOT
    / "data/run_inputs/known_mining_neutral_ratio_1_10_supervised_all_features_v1"
    / "model_dataset_known_mining_neutral_ratio_1_10_supervised_all_features_v1.csv"
)
REQUIRED_FILES = (
    "NASPA_COOL_TOTALPRECIP.txt",
    "NASPA_GRID_POINTS.txt",
    "NASPA_COOL_STATS.txt",
    "naspa2020readme.txt",
    "king2024-WNATA.txt",
    "king2024-WNATA-gridcells.txt",
)
PERIODS = (
    (1553, 1600),
    (1601, 1700),
    (1701, 1800),
    (1801, 1850),
    (1851, 1900),
    (1901, 1950),
    (1951, 1980),
    (1981, 2016),
)
PILOT_START = 1700
PILOT_END = 1850


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean_number(value: object, digits: int = 6) -> float | None:
    if value is None or not np.isfinite(value):
        return None
    return round(float(value), digits)


def read_wnata(paleo_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    grid = pd.read_csv(
        paleo_dir / "king2024-WNATA-gridcells.txt", sep="\t", comment="#"
    )
    grid = grid.loc[:, ~grid.columns.str.startswith("Unnamed:")]
    values = pd.read_csv(
        paleo_dir / "king2024-WNATA.txt", sep="\t", comment="#", index_col=0
    )
    values = values.loc[:, ~values.columns.str.startswith("Unnamed:")]
    values.index = values.index.astype(int)
    assert grid["Gridpoint"].is_unique
    assert list(values.columns) == grid["Gridpoint"].astype(str).tolist()
    assert values.index.is_unique
    assert values.index.tolist() == list(range(1553, 2021))
    return grid, values


def read_naspa(paleo_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    grid = pd.read_csv(
        paleo_dir / "NASPA_GRID_POINTS.txt",
        sep=r"\s+",
        header=None,
        names=["LON", "LAT"],
    )
    stats = pd.read_csv(paleo_dir / "NASPA_COOL_STATS.txt", sep=r"\s+")
    assert len(grid) == len(stats)
    assert np.allclose(grid["LON"], stats["lon"])
    assert np.allclose(grid["LAT"], stats["lat"])
    values = pd.read_csv(
        paleo_dir / "NASPA_COOL_TOTALPRECIP.txt", sep=r"\s+", index_col=0
    )
    values.index = values.index.astype(int)
    values = values.replace(-99.999, np.nan)
    assert len(values.columns) == len(grid)
    assert values.index.is_unique
    assert values.index.tolist() == list(range(0, 2017))
    assert np.nanmin(values.to_numpy()) >= 0
    return grid, stats, values


def match_samples(samples: pd.DataFrame, grid: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    sample_coords = samples[["latitude", "longitude"]].to_numpy(dtype=float)
    grid_coords = grid[["LAT", "LON"]].to_numpy(dtype=float)
    nearest = cKDTree(grid_coords).query(sample_coords)[1]
    # Both products use 0.5-degree cells; nearest-center distance alone can cross gaps.
    inside_cell = np.all(np.abs(sample_coords - grid_coords[nearest]) <= 0.250001, axis=1)
    return nearest, inside_cell


def quality_summary(stats: pd.DataFrame, nearest: np.ndarray, matched: np.ndarray) -> dict:
    subset = stats.iloc[nearest[matched]]
    fields = {}
    for field in ("CRSQ", "CVRE", "VRE", "VCE"):
        key = field if field in subset else field.lower()
        values = pd.to_numeric(subset[key], errors="coerce")
        fields[field] = {
            "available_samples": int(values.notna().sum()),
            "missing_samples": int(values.isna().sum()),
            "positive_samples": int((values > 0).sum()),
            "nonpositive_samples": int((values <= 0).sum()),
            "median_available": clean_number(values.median()),
        }
    return fields


def period_summary(values: pd.DataFrame, nearest: np.ndarray, matched: np.ndarray) -> list[dict]:
    rows = []
    for start, end in PERIODS:
        segment = values.loc[start:end].to_numpy(dtype=float)
        selected = segment[:, nearest[matched]]
        fractions = np.isfinite(selected).mean(axis=0)
        rows.append(
            {
                "start_year": start,
                "end_year": end,
                "n_years": end - start + 1,
                "all_grid_valid_fraction": clean_number(np.isfinite(segment).mean()),
                "matched_sample_valid_fraction_min": clean_number(fractions.min()),
                "matched_sample_valid_fraction_median": clean_number(np.median(fractions)),
                "matched_samples_with_all_years": int(np.count_nonzero(fractions == 1)),
                "matched_samples_with_at_least_80pct_years": int(
                    np.count_nonzero(fractions >= 0.8)
                ),
            }
        )
    return rows


def point_quality(stats: pd.DataFrame, index: int, field: str) -> float | None:
    key = field if field in stats else field.lower()
    return clean_number(stats.iloc[index][key])


def write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paleo-dir", type=Path, required=True)
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/rebuttal_experiments/12_paleoclimate_alignment",
    )
    args = parser.parse_args()

    for name in REQUIRED_FILES:
        if not (args.paleo_dir / name).is_file():
            parser.error(f"Missing input: {args.paleo_dir / name}")
    if not args.samples.is_file():
        parser.error(f"Missing samples: {args.samples}")

    samples = pd.read_csv(
        args.samples,
        usecols=["sample_id", "Y_label", "state", "latitude", "longitude"],
    )
    assert samples[["latitude", "longitude"]].notna().all().all()
    samples["global_sample_id"] = samples["state"].astype(str) + "|" + samples["sample_id"].astype(str)
    assert samples["global_sample_id"].is_unique

    wgrid, wvalues = read_wnata(args.paleo_dir)
    ngrid, nstats, nvalues = read_naspa(args.paleo_dir)
    wi, wok = match_samples(samples, wgrid)
    ni, nok = match_samples(samples, ngrid)
    both = wok & nok

    wsegment = wvalues.loc[PILOT_START:PILOT_END].to_numpy(dtype=float)
    nsegment = nvalues.loc[PILOT_START:PILOT_END].to_numpy(dtype=float)
    wmean, wstd = np.nanmean(wsegment, axis=0), np.nanstd(wsegment, axis=0)
    nmean, nstd = np.nanmean(nsegment, axis=0), np.nanstd(nsegment, axis=0)
    wcoverage = np.isfinite(wsegment).mean(axis=0)
    ncoverage = np.isfinite(nsegment).mean(axis=0)

    state_counts = []
    for state, group in samples.groupby("state", sort=True):
        indices = group.index.to_numpy()
        state_counts.append(
            {
                "state": state,
                "samples": len(group),
                "positive_samples": int(group["Y_label"].sum()),
                "wnata_matched": int(wok[indices].sum()),
                "naspa_matched": int(nok[indices].sum()),
                "both_matched": int(both[indices].sum()),
            }
        )

    inputs = {name: {"bytes": (args.paleo_dir / name).stat().st_size, "sha256": file_sha256(args.paleo_dir / name)} for name in REQUIRED_FILES}
    inputs["samples"] = {"bytes": args.samples.stat().st_size, "sha256": file_sha256(args.samples)}
    summary = {
        "status": "data_feasibility_only_no_model_validation",
        "sources": {
            "WNATA": "Jun-Aug maximum temperature anomaly, degC relative to 1951-1980",
            "NASPA": "Dec-Apr total precipitation reconstruction, mm",
        },
        "pilot_window": {"start_year": PILOT_START, "end_year": PILOT_END, "inclusive": True},
        "sample_count": len(samples),
        "positive_count": int(samples["Y_label"].sum()),
        "wnata": {
            "grid_count": len(wgrid),
            "year_min": int(wvalues.index.min()),
            "year_max": int(wvalues.index.max()),
            "matched_samples": int(wok.sum()),
            "matched_positive": int(samples.loc[wok, "Y_label"].sum()),
            "quality": quality_summary(wgrid, wi, wok),
            "periods": period_summary(wvalues, wi, wok),
            "pilot_all_years_samples": int(np.count_nonzero(wcoverage[wi[wok]] == 1)),
        },
        "naspa": {
            "grid_count": len(ngrid),
            "year_min": int(nvalues.index.min()),
            "year_max": int(nvalues.index.max()),
            "matched_samples": int(nok.sum()),
            "matched_positive": int(samples.loc[nok, "Y_label"].sum()),
            "quality": quality_summary(nstats, ni, nok),
            "periods": period_summary(nvalues, ni, nok),
            "pilot_all_years_samples": int(np.count_nonzero(ncoverage[ni[nok]] == 1)),
        },
        "both_matched_samples": int(both.sum()),
        "both_matched_positive": int(samples.loc[both, "Y_label"].sum()),
        "by_state": state_counts,
        "input_files": inputs,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "summary.json", summary)
    with (args.out_dir / "sample_alignment.jsonl").open("w", encoding="utf-8") as stream:
        for row_index, row in samples.iterrows():
            record = {
                "global_sample_id": row.global_sample_id,
                "sample_id": row.sample_id,
                "state": row.state,
                "label": int(row.Y_label),
                "latitude": clean_number(row.latitude),
                "longitude": clean_number(row.longitude),
                "wnata_gridpoint": int(wgrid.iloc[wi[row_index]].Gridpoint) if wok[row_index] else None,
                "wnata_jja_tmax_anomaly_mean_degC_1700_1850": clean_number(wmean[wi[row_index]]) if wok[row_index] else None,
                "wnata_jja_tmax_anomaly_std_degC_1700_1850": clean_number(wstd[wi[row_index]]) if wok[row_index] else None,
                "wnata_valid_fraction_1700_1850": clean_number(wcoverage[wi[row_index]]) if wok[row_index] else None,
                "wnata_CVRE": point_quality(wgrid, wi[row_index], "CVRE") if wok[row_index] else None,
                "wnata_VCE": point_quality(wgrid, wi[row_index], "VCE") if wok[row_index] else None,
                "naspa_gridpoint": int(ni[row_index] + 1) if nok[row_index] else None,
                "naspa_cool_precip_mean_mm_1700_1850": clean_number(nmean[ni[row_index]]) if nok[row_index] else None,
                "naspa_cool_precip_std_mm_1700_1850": clean_number(nstd[ni[row_index]]) if nok[row_index] else None,
                "naspa_valid_fraction_1700_1850": clean_number(ncoverage[ni[row_index]]) if nok[row_index] else None,
                "naspa_CVRE": point_quality(nstats, ni[row_index], "CVRE") if nok[row_index] else None,
                "naspa_VCE": point_quality(nstats, ni[row_index], "VCE") if nok[row_index] else None,
            }
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")

    missing = samples.loc[~both, ["global_sample_id", "state", "latitude", "longitude", "Y_label"]]
    missing.to_json(args.out_dir / "unmatched_samples.json", orient="records", force_ascii=False, indent=2)
    state_table = "\n".join(
        f"| {item['state']} | {item['samples']} | {item['positive_samples']} | "
        f"{item['wnata_matched']} | {item['naspa_matched']} | {item['both_matched']} |"
        for item in state_counts
    )
    period_table = "\n".join(
        f"| {wp['start_year']}–{wp['end_year']} | "
        f"{wp['matched_samples_with_all_years']}/{int(wok.sum())} | "
        f"{np_period['matched_samples_with_all_years']}/{int(nok.sum())} |"
        for wp, np_period in zip(summary['wnata']['periods'], summary['naspa']['periods'])
    )
    wvce = summary["wnata"]["quality"]["VCE"]
    nvce = summary["naspa"]["quality"]["VCE"]
    wcvre = summary["wnata"]["quality"]["CVRE"]
    ncvre = summary["naspa"]["quality"]["CVRE"]
    report = f"""# 古气候数据可用性检查

## 检查范围

- 论文主实验样本：{len(samples)} 个，其中正样本 {int(samples['Y_label'].sum())} 个。
- WNATA：{len(wgrid)} 格点，1553–2020 年；数值为相对 1951–1980 年的 6–8 月最高气温距平（°C），不是绝对气温。
- NASPA：{len(ngrid)} 格点，0–2016 年；数值为 12–4 月总降水重建（mm），`-99.999` 已按缺失值处理。
- 空间判定：将样本匹配到最近格点中心，且经纬度分别不得偏离超过 0.25°。未覆盖样本保留为空值，不外推填补。
- 初步汇总窗口：{PILOT_START}–{PILOT_END} 年（含首尾）；仅用于可用性与特征构建检查，并非已确定的论文主分析窗口。

## 结果

| 项目 | 样本数 | 正样本数 |
|---|---:|---:|
| NASPA 覆盖 | {int(nok.sum())} | {int(samples.loc[nok, 'Y_label'].sum())} |
| WNATA 覆盖 | {int(wok.sum())} | {int(samples.loc[wok, 'Y_label'].sum())} |
| 两者共同覆盖 | {int(both.sum())} | {int(samples.loc[both, 'Y_label'].sum())} |
| 两者共同覆盖之外 | {int((~both).sum())} | {int(samples.loc[~both, 'Y_label'].sum())} |

{int((~both).sum())} 个未共同覆盖样本均列于 `results/unmatched_samples.json`。逐州覆盖、验证统计量的可用/缺失数量、各年代窗口的有效年份比例见 `results/summary.json`；逐点格点映射和试算特征见 `results/sample_alignment.jsonl`。

### 逐州覆盖

| 州 | 样本 | 正样本 | WNATA | NASPA | 共同覆盖 |
|---|---:|---:|---:|---:|---:|
{state_table}

### 时间与质量

1700–1850 年窗口内，WNATA {summary['wnata']['pilot_all_years_samples']}/{int(wok.sum())} 个已匹配样本、NASPA {summary['naspa']['pilot_all_years_samples']}/{int(nok.sum())} 个已匹配样本具有完整的逐年值。

| 年代窗口 | WNATA 全年有效样本 | NASPA 全年有效样本 |
|---|---:|---:|
{period_table}

- WNATA：CVRE 可用 {wcvre['available_samples']}/{int(wok.sum())}，其中大于 0 的为 {wcvre['positive_samples']}；VCE 可用 {wvce['available_samples']}，缺失 {wvce['missing_samples']}，可用值中大于 0 的为 {wvce['positive_samples']}。
- NASPA：CVRE 可用 {ncvre['available_samples']}/{int(nok.sum())}，其中大于 0 的为 {ncvre['positive_samples']}；VCE 可用 {nvce['available_samples']}，缺失 {nvce['missing_samples']}，可用值中大于 0 的为 {nvce['positive_samples']}。
- 这里按样本计数；同一格点可能对应多个样本，不能把这些数字解释为独立格点数。

## 使用边界

1. 两套数据来自不同重建产品，格点编号不可直接互连；必须分别用坐标匹配。
2. WNATA 是夏季最高气温距平，NASPA 是冷季总降水；不可称为同季节或年平均气候的配对变量。
3. CVRE、VRE、VCE 是重建验证统计量。缺失的 VCE 不是 0，也不代表该样本没有气候数据；具体阈值需在模型实验前预先规定。
4. 本检查只说明覆盖和数据质量，不证明古气候特征改善模型，也不能证明古气候是成矿时期的真实气候。
5. 下一步若进行模型敏感性实验，应在固定的 11 州 GroupKFold 中，对同一共同覆盖样本比较现代特征基线、古气候补充与替换配置；缺口处理、特征选择和标准化必须折内拟合。
"""
    (args.out_dir / "report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
