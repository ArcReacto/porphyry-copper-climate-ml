from __future__ import annotations

import json
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = PROJECT_ROOT / "figures"
MANIFEST = Path(__file__).with_name("figure_manifest.yaml")
REPORT = PROJECT_ROOT / "docs" / "paper_figure_status.json"


def read_png_dpi(path: Path) -> tuple[float | None, float | None]:
    try:
        from PIL import Image
    except ImportError:
        return None, None

    with Image.open(path) as img:
        dpi = img.info.get("dpi")
    if not dpi:
        return None, None
    return float(dpi[0]), float(dpi[1])


def main() -> None:
    with MANIFEST.open("r", encoding="utf-8") as f:
        manifest = yaml.safe_load(f)

    rows = []
    for item in manifest["figures"]:
        stem = item["stem"]
        png = FIGURE_DIR / f"{stem}.png"
        pdf = FIGURE_DIR / f"{stem}.pdf"
        dpi_x, dpi_y = read_png_dpi(png) if png.exists() else (None, None)
        rows.append(
            {
                "id": item["id"],
                "stem": stem,
                "role": item["role"],
                "status": item["status"],
                "png_exists": png.exists(),
                "pdf_exists": pdf.exists(),
                "png_size_bytes": png.stat().st_size if png.exists() else None,
                "pdf_size_bytes": pdf.stat().st_size if pdf.exists() else None,
                "png_dpi_x": dpi_x,
                "png_dpi_y": dpi_y,
                "png_dpi_ge_600": bool(dpi_x and dpi_y and dpi_x >= 599 and dpi_y >= 599),
                "claim": item["claim"],
            }
        )

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    missing = [r["stem"] for r in rows if not r["png_exists"] or not r["pdf_exists"]]
    low_dpi = [r["stem"] for r in rows if r["png_exists"] and not r["png_dpi_ge_600"]]

    print(f"Checked figures: {len(rows)}")
    print(f"Missing png/pdf pairs: {len(missing)}")
    print(f"PNG below 600 dpi or dpi unreadable: {len(low_dpi)}")
    print(REPORT)
    if missing:
        print("Missing:", ", ".join(missing))
    if low_dpi:
        print("Low/unknown dpi:", ", ".join(low_dpi))


if __name__ == "__main__":
    main()
