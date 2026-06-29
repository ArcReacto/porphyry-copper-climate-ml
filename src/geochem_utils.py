from __future__ import annotations

import re

import pandas as pd


def numeric_from_qualified(value) -> float | None:
    if pd.isna(value):
        return None
    match = re.search(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", str(value))
    if not match:
        return None
    return float(match.group(0))


def target_element_columns(columns: list[str], elements: list[str]) -> list[str]:
    wanted = []
    for element in elements:
        prefixes = (f"{element}_", f"{element.lower()}_")
        exact = {element, element.lower(), element.upper()}
        for column in columns:
            if column in exact or column.startswith(prefixes):
                wanted.append(column)
    return sorted(set(wanted))
