from __future__ import annotations

from pathlib import Path

import pandas as pd

from ecommerce_forecasting_recommender.utils.io import write_json


def generate_simple_drift_report(reference_path: Path, current_path: Path, output_path: Path) -> Path:
    """Fallback drift report with lightweight stats for local runs."""
    reference = pd.read_csv(reference_path)
    current = pd.read_csv(current_path)

    common_numeric = [
        col
        for col in reference.columns
        if col in current.columns and pd.api.types.is_numeric_dtype(reference[col])
    ]
    report: dict[str, dict[str, float]] = {}
    for col in common_numeric:
        ref_mean = float(reference[col].mean())
        cur_mean = float(current[col].mean())
        report[col] = {
            "reference_mean": ref_mean,
            "current_mean": cur_mean,
            "relative_shift": 0.0 if ref_mean == 0 else (cur_mean - ref_mean) / abs(ref_mean),
        }

    write_json(output_path, {"columns": report})
    return output_path
