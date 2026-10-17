"""Reproducible sample data for exploring the interface offline."""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_demo_series(seed: int = 42) -> tuple[pd.DataFrame, list[tuple[pd.Timestamp, pd.Timestamp]]]:
    rng = np.random.default_rng(seed)
    count = 1200
    timestamps = pd.date_range("2026-01-01", periods=count, freq="5min", tz="UTC")
    step = np.arange(count)
    values = 44 + 11 * np.sin(2 * np.pi * step / 288) + rng.normal(0, 2, count)
    windows = []
    for start, end, change in [(260, 279, 29), (690, 710, -25), (1040, 1055, 35)]:
        values[start : end + 1] += change
        windows.append((timestamps[start], timestamps[end]))
    frame = pd.DataFrame({"timestamp": timestamps, "value": values})
    return frame, windows
