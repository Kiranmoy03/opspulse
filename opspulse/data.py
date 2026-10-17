"""Load and prepare chronological metric series."""

from __future__ import annotations

import io
from collections.abc import Sequence

import pandas as pd
import requests

NAB_BASE = "https://raw.githubusercontent.com/numenta/NAB/master"
NAB_SERIES = {
    "EC2 CPU — 5f5533": "realAWSCloudwatch/ec2_cpu_utilization_5f5533.csv",
    "EC2 CPU — 24ae8d": "realAWSCloudwatch/ec2_cpu_utilization_24ae8d.csv",
    "EC2 network in": "realAWSCloudwatch/ec2_network_in_5abac7.csv",
    "RDS CPU": "realAWSCloudwatch/rds_cpu_utilization_cc0c53.csv",
}


def prepare_series(frame: pd.DataFrame, timestamp_col: str = "timestamp", value_col: str = "value") -> pd.DataFrame:
    if timestamp_col not in frame or value_col not in frame:
        raise ValueError(f"CSV must contain '{timestamp_col}' and '{value_col}' columns.")
    result = frame[[timestamp_col, value_col]].copy()
    result.columns = ["timestamp", "value"]
    result["timestamp"] = pd.to_datetime(result["timestamp"], errors="coerce", utc=True)
    result["value"] = pd.to_numeric(result["value"], errors="coerce")
    result = result.dropna(subset=["timestamp", "value"])
    result = result.sort_values("timestamp", kind="stable").drop_duplicates(subset="timestamp", keep="last")
    result = result.reset_index(drop=True)
    if len(result) < 100:
        raise ValueError("At least 100 valid timestamp/value rows are required.")
    return result


def read_csv(file: bytes, timestamp_col: str = "timestamp", value_col: str = "value") -> pd.DataFrame:
    return prepare_series(pd.read_csv(io.BytesIO(file)), timestamp_col, value_col)


def parse_windows(payload: dict, series_path: str) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    windows = payload.get(series_path, [])
    return [(pd.to_datetime(start, utc=True), pd.to_datetime(end, utc=True)) for start, end in windows]


def label_anomalies(timestamps: pd.Series, windows: Sequence[tuple[pd.Timestamp, pd.Timestamp]]) -> pd.Series:
    labels = pd.Series(False, index=timestamps.index)
    for start, end in windows:
        labels |= timestamps.between(start, end, inclusive="both")
    return labels


def load_nab(series_path: str, timeout: int = 20) -> tuple[pd.DataFrame, list[tuple[pd.Timestamp, pd.Timestamp]]]:
    if series_path not in NAB_SERIES.values():
        raise ValueError("Unrecognized NAB series")
    response = requests.get(f"{NAB_BASE}/data/{series_path}", timeout=timeout)
    response.raise_for_status()
    frame = read_csv(response.content)
    response = requests.get(f"{NAB_BASE}/labels/combined_windows.json", timeout=timeout)
    response.raise_for_status()
    windows = parse_windows(response.json(), series_path)
    return frame, windows
