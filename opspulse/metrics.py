from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from opspulse.data import label_anomalies


def summarize(
    timestamps: pd.Series,
    predictions: pd.Series,
    windows: Sequence[tuple[pd.Timestamp, pd.Timestamp]],
) -> dict[str, float | int]:
    timestamps = timestamps.reset_index(drop=True)
    predicted = predictions.reset_index(drop=True).astype(bool)
    if len(timestamps) != len(predicted):
        raise ValueError("timestamps and predictions must have equal length")
    true_labels = label_anomalies(timestamps, windows)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predicted, average="binary", zero_division=0
    )
    intersecting = [(a, b) for a, b in windows if timestamps.between(a, b).any()]
    detected = sum(
        bool((predicted & timestamps.between(start, end)).any())
        for start, end in intersecting
    )
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "alerts": int(predicted.sum()),
        "alert_rate": float(predicted.mean()) if len(predicted) else 0.0,
        "events_detected": detected,
        "events_total": len(intersecting),
    }
