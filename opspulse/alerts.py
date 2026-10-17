from __future__ import annotations

import pandas as pd


def confirm_alerts(flags: pd.Series, required: int = 2, window: int = 3) -> pd.Series:
    if not 1 <= required <= window:
        raise ValueError("required must be between 1 and window")
    raw = flags.astype(bool)
    if required == 1:
        return raw.copy()
    confirmations = raw.astype(int).rolling(window, min_periods=required).sum()
    return raw & confirmations.ge(required)


def group_incidents(points: pd.DataFrame, max_gap: int = 6) -> pd.DataFrame:
    if max_gap < 0:
        raise ValueError("max_gap must be nonnegative")
    if not {"timestamp", "score", "confirmed_alert"}.issubset(points):
        raise ValueError("Expected timestamp, score and confirmed_alert columns")

    positions = [int(i) for i in range(len(points)) if bool(points.iloc[i]["confirmed_alert"])]
    rows: list[dict] = []
    if not positions:
        return pd.DataFrame(columns=["start", "end", "alert_points", "peak_score"])

    groups: list[list[int]] = [[positions[0]]]
    for position in positions[1:]:
        if position - groups[-1][-1] - 1 <= max_gap:
            groups[-1].append(position)
        else:
            groups.append([position])

    for group in groups:
        alert_points = points.iloc[group]
        rows.append(
            {
                "start": points.iloc[group[0]]["timestamp"],
                "end": points.iloc[group[-1]]["timestamp"],
                "alert_points": len(group),
                "peak_score": float(alert_points["score"].max()),
            }
        )
    return pd.DataFrame(rows)
