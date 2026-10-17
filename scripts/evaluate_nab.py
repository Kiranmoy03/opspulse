from __future__ import annotations

import argparse

import pandas as pd

from opspulse.alerts import confirm_alerts, group_incidents
from opspulse.data import NAB_SERIES, load_nab
from opspulse.metrics import summarize
from opspulse.model import detect, rolling_zscore


def evaluate_series(series: str, alert_rate: float, train_fraction: float, required: int, max_gap: int) -> list[dict]:
    frame, windows = load_nab(series)
    result = detect(frame, alert_rate=alert_rate, train_fraction=train_fraction)
    points = result.points.copy()
    points["confirmed_alert"] = confirm_alerts(points["is_anomaly"], required=required)
    points["baseline_alert"] = rolling_zscore(points)
    evaluation = points.iloc[result.train_rows:].copy().reset_index(drop=True)
    grouped = group_incidents(evaluation, max_gap=max_gap)
    rows = []
    for method, column in [
        ("Isolation Forest (raw)", "is_anomaly"),
        ("Confirmed alerts", "confirmed_alert"),
        ("Rolling z-score", "baseline_alert"),
    ]:
        report = summarize(evaluation["timestamp"], evaluation[column], windows)
        rows.append({"series": series, "method": method, "train_rows": result.train_rows, "eval_rows": len(evaluation), "incidents": len(grouped) if method == "Confirmed alerts" else None, **report})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare fixed detection settings across NAB CloudWatch series.")
    parser.add_argument("--series", choices=list(NAB_SERIES.values()), action="append", help="Evaluate one or more series (default: all four).")
    parser.add_argument("--alert-rate", type=float, default=0.04)
    parser.add_argument("--train-fraction", type=float, default=0.6)
    parser.add_argument("--confirm", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--merge-gap", type=int, default=6)
    parser.add_argument("--output", help="Optional CSV output path")
    args = parser.parse_args()

    reports = []
    for series in args.series or list(NAB_SERIES.values()):
        print(f"Evaluating {series}…", flush=True)
        reports.extend(evaluate_series(series, args.alert_rate, args.train_fraction, args.confirm, args.merge_gap))
    table = pd.DataFrame(reports)
    print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("Point-level metrics only; this is not the official NAB score. Labels were not used to choose settings.")
    if args.output:
        table.to_csv(args.output, index=False)
        print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
