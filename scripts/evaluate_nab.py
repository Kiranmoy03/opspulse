"""Evaluate an Isolation Forest against a labeled NAB metric series."""

from __future__ import annotations

import argparse

from opspulse.data import NAB_SERIES, load_nab
from opspulse.metrics import summarize
from opspulse.model import detect, rolling_zscore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series", choices=NAB_SERIES.values(), default=next(iter(NAB_SERIES.values())))
    parser.add_argument("--contamination", type=float, default=0.04)
    arguments = parser.parse_args()
    frame, windows = load_nab(arguments.series)
    result = detect(frame, contamination=arguments.contamination)
    evaluation = result.points.iloc[result.train_rows :]
    output = {
        "isolation_forest": summarize(evaluation["timestamp"], evaluation["is_anomaly"], windows),
        "rolling_zscore": summarize(evaluation["timestamp"], rolling_zscore(result.points).iloc[result.train_rows :], windows),
    }
    print(f"Series: {arguments.series}")
    print(f"Training: {result.train_rows} rows; evaluation: {len(evaluation)} rows")
    for model, metrics in output.items():
        print(f"{model}: {metrics}")
    print("Point-level evaluation; this is not the official NAB score.")


if __name__ == "__main__":
    main()
