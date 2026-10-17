import pandas as pd
import pytest

from opspulse.data import label_anomalies, prepare_series
from opspulse.demo import make_demo_series
from opspulse.metrics import summarize
from opspulse.model import detect, make_features, rolling_zscore


def test_prepare_series_sorts_and_deduplicates():
    timestamps = pd.date_range("2026-01-01", periods=104, freq="min", tz="UTC")
    frame = pd.DataFrame({"timestamp": list(timestamps[::-1]) + [timestamps[0]], "value": list(range(104)) + [999]})
    result = prepare_series(frame)
    assert len(result) == 104
    assert result["timestamp"].is_monotonic_increasing
    assert result.iloc[0]["value"] == 999


def test_prepare_series_rejects_small_datasets():
    with pytest.raises(ValueError, match="100"):
        prepare_series(pd.DataFrame({"timestamp": ["2026-01-01"], "value": [1]}))


def test_features_use_only_current_and_previous_values():
    series, _ = make_demo_series()
    features = make_features(series)
    changed = series.copy()
    changed.loc[850:, "value"] += 1000
    other = make_features(changed)
    pd.testing.assert_frame_equal(features.iloc[:800], other.iloc[:800])
    assert features.iloc[0]["timestamp"] == series.iloc[24]["timestamp"]


def test_model_is_reproducible_and_split_is_chronological():
    series, _ = make_demo_series()
    first = detect(series)
    second = detect(series)
    assert first.train_rows == second.train_rows
    assert first.points["score"].equals(second.points["score"])
    assert first.points.iloc[:first.train_rows]["split"].eq("Training").all()
    assert first.points.iloc[first.train_rows:]["split"].eq("Evaluation").all()
    assert first.points["is_anomaly"].any()


def test_baseline_and_labels():
    series, windows = make_demo_series()
    labels = label_anomalies(series["timestamp"], windows)
    assert labels.sum() == 20 + 21 + 16
    assert not labels.iloc[0]
    assert rolling_zscore(series).dtype == bool


def test_summary_counts_alerts_and_events():
    times = pd.Series(pd.date_range("2026-01-01", periods=10, freq="h", tz="UTC"))
    predicted = pd.Series([False, False, True, False, False, False, False, True, False, False])
    windows = [(times.iloc[1], times.iloc[3]), (times.iloc[7], times.iloc[8])]
    report = summarize(times, predicted, windows)
    assert report["alerts"] == 2
    assert report["events_detected"] == 2
    assert report["events_total"] == 2
    assert report["precision"] == 1.0


def test_parse_nab_windows():
    from opspulse.data import parse_windows
    payload = {"realAWSCloudwatch/example.csv": [["2014-04-01 00:00:00", "2014-04-01 01:00:00"]]}
    windows = parse_windows(payload, "realAWSCloudwatch/example.csv")
    assert len(windows) == 1
    assert str(windows[0][0].tzinfo) == "UTC"
