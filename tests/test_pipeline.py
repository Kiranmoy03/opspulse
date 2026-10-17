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


def test_training_threshold_is_calibrated_without_test_data():
    series, _ = make_demo_series()
    model = detect(series, alert_rate=0.01)
    training = model.points.iloc[:model.train_rows]
    assert 0 < training["is_anomaly"].sum() <= round(len(training) * 0.02)
    assert model.threshold > 0
    other = series.copy()
    other.loc[1000:, "value"] += 1000
    shifted = detect(other, alert_rate=0.01)
    assert model.threshold == shifted.threshold
    pd.testing.assert_series_equal(
        model.points.iloc[:900]["score"], shifted.points.iloc[:900]["score"]
    )


def test_stricter_training_alert_rate_reduces_raw_alerts():
    series, _ = make_demo_series()
    strict = detect(series, alert_rate=0.005)
    liberal = detect(series, alert_rate=0.05)
    assert strict.threshold > liberal.threshold
    assert strict.points["is_anomaly"].sum() <= liberal.points["is_anomaly"].sum()


def test_confirmation_only_uses_current_and_past_readings():
    from opspulse.alerts import confirm_alerts
    flags = pd.Series([True, False, True, True, False, True, False])
    confirmed = confirm_alerts(flags, required=2, window=3)
    assert confirmed.tolist() == [False, False, True, True, False, True, False]
    assert confirm_alerts(flags, required=1).equals(flags)
    with pytest.raises(ValueError):
        confirm_alerts(flags, required=4, window=3)


def test_incidents_merge_short_gaps_but_not_unrelated_alerts():
    from opspulse.alerts import group_incidents
    flags = [False, True, False, True, False, False, False, True, False, False, False, False, True]
    times = pd.date_range("2026-01-01", periods=len(flags), freq="min", tz="UTC")
    points = pd.DataFrame({"timestamp": times, "score": range(len(flags)), "confirmed_alert": flags})
    groups = group_incidents(points, max_gap=2)
    assert len(groups) == 3
    assert groups.iloc[0]["alert_points"] == 2
    assert groups.iloc[0]["peak_score"] == 3
    assert groups.iloc[1]["start"] == times[7]
    assert group_incidents(points.assign(confirmed_alert=False)).empty


def test_evaluation_metrics_ignore_series_index():
    times = pd.Series(pd.date_range("2026-01-01", periods=10, freq="h", tz="UTC"), index=range(50, 60))
    flags = pd.Series([False] * 7 + [True] + [False] * 2, index=range(50, 60))
    report = summarize(times, flags, [(times.iloc[7], times.iloc[7])])
    assert report["events_detected"] == 1
    assert report["alert_rate"] == pytest.approx(0.1)


def test_batch_evaluation_produces_three_comparable_rows(monkeypatch):
    from scripts import evaluate_nab
    series, windows = make_demo_series()
    monkeypatch.setattr(evaluate_nab, "load_nab", lambda _: (series.copy(), windows))
    report = evaluate_nab.evaluate_series(
        "realAWSCloudwatch/ec2_cpu_utilization_5f5533.csv",
        alert_rate=0.04,
        train_fraction=0.6,
        required=2,
        max_gap=6,
    )
    assert [row["method"] for row in report] == [
        "Isolation Forest (raw)", "Confirmed alerts", "Rolling z-score"
    ]
    assert all(row["eval_rows"] > 0 for row in report)
    assert report[1]["alerts"] <= report[0]["alerts"]
    assert report[1]["incidents"] is not None
