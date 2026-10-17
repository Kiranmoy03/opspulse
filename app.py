from __future__ import annotations

import io

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

from opspulse.alerts import confirm_alerts, group_incidents
from opspulse.data import NAB_SERIES, label_anomalies, load_nab, prepare_series
from opspulse.demo import make_demo_series
from opspulse.metrics import summarize
from opspulse.model import detect, rolling_zscore
from scripts.evaluate_nab import evaluate_series

st.set_page_config(page_title="OpsPulse | Anomaly Detection", page_icon="📡", layout="wide")


@st.cache_data(ttl=86400, show_spinner=False)
def cached_nab(path: str):
    return load_nab(path)


@st.cache_data(ttl=3600, show_spinner=False)
def compare_four_series(rate: float, fraction: float, confirmation: int, gap: int):
    reports = []
    for path in NAB_SERIES.values():
        reports.extend(evaluate_series(path, rate, fraction, confirmation, gap))
    return pd.DataFrame(reports)


st.title("OpsPulse")
st.caption("Infrastructure anomaly detection · Historical time-series analysis")

with st.sidebar:
    st.header("Data source")
    source = st.selectbox("Series", ["Demo (synthetic)", *NAB_SERIES, "Upload CSV"])
    uploaded = st.file_uploader("CSV with timestamp and value", type="csv") if source == "Upload CSV" else None
    st.divider()
    st.header("Detection settings")
    alert_rate = st.slider("Training alert rate target (%)", 0.5, 5.0, 4.0, step=0.5) / 100
    train_fraction = st.slider("Initial training portion", 0.50, 0.80, 0.60, step=0.05)
    required = st.selectbox("Alert confirmation", [1, 2, 3], index=1, format_func=lambda n: {1: "Off (all alerts)", 2: "2 of 3 readings", 3: "3 of 3 readings"}[n])
    max_gap = st.slider("Merge nearby alerts (readings)", 0, 12, 6)
    with st.expander("Baseline comparison"):
        z_threshold = st.slider("Rolling z-score threshold", 2.0, 6.0, 3.0, step=0.5)

try:
    if source == "Demo (synthetic)":
        series, windows = make_demo_series()
        st.info("This demo uses synthetic data. Select an AWS CloudWatch series for NAB data.")
    elif source == "Upload CSV":
        if uploaded is None:
            st.info("Upload a CSV with timestamp and value columns to begin.")
            st.stop()
        series = prepare_series(pd.read_csv(io.BytesIO(uploaded.getvalue())))
        windows = []
    else:
        with st.spinner("Loading NAB data…"):
            series, windows = cached_nab(NAB_SERIES[source])

    result = detect(series, alert_rate=alert_rate, train_fraction=train_fraction)
except (ValueError, pd.errors.ParserError, OSError, requests.RequestException) as exc:
    st.error(f"Unable to analyze this series: {exc}")
    st.stop()

points = result.points.copy()
points["confirmed_alert"] = confirm_alerts(points["is_anomaly"], required=required)
points["baseline_alert"] = rolling_zscore(points, z_threshold)
points["reference_anomaly"] = label_anomalies(points["timestamp"], windows).to_numpy()
evaluation = points.iloc[result.train_rows:].copy().reset_index(drop=True)
incidents = group_incidents(evaluation, max_gap=max_gap)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Evaluation readings", f"{len(evaluation):,}")
col2.metric("Raw model alerts", int(evaluation["is_anomaly"].sum()))
col3.metric("Confirmed alerts", int(evaluation["confirmed_alert"].sum()))
col4.metric("Grouped incidents", len(incidents))
st.caption(f"Threshold calibrated on the first {result.train_rows:,} feature rows. Confirmations and incident grouping affect alert presentation, not model training.")

st.subheader("Metric timeline")
figure = go.Figure()
figure.add_trace(go.Scatter(x=points["timestamp"], y=points["value"], mode="lines", name="Metric", line=dict(color="#7DD3FC", width=1.5)))
raw = evaluation.loc[evaluation["is_anomaly"]]
confirmed = evaluation.loc[evaluation["confirmed_alert"]]
figure.add_trace(go.Scatter(x=raw["timestamp"], y=raw["value"], mode="markers", name="Raw alerts", marker=dict(color="#64748B", size=5, opacity=0.5)))
figure.add_trace(go.Scatter(x=confirmed["timestamp"], y=confirmed["value"], mode="markers", name="Confirmed alerts", marker=dict(color="#FB923C", size=7)))
for start, end in windows:
    figure.add_vrect(x0=start, x1=end, fillcolor="#F87171", opacity=0.13, line_width=0)
figure.add_vline(x=points.iloc[result.train_rows]["timestamp"], line_color="#94A3B8", line_dash="dash")
figure.update_layout(height=450, margin=dict(t=26, b=10, l=10, r=10), hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.03), xaxis_title="Time (UTC)", yaxis_title="Metric value")
st.plotly_chart(figure, use_container_width=True)

st.subheader("Chronological evaluation")
st.caption("The first portion trains the model; only later observations are evaluated. NAB windows are broad reference labels. This is not the official NAB benchmark score.")
if windows:
    reports = {
        "Isolation Forest (raw)": summarize(evaluation["timestamp"], evaluation["is_anomaly"], windows),
        "Confirmed alerts": summarize(evaluation["timestamp"], evaluation["confirmed_alert"], windows),
        "Rolling z-score": summarize(evaluation["timestamp"], evaluation["baseline_alert"], windows),
    }
    st.dataframe(pd.DataFrame.from_dict(reports, orient="index"), use_container_width=True)
    st.caption("Lower alert counts are not automatically better: compare precision, recall, and detected events before drawing conclusions.")
else:
    st.write("No reference labels supplied. Precision and recall cannot be calculated for this upload.")

with st.expander("Compare all four AWS datasets", expanded=False):
    st.caption("Uses the same settings for every series. Reference labels are used only for evaluation, not threshold selection.")
    if st.button("Run four-series comparison"):
        try:
            with st.spinner("Loading and evaluating four NAB series…"):
                comparison = compare_four_series(alert_rate, train_fraction, required, max_gap)
            selected = comparison[["series", "method", "precision", "recall", "f1", "alert_rate", "events_detected", "events_total"]]
            st.dataframe(selected, use_container_width=True, hide_index=True)
            st.download_button("Download comparison (CSV)", data=comparison.to_csv(index=False), file_name="opspulse_comparison.csv", mime="text/csv")
        except (ValueError, OSError, requests.RequestException) as exc:
            st.error(f"Comparison could not finish: {exc}")

with st.expander("Review grouped incidents", expanded=False):
    st.dataframe(incidents, use_container_width=True, hide_index=True)
with st.expander("Review evaluation alerts", expanded=False):
    st.dataframe(evaluation.loc[evaluation["confirmed_alert"], ["timestamp", "value", "score"]], use_container_width=True, hide_index=True)

first, second = st.columns(2)
with first:
    alert_csv = evaluation.loc[evaluation["confirmed_alert"], ["timestamp", "value", "score"]].to_csv(index=False)
    st.download_button("Download confirmed alerts (CSV)", data=alert_csv, file_name="opspulse_alerts.csv", mime="text/csv")
with second:
    st.download_button("Download incidents (CSV)", data=incidents.to_csv(index=False), file_name="opspulse_incidents.csv", mime="text/csv")

st.caption("Exploratory offline model fitting, not live infrastructure monitoring. Dataset: Numenta Anomaly Benchmark (NAB).")
