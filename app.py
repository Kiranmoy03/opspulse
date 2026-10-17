"""OpsPulse dashboard."""

from __future__ import annotations

import io

import pandas as pd
import requests
import plotly.graph_objects as go
import streamlit as st

from opspulse.data import NAB_SERIES, label_anomalies, load_nab, prepare_series
from opspulse.demo import make_demo_series
from opspulse.metrics import summarize
from opspulse.model import detect, rolling_zscore

st.set_page_config(page_title="OpsPulse | Anomaly Detection", page_icon="📡", layout="wide")


@st.cache_data(ttl=86400, show_spinner=False)
def cached_nab(path: str):
    return load_nab(path)


st.title("OpsPulse")
st.caption("Infrastructure anomaly detection · Historical time-series analysis")

with st.sidebar:
    st.header("Data source")
    source = st.selectbox("Series", ["Demo (synthetic)", *NAB_SERIES, "Upload CSV"])
    uploaded = None
    if source == "Upload CSV":
        uploaded = st.file_uploader("Upload timestamp/value CSV", type="csv")
    st.divider()
    st.header("Model settings")
    contamination = st.slider("Expected anomaly proportion", 0.01, 0.15, 0.04, step=0.01)
    train_fraction = st.slider("Initial training portion", 0.50, 0.80, 0.60, step=0.05)
    z_threshold = st.slider("Rolling z-score threshold", 2.0, 6.0, 3.0, step=0.5)

try:
    if source == "Demo (synthetic)":
        series, windows = make_demo_series()
        st.info("Demo uses generated data with injected anomalies. Select an AWS CloudWatch series to explore real NAB data.")
    elif source == "Upload CSV":
        if uploaded is None:
            st.info("Upload a CSV with `timestamp` and `value` columns to begin.")
            st.stop()
        series = prepare_series(pd.read_csv(io.BytesIO(uploaded.getvalue())))
        windows = []
    else:
        with st.spinner("Loading NAB series and anomaly windows…"):
            series, windows = cached_nab(NAB_SERIES[source])
except (ValueError, pd.errors.ParserError, OSError, requests.RequestException) as exc:
    st.error(f"Unable to load data: {exc}")
    st.stop()

result = detect(series, contamination=contamination, train_fraction=train_fraction)
points = result.points
points["baseline_alert"] = rolling_zscore(points, z_threshold)
points["reference_anomaly"] = label_anomalies(points["timestamp"], windows).to_numpy()
evaluation = points.loc[points["split"] == "Evaluation"].copy()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total samples", f"{len(series):,}")
col2.metric("Evaluation samples", f"{len(evaluation):,}")
col3.metric("Model alerts (evaluation)", int(evaluation["is_anomaly"].sum()))
col4.metric("Reference windows", len(windows) if windows else "Not available")

st.subheader("Metric timeline")
figure = go.Figure()
figure.add_trace(go.Scatter(x=points["timestamp"], y=points["value"], mode="lines", name="Metric", line=dict(color="#7DD3FC", width=1.5)))
alerts = points.loc[points["is_anomaly"]]
figure.add_trace(go.Scatter(x=alerts["timestamp"], y=alerts["value"], mode="markers", name="Isolation Forest alerts", marker=dict(color="#FB923C", size=7)))
for start, end in windows:
    figure.add_vrect(x0=start, x1=end, fillcolor="#F87171", opacity=0.13, line_width=0)
figure.add_vline(x=points.loc[result.train_rows, "timestamp"], line_color="#94A3B8", line_dash="dash")
figure.update_layout(height=450, margin=dict(t=26, b=10, l=10, r=10), hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.03), xaxis_title="Time (UTC)", yaxis_title="Metric value")
st.plotly_chart(figure, use_container_width=True)
if windows:
    st.caption("Orange markers: Isolation Forest alerts · Red shading: reference anomaly windows · Dashed line: evaluation start")

st.subheader("Chronological evaluation")
st.caption("The first portion is used to train the model; only the later portion is evaluated. Windows come from NAB when available. Point-level labels cover each entire reference window, so these figures are descriptive, not the official NAB benchmark score.")
if windows:
    model_metrics = summarize(evaluation["timestamp"], evaluation["is_anomaly"], windows)
    baseline_metrics = summarize(evaluation["timestamp"], evaluation["baseline_alert"], windows)
    st.dataframe(pd.DataFrame({"Isolation Forest": model_metrics, "Rolling z-score": baseline_metrics}).T, use_container_width=True)
else:
    st.write("No reference labels supplied. Uploads show alert patterns, not precision or recall.")

with st.expander("Review evaluation alerts", expanded=False):
    st.dataframe(evaluation.loc[evaluation["is_anomaly"], ["timestamp", "value", "score"]], use_container_width=True, hide_index=True)

csv = evaluation.loc[evaluation["is_anomaly"], ["timestamp", "value", "score"]].to_csv(index=False)
st.download_button("Download evaluation alerts (CSV)", data=csv, file_name="opspulse_alerts.csv", mime="text/csv")
st.caption("Exploratory analysis only. Not a live monitoring service or operational alerting system. NAB data: Numenta Anomaly Benchmark (GitHub).")
