# OpsPulse

Infrastructure time-series anomaly detection with Python, scikit-learn, and Streamlit.

OpsPulse trains an **Isolation Forest** on an initial chronological segment of a metric series, scores later observations, and visualizes unusual readings. It includes a rolling z-score comparison and optional evaluation against labeled anomaly windows from the [Numenta Anomaly Benchmark (NAB)](https://github.com/numenta/NAB).

## Features

- Analyze historical AWS CloudWatch metric series from NAB or upload a CSV with `timestamp` and `value` columns.
- Engineer rolling statistics and rate-of-change features using current and earlier observations only.
- Train an unsupervised Isolation Forest and evaluate on a later chronological segment.
- Compare alerts with a rolling z-score baseline.
- Explore flagged readings on an interactive timeline and export evaluation alerts as CSV.
- Try the interface without a download using a reproducible **synthetic demo**.

## Tech stack

`Python` · `pandas` · `scikit-learn` · `Streamlit` · `Plotly` · `pytest`

## Run locally

Python 3.11+ is recommended.

```bash
git clone https://github.com/Kiranmoy03/opspulse.git
cd opspulse
python -m venv .venv
```

Activate the virtual environment:

- Windows PowerShell: `.venv\Scripts\Activate.ps1`
- macOS/Linux: `source .venv/bin/activate`

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

The default dataset is synthetic and available offline. Real NAB series are fetched from GitHub when selected, so those options require internet access.

## Evaluate a NAB series

```bash
python -m scripts.evaluate_nab
```

Use `python -m scripts.evaluate_nab --help` to see the available dataset paths and model settings.

The dashboard reports **point-level precision, recall, F1, and anomaly-window hit counts** for the evaluation portion of the series. These are descriptive measures, **not the official NAB benchmark score**. The labels cover broad time windows, so point-level scores can be misleading for early detection. No claim of production accuracy or generalization is made.

## Test

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Tests check data cleaning, chronological splitting, deterministic scoring, feature leakage, and evaluation metrics.

## Structure

```text
app.py                    Streamlit interface
opspulse/
  data.py                 CSV input and NAB loading
  demo.py                 Reproducible synthetic sample
  model.py                Time-series features and Isolation Forest
  metrics.py              Evaluation metrics
scripts/evaluate_nab.py   Command-line NAB evaluation
tests/test_pipeline.py    Automated tests
```

## Data source and limitations

Real metric data and anomaly windows are provided by [Numenta/NAB](https://github.com/numenta/NAB), specifically the `data/realAWSCloudwatch` series and `labels/combined_windows.json`. See the upstream repository for dataset terms and attribution. The example dataset generated in `opspulse/demo.py` is **synthetic** and is not representative of real server performance.

OpsPulse currently analyzes **one historical metric at a time**. Its rolling features use observations available up to each timestamp, but model fitting is offline. It does not connect to infrastructure monitoring systems, send live alerts, or replace an operational monitoring solution. The initial training portion is not guaranteed anomaly-free, and outcomes depend on the chosen time series and contamination setting.
