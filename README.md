# OpsPulse

**Historical infrastructure anomaly detection with Python, scikit-learn, and Streamlit.**

**Live dashboard:** https://opspulse-ml.streamlit.app/

OpsPulse analyzes one metric series at a time using Isolation Forest. It trains on the earlier part of the series, flags unusual readings in the later part, and compares the results with a rolling z-score baseline. Optional alert confirmation and incident grouping help explore alert noise without changing the trained model.

## Features

- Load historical AWS CloudWatch metrics from the [Numenta Anomaly Benchmark (NAB)](https://github.com/numenta/NAB), use a synthetic example, or upload a CSV.
- Build rolling and rate-of-change features using current and past readings.
- Calibrate the Isolation Forest alert threshold on the training segment only.
- Adjust the training alert rate, require repeated alerts, and group nearby alerts into incidents.
- Compare raw and confirmed alerts against NAB anomaly windows, including a one-click comparison across four AWS series.
- Export confirmed alerts and incident summaries as CSV.

`Python` · `pandas` · `scikit-learn` · `Streamlit` · `Plotly` · `pytest`

## Run locally

Python 3.11+ is recommended.

```bash
git clone https://github.com/Kiranmoy03/opspulse.git
cd opspulse
python -m pip install -r requirements.txt
streamlit run app.py
```

The default demo is synthetic and works offline. NAB data requires an internet connection. Uploaded CSV files need `timestamp` and `value` columns and enough valid readings for the chronological split.

## Evaluate the model

Run the fixed configuration on all four included AWS series:

```bash
python -m scripts.evaluate_nab --output evaluation.csv
```

To view available settings:

```bash
python -m scripts.evaluate_nab --help
```

The report compares raw Isolation Forest alerts, confirmed alerts, and rolling z-score using precision, recall, F1, alert rate, and detected anomaly windows. The threshold is calibrated without labels. Reducing the number of alerts may also reduce recall. **These point-level measures are not the official NAB score.**

## Tests

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

## Structure

```text
app.py                    Streamlit dashboard
opspulse/
  alerts.py               Alert confirmation and incident grouping
  data.py                 CSV and NAB loading
  demo.py                 Synthetic example
  model.py                Feature engineering and Isolation Forest
  metrics.py              Point-level and window-level evaluation
scripts/evaluate_nab.py   Reproducible four-series evaluation
tests/test_pipeline.py    Automated tests
```

## Limitations

OpsPulse is an exploratory offline model, not a live monitoring system. It processes a single historical metric and does not collect production telemetry or send operational alerts. NAB labels are broad windows; a detected window does not establish that every anomaly point was detected. Distribution changes can produce false positives, and training data may contain anomalies. The evaluation uses a chronological holdout, not cross-validation or the official NAB scoring procedure.
