from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = (
    "value",
    "change",
    "abs_change",
    "previous_mean_6",
    "previous_std_6",
    "previous_mean_24",
    "previous_std_24",
    "deviation_6",
)


def make_features(frame: pd.DataFrame) -> pd.DataFrame:
    values = frame["value"].astype(float)
    prior = values.shift(1)
    mean_6 = prior.rolling(6, min_periods=6).mean()
    result = frame[["timestamp", "value"]].copy()
    result["change"] = values.diff()
    result["abs_change"] = result["change"].abs()
    result["previous_mean_6"] = mean_6
    result["previous_std_6"] = prior.rolling(6, min_periods=6).std()
    result["previous_mean_24"] = prior.rolling(24, min_periods=24).mean()
    result["previous_std_24"] = prior.rolling(24, min_periods=24).std()
    result["deviation_6"] = values - mean_6
    return result.dropna().reset_index(drop=True)


@dataclass(frozen=True)
class DetectionResult:
    points: pd.DataFrame
    train_rows: int
    threshold: float


def detect(
    frame: pd.DataFrame,
    alert_rate: float = 0.04,
    train_fraction: float = 0.60,
) -> DetectionResult:
    if not 0.005 <= alert_rate <= 0.15:
        raise ValueError("alert_rate must be between 0.005 and 0.15")
    if not 0.3 <= train_fraction <= 0.8:
        raise ValueError("train_fraction must be between 0.3 and 0.8")

    features = make_features(frame)
    boundary = int(len(features) * train_fraction)
    if boundary < 50 or len(features) - boundary < 20:
        raise ValueError("Not enough data for training and evaluation; add more observations.")

    pipeline = make_pipeline(
        StandardScaler(),
        IsolationForest(n_estimators=200, contamination="auto", random_state=42, n_jobs=-1),
    )
    training = features.iloc[:boundary][list(FEATURES)]
    pipeline.fit(training)
    training_scores = -pipeline.score_samples(training)
    threshold = float(np.quantile(training_scores, 1 - alert_rate))

    result = features[["timestamp", "value"]].copy()
    result["score"] = -pipeline.score_samples(features[list(FEATURES)])
    result["is_anomaly"] = result["score"] > threshold
    result["split"] = np.where(result.index < boundary, "Training", "Evaluation")
    return DetectionResult(points=result, train_rows=boundary, threshold=threshold)


def rolling_zscore(frame: pd.DataFrame, threshold: float = 3.0) -> pd.Series:
    values = frame["value"].astype(float)
    prior = values.shift(1)
    mean = prior.rolling(24, min_periods=24).mean()
    std = prior.rolling(24, min_periods=24).std()
    score = (values - mean).abs() / std.replace(0, np.nan)
    return (score > threshold).fillna(False)
