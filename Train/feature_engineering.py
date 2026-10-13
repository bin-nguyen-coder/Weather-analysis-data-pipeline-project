"""Feature engineering for the LSTM forecaster.

Pipeline
--------
1. Load the scaled feature matrix written by
   ``Transform/transform_features.py`` (``data/scaled_features.parquet``).
2. Add **cyclical (sine/cosine) temporal encodings** for Hour / Day /
   Month so the model can learn time-of-day and seasonal structure:
       hour_sin/cos  = period 24 h
       day_sin/cos   = period 31 days
       month_sin/cos = period 12 months
3. Build a **168-hour lookback sliding window** dataset: every sample is a
   ``(168, n_features)`` matrix and the label is the *next-hour* scaled
   temperature.
4. Persist the compiled dataset to ``data/windows.npz`` together with the
   per-sample city / timestamp metadata and the feature name order
   (contractual for ``train_lstm.py`` and ``predict.py``).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Transform.transform_features import FEATURE_COLUMNS  # noqa: E402
from Workflow.config import load_config  # noqa: E402

logger = logging.getLogger(__name__)

TARGET_COLUMN = "temperature_c"  # must be within FEATURE_COLUMNS

# Cyclical temporal encodings (periods shown in the comments).
CYCLICAL_COLUMNS = [
    "hour_sin", "hour_cos",   # period 24 h
    "day_sin", "day_cos",     # period 31 days
    "month_sin", "month_cos", # period 12 months
]
_HOUR_PERIOD = 24
_DAY_PERIOD = 31
_MONTH_PERIOD = 12

# Full ordered feature list consumed by the LSTM: scaled features first
# (column 0 == temperature_c == the target) followed by the cyclic encodings.
MODEL_FEATURE_COLUMNS = FEATURE_COLUMNS + CYCLICAL_COLUMNS
TEMP_INDEX = MODEL_FEATURE_COLUMNS.index(TARGET_COLUMN)


def add_cyclical_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append sine/cosine encodings for hour, day-of-month and month."""
    df = df.copy()
    timestamps = pd.to_datetime(df["observation_time"], utc=True)

    hour = timestamps.dt.hour
    day = timestamps.dt.day
    month = timestamps.dt.month

    df["hour_sin"] = np.sin(2 * np.pi * hour / _HOUR_PERIOD)
    df["hour_cos"] = np.cos(2 * np.pi * hour / _HOUR_PERIOD)
    df["day_sin"] = np.sin(2 * np.pi * day / _DAY_PERIOD)
    df["day_cos"] = np.cos(2 * np.pi * day / _DAY_PERIOD)
    df["month_sin"] = np.sin(2 * np.pi * month / _MONTH_PERIOD)
    df["month_cos"] = np.cos(2 * np.pi * month / _MONTH_PERIOD)
    return df


def build_full_matrix(scaled_features: pd.DataFrame) -> pd.DataFrame:
    """Combine scaled features with the derived cyclic temporal columns."""
    frame = add_cyclical_time_features(scaled_features)
    missing = [c for c in MODEL_FEATURE_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing model feature columns: {missing}")
    if frame[MODEL_FEATURE_COLUMNS].isna().any().any():
        raise ValueError(
            "Feature matrix contains NaN - run Transform/transform_features.py."
        )
    return frame


def create_sliding_windows(
    frame: pd.DataFrame,
    lookback: int = 168,
    horizon: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build per-city sliding windows.

    Parameters
    ----------
    frame :
        Sorted (city, time) feature matrix (scaled + cyclic columns).
    lookback :
        Number of historical hours used as model input.
    horizon :
        Hours *ahead* of the window that the label targets
        (``horizon=1`` -> next-hour temperature).

    Returns
    -------
    (X, y, cities, timestamps)
        ``X``  shape ``(n, lookback, n_features)`` float32
        ``y``  shape ``(n,)`` float32 (scaled temperature at target hour)
    """
    X_parts = []
    y_parts = []
    city_parts = []
    time_parts = []

    for city, group in frame.groupby("city", sort=False):
        group = group.sort_values("observation_time")
        values = group[MODEL_FEATURE_COLUMNS].to_numpy(dtype="float32")
        times = group["observation_time"].to_numpy()

        if len(values) < lookback + horizon:
            logger.warning(
                "City %s has only %d rows (< %d + %d): skipping windows.",
                city, len(values), lookback, horizon,
            )
            continue

        starts = np.arange(len(values) - lookback - horizon + 1)
        for start in starts:
            end = int(start + lookback)
            X_parts.append(values[start:end])
            y_parts.append(values[end + horizon - 1, TEMP_INDEX])
            city_parts.append(city)
            time_parts.append(times[end + horizon - 1])

    if not X_parts:
        raise RuntimeError(
            "No sliding windows could be built - is there at least "
            f"{lookback + horizon} hours of data per city?"
        )

    X = np.asarray(X_parts, dtype="float32")
    y = np.asarray(y_parts, dtype="float32")
    cities = np.asarray(city_parts)
    timestamps = np.asarray(time_parts)
    logger.info("Built %d windows of shape %s", len(X), X.shape)
    return X, y, cities, timestamps


def run(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Build the windowed training dataset and persist ``data/windows.npz``."""
    config = config if config is not None else load_config()
    scaled_path = Path(config["paths"]["scaled_features"])
    windows_path = Path(config["paths"]["windows"])
    lookback = int(config["pipeline"].get("lookback_hours", 168))

    if not scaled_path.exists():
        raise FileNotFoundError(
            f"{scaled_path} not found - run Transform/transform_features.py first."
        )

    scaled_features = pd.read_parquet(scaled_path)
    frame = build_full_matrix(scaled_features)
    X, y, cities, timestamps = create_sliding_windows(
        frame, lookback=lookback, horizon=1
    )

    windows_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        windows_path,
        X=X,
        y=y,
        cities=cities,
        timestamps=timestamps,
        feature_names=np.asarray(MODEL_FEATURE_COLUMNS),
        lookback=np.asarray(lookback),
    )
    logger.info("Saved windowed dataset to %s", windows_path)

    return {
        "windows_path": str(windows_path),
        "n_windows": int(len(X)),
        "window_shape": list(X.shape),
        "feature_columns": MODEL_FEATURE_COLUMNS,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    report = run()
    print(report)
