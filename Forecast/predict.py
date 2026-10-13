"""24-hour temperature forecasting with the trained LSTM.

The model is trained on **scaled** features (``Transform/transform_features
.py``), so the prediction flow is:

1. Load ``Model/lstm_weather_model.h5`` and ``Model/scaler.pkl``.
2. Rebuild the full model feature matrix (scaled features + cyclical
   Hour/Day/Month sine-cosine encodings) from ``data/scaled_features.parquet``.
3. For each city, seed the model with the **last 168 hours** and recursively
   predict the next temperature hour-by-hour for the configured horizon,
   feeding each prediction back into the window.
4. Invert the target scaling back to physical Celsius and export a 24-hour
   forecast to ``Forecast/forecast.json``.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Train.feature_engineering import (  # noqa: E402
    MODEL_FEATURE_COLUMNS,
    TEMP_INDEX,
)
from Transform.transform_features import load_scaler  # noqa: E402
from Workflow.config import load_config  # noqa: E402

logger = logging.getLogger(__name__)

# Physically plausible output range (helps damp recursive prediction drift).
MIN_TEMPERATURE_C = -60.0
MAX_TEMPERATURE_C = 60.0


def _isoformat_utc(value) -> str:
    """Render a (possibly tz-aware) timestamp as an ISO-8601 UTC string."""
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def _inverse_temperature_prediction(
    scaler,
    scaled_temperature: float,
) -> float:
    """Convert a scaled temperature back to Celsius using the fitted scaler.

    ``StandardScaler`` is element-wise, so the inverse of a single column is
    ``value * scale_[temp_idx] + mean_[temp_idx]``.
    """
    temp_index = TEMP_INDEX
    raw = (
        float(scaled_temperature) * float(scaler.scale_[temp_index])
        + float(scaler.mean_[temp_index])
    )
    return float(np.clip(raw, MIN_TEMPERATURE_C, MAX_TEMPERATURE_C))


def predict_city_horizon(
    model,
    scaler,
    city_frame: pd.DataFrame,
    horizon: int = 24,
    lookback: int = 168,
) -> Dict[str, Any]:
    """Recursively forecast the next ``horizon`` hours for one city.

    ``city_frame`` is the *full* model feature matrix (scaled + cyclic) for a
    single city sorted by time. Returns ``{"timestamps": [...], "temperature_c": [...]}``.
    """
    if len(city_frame) < lookback:
        raise ValueError(
            f"Only {len(city_frame)} rows available, need at least {lookback}."
        )

    window = city_frame[MODEL_FEATURE_COLUMNS].to_numpy(dtype="float32")[-lookback:]
    last_time = pd.Timestamp(city_frame["observation_time"].iloc[-1])
    if last_time.tzinfo is None:
        last_time = last_time.tz_localize("UTC")

    timestamps: List[str] = []
    temperatures: List[float] = []

    # Reference row (scaled) used to carry forward static feature columns
    # while we only advance temperature + the cyclical time encodings.
    last_row = np.copy(window[-1])

    for step in range(1, horizon + 1):
        target_time = last_time + pd.Timedelta(hours=step)

        x_batch = window[np.newaxis, :, :]  # (1, lookback, n_features)
        predicted_scaled = float(model.predict(x_batch, verbose=0)[0, 0])

        # Update the newest row: predicted temperature + fresh cyclics.
        next_row = np.copy(last_row)
        next_row[TEMP_INDEX] = predicted_scaled
        cyclic_values = _cyclic_features_for(target_time)
        next_row[len(next_row) - len(cyclic_values):] = cyclic_values

        # Roll the window forward by one step.
        window = np.vstack([window[1:], next_row])
        last_row = next_row

        timestamps.append(_isoformat_utc(target_time))
        temperatures.append(_inverse_temperature_prediction(scaler, predicted_scaled))

    return {
        "timestamps": timestamps,
        "temperature_c": [round(value, 2) for value in temperatures],
    }


def _cyclic_features_for(timestamp: pd.Timestamp) -> np.ndarray:
    """Raw sine/cosine encodings for a timestamp (kept unscaled on purpose)."""
    hour = timestamp.hour
    day = timestamp.day
    month = timestamp.month
    return np.asarray(
        [
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
            np.sin(2 * np.pi * day / 31),
            np.cos(2 * np.pi * day / 31),
            np.sin(2 * np.pi * month / 12),
            np.cos(2 * np.pi * month / 12),
        ],
        dtype="float32",
    )


def load_trained_model(model_path: str | Path):
    """Load the Keras model, failing loudly if it is missing/empty."""
    model_path = Path(model_path)
    if not model_path.exists() or model_path.stat().st_size == 0:
        raise FileNotFoundError(
            f"Model file {model_path} is missing or empty - run "
            "Train/train_lstm.py first."
        )
    from tensorflow.keras.models import load_model

    # Thêm compile=False để tránh lỗi khi nạp lại model HDF5 cũ
    return load_model(str(model_path), compile=False)

def run(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Load the trained model and export a 24-hour temperature forecast.

    Writes ``Forecast/forecast.json`` with one entry per city.
    """
    config = config if config is not None else load_config()
    model_path = Path(config["paths"]["model"])
    scaler_path = Path(config["paths"]["scaler"])
    scaled_path = Path(config["paths"]["scaled_features"])
    output_path = Path(config["paths"]["forecast_output"])
    horizon = int(config["pipeline"].get("forecast_horizon_hours", 24))
    lookback = int(config["pipeline"].get("lookback_hours", 168))

    model = load_trained_model(model_path)
    scaler = load_scaler(scaler_path)

    if not scaled_path.exists():
        raise FileNotFoundError(
            f"{scaled_path} not found - run Transform/transform_features.py first."
        )
    scaled_features = pd.read_parquet(scaled_path)
    scaled_features["observation_time"] = pd.to_datetime(
        scaled_features["observation_time"], utc=True
    )

    # Rebuild the full matrix (scaled features + cyclic encodings) exactly as
    # the training stage saw it.
    from Train.feature_engineering import build_full_matrix

    frame = build_full_matrix(scaled_features)

    forecasts: List[Dict[str, Any]] = []
    skipped: List[str] = []
    for city, group in frame.groupby("city", sort=False):
        city_frame = group.sort_values("observation_time").reset_index(drop=True)
        if len(city_frame) < lookback:
            logger.warning(
                "City %s has only %d rows (< %d); skipping forecast.",
                city, len(city_frame), lookback,
            )
            skipped.append(str(city))
            continue
        forecast = predict_city_horizon(
            model, scaler, city_frame, horizon=horizon, lookback=lookback
        )
        forecast["city"] = str(city)
        forecasts.append(forecast)
        logger.info(
            "Forecast for %s: %s -> %s C",
            city, forecast["timestamps"][0], forecast["temperature_c"][0],
        )

    if not forecasts:
        raise RuntimeError("No forecasts could be generated for any city.")

    payload = {
        "generated_at": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model_path": str(model_path),
        "lookback_hours": lookback,
        "horizon_hours": horizon,
        "forecasts": forecasts,
        "skipped_cities": skipped,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    logger.info("Wrote 24-hour forecast to %s", output_path)

    return {
        "forecast_path": str(output_path),
        "cities": [f["city"] for f in forecasts],
        "horizon_hours": horizon,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    report = run()
    print(report)
