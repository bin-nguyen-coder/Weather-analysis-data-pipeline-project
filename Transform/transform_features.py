"""Feature-vector preparation: impute + scale the metrics table.

This step turns ``analytics.weather_metrics`` into a **scaled feature matrix**
consumed by the ML stages:

* selects the numeric ML feature columns (the *temperature target* is one of
  them, so a single ``StandardScaler`` covers features *and* the target);
* imputes any leftover missing values (per-city interpolation first, then a
  conservative global fallback);
* fits a ``sklearn.preprocessing.StandardScaler`` and applies it;
* persists the scaled matrix to ``data/scaled_features.parquet`` and the
  fitted scaler to ``Model/scaler.pkl`` (required later by ``predict.py`` to
  convert model output back to physical Celsius).
"""

from __future__ import annotations

import logging
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Ingestion.db_upsert import get_connection, init_schema  # noqa: E402
from Workflow.config import load_config  # noqa: E402

logger = logging.getLogger(__name__)

# ML feature columns (order is contractual for training & prediction).
# The first column, ``temperature_c``, is the forecasting target.
FEATURE_COLUMNS: List[str] = [
    "temperature_c",
    "relative_humidity",
    "dew_point_c",
    "pressure_msl",
    "wind_speed_kmh",
    "wind_direction",
    "cloud_cover",
    "heat_index_c",
    "wind_chill_c",
]

TARGET_COLUMN = "temperature_c"


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, StandardScaler]:
    """Impute, then standardize the ML feature columns.

    Returns
    -------
    (scaled_frame, scaler)
        ``scaled_frame`` carries ``city`` / ``observation_time`` plus the
        standardized feature columns.
    """
    df = df.sort_values(["city", "observation_time"]).copy()
    missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Metrics table missing feature columns: {missing}")

    features = df[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).astype("float64")

    # Per-city time interpolation, then a neutral global fallback so that
    # scaler.fit never sees NaN.
    for column in FEATURE_COLUMNS:
        if features[column].isna().sum() == 0:
            continue
        features[column] = df.groupby("city", sort=False)[column].transform(
            lambda s, col=column: s.interpolate(method="linear", limit_direction="both")
            .ffill()
            .bfill()
        )
        global_fallback = features[column].median()
        if pd.isna(global_fallback):  # whole column was empty
            global_fallback = 0.0
        features[column] = features[column].fillna(global_fallback)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    scaled_frame = pd.DataFrame(
        scaled, columns=FEATURE_COLUMNS, index=df.index
    )
    scaled_frame.insert(0, "observation_time", df["observation_time"].values)
    scaled_frame.insert(0, "city", df["city"].values)

    logger.debug("Fitted StandardScaler with %d features", len(FEATURE_COLUMNS))
    return scaled_frame.reset_index(drop=True), scaler


def save_scaler(scaler: StandardScaler, path: str | Path) -> None:
    """Persist the fitted scaler to ``path`` (typically ``Model/scaler.pkl``)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        pickle.dump(scaler, handle)
    logger.info("Saved scaler to %s", path)


def load_scaler(path: str | Path) -> StandardScaler:
    """Load a previously fitted ``StandardScaler`` from ``path``."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Scaler file {path} not found - run Transform/transform_features.py first."
        )
    with open(path, "rb") as handle:
        return pickle.load(handle)


def run(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Prepare scaled feature vectors from ``analytics.weather_metrics``.

    Writes ``data/scaled_features.parquet`` and ``Model/scaler.pkl``.
    """
    config = config if config is not None else load_config()
    scaled_path = Path(config["paths"]["scaled_features"])
    scaler_path = Path(config["paths"]["scaler"])
    scaled_path.parent.mkdir(parents=True, exist_ok=True)

    conn = get_connection(config)
    try:
        init_schema(conn)
        frame = pd.read_sql_query(
            "SELECT * FROM analytics.weather_metrics "
            "ORDER BY city, observation_time",
            conn,
        )
        if frame.empty:
            raise RuntimeError(
                "analytics.weather_metrics is empty - run the transform "
                "stage before preparing features."
            )

        scaled_frame, scaler = prepare_features(frame)
        scaled_frame.to_parquet(scaled_path, index=False)
        save_scaler(scaler, scaler_path)

        logger.info(
            "Wrote %d scaled feature rows to %s", len(scaled_frame), scaled_path
        )
        return {
            "scaled_rows": int(len(scaled_frame)),
            "features": FEATURE_COLUMNS,
            "scaled_features_path": str(scaled_path),
            "scaler_path": str(scaler_path),
        }
    finally:
        conn.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    report = run()
    print(report)
