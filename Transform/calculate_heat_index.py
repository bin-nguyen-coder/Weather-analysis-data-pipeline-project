"""Comfort metrics: Heat Index, Wind Chill and rolling-window statistics.

Formulas implemented
--------------------
* **Heat Index** (NOAA Rothfusz regression): valid for T >= 27 °C and
  RH >= 40 %. The NWS adjustments for low-RH / high-RH regimes are applied.
  Outside its validity envelope the heat index is set to the air temperature
  so downstream aggregates stay continuous.
* **Wind Chill** (Environment Canada / NWS): valid when T <= 10 °C and
  wind speed >= 4.8 km/h. Otherwise set to the air temperature.
* **Rolling metrics**: trailing ``window``-hour means / std / min / max of
  temperature and a trailing mean of relative humidity, computed per city.

The derived metrics are UPSERTed into ``analytics.weather_metrics``, which
feeds both the SQL aggregation queries and the ML feature engineering.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Ingestion.db_upsert import _to_db_value, get_connection, init_schema  # noqa: E402
from Workflow.config import load_config  # noqa: E402

logger = logging.getLogger(__name__)

# Column order for ``analytics.weather_metrics``.
METRICS_COLUMNS = [
    "city", "latitude", "longitude", "observation_time",
    "temperature_c", "relative_humidity", "dew_point_c",
    "pressure_msl", "wind_speed_kmh", "wind_direction",
    "weather_code", "precipitation_mm", "cloud_cover",
    "heat_index_c", "wind_chill_c",
    "temp_rolling_mean_24h", "temp_rolling_std_24h",
    "temp_rolling_min_24h", "temp_rolling_max_24h",
    "humidity_rolling_mean_24h",
]


def calculate_heat_index(
    temperature_c: pd.Series, relative_humidity: pd.Series
) -> pd.Series:
    """NOAA heat index in Celsius (with NWS low/high-RH adjustments)."""
    t_f = temperature_c * 9.0 / 5.0 + 32.0
    rh = relative_humidity.clip(0, 100)

    # Rothfusz regression.
    heat_index_f = (
        -42.379
        + 2.04901523 * t_f
        + 10.14333127 * rh
        - 0.22475541 * t_f * rh
        - 0.00683783 * t_f ** 2
        - 0.05481717 * rh ** 2
        + 0.00122874 * t_f ** 2 * rh
        + 0.00085282 * t_f * rh ** 2
        - 0.00000199 * t_f ** 2 * rh ** 2
    )

    # NWS adjustment: very dry conditions.
    dry_adj = np.where(
        (rh <= 13) & (t_f >= 80) & (t_f <= 112),
        (13 - rh) / 4 * np.sqrt(np.maximum(17 - np.abs(t_f - 95), 0) / 17),
        0.0,
    )
    heat_index_f = heat_index_f - dry_adj

    # NWS adjustment: very humid & warm conditions.
    humid_adj = np.where(
        (rh > 85) & (t_f >= 80) & (t_f <= 87),
        (rh - 85) / 10 * ((87 - t_f) / 5),
        0.0,
    )
    heat_index_f = heat_index_f + humid_adj

    # If the computed value is lower than T, the heat index equals T.
    heat_index_f = np.maximum(heat_index_f, t_f)

    result = (heat_index_f - 32) * 5.0 / 9.0
    # Outside the validity envelope (T < 27 C or RH < 40%) fall back to T.
    valid = (temperature_c >= 27) & (relative_humidity >= 40)
    return pd.Series(
        np.where(valid, result, temperature_c),
        index=temperature_c.index,
        name="heat_index_c",
    )


def calculate_wind_chill(
    temperature_c: pd.Series, wind_speed_kmh: pd.Series
) -> pd.Series:
    """Environment Canada / NWS wind chill in Celsius."""
    wind = wind_speed_kmh.clip(lower=0)
    wind_chill = (
        13.12
        + 0.6215 * temperature_c
        - 11.37 * wind ** 0.16
        + 0.3965 * temperature_c * wind ** 0.16
    )
    valid = (temperature_c <= 10) & (wind >= 4.8)
    return pd.Series(
        np.where(valid, wind_chill, temperature_c),
        index=temperature_c.index,
        name="wind_chill_c",
    )


def add_rolling_metrics(
    df: pd.DataFrame, window: int = 24
) -> pd.DataFrame:
    """Trailing ``window``-hour rolling stats, computed per city series."""
    df = df.sort_values(["city", "observation_time"]).copy()
    grouped = df.groupby("city", sort=False)

    spec = [
        ("temperature_c", "temp_rolling_mean_24h", "mean"),
        ("temperature_c", "temp_rolling_std_24h", "std"),
        ("temperature_c", "temp_rolling_min_24h", "min"),
        ("temperature_c", "temp_rolling_max_24h", "max"),
        ("relative_humidity", "humidity_rolling_mean_24h", "mean"),
    ]
    for source, target, operation in spec:
        min_periods = 2 if operation == "std" else 1
        df[target] = grouped[source].transform(
            lambda s, op=operation, mp=min_periods: s.rolling(
                window=window, min_periods=mp
            ).agg(op)
        )
    return df


def add_comfort_metrics(
    df: pd.DataFrame, window: int = 24
) -> pd.DataFrame:
    """Add heat index, wind chill and rolling metrics to ``df``."""
    df = df.copy()
    df["heat_index_c"] = calculate_heat_index(df["temperature_c"], df["relative_humidity"])
    df["wind_chill_c"] = calculate_wind_chill(df["temperature_c"], df["wind_speed_kmh"])
    return add_rolling_metrics(df, window=window)


def upsert_metrics(conn: Any, df: pd.DataFrame) -> int:
    """Idempotently UPSERT derived metrics into ``analytics.weather_metrics``."""
    missing = [col for col in METRICS_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing metric columns: {missing}")

    rows = [
        tuple(_to_db_value(value) for value in record)
        for record in df[METRICS_COLUMNS].itertuples(index=False, name=None)
    ]
    if not rows:
        return 0

    from psycopg2.extras import execute_values

    column_list = ", ".join(METRICS_COLUMNS)
    update_columns = [c for c in METRICS_COLUMNS if c not in ("city", "observation_time")]
    assignments = ", ".join(f"{c} = EXCLUDED.{c}" for c in update_columns)
    sql = (
        f"INSERT INTO analytics.weather_metrics ({column_list}) VALUES %s "
        f"ON CONFLICT (city, observation_time) DO UPDATE SET {assignments}"
    )
    with conn.cursor() as cursor:
        execute_values(cursor, sql, rows, page_size=500)
    conn.commit()
    logger.info("UPSERTED %d rows into analytics.weather_metrics", len(rows))
    return len(rows)


def run(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Compute comfort metrics over the cleaned layer and persist them."""
    config = config if config is not None else load_config()
    window = int(config["pipeline"].get("rolling_window_hours", 24))

    conn = get_connection(config)
    try:
        init_schema(conn)
        frame = pd.read_sql_query(
            "SELECT * FROM cleaned.weather_observations "
            "ORDER BY city, observation_time",
            conn,
        )
        if frame.empty:
            logger.warning("No cleaned rows to transform.")
            return {"metrics_rows": 0}

        metrics = add_comfort_metrics(frame, window=window)
        count = upsert_metrics(conn, metrics)
        return {"metrics_rows": count}
    finally:
        conn.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    report = run()
    print(report)
