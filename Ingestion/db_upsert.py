"""PostgreSQL connectivity + idempotent UPSERT for the pipeline warehouse.

Responsibilities
----------------
* Create (if missing) the schemas and tables used by the whole pipeline:
    - ``raw``            : raw hourly observations straight from Open-Meteo
    - ``cleaned``        : deduplicated, UTC-normalized observations
    - ``analytics``      : comfort metrics, daily aggregates, city stats
* Persist raw observations with an ``INSERT ... ON CONFLICT DO UPDATE``
  UPSERT keyed on ``(city, observation_time)`` so re-runs are idempotent.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

# Allow ``python Ingestion/db_upsert.py`` to run from the repo root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Workflow.config import load_config  # noqa: E402

logger = logging.getLogger(__name__)

# Column order for ``raw.weather_observations`` (mirrors the Open-Meteo frame).
RAW_COLUMNS: List[str] = [
    "city", "latitude", "longitude", "observation_time",
    "temperature_c", "relative_humidity", "dew_point_c",
    "pressure_msl", "wind_speed_kmh", "wind_direction",
    "weather_code", "precipitation_mm", "cloud_cover", "is_forecast",
]

# Unique key for the raw UPSERT.
UNIQUE_KEY = ("city", "observation_time")

# Columns updated on conflict (identity/key columns are excluded).
_RAW_UPDATE_COLUMNS = [
    "latitude", "longitude", "temperature_c", "relative_humidity",
    "dew_point_c", "pressure_msl", "wind_speed_kmh", "wind_direction",
    "weather_code", "precipitation_mm", "cloud_cover", "is_forecast",
    "ingested_at",
]

SCHEMA_DDL = """
-- ---------------------------------------------------------------------------
-- Warehouse schema (idempotent)
-- ---------------------------------------------------------------------------
CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS cleaned;
CREATE SCHEMA IF NOT EXISTS analytics;

-- 1. Raw ingestion landing table
CREATE TABLE IF NOT EXISTS raw.weather_observations (
    city                TEXT           NOT NULL,
    latitude            DOUBLE PRECISION,
    longitude           DOUBLE PRECISION,
    observation_time    TIMESTAMPTZ    NOT NULL,
    temperature_c       DOUBLE PRECISION,
    relative_humidity   DOUBLE PRECISION,
    dew_point_c         DOUBLE PRECISION,
    pressure_msl        DOUBLE PRECISION,
    wind_speed_kmh      DOUBLE PRECISION,
    wind_direction      DOUBLE PRECISION,
    weather_code        SMALLINT,
    precipitation_mm    DOUBLE PRECISION,
    cloud_cover         DOUBLE PRECISION,
    is_forecast         BOOLEAN        NOT NULL DEFAULT FALSE,
    ingested_at         TIMESTAMPTZ    NOT NULL DEFAULT NOW(),
    PRIMARY KEY (city, observation_time)
);
CREATE INDEX IF NOT EXISTS idx_raw_observations_time
    ON raw.weather_observations (observation_time);

-- 2. Cleaned layer
CREATE TABLE IF NOT EXISTS cleaned.weather_observations (
    city                TEXT           NOT NULL,
    latitude            DOUBLE PRECISION,
    longitude           DOUBLE PRECISION,
    observation_time    TIMESTAMPTZ    NOT NULL,
    temperature_c       DOUBLE PRECISION NOT NULL,
    relative_humidity   DOUBLE PRECISION NOT NULL,
    dew_point_c         DOUBLE PRECISION,
    pressure_msl        DOUBLE PRECISION,
    wind_speed_kmh      DOUBLE PRECISION,
    wind_direction      DOUBLE PRECISION,
    weather_code        SMALLINT,
    precipitation_mm    DOUBLE PRECISION,
    cloud_cover         DOUBLE PRECISION,
    cleaned_at          TIMESTAMPTZ    NOT NULL DEFAULT NOW(),
    PRIMARY KEY (city, observation_time)
);
CREATE INDEX IF NOT EXISTS idx_cleaned_observations_time
    ON cleaned.weather_observations (observation_time);

-- 3. Analytics: per-hour comfort metrics (Heat Index, Wind Chill, rollings)
CREATE TABLE IF NOT EXISTS analytics.weather_metrics (
    city                       TEXT           NOT NULL,
    latitude                   DOUBLE PRECISION,
    longitude                  DOUBLE PRECISION,
    observation_time           TIMESTAMPTZ    NOT NULL,
    temperature_c              DOUBLE PRECISION NOT NULL,
    relative_humidity          DOUBLE PRECISION NOT NULL,
    dew_point_c                DOUBLE PRECISION,
    pressure_msl               DOUBLE PRECISION,
    wind_speed_kmh             DOUBLE PRECISION,
    wind_direction             DOUBLE PRECISION,
    weather_code               SMALLINT,
    precipitation_mm           DOUBLE PRECISION,
    cloud_cover                DOUBLE PRECISION,
    heat_index_c               DOUBLE PRECISION,
    wind_chill_c               DOUBLE PRECISION,
    temp_rolling_mean_24h      DOUBLE PRECISION,
    temp_rolling_std_24h       DOUBLE PRECISION,
    temp_rolling_min_24h       DOUBLE PRECISION,
    temp_rolling_max_24h       DOUBLE PRECISION,
    humidity_rolling_mean_24h  DOUBLE PRECISION,
    PRIMARY KEY (city, observation_time)
);
CREATE INDEX IF NOT EXISTS idx_metrics_time
    ON analytics.weather_metrics (observation_time);

-- 4. Analytics: per-city daily aggregates (populated by daily_aggregates.sql)
CREATE TABLE IF NOT EXISTS analytics.daily_aggregates (
    city                    TEXT        NOT NULL,
    observation_date        DATE        NOT NULL,
    avg_temperature_c       DOUBLE PRECISION,
    min_temperature_c       DOUBLE PRECISION,
    max_temperature_c       DOUBLE PRECISION,
    avg_relative_humidity   DOUBLE PRECISION,
    avg_heat_index_c        DOUBLE PRECISION,
    avg_wind_chill_c        DOUBLE PRECISION,
    avg_wind_speed_kmh      DOUBLE PRECISION,
    total_precipitation_mm  DOUBLE PRECISION,
    num_observations        INTEGER,
    PRIMARY KEY (city, observation_date)
);

-- 5. Analytics: cross-city summary statistics (populated by city_stats.sql)
CREATE TABLE IF NOT EXISTS analytics.city_stats (
    city                   TEXT NOT NULL PRIMARY KEY,
    observation_count      INTEGER,
    first_observation      TIMESTAMPTZ,
    last_observation       TIMESTAMPTZ,
    avg_temperature_c      DOUBLE PRECISION,
    min_temperature_c      DOUBLE PRECISION,
    max_temperature_c      DOUBLE PRECISION,
    stddev_temperature_c   DOUBLE PRECISION,
    avg_relative_humidity  DOUBLE PRECISION,
    avg_heat_index_c       DOUBLE PRECISION,
    avg_wind_speed_kmh     DOUBLE PRECISION,
    total_precipitation_mm DOUBLE PRECISION
);
"""


def get_connection(config: Optional[Dict[str, Any]] = None) -> psycopg2.extensions.connection:
    """Open a psycopg2 connection using ``workflow_config.json`` + env vars.

    The connection is created uncommitted; callers must ``commit()`` after
    mutating statements (helpers in this module do it for you).
    """
    config = config if config is not None else load_config()
    db = config["database"]
    logger.info(
        "Connecting to postgresql://%s@%s:%s/%s",
        db.get("user"), db.get("host"), db.get("port"), db.get("dbname"),
    )
    return psycopg2.connect(
        host=db.get("host", "localhost"),
        port=int(db.get("port", 5432)),
        dbname=db.get("dbname", "weather"),
        user=db.get("user", "weather_user"),
        password=db.get("password", "weather_password"),
    )


def init_schema(conn: psycopg2.extensions.connection) -> None:
    """Create all schemas / tables used by the pipeline (idempotent)."""
    with conn.cursor() as cursor:
        cursor.execute(SCHEMA_DDL)
    conn.commit()
    logger.info("Warehouse schema ensured (raw, cleaned, analytics).")


def _to_db_value(value: Any) -> Any:
    """Convert pandas primitives into PostgreSQL-friendly Python values.

    Handles ``pd.NaT``/``np.nan`` -> ``None``, tz-aware ``pd.Timestamp`` ->
    native ``datetime``, and ``numpy`` scalars -> Python scalars.
    """
    if value is None:
        return None
    if pd.isna(value):  # catches np.nan and pd.NaT
        return None
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def upsert_dataframe(conn: psycopg2.extensions.connection, df: pd.DataFrame) -> int:
    """Idempotently UPSERT ``df`` into ``raw.weather_observations``.

    Duplicate ``(city, observation_time)`` keys update the payload columns;
    new hours are inserted. Returns the number of rows UPSERTed.
    """
    missing = [col for col in RAW_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"DataFrame missing required columns: {missing}; got {list(df.columns)}"
        )

    rows = [
        tuple(_to_db_value(value) for value in record)
        for record in df[RAW_COLUMNS].itertuples(index=False, name=None)
    ]
    if not rows:
        logger.warning("No rows to UPSERT into raw.weather_observations.")
        return 0

    column_list = ", ".join(RAW_COLUMNS)
    assignments = ", ".join(f"{col} = EXCLUDED.{col}" for col in _RAW_UPDATE_COLUMNS)
    sql = (
        f"INSERT INTO raw.weather_observations ({column_list}) VALUES %s "
        f"ON CONFLICT ({', '.join(UNIQUE_KEY)}) DO UPDATE SET {assignments}"
    )

    with conn.cursor() as cursor:
        execute_values(cursor, sql, rows, page_size=500)
    conn.commit()
    logger.info("UPSERTED %d rows into raw.weather_observations", len(rows))
    return len(rows)


def fetch_and_upsert(config: Optional[Dict[str, Any]] = None) -> int:
    """End-to-end ingestion: fetch from Open-Meteo, then UPSERT to Postgres."""
    config = config if config is not None else load_config()
    # Local import avoids pulling `requests`/`tenacity` when only DB is needed.
    from Ingestion.client import run as fetch_data  # noqa: PLC0415

    frame = fetch_data(config)
    conn = get_connection(config)
    try:
        init_schema(conn)
        return upsert_dataframe(conn, frame)
    finally:
        conn.close()


def run(config: Optional[Dict[str, Any]] = None) -> int:
    """Pipeline entrypoint alias for ``fetch_and_upsert``."""
    return fetch_and_upsert(config)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    count = run()
    print(f"Ingested {count} raw rows.")
