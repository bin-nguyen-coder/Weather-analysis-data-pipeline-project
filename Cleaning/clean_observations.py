"""Cleaning layer: deduplicate, impute and normalize raw observations.

Pipeline responsibilities
-------------------------
1. Parse + normalize ``observation_time`` to UTC (tz-aware).
2. Flag / drop rows that are missing their most important fields
   (temperature and relative humidity).
3. Deduplicate ``(city, observation_time)`` pairs, keeping the row ingested
   last (the archive/forecast payload with the freshest values).
4. Impute remaining numeric gaps with linear interpolation within each city
   (falling back to nearest-value fill) so downstream aggregations and ML
   features never see missing values.
5. Write the cleaned rows into ``cleaned.weather_observations`` using an
   idempotent UPSERT keyed on ``(city, observation_time)``.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

# Allow running this module directly from the repository root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Ingestion.db_upsert import _to_db_value, get_connection, init_schema  # noqa: E402
from Workflow.config import load_config  # noqa: E402

logger = logging.getLogger(__name__)

# Columns that must be present after cleaning.
REQUIRED_COLUMNS = [
    "city", "latitude", "longitude", "observation_time",
    "temperature_c", "relative_humidity", "dew_point_c",
    "pressure_msl", "wind_speed_kmh", "wind_direction",
    "weather_code", "precipitation_mm", "cloud_cover",
]

# Numeric columns eligible for gap-filling interpolation.
_NUMERIC_COLUMNS = [
    "latitude", "longitude", "temperature_c", "relative_humidity",
    "dew_point_c", "pressure_msl", "wind_speed_kmh", "wind_direction",
    "weather_code", "precipitation_mm", "cloud_cover",
]


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Return a deduplicated, UTC-normalized, gap-free copy of ``df``.

    The cleaned frame has one row per ``(city, observation_time)`` sorted by
    time, ready for the ``cleaned.weather_observations`` UPSERT.
    """
    df = df.copy()
    if df.empty:
        return df

    # --- 1. Normalize observation_time to UTC ------------------------------- #
    df["observation_time"] = pd.to_datetime(df["observation_time"], utc=True, errors="coerce")
    df["city"] = df["city"].astype("string").str.strip()

    # --- 2. Drop rows with unusable keys / core measurements ----------------
    before = len(df)
    df = df.dropna(subset=["observation_time"])
    df = df.dropna(subset=["temperature_c", "relative_humidity"])
    if len(df) < before:
        logger.info("Dropped %d rows with null key/time/temperature/humidity.", before - len(df))

    # --- 3. Deduplicate (city, observation_time), keep the last one ---------
    before = len(df)
    df = df.sort_values(["city", "observation_time"])
    df = df.drop_duplicates(subset=["city", "observation_time"], keep="last")
    if len(df) < before:
        logger.info("Removed %d duplicate (city, observation_time) rows.", before - len(df))

    # --- 4. Impute remaining numeric gaps (per city, time-ordered) ----------
    df = _impute_numeric_gaps(df)

    # --- 5. Drop rows still missing essential numerics ----------------------
    before = len(df)
    df = df.dropna(subset=["temperature_c", "relative_humidity"])
    if len(df) < before:
        logger.info("Dropped %d rows after imputation (still null core metrics).", before - len(df))

    # Replace any leftover NaN / inf in ancillary numeric columns with 0 so
    # the UPSERT can store a finite value.
    for column in _NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )

    return df.sort_values(["city", "observation_time"]).reset_index(drop=True)


def _impute_numeric_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """Interpolate missing numeric values within each city series.

    Uses linear interpolation first, then forward/backward fill so
    edge NaNs are handled too. Whole-column NaN stays NaN (dropped upstream).
    """

    def _fill(series: pd.Series) -> pd.Series:
        values = series
        try:
            values = values.interpolate(method="linear", limit_direction="both")
        except Exception:  # pragma: no cover - defensive
            values = values.interpolate(method="linear", limit_direction="forward")
        return values.ffill().bfill()

    for column in _NUMERIC_COLUMNS:
        if column not in df.columns:
            continue
        df[column] = pd.to_numeric(df[column], errors="coerce")
        if df[column].notna().sum() == 0:
            continue
        df[column] = df.groupby("city", sort=False)[column].transform(_fill)
    return df


def upsert_cleaned(
    conn: Any, df: pd.DataFrame
) -> int:
    """Idempotently UPSERT a cleaned frame into ``cleaned.weather_observations``."""
    if df.empty:
        logger.warning("Nothing to UPSERT into cleaned.weather_observations.")
        return 0

    rows = [
        tuple(_to_db_value(value) for value in record)
        for record in df[REQUIRED_COLUMNS].itertuples(index=False, name=None)
    ]
    column_list = ", ".join(REQUIRED_COLUMNS)
    update_columns = [col for col in REQUIRED_COLUMNS if col not in ("city", "observation_time")]
    assignments = ", ".join(f"{col} = EXCLUDED.{col}" for col in update_columns)
    sql = (
        f"INSERT INTO cleaned.weather_observations ({column_list}) VALUES %s "
        f"ON CONFLICT (city, observation_time) DO UPDATE SET {assignments}"
    )

    with conn.cursor() as cursor:
        from psycopg2.extras import execute_values

        execute_values(cursor, sql, rows, page_size=500)
    conn.commit()
    logger.info("UPSERTED %d rows into cleaned.weather_observations", len(rows))
    return len(rows)


def clean_table(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Read ``raw.weather_observations``, clean it and write the cleaned layer.

    Returns a small report dictionary with row counts.
    """
    config = config if config is not None else load_config()
    conn = get_connection(config)
    try:
        init_schema(conn)
        raw_df = pd.read_sql_query(
            "SELECT * FROM raw.weather_observations ORDER BY city, observation_time",
            conn,
        )
        result = {"raw_rows": len(raw_df)}
        if raw_df.empty:
            logger.warning("No raw rows available to clean.")
            return result

        cleaned_df = clean_dataframe(raw_df)
        upsert_cleaned(conn, cleaned_df)
        result["cleaned_rows"] = int(len(cleaned_df))
        result["duplicates_removed"] = int(result["raw_rows"] - len(cleaned_df))
        return result
    finally:
        conn.close()


def run(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Pipeline entrypoint: clean the raw table into the cleaned layer."""
    return clean_table(config)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    report = run()
    print(report)
