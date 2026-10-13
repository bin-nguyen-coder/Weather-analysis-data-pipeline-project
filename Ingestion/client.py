"""Open-Meteo REST API client with rate limiting and retries.

Fetches both **historical hourly** data (archive API:
``https://archive-api.open-meteo.com/v1/archive``) and **live hourly**
forecast data (``https://api.open-meteo.com/v1/forecast``) for every city
listed in the workflow configuration.

Resilience:
  * ``tenacity`` applies exponential-backoff retries (1 initial attempt
    followed by up to 3 retries) for transient network errors and non-2xx
    HTTP responses.
  * A simple rate limiter guarantees a configurable minimum interval
    (default 1s) between consecutive requests to stay well under the
    Open-Meteo free-tier limits.
"""

from __future__ import annotations

import logging
import time
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from Workflow.config import load_config

logger = logging.getLogger(__name__)

# Hourly variables requested from both Open-Meteo APIs. Names are the exact
# Open-Meteo parameters; they map 1:1 onto our warehouse columns.
HOURLY_VARIABLES: List[str] = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "precipitation",
    "pressure_msl",
    "cloud_cover",
    "wind_speed_10m",
    "wind_direction_10m",
    "weather_code",
]

# 1 initial attempt + 3 retries = "3x exponential backoff".
MAX_RETRY_ATTEMPTS = 4
RETRYABLE_EXCEPTIONS = (requests.RequestException, requests.Timeout)


class OpenMeteoAPIError(RuntimeError):
    """Raised when Open-Meteo returns an HTTP error or an invalid payload."""


class OpenMeteoClient:
    """Small, resilient client for the Open-Meteo forecast + archive APIs."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        meteo = config["open_meteo"]
        self.forecast_url = meteo["forecast_url"]
        self.archive_url = meteo["archive_url"]
        self.min_interval = float(meteo.get("min_interval_seconds", 1.0))
        self.timeout = int(meteo.get("timeout_seconds", 30))
        self._last_request_monotonic = 0.0

    # ------------------------------------------------------------------ #
    # Rate limiting + resilient GET
    # ------------------------------------------------------------------ #
    def _rate_limit(self) -> None:
        """Sleep if needed to keep a minimum gap between API requests."""
        elapsed = time.monotonic() - self._last_request_monotonic
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_request_monotonic = time.monotonic()

    @retry(
        reraise=True,
        stop=stop_after_attempt(MAX_RETRY_ATTEMPTS),
        wait=wait_exponential(multiplier=1.0, min=2, max=30),
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS + (OpenMeteoAPIError,)),
    )
    def _get_json(self, url: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """GET ``url`` with rate limiting and exponential-backoff retries."""
        self._rate_limit()
        logger.info("GET %s params=%s", url, params)
        response = requests.get(url, params=params, timeout=self.timeout)
        if response.status_code != 200:
            raise OpenMeteoAPIError(
                f"Open-Meteo returned HTTP {response.status_code}: "
                f"{response.text[:200]}"
            )
        return response.json()

    # ------------------------------------------------------------------ #
    # Public fetch APIs
    # ------------------------------------------------------------------ #
    def fetch_historical(
        self,
        city: Dict[str, Any],
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """Hourly historical frame for ``city`` in ``[start, end]`` (UTC).

        NOTE: the Open-Meteo archive endpoint refuses future dates, so the
        caller normally passes ``end_date = yesterday``.
        """
        params: Dict[str, Any] = {
            "latitude": city["latitude"],
            "longitude": city["longitude"],
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "hourly": ",".join(HOURLY_VARIABLES),
            "timezone": "UTC",
        }
        payload = self._get_json(self.archive_url, params)
        return self._payload_to_frame(
            payload, city["name"], city["latitude"], city["longitude"],
            is_forecast=False,
        )

    def fetch_forecast(
        self,
        city: Dict[str, Any],
        forecast_days: int = 5,
    ) -> pd.DataFrame:
        """Live hourly forecast frame for ``city``.

        Only forecast hours still in the *future* are kept so historical
        (authoritative) rows are never overwritten by a forecast row.
        """
        params: Dict[str, Any] = {
            "latitude": city["latitude"],
            "longitude": city["longitude"],
            "hourly": ",".join(HOURLY_VARIABLES),
            "forecast_days": forecast_days,
            "timezone": "UTC",
        }
        payload = self._get_json(self.forecast_url, params)
        frame = self._payload_to_frame(
            payload, city["name"], city["latitude"], city["longitude"],
            is_forecast=True,
        )
        now_utc = pd.Timestamp.now(tz="UTC").floor("h")
        return frame[frame["observation_time"] >= now_utc].copy()

    def fetch_all(self) -> pd.DataFrame:
        """Fetch historical + forecast data for every configured city.

        Returns
        -------
        pandas.DataFrame
            Raw hourly observations with the columns expected by
            ``Ingestion.db_upsert.upsert_dataframe``.
        """
        meteo = self.config["open_meteo"]
        history_days = int(meteo.get("history_days", 45))
        forecast_days = int(meteo.get("forecast_days", 5))

        yesterday = date.today() - timedelta(days=1)
        start_date = yesterday - timedelta(days=history_days - 1)

        frames: List[pd.DataFrame] = []
        for city in self.config["cities"]:
            logger.info(
                "Fetching %s: historical=%s..%s forecast=%sd",
                city["name"], start_date, yesterday, forecast_days,
            )
            frames.append(self.fetch_historical(city, start_date, yesterday))
            frames.append(self.fetch_forecast(city, forecast_days=forecast_days))

        if not frames:
            return pd.DataFrame()

        raw = pd.concat(frames, ignore_index=True)
        logger.info(
            "Fetched %d raw rows for %d cities",
            len(raw), raw["city"].nunique(),
        )
        return raw

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _payload_to_frame(
        self,
        payload: Dict[str, Any],
        city_name: str,
        latitude: float,
        longitude: float,
        is_forecast: bool,
    ) -> pd.DataFrame:
        """Transform an Open-Meteo JSON payload into a tidy DataFrame."""
        hourly = payload.get("hourly") or {}
        times = pd.to_datetime(hourly.get("time", []), utc=True)
        if len(times) == 0:
            raise OpenMeteoAPIError(
                f"Empty 'hourly.time' block for city={city_name}"
            )

        frame = pd.DataFrame(index=range(len(times)))
        for column in HOURLY_VARIABLES:
            values = hourly.get(column)
            frame[column] = (
                pd.to_numeric(values, errors="coerce")
                if values is not None
                else np.nan
            )

        frame.insert(0, "observation_time", times)
        frame.insert(0, "city", city_name)
        frame["latitude"] = float(latitude)
        frame["longitude"] = float(longitude)
        frame["is_forecast"] = bool(is_forecast)

        # Rename Open-Meteo variables to warehouse column names.
        return frame.rename(columns=_RENAMES)


_RENAMES = {
    "temperature_2m": "temperature_c",
    "relative_humidity_2m": "relative_humidity",
    "dew_point_2m": "dew_point_c",
    "precipitation": "precipitation_mm",
    "pressure_msl": "pressure_msl",
    "cloud_cover": "cloud_cover",
    "wind_speed_10m": "wind_speed_kmh",
    "wind_direction_10m": "wind_direction",
    "weather_code": "weather_code",
}


def run(config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """Entrypoint used by ``run_pipeline`` / the Airflow DAG.

    Fetches every city's historical + forecast data and returns a single
    raw DataFrame ready for ``db_upsert.upsert_dataframe``.
    """
    config = config if config is not None else load_config()
    return OpenMeteoClient(config).fetch_all()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    frame = run()
    print(frame.head())
    print(frame.describe())
