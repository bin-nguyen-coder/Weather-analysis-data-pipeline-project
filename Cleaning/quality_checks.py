"""Quality gates for the pipeline.

Checks enforced here:

* **Temperature bounds**      : must stay within ``[-60, 60]`` °C
* **Relative humidity bounds** : must stay within ``[0, 100]`` %
* **Schema completeness**      : all expected columns must exist on
  ``cleaned.weather_observations`` and the table must be non-empty

A ``QualityCheckError`` is raised (failing the pipeline run) whenever a hard
check fails. Boundary violations are reported in the returned dictionary.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Ingestion.db_upsert import get_connection, init_schema  # noqa: E402
from Workflow.config import load_config  # noqa: E402

logger = logging.getLogger(__name__)

# Physically plausible bounds.
TEMPERATURE_RANGE = (-60.0, 60.0)
HUMIDITY_RANGE = (0.0, 100.0)

# Every column the cleaned layer is expected to expose (schema contract).
EXPECTED_SCHEMA = [
    "city", "latitude", "longitude", "observation_time",
    "temperature_c", "relative_humidity", "dew_point_c",
    "pressure_msl", "wind_speed_kmh", "wind_direction",
    "weather_code", "precipitation_mm", "cloud_cover",
]


class QualityCheckError(RuntimeError):
    """Raised when a hard quality gate fails."""


def check_temperature(series: pd.Series) -> Dict[str, Any]:
    """Count temperature observations outside the plausible range."""
    if series.empty:
        return {"check": "temperature_bounds", "violations": 0, "passed": True}
    violations = int(
        ((series < TEMPERATURE_RANGE[0]) | (series > TEMPERATURE_RANGE[1])).sum()
    )
    return {
        "check": "temperature_bounds",
        "min": float(series.min()),
        "max": float(series.max()),
        "violations": violations,
        "passed": violations == 0,
    }


def check_humidity(series: pd.Series) -> Dict[str, Any]:
    """Count relative-humidity values outside ``[0, 100]``."""
    if series.empty:
        return {"check": "humidity_bounds", "violations": 0, "passed": True}
    violations = int(
        ((series < HUMIDITY_RANGE[0]) | (series > HUMIDITY_RANGE[1])).sum()
    )
    return {
        "check": "humidity_bounds",
        "min": float(series.min()),
        "max": float(series.max()),
        "violations": violations,
        "passed": violations == 0,
    }


def check_schema_completeness(
    frame: pd.DataFrame,
    table_name: str = "cleaned.weather_observations",
) -> Dict[str, Any]:
    """Verify every expected column exists on ``frame``."""
    missing = [col for col in EXPECTED_SCHEMA if col not in frame.columns]
    return {
        "check": "schema_completeness",
        "table": table_name,
        "missing_columns": missing,
        "passed": not missing and not frame.empty,
    }


def run_quality_checks(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run every quality gate against the cleaned layer.

    Returns a structured report. Raises ``QualityCheckError`` on any hard
    failure (schema incompleteness, empty table, or boundary violations).
    """
    config = config if config is not None else load_config()
    conn = get_connection(config)
    try:
        init_schema(conn)
        frame = pd.read_sql_query(
            "SELECT * FROM cleaned.weather_observations", conn
        )
        if frame.empty:
            raise QualityCheckError(
                "cleaned.weather_observations is empty - run the cleaning "
                "stage before quality checks."
            )

        schema = check_schema_completeness(frame)
        temperature = check_temperature(frame["temperature_c"])
        humidity = check_humidity(frame["relative_humidity"])

        report: Dict[str, Any] = {
            "rows_checked": int(len(frame)),
            "cities": sorted(frame["city"].unique().tolist()),
            "checks": [schema, temperature, humidity],
        }
        for check in report["checks"]:
            logger.info("Quality check '%s': passed=%s", check["check"], check["passed"])

        failed = [c for c in report["checks"] if not c["passed"]]
        if failed:
            details = {
                c["check"]: {k: v for k, v in c.items() if k != "check"}
                for c in failed
            }
            raise QualityCheckError(f"Quality gates failed: {details}")

        return report
    finally:
        conn.close()


def run(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Pipeline entrypoint alias for ``run_quality_checks``."""
    return run_quality_checks(config)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    report = run()
    print(report)
