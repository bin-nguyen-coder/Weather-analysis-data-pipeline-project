"""Airflow DAG: ``weather_pipeline``.

Runs the full weather data pipeline every day, sequentially:

    Ingestion -> Cleaning -> QualityChecks -> Transform (metrics)
    -> FeaturePrep (scaling) -> Aggregation (SQL)
    -> FeatureEngineering (windows) -> ML Train -> Predict
    -> EDA reports

The DAG uses ``PythonOperator``; module imports are kept **inside** the task
callables so Airflow's DAG parser does not pay the TensorFlow import cost on
every parse. All pipeline modules import cleanly via the repository root,
which the scheduler has mounted at ``/opt/airflow/project`` in docker-compose.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Make the pipeline modules importable.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from airflow import DAG  # noqa: E402
from airflow.operators.python import PythonOperator  # noqa: E402

from Workflow.config import load_config  # noqa: E402

logger = logging.getLogger(__name__)

DAG_ID = "weather_pipeline"
DEFAULT_ARGS = {
    "owner": "weather-pipeline",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def _load_config() -> dict:
    return load_config()


def _task_ingest() -> dict:
    """Ingestion: Open-Meteo -> raw.weather_observations (UPSERT)."""
    from Ingestion import db_upsert

    result = db_upsert.run(_load_config())
    logger.info("Ingestion result: %s", result)
    return {"ingested_rows": result}


def _task_clean() -> dict:
    """Cleaning: raw -> cleaned.weather_observations."""
    from Cleaning import clean_observations

    result = clean_observations.run(_load_config())
    logger.info("Cleaning result: %s", result)
    return result


def _task_quality_checks() -> dict:
    """Quality gates on the cleaned layer."""
    from Cleaning import quality_checks

    result = quality_checks.run(_load_config())
    logger.info("Quality check result: %s", result)
    return result


def _task_transform_metrics() -> dict:
    """Transform: heat index / wind chill / rolling metrics."""
    from Transform import calculate_heat_index

    result = calculate_heat_index.run(_load_config())
    logger.info("Transform result: %s", result)
    return result


def _task_feature_prep() -> dict:
    """Transform: scaled feature vectors."""
    from Transform import transform_features

    result = transform_features.run(_load_config())
    logger.info("Feature preparation result: %s", result)
    return result


def _task_aggregation() -> dict:
    """Aggregation: daily_aggregates.sql + city_stats.sql."""
    from Aggregation import run_aggregations

    result = run_aggregations.run(_load_config())
    logger.info("Aggregation result: %s", result)
    return result


def _task_feature_engineering() -> dict:
    """ML: cyclical encodings + 168-hour sliding windows."""
    from Train import feature_engineering

    result = feature_engineering.run(_load_config())
    logger.info("Feature engineering result: %s", result)
    return result


def _task_train_lstm() -> dict:
    """ML: LSTM training -> Model/lstm_weather_model.h5 + scaler.pkl."""
    from Train import train_lstm

    result = train_lstm.run(_load_config())
    logger.info("Training result: %s", result)
    return result


def _task_predict() -> dict:
    """Forecast: 24-hour temperatures -> Forecast/forecast.json."""
    from Forecast import predict

    result = predict.run(_load_config())
    logger.info("Prediction result: %s", result)
    return result


def _task_eda() -> dict:
    """EDA: analytical charts + summary statistics."""
    from EDA import generate_reports

    result = generate_reports.run(_load_config())
    logger.info("EDA result: %s", result)
    return result


with DAG(
    dag_id=DAG_ID,
    default_args=DEFAULT_ARGS,
    description=(
        "Weather data pipeline: ingest Open-Meteo -> clean -> transform -> "
        "aggregate -> train LSTM -> forecast -> EDA reports"
    ),
    schedule="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["weather", "lstm", "pipeline"],
) as dag:

    ingest = PythonOperator(task_id="ingest", python_callable=_task_ingest)
    clean = PythonOperator(task_id="clean", python_callable=_task_clean)
    quality_checks = PythonOperator(
        task_id="quality_checks", python_callable=_task_quality_checks
    )
    transform_metrics = PythonOperator(
        task_id="transform_metrics", python_callable=_task_transform_metrics
    )
    feature_prep = PythonOperator(
        task_id="feature_prep", python_callable=_task_feature_prep
    )
    aggregation = PythonOperator(
        task_id="aggregation", python_callable=_task_aggregation
    )
    feature_engineering = PythonOperator(
        task_id="feature_engineering", python_callable=_task_feature_engineering
    )
    train_lstm = PythonOperator(
        task_id="train_lstm", python_callable=_task_train_lstm
    )
    predict = PythonOperator(task_id="predict", python_callable=_task_predict)
    eda = PythonOperator(task_id="eda", python_callable=_task_eda)

    # Linear execution chain.
    (
        ingest
        >> clean
        >> quality_checks
        >> transform_metrics
        >> feature_prep
        >> aggregation
        >> feature_engineering
        >> train_lstm
        >> predict
        >> eda
    )
