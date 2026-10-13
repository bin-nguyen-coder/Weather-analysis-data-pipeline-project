"""EDA reports: analytical summary charts + summary statistics.

Reads the analytics layer (``analytics.weather_metrics``) and renders a set
of PNG charts plus a JSON summary into ``EDA/reports/``:

* ``temperature_trends.png``        : per-city temperature time series
* ``humidity_distribution.png``     : per-city humidity histograms
* ``correlation_heatmap.png``       : numeric feature correlation matrix
* ``temp_vs_humidity.png``          : bivariate scatter (coloured by city)
* ``daily_average_temperature.png`` : per-city daily mean temperature
* ``summary_statistics.json``       : per-city descriptive statistics

All plots use the non-interactive ``Agg`` backend so the report generation
works headless (Airflow scheduler / CI).
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")  # headless backend - imports must come afterwards.

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Ingestion.db_upsert import get_connection, init_schema  # noqa: E402
from Workflow.config import load_config  # noqa: E402

logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid")


def _plot_temperature_trends(df: pd.DataFrame, output_dir: Path) -> Path:
    pivot = df.pivot_table(
        index="observation_time", columns="city", values="temperature_c"
    )
    pivot = pivot.resample("D").mean()
    figure, axis = plt.subplots(figsize=(14, 6))
    for city in pivot.columns:
        axis.plot(pivot.index, pivot[city], label=str(city), linewidth=1.5)
    axis.set_title("Daily average temperature per city (°C)")
    axis.set_xlabel("Date")
    axis.set_ylabel("Temperature (°C)")
    axis.legend(loc="best")
    figure.autofmt_xdate()
    path = output_dir / "temperature_trends.png"
    figure.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(figure)
    return path


def _plot_humidity_distribution(df: pd.DataFrame, output_dir: Path) -> Path:
    figure, axis = plt.subplots(figsize=(10, 6))
    for city, group in df.groupby("city", sort=False):
        axis.hist(
            group["relative_humidity"],
            bins=30,
            alpha=0.5,
            label=str(city),
        )
    axis.set_title("Relative humidity distribution per city (%)")
    axis.set_xlabel("Relative humidity (%)")
    axis.set_ylabel("Observations")
    axis.legend(loc="best")
    path = output_dir / "humidity_distribution.png"
    figure.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(figure)
    return path


def _plot_correlation_heatmap(df: pd.DataFrame, output_dir: Path) -> Path:
    numeric = df.select_dtypes(include=[np.number])
    correlation = numeric.corr()
    figure, axis = plt.subplots(figsize=(12, 9))
    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        ax=axis,
    )
    axis.set_title("Feature correlation heatmap")
    path = output_dir / "correlation_heatmap.png"
    figure.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(figure)
    return path


def _plot_temp_vs_humidity(df: pd.DataFrame, output_dir: Path) -> Path:
    figure, axis = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        data=df.sample(min(len(df), 20_000), random_state=42),
        x="temperature_c",
        y="relative_humidity",
        hue="city",
        alpha=0.4,
        s=12,
        ax=axis,
    )
    axis.set_title("Temperature vs relative humidity")
    axis.set_xlabel("Temperature (°C)")
    axis.set_ylabel("Relative humidity (%)")
    path = output_dir / "temp_vs_humidity.png"
    figure.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(figure)
    return path


def _plot_daily_average(df: pd.DataFrame, output_dir: Path) -> Path:
    daily = (
        df.set_index("observation_time")
        .groupby("city", sort=False)["temperature_c"]
        .resample("D")
        .mean()
        .reset_index()
    )
    figure, axis = plt.subplots(figsize=(14, 6))
    sns.lineplot(
        data=daily, x="observation_time", y="temperature_c",
        hue="city", marker="o", markersize=4, ax=axis,
    )
    axis.set_title("Per-city daily average temperature (°C)")
    axis.set_xlabel("Date")
    axis.set_ylabel("Temperature (°C)")
    figure.autofmt_xdate()
    path = output_dir / "daily_average_temperature.png"
    figure.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(figure)
    return path


def _build_summary_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """Per-city descriptive statistics exported as JSON."""
    numeric_columns = [
        "temperature_c", "relative_humidity", "dew_point_c",
        "pressure_msl", "wind_speed_kmh", "heat_index_c", "wind_chill_c",
    ]
    numeric_columns = [c for c in numeric_columns if c in df.columns]

    summary: Dict[str, Any] = {}
    for city, group in df.groupby("city", sort=False):
        stats = {
            "observation_count": int(len(group)),
            "first_observation": str(group["observation_time"].min()),
            "last_observation": str(group["observation_time"].max()),
        }
        for column in numeric_columns:
            stats[column] = {
                "mean": round(float(group[column].mean()), 3),
                "std": round(float(group[column].std()), 3),
                "min": round(float(group[column].min()), 3),
                "max": round(float(group[column].max()), 3),
            }
        summary[str(city)] = stats
    return {"generated_at": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"), "cities": summary}


def generate_reports(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Generate all EDA charts + summary statistics from the analytics layer."""
    config = config if config is not None else load_config()
    output_dir = Path(config["paths"]["eda_reports"])
    output_dir.mkdir(parents=True, exist_ok=True)

    conn = get_connection(config)
    try:
        init_schema(conn)
        frame = pd.read_sql_query(
            "SELECT * FROM analytics.weather_metrics ORDER BY city, observation_time",
            conn,
        )
        if frame.empty:
            raise RuntimeError(
                "analytics.weather_metrics is empty - run the transform "
                "stage before generating reports."
            )
        frame["observation_time"] = pd.to_datetime(frame["observation_time"], utc=True)

        charts: List[Path] = [
            _plot_temperature_trends(frame, output_dir),
            _plot_humidity_distribution(frame, output_dir),
            _plot_correlation_heatmap(frame, output_dir),
            _plot_temp_vs_humidity(frame, output_dir),
            _plot_daily_average(frame, output_dir),
        ]

        summary = _build_summary_statistics(frame)
        summary_path = output_dir / "summary_statistics.json"
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)

        logger.info("Generated %d charts + summary in %s", len(charts), output_dir)
        return {
            "report_dir": str(output_dir),
            "charts": [str(path) for path in charts],
            "summary_path": str(summary_path),
        }
    finally:
        conn.close()


def run(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Pipeline entrypoint alias for ``generate_reports``."""
    return generate_reports(config)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    report = run()
    print(report)
