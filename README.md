# Weather Analysis Data Pipeline

An end-to-end, automated weather **ingestion → cleaning → transformation →
aggregation → ML forecasting → EDA** pipeline built around the free
[Open-Meteo](https://open-meteo.com/) APIs, PostgreSQL and Apache Airflow.

## Architecture

```
Open-Meteo REST API
      │
      ▼
┌───────────────────────┐        ┌───────────────────────────┐
│ Ingestion/client.py   │  ──►   │ PostgreSQL warehouse      │
│ (rate-limited fetch)  │        │ raw.weather_observations  │
└───────────────────────┘        └───────────────────────────┘
                                           │
                                           ▼
                                Cleaning / quality gates
                                           │
                                           ▼
                          Transform (heat index, wind chill,
                             rolling metrics, scaling)
                                           │
                    ┌──────────────────────┼───────────────────────┐
                    ▼                      ▼                       ▼
           Aggregation (SQL)        Feature engineering      EDA reports
           daily + city stats       (cyclical encodings,     (charts + stats)
                                    168h sliding windows)
                                            │
                                            ▼
                                     LSTM training
                                            │
                                            ▼
                                     24h forecast
                                    Forecast/forecast.json
```

## Directory layout

| Folder | Purpose |
|---|---|
| `Ingestion/` | Open-Meteo client with retries/rate-limit + PostgreSQL UPSERT |
| `Cleaning/` | Dedupe, impute, UTC normalization, quality gates |
| `Transform/` | Heat Index / Wind Chill / rolling metrics + scaled feature vectors |
| `Aggregation/` | SQL queries for daily aggregates & city statistics |
| `Train/` | LSTM feature engineering + Keras training |
| `Forecast/` | 24-hour forecast export (`forecast.json`) |
| `EDA/` | Analytical charts + summary statistics |
| `Orchestration/` | Apache Airflow DAG |
| `Automation Pipeline/` | Local end-to-end runner (no Airflow) |
| `Workflow/` | Shared pipeline configuration |
| `Model/` | Trained model (`lstm_weather_model.h5`) + scaler (`scaler.pkl`) |
| `.github/workflows/` | Daily GitHub Actions trigger |

## Quick start

### 1. Infrastructure (Docker)

```bash
docker compose up -d          # PostgreSQL 16 + Airflow (LocalExecutor)
docker compose ps             # confirm postgres/airflow-scheduler/webserver are up
```

Airflow UI: <http://localhost:8080> (admin / admin).
PostgreSQL: `localhost:5432` (db `weather`, user `weather_user`,
password `weather_password`).

### 2. Local run (single machine, no Airflow)

```bash
python -m pip install -r requirements.txt
python "Automation Pipeline/run_pipeline.py"          # all stages
python "Automation Pipeline/run_pipeline.py" --only ingest
python "Automation Pipeline/run_pipeline.py" --skip train eda
```

A run report is written to `data/pipeline_run.json`.

### 3. Airflow

Mount the repo so the DAG can import the pipeline modules (done in
`docker-compose.yml`), then trigger the `weather_pipeline` DAG. Runs daily.

### 4. GitHub Actions

The workflow `.github/workflows/pipeline_cron.yml` runs the full pipeline
every day at 02:00 UTC (and can be triggered manually via `workflow_dispatch`).

## Configuration

All settings live in `Workflow/workflow_config.json`:

* **cities** – list of `{name, latitude, longitude}` monitored by the pipeline;
* **open_meteo** – API URLs, historical lookback (`history_days`), forecast
  window, rate-limit interval;
* **pipeline** – LSTM lookback (`168` h), forecast horizon (`24` h), training
  hyper-parameters;
* **paths** – artifact locations (relative paths resolve against the repo root).

Database settings can be overridden with the standard `PGHOST` / `PGPORT` /
`PGDATABASE` / `PGUSER` / `PGPASSWORD` environment variables.

## Outputs

| Artifact | Location |
|---|---|
| 24-hour temperature forecast | `Forecast/forecast.json` |
| Trained LSTM | `Model/lstm_weather_model.h5` |
| Feature/target scaler | `Model/scaler.pkl` |
| Scaled feature matrix | `data/scaled_features.parquet` |
| Windowed ML dataset | `data/windows.npz` |
| EDA charts + summary | `EDA/reports/*.png`, `summary_statistics.json` |
| Pipeline run report | `data/pipeline_run.json` |
