-- ============================================================================
-- daily_aggregates.sql
-- Per-city, per-day analytical metrics (temperature extremes, humidity,
-- comfort indices, wind and precipitation).
--
-- Idempotent: re-running this script performs an UPSERT keyed on
-- (city, observation_date). The target table is created by
-- Ingestion/db_upsert.init_schema().
-- ============================================================================

INSERT INTO analytics.daily_aggregates (
    city,
    observation_date,
    avg_temperature_c,
    min_temperature_c,
    max_temperature_c,
    avg_relative_humidity,
    avg_heat_index_c,
    avg_wind_chill_c,
    avg_wind_speed_kmh,
    total_precipitation_mm,
    num_observations
)
SELECT
    city,
    observation_time::date                                                                            AS observation_date,
    ROUND(AVG(temperature_c)::numeric, 2)                                                             AS avg_temperature_c,
    ROUND(MIN(temperature_c)::numeric, 2)                                                             AS min_temperature_c,
    ROUND(MAX(temperature_c)::numeric, 2)                                                             AS max_temperature_c,
    ROUND(AVG(relative_humidity)::numeric, 2)                                                         AS avg_relative_humidity,
    ROUND(AVG(heat_index_c)::numeric, 2)                                                              AS avg_heat_index_c,
    ROUND(AVG(wind_chill_c)::numeric, 2)                                                              AS avg_wind_chill_c,
    ROUND(AVG(wind_speed_kmh)::numeric, 2)                                                            AS avg_wind_speed_kmh,
    ROUND(SUM(COALESCE(precipitation_mm, 0))::numeric, 2)                                             AS total_precipitation_mm,
    COUNT(*)::int                                                                                     AS num_observations
FROM analytics.weather_metrics
GROUP BY city, observation_time::date
ON CONFLICT (city, observation_date) DO UPDATE SET
    avg_temperature_c       = EXCLUDED.avg_temperature_c,
    min_temperature_c       = EXCLUDED.min_temperature_c,
    max_temperature_c       = EXCLUDED.max_temperature_c,
    avg_relative_humidity   = EXCLUDED.avg_relative_humidity,
    avg_heat_index_c        = EXCLUDED.avg_heat_index_c,
    avg_wind_chill_c        = EXCLUDED.avg_wind_chill_c,
    avg_wind_speed_kmh      = EXCLUDED.avg_wind_speed_kmh,
    total_precipitation_mm  = EXCLUDED.total_precipitation_mm,
    num_observations        = EXCLUDED.num_observations;
