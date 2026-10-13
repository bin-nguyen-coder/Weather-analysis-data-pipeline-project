-- ============================================================================
-- city_stats.sql
-- Cross-city summary statistics computed over the full analytics metric set.
--
-- Idempotent: re-running this script performs an UPSERT keyed on the primary
-- key (city). The target table is created by
-- Ingestion/db_upsert.init_schema().
-- ============================================================================

INSERT INTO analytics.city_stats (
    city,
    observation_count,
    first_observation,
    last_observation,
    avg_temperature_c,
    min_temperature_c,
    max_temperature_c,
    stddev_temperature_c,
    avg_relative_humidity,
    avg_heat_index_c,
    avg_wind_speed_kmh,
    total_precipitation_mm
)
SELECT
    city,
    COUNT(*)::int                                  AS observation_count,
    MIN(observation_time)                          AS first_observation,
    MAX(observation_time)                          AS last_observation,
    ROUND(AVG(temperature_c)::numeric, 2)          AS avg_temperature_c,
    ROUND(MIN(temperature_c)::numeric, 2)          AS min_temperature_c,
    ROUND(MAX(temperature_c)::numeric, 2)          AS max_temperature_c,
    ROUND(STDDEV_SAMP(temperature_c)::numeric, 2)  AS stddev_temperature_c,
    ROUND(AVG(relative_humidity)::numeric, 2)      AS avg_relative_humidity,
    ROUND(AVG(heat_index_c)::numeric, 2)           AS avg_heat_index_c,
    ROUND(AVG(wind_speed_kmh)::numeric, 2)         AS avg_wind_speed_kmh,
    ROUND(SUM(COALESCE(precipitation_mm, 0))::numeric, 2) AS total_precipitation_mm
FROM analytics.weather_metrics
GROUP BY city
ON CONFLICT (city) DO UPDATE SET
    observation_count       = EXCLUDED.observation_count,
    first_observation       = EXCLUDED.first_observation,
    last_observation        = EXCLUDED.last_observation,
    avg_temperature_c       = EXCLUDED.avg_temperature_c,
    min_temperature_c       = EXCLUDED.min_temperature_c,
    max_temperature_c       = EXCLUDED.max_temperature_c,
    stddev_temperature_c    = EXCLUDED.stddev_temperature_c,
    avg_relative_humidity   = EXCLUDED.avg_relative_humidity,
    avg_heat_index_c        = EXCLUDED.avg_heat_index_c,
    avg_wind_speed_kmh      = EXCLUDED.avg_wind_speed_kmh,
    total_precipitation_mm  = EXCLUDED.total_precipitation_mm;
