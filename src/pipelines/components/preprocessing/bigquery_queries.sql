-- src/pipelines/components/preprocessing/bigquery_queries.sql

-- Crée le schéma si nécessaire (optionnel, peut être géré en amont)
-- CREATE SCHEMA IF NOT EXISTS `{PROJECT_ID}.{BQ_DATASET}` OPTIONS(location="{{ location }}");

-- Crée ou remplace la table préparée pour le forecasting
CREATE OR REPLACE TABLE `{PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE_PREPARED}` AS
WITH
  raw_trips AS (
    -- Sélectionne les données brutes nécessaires dans la plage temporelle définie
    SELECT trip_start_timestamp, pickup_community_area
    FROM `{SOURCE_TABLE}` -- Utilise la table source spécifiée
    WHERE
      pickup_community_area IS NOT NULL
      -- Filtrage temporel basé sur les timestamps calculés par le script Python
      AND trip_start_timestamp >= TIMESTAMP("{start_timestamp_str}")
      AND trip_start_timestamp <= TIMESTAMP("{end_timestamp_str}") -- Inclut l'heure de fin
  ),
  hours AS (
    -- Génère toutes les heures possibles dans la plage filtrée
    SELECT timestamp_hour
    FROM UNNEST(GENERATE_TIMESTAMP_ARRAY(
            TIMESTAMP("{start_timestamp_str}"),
            TIMESTAMP("{end_timestamp_str}"),
            INTERVAL 1 HOUR)) AS timestamp_hour
  ),
  areas AS (
    -- Obtient les zones uniques à partir des données DÉJÀ filtrées
    SELECT DISTINCT pickup_community_area
    FROM raw_trips
  ),
  all_combinations AS (
    -- Crée la grille complète : toutes les heures x toutes les zones concernées
    SELECT
      h.timestamp_hour,
      a.pickup_community_area
    FROM hours h
    CROSS JOIN areas a
  ),
  aggregated AS (
    -- Agrège le nombre de courses par heure et zone DANS la période filtrée
    SELECT
      TIMESTAMP_TRUNC(trip_start_timestamp, HOUR) AS timestamp_hour,
      pickup_community_area,
      COUNT(*) AS trip_count
    FROM raw_trips -- Utilise les données déjà filtrées
    GROUP BY 1, 2
  ),
  filled AS (
    -- Joint la grille complète avec les agrégats et remplit les vides (0 course)
    SELECT
      ac.timestamp_hour,
      ac.pickup_community_area,
      IFNULL(agg.trip_count, 0) AS trip_count
    FROM all_combinations ac
    LEFT JOIN aggregated agg
      ON ac.timestamp_hour = agg.timestamp_hour
     AND ac.pickup_community_area = agg.pickup_community_area
  )
-- Sélection finale avec toutes les features temporelles requises par Vertex AI Forecast
SELECT
  timestamp_hour,
  pickup_community_area,
  trip_count,
  EXTRACT(HOUR FROM timestamp_hour) AS hour,
  -- Note: BQ DAYOFWEEK: Dimanche=1, ..., Samedi=7. Vertex AI gère différents formats.
  -- Si besoin d'aligner sur Python (Lundi=0), utiliser: MOD(EXTRACT(DAYOFWEEK FROM timestamp_hour) + 5, 7)
  EXTRACT(DAYOFWEEK FROM timestamp_hour) AS day_of_week,
  EXTRACT(MONTH FROM timestamp_hour) AS month,
  EXTRACT(YEAR FROM timestamp_hour) AS year,
  EXTRACT(DAYOFYEAR FROM timestamp_hour) AS day_of_year,
  -- Utiliser ISOWEEK pour la semaine de l'année (plus standard)
  CAST(EXTRACT(ISOWEEK FROM timestamp_hour) AS INT64) as week_of_year,
  -- is_weekend: 1 si Samedi (7) ou Dimanche (1), 0 sinon
  IF(EXTRACT(DAYOFWEEK FROM timestamp_hour) IN (1, 7), 1, 0) AS is_weekend,
  -- Features cycliques pour l'heure
  SIN(2 * ACOS(-1) * EXTRACT(HOUR FROM timestamp_hour) / 24) AS hour_sin,
  COS(2 * ACOS(-1) * EXTRACT(HOUR FROM timestamp_hour) / 24) AS hour_cos
FROM filled
ORDER BY timestamp_hour, pickup_community_area;