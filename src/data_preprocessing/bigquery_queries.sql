-- Create schema if not exists
CREATE SCHEMA IF NOT EXISTS `avisia-certification-ml-yde.chicago_taxis`
OPTIONS(location="US");

-- Main query: hourly demand per pickup_community_area, with all hours × zones filled
CREATE OR REPLACE TABLE `avisia-certification-ml-yde.chicago_taxis.demand_by_hour` AS
WITH

-- 1. Déterminer les plages horaires complètes (min à max)
hours AS (
  SELECT
    MIN(TIMESTAMP_TRUNC(trip_start_timestamp, HOUR)) AS min_hour,
    MAX(TIMESTAMP_TRUNC(trip_start_timestamp, HOUR)) AS max_hour
  FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`
),

-- 2. Générer toutes les heures entre min et max
all_hours AS (
  SELECT timestamp_hour
  FROM hours,
  UNNEST(GENERATE_TIMESTAMP_ARRAY(min_hour, max_hour, INTERVAL 1 HOUR)) AS timestamp_hour
),

-- 3. Extraire toutes les zones (pickup_community_area)
areas AS (
  SELECT DISTINCT pickup_community_area
  FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`
  WHERE pickup_community_area IS NOT NULL
),

-- 4. Générer le produit cartésien (heure x zone)
all_combinations AS (
  SELECT
    h.timestamp_hour,
    a.pickup_community_area
  FROM all_hours h
  CROSS JOIN areas a
),

-- 5. Compter les trajets agrégés par heure x zone
aggregated AS (
  SELECT
    TIMESTAMP_TRUNC(trip_start_timestamp, HOUR) AS timestamp_hour,
    pickup_community_area,
    COUNT(*) AS trip_count
  FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`
  WHERE pickup_community_area IS NOT NULL
  GROUP BY 1, 2
),

-- 6. Joindre toutes les combinaisons avec les données agrégées
filled AS (
  SELECT
    ac.timestamp_hour,
    ac.pickup_community_area,
    IFNULL(agg.trip_count, 0) AS trip_count
  FROM all_combinations ac
  LEFT JOIN aggregated agg
    ON ac.timestamp_hour = agg.timestamp_hour
   AND ac.pickup_community_area = agg.pickup_community_area
)

-- 7. Ajouter les features temporelles
SELECT
  timestamp_hour,
  pickup_community_area,
  trip_count,
  EXTRACT(HOUR FROM timestamp_hour) AS hour,
  EXTRACT(DAYOFWEEK FROM timestamp_hour) AS day_of_week,
  EXTRACT(MONTH FROM timestamp_hour) AS month,
  EXTRACT(YEAR FROM timestamp_hour) AS year,
  EXTRACT(DAYOFYEAR FROM timestamp_hour) AS day_of_year,
  IF(EXTRACT(DAYOFWEEK FROM timestamp_hour) IN (1, 7), 1, 0) AS is_weekend,
  -- Simple US holiday heuristic (à remplacer par table dédiée si besoin)
  IF(FORMAT_DATE('%m-%d', DATE(timestamp_hour)) IN ('01-01','07-04','12-25'), 1, 0) AS is_holiday
FROM filled
ORDER BY timestamp_hour, pickup_community_area;
