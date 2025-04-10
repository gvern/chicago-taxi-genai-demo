from typing import Optional
from kfp.dsl import component

@component(
    packages_to_install=["google-cloud-bigquery"],
    base_image="python:3.9"
)
def run_bq_forecasting_query(
    project_id: str,
    location: str = "europe-west1",
    dataset_id: str = "chicago_taxis",
    destination_table: str = "demand_by_hour",
    sql_output_table: Optional[str] = None,
) -> str:
    """
    Exécute la requête BigQuery pour générer la table d'agrégation forecasting (timestamp_hour × pickup_community_area).
    Retourne l'URI de la table créée.
    """
    from google.cloud import bigquery  # ← Cette ligne est indispensable

    client = bigquery.Client(project=project_id, location=location)
    
    table_uri = f"{project_id}.{dataset_id}.{destination_table}"
    
    query = f"""
    -- Génère la table forecasting avec toutes les heures et zones
    CREATE SCHEMA IF NOT EXISTS `{project_id}.{dataset_id}` OPTIONS(location="{location}");
    
    CREATE OR REPLACE TABLE `{table_uri}` AS
    WITH
      hours AS (
        SELECT ts AS timestamp_hour
        FROM UNNEST(GENERATE_TIMESTAMP_ARRAY(
          TIMESTAMP("2013-01-01 00:00:00"),
          TIMESTAMP("2023-11-22 09:00:00"),
          INTERVAL 1 HOUR)) AS ts
      ),
      areas AS (
        SELECT DISTINCT pickup_community_area
        FROM `{project_id}.{dataset_id}.taxi_trips`
        WHERE pickup_community_area IS NOT NULL
      ),
      hourly_grid AS (
        SELECT timestamp_hour, pickup_community_area
        FROM hours CROSS JOIN areas
      ),
      trip_counts AS (
        SELECT
          TIMESTAMP_TRUNC(trip_start_timestamp, HOUR) AS timestamp_hour,
          pickup_community_area,
          COUNT(unique_key) AS trip_count,
          AVG(NULLIF(weather.temperature,0)) AS temperature,
          AVG(NULLIF(weather.precipitation,0)) AS precipitation,
          AVG(NULLIF(weather.wind_speed,0)) AS wind_speed
        FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips` AS trips
        LEFT JOIN `bigquery-public-data.chicago_taxi_trips.taxi_weather` AS weather
          ON DATE(trips.trip_start_timestamp) = weather.date
        WHERE
          trip_start_timestamp BETWEEN TIMESTAMP("2013-01-01") AND TIMESTAMP("2023-11-22")
          AND pickup_community_area IS NOT NULL
        GROUP BY timestamp_hour, pickup_community_area
      )
    SELECT
      grid.timestamp_hour,
      grid.pickup_community_area,
      IFNULL(tc.trip_count, 0) AS trip_count,
      EXTRACT(HOUR FROM grid.timestamp_hour) AS hour,
      EXTRACT(DAYOFWEEK FROM grid.timestamp_hour) AS day_of_week,
      EXTRACT(MONTH FROM grid.timestamp_hour) AS month,
      EXTRACT(DAYOFYEAR FROM grid.timestamp_hour) AS day_of_year,
      IF(EXTRACT(DAYOFWEEK FROM grid.timestamp_hour) IN (1,7), 1, 0) AS is_weekend,
      tc.temperature,
      tc.precipitation,
      tc.wind_speed
    FROM hourly_grid AS grid
    LEFT JOIN trip_counts AS tc
      ON grid.timestamp_hour = tc.timestamp_hour
      AND grid.pickup_community_area = tc.pickup_community_area
    ORDER BY timestamp_hour ASC
    """

    job = client.query(query)
    job.result()

    return table_uri
