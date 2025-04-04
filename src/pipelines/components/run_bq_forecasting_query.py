from typing import Optional
from kfp.dsl import component
from google.cloud import bigquery

@component(
    packages_to_install=["google-cloud-bigquery"],
    base_image="python:3.9"
)
def run_bq_forecasting_query(
    project_id: str,
    location: str = "US",
    dataset_id: str = "chicago_taxis",
    destination_table: str = "demand_by_hour",
    sql_output_table: Optional[str] = None,
) -> str:
    """
    Exécute la requête BigQuery pour générer la table d'agrégation forecasting (timestamp_hour × pickup_community_area).
    Retourne l'URI de la table créée.
    """
    client = bigquery.Client(project=project_id, location=location)
    
    table_uri = f"{project_id}.{dataset_id}.{destination_table}"
    
    query = f"""
    -- Génère la table forecasting avec toutes les heures et zones
    CREATE SCHEMA IF NOT EXISTS `{project_id}.{dataset_id}` OPTIONS(location="{location}");

    CREATE OR REPLACE TABLE `{table_uri}` AS
    WITH
      hours AS (
        SELECT
          MIN(TIMESTAMP_TRUNC(trip_start_timestamp, HOUR)) AS min_hour,
          MAX(TIMESTAMP_TRUNC(trip_start_timestamp, HOUR)) AS max_hour
        FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`
      ),
      all_hours AS (
        SELECT timestamp_hour
        FROM hours,
        UNNEST(GENERATE_TIMESTAMP_ARRAY(min_hour, max_hour, INTERVAL 1 HOUR)) AS timestamp_hour
      ),
      areas AS (
        SELECT DISTINCT pickup_community_area
        FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`
        WHERE pickup_community_area IS NOT NULL
      ),
      all_combinations AS (
        SELECT
          h.timestamp_hour,
          a.pickup_community_area
        FROM all_hours h
        CROSS JOIN areas a
      ),
      aggregated AS (
        SELECT
          TIMESTAMP_TRUNC(trip_start_timestamp, HOUR) AS timestamp_hour,
          pickup_community_area,
          COUNT(*) AS trip_count
        FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`
        WHERE pickup_community_area IS NOT NULL
        GROUP BY 1, 2
      ),
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
      IF(FORMAT_DATE('%m-%d', DATE(timestamp_hour)) IN ('01-01','07-04','12-25'), 1, 0) AS is_holiday
    FROM filled
    ORDER BY timestamp_hour, pickup_community_area;
    """

    job = client.query(query)
    job.result()

    return table_uri
