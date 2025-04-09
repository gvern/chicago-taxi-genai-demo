from kfp.v2.dsl import component
from google.cloud import bigquery

@component(
    base_image="python:3.10",
    packages_to_install=["google-cloud-bigquery"],
)
def create_demand_table(project_id: str, dataset_id: str, table_id: str) -> str:
    """
    Crée une table BigQuery agrégée avec le nombre de trajets par heure et par zone.
    """
    client = bigquery.Client(project=project_id)

    query = f"""
    CREATE SCHEMA IF NOT EXISTS `{project_id}.{dataset_id}`
    OPTIONS(location="US");

    CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{table_id}` AS
    SELECT
      TIMESTAMP_TRUNC(pickup_datetime, HOUR) AS timestamp_hour,
      pickup_community_area,
      COUNT(trip_id) AS trip_count,
      EXTRACT(HOUR FROM pickup_datetime) AS hour,
      EXTRACT(DAYOFWEEK FROM pickup_datetime) AS day_of_week,
      EXTRACT(MONTH FROM pickup_datetime) AS month,
      EXTRACT(DAYOFYEAR FROM pickup_datetime) AS day_of_year,
      IF(EXTRACT(DAYOFWEEK FROM pickup_datetime) IN (1,7), 1, 0) AS is_weekend,
      AVG(temperature) AS temperature,
      AVG(precipitation) AS precipitation,
      AVG(wind_speed) AS wind_speed
    FROM
      `bigquery-public-data.chicago_taxi_trips.taxi_trips`
    WHERE
      pickup_datetime BETWEEN TIMESTAMP("2013-01-01") AND TIMESTAMP("2023-11-22")
      AND pickup_community_area IS NOT NULL
    GROUP BY
      timestamp_hour, pickup_community_area
    HAVING 
      COUNT(trip_id) > 0
    ;
    """

    job = client.query(query)
    job.result()  # wait for completion

    print(f"✅ Table créée : {project_id}.{dataset_id}.{table_id}")
    return f"{project_id}.{dataset_id}.{table_id}"
