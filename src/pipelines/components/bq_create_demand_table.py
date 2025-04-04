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
      TIMESTAMP_TRUNC(trip_start_timestamp, HOUR) AS timestamp_hour,
      pickup_community_area,
      COUNT(*) AS trip_count
    FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`
    WHERE pickup_community_area IS NOT NULL
    GROUP BY timestamp_hour, pickup_community_area
    ORDER BY timestamp_hour, pickup_community_area;
    """

    job = client.query(query)
    job.result()  # wait for completion

    print(f"✅ Table créée : {project_id}.{dataset_id}.{table_id}")
    return f"{project_id}.{dataset_id}.{table_id}"
