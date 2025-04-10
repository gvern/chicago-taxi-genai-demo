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
    start_date: str = "2021-01-01",  # Paramètre par défaut pour limiter la taille des séries temporelles
    end_date: str = "2023-11-22"     # Paramètre par défaut pour limiter la taille des séries temporelles
) -> str:
    """
    Exécute la requête BigQuery pour générer la table d'agrégation forecasting (timestamp_hour × pickup_community_area).
    Retourne l'URI de la table créée.
    
    Parameters:
        project_id: ID du projet GCP
        location: Région GCP
        dataset_id: ID du dataset BigQuery
        destination_table: Nom de la table de destination
        sql_output_table: Nom de la table SQL optionnelle
        start_date: Date de début pour limiter les séries temporelles (format: YYYY-MM-DD)
        end_date: Date de fin pour limiter les séries temporelles (format: YYYY-MM-DD)
    """
    from google.cloud import bigquery  # ← Cette ligne est indispensable
    from datetime import datetime

    client = bigquery.Client(project=project_id, location=location)
    
    table_uri = f"{project_id}.{dataset_id}.{destination_table}"
    
    # Calculer le nombre approximatif d'heures entre les dates
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    hours_diff = int((end_dt - start_dt).total_seconds() / 3600)
    
    print(f"Génération des données de {start_date} à {end_date}")
    print(f"Nombre d'heures approximatif: {hours_diff}")
    
    # Vérification de la limite de 3000 points de données
    if hours_diff > 3000:
        print(f"⚠️ ATTENTION: Le nombre d'heures ({hours_diff}) dépasse la limite de 3000 points de Vertex AI.")
        print(f"La période a été automatiquement ajustée pour respecter cette limite.")
        # Ajuster la date de début pour rester sous 3000 heures
        from datetime import timedelta
        adjusted_start = end_dt - timedelta(hours=2950)  # Laisser une marge de sécurité
        start_date = adjusted_start.strftime("%Y-%m-%d")
        print(f"Nouvelle période: {start_date} à {end_date}")
    
    query = f"""
    -- Génère la table forecasting avec toutes les heures et zones
    CREATE SCHEMA IF NOT EXISTS `{project_id}.{dataset_id}` OPTIONS(location="{location}");
    
    CREATE OR REPLACE TABLE `{table_uri}` AS
    WITH
      hours AS (
        SELECT ts AS timestamp_hour
        FROM UNNEST(GENERATE_TIMESTAMP_ARRAY(
          TIMESTAMP("{start_date} 00:00:00"),
          TIMESTAMP("{end_date} 23:00:00"),
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
          trip_start_timestamp BETWEEN TIMESTAMP("{start_date}") AND TIMESTAMP("{end_date}")
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
    
    # Obtention de statistiques sur la table créée
    count_query = f"""
    SELECT 
      COUNT(DISTINCT pickup_community_area) as num_areas,
      COUNT(DISTINCT timestamp_hour) as num_timestamps,
      MIN(timestamp_hour) as min_timestamp,
      MAX(timestamp_hour) as max_timestamp
    FROM `{table_uri}`
    """
    count_job = client.query(count_query)
    stats = next(count_job.result())
    
    print(f"✅ Table créée: {table_uri}")
    print(f"📊 Statistiques:")
    print(f"   - Nombre de zones: {stats.num_areas}")
    print(f"   - Nombre d'horodatages: {stats.num_timestamps}")
    print(f"   - Période: {stats.min_timestamp} à {stats.max_timestamp}")
    print(f"   - Points de données par série: {stats.num_timestamps} (limite Vertex AI: 3000)")

    return table_uri
