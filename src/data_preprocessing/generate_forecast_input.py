from google.cloud import bigquery
from datetime import datetime, timedelta
import holidays
import pandas as pd

# Config
PROJECT_ID = "avisia-certification-ml-yde"
DATASET = "chicago_taxis"
TABLE_INPUT = f"{PROJECT_ID}.{DATASET}.forecast_input"
TABLE_REF = f"{PROJECT_ID}.{DATASET}.demand_by_hour"

# Initialiser le client BQ
client = bigquery.Client(project=PROJECT_ID)

def get_future_timestamps(n_hours=24):
    """Génère une liste d'horodatages horaires futurs (arrondis)"""
    now = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    return [now + timedelta(hours=i) for i in range(1, n_hours + 1)]

def get_chicago_holidays(start_date, end_date):
    """Renvoie les jours fériés américains (Chicago)"""
    us_holidays = holidays.US(years=range(start_date.year, end_date.year + 1))
    return set(us_holidays.keys())

def generate_forecast_input():
    # Étape 1: Récupérer toutes les pickup_community_area existantes
    query = f"""
        SELECT DISTINCT pickup_community_area
        FROM `{TABLE_REF}`
        WHERE pickup_community_area IS NOT NULL
    """
    areas_df = client.query(query).to_dataframe()
    
    # Étape 2: Générer les combinaisons zone x future_hour
    timestamps = get_future_timestamps()
    start, end = timestamps[0], timestamps[-1]
    holidays_set = get_chicago_holidays(start, end)
    
    rows = []
    for area in areas_df["pickup_community_area"]:
        for ts in timestamps:
            hour = ts.hour
            day_of_week = ts.weekday()
            is_weekend = day_of_week >= 5
            is_holiday = ts.date() in holidays_set
            rows.append({
                "pickup_community_area": area,
                "timestamp_hour": ts,
                "hour": hour,
                "day_of_week": day_of_week,
                "is_weekend": is_weekend,
                "is_holiday": is_holiday
            })
    
    df = pd.DataFrame(rows)

    # Étape 3: Uploader dans BigQuery
    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_TRUNCATE",
        schema=[
            bigquery.SchemaField("pickup_community_area", "INTEGER"),
            bigquery.SchemaField("timestamp_hour", "TIMESTAMP"),
            bigquery.SchemaField("hour", "INTEGER"),
            bigquery.SchemaField("day_of_week", "INTEGER"),
            bigquery.SchemaField("is_weekend", "BOOLEAN"),
            bigquery.SchemaField("is_holiday", "BOOLEAN")
        ]
    )
    job = client.load_table_from_dataframe(df, TABLE_INPUT, job_config=job_config)
    job.result()
    print(f"✅ Table {TABLE_INPUT} créée avec {len(df)} lignes.")

if __name__ == "__main__":
    generate_forecast_input()
