import pandas as pd
from datetime import datetime, timedelta
from google.cloud import bigquery
import holidays

# Configuration
PROJECT_ID = "avisia-certification-ml-yde"
DATASET = "chicago_taxis"
TABLE_INPUT = f"{PROJECT_ID}.{DATASET}.forecast_input"
TABLE_REF = f"{PROJECT_ID}.{DATASET}.demand_by_hour"
HORIZON_HOURS = 24  # à ajuster si besoin

# Initialise le client BQ
client = bigquery.Client(project=PROJECT_ID)

def get_unique_zones():
    """Récupère les pickup_community_area uniques depuis la table de training."""
    query = f"""
    SELECT DISTINCT pickup_community_area
    FROM `{TABLE_REF}`
    WHERE pickup_community_area IS NOT NULL
    """
    return client.query(query).to_dataframe()

def get_future_timestamps(n_hours=HORIZON_HOURS):
    """Génère une liste d'horodatages horaires futurs (arrondis)"""
    now = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    return [now + timedelta(hours=i) for i in range(1, n_hours + 1)]

def get_chicago_holidays(start_date, end_date):
    """Renvoie les jours fériés américains (Chicago)"""
    us_holidays = holidays.US(years=range(start_date.year, end_date.year + 1))
    return set(us_holidays.keys())

def generate_forecast_input():
    # Étape 1 : Récupérer les zones
    zones_df = get_unique_zones()
    
    # Étape 2 : Générer le cartésien zones × timestamps
    timestamps = get_future_timestamps()
    start, end = timestamps[0], timestamps[-1]
    holidays_set = get_chicago_holidays(start, end)

    rows = []
    for zone in zones_df["pickup_community_area"]:
        for ts in timestamps:
            rows.append({
                "pickup_community_area": zone,
                "timestamp_hour": ts,
                "hour": ts.hour,
                "day_of_week": ts.weekday(),
                "month": ts.month,
                "is_weekend": ts.weekday() >= 5,
                "is_holiday": ts.date() in holidays_set
            })
    
    df = pd.DataFrame(rows)
    return df

def write_to_bigquery(df: pd.DataFrame):
    """Écrit le dataframe dans BigQuery avec schéma défini"""
    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_TRUNCATE",
        schema=[
            bigquery.SchemaField("pickup_community_area", "INTEGER"),
            bigquery.SchemaField("timestamp_hour", "TIMESTAMP"),
            bigquery.SchemaField("hour", "INTEGER"),
            bigquery.SchemaField("day_of_week", "INTEGER"),
            bigquery.SchemaField("month", "INTEGER"),
            bigquery.SchemaField("is_weekend", "BOOLEAN"),
            bigquery.SchemaField("is_holiday", "BOOLEAN")
        ]
    )
    job = client.load_table_from_dataframe(df, TABLE_INPUT, job_config=job_config)
    job.result()
    print(f"✅ Table {TABLE_INPUT} créée avec {len(df)} lignes.")

if __name__ == "__main__":
    df_input = generate_forecast_input()
    print(f"✅ Données générées : {df_input.shape}")
    write_to_bigquery(df_input)
