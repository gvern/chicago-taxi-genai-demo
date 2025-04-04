import pandas as pd
from datetime import datetime, timedelta
from google.cloud import bigquery

PROJECT_ID = "avisia-certification-ml-yde"
BQ_TABLE_TARGET = "chicago_taxis.forecast_input"
HORIZON_HOURS = 24  # modifier si nécessaire
BQ_TRAINING_TABLE = "chicago_taxis.demand_by_hour"


def get_unique_zones(client, table=BQ_TRAINING_TABLE):
    """Récupère les pickup_community_area uniques depuis la table de training."""
    query = f"""
    SELECT DISTINCT pickup_community_area
    FROM `{PROJECT_ID}.{table}`
    WHERE pickup_community_area IS NOT NULL
    """
    return client.query(query).to_dataframe()


def generate_future_timestamps(horizon_hours: int):
    """Génère les timestamps futurs horaires à horizon donné."""
    now = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    return [now + timedelta(hours=i) for i in range(1, horizon_hours + 1)]


def generate_forecast_input(horizon_hours: int = HORIZON_HOURS,
                            project_id: str = PROJECT_ID,
                            training_table: str = BQ_TRAINING_TABLE) -> pd.DataFrame:
    client = bigquery.Client(project=project_id)
    zones = get_unique_zones(client, table=training_table)
    future_timestamps = generate_future_timestamps(horizon_hours)

    # Produit le cartésien des zones × timestamps
    input_data = pd.DataFrame(
        [(zone, ts) for zone in zones["pickup_community_area"] for ts in future_timestamps],
        columns=["pickup_community_area", "timestamp_hour"]
    )

    # Ajout éventuel de features temporelles
    input_data["hour"] = input_data["timestamp_hour"].dt.hour
    input_data["day_of_week"] = input_data["timestamp_hour"].dt.dayofweek
    input_data["month"] = input_data["timestamp_hour"].dt.month

    return input_data


def write_to_bigquery(df: pd.DataFrame,
                      table: str = BQ_TABLE_TARGET,
                      project_id: str = PROJECT_ID):
    df.to_gbq(destination_table=table, project_id=project_id, if_exists="replace")
    print(f"✅ Données écrites dans BigQuery : {table}")


if __name__ == "__main__":
    df_input = generate_forecast_input()
    print(f"✅ Données générées : {df_input.shape}")
    write_to_bigquery(df_input)
