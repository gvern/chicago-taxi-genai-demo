# src/pipelines/components/generate_forecasting_data/generate_forecast_input.py

# Correction : Ajouter Artifact aux imports kfp.v2.dsl

# --- MODIFICATION : Importer Input ET Artifact ---
from kfp.v2.dsl import component, Output, Dataset, Input, Artifact # Artifact ajouté ici
import logging
import argparse
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Fonction d'implémentation interne ---
# Prend maintenant le chemin du fichier en entrée
def _generate_forecast_input_impl(
    project_id: str,
    bq_dataset: str,
    bq_table_prepared_path: str, # Chemin vers le fichier contenant le nom de table
    id_col: str,
    time_col: str,
    forecast_horizon_hours: int,
    forecast_start_time: str,
    output_gcs_path: str
):
    """
    Génère le DataFrame de features d'entrée pour les prédictions futures.
    Lit le nom de la table de données préparées depuis un chemin de fichier d'entrée.

    Args:
        project_id: ID du projet GCP.
        bq_dataset: ID du dataset BigQuery.
        bq_table_prepared_path: Chemin local vers le fichier contenant le nom de la table BQ préparée.
        id_col: Nom de la colonne d'identifiant de série.
        time_col: Nom de la colonne de timestamp dans le fichier CSV de sortie.
        forecast_horizon_hours: Nombre d'heures à prévoir.
        forecast_start_time: Timestamp de début pour la période de prévision (string UTC).
        output_gcs_path: Chemin GCS pour sauvegarder le fichier CSV de sortie.
    """
    # Imports internes
    import pandas as pd
    import numpy as np
    from google.cloud import bigquery
    from google.cloud import storage
    # Import des fonctions de feature engineering
    from src.pipelines.components.preprocessing.feature_engineering import preprocess_data_for_xgboost

    # Lire le nom de la table depuis le fichier
    try:
        with open(bq_table_prepared_path, 'r') as f:
            bq_table_name_str = f.read().strip()
        if not bq_table_name_str:
            raise ValueError(f"Le fichier de nom de table préparée ({bq_table_prepared_path}) est vide.")
        logging.info(f"Nom de la table préparée lu depuis {bq_table_prepared_path}: '{bq_table_name_str}'")
    except Exception as read_e:
        logging.error(f"Erreur lors de la lecture du nom de table depuis {bq_table_prepared_path}: {read_e}")
        raise RuntimeError(f"Impossible de lire le nom de la table préparée: {read_e}") from read_e

    client = bigquery.Client(project=project_id)

    # Obtenir les IDs de série distincts depuis la table préparée (en utilisant le nom lu)
    query_ids = f"""
        SELECT DISTINCT `{id_col}`
        FROM `{project_id}.{bq_dataset}.{bq_table_name_str}`
        WHERE `{id_col}` IS NOT NULL
        ORDER BY `{id_col}`
    """
    logging.info(f"Récupération des IDs distincts avec la requête : {query_ids}")
    try:
        ids_df = client.query(query_ids).to_dataframe()
        ids = ids_df[id_col].unique()
        logging.info(f"Trouvé {len(ids)} IDs distincts.")
        if len(ids) == 0:
            raise ValueError("Aucun ID distinct trouvé dans la table source.")
    except Exception as e:
        logging.error(f"Échec de la récupération des IDs distincts : {e}")
        raise

    # Générer les timestamps futurs (logique identique)
    try:
        start_ts = pd.to_datetime(forecast_start_time, utc=True)
        future_timestamps = pd.date_range(
            start=start_ts, periods=forecast_horizon_hours, freq='H', tz='UTC', name=time_col
        )
        logging.info(f"Génération de {forecast_horizon_hours} timestamps à partir de {start_ts}.")
    except Exception as e:
        logging.error(f"Échec de la génération des timestamps futurs : {e}")
        raise

    # Créer le DataFrame futur de base (logique identique)
    future_df = pd.MultiIndex.from_product([ids, future_timestamps], names=[id_col, time_col])
    future_df = pd.DataFrame(index=future_df).reset_index()
    logging.info(f"Création de la grille future DataFrame shape: {future_df.shape}")

    # --- FEATURE ENGINEERING (logique identique) ---
    logging.info("Génération des features temporelles de base (hour, dayofweek)...")
    future_df['timestamp'] = pd.to_datetime(future_df[time_col])
    future_df['hour'] = future_df['timestamp'].dt.hour
    future_df['dayofweek'] = future_df['timestamp'].dt.dayofweek

    logging.info("Génération des features non-target (cyclical, OHE)...")
    try:
        cols_for_fe = ['timestamp', 'hour', 'dayofweek', id_col]
        rename_dict = {id_col: 'pickup_community_area'} if id_col != 'pickup_community_area' else {}
        temp_df = future_df[cols_for_fe].rename(columns=rename_dict)
        future_features_df = preprocess_data_for_xgboost(temp_df, is_train=False)
    except Exception as e:
        logging.error(f"Erreur pendant l'ingénierie des features futures : {e}")
        raise

    # Réattacher les identifiants (logique identique)
    future_features_df = future_features_df.set_index(future_df.index)
    future_features_df[id_col] = future_df[id_col]
    future_features_df[time_col] = future_df[time_col]

    # Sélectionner et ordonner les colonnes (logique identique)
    output_cols = [id_col, time_col] + [col for col in future_features_df.columns if col not in [id_col, time_col]]
    output_cols = sorted(list(set(output_cols)), key=lambda x: (x != id_col, x != time_col, x))
    final_future_df = future_features_df[output_cols]

    logging.info(f"Shape final du DataFrame des features futures : {final_future_df.shape}")
    logging.info(f"Colonnes finales des features futures : {final_future_df.columns.tolist()}")

    # --- SAVE TO GCS --- (logique identique)
    logging.info(f"Sauvegarde des features futures vers {output_gcs_path}")
    try:
        if not output_gcs_path.startswith("gs://"):
            raise ValueError("output_gcs_path doit commencer par gs://")
        final_future_df.to_csv(output_gcs_path, index=False)
        logging.info("Données des features futures sauvegardées avec succès.")
    except Exception as e:
        logging.error(f"Échec de la sauvegarde des features futures sur GCS: {e}")
        raise

# --- KFP Component Definition ---
@component(
    base_image="europe-west1-docker.pkg.dev/avisia-certification-ml-yde/chicago-taxis-demo/forecasting-pipeline:latest",
    # packages_to_install=[...] # Dépendances dans l'image
)
def generate_forecast_input(
    project_id: str,
    bq_dataset: str,
    # --- MODIFICATION : Utiliser Input[Artifact] ---
    bq_table_prepared_path: Input[Artifact], # Input Artifact pointant vers le fichier
    id_col: str,
    time_col: str,
    forecast_horizon_hours: int,
    forecast_start_time: str,
    output_gcs_path: str,
    future_features: Output[Dataset]
):
    """
    Composant KFP pour générer les features pour les prédictions futures.
    Lit le nom de la table BQ préparée depuis un fichier d'artefact d'entrée.
    """
    import logging

    # Appeler la fonction d'implémentation avec le chemin du fichier .path de l'artefact
    _generate_forecast_input_impl(
        project_id=project_id,
        bq_dataset=bq_dataset,
        bq_table_prepared_path=bq_table_prepared_path.path, # Passer le chemin .path
        id_col=id_col,
        time_col=time_col,
        forecast_horizon_hours=forecast_horizon_hours,
        forecast_start_time=forecast_start_time,
        output_gcs_path=output_gcs_path
    )
    # Définir l'URI pour l'artefact de sortie (reste identique)
    future_features.uri = output_gcs_path
    logging.info(f"Component finished. Output artifact URI: {future_features.uri}")

# --- Main execution block for local testing ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate future forecast input features.")
    parser.add_argument('--project_id', type=str, required=True, help='GCP Project ID')
    parser.add_argument('--bq_dataset', type=str, required=True, help='BigQuery dataset ID')
    parser.add_argument('--bq_table_prepared_path', type=str, required=True, help='Local path to the file containing the prepared BQ table name')
    parser.add_argument('--id_col', type=str, default='pickup_community_area', help='Name of the series ID column')
    parser.add_argument('--time_col', type=str, default='timestamp_hour', help='Name for the timestamp column in the output CSV')
    parser.add_argument('--forecast_horizon_hours', type=int, required=True, help='Number of hours to forecast')
    parser.add_argument('--forecast_start_time', type=str, required=True, help='Forecast start timestamp (UTC), e.g., yyyy-MM-ddTHH:mm:ss')
    parser.add_argument('--output_gcs_path', type=str, required=True, help='GCS path for the output CSV file (gs://...)')
    args = parser.parse_args()

    class DummyOutput:
        def __init__(self): self.uri = None
    future_features_output = DummyOutput()

    try:
        _generate_forecast_input_impl(
            project_id=args.project_id,
            bq_dataset=args.bq_dataset,
            bq_table_prepared_path=args.bq_table_prepared_path,
            id_col=args.id_col,
            time_col=args.time_col,
            forecast_horizon_hours=args.forecast_horizon_hours,
            forecast_start_time=args.forecast_start_time,
            output_gcs_path=args.output_gcs_path
        )
        future_features_output.uri = args.output_gcs_path
        logging.info(f"Local execution finished. Output would be at: {future_features_output.uri}")
    except Exception as e:
        logging.error(f"Local execution failed: {e}")