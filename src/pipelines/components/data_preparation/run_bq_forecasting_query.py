# src/pipelines/components/data_preparation/run_bq_forecasting_query.py

# Correction : Utilisation de NamedTuple pour la sortie du nom de table
# + Logique de succès/échec pour le fallback modifiée

from kfp.v2.dsl import component, Output, Artifact
import logging
import traceback # Importer traceback
# --- MODIFICATION : Import ajouté ---


# Configure logging - Peut rester au niveau module ou être déplacé dans la fonction
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


@component(
    base_image="europe-west1-docker.pkg.dev/avisia-certification-ml-yde/chicago-taxis-demo/forecasting-pipeline:latest",
    # packages_to_install=["google-cloud-bigquery", ...] # Dépendances dans l'image
)
def run_bq_forecasting_query(
    project_id: str,
    location: str,
    dataset_id: str,
    source_table: str,
    destination_table_name: str, # Le nom d'entrée reste le même
    sql_template_path_in_container: str,
    end_date_str: str,
    max_data_points: int,
    destination_table_uri: Output[Artifact], # Conserver ce paramètre Artifact
    destination_table_name_out: Output[Artifact], # Conserver ce paramètre Artifact
): # --- MODIFICATION : Annotation de retour supprimée ---
    """
    Exécute une requête BigQuery (définie dans un template SQL) pour préparer les données de forecasting.

    Lit un template SQL, le formate avec les paramètres fournis (dates, tables),
    exécute la requête pour créer ou remplacer une table de destination dans BigQuery.
    Inclut un mécanisme de secours utilisant Pandas si la requête BQ échoue.

    Args:
        project_id: ID du projet GCP.
        location: Localisation BigQuery (ex: 'US', 'EU').
        dataset_id: ID du dataset BigQuery pour la table de destination.
        source_table: ID complet de la table source BigQuery (ex: projet.dataset.table).
        destination_table_name: Nom de la table de destination BigQuery (sans projet/dataset).
        sql_template_path_in_container: Chemin vers le fichier template .sql dans le conteneur.
        end_date_str: Date de fin pour l'extraction des données (YYYY-MM-DD).
        max_data_points: Nombre maximum d'heures (points de données) par série à inclure avant end_date.
        destination_table_uri: Artefact de sortie pour stocker l'URI BQ de la table créée.
        # destination_table_name_out: SUPPRIMÉ

    Raises:
        RuntimeError: Si la préparation des données échoue après tentative BQ et fallback Pandas.
        FileNotFoundError: Si le fichier template SQL n'est pas trouvé.
        ValueError: Si les paramètres d'entrée (comme la date) sont invalides.
        Exception: Pour d'autres erreurs inattendues.

    Returns:
        DataPrepOutputs: Un NamedTuple contenant :
            destination_table_name_str: Le nom simple (string) de la table créée.
    """
    # Imports nécessaires (restent les mêmes)
    from datetime import timedelta, datetime
    from google.cloud import bigquery
    from google.cloud.exceptions import NotFound
    import google.api_core.exceptions
    import pandas as pd
    import logging
    import os

    # Import de la fonction fallback (reste le même)
    process_data_with_pandas = None
    try:
        from src.pipelines.components.preprocessing.fallback_bq import process_data_with_pandas
        logging.info("Fonction fallback 'process_data_with_pandas' importée avec succès.")
    except ImportError:
        logging.error("Impossible d'importer 'process_data_with_pandas' depuis fallback_bq. Le fallback ne sera pas disponible.")

    # --- Validation et Préparation des Paramètres --- (reste le même)
    try:
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    except ValueError:
        logging.error(f"Format invalide pour end_date_str: {end_date_str}. Attendu : YYYY-MM-DD.")
        raise

    start_date = end_date - timedelta(hours=max_data_points - 1)
    start_date_str_calc = start_date.strftime("%Y-%m-%d")
    hours_diff = max_data_points

    if hours_diff > 3000:
        logging.warning(f"Les {hours_diff} points demandés dépassent la limite Vertex AI (3000). Ajustement de la date de début.")
        start_date = end_date - timedelta(hours=2999)
        start_date_str_calc = start_date.strftime("%Y-%m-%d")
        hours_diff = 3000
    else:
        logging.info(f"Plage de données (~{hours_diff} heures) respecte la limite Vertex AI.")

    start_timestamp_str = f"{start_date_str_calc} 00:00:00"
    end_timestamp_str = f"{end_date_str} 23:59:59"
    logging.info("--- Plage Temporelle des Données ---")
    logging.info(f"  Timestamp Début: {start_timestamp_str}")
    logging.info(f"  Timestamp Fin  : {end_timestamp_str}")

    # --- Lecture du Template SQL --- (reste le même)
    logging.info(f"Lecture du template SQL depuis : {sql_template_path_in_container}")
    try:
        with open(sql_template_path_in_container, 'r') as f:
            sql_template_content = f.read()
        logging.info("Lecture du template SQL réussie.")
    except FileNotFoundError:
        logging.error(f"Fichier template SQL non trouvé : {sql_template_path_in_container}")
        raise
    except Exception as e:
        logging.error(f"Erreur lors de la lecture du fichier template SQL : {e}")
        raise

    # --- Initialisation et Tentatives de Préparation --- (reste le même)
    client = bigquery.Client(project=project_id, location=location)
    export_success = False
    final_destination_table_id = f"{project_id}.{dataset_id}.{destination_table_name}"

    # --- MÉTHODE 1 : Requête BigQuery Directe --- (reste le même)
    try:
        logging.info(f"Méthode 1: Tentative d'exécution de la requête BigQuery (Source: {source_table}, Dest: {final_destination_table_id})")
        sql_formatted = sql_template_content.format(
            PROJECT_ID=project_id, BQ_DATASET=dataset_id, BQ_TABLE_PREPARED=destination_table_name,
            SOURCE_TABLE=source_table, start_timestamp_str=start_timestamp_str, end_timestamp_str=end_timestamp_str
        )
        logging.info("Requête SQL formatée (début) : " + sql_formatted[:200].replace('\n', ' ') + "...")
        logging.info("Exécution de la requête SQL formatée...")
        query_job = client.query(sql_formatted)
        query_job.result()
        logging.info(f"Méthode 1: Requête BigQuery exécutée avec succès. Données écrites dans {final_destination_table_id}")
        export_success = True

    except Exception as e:
        logging.warning(f"Méthode 1: Échec de la requête BigQuery : {e}")

        # --- MÉTHODE 2 : Fallback avec Pandas --- (reste le même)
        logging.info("--- Tentative Méthode 2: Fallback avec Pandas ---")
        if process_data_with_pandas is None:
             logging.error("Fonction fallback 'process_data_with_pandas' non disponible. Impossible de continuer.")
        else:
            try:
                logging.info(f"Méthode 2: Appel de process_data_with_pandas pour exporter vers {final_destination_table_id}")
                processed_df_fallback = process_data_with_pandas(
                    df_raw=None,
                    start_timestamp_str=start_timestamp_str,
                    end_timestamp_str=end_timestamp_str,
                    PROJECT_ID=project_id,
                    source_table=source_table,
                    dataset_id=dataset_id,
                    destination_table_name=destination_table_name
                )
                logging.info(f"Méthode 2: La fonction fallback process_data_with_pandas s'est terminée avec succès.")
                export_success = True

            except Exception as fallback_e:
                logging.error(f"Méthode 2: Échec critique pendant l'exécution du fallback Pandas : {fallback_e}")
                traceback.print_exc()

 # --- Sortie Finale du Composant ---
    if export_success:
        logging.info(f"Préparation des données réussie. Table de sortie : {final_destination_table_id}")

        # Définir l'artefact de sortie (URI BQ) - reste le même
        destination_table_uri.uri = f"bq://{final_destination_table_id}"
        logging.info(f"URI de l'artefact de sortie : {destination_table_uri.uri}")

        # --- MODIFICATION : Écrire le nom de la table dans le fichier de sortie ---
        # Utiliser destination_table_name_out.path qui est fourni par KFP
        try:
            output_file_path = destination_table_name_out.path
            logging.info(f"Écriture du nom de la table '{destination_table_name}' dans le fichier de sortie : {output_file_path}")
            # S'assurer que le répertoire existe (normalement KFP le gère)
            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
            with open(output_file_path, 'w') as f:
                f.write(destination_table_name)
            logging.info("Écriture du nom de la table dans le fichier réussie.")
        except Exception as write_e:
            logging.error(f"ERREUR CRITIQUE: Échec de l'écriture du nom de table dans le fichier de sortie {destination_table_name_out.path}: {write_e}")
            # Lever une erreur pour faire échouer la tâche KFP si l'écriture échoue
            raise RuntimeError(f"Échec de l'écriture du nom de table dans le fichier : {write_e}") from write_e

        # Pas de 'return' ici car les sorties sont gérées par les paramètres Output[...]

    else:
        # Cas où ni la Méthode 1 ni la Méthode 2 n'ont réussi
        logging.error("La préparation des données a échoué globalement. Aucune méthode n'a réussi.")
        # Lever une exception pour signaler l'échec au pipeline KFP
        raise RuntimeError("La préparation des données a échoué.")