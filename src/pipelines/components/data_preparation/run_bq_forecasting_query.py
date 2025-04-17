# src/pipelines/components/run_bq_forecasting_query.py

from kfp.v2.dsl import component, Output, Artifact
from datetime import timedelta, datetime
from google.cloud import bigquery
from google.cloud.exceptions import NotFound
import google.api_core.exceptions
import pandas as pd # Import pandas
from src.pipelines.components.preprocessing.fallback_bq import process_data_with_pandas

@component(
    base_image="python:3.9",
    packages_to_install=["google-cloud-bigquery"]
)
def run_bq_forecasting_query(
    project_id: str,
    location: str,
    dataset_id: str,
    source_table: str,
    destination_table_name: str,
    sql_template_path: str,
    end_date_str: str,
    max_data_points: int,
    destination_table_uri: Output[Artifact]
):
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    START_DATE = end_date - timedelta(hours=max_data_points - 1)
    start_date_str = START_DATE.strftime("%Y-%m-%d")
    hours_diff = int((end_date - START_DATE).total_seconds() // 3600)
    
    # Vérification de la limite de Vertex AI (3000 points par série)
    if hours_diff > 3000:
        print(f"⚠️ ATTENTION: Le nombre d'heures ({hours_diff}) dépasse la limite de 3000 points de Vertex AI.")
        print("La période va être automatiquement ajustée pour respecter cette limite.")
        
        # Ajuster la date de début pour rester sous ~3000 heures
        adjusted_start = end_date - timedelta(hours=2950)  # Marge de sécurité
        start_date_str = adjusted_start.strftime("%Y-%m-%d")
        hours_diff = 2950
        print(f"📅 Nouvelle plage ajustée: {start_date_str} à {end_date_str} (~{hours_diff} heures)")
    else:
        print(f"✅ La plage respecte la limite de Vertex AI (max 3000 points par série): {hours_diff} heures.")

    # Définir les timestamps complets pour le filtrage (SQL, Pandas, etc.)
    # On inclut tout le jour de début (00:00:00) et tout le jour de fin (23:59:59)
    start_timestamp_str = f"{start_date_str} 00:00:00"
    end_timestamp_str   = f"{end_date_str} 23:59:59"
    print("=== Récapitulatif final ===")
    print(f"  Start Timestamp   : {start_timestamp_str}")
    print(f"  End Timestamp     : {end_timestamp_str}\n")
    print(f"Nombre d'heures dans la plage : {hours_diff} heures")

    # Read SQL template
    with open(sql_template_path, "r") as f:
        sql_template_content = f.read()

    client = bigquery.Client(project=project_id, location=location)
    export_success = False
    final_source_used = None

    # --- METHOD 1: Try BigQuery Query ---
    try:
        print(f"\nMéthode 1: Tentative d'exécution de la requête BigQuery avec la table source: {source_table}")
        sql_formatted = sql_template_content.format(
            PROJECT_ID=project_id,
            BQ_DATASET=dataset_id,
            BQ_TABLE_PREPARED=destination_table_name,
            SOURCE_TABLE=source_table,
            start_timestamp_str=start_timestamp_str,
            end_timestamp_str=end_timestamp_str
        )
        job = client.query(sql_formatted)
        job.result()  # Wait for the job to complete
        print(f"✅ Méthode 1: Requête BigQuery exécutée avec succès en utilisant {source_table}.")
        export_success = True
        final_source_used = source_table

    except (NotFound, google.api_core.exceptions.BadRequest, Exception) as e:
        print(f"⚠️ Méthode 1: Échec de la requête BigQuery avec la table source ({source_table}): {e}")
        print("\n--- Passage à la Méthode 2: Traitement et export via Pandas ---")

        # --- METHOD 2: Fallback to Pandas Processing ---
        try:
            print(f"\nMéthode 2: Traitement avec Pandas et tentative d'export vers {destination_table_name}")
            # Process data with Pandas
            processed_df = process_data_with_pandas(
                df_raw=None, # Pass None to trigger reading inside the function
                start_timestamp_str=start_timestamp_str,
                end_timestamp_str=end_timestamp_str,
                PROJECT_ID=project_id,
                source_table_id=source_table,
                dataset_id=dataset_id, # Pass dataset_id for export target
                destination_table_name=destination_table_name # Pass destination for export target
            )

            if processed_df is not None:
                 # Check if the export inside process_data_with_pandas succeeded
                 # We infer success if the function completed without raising an exception
                 # and returned a DataFrame. A more robust check might involve querying
                 # the destination table's metadata or row count after the call.
                 print(f"✅ Méthode 2: Traitement Pandas et tentative d'export vers {destination_table_name} terminés.")
                 # We assume the export within process_data_with_pandas worked if no exception was raised
                 export_success = True
                 final_source_used = source_table # Indicate fallback was used
            else:
                 print(f"❌ Méthode 2: Le traitement Pandas a échoué ou n'a retourné aucune donnée.")
                 raise RuntimeError("Fallback processing with Pandas failed.") # Raise error if fallback fails

        except Exception as fallback_e:
            print(f"❌ Méthode 2: Échec critique du fallback Pandas: {fallback_e}")
            import traceback
            traceback.print_exc()
            # Re-raise the exception if the fallback also fails critically
            raise fallback_e

        # --- Final Output ---
        if export_success:
            destination_table_full_id = f"{project_id}.{dataset_id}.{destination_table_name}"
            destination_table_uri.uri = f"bq://{destination_table_full_id}"
            print(f"\n✅ Pipeline step succeeded. Data prepared in: {destination_table_full_id}")
            print(f"   Source data used: {final_source_used}")
            print(f"   Output Artifact URI: {destination_table_uri.uri}")
        else:
            # This part should ideally not be reached if exceptions are raised correctly
            print("\n❌ Pipeline step failed. Aucune méthode n'a pu préparer les données.")
            raise RuntimeError("Data preparation failed using both BigQuery query and Pandas fallback.")
