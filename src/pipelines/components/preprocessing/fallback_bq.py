\
# filepath: /Users/gustavevernay/Desktop/Projets/Pro/Avisia/chicago-taxi-genai-demo/src/pipelines/components/preprocessing/fallback_bq.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from google.cloud import bigquery
from pandas_gbq import read_gbq

def process_data_with_pandas(
    df_raw: pd.DataFrame,
    start_timestamp_str: str,
    end_timestamp_str: str,
    PROJECT_ID: str,
    source_table: str,
    dataset_id: str, # Added dataset_id
    destination_table_name: str # Added destination_table_name
) -> pd.DataFrame | None:
    """
    Effectue l'agrégation, la complétion et l'ingénierie de features sur les données brutes
    en utilisant Pandas, comme alternative à une requête BigQuery directe, et tente d'exporter
    le résultat vers BigQuery.

    Args:
        df_raw: DataFrame Pandas contenant les données brutes ('trip_start_timestamp', 'pickup_community_area').
        start_timestamp_str: Timestamp de début (YYYY-MM-DD HH:MM:SS) pour le filtrage.
        end_timestamp_str: Timestamp de fin (YYYY-MM-DD HH:MM:SS) pour le filtrage.
        PROJECT_ID: ID du projet Google Cloud pour l'accès à BigQuery.
        dataset_id: ID du dataset BigQuery de destination.
        destination_table_name: Nom de la table BigQuery de destination.


    Returns:
        DataFrame Pandas traité et complété, ou None si une erreur survient lors du traitement initial.
        L'export vers BigQuery est une étape supplémentaire tentée à la fin.
    """
    # Instantiate BigQuery client
    client = bigquery.Client(project=PROJECT_ID)
    BQ_TABLE_ID = f"{PROJECT_ID}.{dataset_id}.{destination_table_name}"

    forecasting_query = f"""
        SELECT
        trip_start_timestamp,
        pickup_community_area
        FROM `{source_table}`
        WHERE 
        pickup_community_area IS NOT NULL
        AND trip_start_timestamp >= '{start_timestamp_str}'
        AND trip_start_timestamp < '{end_timestamp_str}'
        """
    if df_raw is None or df_raw.empty:
        try:
            df_raw = read_gbq(forecasting_query, project_id=PROJECT_ID)
            print(f"✅ Dataset chargé avec succès. Nombre de lignes : {len(df_raw):,}")
        except Exception as e:
            print("Fallback Pandas: DataFrame brut vide ou non fourni.")
            return None

    print("Fallback Pandas: Début du traitement des données...")

    try:
        # S'assurer que les colonnes nécessaires existent
        if not {'trip_start_timestamp', 'pickup_community_area'}.issubset(df_raw.columns):
            print("Fallback Pandas: Colonnes 'trip_start_timestamp' ou 'pickup_community_area' manquantes.")
            return None

        # 1. Conversion et Filtrage Temporel Initial
        print(f"Fallback Pandas: Filtrage entre {start_timestamp_str} et {end_timestamp_str}...")
        df_raw['trip_start_timestamp'] = pd.to_datetime(df_raw['trip_start_timestamp'], errors='coerce', utc=True)
        df_raw = df_raw.dropna(subset=['trip_start_timestamp', 'pickup_community_area']) # Supprimer NaT et NaN area

        start_dt = pd.to_datetime(start_timestamp_str, utc=True)
        end_dt = pd.to_datetime(end_timestamp_str, utc=True)

        df_filtered = df_raw[
            (df_raw['trip_start_timestamp'] >= start_dt) &
            (df_raw['trip_start_timestamp'] <= end_dt)
        ].copy()

        if df_filtered.empty:
            print("Fallback Pandas: Aucune donnée dans la plage temporelle spécifiée.")
            return None
        print(f"Fallback Pandas: {len(df_filtered):,} lignes après filtrage temporel.")


        # 2. Agrégation par Heure et Zone (Similaire à la section 3.2 du notebook)
        print("Fallback Pandas: Agrégation par heure et zone...")
        df_filtered["timestamp_hour"] = df_filtered["trip_start_timestamp"].dt.floor("H")
        df_demand = (
            df_filtered
            .groupby(["timestamp_hour", "pickup_community_area"])
            .size()
            .reset_index(name="trip_count")
            .sort_values(["timestamp_hour", "pickup_community_area"])
        )
        print(f"Fallback Pandas: {len(df_demand):,} lignes après agrégation.")

        # 3. Complétion des Séries Temporelles (Similaire à la section 3.3)
        print("Fallback Pandas: Complétion des séries temporelles...")
        min_time = df_demand["timestamp_hour"].min()
        max_time = df_demand["timestamp_hour"].max()

        # S'assurer que la plage générée correspond bien aux bornes filtrées
        # (peut différer légèrement si les données agrégées ne couvrent pas exactement les bornes)
        all_hours = pd.date_range(start=min_time, end=max_time, freq="H", tz='UTC')

        all_zones = df_demand["pickup_community_area"].dropna().unique()
        all_zones = sorted(all_zones)

        if not all_hours.empty and len(all_zones) > 0:
            complete_index = pd.MultiIndex.from_product(
                [all_hours, all_zones],
                names=["timestamp_hour", "pickup_community_area"]
            )
            df_complete = pd.DataFrame(index=complete_index).reset_index()

            df_demand_complete = pd.merge(
                df_complete,
                df_demand,
                on=["timestamp_hour", "pickup_community_area"],
                how="left"
            )
            df_demand_complete["trip_count"] = df_demand_complete["trip_count"].fillna(0).astype(int)
            print(f"Fallback Pandas: {len(df_demand_complete):,} lignes après complétion.")
        else:
             print("Fallback Pandas: Impossible de créer l'index complet (pas d'heures ou de zones).")
             # Retourner les données agrégées mais non complétées si nécessaire ?
             # Ou retourner None car la complétion est souvent essentielle.
             return None


        # 4. Ingénierie de Features Temporelles (Similaire à la section 3.4)
        print("Fallback Pandas: Ajout des features temporelles...")
        df_demand_complete["hour"] = df_demand_complete["timestamp_hour"].dt.hour
        # Note: dayofweek en Pandas: Lundi=0, Dimanche=6. BQ: Dimanche=1, Samedi=7.
        # Vertex AI gère différents formats, mais soyons cohérents si possible.
        # Gardons Lundi=0 pour l'instant.
        df_demand_complete["day_of_week"] = df_demand_complete["timestamp_hour"].dt.dayofweek
        df_demand_complete["month"] = df_demand_complete["timestamp_hour"].dt.month
        df_demand_complete["day_of_year"] = df_demand_complete["timestamp_hour"].dt.dayofyear
        # Utiliser .astype(int) pour éviter les types Int64 potentiellement problématiques pour BQ
        df_demand_complete["week_of_year"] = df_demand_complete["timestamp_hour"].dt.isocalendar().week.astype(int)
        df_demand_complete["year"] = df_demand_complete["timestamp_hour"].dt.year
        df_demand_complete["is_weekend"] = df_demand_complete["day_of_week"].isin([5, 6]).astype(int)


        print("Fallback Pandas: Traitement terminé avec succès.")

        # --- NOUVELLE LOGIQUE D'EXPORT ---
        export_success = False
        try:
            print("\nTentative d'export via pandas DataFrame (Méthode 2)...")
            # S'assurer que le DataFrame existe et n'est pas vide
            if "df_demand_complete" in locals() and df_demand_complete is not None and not df_demand_complete.empty:
                # --- FILTRAGE DU DATAFRAME PANDAS ---
                # Appliquer le même filtre temporel que la requête SQL (peut être redondant mais sert de vérification)
                print(f"Filtrage du DataFrame Pandas entre {start_timestamp_str} et {end_timestamp_str}...")
                start_dt_filter = pd.to_datetime(start_timestamp_str, utc=True)
                end_dt_filter = pd.to_datetime(end_timestamp_str, utc=True)

                # Ensure timestamp_hour is timezone-aware for comparison
                if df_demand_complete['timestamp_hour'].dt.tz is None:
                     df_demand_complete['timestamp_hour'] = df_demand_complete['timestamp_hour'].dt.tz_localize('UTC')


                df_filtered_pandas = df_demand_complete[
                    (df_demand_complete['timestamp_hour'] >= start_dt_filter) &
                    (df_demand_complete['timestamp_hour'] <= end_dt_filter) # Note: <= includes the end hour
                ].copy() # Utiliser .copy() pour éviter SettingWithCopyWarning
                print(f"Nombre de lignes après filtrage Pandas: {len(df_filtered_pandas):,}")

                if not df_filtered_pandas.empty:
                    # Sélectionner les colonnes pertinentes (celles générées précédemment ET dans la requête SQL)
                    columns_to_keep = [
                        "timestamp_hour", "pickup_community_area", "trip_count",
                        "hour", "day_of_week", "month", "year", "day_of_year",
                        "week_of_year", # Assurez-vous qu'elle existe si utilisée
                        "is_weekend", "hour_sin", "hour_cos"
                    ]
                    # Garder uniquement les colonnes qui existent réellement dans le DataFrame
                    columns_to_export = [col for col in columns_to_keep if col in df_filtered_pandas.columns]

                    # Convertir les colonnes potentiellement Int64 en int64 standard ou float64 si NaN présent
                    for col in df_to_export.select_dtypes(include=['Int64']).columns:
                         if df_to_export[col].isnull().any():
                              df_to_export[col] = df_to_export[col].astype('float64')
                         else:
                              df_to_export[col] = df_to_export[col].astype('int64')


                    df_to_export = df_filtered_pandas[columns_to_export]


                    # Configurer le job d'export
                    job_config = bigquery.LoadJobConfig(
                        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
                        # Détecter automatiquement le schéma ou le spécifier si nécessaire
                        autodetect=True,
                        # Ou spécifier le schéma explicitement si autodetect pose problème
                        # schema=[
                        #     bigquery.SchemaField("timestamp_hour", "TIMESTAMP"),
                        #     bigquery.SchemaField("pickup_community_area", "INTEGER"),
                        #     bigquery.SchemaField("trip_count", "INTEGER"),
                        #     # ... autres colonnes avec leurs types BQ ...
                        # ]
                    )

                    # Exporter le DataFrame filtré
                    print(f"Exportation vers: {BQ_TABLE_ID}")
                    job = client.load_table_from_dataframe(
                        df_to_export, BQ_TABLE_ID, job_config=job_config
                    )
                    job.result()  # Attendre la fin du job
                    print(f"✅ Méthode 2: Table exportée vers BigQuery via pandas : {BQ_TABLE_ID}")
                    export_success = True

                    # Afficher un extrait
                    print("\nExtrait des données exportées via Pandas:")
                    # Remplacer display() par print() pour les scripts
                    print(df_to_export.head())
                else:
                    print("⚠️ Le DataFrame est vide après filtrage, export annulé.")
            else:
                print("⚠️ DataFrame 'df_demand_complete' non trouvé ou vide. Export via Pandas impossible.")

        except Exception as e2:
            print(f"⚠️ Échec de l'export via pandas : {e2}")
            import traceback
            traceback.print_exc() # Print full traceback for export error

        # Retourner le DataFrame traité, même si l'export a échoué
        return df_demand_complete.sort_values(["timestamp_hour", "pickup_community_area"])

    except Exception as e:
        print(f"Fallback Pandas: Erreur lors du traitement des données: {e}")
        import traceback
        traceback.print_exc()
        return None

# Exemple d'utilisation (pour test local uniquement)
if __name__ == "__main__":
    # Créer un exemple de DataFrame brut
    data = {
        'trip_start_timestamp': pd.to_datetime(['2023-01-01 00:15:00', '2023-01-01 00:45:00', '2023-01-01 01:10:00', '2023-01-01 00:20:00', '2023-01-02 03:00:00'], utc=True),
        'pickup_community_area': [8.0, 8.0, 76.0, 8.0, 8.0],
        'other_col': [1, 2, 3, 4, 5]
    }
    df_test_raw = pd.DataFrame(data)
    # La localisation UTC est maintenant faite dans la fonction, mais c'est bien de l'avoir ici aussi pour la clarté
    # df_test_raw['trip_start_timestamp'] = df_test_raw['trip_start_timestamp'].dt.tz_localize('UTC')

    # Définir des valeurs d'exemple pour les arguments requis
    test_project_id = "votre-projet-gcp-test" # Remplacez par un ID de projet si nécessaire pour les tests BQ
    test_source_table = "votre-projet-gcp-test.votre_dataset.table_source_test"
    test_dataset_id = "votre_dataset_test"
    test_dest_table = "table_destination_test"
    start_ts = "2023-01-01 00:00:00"
    end_ts = "2023-01-01 23:59:59"

    print("--- Démarrage du test local ---")
    processed_df = process_data_with_pandas(
        df_raw=df_test_raw.copy(), # Passer le DataFrame de test
        start_timestamp_str=start_ts,
        end_timestamp_str=end_ts,
        PROJECT_ID=test_project_id,
        source_table=test_source_table, # Utilisé si df_raw est None
        dataset_id=test_dataset_id,
        destination_table_name=test_dest_table
    )

    if processed_df is not None:
        print("\n--- DataFrame traité (test local) ---")
        print(processed_df.head())
        print("\nTypes de données:")
        print(processed_df.dtypes)
    else:
        print("\n--- Échec du traitement de l'exemple local ---")

    print("--- Fin du test local ---")

