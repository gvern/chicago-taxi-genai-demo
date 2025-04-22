# src/pipelines/components/preprocessing/fallback_bq.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from google.cloud import bigquery
from pandas_gbq import read_gbq
import logging
import traceback # Importer traceback pour l'utiliser

# Configurer le logging (peut être fait une seule fois au niveau module)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_data_with_pandas(
    df_raw: pd.DataFrame | None, # Accepter None explicitement
    start_timestamp_str: str,
    end_timestamp_str: str,
    PROJECT_ID: str,
    source_table: str, # Nom corrigé (était source_table_id dans l'appel précédent)
    dataset_id: str,
    destination_table_name: str
) -> pd.DataFrame: # Le retour est DataFrame, car une erreur lève une exception
    """
    Effectue l'agrégation, la complétion et l'ingénierie de features en utilisant Pandas
    comme alternative à une requête BigQuery directe, et tente d'exporter
    le résultat vers BigQuery. Lève une exception en cas d'échec.

    Args:
        df_raw: DataFrame Pandas contenant les données brutes ('trip_start_timestamp', 'pickup_community_area').
                Peut être None pour déclencher la lecture depuis BQ.
        start_timestamp_str: Timestamp de début (YYYY-MM-DD HH:MM:SS) pour le filtrage.
        end_timestamp_str: Timestamp de fin (YYYY-MM-DD HH:MM:SS) pour le filtrage.
        PROJECT_ID: ID du projet Google Cloud pour l'accès à BigQuery.
        source_table: Nom complet de la table source dans BigQuery (ex: projet.dataset.table).
        dataset_id: ID du dataset BigQuery de destination.
        destination_table_name: Nom de la table BigQuery de destination (sans projet/dataset).

    Returns:
        pd.DataFrame: DataFrame Pandas traité et complété si le traitement et l'export réussissent.

    Raises:
        RuntimeError: Si une étape critique (lecture, traitement, export) échoue.
        ValueError: Si les données ou la configuration sont invalides.
    """
    # Initialiser les variables importantes
    client = bigquery.Client(project=PROJECT_ID)
    BQ_TABLE_ID = f"{PROJECT_ID}.{dataset_id}.{destination_table_name}"
    df_demand_complete = None # Initialiser pour le retour final

    # --- Lecture des données brutes ---
    if df_raw is None or df_raw.empty:
        logging.info("DataFrame brut non fourni, lecture depuis BigQuery via pandas-gbq...")
        # Utiliser <= pour end_timestamp pour être cohérent avec le filtrage pandas
        forecasting_query = f"""
            SELECT trip_start_timestamp, pickup_community_area
            FROM `{source_table}`
            WHERE pickup_community_area IS NOT NULL
            AND trip_start_timestamp >= TIMESTAMP('{start_timestamp_str}')
            AND trip_start_timestamp <= TIMESTAMP('{end_timestamp_str}')
            """ # Utiliser <= pour inclure la fin
        try:
            # Note: read_gbq peut consommer beaucoup de mémoire pour de gros datasets
            # Désactiver la barre de progression tqdm pour éviter les warnings dans les logs KFP
            df_raw = read_gbq(forecasting_query, project_id=PROJECT_ID, progress_bar_type=None)
            logging.info(f"✅ Dataset chargé avec succès via pandas-gbq. Nombre de lignes : {len(df_raw):,}")
            if df_raw.empty:
                 logging.warning("Les données lues depuis BQ sont vides pour la période donnée.")
                 # Lever une erreur pour signaler clairement qu'il n'y a pas de données à traiter
                 raise ValueError("No data returned from BigQuery for the specified period.")
        except Exception as e:
            logging.error(f"Fallback Pandas: Échec du chargement depuis BQ via pandas-gbq: {e}")
            # Lever une exception pour signaler l'échec de lecture
            raise RuntimeError(f"Failed to load data from BigQuery using pandas-gbq: {e}") from e

    logging.info("Fallback Pandas: Début du traitement des données...")

    try:
        # --- Traitement Pandas (Agrégation, Complétion, Features) ---

        # S'assurer que les colonnes nécessaires existent
        if not {'trip_start_timestamp', 'pickup_community_area'}.issubset(df_raw.columns):
            logging.error("Fallback Pandas: Colonnes 'trip_start_timestamp' ou 'pickup_community_area' manquantes.")
            raise ValueError("Missing required columns in raw data.")

        # 1. Conversion et Filtrage Temporel Initial
        logging.info(f"Fallback Pandas: Filtrage entre {start_timestamp_str} et {end_timestamp_str}...")
        # Convertir en datetime AVANT de filtrer
        df_raw['trip_start_timestamp'] = pd.to_datetime(df_raw['trip_start_timestamp'], errors='coerce', utc=True)
        df_raw = df_raw.dropna(subset=['trip_start_timestamp', 'pickup_community_area']) # Supprimer NaT et NaN area

        start_dt = pd.to_datetime(start_timestamp_str, utc=True)
        end_dt = pd.to_datetime(end_timestamp_str, utc=True)

        # Filtrer les données brutes converties
        df_filtered = df_raw[
            (df_raw['trip_start_timestamp'] >= start_dt) &
            (df_raw['trip_start_timestamp'] <= end_dt)
        ].copy()

        if df_filtered.empty:
            logging.warning("Fallback Pandas: Aucune donnée dans la plage temporelle spécifiée après filtrage initial.")
            # Lever une erreur pour signaler le problème au lieu de continuer avec un DataFrame vide
            raise ValueError("No data found within the specified time range after initial filtering.")
        logging.info(f"Fallback Pandas: {len(df_filtered):,} lignes après filtrage temporel.")

        # 2. Agrégation par Heure et Zone
        logging.info("Fallback Pandas: Agrégation par heure et zone...")
        # Utiliser dt.floor pour être sûr d'avoir l'heure pile
        df_filtered["timestamp_hour"] = df_filtered["trip_start_timestamp"].dt.floor("H")
        df_demand = (
            df_filtered
            # Utiliser observed=True pour performance si pandas >= 1.5, sinon l'omettre
            .groupby(["timestamp_hour", "pickup_community_area"]) # Retiré observed=True pour compatibilité
            .size()
            .reset_index(name="trip_count")
            .sort_values(["timestamp_hour", "pickup_community_area"])
        )
        logging.info(f"Fallback Pandas: {len(df_demand):,} lignes après agrégation.")

        # 3. Complétion des Séries Temporelles
        logging.info("Fallback Pandas: Complétion des séries temporelles...")
        if df_demand.empty:
             logging.warning("Fallback Pandas: Aucune donnée après agrégation, impossible de compléter.")
             raise ValueError("No data after aggregation, cannot perform completion.")

        min_time = df_demand["timestamp_hour"].min()
        max_time = df_demand["timestamp_hour"].max()
        # Générer la plage complète basée sur les dates de filtrage pour assurer la couverture
        all_hours = pd.date_range(start=min_time, end=max_time, freq="H", tz='UTC') # Utiliser min/max des données agrégées
        all_zones = sorted(df_demand["pickup_community_area"].dropna().unique()) # Utiliser les zones des données agrégées

        if not all_hours.empty and len(all_zones) > 0:
            complete_index = pd.MultiIndex.from_product(
                [all_hours, all_zones],
                names=["timestamp_hour", "pickup_community_area"]
            )
            df_complete = pd.DataFrame(index=complete_index).reset_index()
            # Assurer que les types sont corrects pour le merge
            df_demand['timestamp_hour'] = pd.to_datetime(df_demand['timestamp_hour'])
            # Essayer de convertir 'pickup_community_area' en int si possible, sinon float
            try:
                 df_demand['pickup_community_area'] = df_demand['pickup_community_area'].astype(int)
                 df_complete['pickup_community_area'] = df_complete['pickup_community_area'].astype(int)
            except (ValueError, TypeError):
                 logging.warning("Conversion de 'pickup_community_area' en int échouée lors de la préparation du merge, utilisation de float.")
                 df_demand['pickup_community_area'] = df_demand['pickup_community_area'].astype(float)
                 df_complete['pickup_community_area'] = df_complete['pickup_community_area'].astype(float)


            df_demand_complete = pd.merge(
                df_complete,
                df_demand,
                on=["timestamp_hour", "pickup_community_area"],
                how="left"
            )
            df_demand_complete["trip_count"] = df_demand_complete["trip_count"].fillna(0).astype(int)
            logging.info(f"Fallback Pandas: {len(df_demand_complete):,} lignes après complétion.")
        else:
             logging.error("Fallback Pandas: Impossible de créer l'index complet (pas d'heures ou de zones valides).")
             raise RuntimeError("Failed to create complete time series index.")

        # 4. Ingénierie de Features Temporelles
        logging.info("Fallback Pandas: Ajout des features temporelles...")
        df_demand_complete["hour"] = df_demand_complete["timestamp_hour"].dt.hour
        df_demand_complete["day_of_week"] = df_demand_complete["timestamp_hour"].dt.dayofweek # Lundi=0
        df_demand_complete["month"] = df_demand_complete["timestamp_hour"].dt.month
        df_demand_complete["day_of_year"] = df_demand_complete["timestamp_hour"].dt.dayofyear
        # Utiliser .isocalendar().week pour semaine ISO
        df_demand_complete["week_of_year"] = df_demand_complete["timestamp_hour"].dt.isocalendar().week.astype(int)
        df_demand_complete["year"] = df_demand_complete["timestamp_hour"].dt.year
        df_demand_complete["is_weekend"] = df_demand_complete["day_of_week"].isin([5, 6]).astype(int)
        # Ajouter les features sin/cos pour cohérence avec la requête SQL
        df_demand_complete['hour_sin'] = np.sin(2 * np.pi * df_demand_complete['hour'] / 24)
        df_demand_complete['hour_cos'] = np.cos(2 * np.pi * df_demand_complete['hour'] / 24)

        logging.info("Fallback Pandas: Traitement des données terminé avec succès.")

    except Exception as e:
        logging.error(f"Fallback Pandas: Erreur inattendue pendant le traitement des données: {e}")
        traceback.print_exc()
        # Lever l'erreur pour signaler l'échec du traitement
        raise RuntimeError(f"Unexpected error during fallback data processing: {e}") from e

    # --- LOGIQUE D'EXPORT ---
    df_to_export = None # **CORRECTION 1: Initialiser ici**
    try:
        logging.info("Tentative d'export via pandas DataFrame (Méthode 2)...")
        # Vérifier si le DataFrame traité existe et n'est pas vide
        if df_demand_complete is not None and not df_demand_complete.empty:
            # --- FILTRAGE FINAL (Juste avant l'export) ---
            logging.info(f"Filtrage final du DataFrame Pandas traité entre {start_timestamp_str} et {end_timestamp_str}...")
            start_dt_filter = pd.to_datetime(start_timestamp_str, utc=True)
            end_dt_filter = pd.to_datetime(end_timestamp_str, utc=True)

            # S'assurer que la colonne est bien datetime avant de filtrer
            if not pd.api.types.is_datetime64_any_dtype(df_demand_complete['timestamp_hour']):
                 df_demand_complete['timestamp_hour'] = pd.to_datetime(df_demand_complete['timestamp_hour'], utc=True)
            elif df_demand_complete['timestamp_hour'].dt.tz is None: # Localiser si pas déjà fait
                 df_demand_complete['timestamp_hour'] = df_demand_complete['timestamp_hour'].dt.tz_localize('UTC')

            df_filtered_pandas = df_demand_complete[
                (df_demand_complete['timestamp_hour'] >= start_dt_filter) &
                (df_demand_complete['timestamp_hour'] <= end_dt_filter)
            ].copy()
            logging.info(f"Nombre de lignes après filtrage final Pandas: {len(df_filtered_pandas):,}")

            if not df_filtered_pandas.empty:
                # --- Préparation pour l'export ---
                columns_to_keep = [ # S'assurer que ces colonnes existent bien dans df_filtered_pandas
                    "timestamp_hour", "pickup_community_area", "trip_count",
                    "hour", "day_of_week", "month", "year", "day_of_year",
                    "week_of_year", "is_weekend", "hour_sin", "hour_cos"
                ]
                columns_to_export_actual = [col for col in columns_to_keep if col in df_filtered_pandas.columns]
                logging.info(f"Colonnes sélectionnées pour l'export: {columns_to_export_actual}")

                # Assigner df_to_export
                df_to_export = df_filtered_pandas[columns_to_export_actual].copy() # Utiliser copy()

                # Conversion des types pour BQ
                logging.info("Conversion des types pour l'export BQ...")
                # **CORRECTION 2: Utiliser un bloc try/except autour de cette section critique**
                try:
                    # Vérifier que df_to_export est bien un DataFrame avant la boucle
                    if not isinstance(df_to_export, pd.DataFrame):
                        raise TypeError(f"df_to_export is not a DataFrame (type: {type(df_to_export)}). Cannot proceed with type conversion.")

                    for col in df_to_export.select_dtypes(include=['Int64', 'Float64', 'float64', 'int64']).columns: # Plus de types
                         # Ne pas reconvertir si déjà correct
                         if pd.api.types.is_integer_dtype(df_to_export[col]) or pd.api.types.is_float_dtype(df_to_export[col]):
                             # Convertir en float si NaN, sinon en int standard (numpy int64)
                             if df_to_export[col].isnull().any():
                                 df_to_export[col] = df_to_export[col].astype('float64')
                             elif pd.api.types.is_float_dtype(df_to_export[col]):
                                 # Tenter int64 si pas de NaN et si les floats sont des entiers
                                 if (df_to_export[col] == df_to_export[col].round()).all():
                                      df_to_export[col] = df_to_export[col].astype('int64')
                                 # Sinon, laisser en float64
                                 else:
                                     df_to_export[col] = df_to_export[col].astype('float64')
                             elif pd.api.types.is_integer_dtype(df_to_export[col]):
                                 # Si déjà entier, s'assurer que c'est int64 standard
                                 df_to_export[col] = df_to_export[col].astype('int64')

                    # Gérer pickup_community_area spécifiquement
                    if 'pickup_community_area' in df_to_export.columns:
                         if df_to_export['pickup_community_area'].isnull().any():
                              df_to_export['pickup_community_area'] = df_to_export['pickup_community_area'].astype('float64')
                         else:
                             try: # Tenter la conversion en int64, fallback en float64
                                 df_to_export['pickup_community_area'] = df_to_export['pickup_community_area'].astype('int64')
                             except (ValueError, TypeError):
                                 logging.warning("Conversion de 'pickup_community_area' en int64 échouée, tentative float64")
                                 df_to_export['pickup_community_area'] = df_to_export['pickup_community_area'].astype('float64')

                    # Assurer que timestamp_hour est bien datetime pour BQ
                    df_to_export['timestamp_hour'] = pd.to_datetime(df_to_export['timestamp_hour'])

                    logging.info("Types convertis. Extrait avant export:\n" + str(df_to_export.dtypes))

                except Exception as type_e:
                    logging.error(f"Erreur pendant la conversion des types: {type_e}")
                    # **CORRECTION 3: Lever une exception si la conversion échoue**
                    raise RuntimeError(f"Type conversion failed before export: {type_e}") from type_e


                # --- Export ---
                # Utiliser un schéma explicite est plus sûr
                schema=[
                        bigquery.SchemaField("timestamp_hour", "TIMESTAMP"),
                        bigquery.SchemaField("pickup_community_area", "INTEGER"), # Ajuster en FLOAT si nécessaire
                        bigquery.SchemaField("trip_count", "INTEGER"),
                        bigquery.SchemaField("hour", "INTEGER"),
                        bigquery.SchemaField("day_of_week", "INTEGER"),
                        bigquery.SchemaField("month", "INTEGER"),
                        bigquery.SchemaField("year", "INTEGER"),
                        bigquery.SchemaField("day_of_year", "INTEGER"),
                        bigquery.SchemaField("week_of_year", "INTEGER"),
                        bigquery.SchemaField("is_weekend", "INTEGER"),
                        bigquery.SchemaField("hour_sin", "FLOAT"),
                        bigquery.SchemaField("hour_cos", "FLOAT"),
                    ]
                # Ajuster le type de pickup_community_area si nécessaire
                for field in schema:
                    if field.name == 'pickup_community_area' and pd.api.types.is_float_dtype(df_to_export.get('pickup_community_area')):
                        field.field_type = 'FLOAT'
                        logging.info("Schéma ajusté: pickup_community_area sera FLOAT.")

                job_config = bigquery.LoadJobConfig(
                    write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
                    schema=schema, # Utiliser le schéma défini
                    # autodetect=False, # Désactivé car schéma fourni
                )
                logging.info(f"Exportation vers: {BQ_TABLE_ID}...")

                # Vérifier une dernière fois que df_to_export est un DataFrame avant l'export
                if not isinstance(df_to_export, pd.DataFrame):
                    raise RuntimeError(f"df_to_export is not a DataFrame (type: {type(df_to_export)}) just before BigQuery export call.")

                job = client.load_table_from_dataframe(
                    df_to_export, BQ_TABLE_ID, job_config=job_config
                )
                job.result()  # Attendre la fin du job
                logging.info(f"✅ Méthode 2: Table exportée vers BigQuery via pandas : {BQ_TABLE_ID}")
                logging.info("\nExtrait des données exportées via Pandas:")
                logging.info(df_to_export.head())

                # Si on arrive ici, l'export a réussi, on retourne le DataFrame traité initial
                # (df_demand_complete) car c'est lui qui contient toutes les données traitées.
                # L'appelant sait que si cette fonction retourne sans erreur, l'export a réussi.
                return df_demand_complete.sort_values(["timestamp_hour", "pickup_community_area"])

            else: # Cas où df_filtered_pandas est vide
                logging.warning("Le DataFrame est vide après filtrage final, export annulé.")
                # **CORRECTION 4: Lever une exception ici**
                raise RuntimeError("Fallback failed: DataFrame empty after final filtering, export cancelled.")
        else: # Cas où df_demand_complete est None ou vide
            logging.warning("DataFrame traité 'df_demand_complete' non trouvé ou vide. Export via Pandas impossible.")
            # **CORRECTION 5: Lever une exception ici**
            raise RuntimeError("Fallback failed: Processed DataFrame not found or empty, export impossible.")

    except Exception as e2: # Attrape TOUTE erreur durant la tentative d'export
        logging.error(f"⚠️ Échec critique lors de la tentative d'export via pandas : {e2}")
        traceback.print_exc()
        # **CORRECTION 6: Lever une exception pour signaler l'échec de l'export à l'appelant**
        raise RuntimeError(f"Fallback failed during export attempt: {e2}") from e2

# --- Bloc de test local ---
if __name__ == "__main__":
    # ... (Le bloc __main__ reste inchangé, mais il attrapera les RuntimeErrors maintenant) ...
    # Définir des valeurs d'exemple pour les arguments requis
    test_project_id = "votre-projet-gcp-test" # Remplacez par un ID de projet si nécessaire pour les tests BQ
    test_source_table = "votre-projet-gcp-test.votre_dataset.table_source_test"
    test_dataset_id = "votre_dataset_test"
    test_dest_table = "table_destination_test"
    start_ts = "2023-01-01 00:00:00"
    end_ts = "2023-01-01 23:59:59"

    print("--- Démarrage du test local ---")
    try: # Ajouter un try/except pour le test local
        # Créer un exemple de DataFrame brut
        data = {
            'trip_start_timestamp': pd.to_datetime(['2023-01-01 00:15:00', '2023-01-01 00:45:00', '2023-01-01 01:10:00', '2023-01-01 00:20:00', '2023-01-02 03:00:00'], utc=True),
            'pickup_community_area': [8.0, 8.0, 76.0, 8.0, 8.0], # Laisser en float pour tester la conversion
            'other_col': [1, 2, 3, 4, 5]
        }
        df_test_raw = pd.DataFrame(data)

        processed_df = process_data_with_pandas(
            df_raw=df_test_raw.copy(), # Passer le DataFrame de test
            start_timestamp_str=start_ts,
            end_timestamp_str=end_ts,
            PROJECT_ID=test_project_id,
            source_table=test_source_table, # Utilisé si df_raw est None
            dataset_id=test_dataset_id,
            destination_table_name=test_dest_table
        )

        if processed_df is not None: # Ne devrait pas être None si succès
            print("\n--- DataFrame traité (test local) ---")
            print(processed_df.head())
            print("\nTypes de données:")
            print(processed_df.dtypes)
        # else: # Ce cas ne devrait plus arriver, une exception serait levée avant
        #     print("\n--- Échec du traitement de l'exemple local (retour None) ---")

    except Exception as test_e:
         print(f"\n--- Échec du traitement de l'exemple local (Exception) : {test_e} ---")

    print("--- Fin du test local ---")