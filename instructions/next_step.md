Okay, je vais adapter la structure et le contenu des fichiers dans le dossier `src` en me basant sur la logique des notebooks fournis (`notebooks.rtf`) et en intégrant l'approche de **Custom Job avec Hyperparameter Tuning (HPT)** pour l'entraînement du modèle, comme détaillé dans votre explication.

Nous allons remplacer l'approche AutoML par un entraînement personnalisé (par exemple avec XGBoost) dont les hyperparamètres seront optimisés via le service HPT de Vertex AI.

Voici la structure et le contenu adaptés pour les fichiers pertinents dans `src/` :

**1. `src/data_preprocessing/bigquery_queries.sql`**

Ce fichier contient la requête SQL clé pour préparer les données agrégées directement dans BigQuery. Il correspond à la **Méthode 1** préférée dans le Notebook 1 (Section 3.6) et inclut l'agrégation, la complétion, le filtrage temporel et l'ingénierie de features.

```sql
-- src/data_preprocessing/bigquery_queries.sql

-- Crée le schéma si nécessaire
CREATE SCHEMA IF NOT EXISTS `{{ project_id }}.{{ dataset_id }}`
OPTIONS(location="{{ location }}"); -- Utilise la localisation du projet/config

-- Crée ou remplace la table d'entraînement agrégée et filtrée
-- Les paramètres {{ project_id }}, {{ dataset_id }}, {{ destination_table }},
-- {{ start_timestamp_str }}, {{ end_timestamp_str }} seront remplacés par le script/composant KFP
CREATE OR REPLACE TABLE `{{ project_id }}.{{ dataset_id }}.{{ destination_table }}` AS
WITH
  raw_trips AS (
    -- Sélectionne les colonnes nécessaires des données brutes
    SELECT trip_start_timestamp, pickup_community_area
    FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`
    WHERE
      pickup_community_area IS NOT NULL
      -- *** FILTRAGE TEMPOREL ***
      -- Appliqué ici pour limiter la quantité de données traitées dès le début
      AND trip_start_timestamp >= TIMESTAMP("{{ start_timestamp_str }}")
      AND trip_start_timestamp <= TIMESTAMP("{{ end_timestamp_str }}")
  ),
  hours AS (
    -- Génère toutes les heures dans la plage temporelle **filtrée**
    SELECT timestamp_hour
    FROM UNNEST(GENERATE_TIMESTAMP_ARRAY(
            TIMESTAMP("{{ start_timestamp_str }}"),
            TIMESTAMP("{{ end_timestamp_str }}"),
            INTERVAL 1 HOUR)) AS timestamp_hour
  ),
  areas AS (
    -- Obtient les zones uniques à partir des données **déjà filtrées**
    SELECT DISTINCT pickup_community_area
    FROM raw_trips
  ),
  all_combinations AS (
    -- Crée la grille complète (heure x zone) pour la période et les zones concernées
    SELECT
      h.timestamp_hour,
      a.pickup_community_area
    FROM hours h
    CROSS JOIN areas a
  ),
  aggregated AS (
    -- Agrège les courses par heure et zone sur les données **filtrées**
    SELECT
      TIMESTAMP_TRUNC(trip_start_timestamp, HOUR) AS timestamp_hour,
      pickup_community_area,
      COUNT(*) AS trip_count
    FROM raw_trips -- Utilise les données déjà filtrées
    GROUP BY 1, 2
  ),
  filled AS (
    -- Joint la grille complète avec les agrégats et remplit les vides avec 0
    SELECT
      ac.timestamp_hour,
      ac.pickup_community_area,
      IFNULL(agg.trip_count, 0) AS trip_count
    FROM all_combinations ac
    LEFT JOIN aggregated agg
      ON ac.timestamp_hour = agg.timestamp_hour
     AND ac.pickup_community_area = agg.pickup_community_area
  )
-- Sélection finale avec toutes les features temporelles requises pour l'entraînement
SELECT
  f.timestamp_hour,
  f.pickup_community_area,
  f.trip_count,
  -- Features temporelles pour le modèle
  EXTRACT(HOUR FROM f.timestamp_hour) AS hour,
  EXTRACT(DAYOFWEEK FROM f.timestamp_hour) AS day_of_week, -- Dimanche=1..Samedi=7 en BQ standard SQL
  EXTRACT(MONTH FROM f.timestamp_hour) AS month,
  EXTRACT(YEAR FROM f.timestamp_hour) AS year,
  EXTRACT(DAYOFYEAR FROM f.timestamp_hour) AS day_of_year,
  CAST(EXTRACT(ISOWEEK FROM f.timestamp_hour) AS INT64) as week_of_year, # Semaine ISO
  IF(EXTRACT(DAYOFWEEK FROM f.timestamp_hour) IN (1, 7), 1, 0) AS is_weekend,
  -- Features cycliques
  SIN(2 * ACOS(-1) * EXTRACT(HOUR FROM f.timestamp_hour) / 24) AS hour_sin,
  COS(2 * ACOS(-1) * EXTRACT(HOUR FROM f.timestamp_hour) / 24) AS hour_cos
  -- Ajouter d'autres features si nécessaire (ex: holidays, weather si jointes)
FROM filled f
ORDER BY f.timestamp_hour, f.pickup_community_area; -- Tri important pour les séries temporelles
```

**2. `src/model_training/train_xgboost_hpt.py`**

Ce script réalise l'entraînement personnalisé avec XGBoost et est conçu pour être exécuté par un Vertex AI Hyperparameter Tuning Job. Il intègre `argparse` et `hypertune`.

```python
# src/model_training/train_xgboost_hpt.py

import argparse
import os
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupShuffleSplit # Pour split chronologique groupé par série
from sklearn.metrics import mean_squared_error
from google.cloud import bigquery
import hypertune # Librairie pour rapporter la métrique à Vertex AI HPT
import db_dtypes # Nécessaire pour read_gbq avec certains types BQ
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO) # Pour voir les logs dans Vertex AI Training Jobs

# Fonction pour gérer le split chronologique groupé
def time_series_split(df, time_col, group_col, train_ratio=0.85):
    """
    Effectue un split chronologique groupé par série temporelle.
    Assure que les points de validation sont postérieurs aux points d'entraînement
    pour chaque série.
    """
    logging.info(f"Performing time series split on {len(df)} rows...")
    df_sorted = df.sort_values(by=[group_col, time_col])
    
    train_dfs = []
    val_dfs = []
    
    # GroupShuffleSplit n'est pas directement chronologique, on le fait manuellement
    unique_groups = df_sorted[group_col].unique()
    
    for group_id in unique_groups:
        group_df = df_sorted[df_sorted[group_col] == group_id]
        split_idx = int(len(group_df) * train_ratio)
        
        # Ensure we have at least one sample in validation if possible
        if split_idx == len(group_df):
             split_idx -= 1 # Move one sample to validation
        if split_idx <= 0: # If only one or two samples, put one in train, rest in val
            split_idx = 1
            
        if len(group_df) > 1: # Only split if more than one sample
            train_dfs.append(group_df.iloc[:split_idx])
            val_dfs.append(group_df.iloc[split_idx:])
        elif len(group_df) == 1: # If only one sample, put it in train
             train_dfs.append(group_df)

    train_df = pd.concat(train_dfs) if train_dfs else pd.DataFrame(columns=df.columns)
    val_df = pd.concat(val_dfs) if val_dfs else pd.DataFrame(columns=df.columns)
    
    logging.info(f"Split complete: Train={len(train_df)}, Validation={len(val_df)}")
    
    # Vérification rapide de non-chevauchement temporel pour une série
    if not train_df.empty and not val_df.empty:
        sample_group_id = train_df[group_col].iloc[0]
        max_train_time = train_df[train_df[group_col] == sample_group_id][time_col].max()
        min_val_time = val_df[val_df[group_col] == sample_group_id][time_col].min()
        if pd.notna(max_train_time) and pd.notna(min_val_time):
             assert max_train_time < min_val_time, f"Time overlap detected for group {sample_group_id}!"
             
    return train_df, val_df


def main():
    parser = argparse.ArgumentParser()

    # --- Arguments STATIQUES (fournis via config/job spec) ---
    parser.add_argument('--input_table_id', type=str, required=True, help='Full BigQuery table ID (project.dataset.table)')
    parser.add_argument('--time_column', type=str, required=True, help='Name of the timestamp column')
    parser.add_argument('--target_column', type=str, required=True, help='Name of the target variable column')
    parser.add_argument('--series_id_column', type=str, required=True, help='Name of the time series identifier column')
    parser.add_argument('--feature_columns', nargs='+', type=str, required=True, help='List of feature column names')
    parser.add_argument('--train_ratio', type=float, default=0.85, help='Fraction of data for training (chronological split)')

    # --- Hyperparamètres TUNÉS (fournis par Vertex AI HPT Service) ---
    # Noms doivent correspondre aux parameter_id dans la config HPT
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--max_depth', type=int, default=5)
    parser.add_argument('--subsample', type=float, default=0.8)
    parser.add_argument('--colsample_bytree', type=float, default=0.8)
    # Ajoutez d'autres hyperparamètres tunés ici si configurés dans le YAML

    args = parser.parse_args()

    logging.info("--- Configuration Reçue ---")
    logging.info(f"Input Table: {args.input_table_id}")
    logging.info(f"Time Column: {args.time_column}")
    logging.info(f"Target Column: {args.target_column}")
    logging.info(f"Series ID Column: {args.series_id_column}")
    logging.info(f"Features: {args.feature_columns}")
    logging.info(f"Train Ratio: {args.train_ratio}")
    logging.info(f"HPT - Learning Rate: {args.learning_rate}")
    logging.info(f"HPT - N Estimators: {args.n_estimators}")
    logging.info(f"HPT - Max Depth: {args.max_depth}")
    logging.info(f"HPT - Subsample: {args.subsample}")
    logging.info(f"HPT - Colsample Bytree: {args.colsample_bytree}")
    logging.info("--------------------------")

    # --- 1. Charger les données depuis BigQuery ---
    logging.info("Chargement des données depuis BigQuery...")
    bq_client = bigquery.Client()
    # L'ID de table doit être au format project.dataset.table pour read_gbq
    # bq:// préfixe est géré par read_gbq
    query = f"SELECT * FROM `{args.input_table_id}` ORDER BY {args.series_id_column}, {args.time_column}" # Trier est crucial
    try:
        df = bq_client.query(query).to_dataframe()
        logging.info(f"Données chargées : {len(df)} lignes")
    except Exception as e:
         logging.error(f"Erreur lors du chargement depuis BigQuery: {e}")
         raise # Arrêter si les données ne peuvent pas être chargées

    # --- 2. Préparation / Feature Engineering (si nécessaire) ---
    # Assurez-vous que les types sont corrects. XGBoost gère 'category' type.
    logging.info("Préparation des données...")
    df[args.time_column] = pd.to_datetime(df[args.time_column], errors='coerce')
    df = df.dropna(subset=[args.time_column]) # Supprimer les lignes où le temps est invalide

    # Convertir les features catégorielles explicitement si nécessaire
    categorical_features = ['is_weekend', 'day_of_week', 'month', 'hour'] # Exemple
    for col in args.feature_columns:
        if col in categorical_features:
            df[col] = df[col].astype('category')
            logging.info(f"Converted {col} to category type.")
        elif col not in [args.time_column, args.series_id_column, args.target_column]:
             # Assurez-vous que les autres features sont numériques
             df[col] = pd.to_numeric(df[col], errors='coerce')
             if df[col].isnull().any():
                 logging.warning(f"NaNs introduced/present in numeric feature: {col}. Consider imputation.")
                 # Optionnel: Imputation simple (moyenne/médiane) - ATTENTION aux fuites de données
                 # median_val = df[col].median()
                 # df[col] = df[col].fillna(median_val)

    # Vérification des colonnes utilisées par le modèle
    features_for_model = [col for col in args.feature_columns if col in df.columns]
    if len(features_for_model) != len(args.feature_columns):
        logging.warning("Certaines features spécifiées n'ont pas été trouvées dans le DataFrame chargé.")
        logging.warning(f"Features demandées: {args.feature_columns}")
        logging.warning(f"Features trouvées et utilisées: {features_for_model}")
    
    if not features_for_model:
        raise ValueError("Aucune feature valide trouvée pour l'entraînement.")


    # --- 3. Split Train / Validation (Chronologique et Groupé par Série) ---
    logging.info("Split Train/Validation chronologique...")
    df_train, df_val = time_series_split(df, args.time_column, args.series_id_column, args.train_ratio)

    if df_train.empty or df_val.empty:
        logging.error(f"Split a résulté en un set vide. Train: {len(df_train)}, Val: {len(df_val)}")
        # Rapporter une très mauvaise métrique pour que HPT ignore cet essai
        hpt = hypertune.HyperTune()
        hpt.report_hyperparameter_tuning_metric(
            hyperparameter_metric_tag='rmse', metric_value=float('inf'), global_step=0
        )
        logging.info("Métrique infinie rapportée en raison d'un split invalide.")
        return # Arrêter cet essai

    X_train = df_train[features_for_model]
    y_train = df_train[args.target_column]
    X_val = df_val[features_for_model]
    y_val = df_val[args.target_column]
    logging.info(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}")

    # --- 4. Entraîner le modèle XGBoost ---
    logging.info("Entraînement du modèle XGBoost...")
    model = xgb.XGBRegressor(
        objective='reg:squarederror', # Objectif de régression
        learning_rate=args.learning_rate,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        # Ajoutez d'autres hyperparamètres ici si tunés
        random_state=42,
        enable_categorical=True, # Important pour utiliser les types 'category' de Pandas
        n_jobs=-1, # Utiliser tous les CPUs disponibles
        tree_method='hist' # Souvent plus rapide pour les grands datasets
    )

    # Utiliser early stopping basé sur le set de validation
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              eval_metric='rmse', # Métrique pour early stopping
              early_stopping_rounds=20, # Arrêter si la validation RMSE ne s'améliore pas pendant 20 tours
              verbose=False) # Mettre True pour voir la progression détaillée

    # --- 5. Évaluer sur le jeu de validation ---
    logging.info("Évaluation sur le jeu de validation...")
    y_pred_val = model.predict(X_val)
    # Assurer que les prédictions ne sont pas négatives (si la cible ne peut pas l'être)
    y_pred_val = y_pred_val.clip(min=0)
    
    rmse = mean_squared_error(y_val, y_pred_val, squared=False)
    logging.info(f"Validation RMSE: {rmse}")

    # --- 6. Rapporter la métrique pour HPT ---
    logging.info("Rapport de la métrique à Vertex AI Hyperparameter Tuning...")
    hpt = hypertune.HyperTune()
    hpt.report_hyperparameter_tuning_metric(
        hyperparameter_metric_tag='rmse', # Doit correspondre au metric_tag dans la config HPT
        metric_value=rmse,
        # global_step peut être le nombre d'itérations/arbres utilisés
        # model.best_iteration est disponible si early stopping est activé
        global_step=model.best_iteration if hasattr(model, 'best_iteration') and model.best_iteration is not None else args.n_estimators
    )
    logging.info("Métrique rapportée.")

    # --- 7. (Optionnel mais recommandé) Sauvegarder le modèle pour le meilleur trial ---
    # Vertex AI HPT définit AIP_MODEL_DIR UNIQUEMENT pour le meilleur trial si configuré.
    # On peut aussi sauvegarder chaque modèle et laisser le composant KPT choisir.
    # Pour l'instant, on loggue juste que le trial est terminé.
    # Le composant KFP `launch_hpt_job.py` récupérera les détails du meilleur trial.

    # model_dir = os.environ.get("AIP_MODEL_DIR") # Vérifier si cette variable est définie
    # if model_dir:
    #     logging.info(f"Variable AIP_MODEL_DIR trouvée: {model_dir}. Sauvegarde du modèle...")
    #     os.makedirs(model_dir, exist_ok=True)
    #     # Sauvegarder au format JSON ou binaire (ubj)
    #     model_path = os.path.join(model_dir, "model.ubj")
    #     model.save_model(model_path)
    #     logging.info(f"Modèle sauvegardé dans {model_path}")
    # else:
    #     logging.info("Variable AIP_MODEL_DIR non définie. Ce n'est probablement pas le meilleur trial (ou HPT non configuré pour exporter).")

if __name__ == '__main__':
    main()
```

**3. `src/pipelines/components/run_bq_forecasting_query.py`**

Ce composant KFP exécute la requête SQL de préparation des données, en gérant le remplacement des variables et la logique de la fenêtre temporelle.

```python
# src/pipelines/components/run_bq_forecasting_query.py

from kfp.dsl import component, Output, Artifact
from typing import Optional

@component(
    packages_to_install=["google-cloud-bigquery", "pandas", "db-dtypes"],
    base_image="python:3.9" # Ou python:3.10 si préféré
)
def run_bq_forecasting_query(
    project_id: str,
    location: str, # Région GCP pour BQ
    dataset_id: str, # Dataset BQ
    destination_table_name: str, # Nom de la table de sortie
    sql_template_path: str, # Chemin vers le fichier .sql (sera lu)
    # Paramètres pour la fenêtre temporelle
    end_date_str: str, # Ex: "2023-11-22"
    max_data_points: int, # Ex: 2950 (limite Vertex AI - marge)
    # --- Sorties ---
    destination_table_uri: Output[Artifact], # URI BQ de la table créée (format project.dataset.table)
    num_series: Output[int], # Nombre de séries temporelles distinctes
    num_timestamps: Output[int], # Nombre de points temporels par série
    actual_start_date: Output[str], # Date de début réellement utilisée
    actual_end_date: Output[str], # Date de fin réellement utilisée
) -> None: # Retourne None car les sorties sont via Output[Artifact]
    """
    Exécute une requête SQL paramétrée pour générer la table de forecasting agrégée.
    Calcule la date de début en fonction de la date de fin et du nombre max de points.
    """
    from google.cloud import bigquery
    from datetime import datetime, timedelta
    import logging
    import pandas as pd

    logging.basicConfig(level=logging.INFO)
    client = bigquery.Client(project=project_id, location=location)

    # --- Calcul de la fenêtre temporelle ---
    logging.info(f"Calcul de la fenêtre temporelle: end_date='{end_date_str}', max_points={max_data_points}")
    try:
        end_dt = datetime.strptime(end_date_str, "%Y-%m-%d")
        # Calculer la date de début pour avoir au maximum 'max_data_points' heures
        # +1 car la génération est inclusive au début et à la fin
        start_dt = end_dt - timedelta(hours=max_data_points - 1)
        _start_date_str = start_dt.strftime("%Y-%m-%d")
        _end_date_str = end_dt.strftime("%Y-%m-%d") # Conserve le format YYYY-MM-DD

        # Définir les timestamps complets pour la requête SQL (inclut toute la journée)
        start_timestamp_str = f"{_start_date_str} 00:00:00"
        end_timestamp_str = f"{_end_date_str} 23:59:59" # Fin de journée incluse

        hours_diff = int((end_dt - start_dt).total_seconds() // 3600) + 24 # Approx heures

        logging.info(f"Plage temporelle calculée: {start_timestamp_str} à {end_timestamp_str} (~{hours_diff} heures)")
        if hours_diff > 3000:
             logging.warning(f"Le nombre d'heures ({hours_diff}) semble dépasser la limite de 3000. Vérifiez max_data_points.")

    except ValueError as e:
        logging.error(f"Format de date invalide: {e}")
        raise

    # --- Préparation et Exécution de la Requête SQL ---
    table_uri = f"{project_id}.{dataset_id}.{destination_table_name}"
    logging.info(f"Préparation de la requête pour la table: {table_uri}")

    try:
        # Lire le template SQL
        with open(sql_template_path, 'r') as f:
            sql_template = f.read()

        # Remplacer les placeholders
        query = sql_template.replace("{{ project_id }}", project_id)
        query = query.replace("{{ dataset_id }}", dataset_id)
        query = query.replace("{{ destination_table }}", destination_table_name)
        query = query.replace("{{ location }}", location)
        query = query.replace("{{ start_timestamp_str }}", start_timestamp_str)
        query = query.replace("{{ end_timestamp_str }}", end_timestamp_str)

        logging.info("Exécution de la requête BigQuery...")
        job = client.query(query)
        job.result()  # Attendre la fin du job
        logging.info(f"✅ Table créée/remplacée avec succès : {table_uri}")

    except FileNotFoundError:
         logging.error(f"Fichier template SQL non trouvé : {sql_template_path}")
         raise
    except Exception as e:
        logging.error(f"Erreur lors de l'exécution de la requête BigQuery : {e}")
        logging.error(f"Requête tentée:\n{query[:1000]}...") # Log début requête
        raise

    # --- Obtention des statistiques sur la table créée ---
    logging.info("Récupération des statistiques de la table créée...")
    try:
        stats_query = f"""
        SELECT
          COUNT(DISTINCT pickup_community_area) as num_areas,
          COUNT(DISTINCT timestamp_hour) as num_timestamps_per_series,
          MIN(timestamp_hour) as min_timestamp,
          MAX(timestamp_hour) as max_timestamp,
          COUNT(*) as total_rows
        FROM `{table_uri}`
        """
        stats_job = client.query(stats_query)
        stats = next(stats_job.result())

        _num_series = stats.num_areas
        _num_timestamps = stats.num_timestamps_per_series

        logging.info(f"📊 Statistiques:")
        logging.info(f"   - Nombre de zones (séries): {_num_series}")
        logging.info(f"   - Nombre d'horodatages par série: {_num_timestamps}")
        logging.info(f"   - Période réelle: {stats.min_timestamp} à {stats.max_timestamp}")
        logging.info(f"   - Nombre total de lignes: {stats.total_rows}")
        if _num_timestamps > 3000:
            logging.warning(f"⚠️ ATTENTION: Le nombre d'horodatages ({_num_timestamps}) dépasse la limite recommandée de 3000 pour Vertex AI Forecasting.")

    except Exception as e:
        logging.error(f"Erreur lors de la récupération des statistiques: {e}")
        _num_series = -1 # Valeur d'erreur
        _num_timestamps = -1

    # --- Écriture des sorties KFP ---
    with open(destination_table_uri.path, "w") as f:
        f.write(table_uri) # Écrit l'URI BQ complet

    # Assignation directe aux sorties numériques/chaînes
    num_series.value = _num_series
    num_timestamps.value = _num_timestamps
    actual_start_date.value = _start_date_str
    actual_end_date.value = _end_date_str

    logging.info("Sorties du composant écrites.")
```

**4. `src/pipelines/components/launch_hpt_job.py`** (Nouveau Composant)

Ce composant lance le Vertex AI Hyperparameter Tuning Job.

```python
# src/pipelines/components/launch_hpt_job.py

from kfp.dsl import component, Input, Output, Artifact, Metrics, Model
from typing import Dict, List, NamedTuple

@component(
    packages_to_install=["google-cloud-aiplatform"],
    base_image="python:3.9" # Ou python:3.10
)
def launch_hpt_job(
    project: str,
    location: str,
    staging_bucket: str, # Bucket GCS pour les artefacts Vertex AI
    display_name_prefix: str, # Préfixe pour le nom du job HPT
    # --- Spécifications du Worker Pool ---
    worker_pool_spec: Dict, # Contient machine_type, replica_count, container_uri
    # --- Configuration du HPT ---
    hpt_config: Dict, # Contient metric_tag, metric_goal, max_trial_count, parallel_trial_count, search_algorithm, parameter_spec
    # --- Arguments STATIQUES pour le script d'entraînement ---
    static_args: Dict[str, str], # Dictionnaire des arguments statiques (--key=value)
    # --- Dépendances ---
    training_data_uri: Input[Artifact], # URI BQ de la table d'entraînement (sortie de run_bq_forecasting_query)
    # --- Sorties ---
    best_trial_metrics: Output[Metrics], # Métriques du meilleur essai
    best_hyperparameters: Output[Dict], # Hyperparamètres du meilleur essai
    # Optionnel: Sortie vers un modèle si le meilleur modèle est exporté par le script
    # best_model_artifact: Output[Model] = None
) -> NamedTuple("Outputs", [("best_trial_id", str), ("best_rmse", float)]): # Retourne ID et métrique du meilleur essai
    """
    Lance un Vertex AI Hyperparameter Tuning Job pour un script d'entraînement personnalisé.
    """
    from google.cloud import aiplatform
    from datetime import datetime
    import logging
    import json

    logging.basicConfig(level=logging.INFO)
    aiplatform.init(project=project, location=location, staging_bucket=staging_bucket)

    # Lire l'URI de la table d'entraînement depuis l'artefact d'entrée
    with open(training_data_uri.path, "r") as f:
        bq_table_id_full = f.read().strip() # Format project.dataset.table
    logging.info(f"Utilisation de la table d'entraînement: {bq_table_id_full}")

    # --- Préparer les arguments pour le script ---
    # Convertir le dictionnaire d'arguments statiques en liste pour l'API
    script_args_list = []
    for key, value in static_args.items():
        # Gérer la liste de features spécialement si c'est un argument 'nargs=+'
        if key == 'feature_columns' and isinstance(value, list):
            script_args_list.append(f"--{key}")
            script_args_list.extend(value)
        else:
            script_args_list.append(f"--{key}={value}")

    # Ajouter l'URI BQ comme argument pour le script
    script_args_list.append(f"--input_table_id={bq_table_id_full}")

    logging.info(f"Arguments passés au script d'entraînement: {script_args_list}")

    # --- Préparer les spécifications du worker pool ---
    # Assurer que la commande est vide si ENTRYPOINT est utilisé dans Docker
    container_spec = {
        "image_uri": worker_pool_spec["container_uri"],
        "args": script_args_list,
        "command": worker_pool_spec.get("command", []) # Souvent vide
    }

    final_worker_pool_spec = [{
        "machine_spec": {
            "machine_type": worker_pool_spec["machine_type"],
            # Ajouter accelerator_type/count si nécessaire
        },
        "replica_count": worker_pool_spec.get("replica_count", 1), # Toujours 1 pour le maître HPT
        "container_spec": container_spec
    }]
    logging.info(f"Worker Pool Spec: {final_worker_pool_spec}")


    # --- Préparer la spécification du HPT Job ---
    # Convertir parameter_spec du YAML en objet API
    study_spec_metrics = [{
        "metric_id": hpt_config["metric_tag"],
        "goal": hpt_config["metric_goal"].upper() # Assurer MAJUSCULES
    }]

    # Transformer la spec des paramètres YAML en objets ParameterSpec de l'API
    parameter_spec_list = []
    for param_name, spec in hpt_config["parameter_spec"].items():
        api_spec = {"parameter_id": spec["parameter_id"]}
        param_type = spec["type"].upper()

        if param_type == "DOUBLE":
            api_spec["double_value_spec"] = {"min_value": spec["min_value"], "max_value": spec["max_value"]}
            if "scale_type" in spec:
                 api_spec["scale_type"] = spec["scale_type"].upper()
        elif param_type == "INTEGER":
            api_spec["integer_value_spec"] = {"min_value": int(spec["min_value"]), "max_value": int(spec["max_value"])}
            if "scale_type" in spec:
                 api_spec["scale_type"] = spec["scale_type"].upper()
        elif param_type == "CATEGORICAL":
             api_spec["categorical_value_spec"] = {"values": spec["values"]}
        elif param_type == "DISCRETE":
             api_spec["discrete_value_spec"] = {"values": [float(v) for v in spec["values"]]} # Assurer float
        else:
            raise ValueError(f"Type de paramètre non supporté: {param_type}")

        parameter_spec_list.append(api_spec)


    study_spec = {
        "metrics": study_spec_metrics,
        "parameters": parameter_spec_list,
        "algorithm": hpt_config.get("search_algorithm", "ALGORITHM_UNSPECIFIED"), # Ex: RANDOM_SEARCH, GRID_SEARCH
        # Ajouter d'autres options si nécessaire (measurement_selection_type, etc.)
    }
    logging.info(f"Study Spec: {study_spec}")


    # --- Créer et Lancer le Job HPT ---
    job_display_name = f"{display_name_prefix}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    logging.info(f"Lancement du HyperparameterTuningJob: {job_display_name}")

    hpt_job = aiplatform.HyperparameterTuningJob(
        display_name=job_display_name,
        custom_job_spec={ # Spécification du CustomJob que HPT va lancer pour chaque essai
            "worker_pool_specs": final_worker_pool_spec,
            # "base_output_directory": {"output_uri_prefix": ...} # Optionnel
        },
        study_spec=study_spec,
        max_trial_count=hpt_config["max_trial_count"],
        parallel_trial_count=hpt_config["parallel_trial_count"],
        # Optionnel: Configurer pour sauvegarder le modèle du meilleur essai
        # trial_job_spec={
        #     "worker_pool_specs": final_worker_pool_spec,
        #     "base_output_directory": {
        #         "output_uri_prefix": f"{staging_bucket}/hpt_output/{job_display_name}"
        #     }
        # },
        project=project,
        location=location,
    )

    try:
        # Utiliser sync=False pour lancer et ne pas attendre ici (KFP gère l'attente)
        hpt_job.run(sync=False) # Important: KFP attendra la fin du job
        logging.info(f"HyperparameterTuningJob {hpt_job.resource_name} lancé.")
        # KFP va automatiquement attendre la fin du job avant de passer à l'étape suivante

        # --- ATTENTION: Le code ci-dessous s'exécuterait APRÈS la fin du job dans KFP ---
        # Une fois le job terminé, récupérer les résultats
        # Il est préférable de le faire dans un composant *séparé* après celui-ci,
        # mais on peut le mettre ici pour l'instant si le composant attend (sync=True non recommandé dans KFP)
        # Pour KFP, on se contente de lancer le job.

        # --- Récupérer les résultats (Idéalement dans un composant suivant) ---
        # Pour l'instant, on logue juste le nom, KFP suivra l'état
        logging.info(f"Le suivi de l'état et la récupération des résultats seront gérés par KFP.")
        # Si on mettait sync=True (non recommandé):
        # best_trial = hpt_job.trials[0] # Le premier est le meilleur après fin
        # logging.info(f"Meilleur essai trouvé: {best_trial.id}")
        # best_params = best_trial.parameters
        # best_metrics_list = best_trial.final_measurement.metrics
        #
        # # Log metrics
        # best_metrics_dict = {m.metric_id: m.value for m in best_metrics_list}
        # best_rmse_value = best_metrics_dict.get(hpt_config["metric_tag"], float('nan'))
        # logging.info(f"Meilleurs Hyperparamètres: {best_params}")
        # logging.info(f"Meilleures Métriques: {best_metrics_dict}")
        #
        # # Écrire les sorties KFP
        # best_trial_metrics.log_metric(hpt_config["metric_tag"], best_rmse_value)
        # for param in best_params:
        #      best_hyperparameters.metadata[param.parameter_id] = param.value # Stocker dans metadata
        #
        # return (best_trial.id, best_rmse_value)

        # Placeholder pour la sortie (sera remplacé par les vraies valeurs si sync=True)
        # Comme on utilise sync=False, on ne peut pas retourner les vraies valeurs ici.
        # Le composant suivant devra récupérer le job HPT par son nom/ID et extraire les résultats.
        # Pour l'instant, on retourne des placeholders.
        placeholder_id = hpt_job.resource_name # On retourne le nom du job pour le retrouver
        placeholder_rmse = -1.0

    except Exception as e:
        logging.error(f"Erreur lors du lancement ou suivi du HPT job: {e}")
        import traceback
        logging.error(traceback.format_exc())
        placeholder_id = "ERROR"
        placeholder_rmse = float('inf')
        # Faire échouer le composant KFP
        raise

    # Retourne des placeholders car sync=False. Le composant suivant doit récupérer les résultats.
    # Retourner le nom du job HPT est utile pour le composant suivant.
    from collections import namedtuple
    outputs = namedtuple("Outputs", ["best_trial_id", "best_rmse"])
    return outputs(best_trial_id=placeholder_id, best_rmse=placeholder_rmse)


```

**5. `src/pipelines/forecasting_pipeline.py`** (Adapté)

Redéfinit la pipeline KFP pour utiliser les nouveaux composants.

```python
# src/pipelines/forecasting_pipeline.py

from kfp import dsl
from kfp.dsl import Input, Output, Artifact, Model, Metrics # Import specific types
from typing import Dict, List, NamedTuple

# Import des composants (supposant qu'ils sont dans le même répertoire ou un sous-répertoire)
# Ajuster les chemins si nécessaire
from .components.run_bq_forecasting_query import run_bq_forecasting_query
from .components.launch_hpt_job import launch_hpt_job
# Importer d'autres composants si créés (register_model, batch_predict, etc.)
# from .components.register_best_model import register_best_model
# from .components.generate_prediction_input import generate_prediction_input
# from .components.run_batch_prediction import run_batch_prediction


@dsl.pipeline(
    name="custom-forecasting-hpt-pipeline",
    description="Pipeline de forecasting avec entraînement personnalisé et HPT"
)
def custom_forecasting_pipeline(
    # --- Paramètres Généraux ---
    project: str,
    location: str, # Région GCP pour Vertex AI & BQ
    staging_bucket: str, # URI GCS (gs://...) pour artefacts KFP/Vertex
    bq_dataset_id: str, # Dataset BQ pour les tables
    train_table_name: str = "demand_by_hour_train", # Nom de la table d'entraînement
    sql_template_path: str = "src/data_preprocessing/bigquery_queries.sql", # Chemin vers le template SQL

    # --- Paramètres Fenêtre Temporelle (pour BQ Query) ---
    end_date_str: str = "2023-11-22", # Date de fin pour données d'entraînement
    max_data_points: int = 2950,      # Max points/série pour entraînement

    # --- Paramètres Script Entraînement (Statiques) ---
    time_column: str = "timestamp_hour",
    target_column: str = "trip_count",
    series_id_column: str = "pickup_community_area",
    feature_columns: List[str] = [ # Liste des features à utiliser
        "hour", "day_of_week", "month", "year", "day_of_year",
        "week_of_year", "is_weekend", "hour_sin", "hour_cos"
    ],
    train_ratio: float = 0.85, # Pour le split dans le script d'entraînement

    # --- Configuration HPT (depuis YAML/dict) ---
    hpt_display_name_prefix: str = "hpt_xgboost_taxi",
    hpt_metric_tag: str = "rmse",
    hpt_metric_goal: str = "MINIMIZE",
    hpt_max_trial_count: int = 20,
    hpt_parallel_trial_count: int = 4,
    hpt_search_algorithm: str = "RANDOM_SEARCH", # Ou BAYESIAN_OPTIMIZATION
    hpt_parameter_spec: Dict = { # Doit correspondre au format attendu par launch_hpt_job
        "learning_rate": {"parameter_id": "learning_rate", "type": "DOUBLE", "scale_type": "SCALE_TYPE_LOG", "min_value": 0.001, "max_value": 0.1},
        "n_estimators": {"parameter_id": "n_estimators", "type": "INTEGER", "scale_type": "UNIT_LINEAR_SCALE", "min_value": 50, "max_value": 500},
        "max_depth": {"parameter_id": "max_depth", "type": "INTEGER", "scale_type": "UNIT_LINEAR_SCALE", "min_value": 3, "max_value": 10},
        "subsample": {"parameter_id": "subsample", "type": "DOUBLE", "scale_type": "UNIT_LINEAR_SCALE", "min_value": 0.5, "max_value": 1.0},
        "colsample_bytree": {"parameter_id": "colsample_bytree", "type": "DOUBLE", "scale_type": "UNIT_LINEAR_SCALE", "min_value": 0.5, "max_value": 1.0}
    },

    # --- Configuration Worker Pool (depuis YAML/dict) ---
    worker_machine_type: str = "n1-standard-4",
    worker_container_uri: str = "", # IMPORTANT: URI de l'image Docker sur Artifact Registry

    # --- Paramètres Prédiction Batch (Optionnel, si inclus dans pipeline) ---
    # predict_table_name: str = "forecast_input",
    # predict_output_prefix: str = "forecast_output",
    # predict_horizon: int = 24, # Horizon pour la génération des données de prédiction
):
    # Valider les entrées essentielles
    if not worker_container_uri:
        raise ValueError("L'URI du conteneur Docker (worker_container_uri) est requis.")

    # --- Étape 1: Préparer les données d'entraînement dans BigQuery ---
    prepare_data_op = run_bq_forecasting_query(
        project_id=project,
        location=location,
        dataset_id=bq_dataset_id,
        destination_table_name=train_table_name,
        sql_template_path=sql_template_path,
        end_date_str=end_date_str,
        max_data_points=max_data_points
    )

    # --- Étape 2: Lancer le Job d'Hyperparameter Tuning ---
    # Regrouper les configurations HPT et Worker Pool
    hpt_full_config = {
        "metric_tag": hpt_metric_tag,
        "metric_goal": hpt_metric_goal,
        "max_trial_count": hpt_max_trial_count,
        "parallel_trial_count": hpt_parallel_trial_count,
        "search_algorithm": hpt_search_algorithm,
        "parameter_spec": hpt_parameter_spec
    }
    worker_full_spec = {
        "machine_type": worker_machine_type,
        "container_uri": worker_container_uri,
        # replica_count est géré dans le composant HPT (fixé à 1 pour le master)
    }
    # Regrouper les arguments statiques pour le script
    static_script_args = {
        "time_column": time_column,
        "target_column": target_column,
        "series_id_column": series_id_column,
        "feature_columns": feature_columns, # Passe la liste directement
        "train_ratio": str(train_ratio) # Convertir float en str pour args
    }

    launch_hpt_op = launch_hpt_job(
        project=project,
        location=location,
        staging_bucket=staging_bucket,
        display_name_prefix=hpt_display_name_prefix,
        worker_pool_spec=worker_full_spec,
        hpt_config=hpt_full_config,
        static_args=static_script_args,
        training_data_uri=prepare_data_op.outputs["destination_table_uri"] # Utilise la sortie de l'étape 1
    )
    # Assurer que HPT ne démarre qu'après la préparation des données
    launch_hpt_op.after(prepare_data_op)

    # --- Étapes suivantes (Optionnelles, à ajouter si nécessaire) ---

    # # Étape 3: Récupérer le meilleur modèle et l'enregistrer (Nécessite un composant dédié)
    # register_model_op = register_best_model(
    #     project=project,
    #     location=location,
    #     hpt_job_name=launch_hpt_op.outputs["best_trial_id"], # Utiliser le nom du HPT job pour le retrouver
    #     hpt_metric_tag=hpt_metric_tag,
    #     model_display_name_prefix="best_taxi_model",
    #     serving_container_image_uri=worker_container_uri # Ou une image de service dédiée
    # )
    # register_model_op.after(launch_hpt_op)

    # # Étape 4: Générer les données d'entrée pour la prédiction (Nécessite un composant dédié)
    # generate_predict_input_op = generate_prediction_input(
    #     project_id=project,
    #     dataset_id=bq_dataset_id,
    #     table_id=predict_table_name,
    #     training_data_table_uri=prepare_data_op.outputs["destination_table_uri"], # Pour obtenir les zones
    #     horizon_hours=predict_horizon,
    #     feature_columns=feature_columns # Inclure les features nécessaires
    # )

    # # Étape 5: Lancer la prédiction Batch (Nécessite un composant dédié)
    # run_batch_predict_op = run_batch_prediction(
    #     project=project,
    #     location=location,
    #     model_resource_name=register_model_op.outputs["model_resource_name"],
    #     job_display_name_prefix="batch_pred_taxi",
    #     bq_input_uri=generate_predict_input_op.outputs["destination_table_uri"],
    #     bq_output_prefix=f"bq://{project}.{bq_dataset_id}.{predict_output_prefix}"
    # )
    # run_batch_predict_op.after(register_model_op, generate_predict_input_op)

```

**6. `src/generate_forecast_input.py`**

Ce script génère les données futures nécessaires à la prédiction batch. Il est largement basé sur le Notebook 3, Section 4. Il peut être exécuté en standalone ou transformé en composant KFP (`generate_prediction_input.py`).

```python
# src/generate_forecast_input.py

import pandas as pd
from datetime import datetime, timedelta
from google.cloud import bigquery
import holidays
import argparse
import logging
import yaml # Pour lire la config

logging.basicConfig(level=logging.INFO)

def load_config(config_path="config/pipeline_config.yaml"):
    """Charge la configuration depuis YAML."""
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
         logging.warning(f"Config file {config_path} not found. Trying parent dir.")
         try:
            with open(f"../{config_path}", "r") as f:
                 return yaml.safe_load(f)
         except FileNotFoundError:
            logging.error(f"Config file not found in {config_path} or parent dir.")
            return {} # Retourne un dict vide pour éviter les erreurs si appelé sans config

def get_unique_zones(client: bigquery.Client, project_id: str, dataset_id: str, training_table: str):
    """Récupère les pickup_community_area uniques depuis la table de training."""
    table_ref = f"{project_id}.{dataset_id}.{training_table}"
    query = f"""
    SELECT DISTINCT pickup_community_area
    FROM `{table_ref}`
    WHERE pickup_community_area IS NOT NULL
    ORDER BY pickup_community_area
    """
    logging.info(f"Querying unique zones from {table_ref}...")
    try:
        df = client.query(query).to_dataframe()
        logging.info(f"Found {len(df)} unique zones.")
        return df
    except Exception as e:
         logging.error(f"Error querying unique zones: {e}")
         raise

def get_future_timestamps(n_hours: int):
    """Génère une liste d'horodatages horaires futurs (UTC)."""
    # Commencer à partir de la prochaine heure arrondie UTC
    # Utiliser UTC est souvent plus sûr pour les serveurs et BQ
    now_utc = datetime.utcnow()
    start_utc = (now_utc.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1))
    logging.info(f"Generating {n_hours} future timestamps starting from {start_utc} UTC")
    return [start_utc + timedelta(hours=i) for i in range(n_hours)]

def get_chicago_holidays(start_date, end_date):
    """Renvoie les jours fériés US (pertinent pour Chicago) dans l'intervalle donné."""
    # holidays utilise les dates locales implicitement, mais pour les features, la date UTC suffit généralement
    us_holidays = holidays.US(years=range(start_date.year, end_date.year + 1))
    # state='IL' pourrait être utilisé mais holidays.US couvre les jours fédéraux majeurs.
    return set(us_holidays.keys())

def generate_forecast_input_df(unique_zones_df: pd.DataFrame, timestamps: list, feature_columns: list):
    """Génère le DataFrame d'entrée pour la prédiction batch avec les features requises."""
    if not timestamps:
         logging.warning("Timestamp list is empty. Cannot generate input.")
         return pd.DataFrame()

    start, end = timestamps[0].date(), timestamps[-1].date() # Utiliser .date() pour holidays
    holidays_set = get_chicago_holidays(start, end)
    logging.info(f"Identified {len(holidays_set)} holidays between {start} and {end}")

    rows = []
    required_features = set(feature_columns) # Features que le modèle attend

    for zone in unique_zones_df["pickup_community_area"]:
        for ts_utc in timestamps:
            feature_dict = {
                "pickup_community_area": zone,
                "timestamp_hour": ts_utc # Garder le timestamp UTC
            }
            # Générer les features temporelles requises
            # NOTE: Ces calculs utilisent l'heure UTC du timestamp
            if "hour" in required_features: feature_dict["hour"] = ts_utc.hour
            if "day_of_week" in required_features: feature_dict["day_of_week"] = ts_utc.weekday() + 1 # BQ: Sun=1..Sat=7 ; Python: Mon=0..Sun=6. Ajuster si modèle attend 1-7.
            if "month" in required_features: feature_dict["month"] = ts_utc.month
            if "year" in required_features: feature_dict["year"] = ts_utc.year
            if "day_of_year" in required_features: feature_dict["day_of_year"] = ts_utc.timetuple().tm_yday
            if "week_of_year" in required_features: feature_dict["week_of_year"] = ts_utc.isocalendar().week
            if "is_weekend" in required_features: feature_dict["is_weekend"] = 1 if ts_utc.weekday() >= 5 else 0 # Sat=5, Sun=6
            if "is_holiday" in required_features: feature_dict["is_holiday"] = 1 if ts_utc.date() in holidays_set else 0
            if "hour_sin" in required_features: feature_dict["hour_sin"] = pd.np.sin(2 * pd.np.pi * ts_utc.hour / 24)
            if "hour_cos" in required_features: feature_dict["hour_cos"] = pd.np.cos(2 * pd.np.pi * ts_utc.hour / 24)
            # Ajouter d'autres features si le modèle en attend (ex: lag, météo future - si disponible)

            rows.append(feature_dict)

    df = pd.DataFrame(rows)

    # S'assurer que toutes les colonnes requises sont présentes, même si vides
    for col in required_features:
         if col not in df.columns and col not in ["pickup_community_area", "timestamp_hour"]:
             logging.warning(f"Required feature '{col}' not generated, adding as NaN.")
             df[col] = pd.NA # Ou 0, ou une valeur par défaut appropriée

    # Réordonner les colonnes pour correspondre (optionnel mais propre)
    final_cols = ["timestamp_hour", "pickup_community_area"] + [col for col in feature_columns if col in df.columns]
    df = df[final_cols]

    return df

def write_to_bigquery(df: pd.DataFrame, client: bigquery.Client, table_id: str):
    """Écrit le dataframe dans BigQuery avec écrasement."""
    if df.empty:
        logging.warning(f"DataFrame is empty. Skipping write to BigQuery table {table_id}.")
        return

    logging.info(f"Writing {len(df)} rows to BigQuery table: {table_id}")
    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_TRUNCATE",
        autodetect=True # Ou spécifier le schéma si nécessaire
    )
    try:
        job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
        job.result()  # Attendre la fin du job
        logging.info(f"✅ Successfully wrote data to {table_id}")
    except Exception as e:
        logging.error(f"Error writing to BigQuery table {table_id}: {e}")
        raise

def parse_arguments():
    """Parse les arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(description="Generate forecast input data and upload to BigQuery.")
    parser.add_argument('--project_id', type=str, required=True, help='Google Cloud Project ID.')
    parser.add_argument('--dataset_id', type=str, required=True, help='BigQuery Dataset ID.')
    parser.add_argument('--output_table', type=str, required=True, help='Name of the BigQuery output table for forecast input.')
    parser.add_argument('--training_table', type=str, required=True, help='Name of the BigQuery training data table (to get unique zones).')
    parser.add_argument('--horizon_hours', type=int, default=24, help='Number of future hours to generate.')
    parser.add_argument('--config_path', type=str, default='config/pipeline_config.yaml', help='Path to the pipeline configuration YAML file.')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    # Charger la config pour obtenir la liste des features attendues par le modèle
    config = load_config(args.config_path)
    feature_cols = config.get('custom_training', {}).get('static_args', {}).get('feature_columns', [])
    if not feature_cols:
         logging.warning("Feature columns not found in config. Input might lack necessary features.")
         # Définir une liste par défaut si la config est absente
         feature_cols = ['hour', 'day_of_week', 'month', 'year', 'day_of_year', 'is_weekend', 'hour_sin', 'hour_cos']


    bq_client = bigquery.Client(project=args.project_id)

    try:
        # 1. Obtenir les zones uniques
        zones_df = get_unique_zones(bq_client, args.project_id, args.dataset_id, args.training_table)

        # 2. Générer les timestamps futurs
        future_ts = get_future_timestamps(args.horizon_hours)

        # 3. Générer le DataFrame d'entrée avec les features requises
        df_input = generate_forecast_input_df(zones_df, future_ts, feature_cols)

        # 4. Écrire dans BigQuery
        output_table_id = f"{args.project_id}.{args.dataset_id}.{args.output_table}"
        write_to_bigquery(df_input, bq_client, output_table_id)

        logging.info("✅ Forecast input generation completed successfully.")

    except Exception as e:
        logging.error(f"❌ Forecast input generation failed: {e}")
        import traceback
        logging.error(traceback.format_exc())
```

**7. `src/visualization.py`**

Ce fichier contient les fonctions de visualisation réutilisables basées sur celles des notebooks (plots temporels, spatiaux, heatmaps, etc.).

```python
# src/visualization.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
from typing import List, Optional

# Configuration visuelle de base (peut être surchargée)
sns.set(style="whitegrid", context="talk")
plt.rcParams['figure.figsize'] = [14, 7] # Taille par défaut

def plot_time_series_plotly(df: pd.DataFrame,
                           time_col: str,
                           value_col: str,
                           color_col: Optional[str] = None,
                           title: str = "Time Series Plot",
                           xaxis_title: str = "Time",
                           yaxis_title: str = "Value",
                           hover_name: Optional[str] = None):
    """Crée un graphique de série temporelle interactif avec Plotly."""
    if df.empty:
        logging.warning("DataFrame is empty, skipping plot.")
        return
    logging.info(f"Plotting {value_col} vs {time_col}, colored by {color_col}")
    try:
        fig = px.line(
            df,
            x=time_col,
            y=value_col,
            color=color_col,
            title=title,
            labels={time_col: xaxis_title, value_col: yaxis_title, color_col: color_col},
            hover_name=hover_name
        )
        fig.update_layout(hovermode="x unified")
        fig.show()
    except Exception as e:
        logging.error(f"Error creating Plotly time series plot: {e}")


def plot_hourly_pattern(df: pd.DataFrame,
                        hour_col: str,
                        value_col: str,
                        agg_func: str = 'mean',
                        title: str = "Average Pattern by Hour of Day"):
    """Visualise le pattern horaire moyen."""
    if df.empty:
        logging.warning("DataFrame is empty, skipping plot.")
        return
    logging.info(f"Plotting hourly pattern for {value_col} aggregated by {agg_func}")
    try:
        hourly_agg = df.groupby(hour_col)[value_col].agg(agg_func).reset_index()
        plt.figure(figsize=(14, 7))
        sns.barplot(x=hour_col, y=value_col, data=hourly_agg, palette='viridis')
        plt.title(title, fontsize=16)
        plt.xlabel('Hour of Day (0-23)', fontsize=14)
        plt.ylabel(f'{agg_func.capitalize()} {value_col}', fontsize=14)
        plt.xticks(range(0, 24, 2)) # Ticks toutes les 2 heures
        plt.grid(True, linestyle='--', alpha=0.7, axis='y')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        logging.error(f"Error plotting hourly pattern: {e}")


def plot_daily_heatmap(df: pd.DataFrame,
                       day_col: str,
                       hour_col: str,
                       value_col: str,
                       agg_func: str = 'mean',
                       day_names: List[str] = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], # Ou complets
                       title: str = "Average Pattern by Day and Hour"):
    """Crée une heatmap des valeurs moyennes par jour de la semaine et heure."""
    if df.empty:
        logging.warning("DataFrame is empty, skipping plot.")
        return
    logging.info(f"Plotting daily heatmap for {value_col} aggregated by {agg_func}")
    try:
        # Assurer que day_col correspond aux indices de day_names (0-6)
        day_hour_agg = df.groupby([day_col, hour_col])[value_col].agg(agg_func).reset_index()

        # Ajuster day_col si nécessaire (ex: si BQ donne 1-7 et day_names est 0-6)
        if day_hour_agg[day_col].min() == 1: # Assume BQ format 1(Sun)-7(Sat)
            # Convertir en Mon=0..Sun=6 pour correspondre à l'ordre python/day_names
            day_map = {2:0, 3:1, 4:2, 5:3, 6:4, 7:5, 1:6} # BQ day -> Python day index
            day_hour_agg[day_col] = day_hour_agg[day_col].map(day_map)
        elif day_hour_agg[day_col].max() > 6:
             logging.warning(f"Max value in '{day_col}' ({day_hour_agg[day_col].max()}) > 6. Heatmap labels might be misaligned.")


        day_hour_pivot = day_hour_agg.pivot(index=hour_col, columns=day_col, values=value_col)

        # Trier les colonnes (jours) selon l'ordre 0-6
        day_hour_pivot = day_hour_pivot.reindex(columns=range(len(day_names)))

        # Appliquer les noms des jours
        if len(day_names) == day_hour_pivot.shape[1]:
             day_hour_pivot.columns = day_names
        else:
             logging.warning("Length of day_names does not match number of day columns in pivot table.")


        plt.figure(figsize=(15, 10))
        sns.heatmap(
            day_hour_pivot,
            cmap='viridis',
            annot=True,
            fmt='.1f',
            linewidths=0.5,
            annot_kws={"size": 8} # Ajuster la taille des annotations
        )
        plt.title(title, fontsize=16)
        plt.xlabel('Day of Week', fontsize=14)
        plt.ylabel('Hour of Day', fontsize=14)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        logging.error(f"Error plotting daily heatmap: {e}")


def plot_top_zones(df: pd.DataFrame,
                   zone_col: str,
                   value_col: str,
                   agg_func: str = 'mean', # Ou 'sum'
                   top_n: int = 15,
                   title: str = "Top Zones by Average Value"):
    """Visualise les N zones principales par valeur agrégée."""
    if df.empty:
        logging.warning("DataFrame is empty, skipping plot.")
        return
    logging.info(f"Plotting top {top_n} zones for {value_col} aggregated by {agg_func}")
    try:
        zone_agg = df.groupby(zone_col)[value_col].agg(agg_func).reset_index()
        zone_agg = zone_agg.sort_values(value_col, ascending=False).head(top_n)

        plt.figure(figsize=(16, 8))
        sns.barplot(
            x=zone_col,
            y=value_col,
            data=zone_agg,
            palette='viridis',
            order=zone_agg[zone_col].astype(str) # Convertir en str pour éviter tri numérique
        )
        plt.title(f'Top {top_n} Zones by {agg_func.capitalize()} {value_col}', fontsize=16)
        plt.xlabel('Zone', fontsize=14)
        plt.ylabel(f'{agg_func.capitalize()} {value_col}', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, linestyle='--', alpha=0.7, axis='y')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        logging.error(f"Error plotting top zones: {e}")


def plot_prediction_vs_actual(df_pred: pd.DataFrame,
                             df_actual: pd.DataFrame,
                             time_col: str,
                             pred_col: str,
                             actual_col: str,
                             id_col: str,
                             sample_ids: List,
                             title: str = "Prediction vs Actual"):
    """Compare les prédictions aux valeurs réelles pour des séries spécifiques."""
    if df_pred.empty or df_actual.empty:
        logging.warning("Prediction or actual DataFrame is empty, skipping plot.")
        return
    logging.info(f"Plotting prediction vs actual for IDs: {sample_ids}")
    try:
        df_pred_filtered = df_pred[df_pred[id_col].isin(sample_ids)]
        df_actual_filtered = df_actual[df_actual[id_col].isin(sample_ids)]

        # Assurer que les time_col sont des datetime
        df_pred_filtered[time_col] = pd.to_datetime(df_pred_filtered[time_col])
        df_actual_filtered[time_col] = pd.to_datetime(df_actual_filtered[time_col])


        fig = make_subplots(rows=len(sample_ids), cols=1,
                           shared_xaxes=True,
                           subplot_titles=[f"Zone {id_}" for id_ in sample_ids])

        for i, id_val in enumerate(sample_ids):
            pred_series = df_pred_filtered[df_pred_filtered[id_col] == id_val]
            actual_series = df_actual_filtered[df_actual_filtered[id_col] == id_val]

            fig.add_trace(go.Scatter(x=actual_series[time_col], y=actual_series[actual_col],
                                     mode='lines', name=f'Actual (Zone {id_val})',
                                     line=dict(color='royalblue', width=2)),
                          row=i+1, col=1)

            fig.add_trace(go.Scatter(x=pred_series[time_col], y=pred_series[pred_col],
                                     mode='lines', name=f'Predicted (Zone {id_val})',
                                     line=dict(color='darkorange', dash='dash')),
                          row=i+1, col=1)

        fig.update_layout(title_text=title, height=300 * len(sample_ids), hovermode="x unified")
        fig.show()

    except Exception as e:
         logging.error(f"Error plotting prediction vs actual: {e}")

```

**Autres fichiers (`src/`)**

* **`data_processing.py`**: Peut être simplifié pour ne contenir que des fonctions utilitaires si la majorité du prétraitement est dans BQ (ex: `load_config`).
* **`utils/helpers.py`**: Conserver les fonctions utiles (`load_config`, `extract_temporal_features` si nécessaire ailleurs).
* **`pipelines/components/create_timeseries_dataset.py`**: Supprimer ou commenter (non utilisé pour Custom Jobs).
* **`pipelines/components/train_forecasting_model.py`**: Supprimer (remplacé par `launch_hpt_job.py`).
* **`model_training/train_model.py`**: Supprimer (remplacé par `train_xgboost_hpt.py`).
* **`__init__.py`**: Conserver dans `src/pipelines` et `src/pipelines/components` pour la reconnaissance des packages.

**Configuration (`config/pipeline_config.yaml`)**

Doit être restructurée comme suggéré dans votre explication pour inclure les sections `custom_training`, `hyperparameter_tuning`, et `worker_pool_spec` et supprimer la section `vertex_ai_forecast` liée à AutoML.

**Prochaines Étapes (Implémentation)**

1.  **Créer le `Dockerfile`** et `requirements.txt`.
2.  **Construire et pousser l'image Docker** vers Artifact Registry.
3.  **Mettre à jour `config/pipeline_config.yaml`** avec la nouvelle structure et l'URI de l'image Docker.
4.  **Compiler et exécuter la pipeline KFP** (`custom_forecasting_pipeline`) à partir d'un notebook ou d'un script.
5.  **Adapter le Notebook 3** pour récupérer le *meilleur modèle* enregistré par la pipeline (ou directement depuis le job HPT si non enregistré) avant de lancer la prédiction batch et la visualisation.

Cette structure et ce contenu fournissent une base solide pour implémenter le workflow de forecasting demandé en utilisant des Custom Jobs et l'Hyperparameter Tuning sur Vertex AI.