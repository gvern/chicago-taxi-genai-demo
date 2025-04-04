# 📄 Whitepaper – Forecasting de la Demande de Taxis à Chicago

## 🎯 Executive Summary

Ce projet propose une solution de prévision horaire du volume de trajets de taxi dans la ville de Chicago, à l’échelle de chaque quartier (`pickup_community_area`). Basé sur le service Vertex AI Forecast de Google Cloud, le pipeline permet d’anticiper la demande pour optimiser l’allocation des taxis, réduire les temps d’attente des clients et mieux répondre aux pics de trafic ou événements.

La solution utilise BigQuery pour l’agrégation à grande échelle du dataset public Chicago Taxi Trips (187M+ lignes), intègre les meilleures pratiques de scalabilité, de modélisation temporelle, et s'appuie sur une architecture moderne GCP de bout-en-bout.

---

## 💼 Business Goal

L’objectif métier principal est de permettre à l’opérateur de taxis de :

- **Prévoir le volume de demandes par heure et par quartier**
- **Optimiser la répartition de la flotte de véhicules**
- **Réduire les délais d’attente et les trajets à vide**
- **Mieux gérer les jours fériés, événements, et périodes de forte demande**

---

## 🔧 ML Use Case: Time Series Forecasting multi-séries

- **Type de tâche :** Prévision de séries temporelles (Forecasting)
- **Variable cible :** `trip_count` (nombre de trajets par heure)
- **Identifiant de série :** `pickup_community_area`
- **Granularité :** Horaire (`timestamp_hour`)
- **Horizon de prévision :** 24h (configurable)
- **Approche :** Forecasting multi-séries automatisé avec Vertex AI Forecast (AutoML)
- **Métriques de référence :** RMSE, MAE, MAPE (WAPE, MASE)

---

## 📊 Data Exploration & Feature Engineering

### 🔍 Exploration

- Analyse des séries temporelles sur 1 an (tendances, saisonnalité)
- Détection des zones à forte variabilité de demande
- Analyse des heures de pointe, jours fériés, jours de la semaine

### 🛠️ Feature Engineering

- Agrégation BQ par `pickup_community_area × timestamp_hour`
- Génération des séries complètes via `GENERATE_TIMESTAMP_ARRAY`
- Encodage temporel :
  - `hour`, `day_of_week`, `month`, `is_weekend`, `is_holiday`
- Intégration exogène :
  - Données météo (via API NOAA, optionnel)
  - Calendrier des événements publics (optionnel)
- Création automatique des lags et fenêtres (par Vertex AI Forecast)

---

## 🏗️ Data Pipeline & Preprocessing

### 🔗 Source

- Dataset : `bigquery-public-data.chicago_taxi_trips.taxi_trips`
- Pipeline SQL dans `bigquery_queries.sql`
- Table finale : `avisia-certification-ml-yde.chicago_taxis.demand_by_hour`

### ⚙️ Étapes du pipeline

1. Troncature à l’heure (`TIMESTAMP_TRUNC`)
2. Agrégation : `COUNT(*) AS trip_count`
3. Séries complètes via CROSS JOIN + LEFT JOIN
4. Remplissage des valeurs manquantes
5. Enrichissement temporel
6. Export possible vers GCS si besoin

---

## 🤖 Model Development – Vertex AI Forecast

### 📦 Dataset

- Création via `aiplatform.TimeSeriesDataset.create_from_bigquery()`
- Colonnes spécifiées :
  - `time_column`: `timestamp_hour`
  - `target_column`: `trip_count`
  - `time_series_identifier_column`: `pickup_community_area`
  - Features disponibles : `hour`, `day_of_week`, `month`, `is_holiday`, etc.

### 🚀 Entraînement

- Job : `AutoMLForecastingTrainingJob`
- Paramètres :
  - `forecast_horizon`: 24
  - `context_window`: 168 (1 semaine)
  - `optimization_objective`: `minimize-rmse`
  - Budget : 1h à 4h selon configuration
- Résultat : modèle déployable dans Vertex AI

### 📈 Évaluation

- Backtesting intégré
- Métriques :
  - RMSE, MAE, R², quantiles
- Visualisation des résultats dans `3_Forecasting_Training.ipynb`

---

## 🚚 Deployment – Batch Prediction Strategy

- Usage de `Vertex AI Batch Prediction`
- Données d’entrée : table BigQuery `forecast_input`
- Résultats : table `forecast_output` contenant les prédictions
- Script : `4_Batch_Prediction_Demo.ipynb`
- Visualisation :
  - Graphiques temporels par zone
  - Comparaison entre zones

---

## 🛡️ Security & Privacy

- Données publiques (pas de PII)
- Stockage GCP sécurisé :
  - Bucket GCS privé
  - Tables BQ avec IAM restreint
- Gestion des accès Vertex AI et BQ via service account
- IAM suivant le principe du moindre privilège

---

## ✅ Résultats & Impact

- Précision de prévision RMSE ~ faible sur zones denses
- Diminution anticipée des temps d’attente client
- Allocation optimale de la flotte de taxis
- Pipeline industrialisable et reproductible

---

## 📌 Architecture Résumée

BigQuery (raw) │ ├──> SQL (agrégation horaire + FE) │ ↓ BQ (demand_by_hour) │ └──> Vertex AI Forecast ├──> Training Job ├──> Model └──> Batch Prediction ↓ BQ Output + Viz
---

## 📅 Prochaines étapes

- Intégration des données météo
- Utilisation de Looker Studio pour dashboard prédictif
- Déploiement du pipeline KFP complet
- Enrichissement avec les données en temps réel (Realtime Forecast)

---

## 📂 Références & Artifacts

| Élément | Chemin |
|--------|--------|
| Données BQ | `chicago_taxis.demand_by_hour` |
| Script prétraitement | `src/data_preprocessing/bigquery_queries.sql` |
| Notebook entraînement | `notebooks/3_Forecasting_Training.ipynb` |
| Notebook prédiction | `notebooks/4_Batch_Prediction_Demo.ipynb` |
| Visualisation | `src/visualization.py` |
| Configuration | `config/pipeline_config.yaml` |

---

