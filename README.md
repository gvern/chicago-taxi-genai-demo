# 🚖 Forecasting de la Demande de Taxis à Chicago avec Vertex AI

Ce projet propose une solution end-to-end pour prédire le nombre de trajets de taxi (`trip_count`) par **heure** et par **zone (`pickup_community_area`)** dans la ville de Chicago, en exploitant le dataset public BigQuery **Chicago Taxi Trips (187M lignes)**. Il s'appuie sur **BigQuery pour le prétraitement**, **Vertex AI Forecast pour l'entraînement AutoML**, et un déploiement via **prédiction batch**.

---

## 🎯 Objectif Métier

Optimiser la répartition des taxis sur la ville en anticipant la demande horaire par quartier, permettant :
- une meilleure couverture territoriale,
- une réduction du temps d'attente client,
- une amélioration de l'efficacité de la flotte.

---

## 🔧 Stack Technique

- **BigQuery** pour l'agrégation horaire par zone
- **Vertex AI Forecast (AutoML)** pour l'entraînement
- **GCS / BQ** pour les entrées/sorties du modèle
- **Dataflow** (optionnel) pour le prétraitement distribué
- **Vertex AI Batch Prediction** pour les prédictions futures
- **Python / SDK Vertex AI / KFP** pour l'orchestration

---

## 📁 Structure du dépôt

```
gvern-chicago-taxi-genai-demo/
├── config/
│   ├── config.yaml                      # Paramètres généraux
│   ├── gcloud_setup.md                  # Setup des ressources GCP
│   └── pipeline_config.yaml             # Paramètres de forecasting
├── docs/
│   └── whitepaper.md                    # Documentation projet
├── notebooks/
│   ├── 1_EDA.ipynb                      # Analyse exploratoire
│   ├── 2_forecasting_preparation.ipynb # Préparation des séries temporelles
│   ├── 3_Forecasting_Training.ipynb     # Entraînement modèle Vertex AI
│   └── 4_Batch_Prediction_Demo.ipynb    # Prédiction batch + visualisation
├── src/
│   ├── data_processing.py               # Fonctions de traitement pandas
│   ├── visualization.py                 # Fonctions de visualisation
│   ├── model_training/
│   │   └── train_model.py               # (obsolète) entraînement manuel
│   ├── data_preprocessing/
│   │   ├── bigquery_queries.sql         # Requêtes d'agrégation
│   │   └── dataflow_pipeline.py         # Pipeline Beam (optionnel)
│   ├── pipelines/
│   │   └── forecasting_pipeline.py      # Pipeline KFP (optionnel)

├── tests/
│   └── test_data_processing.py          # Tests unitaires
├── requirements.txt
└── README.md
```

---

## 🚀 Mise en route

### 1. Activer les API GCP
- Vertex AI
- BigQuery
- Dataflow (optionnel)
- Cloud Storage

### 2. Configurer le projet GCP
Modifier `config/gcloud_setup.md` avec :
- `project_id`
- `region`
- `bucket`
- `dataset BigQuery cible`

### 3. Préparer les données (via BigQuery)

Lancer les requêtes SQL dans `notebooks/2_forecasting_preparation.ipynb` pour créer la table :
```
[PROJECT_ID].chicago_taxis.demand_by_hour
```
contient : `timestamp_hour`, `pickup_community_area`, `trip_count`, et les features temporelles.

### 4. Entraîner le modèle Vertex AI Forecast

Utiliser le notebook :
```
notebooks/3_Forecasting_Training.ipynb
```
Il crée le dataset, lance l'entraînement AutoML Forecast, et enregistre le modèle.

### 5. Lancer la prédiction batch

Utiliser :
```
notebooks/4_Batch_Prediction_Demo.ipynb
```
Ce notebook :
- prédit le nombre de courses futures (via `model.batch_predict`)
- stocke les résultats dans BigQuery
- visualise la demande attendue par quartier

---

## 📊 Exemple de Prédiction

![forecasting](docs/img/sample_forecast.png)

---

## 📘 Documentation associée

- `docs/whitepaper.md` : Explication complète du pipeline ML, des choix d'architecture, des métriques, des résultats.
- `config/pipeline_config.yaml` : Configuration du forecasting (horizon, fenêtre, colonnes).
- `config/gcloud_setup.md` : Instructions pour la configuration de l'environnement GCP.

---

## 🧪 Tests

```bash
pytest tests/
```

---

## 📝 Licence

Ce projet est sous licence MIT. Voir `LICENSE`.

---

## 🙌 Crédits

Ce projet a été réalisé dans le cadre de la spécialisation Machine Learning de Google Cloud, démonstration #4 – Forecasting à grande échelle avec Vertex AI.
