# Chicago Taxi GenAI Demo

Ce projet démontre l'utilisation de l'Intelligence Artificielle pour prédire les tarifs de taxi à Chicago en utilisant les services Google Cloud Platform (GCP). Il s'agit d'une solution complète qui va de l'analyse des données à la mise en production d'un modèle ML.

## Objectifs

- Analyser les données historiques des taxis de Chicago
- Développer un modèle de prédiction des tarifs
- Mettre en place un pipeline ML automatisé sur GCP
- Déployer une API de prédiction

## Structure du Projet

```
chicago-taxi-genai-demo/
│
├── README.md                # Présentation du projet, objectifs, aperçu de la solution
├── LICENSE                  # Licence du projet
├── .gitignore               # Fichiers et dossiers à ignorer
│
├── docs/                    # Documentation globale et whitepaper
│   ├── whitepaper.md        # Rédaction du whitepaper (business goal, ML pipeline, sécurité, etc.)
│   └── design_diagrams/     # Schémas d'architecture (pipeline, déploiement, etc.)
│
├── notebooks/               # Notebooks Jupyter pour l'analyse exploratoire (EDA)
│   ├── 1_EDA.ipynb          # Analyse exploratoire initiale du dataset
│   └── 2_Feature_Engineering.ipynb   # Traitement des données et ingénierie des features
│
├── src/                     # Code source pour la solution
│   ├── data_preprocessing/  # Scripts pour le pré-traitement des données
│   │   ├── bigquery_queries.sql  # Requêtes SQL pour préparer les données dans BigQuery
│   │   └── dataflow_pipeline.py   # Code Dataflow pour ingérer et transformer les données
│   │
│   ├── model_training/      # Scripts et notebooks pour l'entraînement du modèle
│   │   ├── train_model.py   # Script principal pour entraîner le modèle
│   │   └── model_pipeline.yaml  # Pipeline de déploiement
│   │
│   ├── deployment/          # Code pour déployer et servir le modèle
│   │   ├── api_server.py    # API pour faire des prédictions
│   │   └── deployment_instructions.md  # Instructions de déploiement
│   │
│   └── utils/               # Fonctions utilitaires et helpers
│       └── helpers.py
│
├── config/                  # Fichiers de configuration pour Google Cloud
│   ├── gcloud_setup.md      # Instructions de configuration du projet GCP
│   └── pipeline_config.yaml # Configurations pour les pipelines de ML
│
└── tests/                   # Tests unitaires et d'intégration
    └── test_helpers.py
```

## Prérequis

- Python 3.8+
- Google Cloud Platform account
- Accès aux services GCP suivants :
  - BigQuery
  - Vertex AI
  - Cloud Storage
  - Cloud Functions

## Installation

1. Cloner le repository :
```bash
git clone https://gitlab.avisia.fr/gvernay/chicago-taxi-genai-demo.git
cd chicago-taxi-genai-demo
```

2. Créer un environnement virtuel Python :
```bash
python -m venv venv
source venv/bin/activate  # Sur Unix/macOS
# ou
.\venv\Scripts\activate  # Sur Windows
```

3. Installer les dépendances :
```bash
pip install -r requirements.txt
```

4. Configurer les credentials GCP :
```bash
gcloud auth application-default login
```

## Utilisation

1. Lancer l'analyse exploratoire :
```bash
jupyter notebook notebooks/1_EDA.ipynb
```

2. Exécuter le pipeline de traitement des données :
```bash
python src/data_preprocessing/dataflow_pipeline.py
```

3. Entraîner le modèle :
```bash
python src/model_training/train_model.py
```

4. Déployer l'API :
```bash
python src/deployment/api_server.py
```

## Contribution

Les contributions sont les bienvenues ! N'hésitez pas à :
1. Fork le projet
2. Créer une branche pour votre feature
3. Commiter vos changements
4. Pousser vers la branche
5. Ouvrir une Pull Request

## Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

