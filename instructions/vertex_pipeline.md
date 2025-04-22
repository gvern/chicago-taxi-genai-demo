Objectif : Adapter une pipeline Kubeflow existante pour qu'elle s'exécute sur Vertex AI Pipelines en utilisant une image Docker personnalisée hébergée sur Google Artifact Registry et en passant les paramètres via l'API Vertex AI Pipelines, au lieu de lire un fichier de configuration localement dans les composants.

Contexte : Le projet a la structure suivante (fournie par l'utilisateur) :
gvern-chicago-taxi-genai-demo/
├── README.md
├── Dockerfile           # À créer/modifier
├── LICENSE
├── Makefile
├── requirements.txt     # À créer/modifier
├── run_pipeline.py      # À créer/modifier
├── compile_pipeline.py  # À créer (optionnel, mais utile)
├── stream_bq_us_to_eu.py
├── config/
│   └── pipeline_config.yaml  # Source des paramètres
├── src/
│   ├── __init__.py
│   ├── pipelines/
│   │   ├── __init__.py
│   │   ├── forecasting_pipeline.py # À modifier LOURDEMENT
│   │   └── components/             # Tous les composants ici à modifier
│   │       ├── __init__.py
│   │       ├── data_preparation/
│   │       │   └── run_bq_forecasting_query.py
│   │       ├── evaluation/
│   │       │   └── evaluate_visualize_predictions.py
│   │       ├── generate_forecasting_data/
│   │       │   └── generate_forecast_input.py
│   │       ├── model_training/
│   │       │   ├── launch_hpt_job.py
│   │       │   └── train_xgboost_hpt.py
│   │       ├── predictions/
│   │       │   ├── batch_predict_xgboost.py
│   │       │   └── xgboost_batch_predict.py
│   │       ├── preprocessing/
│   │       │   ├── bigquery_queries.sql # Chemin à gérer
│   │       │   ├── fallback_bq.py
│   │       │   └── feature_engineering.py
│   │       └── setup/
│   │           └── setup.py # NE PLUS UTILISER CE FICHIER DANS LA PIPELINE
│   └── utils/ # Potentiellement utilisé par les composants
│   └── visualization.py # Potentiellement utilisé par les composants
└── ... (autres dossiers comme docs, notebooks, tests)

Tâches à effectuer :

1.  **Créer/Mettre à jour `requirements.txt`** à la racine :
    * Basé sur les imports dans tous les fichiers `.py` du dossier `src`, lister toutes les dépendances externes nécessaires. Inclure au minimum :
        `google-cloud-aiplatform`, `google-cloud-bigquery`, `google-cloud-storage`, `pandas`, `numpy`, `xgboost`, `scikit-learn`, `matplotlib`, `pyyaml`, `hypertune`, `kfp`, `pyarrow`, `db-dtypes`, `pandas-gbq`. Spécifier des versions si possible.

2.  **Créer/Mettre à jour `Dockerfile`** à la racine :
    * Utiliser une image de base `python:3.9-slim` ou `python:3.10-slim`.
    * Copier `requirements.txt` et l'installer via `pip install --no-cache-dir -r requirements.txt`.
    * Copier l'intégralité du dossier `src` dans `/app/src` dans l'image.
    * Définir la variable d'environnement `ENV PYTHONPATH="/app:${PYTHONPATH}"`.

3.  **Modifier LOURDEMENT `src/pipelines/forecasting_pipeline.py` :**
    * Supprimer toute importation et utilisation de `src.pipelines.components.setup.setup` (particulièrement `get_config`).
    * Modifier la signature de la fonction `forecasting_pipeline` pour qu'elle accepte **tous** les paramètres actuellement lus depuis `pipeline_config.yaml` comme arguments de fonction avec des annotations de type (ex: `project: str`, `location: str`, `bq_dataset_id: str`, `hpt_config: dict`, `worker_pool_spec: dict`, etc.).
    * À l'intérieur de la fonction `forecasting_pipeline`, passer explicitement ces arguments aux composants KFP correspondants lors de leur appel. Ne plus lire depuis une variable `config`.

4.  **Modifier TOUS les composants KFP** dans `src/pipelines/components/` :
    * Pour chaque fichier contenant `@component` (ex: `run_bq_forecasting_query.py`, `launch_hpt_job.py`, `evaluate_visualize_predictions.py`, etc.) :
        * Dans le décorateur `@component`, remplacer l'argument `base_image` par l'URL de l'image Docker personnalisée qui sera construite (utiliser un placeholder clair comme `base_image="YOUR_CUSTOM_IMAGE_URI"` pour l'instant).
        * Dans le décorateur `@component`, **supprimer** l'argument `packages_to_install`.
    * **Spécifiquement pour `run_bq_forecasting_query.py`:**
        * Le paramètre `sql_template_path` ne peut plus être un chemin local relatif au projet *avant* la construction de l'image. Gérer cela de l'une des façons suivantes (préférer l'option a ou b) :
            a) Modifier le composant pour accepter le *contenu* de la requête SQL comme un argument `str` (`sql_query_content: str`) au lieu d'un chemin. La pipeline lira le fichier SQL *avant* d'appeler le composant et passera son contenu.
            b) Ou, si on garde le chemin, s'assurer que le chemin utilisé *dans* le composant correspond au chemin *à l'intérieur* de l'image Docker (ex: `/app/src/pipelines/components/preprocessing/bigquery_queries.sql`) et que ce fichier est bien copié par le `Dockerfile`.

5.  **(Optionnel mais recommandé) Créer `compile_pipeline.py`** à la racine :
    * Ce script importera la fonction `forecasting_pipeline` modifiée.
    * Utilisera `kfp.compiler.Compiler().compile()` pour générer un fichier `forecasting_pipeline.json`.

6.  **Créer `run_pipeline.py`** à la racine :
    * Importer `google.cloud.aiplatform` et `yaml`.
    * Définir les constantes `PROJECT_ID`, `REGION`, `PIPELINE_ROOT` (sur GCS).
    * Définir le chemin vers le fichier JSON compilé (`forecasting_pipeline.json`).
    * Lire le fichier `config/pipeline_config.yaml` pour obtenir les valeurs des paramètres.
    * Créer un dictionnaire `parameter_values` qui mappe les noms des arguments de la fonction `forecasting_pipeline` (définis à l'étape 3) à leurs valeurs lues depuis le fichier YAML.
    * Initialiser `aiplatform.init(...)`.
    * Créer un `aiplatform.PipelineJob` en lui passant le `template_path` (JSON compilé), `pipeline_root`, et le dictionnaire `parameter_values`.
    * Lancer le job avec `pipeline_job.run()`.

Merci d'appliquer ces modifications de manière cohérente sur l'ensemble des fichiers concernés. Utiliser les imports relatifs corrects maintenant que tout `src` est dans `PYTHONPATH` dans le conteneur (ex: `from src.pipelines.components.preprocessing.feature_engineering import ...`). Remplacer les placeholders comme `YOUR_CUSTOM_IMAGE_URI`.