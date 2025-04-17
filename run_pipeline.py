from kfp import local
from kfp import dsl
from google.cloud import aiplatform
from src.pipelines.forecasting_pipeline import forecasting_pipeline
import os
import sys

# Ajouter le répertoire src au PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

# Initialize the local environment with a valid runner
local.init(runner=local.SubprocessRunner())

# === Plus de gestion de configuration ici : tout est dans forecasting_pipeline/setup ===

# Initialisation du client BigQuery si besoin (optionnel, à retirer si inutilisé ici)
# from google.cloud import bigquery
# client = bigquery.Client()  # Peut être supprimé si non utilisé dans ce fichier

# Lancement du pipeline (toute la configuration est gérée en interne)
job = forecasting_pipeline()
job.run(sync=True)

print("✅ Pipeline lancée avec succès sur Vertex AI Pipelines.")
