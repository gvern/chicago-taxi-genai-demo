from kfp import local
from kfp import dsl
from google.cloud import aiplatform
from src.pipelines.forecasting_pipeline import forecasting_pipeline
import yaml
import os
import sys

# Ajouter le répertoire src au PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

# Initialize the local environment with a valid runner
local.init(runner=local.SubprocessRunner())

# === 1. Charger la configuration YAML ===
with open("config/pipeline_config.yaml", "r") as f:
    config = yaml.safe_load(f)

# === 2. Paramètres GCP / Pipeline ===
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "avisia-certification-ml-yde")
REGION = "europe-west1"

PIPELINE_ROOT = f"gs://{PROJECT_ID}-vertex-bucket/pipeline_artifacts"

# Table BQ de sortie générée par la requête d'agrégation
BQ_DATASET = "chicago_taxis"
BQ_TABLE = "demand_by_hour"
BQ_URI = f"bq://{PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE}"

# Nom du job
PIPELINE_NAME = "chicago-taxi-forecasting"

# === 3. Lancer le pipeline ===
aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=PIPELINE_ROOT)

job = forecasting_pipeline(
    project=PROJECT_ID,
    location=REGION,
    bq_query="",  # le composant run_bq_forecasting_query embarque la requête
    bq_output_uri=BQ_URI,
    dataset_display_name=config["vertex_ai_forecast"]["display_name"] + "-dataset",
    forecast_model_display_name=config["vertex_ai_forecast"]["display_name"],
    target_column=config["forecasting"]["target_column"],
    time_column=config["forecasting"]["time_column"],
    time_series_identifier_column=config["forecasting"]["context_column"],
    forecast_horizon=config["forecasting"]["forecast_horizon"],
    context_window=config["forecasting"]["window_size"],
    data_granularity_unit=config["forecasting"]["data_granularity_unit"],
    data_granularity_count=1,
    optimization_objective=config["vertex_ai_forecast"]["optimization_objective"],
    available_at_forecast_columns=config["forecasting"]["available_at_forecast"],
    unavailable_at_forecast_columns=config["forecasting"]["unavailable_at_forecast"],
    budget_milli_node_hours=config["vertex_ai_forecast"]["budget_milli_node_hours"],
    column_transformations=config["vertex_ai_forecast"]["column_transformations"]
)

job.run(sync=True)

print(f"✅ Pipeline '{PIPELINE_NAME}' lancé avec succès sur Vertex AI Pipelines.")
