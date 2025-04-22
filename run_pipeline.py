import yaml
import google.cloud.aiplatform as aiplatform
import argparse
import os
from datetime import datetime
from typing import Dict, Any, List, Union # Ajouté Union

# --- Configuration ---
DEFAULT_CONFIG_FILE = "config/pipeline_config.yaml"
DEFAULT_PIPELINE_JSON = "forecasting_pipeline.json" # Compiled pipeline definition

# --- Helper Function for Safe Config Access ---
def get_nested_config(cfg: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    """Safely gets a nested key from a dictionary."""
    val = cfg
    for key in keys:
        if isinstance(val, dict):
            val = val.get(key)
        else:
            # If we expect a dict but found something else, return default
            return default
        if val is None:
            # Key not found at this level, return default
            return default
    return val

# --- Function to load and validate config ---
def load_config(config_path: str) -> Dict[str, Any]:
    """Loads configuration from YAML file."""
    print(f"Loading configuration from: {config_path}")
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        if config is None:
            print(f"Error: Configuration file {config_path} is empty or invalid.")
            exit(1)
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file {config_path}: {e}")
        exit(1)

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Run Vertex AI Forecasting Pipeline")
parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_FILE, help=f"Path to config YAML (default: {DEFAULT_CONFIG_FILE})")
parser.add_argument("--pipeline_json", type=str, default=DEFAULT_PIPELINE_JSON, help=f"Path to compiled pipeline JSON (default: {DEFAULT_PIPELINE_JSON})")
parser.add_argument("--project_id", type=str, help="GCP Project ID (overrides config)")
parser.add_argument("--region", type=str, help="GCP Region (overrides config)")
parser.add_argument("--staging_bucket", type=str, help="GCS Staging Bucket URI (gs://...) (overrides config)")
parser.add_argument("--image_uri", type=str, help="Custom Docker Image URI (overrides config)")
parser.add_argument("--enable_caching", action=argparse.BooleanOptionalAction, help="Enable/Disable pipeline caching (--enable_caching or --no-enable_caching)") # Utiliser BooleanOptionalAction
parser.add_argument("--pipeline_root", type=str, help="GCS path for pipeline outputs (overrides staging_bucket for root).")
parser.add_argument("--service_account", type=str, help="Service account for pipeline execution (overrides config)")
parser.add_argument("--sync", action='store_true', help="Wait for pipeline completion (run synchronously).")

args = parser.parse_args()

# --- Load Configuration ---
config = load_config(args.config)

# --- Determine Final Parameter Values (Config + Overrides) ---
project_id = args.project_id or get_nested_config(config, ['gcp', 'project_id'])
region = args.region or get_nested_config(config, ['gcp', 'region'])
staging_bucket = args.staging_bucket or get_nested_config(config, ['gcp', 'staging_bucket'])
image_uri = args.image_uri or get_nested_config(config, ['gcp', 'custom_image_uri']) # Prioriser la clé directe si elle existe
if not image_uri: # Fallback si la clé directe n'existe pas
     image_uri = get_nested_config(config, ['training', 'worker_pool_spec', 'container_uri']) # Vérifier sous training

pipeline_root = args.pipeline_root or staging_bucket
service_account = args.service_account or get_nested_config(config, ['pipeline', 'service_account'])

# Caching: CLI > Config > Default (False)
enable_caching = args.enable_caching if args.enable_caching is not None else get_nested_config(config, ['pipeline', 'enable_caching'], False)


# --- Validate Required Parameters ---
if not all([project_id, region, staging_bucket, image_uri]):
    missing = [name for name, val in [('project_id', project_id), ('region', region), ('staging_bucket', staging_bucket), ('image_uri', image_uri)] if not val]
    print(f"Error: Missing required configuration values: {', '.join(missing)}.")
    print("Provide via args or ensure keys 'gcp.project_id', 'gcp.region', 'gcp.staging_bucket', 'gcp.custom_image_uri' exist in config.")
    exit(1)

print("--- Using Configuration ---")
print(f"Project ID:       {project_id}")
print(f"Region:           {region}")
print(f"Staging Bucket:   {staging_bucket}")
print(f"Pipeline Root:    {pipeline_root}")
print(f"Custom Image URI: {image_uri}")
print(f"Pipeline JSON:    {args.pipeline_json}")
print(f"Enable Caching:   {enable_caching}")
if service_account: print(f"Service Account:  {service_account}")
print("---------------------------")


# --- Prepare Pipeline Parameters Dictionary ---
# Map YAML config keys to pipeline function arguments
try:
    # Helper function specific for HPT params within this scope
    def get_hpt_param(param_name: str, key: str, default: Any = None) -> Any:
         # Utilise la structure définie dans le YAML: hpt_params.<nom_hp>.<min|max|scale>
        return get_nested_config(config, ['hpt_params', param_name, key], default)

    parameter_values = {
        # GCP & BQ
        "project": project_id,
        "location": region,
        "staging_bucket": staging_bucket,
        "bq_dataset_id": get_nested_config(config, ['bigquery', 'dataset_id']),
        "bq_source_table": get_nested_config(config, ['bigquery', 'source_table_id']), # Clé corrigée
        "bq_train_table_name": get_nested_config(config, ['bigquery', 'training_table_name']),

        # Data Prep
        "sql_query_path": get_nested_config(config, ['data_preprocessing', 'sql_template_path']),
        "sql_end_date_str": get_nested_config(config, ['data_preprocessing', 'time_window', 'end_date_str']),
        "sql_max_data_points": get_nested_config(config, ['data_preprocessing', 'time_window', 'max_data_points']),

        # Training
        "train_time_column": get_nested_config(config, ['training', 'time_column']),
        "train_target_column": get_nested_config(config, ['training', 'target_column']),
        "train_series_id_column": get_nested_config(config, ['training', 'series_id_column']),
        "train_end_date": config.get('train_end_date'), # Accès direct si clé top-level
        "val_end_date": config.get('val_end_date'),     # Accès direct si clé top-level

        # HPT / Custom Job General
        "hpt_display_name_prefix": get_nested_config(config, ['hpt', 'display_name_prefix'], 'xgboost-hpt'),
        "enable_hpt": config.get('enable_hpt'), # Accès direct
        "default_hyperparameters": config.get('default_hyperparameters'), # Accès direct
        "hpt_metric_tag": get_nested_config(config, ['hpt', 'metric_tag']),
        "hpt_metric_goal": get_nested_config(config, ['hpt', 'metric_goal']),
        "hpt_max_trial_count": get_nested_config(config, ['hpt', 'max_trial_count']),
        "hpt_parallel_trial_count": get_nested_config(config, ['hpt', 'parallel_trial_count']),
        "hpt_search_algorithm": get_nested_config(config, ['hpt', 'search_algorithm']),

        # Worker Spec
        "worker_machine_type": get_nested_config(config, ['training', 'worker_machine_type']),
        "worker_container_uri": image_uri, # Utiliser l'URI déterminée plus haut

        # Downstream / Prediction / Evaluation
        "gen_forecast_input_horizon_hours": get_nested_config(config, ['prediction', 'gen_forecast_input_horizon_hours']),
        "gen_forecast_input_start_time": get_nested_config(config, ['prediction', 'gen_forecast_input_start_time'], ""),
        "batch_pred_output_suffix": get_nested_config(config, ['prediction', 'batch_pred_output_suffix'], "predictions/predictions.csv"),
        "eval_output_dir_suffix": get_nested_config(config, ['evaluation', 'eval_output_dir_suffix'], "evaluation_output"),
        "activate_downstream_steps": get_nested_config(config, ['pipeline', 'activate_downstream_steps'], True),

        # HPT Simple Params (Using the helper function)
        "hpt_n_estimators_min": get_hpt_param('n_estimators', 'min'),
        "hpt_n_estimators_max": get_hpt_param('n_estimators', 'max'),
        "hpt_n_estimators_scale": get_hpt_param('n_estimators', 'scale', 'UNIT_LINEAR_SCALE'),
        "hpt_learning_rate_min": get_hpt_param('learning_rate', 'min'),
        "hpt_learning_rate_max": get_hpt_param('learning_rate', 'max'),
        "hpt_learning_rate_scale": get_hpt_param('learning_rate', 'scale', 'UNIT_LOG_SCALE'),
        "hpt_max_depth_min": get_hpt_param('max_depth', 'min'),
        "hpt_max_depth_max": get_hpt_param('max_depth', 'max'),
        "hpt_max_depth_scale": get_hpt_param('max_depth', 'scale', 'UNIT_LINEAR_SCALE'),
        "hpt_reg_lambda_min": get_hpt_param('reg_lambda', 'min'),
        "hpt_reg_lambda_max": get_hpt_param('reg_lambda', 'max'),
        "hpt_reg_lambda_scale": get_hpt_param('reg_lambda', 'scale', 'UNIT_LOG_SCALE'),
        # Ajouter ici l'accès aux autres paramètres simples si définis dans le YAML/pipeline
        # Ex: "hpt_subsample_min": get_hpt_param('subsample', 'min'),
        #     "hpt_subsample_max": get_hpt_param('subsample', 'max'),
        #     "hpt_subsample_scale": get_hpt_param('subsample', 'scale', 'UNIT_LINEAR_SCALE'),
    }

    # Enlever les clés dont la valeur est None pour éviter les erreurs
    # Le pipeline utilisera les valeurs par défaut définies dans sa signature si un paramètre manque
    parameter_values = {k: v for k, v in parameter_values.items() if v is not None}
    print("\n--- Pipeline Parameters Submitted ---")
    for k, v in parameter_values.items():
        print(f"{k}: {v}")
    print("-----------------------------------\n")


except (KeyError, TypeError) as e:
    print(f"Error: Problem accessing key in configuration file '{args.config}': {e}")
    print("Please ensure the config file structure and all required keys are present and have correct values.")
    exit(1)

# --- Initialize Vertex AI ---
print(f"Initializing Vertex AI SDK for project={project_id}, location={region}, staging={staging_bucket}")
try:
    aiplatform.init(project=project_id, location=region, staging_bucket=staging_bucket)
except Exception as e:
    print(f"Error initializing Vertex AI SDK: {e}")
    exit(1)


# --- Create and Run Pipeline Job ---
pipeline_job_name = f"forecasting-pipeline-run-{datetime.now().strftime('%Y%m%d%H%M%S')}"
print(f"Creating pipeline job: {pipeline_job_name}")

try:
    job = aiplatform.PipelineJob(
        display_name=pipeline_job_name,
        template_path=args.pipeline_json,
        pipeline_root=pipeline_root, # Utiliser le pipeline_root défini
        parameter_values=parameter_values,
        enable_caching=enable_caching,
    )

    print("Submitting pipeline job...")
    job.submit(service_account=service_account) # Utiliser submit et passer le SA

    # Lien pour suivre le job dans la console Cloud
    job_link = f"https://console.cloud.google.com/vertex-ai/locations/{region}/pipelines/runs/{job.job_id}?project={project_id}"
    print(f"Pipeline job submitted. Track progress at: {job_link}")
    print(f"Job ID: {job.job_id}")

    if args.sync:
        print("Waiting for pipeline job to complete... (sync=True)")
        job.wait()
        print("Pipeline job finished.")
        # Vous pouvez ajouter ici la logique pour vérifier le statut final si nécessaire
        # print(f"Job State: {job.state}")

except Exception as e:
    print(f"Error creating or submitting pipeline job: {e}")
    exit(1)

print("Script finished.")