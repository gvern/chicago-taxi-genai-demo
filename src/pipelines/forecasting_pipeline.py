# src/pipelines/forecasting_pipeline.py

from kfp import dsl
from kfp.dsl import Input, Output, Artifact, Model, Metrics # Import specific types
from kfp.v2.dsl import component, Output, Dataset, Model
from typing import Dict, List, NamedTuple
import yaml

# Remplacer le chargement direct du YAML par l'import du setup
from src.pipelines.components.setup.setup import get_config

config = get_config()

# Import des composants
from src.pipelines.components.data_preparation.run_bq_forecasting_query import run_bq_forecasting_query
from src.pipelines.components.model_training.launch_hpt_job import launch_hpt_job
from src.pipelines.components.generate_forecasting_data.generate_forecast_input import generate_forecast_input
from src.pipelines.components.predictions.batch_predict_xgboost import batch_predict_xgboost
from src.pipelines.components.evaluation.evaluate_visualize_predictions import evaluate_visualize_predictions

@dsl.pipeline(
    name="custom-forecasting-hpt-pipeline",
    description="Pipeline de forecasting avec entraînement personnalisé et HPT"
)
def forecasting_pipeline():
    # Use config values
    project = config["project"]
    location = config["location"]
    staging_bucket = config["staging_bucket"]
    bq_dataset_id = config["bq_dataset_id"]
    source_table = config["source_table_id"]
    train_table_name = config["train_table_name"]
    sql_template_path = config["sql_template_path"]
    end_date_str = config["end_date_str"]
    max_data_points = config["max_data_points"]
    time_column = config["time_column"]
    target_column = config["target_column"]
    series_id_column = config["series_id_column"]
    feature_columns = config["feature_columns"]
    train_ratio = config["train_ratio"]
    hpt_display_name_prefix = config["hpt_display_name_prefix"]
    hpt_metric_tag = config["hpt_metric_tag"]
    hpt_metric_goal = config["hpt_metric_goal"]
    hpt_max_trial_count = config["hpt_max_trial_count"]
    hpt_parallel_trial_count = config["hpt_parallel_trial_count"]
    hpt_search_algorithm = config["hpt_search_algorithm"]
    hpt_parameter_spec = config["hpt_parameter_spec"]
    worker_machine_type = config["worker_machine_type"]
    worker_container_uri = config["worker_container_uri"]
    forecast_horizon_hours = config["forecast_horizon_hours"]
    forecast_start_time = config["forecast_start_time"]
    id_col = config["id_col"]
    time_col = config["time_col"]

    prepare_data_op = run_bq_forecasting_query(
        project_id=project,
        location=location,
        dataset_id=bq_dataset_id,
        source_table=source_table,
        destination_table_name=train_table_name,
        sql_template_path=sql_template_path,
        end_date_str=end_date_str,
        max_data_points=max_data_points
    )

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
    }
    static_script_args = {
        "time_column": time_column,
        "target_column": target_column,
        "series_id_column": series_id_column,
        "feature_columns": feature_columns,
        "train_ratio": str(train_ratio)
    }

    launch_hpt_op = launch_hpt_job(
        project=project,
        location=location,
        staging_bucket=staging_bucket,
        display_name_prefix=hpt_display_name_prefix,
        worker_pool_spec=worker_full_spec,
        hpt_config=hpt_full_config,
        static_args=static_script_args,
        training_data_uri=prepare_data_op.outputs["destination_table_uri"]
    )
    launch_hpt_op.after(prepare_data_op)

    future_features_gcs_path = f"{staging_bucket}/future_features.csv"
    generate_future_features_op = generate_forecast_input(
        project_id=project,
        bq_dataset=bq_dataset_id,
        bq_table_prepared=train_table_name,
        id_col=id_col,
        time_col=time_col,
        forecast_horizon_hours=forecast_horizon_hours,
        forecast_start_time=forecast_start_time,
        output_gcs_path=future_features_gcs_path,
    )
    generate_future_features_op.after(prepare_data_op)

    model_gcs_path = launch_hpt_op.outputs["model_gcs_path"]
    features_gcs_path = generate_future_features_op.outputs["future_features"]
    output_gcs_path = f"{staging_bucket}/predictions.csv"

    batch_predict_task = batch_predict_xgboost(
        model_gcs_path=model_gcs_path,
        features_gcs_path=features_gcs_path,
        output_gcs_path=output_gcs_path,
    )
    batch_predict_task.after(launch_hpt_op, generate_future_features_op)

    actuals_bq_table_uri = f"bq://{project}.{bq_dataset_id}.{train_table_name}"
    eval_output_dir = f"{staging_bucket}/evaluation_output"

    eval_step = evaluate_visualize_predictions(
        predictions_gcs_path=batch_predict_task.outputs['output_gcs_path'],
        actuals_bq_table_uri=actuals_bq_table_uri,
        output_dir=eval_output_dir
    )
    eval_step.after(batch_predict_task)