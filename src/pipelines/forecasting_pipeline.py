# src/pipelines/forecasting_pipeline.py

# Correction : Passer les specs HPT comme paramètres simples
import logging
import os
from kfp import dsl
from kfp.dsl import Input, Output, Artifact, Model, Metrics, HTML
from typing import Dict, List, Optional, Any

# Import components
from src.pipelines.components.data_preparation.run_bq_forecasting_query import run_bq_forecasting_query
from src.pipelines.components.model_training.launch_hpt_job import launch_hpt_job
from src.pipelines.components.generate_forecasting_data.generate_forecast_input import generate_forecast_input
from src.pipelines.components.predictions.batch_predict_xgboost import batch_predict_xgboost
from src.pipelines.components.evaluation.evaluate_visualize_predictions import evaluate_visualize_predictions

@dsl.pipeline(
    name="chicago-taxi-forecasting-pipeline-v2",
    description="Pipeline de forecasting avec entraînement XGBoost (HPT simplifié)",
    # pipeline_root=...
)
def forecasting_pipeline(
    # --- GCP Config ---
    project: str,
    location: str,
    staging_bucket: str,

    # --- BigQuery Config ---
    bq_dataset_id: str,
    bq_source_table: str,
    bq_train_table_name: str,

    # --- Data Preprocessing Config ---
    sql_query_path: str,
    sql_end_date_str: str,
    sql_max_data_points: int,

    # --- Training Config ---
    train_time_column: str,
    train_target_column: str,
    train_series_id_column: str,
    train_end_date: str,
    val_end_date: str,

    # --- HPT / Custom Job Config ---
    hpt_display_name_prefix: str = "xgboost-hpt",
    enable_hpt: bool = True,
    default_hyperparameters: Dict[str, Any] = { # Garder les défauts pour le cas non-HPT
        "n_estimators": 100, "learning_rate": 0.1, "max_depth": 5,
        "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0,
    },
    # -- Configuration générale HPT --
    hpt_metric_tag: str = "rmse",
    hpt_metric_goal: str = "MINIMIZE",
    hpt_max_trial_count: int = 10,
    hpt_parallel_trial_count: int = 2,
    hpt_search_algorithm: str = "RANDOM_SEARCH",

    # --- MODIFICATION : Supprimer hpt_parameter_spec complexe ---
    # hpt_parameter_spec: List[Dict[str, Any]] = [ ... ], # SUPPRIMÉ

    # --- MODIFICATION : Ajouter des paramètres simples pour HPT ---
    # Exemple pour n_estimators (Integer)
    hpt_n_estimators_min: int = 50,
    hpt_n_estimators_max: int = 500,
    hpt_n_estimators_scale: str = "UNIT_LINEAR_SCALE", # Ou autre si nécessaire
    # Exemple pour learning_rate (Double)
    hpt_learning_rate_min: float = 0.01,
    hpt_learning_rate_max: float = 0.3,
    hpt_learning_rate_scale: str = "UNIT_LOG_SCALE",
    # Exemple pour max_depth (Integer)
    hpt_max_depth_min: int = 3,
    hpt_max_depth_max: int = 10,
    hpt_max_depth_scale: str = "UNIT_LINEAR_SCALE",
    # Exemple pour reg_lambda (Double)
    hpt_reg_lambda_min: float = 0.1,
    hpt_reg_lambda_max: float = 10.0,
    hpt_reg_lambda_scale: str = "UNIT_LOG_SCALE",
    # Ajouter d'autres hyperparamètres à régler ici...

    # --- Worker Pool Spec ---
    worker_machine_type: str = "n1-standard-4",
    worker_container_uri: str = "europe-west1-docker.pkg.dev/avisia-certification-ml-yde/chicago-taxis-demo/forecasting-pipeline:latest",

    # --- Downstream Steps Config ---
    gen_forecast_input_horizon_hours: int = 24*7,
    gen_forecast_input_start_time: str = "",
    batch_pred_output_suffix: str = "predictions/predictions.csv",
    eval_output_dir_suffix: str = "evaluation_output",
    activate_downstream_steps: bool = True,
):
    """Defines the KFP v2 pipeline for training and evaluating an XGBoost forecaster."""

    forecast_start_time = gen_forecast_input_start_time if gen_forecast_input_start_time else val_end_date

    # --- 1. Prepare Data using BigQuery ---
    prepare_data_op = run_bq_forecasting_query(
        project_id=project, location=location, dataset_id=bq_dataset_id,
        source_table=bq_source_table, destination_table_name=bq_train_table_name,
        sql_template_path_in_container=sql_query_path, end_date_str=sql_end_date_str,
        max_data_points=sql_max_data_points
    ).set_display_name("Prepare Training Data BQ")

    # --- 2. Launch Training (HPT or Custom Job) ---
    worker_full_spec = {
        "machine_type": worker_machine_type,
        "container_uri": worker_container_uri,
    }
    # Arguments statiques SANS bq_table (lu depuis artefact)
    static_script_args_for_train = {
        "project_id": project, "location": location, "bq_dataset": bq_dataset_id,
        "train_end_date": train_end_date, "val_end_date": val_end_date,
        "target_col": train_target_column, "id_col": train_series_id_column,
        "time_col": train_time_column,
    }

    launch_hpt_op = launch_hpt_job(
        project=project,
        location=location,
        staging_bucket=staging_bucket,
        display_name_prefix=hpt_display_name_prefix,
        worker_pool_spec=worker_full_spec, # OK
        enable_hpt=enable_hpt, # OK
        # --- MODIFICATION : Supprimer hpt_config ---
        # hpt_config=hpt_full_config, # SUPPRIMÉ
        default_hyperparameters=default_hyperparameters, # OK
        bq_table_name_path=prepare_data_op.outputs["destination_table_name_out"], # OK
        static_args=static_script_args_for_train, # OK
        # --- MODIFICATION : Passer les paramètres HPT simples ---
        hpt_metric_tag=hpt_metric_tag,
        hpt_metric_goal=hpt_metric_goal,
        hpt_max_trial_count=hpt_max_trial_count,
        hpt_parallel_trial_count=hpt_parallel_trial_count,
        hpt_search_algorithm=hpt_search_algorithm,
        # Passer les bornes/échelles individuelles
        hpt_n_estimators_min=hpt_n_estimators_min,
        hpt_n_estimators_max=hpt_n_estimators_max,
        hpt_n_estimators_scale=hpt_n_estimators_scale,
        hpt_learning_rate_min=hpt_learning_rate_min,
        hpt_learning_rate_max=hpt_learning_rate_max,
        hpt_learning_rate_scale=hpt_learning_rate_scale,
        hpt_max_depth_min=hpt_max_depth_min,
        hpt_max_depth_max=hpt_max_depth_max,
        hpt_max_depth_scale=hpt_max_depth_scale,
        hpt_reg_lambda_min=hpt_reg_lambda_min,
        hpt_reg_lambda_max=hpt_reg_lambda_max,
        hpt_reg_lambda_scale=hpt_reg_lambda_scale,
        # Ajouter les autres paramètres simples ici si vous en définissez plus
    ).set_display_name("Launch Training Job (HPT/Custom)")
    launch_hpt_op.after(prepare_data_op)

    # --- Conditional Downstream Steps ---
    with dsl.Condition(activate_downstream_steps == True, name="downstream-steps"):
        # --- 3. Generate Future Features ---
        future_features_gcs_path = f"{staging_bucket}/features/future_features.csv"
        generate_future_features_op = generate_forecast_input(
            project_id=project, bq_dataset=bq_dataset_id,
            bq_table_prepared_path=prepare_data_op.outputs["destination_table_name_out"], # OK
            id_col=train_series_id_column, time_col=train_time_column,
            forecast_horizon_hours=gen_forecast_input_horizon_hours,
            forecast_start_time=forecast_start_time, output_gcs_path=future_features_gcs_path,
        ).set_display_name("Generate Future Features")
        generate_future_features_op.after(prepare_data_op)

        # --- 4. Batch Prediction --- (Reste identique)
        batch_pred_output_gcs_path = f"{staging_bucket}/{batch_pred_output_suffix}"
        batch_predict_task = batch_predict_xgboost(
            model_gcs_path=launch_hpt_op.outputs["model_gcs_path"],
            features_gcs_path=generate_future_features_op.outputs["future_features"],
            output_gcs_path=batch_pred_output_gcs_path,
            id_col=train_series_id_column, time_col=train_time_column,
        ).set_display_name("Batch Predict")
        batch_predict_task.after(launch_hpt_op)
        batch_predict_task.after(generate_future_features_op)

        # --- 5. Evaluate Predictions --- (Reste identique)
        eval_output_dir = f"{staging_bucket}/{eval_output_dir_suffix}"
        eval_step = evaluate_visualize_predictions(
            predictions_gcs_path=batch_predict_task.outputs["predictions"],
            actuals_bq_table_uri=prepare_data_op.outputs["destination_table_uri"],
            output_dir=eval_output_dir, actuals_time_col=train_time_column,
            actuals_id_col=train_series_id_column, actuals_target_col=train_target_column,
            pred_time_col=train_time_column, pred_id_col=train_series_id_column,
            pred_value_col="prediction",
        ).set_display_name("Evaluate Predictions")
        eval_step.after(batch_predict_task)