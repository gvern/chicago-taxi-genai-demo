from kfp.v2 import dsl
from kfp.v2.dsl import Input, Output, Artifact, Dataset, Model, component
from google_cloud_pipeline_components import aiplatform as gcc_aip

from src.pipelines.components.run_bq_forecasting_query import run_bq_forecasting_query
from src.pipelines.components.create_timeseries_dataset import create_timeseries_dataset
from src.pipelines.components.train_forecasting_model import train_forecasting_model

@dsl.pipeline(
    name="forecasting-pipeline",
    description="Pipeline complet de forecasting taxi demand - Chicago"
)
def forecasting_pipeline(
    project: str,
    location: str,
    bq_query: str,
    bq_output_uri: str,
    dataset_display_name: str,
    forecast_model_display_name: str,
    target_column: str,
    time_column: str,
    time_series_identifier_column: str,
    forecast_horizon: int,
    context_window: int,
    data_granularity_unit: str = "hour",
    data_granularity_count: int = 1,
    optimization_objective: str = "minimize-rmse",
    available_at_forecast_columns: list = [],
    unavailable_at_forecast_columns: list = [],
    budget_milli_node_hours: int = 1000,
):
    # Étape 1 : Générer la table BigQuery finale
    query_job = run_bq_forecasting_query(
        query=bq_query,
        project=project,
        location=location
    )

    # Étape 2 : Créer le dataset TimeSeries dans Vertex AI
    dataset_creation = create_timeseries_dataset(
        project=project,
        location=location,
        display_name=dataset_display_name,
        bq_source_uri=bq_output_uri,
        time_column=time_column,
        target_column=target_column,
        time_series_identifier_column=time_series_identifier_column
    )

    # Étape 3 : Entraîner le modèle Vertex AI Forecasting
    train_model = train_forecasting_model(
        project=project,
        location=location,
        display_name=forecast_model_display_name,
        dataset_resource_name=dataset_creation.outputs["dataset_resource_name"],
        target_column=target_column,
        time_column=time_column,
        time_series_identifier_column=time_series_identifier_column,
        forecast_horizon=forecast_horizon,
        context_window=context_window,
        data_granularity_unit=data_granularity_unit,
        data_granularity_count=data_granularity_count,
        optimization_objective=optimization_objective,
        available_at_forecast_columns=available_at_forecast_columns,
        unavailable_at_forecast_columns=unavailable_at_forecast_columns,
        budget_milli_node_hours=budget_milli_node_hours
    )
