from typing import List
from kfp.dsl import component, Input, Output, Artifact

@component(
    base_image="python:3.10",
    packages_to_install=["google-cloud-aiplatform"]
)
def train_forecasting_model(
    project: str,
    location: str,
    display_name: str,
    dataset_resource_name: Input[Artifact],
    target_column: str,
    time_column: str,
    time_series_identifier_column: str,
    forecast_horizon: int,
    context_window: int,
    data_granularity_unit: str,
    data_granularity_count: int,
    optimization_objective: str = "minimize-rmse",
    available_at_forecast_columns: List[str] = [],
    unavailable_at_forecast_columns: List[str] = [],
    budget_milli_node_hours: int = 1000,
    model_resource_name: Output[Artifact] = None
):
    """
    Entraîne un modèle Vertex AI Forecasting.
    """
    # Importer aiplatform à l'intérieur de la fonction
    from google.cloud import aiplatform
    aiplatform.init(project=project, location=location)

    # Lire le nom du dataset depuis le fichier d'artefact
    with open(dataset_resource_name.path, "r") as f:
        dataset_name = f.read().strip()
    
    print(f"📊 Utilisation du dataset : {dataset_name}")

    job = aiplatform.AutoMLForecastingTrainingJob(
        display_name=display_name,
        optimization_objective=optimization_objective
    )

    model = job.run(
        dataset=dataset_name,  # Utiliser le nom du dataset lu depuis le fichier
        target_column=target_column,
        time_column=time_column,
        time_series_identifier_column=time_series_identifier_column,
        unavailable_at_forecast_columns=unavailable_at_forecast_columns,
        available_at_forecast_columns=available_at_forecast_columns,
        data_granularity_unit=data_granularity_unit,
        data_granularity_count=data_granularity_count,
        forecast_horizon=forecast_horizon,
        context_window=context_window,
        budget_milli_node_hours=budget_milli_node_hours,
        model_display_name=f"{display_name}_model"
    )

    # Écrire le nom du modèle dans le fichier d'artefact
    with open(model_resource_name.path, "w") as f:
        f.write(model.resource_name)
    
    print(f"✅ Modèle entraîné : {model.resource_name}")
