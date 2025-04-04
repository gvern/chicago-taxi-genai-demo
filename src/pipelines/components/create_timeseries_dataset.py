from kfp.v2.dsl import component
from google.cloud import aiplatform

@component(
    base_image="python:3.10",
    packages_to_install=["google-cloud-aiplatform"]
)
def create_timeseries_dataset(
    project: str,
    location: str,
    bq_source_uri: str,
    display_name: str,
) -> str:
    """
    Crée un Vertex AI TimeSeriesDataset à partir d'une table BigQuery.
    Args:
        project: ID du projet GCP
        location: région (ex: "us-central1")
        bq_source_uri: URI BigQuery au format bq://project.dataset.table
        display_name: nom du dataset Vertex AI
    Returns:
        dataset_resource_name: ID du dataset créé (utilisable dans les étapes suivantes)
    """
    aiplatform.init(project=project, location=location)

    dataset = aiplatform.TimeSeriesDataset.create(
        display_name=display_name,
        bq_source=bq_source_uri,
    )

    print(f"✅ Dataset créé : {dataset.resource_name}")
    return dataset.resource_name
