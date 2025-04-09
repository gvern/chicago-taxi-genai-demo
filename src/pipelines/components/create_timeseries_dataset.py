from kfp.dsl import component, Output, Artifact

@component(
    base_image="python:3.10",
    packages_to_install=["google-cloud-aiplatform"]
)
def create_timeseries_dataset(
    project: str,
    location: str,
    bq_source_uri: str,
    display_name: str,
    dataset_resource_name: Output[Artifact]
):
    """
    Crée un Vertex AI TimeSeriesDataset à partir d'une table BigQuery.
    """
    from google.cloud import aiplatform
    aiplatform.init(project=project, location=location)

    dataset = aiplatform.TimeSeriesDataset.create(
        display_name=display_name,
        bq_source=bq_source_uri,
    )

    # Stocke le nom du dataset dans un fichier pour qu'il soit disponible pour les étapes suivantes
    with open(dataset_resource_name.path, "w") as f:
        f.write(dataset.resource_name)

    print(f"✅ Dataset créé : {dataset.resource_name}")
