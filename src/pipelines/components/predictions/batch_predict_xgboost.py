from kfp.v2.dsl import component, Output, Dataset, Input, Model # Add Input, Model
import logging # Keep for basicConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@component(
    base_image="europe-west1-docker.pkg.dev/avisia-certification-ml-yde/chicago-taxis-demo/forecasting-pipeline:latest",
    # packages_to_install=["google-cloud-storage", "pandas", "xgboost", "joblib"] # Should be in base image
)
def batch_predict_xgboost(
    model_gcs_path: Input[Model], # Changed from str to Input[Model]
    features_gcs_path: Input[Dataset], # Changed from str to Input[Dataset]
    output_gcs_path: str, # GCS URI for the output predictions CSV
    id_col: str, # Name of the ID column in the features CSV
    time_col: str, # Name of the time column in the features CSV
    predictions: Output[Dataset] # Output artifact for predictions CSV URI
):
    """
    KFP component wrapper to run the batch prediction script.

    Invokes xgboost_batch_predict.py script within the container, passing
    necessary GCS paths and column names.

    Args:
        model_gcs_path: GCS URI of the trained model file.
        features_gcs_path: GCS URI of the input features CSV file.
        output_gcs_path: GCS URI where the predictions CSV will be saved.
        id_col: Name of the series identifier column in the features file.
        time_col: Name of the timestamp column in the features file.
        predictions: Output artifact to store the URI of the predictions CSV.
    """
    # Imports moved inside
    import subprocess
    import logging # Import again if needed inside, or use module logger

    # Path to the prediction script inside the container
    # Ensure this path is correct based on your Dockerfile WORKDIR and COPY commands
    script_path = "/app/src/pipelines/components/predictions/xgboost_batch_predict.py"

    cmd = [
        "python", script_path,
        "--model_gcs_path", model_gcs_path.uri, # Use .uri here
        "--features_gcs_path", features_gcs_path.uri, # Use .uri here
        "--output_gcs_path", output_gcs_path,
        "--id_col", id_col, # Pass id_col
        "--time_col", time_col, # Pass time_col
    ]

    logging.info(f"Running prediction script with command: {' '.join(cmd)}")
    try:
        # Run the script as a subprocess
        process = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logging.info("Prediction script stdout:")
        logging.info(process.stdout)
        logging.info("Prediction script stderr:")
        logging.info(process.stderr)
        logging.info("Prediction script finished successfully.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Prediction script failed with exit code {e.returncode}")
        logging.error("Script stdout:")
        logging.error(e.stdout)
        logging.error("Script stderr:")
        logging.error(e.stderr)
        raise RuntimeError("Batch prediction script failed.") from e
    except FileNotFoundError:
        logging.error(f"Prediction script not found at {script_path}. Check path and Docker image.")
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred while running the prediction script: {e}")
        raise

    # Set the output artifact URI
    predictions.uri = output_gcs_path
    logging.info(f"Output predictions artifact URI set to: {predictions.uri}")
