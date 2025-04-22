# Keep KFP imports and logging config at top level
from kfp.v2.dsl import component, Input, Output, Dataset, Metrics, HTML, Artifact
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@component(
    base_image="europe-west1-docker.pkg.dev/avisia-certification-ml-yde/chicago-taxis-demo/forecasting-pipeline:latest",
    # packages_to_install should be in base image:
    # ["google-cloud-bigquery", "google-cloud-storage", "pandas", "scikit-learn", "matplotlib", "pyarrow", "db-dtypes"]
)
def evaluate_visualize_predictions(
    predictions_gcs_path: Input[Dataset], # Changed from str to Input[Dataset]
    actuals_bq_table_uri: Input[Artifact], # Changed from str to Input[Artifact]
    output_dir: str, # GCS directory path for saving outputs (e.g., gs://bucket/prefix/)
    # --- Column Name Parameters ---
    actuals_time_col: str, # Name of timestamp column in actuals BQ table
    actuals_id_col: str,   # Name of series ID column in actuals BQ table
    actuals_target_col: str, # Name of target value column in actuals BQ table
    pred_time_col: str,    # Name of timestamp column in predictions CSV
    pred_id_col: str,      # Name of series ID column in predictions CSV
    pred_value_col: str,   # Name of predicted value column in predictions CSV
    # --- KFP Outputs ---
    metrics: Output[Metrics],
    html_artifact: Output[HTML]
):
    """
    Compares predictions from a GCS CSV file to actual values from a BigQuery table.

    Calculates evaluation metrics (MAE, RMSE, MAPE) and generates a visualization
    comparing predictions and actuals for a sample series. Outputs KFP Metrics
    and an HTML artifact containing the results and plot.

    Args:
        predictions_gcs_path: GCS URI of the predictions CSV file.
        actuals_bq_table_uri: BigQuery URI of the table containing actual values.
        output_dir: GCS directory URI to save generated artifacts (like plots).
        actuals_time_col: Name of the timestamp column in the actuals BQ table.
        actuals_id_col: Name of the series ID column in the actuals BQ table.
        actuals_target_col: Name of the target value column in the actuals BQ table.
        pred_time_col: Name of the timestamp column in the predictions CSV.
        pred_id_col: Name of the series ID column in the predictions CSV.
        pred_value_col: Name of the predicted value column in the predictions CSV.
        metrics: KFP Output[Metrics] artifact.
        html_artifact: KFP Output[HTML] artifact.
    """
    # Imports moved inside
    import pandas as pd
    from google.cloud import bigquery
    import logging # Import again if needed
    import os
    import matplotlib.pyplot as plt
    import base64
    from google.cloud import storage
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    import numpy as np # Import numpy for MAPE calculation

    # Assuming visualization is available in the PYTHONPATH within the container
    try:
        from src.visualization import plot_prediction_vs_actual # Corrected import path
    except ImportError:
        logging.error("Could not import visualization function. Ensure src/visualization.py exists.")
        plot_prediction_vs_actual = None # Set to None if import fails

    # --- Validate Inputs ---
    if not predictions_gcs_path.uri.startswith("gs://"):
        raise ValueError(f"predictions_gcs_path must be a GCS URI (gs://...), got: {predictions_gcs_path.uri}") # Use .uri
    if not actuals_bq_table_uri.uri.startswith("bq://"): # Use .uri
        raise ValueError(f"actuals_bq_table_uri must start with bq://, got: {actuals_bq_table_uri.uri}") # Use .uri
    if not output_dir.startswith("gs://"):
        raise ValueError(f"output_dir must be a GCS URI (gs://...), got: {output_dir}")

    # --- 1. Load Predictions from GCS ---
    logging.info(f"Loading predictions from: {predictions_gcs_path.uri}") # Use .uri
    try:
        df_pred = pd.read_csv(predictions_gcs_path.uri) # Use .uri
        logging.info(f"Loaded {len(df_pred)} predictions.")
        # Convert time columns to datetime
        df_pred[pred_time_col] = pd.to_datetime(df_pred[pred_time_col])
    except Exception as e:
        logging.error(f"Failed to load predictions CSV from {predictions_gcs_path.uri}: {e}") # Use .uri
        raise

    # --- 2. Load Actuals from BigQuery ---
    logging.info(f"Loading actuals from BQ table: {actuals_bq_table_uri.uri}") # Use .uri
    try:
        bq_table_id = actuals_bq_table_uri.uri[5:] # Use .uri, Remove bq:// prefix
        client = bigquery.Client()
        # Construct query carefully, selecting only necessary columns and filtering by prediction time range
        min_pred_time = df_pred[pred_time_col].min()
        max_pred_time = df_pred[pred_time_col].max()
        logging.info(f"Querying actuals between {min_pred_time} and {max_pred_time}")

        query = f"""
            SELECT
                `{actuals_time_col}`,
                `{actuals_id_col}`,
                `{actuals_target_col}`
            FROM `{bq_table_id}`
            WHERE `{actuals_time_col}` >= TIMESTAMP("{min_pred_time.strftime('%Y-%m-%d %H:%M:%S')}")
              AND `{actuals_time_col}` <= TIMESTAMP("{max_pred_time.strftime('%Y-%m-%d %H:%M:%S')}")
        """
        logging.info(f"Executing BQ query: {query[:200]}...") # Log start of query
        df_actual = client.query(query).to_dataframe()
        logging.info(f"Actuals loaded from BQ. Shape: {df_actual.shape}. Columns: {df_actual.columns.tolist()}")
        # Ensure required actuals columns exist
        if actuals_time_col not in df_actual.columns: raise ValueError(f"Missing actuals time column '{actuals_time_col}'")
        if actuals_id_col not in df_actual.columns: raise ValueError(f"Missing actuals ID column '{actuals_id_col}'")
        if actuals_target_col not in df_actual.columns: raise ValueError(f"Missing actuals target column '{actuals_target_col}'")
        # Convert time column to datetime
        df_actual[actuals_time_col] = pd.to_datetime(df_actual[actuals_time_col])
    except Exception as e:
        logging.error(f"Failed to load or parse actuals from BigQuery: {e}")
        raise

    # --- Merge Predictions and Actuals ---
    logging.info("Merging predictions and actuals...")
    try:
        # Rename columns for consistent merging
        df_pred_renamed = df_pred.rename(columns={
            pred_time_col: 'timestamp',
            pred_id_col: 'series_id',
            pred_value_col: 'prediction'
        })
        df_actual_renamed = df_actual.rename(columns={
            actuals_time_col: 'timestamp',
            actuals_id_col: 'series_id',
            actuals_target_col: 'actual'
        })

        # Ensure merge keys have compatible types (already converted time, check ID type)
        if df_pred_renamed['series_id'].dtype != df_actual_renamed['series_id'].dtype:
            logging.warning(f"Merge key 'series_id' has different dtypes: Pred={df_pred_renamed['series_id'].dtype}, Actual={df_actual_renamed['series_id'].dtype}. Attempting conversion.")
            try:
                # Attempt to convert actuals ID to prediction ID type (often safer)
                df_actual_renamed['series_id'] = df_actual_renamed['series_id'].astype(df_pred_renamed['series_id'].dtype)
            except Exception as cast_e:
                logging.error(f"Failed to cast series_id columns for merging: {cast_e}")
                raise ValueError("Cannot merge due to incompatible series_id types.")


        merge_keys = ['timestamp', 'series_id']
        df_merged = pd.merge(
            df_pred_renamed[[*merge_keys, 'prediction']],
            df_actual_renamed[[*merge_keys, 'actual']],
            on=merge_keys,
            how='inner' # Use inner join to evaluate only where both exist
        )
        logging.info(f"Merged data shape: {df_merged.shape}")

        if df_merged.empty:
            logging.warning("Merged dataframe is empty. No matching data found between predictions and actuals for the specified time range.")
            # Log dummy metrics and write basic HTML
            metrics.log_metric("mae", float('nan'))
            metrics.log_metric("rmse", float('nan'))
            metrics.log_metric("mape", float('nan'))
            with open(html_artifact.path, "w") as f:
                f.write("<h2>Forecast Evaluation</h2><p>Error: No matching data found between predictions and actuals.</p>")
            logging.info("Logged NaN metrics and basic HTML due to empty merge.")
            return # Exit component gracefully

    except Exception as e:
        logging.error(f"Failed during merge operation: {e}")
        raise

    # --- Compute Metrics ---
    logging.info("Computing evaluation metrics...")
    try:
        mae = mean_absolute_error(df_merged['actual'], df_merged['prediction'])
        rmse = mean_squared_error(df_merged['actual'], df_merged['prediction'], squared=False)

        # Calculate MAPE carefully, avoiding division by zero
        mask = df_merged['actual'] != 0
        if mask.sum() > 0:
            mape = np.mean(np.abs((df_merged.loc[mask, 'actual'] - df_merged.loc[mask, 'prediction']) / df_merged.loc[mask, 'actual'])) * 100 # As percentage
        else:
            mape = float('inf') if df_merged['prediction'].abs().sum() > 0 else 0.0 # If actuals are all 0, MAPE is inf if preds are non-zero, else 0

        logging.info(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.4f}%")

        # Log metrics to KFP
        metrics.log_metric("mae", float(mae))
        metrics.log_metric("rmse", float(rmse))
        metrics.log_metric("mape", float(mape)) # Log MAPE percentage

    except Exception as e:
        logging.error(f"Failed to compute metrics: {e}")
        # Log dummy metrics on error? Or let pipeline fail? Let's log NaNs.
        metrics.log_metric("mae", float('nan'))
        metrics.log_metric("rmse", float('nan'))
        metrics.log_metric("mape", float('nan'))
        raise # Re-raise exception to signal failure

    # --- Generate Plot ---
    plot_path_gcs = None
    img_str = None
    if plot_prediction_vs_actual is not None:
        logging.info("Generating prediction vs actual plot...")
        try:
            # Use the renamed columns for plotting
            # The function expects 'pickup_community_area', 'timestamp', 'target_actual', 'prediction'
            # We need to adapt the function or pass the correct columns
            # Let's assume the function can be adapted or we pass the right df
            plot_df = df_merged.rename(columns={
                'series_id': actuals_id_col, # Rename back for the plot function if it expects original name
                'actual': 'target_actual' # Rename for the plot function
            })

            # Select a sample series ID for plotting if multiple exist
            sample_series_id = plot_df[actuals_id_col].unique()[0]
            plot_df_sample = plot_df[plot_df[actuals_id_col] == sample_series_id]
            logging.info(f"Plotting sample series ID: {sample_series_id}")

            fig = plot_prediction_vs_actual(plot_df_sample) # Pass the sample dataframe

            # Save plot locally first
            local_fig_path = "/tmp/prediction_vs_actual.png"
            os.makedirs(os.path.dirname(local_fig_path), exist_ok=True)
            fig.savefig(local_fig_path)
            plt.close(fig) # Close the figure to free memory
            logging.info(f"Plot saved locally to {local_fig_path}")

            # Upload plot to GCS output directory
            storage_client = storage.Client()
            bucket_name, blob_prefix = output_dir.replace("gs://", "").split('/', 1)
            # Ensure blob_prefix ends with / if it's a directory path
            if blob_prefix and not blob_prefix.endswith('/'):
                blob_prefix += '/'
            blob_path = f"{blob_prefix}prediction_vs_actual.png"
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_path)

            logging.info(f"Uploading plot to GCS: gs://{bucket_name}/{blob_path}")
            blob.upload_from_filename(local_fig_path)
            plot_path_gcs = f"gs://{bucket_name}/{blob_path}"
            logging.info("Plot uploaded successfully.")

            # Encode image for embedding in HTML
            with open(local_fig_path, "rb") as img_file:
                img_str = base64.b64encode(img_file.read()).decode('utf-8')

        except Exception as e:
            logging.error(f"Failed to generate or upload plot: {e}")
            # Continue without plot if it fails
    else:
        logging.warning("Plotting function not available. Skipping plot generation.")


    # --- Create HTML Artifact ---
    logging.info("Generating HTML artifact...")
    html_content = f"""
    <html>
    <head><title>Forecast Evaluation</title></head>
    <body>
    <h1>Forecast Evaluation Metrics</h1>
    <ul>
        <li>Mean Absolute Error (MAE): {mae:.4f}</li>
        <li>Root Mean Squared Error (RMSE): {rmse:.4f}</li>
        <li>Mean Absolute Percentage Error (MAPE): {mape:.4f}%</li>
    </ul>
    """

    if img_str:
        html_content += f"""
        <h2>Prediction vs Actual (Sample Series: {sample_series_id})</h2>
        <img src='data:image/png;base64,{img_str}' alt='Prediction vs Actual Plot'>
        <p>Full plot available at: <a href='{plot_path_gcs}' target='_blank'>{plot_path_gcs}</a></p>
        """
    elif plot_path_gcs: # If upload succeeded but embedding failed
         html_content += f"""
        <h2>Prediction vs Actual Plot</h2>
        <p>Plot available at: <a href='{plot_path_gcs}' target='_blank'>{plot_path_gcs}</a></p>
        """
    else:
         html_content += "<p>Plot generation failed or was skipped.</p>"


    html_content += """
    </body>
    </html>
    """

    try:
        with open(html_artifact.path, "w") as f:
            f.write(html_content)
        logging.info(f"HTML artifact saved to {html_artifact.path}")
    except Exception as e:
        logging.error(f"Failed to write HTML artifact: {e}")
        # Don't raise here, pipeline succeeded in metrics calculation

    logging.info("Evaluation component finished.")
