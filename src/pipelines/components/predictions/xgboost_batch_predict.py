import argparse
import pandas as pd
# import joblib # Use joblib if model saved with it
import os
import xgboost as xgb
from google.cloud import storage
import logging # Add logging
import joblib # Use joblib to load model

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def download_from_gcs(gcs_path: str, local_path: str):
    """Downloads a file from GCS to a local path."""
    logging.info(f"Downloading {gcs_path} to {local_path}...")
    try:
        client = storage.Client()
        bucket_name, blob_path = gcs_path.replace("gs://", "").split("/", 1)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True) # Ensure local dir exists
        blob.download_to_filename(local_path)
        logging.info("Download complete.")
    except Exception as e:
        logging.error(f"Failed to download {gcs_path}: {e}")
        raise

def upload_to_gcs(local_path: str, gcs_path: str):
    """Uploads a local file to GCS."""
    logging.info(f"Uploading {local_path} to {gcs_path}...")
    try:
        client = storage.Client()
        bucket_name, blob_path = gcs_path.replace("gs://", "").split("/", 1)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        blob.upload_from_filename(local_path)
        logging.info("Upload complete.")
    except Exception as e:
        logging.error(f"Failed to upload {local_path} to {gcs_path}: {e}")
        raise

def batch_predict_xgboost(model_gcs_path: str, features_gcs_path: str, output_gcs_path: str, id_col: str, time_col: str):
    """
    Loads model and features, performs batch prediction, and saves results.

    Args:
        model_gcs_path: GCS path to the trained model file (e.g., model.joblib).
        features_gcs_path: GCS path to the CSV file containing future features.
        output_gcs_path: GCS path to save the predictions CSV file.
        id_col: Name of the series identifier column in the features file.
        time_col: Name of the timestamp column in the features file.
    """
    local_model_path = "/tmp/model.joblib" # Assuming joblib format
    local_features_path = "/tmp/future_features.csv"
    local_output_path = "/tmp/predictions.csv"

    # Download model and features
    download_from_gcs(model_gcs_path, local_model_path)
    download_from_gcs(features_gcs_path, local_features_path)

    # Load model
    logging.info(f"Loading model from {local_model_path}...")
    try:
        # model = xgb.XGBRegressor() # Create instance first
        # model.load_model(local_model_path) # Use this if saved with model.save_model('model.xgb')
        model = joblib.load(local_model_path) # Use this if saved with joblib.dump
        logging.info("Model loaded successfully.")
        # Log expected features if possible (difficult with joblib, easier with native XGBoost)
        # logging.info(f"Model expected features: {model.feature_names_in_}") # Only works if trained with DataFrame
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        raise

    # Load features
    logging.info(f"Loading features from {local_features_path}...")
    try:
        df_features = pd.read_csv(local_features_path)
        logging.info(f"Features loaded. Shape: {df_features.shape}")
        logging.info(f"Feature columns: {df_features.columns.tolist()}")
    except Exception as e:
        logging.error(f"Failed to load features CSV: {e}")
        raise

    # Prepare features for prediction
    # Ensure columns match the training features (excluding target, id, time)
    # The feature engineering steps should guarantee this.
    identifier_cols = [id_col, time_col]
    feature_cols_for_pred = [col for col in df_features.columns if col not in identifier_cols]

    # Check if model has feature names and if they match
    try:
         # This might fail if the model wasn't trained with feature names (e.g., older XGBoost or numpy input)
         # For XGBoost >= 1.0 trained with DataFrame, feature_names_in_ should exist
         model_feature_names = model.feature_names_in_
         if model_feature_names is not None:
              logging.info(f"Model expected feature names: {model_feature_names}")
              if set(feature_cols_for_pred) != set(model_feature_names):
                   logging.warning("Feature names in input CSV do not exactly match model's expected feature names!")
                   logging.warning(f"CSV features: {sorted(feature_cols_for_pred)}")
                   logging.warning(f"Model expects: {sorted(model_feature_names)}")
                   # Attempt to reorder CSV columns to match model, if names are just different order
                   try:
                       df_features_ordered = df_features[model_feature_names] # Select and order
                       logging.info("Reordered input columns to match model.")
                   except KeyError as ke:
                       missing_in_csv = set(model_feature_names) - set(feature_cols_for_pred)
                       extra_in_csv = set(feature_cols_for_pred) - set(model_feature_names)
                       logging.error(f"Cannot reorder columns. Missing in CSV: {missing_in_csv}. Extra in CSV: {extra_in_csv}")
                       raise ValueError("Input features mismatch model features.")
              else:
                    df_features_ordered = df_features[model_feature_names] # Ensure order even if sets match
                    logging.info("Input feature names match model feature names.")
         else:
              logging.warning("Model does not contain feature names (feature_names_in_ is None). Assuming column order is correct.")
              df_features_ordered = df_features[feature_cols_for_pred] # Use original order

    except AttributeError:
         logging.warning("Could not retrieve feature names from model (feature_names_in_ attribute missing). Assuming column order is correct.")
         df_features_ordered = df_features[feature_cols_for_pred]


    # Perform prediction
    logging.info(f"Starting prediction on {len(df_features_ordered)} rows...")
    try:
        predictions = model.predict(df_features_ordered)
        logging.info("Prediction complete.")
    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        raise

    # Create output DataFrame
    df_output = df_features[[id_col, time_col]].copy()
    df_output['prediction'] = predictions
    # Ensure predictions are non-negative if applicable (e.g., for counts)
    df_output['prediction'] = df_output['prediction'].clip(lower=0)
    logging.info(f"Output DataFrame created. Shape: {df_output.shape}")

    # Save predictions locally and upload
    logging.info(f"Saving predictions locally to {local_output_path}...")
    try:
        # Ensure output directory exists locally
        os.makedirs(os.path.dirname(local_output_path), exist_ok=True)
        df_output.to_csv(local_output_path, index=False)
        logging.info("Local save complete.")
        upload_to_gcs(local_output_path, output_gcs_path)
    except Exception as e:
        logging.error(f"Failed to save or upload predictions: {e}")
        raise

    logging.info(f"Predictions successfully saved to {output_gcs_path}")

# --- Main execution block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform batch prediction using a trained XGBoost model.")
    parser.add_argument('--model_gcs_path', type=str, required=True, help='GCS path to the trained model file (e.g., gs://.../model.joblib)')
    parser.add_argument('--features_gcs_path', type=str, required=True, help='GCS path to the input features CSV file (gs://...)')
    parser.add_argument('--output_gcs_path', type=str, required=True, help='GCS path to save the predictions CSV file (gs://...)')
    # Add arguments for identifier columns, matching generate_forecast_input
    parser.add_argument('--id_col', type=str, default='pickup_community_area', help='Name of the series ID column in the features file')
    parser.add_argument('--time_col', type=str, default='timestamp_hour', help='Name of the timestamp column in the features file')

    args = parser.parse_args()

    batch_predict_xgboost(
        model_gcs_path=args.model_gcs_path,
        features_gcs_path=args.features_gcs_path,
        output_gcs_path=args.output_gcs_path,
        id_col=args.id_col,
        time_col=args.time_col
    )
