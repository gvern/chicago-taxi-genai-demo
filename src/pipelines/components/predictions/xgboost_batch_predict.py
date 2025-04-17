import argparse
import pandas as pd
import joblib
import os
import xgboost as xgb
from google.cloud import storage

def download_from_gcs(gcs_path, local_path):
    client = storage.Client()
    bucket_name, blob_path = gcs_path.replace("gs://", "").split("/", 1)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.download_to_filename(local_path)

def upload_to_gcs(local_path, gcs_path):
    client = storage.Client()
    bucket_name, blob_path = gcs_path.replace("gs://", "").split("/", 1)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(local_path)

def batch_predict_xgboost(model_gcs_path, features_gcs_path, output_gcs_path):
    local_model_path = "/tmp/model.json"
    local_features_path = "/tmp/future_features.csv"
    local_output_path = "/tmp/predictions.csv"

    print("Downloading model and features from GCS...")
    download_from_gcs(model_gcs_path, local_model_path)
    download_from_gcs(features_gcs_path, local_features_path)

    print("Loading model and features...")
    model = xgb.XGBRegressor()
    model.load_model(local_model_path)
    df = pd.read_csv(local_features_path)

    # Remove target column if present
    if 'target' in df.columns:
        df = df.drop(columns=['target'])

    print("Predicting...")
    preds = model.predict(df)
    df['prediction'] = preds

    # Save predictions with identifiers
    print("Saving predictions locally and uploading to GCS...")
    df[['timestamp', 'prediction'] + [col for col in df.columns if col.startswith('area_')]].to_csv(local_output_path, index=False)
    upload_to_gcs(local_output_path, output_gcs_path)
    print(f"Predictions saved to {output_gcs_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_gcs_path', type=str, required=True)
    parser.add_argument('--features_gcs_path', type=str, required=True)
    parser.add_argument('--output_gcs_path', type=str, required=True)
    args = parser.parse_args()
    batch_predict_xgboost(args.model_gcs_path, args.features_gcs_path, args.output_gcs_path)
