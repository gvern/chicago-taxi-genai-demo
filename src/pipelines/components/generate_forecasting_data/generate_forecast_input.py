from kfp.v2.dsl import component, Output, Dataset
import argparse
import pandas as pd
import numpy as np
from google.cloud import bigquery
from google.cloud import storage
import os
from archives.data_processing import engineer_temporal_features
from src.pipelines.components.preprocessing.feature_engineering import preprocess_data_for_xgboost

def _generate_forecast_input_impl(
    project_id: str,
    bq_dataset: str,
    bq_table_prepared: str,
    id_col: str,
    time_col: str,
    forecast_horizon_hours: int,
    forecast_start_time: str,
    output_gcs_path: str
):
    client = bigquery.Client(project=project_id)
    query_ids = f"SELECT DISTINCT {id_col} FROM `{project_id}.{bq_dataset}.{bq_table_prepared}` WHERE {id_col} IS NOT NULL"
    print(f"Fetching distinct IDs with query: {query_ids}")
    ids_df = client.query(query_ids).to_dataframe()
    ids = ids_df[id_col].unique()
    print(f"Found {len(ids)} distinct IDs.")

    start_ts = pd.to_datetime(forecast_start_time, utc=True)
    future_timestamps = pd.date_range(
        start=start_ts,
        periods=forecast_horizon_hours,
        freq='H',
        tz='UTC'
    )
    print(f"Generating timestamps from {start_ts} for {forecast_horizon_hours} hours.")

    future_df = pd.MultiIndex.from_product([ids, future_timestamps], names=[id_col, time_col])
    future_df = pd.DataFrame(index=future_df).reset_index()
    print(f"Created future DataFrame shape: {future_df.shape}")

    print("Generating features for future data...")
    future_df['timestamp_col'] = pd.to_datetime(future_df[time_col])
    future_df['trip_hour'] = future_df['timestamp_col'].dt.hour
    future_df['trip_dayofweek'] = future_df['timestamp_col'].dt.dayofweek
    future_df['trip_month'] = future_df['timestamp_col'].dt.month
    future_df = engineer_temporal_features(future_df, 'trip_hour', 24)
    future_df = pd.get_dummies(future_df, columns=['trip_dayofweek'], prefix='dow')
    all_dow_cols = [f'dow_{i}' for i in range(7)]
    for col in all_dow_cols:
        if col not in future_df.columns:
            future_df[col] = 0
    future_df = future_df.sort_index(axis=1)
    future_features_df = preprocess_data_for_xgboost(future_df, is_train=False)
    print(f"Generated features: {future_features_df.columns.tolist()}")
    print(f"Final future data shape: {future_features_df.shape}")

    print(f"Saving future data to {output_gcs_path}")
    future_features_df.to_csv(output_gcs_path, index=False)
    print("Future data saved successfully.")

@component(
    base_image="python:3.9",
    packages_to_install=["google-cloud-bigquery", "google-cloud-storage", "pandas", "numpy"]
)
def generate_forecast_input(
    project_id: str,
    bq_dataset: str,
    bq_table_prepared: str,
    id_col: str,
    time_col: str,
    forecast_horizon_hours: int,
    forecast_start_time: str,
    output_gcs_path: str,
    future_features: Output[Dataset]
):
    _generate_forecast_input_impl(
        project_id=project_id,
        bq_dataset=bq_dataset,
        bq_table_prepared=bq_table_prepared,
        id_col=id_col,
        time_col=time_col,
        forecast_horizon_hours=forecast_horizon_hours,
        forecast_start_time=forecast_start_time,
        output_gcs_path=output_gcs_path
    )
    future_features.uri = output_gcs_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_id', type=str, required=True)
    parser.add_argument('--bq_dataset', type=str, required=True)
    parser.add_argument('--bq_table_prepared', type=str, required=True)
    parser.add_argument('--id_col', type=str, default='pickup_community_area')
    parser.add_argument('--time_col', type=str, default='trip_start_timestamp_hour')
    parser.add_argument('--forecast_horizon_hours', type=int, required=True)
    parser.add_argument('--forecast_start_time', type=str, required=True)
    parser.add_argument('--output_gcs_path', type=str, required=True)
    args = parser.parse_args()
    class DummyOutput:
        def __init__(self):
            self.uri = None
    future_features = DummyOutput()
    _generate_forecast_input_impl(
        project_id=args.project_id,
        bq_dataset=args.bq_dataset,
        bq_table_prepared=args.bq_table_prepared,
        id_col=args.id_col,
        time_col=args.time_col,
        forecast_horizon_hours=args.forecast_horizon_hours,
        forecast_start_time=args.forecast_start_time,
        output_gcs_path=args.output_gcs_path
    )
    future_features.uri = args.output_gcs_path
