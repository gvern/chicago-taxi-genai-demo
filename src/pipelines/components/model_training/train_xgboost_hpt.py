# src/pipelines/components/model_training/train_xgboost_hpt.py

import argparse
import os
import pandas as pd
import numpy as np
import xgboost as xgb
import hypertune
from google.cloud import bigquery
from src.pipelines.components.preprocessing.feature_engineering import preprocess_data_for_xgboost

def train_evaluate(project_id, bq_dataset, bq_table, model_dir, train_end_date, val_end_date, target_col, id_col, time_col, **hyperparameters):
    client = bigquery.Client(project=project_id)
    table_id = f"{project_id}.{bq_dataset}.{bq_table}"
    query = f"""
        SELECT *
        FROM `{table_id}`
        WHERE {time_col} < TIMESTAMP("{val_end_date}")
        ORDER BY {id_col}, {time_col}
    """
    print(f"Loading data with query: {query}")
    df = client.query(query).to_dataframe()
    print(f"Loaded {len(df)} rows.")

    if df.empty:
        raise ValueError("No data loaded. Check query and table.")

    print("Starting feature engineering...")
    df = preprocess_data_for_xgboost(df, is_train=True)
    X, y = df.drop(columns=[target_col]), df[target_col]
    print(f"Feature engineering complete. Features: {X.columns.tolist()}")

    print("Splitting data...")
    train_mask = pd.to_datetime(df[time_col]) < pd.to_datetime(train_end_date)
    val_mask = (pd.to_datetime(df[time_col]) >= pd.to_datetime(train_end_date)) & \
               (pd.to_datetime(df[time_col]) < pd.to_datetime(val_end_date))

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    print(f"Train size: {len(X_train)}, Validation size: {len(X_val)}")

    if len(X_train) == 0 or len(X_val) == 0:
        raise ValueError("Train or validation set is empty after splitting.")

    print(f"Training XGBoost with hyperparameters: {hyperparameters}")
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        n_jobs=-1,
        **hyperparameters
    )

    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10, verbose=True)

    print("Evaluating model...")
    predictions = model.predict(X_val)
    rmse = np.sqrt(np.mean((predictions - y_val) ** 2))
    print(f"Validation RMSE: {rmse}")

    print("Reporting metric to Hypertune...")
    hpt = hypertune.HyperTune()
    hpt.report_hyperparameter_tuning_metric(
        hyperparameter_metric_tag='rmse',
        metric_value=rmse,
    )

    print("Saving model...")
    model_dir = os.environ.get("AIP_MODEL_DIR", "./model")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.xgb")
    model.save_model(model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_id', type=str, required=True)
    parser.add_argument('--bq_dataset', type=str, required=True)
    parser.add_argument('--bq_table', type=str, required=True)
    parser.add_argument('--train_end_date', type=str, required=True, help='YYYY-MM-DDTHH:MM:SS')
    parser.add_argument('--val_end_date', type=str, required=True, help='YYYY-MM-DDTHH:MM:SS')
    parser.add_argument('--model_dir', type=str, default=os.environ.get("AIP_MODEL_DIR"), help='GCS path for saving the model')
    parser.add_argument('--target_col', type=str, default='num_trips')
    parser.add_argument('--id_col', type=str, default='pickup_community_area')
    parser.add_argument('--time_col', type=str, default='trip_start_timestamp_hour')
    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--max_depth', type=int, default=5)
    parser.add_argument('--subsample', type=float, default=0.8)
    parser.add_argument('--colsample_bytree', type=float, default=0.8)
    args = parser.parse_args()
    hparams = {
        'n_estimators': args.n_estimators,
        'learning_rate': args.learning_rate,
        'max_depth': args.max_depth,
        'subsample': args.subsample,
        'colsample_bytree': args.colsample_bytree,
    }
    train_evaluate(
        project_id=args.project_id,
        bq_dataset=args.bq_dataset,
        bq_table=args.bq_table,
        model_dir=args.model_dir,
        train_end_date=args.train_end_date,
        val_end_date=args.val_end_date,
        target_col=args.target_col,
        id_col=args.id_col,
        time_col=args.time_col,
        **hparams
    )