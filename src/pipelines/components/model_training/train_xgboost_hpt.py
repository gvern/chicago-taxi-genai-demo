# src/pipelines/components/model_training/train_xgboost_hpt.py

import argparse
import os
import pandas as pd
import numpy as np
import xgboost as xgb
try:
    import hypertune
except ImportError:
    hypertune = None # Hypertune might not be installed in all environments
from google.cloud import bigquery
import logging # Add logging
import joblib # Use joblib for saving model

# Import the modified feature engineering function
from src.pipelines.components.preprocessing.feature_engineering import preprocess_data_for_xgboost

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_target_derived_features(df: pd.DataFrame, id_col: str, time_col: str, target_col: str, lags=[1, 2, 3, 24, 48, 168], rolling_windows=[3, 6, 12, 24]) -> pd.DataFrame:
    """Generates lag and rolling mean features based on the target variable."""
    logging.info(f"Generating target-derived features (lags: {lags}, rolling: {rolling_windows})...")
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values([id_col, time_col])

    # Lag features
    for lag in lags:
        df[f'target_lag_{lag}'] = df.groupby(id_col)[target_col].shift(lag)
        logging.info(f"Generated target_lag_{lag}")

    # Rolling mean features (using shift(1) to avoid data leakage)
    for window in rolling_windows:
        df[f'target_rollmean_{window}'] = (
            df.groupby(id_col)[target_col]
            .shift(1) # Use data prior to the current timestamp
            .rolling(window=window, min_periods=1) # Calculate rolling mean on past data
            .mean()
            .reset_index(level=0, drop=True) # Align index after rolling
        )
        logging.info(f"Generated target_rollmean_{window}")

    logging.info("Target-derived features generated.")
    return df


def train_evaluate(project_id: str, bq_dataset: str, bq_table: str, model_dir: str,
                   train_end_date: str, val_end_date: str,
                   target_col: str, id_col: str, time_col: str,
                   hpt_enabled: bool, **hyperparameters):
    """
    Loads data, engineers features, trains XGBoost, evaluates, and saves the model.

    Args:
        project_id: GCP Project ID.
        bq_dataset: BigQuery dataset ID.
        bq_table: BigQuery table name containing prepared data.
        model_dir: GCS directory to save the trained model (from AIP_MODEL_DIR).
        train_end_date: Timestamp string marking the end of the training period.
        val_end_date: Timestamp string marking the end of the validation period.
        target_col: Name of the target variable column.
        id_col: Name of the series identifier column.
        time_col: Name of the timestamp column.
        hpt_enabled: Boolean flag indicating if Hypertune reporting is enabled.
        **hyperparameters: Dictionary of hyperparameters for XGBoost.
    """
    client = bigquery.Client(project=project_id)
    table_id = f"{project_id}.{bq_dataset}.{bq_table}"

    # Load data up to the validation end date
    # Include necessary columns for feature engineering (hour, dayofweek etc. from BQ)
    # Ensure target column name matches BQ table (e.g., trip_count)
    query = f"""
        SELECT *, {time_col} as timestamp, {target_col} as target
        FROM `{table_id}`
        WHERE {time_col} < TIMESTAMP("{val_end_date}")
        ORDER BY {id_col}, {time_col}
    """
    logging.info(f"Loading data with query: {query}")
    df = client.query(query).to_dataframe()
    logging.info(f"Loaded {len(df)} rows.")

    if df.empty:
        raise ValueError("No data loaded. Check query, table, and date range.")

    # Ensure correct types before feature engineering
    df[time_col] = pd.to_datetime(df[time_col])
    # Ensure 'hour', 'dayofweek' exist from BQ query or extract them
    if 'hour' not in df.columns:
        df['hour'] = df[time_col].dt.hour
    if 'dayofweek' not in df.columns:
         # Ensure BQ dayofweek (Sun=1) is converted to Python (Mon=0) if needed by preprocess_data_for_xgboost
         # Assuming BQ query provides 'day_of_week' as Sun=1..Sat=7
         # Convert to Mon=0..Sun=6: MOD(day_of_week + 5, 7)
         if 'day_of_week' in df.columns:
              logging.info("Converting BQ day_of_week (1-7) to Python dayofweek (0-6)...")
              # BQ Sunday=1, Monday=2, ..., Saturday=7
              # Python Monday=0, ..., Sunday=6
              # Mapping: 1->6, 2->0, 3->1, 4->2, 5->3, 6->4, 7->5
              day_map = {1: 6, 2: 0, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5}
              df['dayofweek'] = df['day_of_week'].map(day_map)
              if df['dayofweek'].isnull().any():
                  logging.warning("Null values produced during day_of_week mapping. Check original values.")
         else:
              logging.warning(f"Column 'day_of_week' not found. Extracting from {time_col}. Assuming Monday=0.")
              df['dayofweek'] = df[time_col].dt.dayofweek


    # --- FEATURE ENGINEERING ---
    # 1. Generate target-derived features (lags, rolling means) using 'target' column
    df = generate_target_derived_features(df, id_col, time_col, 'target') # Use 'target' alias

    # 2. Generate non-target features (cyclical, OHE) using the modified function
    # Pass the correct column names expected by the function
    df_processed = preprocess_data_for_xgboost(df, is_train=True)

    # Separate features (X) and target (y) AFTER all processing and NA drops
    if 'target' not in df_processed.columns:
         raise ValueError(f"Target column 'target' not found after feature processing.")
    y = df_processed['target']
    X = df_processed.drop(columns=['target'])
    logging.info(f"Feature engineering complete. Final features: {X.columns.tolist()}")

    # Re-attach timestamp for splitting (it was removed in preprocess_data_for_xgboost)
    # Ensure the index aligns if df_processed had rows dropped
    X[time_col] = df.loc[X.index, time_col] # Use .loc to align based on index

    # --- SPLITTING DATA ---
    logging.info(f"Splitting data using train_end_date: {train_end_date}")
    train_mask = X[time_col] < pd.to_datetime(train_end_date)
    # Validation mask uses the original time_col from X before it's dropped
    val_mask = (X[time_col] >= pd.to_datetime(train_end_date)) & \
               (X[time_col] < pd.to_datetime(val_end_date))

    X_train, y_train = X[train_mask].drop(columns=[time_col]), y[train_mask]
    X_val, y_val = X[val_mask].drop(columns=[time_col]), y[val_mask]
    logging.info(f"Train size: {len(X_train)}, Validation size: {len(X_val)}")

    if len(X_train) == 0 or len(X_val) == 0:
        raise ValueError("Train or validation set is empty after splitting. Check date ranges and data.")

    # --- MODEL TRAINING ---
    logging.info(f"Training XGBoost with hyperparameters: {hyperparameters}")
    # Ensure hyperparameters passed are valid for XGBoost
    valid_xgb_params = {
        'objective': 'reg:squarederror',
        'random_state': 42,
        'n_jobs': -1,
        'n_estimators': int(hyperparameters.get('n_estimators', 100)), # Ensure int
        'learning_rate': float(hyperparameters.get('learning_rate', 0.1)), # Ensure float
        'max_depth': int(hyperparameters.get('max_depth', 5)), # Ensure int
        'subsample': float(hyperparameters.get('subsample', 0.8)), # Ensure float
        'colsample_bytree': float(hyperparameters.get('colsample_bytree', 0.8)), # Ensure float
        'reg_lambda': float(hyperparameters.get('reg_lambda', 1.0)), # Ensure float
        # Add other expected hyperparameters here, ensuring correct types
    }
    model = xgb.XGBRegressor(**valid_xgb_params)

    # Pass feature names if available and using DataFrame
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10, verbose=True)

    # --- EVALUATION ---
    logging.info("Evaluating model on validation set...")
    predictions = model.predict(X_val)
    rmse = np.sqrt(np.mean((predictions - y_val) ** 2))
    logging.info(f"Validation RMSE: {rmse}")

    # --- HPT REPORTING ---
    if hpt_enabled:
        if hypertune is None:
            logging.warning("Hypertune library not found, but HPT reporting was requested.")
        else:
            logging.info(f"Reporting metric 'rmse'={rmse} to Hypertune...")
            hpt = hypertune.HyperTune()
            hpt.report_hyperparameter_tuning_metric(
                hyperparameter_metric_tag='rmse', # Ensure this matches hpt_config in pipeline
                metric_value=rmse,
                # step=1 # Optional: specify step if needed
            )
    else:
        logging.info("Hypertune reporting disabled.")

    # --- SAVING MODEL ---
    logging.info("Saving model...")
    if not model_dir:
        logging.error("model_dir (AIP_MODEL_DIR) not provided. Cannot save model.")
        raise ValueError("Model directory environment variable not set.")

    # Vertex AI expects the model artifact in a specific structure if using managed datasets/models.
    # For custom jobs/HPT, saving directly to AIP_MODEL_DIR is standard.
    # Save as model.joblib for broader compatibility.
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.joblib") # Joblib format
    joblib.dump(model, model_path)
    logging.info(f"Model saved to {model_path}")

# --- Main execution block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train XGBoost model with optional HPT.")
    # GCP and Data Args
    parser.add_argument('--project_id', type=str, required=True, help='GCP Project ID')
    parser.add_argument('--bq_dataset', type=str, required=True, help='BigQuery dataset ID')
    parser.add_argument('--bq_table', type=str, required=True, help='BigQuery table name for training data')
    # Date Args
    parser.add_argument('--train_end_date', type=str, required=True, help='End date for training set (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)')
    parser.add_argument('--val_end_date', type=str, required=True, help='End date for validation set (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)')
    # Model/Column Args
    parser.add_argument('--model_dir', type=str, default=os.environ.get("AIP_MODEL_DIR"), help='GCS path for saving the model (usually from AIP_MODEL_DIR)')
    parser.add_argument('--target_col', type=str, default='trip_count', help='Name of the target column in the BQ table')
    parser.add_argument('--id_col', type=str, default='pickup_community_area', help='Name of the series ID column')
    parser.add_argument('--time_col', type=str, default='timestamp_hour', help='Name of the timestamp column in the BQ table')
    # HPT Arg
    parser.add_argument('--hpt_enabled', type=lambda x: (str(x).lower() == 'true'), default=True, help='Enable Hypertune metric reporting')

    # Hyperparameters (define defaults, can be overridden by HPT or command line)
    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--max_depth', type=int, default=5)
    parser.add_argument('--subsample', type=float, default=0.8)
    parser.add_argument('--colsample_bytree', type=float, default=0.8)
    parser.add_argument('--reg_lambda', type=float, default=1.0)

    args, unknown = parser.parse_known_args()

    # Collect hyperparameters explicitly passed or defaults
    hparams = {
        'n_estimators': args.n_estimators,
        'learning_rate': args.learning_rate,
        'max_depth': args.max_depth,
        'subsample': args.subsample,
        'colsample_bytree': args.colsample_bytree,
        'reg_lambda': args.reg_lambda,
    }
    # If HPT is enabled, hypertune passes hyperparameters like --learning_rate=0.05
    # These will be in 'unknown' args but XGBoost handles them if passed via **hparams.
    # Let's update hparams with any explicitly passed hyperparameter args.
    logging.info(f"Default/CLI Hyperparameters: {hparams}")
    # Note: Vertex AI HPT injects hyperparameters directly as arguments (e.g., --learning_rate=0.05).
    # The **hyperparameters in train_evaluate will capture these. We pass the hparams dict
    # primarily for the non-HPT case and potentially for logging defaults.

    train_evaluate(
        project_id=args.project_id,
        bq_dataset=args.bq_dataset,
        bq_table=args.bq_table,
        model_dir=args.model_dir,
        train_end_date=args.train_end_date,
        val_end_date=args.val_end_date,
        target_col=args.target_col, # Pass BQ target column name
        id_col=args.id_col,
        time_col=args.time_col, # Pass BQ time column name
        hpt_enabled=args.hpt_enabled,
        **hparams # Pass defaults, HPT overrides will be handled by **hyperparameters in function
    )