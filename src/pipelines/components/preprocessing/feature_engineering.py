import pandas as pd
import numpy as np
import logging # Add logging

# Configure logging
# You might want to configure this at a higher level in your application/pipeline
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- MODIFIED FUNCTION ---
def preprocess_data_for_xgboost(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """
    Applies feature engineering steps NOT dependent on the target variable.
    - Adds cyclical time features based on 'hour' and 'dayofweek'.
    - One-hot encodes 'pickup_community_area'.
    - Drops original time/ID columns and rows with NAs (if training).

    Args:
        df (pd.DataFrame): Input DataFrame. Must contain 'timestamp', 'hour',
                           'dayofweek', and 'pickup_community_area' columns.
                           If is_train=True, must also contain lag/rolling features.
        is_train (bool): Flag indicating if this is for training (to handle NA dropping).

    Returns:
        pd.DataFrame: DataFrame with engineered features, ready for XGBoost
                      (excluding the target column).
    """
    logging.info(f"Starting non-target feature engineering. Input shape: {df.shape}. is_train={is_train}")
    df = df.copy()

    # Ensure required columns exist
    required_cols = ['timestamp', 'hour', 'dayofweek', 'pickup_community_area']
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        raise ValueError(f"Missing required columns for feature engineering: {missing}")

    # Ensure timestamp is datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Cyclical time features (using existing 'hour' and 'dayofweek')
    logging.info("Generating cyclical time features...")
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7) # Assumes dayofweek is 0-6 (Mon-Sun)
    df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)

    # One-hot encode pickup_community_area
    logging.info("One-hot encoding pickup_community_area...")
    df = pd.get_dummies(df, columns=['pickup_community_area'], prefix='area', dummy_na=False) # Avoid NA column

    # Define features to keep (excluding original time/id/target and intermediate cols)
    # Keep lag/rolling features if they exist (added before calling this function during training)
    feature_cols = [col for col in df.columns if col.startswith('area_')]
    feature_cols += [col for col in df.columns if col.startswith('target_lag_')]
    feature_cols += [col for col in df.columns if col.startswith('target_rollmean_')]
    feature_cols += ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos']

    # Select only the engineered features + target (if present) + identifiers for potential NA drop
    cols_to_keep = feature_cols + ['timestamp'] # Keep timestamp temporarily for NA drop logic
    if 'target' in df.columns: # Keep target if it exists (training)
         cols_to_keep.append('target')

    original_cols = df.columns.tolist()
    df_processed = df[[col for col in cols_to_keep if col in original_cols]].copy()


    # Drop rows with NA (from lags/rolling means generated *before* this function)
    if is_train:
        initial_rows = len(df_processed)
        df_processed = df_processed.dropna()
        dropped_rows = initial_rows - len(df_processed)
        logging.info(f"Dropped {dropped_rows} rows with NA values (due to lags/rolling means).")
        if df_processed.empty:
             raise ValueError("DataFrame is empty after dropping NA values during training.")

    # Final feature set (remove timestamp after potential NA drop)
    final_feature_cols = [col for col in df_processed.columns if col != 'target' and col != 'timestamp']
    df_final = df_processed[final_feature_cols + (['target'] if 'target' in df_processed.columns else [])]

    logging.info(f"Feature engineering complete. Final shape: {df_final.shape}")
    logging.info(f"Final features: {final_feature_cols}")
    return df_final

# Keep engineer_temporal_features if used elsewhere, but it's partially duplicated now
def engineer_temporal_features(df: pd.DataFrame, timestamp_col: str, hour_period: int = 24) -> pd.DataFrame:
    """Adds basic temporal features (hour, dow, month) and cyclical hour features."""
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df['hour'] = df[timestamp_col].dt.hour
    df['dayofweek'] = df[timestamp_col].dt.dayofweek # Monday=0, Sunday=6
    df['month'] = df[timestamp_col].dt.month
    # Cyclical hour features
    df[f'{timestamp_col}_hour_sin'] = np.sin(2 * np.pi * df['hour'] / hour_period)
    df[f'{timestamp_col}_hour_cos'] = np.cos(2 * np.pi * df['hour'] / hour_period)
    return df
