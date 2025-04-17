import pandas as pd
import numpy as np

def preprocess_data_for_xgboost(df: pd.DataFrame, lags=[1,2,3], rolling_windows=[3,6], is_train=True):
    """
    Feature engineering for XGBoost forecasting.
    - Adds lag features and rolling means for the target.
    - Adds cyclical time features.
    - One-hot encodes 'pickup_community_area'.
    """
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(['pickup_community_area', 'timestamp'])

    # Lag features
    for lag in lags:
        df[f'target_lag_{lag}'] = df.groupby('pickup_community_area')['target'].shift(lag)

    # Rolling mean features
    for window in rolling_windows:
        df[f'target_rollmean_{window}'] = (
            df.groupby('pickup_community_area')['target']
            .shift(1)
            .rolling(window=window, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )

    # Cyclical time features
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)

    # One-hot encode pickup_community_area
    df = pd.get_dummies(df, columns=['pickup_community_area'], prefix='area')

    # Drop rows with NA (from lags)
    if is_train:
        df = df.dropna()

    return df
