import pandas as pd
import numpy as np
from typing import Dict, Any
import yaml


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_data(file_path: str) -> pd.DataFrame:
    """Load the Chicago Taxi Trips dataset from CSV."""
    return pd.read_csv(file_path)


def preprocess_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and convert raw columns to datetime, duration, etc."""
    df['trip_start_timestamp'] = pd.to_datetime(df['trip_start_timestamp'], errors='coerce')
    df['trip_end_timestamp'] = pd.to_datetime(df['trip_end_timestamp'], errors='coerce')

    df = df.dropna(subset=['trip_start_timestamp', 'trip_end_timestamp', 'trip_miles'])

    df['trip_duration'] = (df['trip_end_timestamp'] - df['trip_start_timestamp']).dt.total_seconds() / 60
    df = df[(df['trip_duration'] > 0) & (df['trip_miles'] > 0)]

    df['average_speed'] = df['trip_miles'] / (df['trip_duration'] / 60)
    df['pickup_community_area'] = df['pickup_community_area'].astype('Int64')  # zone
    df = df.dropna(subset=['pickup_community_area'])

    return df


def engineer_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add timestamp_hour and temporal features."""
    df['timestamp_hour'] = df['trip_start_timestamp'].dt.floor('H')
    df['hour'] = df['trip_start_timestamp'].dt.hour
    df['day_of_week'] = df['trip_start_timestamp'].dt.dayofweek
    df['month'] = df['trip_start_timestamp'].dt.month
    df['season'] = df['month'].map({
        12: 'winter', 1: 'winter', 2: 'winter',
        3: 'spring', 4: 'spring', 5: 'spring',
        6: 'summer', 7: 'summer', 8: 'summer',
        9: 'fall', 10: 'fall', 11: 'fall'
    })
    return df


def aggregate_demand_series(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate hourly demand per pickup zone."""
    demand_df = (
        df.groupby(['pickup_community_area', 'timestamp_hour'])
          .size()
          .reset_index(name='trip_count')
    )
    return demand_df


def sample_data(df: pd.DataFrame, sample_size: int, random_seed: int = 42) -> pd.DataFrame:
    """Optional: sample the data if needed."""
    return df.sample(n=min(sample_size, len(df)), random_state=random_seed)
