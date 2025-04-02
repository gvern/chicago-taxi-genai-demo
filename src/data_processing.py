import pandas as pd
import numpy as np
from typing import Dict, Any
import yaml

def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_data(file_path: str) -> pd.DataFrame:
    """Load the Chicago Taxi Trips dataset."""
    return pd.read_csv(file_path)

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the taxi trips data."""
    # Convert datetime columns
    df['trip_start_timestamp'] = pd.to_datetime(df['trip_start_timestamp'])
    df['trip_end_timestamp'] = pd.to_datetime(df['trip_end_timestamp'])
    
    # Calculate trip duration in minutes
    df['trip_duration'] = (df['trip_end_timestamp'] - df['trip_start_timestamp']).dt.total_seconds() / 60
    
    # Filter out invalid trips
    df = df[df['trip_duration'] > 0]
    df = df[df['trip_miles'] > 0]
    
    # Calculate speed in mph
    df['average_speed'] = df['trip_miles'] / (df['trip_duration'] / 60)
    
    return df

def sample_data(df: pd.DataFrame, sample_size: int, random_seed: int = 42) -> pd.DataFrame:
    """Sample the dataset to reduce size."""
    return df.sample(n=min(sample_size, len(df)), random_state=random_seed)
