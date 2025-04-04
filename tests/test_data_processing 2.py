import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.data_processing import preprocess_data, sample_data

@pytest.fixture
def sample_taxi_data():
    """Create sample taxi data for testing."""
    return pd.DataFrame({
        'trip_start_timestamp': [
            '2023-01-01 10:00:00',
            '2023-01-01 11:00:00',
            '2023-01-01 12:00:00'
        ],
        'trip_end_timestamp': [
            '2023-01-01 10:30:00',
            '2023-01-01 11:45:00',
            '2023-01-01 12:15:00'
        ],
        'trip_miles': [5.0, 10.0, 3.0]
    })

def test_preprocess_data(sample_taxi_data):
    """Test the preprocess_data function."""
    processed_df = preprocess_data(sample_taxi_data)
    
    # Check if datetime conversion worked
    assert isinstance(processed_df['trip_start_timestamp'].iloc[0], pd.Timestamp)
    assert isinstance(processed_df['trip_end_timestamp'].iloc[0], pd.Timestamp)
    
    # Check if trip duration is calculated correctly
    assert processed_df['trip_duration'].iloc[0] == 30.0  # 30 minutes
    
    # Check if average speed is calculated correctly
    assert processed_df['average_speed'].iloc[0] == 10.0  # 5 miles / 0.5 hours = 10 mph

def test_sample_data(sample_taxi_data):
    """Test the sample_data function."""
    sample_size = 2
    sampled_df = sample_data(sample_taxi_data, sample_size)
    
    # Check if sample size is correct
    assert len(sampled_df) == sample_size
    
    # Check if sampled data comes from original data
    assert all(sampled_df.index.isin(sample_taxi_data.index))
