import pytest
import pandas as pd
from datetime import datetime, timedelta

from src.generate_forecast_input import generate_future_input

def test_generate_future_input_shape():
    """Test que la sortie contient les colonnes attendues et au moins une ligne"""
    df = generate_future_input()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert 'timestamp_hour' in df.columns
    assert 'pickup_community_area' in df.columns

def test_future_timestamps_are_in_future():
    """Test que toutes les timestamps sont bien dans le futur"""
    df = generate_future_input()
    now = datetime.utcnow()
    assert all(pd.to_datetime(df['timestamp_hour']) > now)

def test_no_duplicate_series():
    """Test qu'il n'y a pas de doublons timestamp + zone"""
    df = generate_future_input()
    assert df.duplicated(subset=['timestamp_hour', 'pickup_community_area']).sum() == 0
