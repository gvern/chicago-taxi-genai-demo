import pytest
import pandas as pd
from datetime import datetime

from src.generate_forecast_input import generate_forecast_input


def test_generate_forecast_input_structure():
    """Test que la sortie contient les colonnes attendues et au moins une ligne"""
    df = generate_forecast_input(horizon_hours=5)  # horizon réduit pour test rapide
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    expected_columns = {
        "pickup_community_area",
        "timestamp_hour",
        "hour",
        "day_of_week",
        "month"
    }
    assert expected_columns.issubset(df.columns)


def test_future_timestamps_are_in_future():
    """Test que toutes les timestamps sont bien dans le futur"""
    df = generate_forecast_input(horizon_hours=5)
    now = datetime.utcnow()
    assert all(pd.to_datetime(df["timestamp_hour"]) > now)


def test_no_duplicate_series():
    """Test qu'il n'y a pas de doublons timestamp + zone"""
    df = generate_forecast_input(horizon_hours=5)
    assert df.duplicated(subset=["timestamp_hour", "pickup_community_area"]).sum() == 0
