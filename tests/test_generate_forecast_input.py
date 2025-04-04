import pytest
import pandas as pd
from datetime import datetime

from src.generate_forecast_input import generate_forecast_input

def test_output_is_dataframe():
    """Test que la sortie est un DataFrame non vide avec les bonnes colonnes"""
    df = generate_forecast_input()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty

    expected_cols = {
        "pickup_community_area",
        "timestamp_hour",
        "hour",
        "day_of_week",
        "month",
        "is_weekend",
        "is_holiday"
    }
    assert expected_cols.issubset(df.columns)

def test_timestamps_are_in_future():
    """Test que tous les timestamp_hour sont bien futurs"""
    df = generate_forecast_input()
    now = datetime.utcnow()
    assert (df["timestamp_hour"] > now).all()

def test_no_duplicates():
    """Test qu'il n'y a pas de doublons zone + timestamp"""
    df = generate_forecast_input()
    assert df.duplicated(subset=["pickup_community_area", "timestamp_hour"]).sum() == 0

def test_data_types():
    """Test des types de données attendus"""
    df = generate_forecast_input()
    assert pd.api.types.is_integer_dtype(df["pickup_community_area"])
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp_hour"])
    assert pd.api.types.is_integer_dtype(df["hour"])
    assert pd.api.types.is_integer_dtype(df["day_of_week"])
    assert pd.api.types.is_integer_dtype(df["month"])
    assert pd.api.types.is_bool_dtype(df["is_weekend"])
    assert pd.api.types.is_bool_dtype(df["is_holiday"])
