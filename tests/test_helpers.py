import pytest
import numpy as np
from datetime import datetime
from src.utils.helpers import (
    calculate_haversine_distance,
    extract_temporal_features,
    calculate_speed,
    prepare_model_input,
    validate_trip_data,
    format_prediction_response
)

def test_calculate_haversine_distance():
    """Test Haversine distance calculation."""
    # Chicago coordinates
    chicago_lat, chicago_lon = 41.8781, -87.6298
    # New York coordinates
    ny_lat, ny_lon = 40.7128, -74.0060
    
    distance = calculate_haversine_distance(chicago_lat, chicago_lon, ny_lat, ny_lon)
    assert isinstance(distance, float)
    assert distance > 0
    assert distance < 2000  # Roughly 1200 km

def test_extract_temporal_features():
    """Test temporal feature extraction."""
    # Test with a specific date
    test_date = datetime(2024, 1, 1, 12, 0)  # Monday, January 1, 2024, 12:00
    features = extract_temporal_features(test_date)
    
    assert features['hour'] == 12
    assert features['day_of_week'] == 0  # Monday
    assert features['month'] == 1
    assert features['day_of_month'] == 1
    assert features['is_weekend'] == 0
    assert features['is_holiday'] == 0

def test_calculate_speed():
    """Test speed calculation."""
    # Test normal case
    speed = calculate_speed(distance=10.0, duration=30.0)  # 10 miles in 30 minutes
    assert speed == 20.0  # 20 mph
    
    # Test zero duration
    speed = calculate_speed(distance=10.0, duration=0.0)
    assert speed == 0.0
    
    # Test negative duration
    speed = calculate_speed(distance=10.0, duration=-30.0)
    assert speed == 0.0

def test_prepare_model_input():
    """Test model input preparation."""
    trip_data = {
        'hour_of_day': 12,
        'day_of_week': 0,
        'month': 1,
        'trip_miles': 5.0,
        'pickup_latitude': 41.8781,
        'pickup_longitude': -87.6298,
        'dropoff_latitude': 41.8781,
        'dropoff_longitude': -87.6298
    }
    
    input_array = prepare_model_input(trip_data)
    assert isinstance(input_array, np.ndarray)
    assert input_array.shape == (1, 8)
    assert input_array[0][0] == 12
    assert input_array[0][3] == 5.0

def test_validate_trip_data():
    """Test trip data validation."""
    # Valid data
    valid_data = {
        'hour_of_day': 12,
        'day_of_week': 0,
        'month': 1,
        'trip_miles': 5.0,
        'pickup_latitude': 41.8781,
        'pickup_longitude': -87.6298,
        'dropoff_latitude': 41.8781,
        'dropoff_longitude': -87.6298
    }
    
    is_valid, message = validate_trip_data(valid_data)
    assert is_valid
    assert message == "Valid"
    
    # Invalid hour
    invalid_data = valid_data.copy()
    invalid_data['hour_of_day'] = 24
    is_valid, message = validate_trip_data(invalid_data)
    assert not is_valid
    assert "Invalid hour_of_day" in message
    
    # Missing field
    invalid_data = valid_data.copy()
    del invalid_data['trip_miles']
    is_valid, message = validate_trip_data(invalid_data)
    assert not is_valid
    assert "Missing required field" in message

def test_format_prediction_response():
    """Test prediction response formatting."""
    prediction = 30.0
    input_data = {
        'hour_of_day': 12,
        'day_of_week': 0,
        'month': 1,
        'trip_miles': 5.0,
        'pickup_latitude': 41.8781,
        'pickup_longitude': -87.6298,
        'dropoff_latitude': 41.8781,
        'dropoff_longitude': -87.6298
    }
    
    response = format_prediction_response(prediction, input_data)
    assert response['predicted_duration_minutes'] == 30.0
    assert response['input_data'] == input_data
    assert 'timestamp' in response
    assert isinstance(response['timestamp'], str) 