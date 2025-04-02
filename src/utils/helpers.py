import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from datetime import datetime
import yaml

def load_config(config_path: str = "config/pipeline_config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def calculate_haversine_distance(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float
) -> float:
    """Calculate the Haversine distance between two points."""
    R = 6371  # Earth's radius in kilometers
    
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    distance = R * c
    return distance

def extract_temporal_features(
    timestamp: datetime
) -> Dict[str, int]:
    """Extract temporal features from timestamp."""
    return {
        'hour': timestamp.hour,
        'day_of_week': timestamp.weekday(),
        'month': timestamp.month,
        'day_of_month': timestamp.day,
        'is_weekend': int(timestamp.weekday() >= 5),
        'is_holiday': int(timestamp.weekday() == 6)  # Simple holiday detection
    }

def calculate_speed(
    distance: float,
    duration: float
) -> float:
    """Calculate average speed in miles per hour."""
    if duration <= 0:
        return 0
    return distance / (duration / 60)  # Convert minutes to hours

def prepare_model_input(
    trip_data: Dict[str, Any]
) -> np.ndarray:
    """Prepare input data for model prediction."""
    features = [
        trip_data['hour_of_day'],
        trip_data['day_of_week'],
        trip_data['month'],
        trip_data['trip_miles'],
        trip_data['pickup_latitude'],
        trip_data['pickup_longitude'],
        trip_data['dropoff_latitude'],
        trip_data['dropoff_longitude']
    ]
    return np.array([features])

def validate_trip_data(
    trip_data: Dict[str, Any]
) -> Tuple[bool, str]:
    """Validate trip data."""
    # Check required fields
    required_fields = [
        'hour_of_day', 'day_of_week', 'month',
        'trip_miles', 'pickup_latitude', 'pickup_longitude',
        'dropoff_latitude', 'dropoff_longitude'
    ]
    
    for field in required_fields:
        if field not in trip_data:
            return False, f"Missing required field: {field}"
    
    # Validate ranges
    if not (0 <= trip_data['hour_of_day'] <= 23):
        return False, "Invalid hour_of_day (must be 0-23)"
    
    if not (0 <= trip_data['day_of_week'] <= 6):
        return False, "Invalid day_of_week (must be 0-6)"
    
    if not (1 <= trip_data['month'] <= 12):
        return False, "Invalid month (must be 1-12)"
    
    if trip_data['trip_miles'] <= 0:
        return False, "Invalid trip_miles (must be positive)"
    
    # Validate coordinates
    for lat in [trip_data['pickup_latitude'], trip_data['dropoff_latitude']]:
        if not (-90 <= lat <= 90):
            return False, "Invalid latitude (must be -90 to 90)"
    
    for lon in [trip_data['pickup_longitude'], trip_data['dropoff_longitude']]:
        if not (-180 <= lon <= 180):
            return False, "Invalid longitude (must be -180 to 180)"
    
    return True, "Valid"

def format_prediction_response(
    prediction: float,
    input_data: Dict[str, Any]
) -> Dict[str, Any]:
    """Format prediction response."""
    return {
        "predicted_duration_minutes": float(prediction),
        "input_data": input_data,
        "timestamp": datetime.now().isoformat()
    } 