import os
import yaml
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from google.cloud import aiplatform
from datetime import datetime

def load_config():
    """Load configuration from YAML file."""
    with open('config/pipeline_config.yaml', 'r') as f:
        return yaml.safe_load(f)

def load_data():
    """Load and prepare data from BigQuery."""
    query = """
    SELECT *
    FROM `chicago_taxi_data.processed_trips`
    """
    return pd.read_gbq(query)

def prepare_features(df):
    """Prepare features for model training."""
    # Select features for training
    feature_columns = [
        'hour_of_day',
        'day_of_week',
        'month',
        'trip_miles',
        'pickup_latitude',
        'pickup_longitude',
        'dropoff_latitude',
        'dropoff_longitude'
    ]
    
    # Prepare X and y
    X = df[feature_columns]
    y = df['trip_duration']
    
    return X, y

def train_model(X_train, y_train, X_val, y_val, config):
    """Train the model using XGBoost."""
    # Prepare DMatrix objects
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    # Set up parameters
    params = config['model']['params']
    
    # Train model
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=config['training']['epochs'],
        evals=[(dtrain, 'train'), (dval, 'validation')],
        early_stopping_rounds=config['training']['early_stopping_rounds']
    )
    
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance."""
    dtest = xgb.DMatrix(X_test)
    y_pred = model.predict(dtest)
    
    # Calculate metrics
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    mae = np.mean(np.abs(y_test - y_pred))
    r2 = 1 - np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)
    
    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }

def save_model(model, metrics):
    """Save model to Vertex AI."""
    # Initialize Vertex AI
    aiplatform.init(project=os.getenv('PROJECT_ID'))
    
    # Create model name with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_name = f"chicago_taxi_demand_{timestamp}"
    
    # Save model
    model.save_model(f"gs://{os.getenv('BUCKET_NAME')}/models/{model_name}/model.json")
    
    # Register model in Vertex AI
    model = aiplatform.Model.upload(
        display_name=model_name,
        artifact_uri=f"gs://{os.getenv('BUCKET_NAME')}/models/{model_name}",
        model_id=model_name
    )
    
    # Log metrics
    model.log_metrics(metrics)
    
    return model

def main():
    """Main function to run the training pipeline."""
    # Load configuration
    config = load_config()
    
    # Load data
    df = load_data()
    
    # Prepare features
    X, y = prepare_features(df)
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=config['data_processing']['train_test_split']
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5
    )
    
    # Train model
    model = train_model(X_train, y_train, X_val, y_val, config)
    
    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test)
    
    # Save model
    saved_model = save_model(model, metrics)
    
    print(f"Model training completed. Metrics: {metrics}")
    print(f"Model saved as: {saved_model.display_name}")

if __name__ == '__main__':
    main() 