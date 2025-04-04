#!/usr/bin/env python
import os
import sys
import subprocess

# Ajouter le répertoire courant au PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Installer le package en mode développement si nécessaire
try:
    import src
except ImportError:
    print("Installation du package en mode développement...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", "."])
    print("Package installé avec succès!")

# Importer et exécuter le pipeline
from src.pipelines.forecasting_pipeline import forecasting_pipeline

if __name__ == "__main__":
    # Paramètres du pipeline
    project = "your-project-id"  # Remplacez par votre ID de projet
    location = "us-central1"     # Remplacez par votre région
    
    # Exécuter le pipeline
    forecasting_pipeline(
        project=project,
        location=location,
        bq_query="SELECT * FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips` LIMIT 1000",
        bq_output_uri="bq://your-project-id.chicago_taxis.demand_by_hour",
        dataset_display_name="chicago-taxi-demand-dataset",
        forecast_model_display_name="chicago-taxi-demand-model",
        target_column="trip_count",
        time_column="timestamp_hour",
        time_series_identifier_column="pickup_community_area",
        forecast_horizon=24,
        context_window=168
    ) 