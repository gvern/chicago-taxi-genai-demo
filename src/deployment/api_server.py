from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google.cloud import aiplatform
import numpy as np
import os
from typing import List, Optional

# Initialize FastAPI app
app = FastAPI(
    title="Chicago Taxi Demand Prediction API",
    description="API for predicting taxi trip duration in Chicago",
    version="1.0.0"
)

# Initialize Vertex AI
aiplatform.init(project=os.getenv('PROJECT_ID'))

# Load the latest model
model = aiplatform.Model.list(
    filter=f"display_name=chicago_taxi_demand_*",
    order_by="create_time desc"
)[0]

# Create model endpoint
endpoint = model.deploy(
    machine_type="n1-standard-2",
    accelerator_type=None,
    accelerator_count=None,
    min_replica_count=1,
    max_replica_count=10
)

class TripRequest(BaseModel):
    hour_of_day: int
    day_of_week: int
    month: int
    trip_miles: float
    pickup_latitude: float
    pickup_longitude: float
    dropoff_latitude: float
    dropoff_longitude: float

class BatchTripRequest(BaseModel):
    trips: List[TripRequest]

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Welcome to Chicago Taxi Demand Prediction API"}

@app.post("/predict")
async def predict_trip_duration(trip: TripRequest):
    """Predict trip duration for a single trip."""
    try:
        # Prepare input data
        input_data = np.array([[
            trip.hour_of_day,
            trip.day_of_week,
            trip.month,
            trip.trip_miles,
            trip.pickup_latitude,
            trip.pickup_longitude,
            trip.dropoff_latitude,
            trip.dropoff_longitude
        ]])
        
        # Make prediction
        prediction = endpoint.predict(input_data)
        
        return {
            "predicted_duration_minutes": float(prediction[0][0]),
            "input_data": trip.dict()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch")
async def predict_batch_trip_duration(trips: BatchTripRequest):
    """Predict trip duration for multiple trips."""
    try:
        # Prepare input data
        input_data = np.array([[
            trip.hour_of_day,
            trip.day_of_week,
            trip.month,
            trip.trip_miles,
            trip.pickup_latitude,
            trip.pickup_longitude,
            trip.dropoff_latitude,
            trip.dropoff_longitude
        ] for trip in trips.trips])
        
        # Make predictions
        predictions = endpoint.predict(input_data)
        
        return {
            "predictions": [
                {
                    "predicted_duration_minutes": float(pred[0]),
                    "input_data": trip.dict()
                }
                for pred, trip in zip(predictions, trips.trips)
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 