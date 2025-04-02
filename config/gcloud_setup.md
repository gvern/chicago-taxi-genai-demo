# Google Cloud Setup Instructions

## Prerequisites
1. Google Cloud account
2. Google Cloud SDK installed
3. Project created in Google Cloud Console

## Project Setup

### 1. Initialize Project
```bash
# Set project ID
export PROJECT_ID=chicago-taxi-analysis
gcloud config set project $PROJECT_ID

# Enable required APIs
gcloud services enable \
  bigquery.googleapis.com \
  compute.googleapis.com \
  aiplatform.googleapis.com \
  storage.googleapis.com
```

### 2. Create Storage Bucket
```bash
# Create bucket for data storage
gsutil mb -l us-central1 gs://${PROJECT_ID}-data
```

### 3. Set Up BigQuery
```bash
# Create dataset
bq mk --dataset ${PROJECT_ID}:chicago_taxi_data

# Create tables
bq mk --table ${PROJECT_ID}:chicago_taxi_data.trips \
  trip_id:STRING,\
  trip_start_timestamp:TIMESTAMP,\
  trip_end_timestamp:TIMESTAMP,\
  trip_miles:FLOAT,\
  pickup_latitude:FLOAT,\
  pickup_longitude:FLOAT,\
  dropoff_latitude:FLOAT,\
  dropoff_longitude:FLOAT
```

### 4. Configure IAM Permissions
```bash
# Grant necessary permissions to service account
gcloud projects add-iam-policy-binding ${PROJECT_ID} \
  --member="serviceAccount:${PROJECT_ID}@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/bigquery.dataEditor"
```

### 5. Vertex AI Setup
```bash
# Enable Vertex AI API
gcloud services enable aiplatform.googleapis.com

# Create service account for Vertex AI
gcloud iam service-accounts create vertex-ai-sa \
  --display-name="Vertex AI Service Account"
```

## Environment Variables
Create a `.env` file with the following variables:
```
PROJECT_ID=chicago-taxi-analysis
BUCKET_NAME=${PROJECT_ID}-data
DATASET_NAME=chicago_taxi_data
REGION=us-central1
```

## Security Considerations
1. Use service accounts with minimal required permissions
2. Enable audit logging
3. Set up VPC security
4. Configure encryption at rest

## Monitoring Setup
1. Enable Cloud Monitoring
2. Set up alerting policies
3. Configure logging sinks
4. Create dashboards for monitoring

## Cost Management
1. Set up budget alerts
2. Configure quotas
3. Monitor resource usage
4. Implement cost optimization strategies 