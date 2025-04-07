# Google Cloud Project Setup Guide

This document guides you through setting up the necessary Google Cloud resources for the Chicago Taxi Forecasting project.

## 1. Prerequisites

*   A Google Cloud Platform account with billing enabled.
*   `gcloud` command-line tool installed and authenticated (`gcloud auth login`).
*   Sufficient permissions to create projects, buckets, datasets, and enable APIs.

## 2. Project Setup

Choose a unique Google Cloud Project ID. We'll use `avisia-certification-ml-yde` as an example.

```bash
# Set your chosen Project ID
export PROJECT_ID="avisia-certification-ml-yde"

# Optional: Create a new project (if you don't have one)
# gcloud projects create ${PROJECT_ID} --name="Chicago Taxi Forecasting Demo"

# Set the active project for gcloud
gcloud config set project ${PROJECT_ID}

# Enable necessary APIs
gcloud services enable \\
    compute.googleapis.com \\
    iam.googleapis.com \\
    aiplatform.googleapis.com \\
    bigquery.googleapis.com \\
    cloudbuild.googleapis.com \\
    cloudresourcemanager.googleapis.com \\
    containerregistry.googleapis.com
```

## 3. Google Cloud Storage (GCS) Bucket

Create a GCS bucket to store pipeline artifacts, model assets, and potentially data staging files. Ensure the bucket name is globally unique.

```bash
# Choose a globally unique bucket name (often PROJECT_ID suffixed)
export BUCKET_NAME="${PROJECT_ID}-vertex-bucket"
export REGION="us-central1" # Or your preferred region

# Create the bucket in the specified region
gsutil mb -p ${PROJECT_ID} -l ${REGION} gs://${BUCKET_NAME}
```
*   `gs://${BUCKET_NAME}` will be used as `PIPELINE_ROOT` and for storing model artifacts.

## 4. BigQuery Dataset

Create a BigQuery dataset to store the processed taxi demand data.

```bash
# Choose a dataset name
export BQ_DATASET="chicago_taxis"

# Create the dataset in the specified region (e.g., US multi-region)
bq mk --location=US --dataset ${PROJECT_ID}:${BQ_DATASET}
```
*   The processed table `demand_by_hour` will reside in `${PROJECT_ID}:${BQ_DATASET}`.

## 5. Service Account (Optional but Recommended)

For running pipelines and interacting with services, it's best practice to use a dedicated service account with specific roles.

```bash
# Choose a name for your service account
export SA_NAME="vertex-pipeline-sa"
export SA_ID="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

# Create the service account
gcloud iam service-accounts create ${SA_NAME} \\
    --display-name="Vertex AI Pipeline Service Account" \\
    --project=${PROJECT_ID}

# Grant necessary roles (adjust based on least privilege principle)
# Common roles needed for Vertex AI Pipelines, BigQuery, GCS:
gcloud projects add-iam-policy-binding ${PROJECT_ID} \\
    --member="serviceAccount:${SA_ID}" \\
    --role="roles/aiplatform.user"
gcloud projects add-iam-policy-binding ${PROJECT_ID} \\
    --member="serviceAccount:${SA_ID}" \\
    --role="roles/storage.objectAdmin" # Access to GCS bucket
gcloud projects add-iam-policy-binding ${PROJECT_ID} \\
    --member="serviceAccount:${SA_ID}" \\
    --role="roles/bigquery.dataEditor" # Read/write BigQuery tables
gcloud projects add-iam-policy-binding ${PROJECT_ID} \\
    --member="serviceAccount:${SA_ID}" \\
    --role="roles/bigquery.jobUser" # Run BigQuery jobs

# You might configure Vertex AI Pipelines to use this service account.
```

## 6. Vertex AI Workbench (Optional)

For interactive development and running notebooks, you can create a Vertex AI Workbench instance.

```bash
# Follow the Google Cloud Console instructions or use gcloud to create a notebook instance.
# Ensure it has access to the services (e.g., via the service account created above or default compute SA).
```

Setup is now complete. You can proceed with cloning the repository and following the instructions in the main `README.md`.

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