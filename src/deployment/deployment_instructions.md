# Deployment Instructions

## Prerequisites
1. Google Cloud project set up
2. Required APIs enabled
3. Service account with necessary permissions
4. Docker installed locally

## Deployment Steps

### 1. Build Docker Image
```bash
# Navigate to project root
cd chicago-taxi-genai-demo

# Build Docker image
docker build -t gcr.io/${PROJECT_ID}/chicago-taxi-api .
```

### 2. Push Docker Image
```bash
# Configure Docker to use Google Cloud credentials
gcloud auth configure-docker

# Push image to Container Registry
docker push gcr.io/${PROJECT_ID}/chicago-taxi-api
```

### 3. Deploy to Cloud Run
```bash
# Deploy the service
gcloud run deploy chicago-taxi-api \
  --image gcr.io/${PROJECT_ID}/chicago-taxi-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars="PROJECT_ID=${PROJECT_ID}"
```

### 4. Configure Vertex AI Endpoint
```bash
# Create endpoint
gcloud ai endpoints create \
  --project=${PROJECT_ID} \
  --region=us-central1 \
  --display-name=chicago-taxi-endpoint

# Deploy model to endpoint
gcloud ai endpoints deploy-model ${ENDPOINT_ID} \
  --project=${PROJECT_ID} \
  --region=us-central1 \
  --model=${MODEL_ID} \
  --display-name=chicago-taxi-model \
  --machine-type=n1-standard-2 \
  --min-replica-count=1 \
  --max-replica-count=10
```

### 5. Set Up Monitoring
```bash
# Enable monitoring
gcloud monitoring dashboards create \
  --project=${PROJECT_ID} \
  --display-name="Chicago Taxi API Dashboard" \
  --dashboard-json=monitoring/dashboard.json
```

## Environment Variables
Required environment variables:
- `PROJECT_ID`: Google Cloud project ID
- `BUCKET_NAME`: GCS bucket for model artifacts
- `MODEL_NAME`: Name of the deployed model
- `ENDPOINT_ID`: Vertex AI endpoint ID

## Security Considerations
1. Enable Cloud IAM authentication
2. Set up VPC security
3. Configure SSL/TLS
4. Implement rate limiting
5. Set up audit logging

## Monitoring and Maintenance
1. Set up Cloud Monitoring alerts
2. Configure log-based metrics
3. Set up error reporting
4. Monitor API usage and costs

## Troubleshooting
Common issues and solutions:
1. Authentication errors
   - Verify service account permissions
   - Check environment variables

2. Model deployment failures
   - Check model artifacts
   - Verify endpoint configuration

3. API errors
   - Check logs in Cloud Logging
   - Verify input data format

## Rollback Procedure
1. Identify the previous working version
2. Deploy previous version:
```bash
gcloud run services update-traffic chicago-taxi-api \
  --to-revisions=REVISION_NAME=100
```

## Cost Management
1. Monitor resource usage
2. Set up budget alerts
3. Configure auto-scaling limits
4. Review and optimize resource allocation

## Support
For issues or questions:
1. Check Cloud Logging
2. Review monitoring dashboards
3. Contact support team
4. Submit issue in project repository 