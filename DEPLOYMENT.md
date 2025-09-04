# ColorAnalyzer - Google Cloud Run Deployment Guide

## 📋 Prerequisites

1. **Google Cloud Account**: Create a free account at [cloud.google.com](https://cloud.google.com)
2. **Google Cloud Project**: Create a new project in the Google Cloud Console
3. **gcloud CLI**: Install from [cloud.google.com/sdk](https://cloud.google.com/sdk/docs/install)
4. **Billing Account**: Link a billing account to your project (required for Cloud Run)

## 🚀 Quick Deployment

### Option 1: Automated Script (Recommended)

1. **Edit the project ID** in `deploy.sh` (Linux/Mac) or `deploy.bat` (Windows):
   ```bash
   PROJECT_ID="your-actual-project-id"
   ```

2. **Run the deployment script**:
   ```bash
   # Linux/Mac
   chmod +x deploy.sh
   ./deploy.sh
   
   # Windows
   deploy.bat
   ```

### Option 2: Manual Deployment

1. **Authenticate with Google Cloud**:
   ```bash
   gcloud auth login
   gcloud config set project YOUR_PROJECT_ID
   ```

2. **Enable required APIs**:
   ```bash
   gcloud services enable cloudbuild.googleapis.com
   gcloud services enable run.googleapis.com
   gcloud services enable containerregistry.googleapis.com
   ```

3. **Deploy using Cloud Build**:
   ```bash
   gcloud builds submit --config cloudbuild.yaml
   ```

4. **Get your service URL**:
   ```bash
   gcloud run services describe coloranalyzer --region=us-central1 --format="value(status.url)"
   ```

## 🔧 Configuration Options

### Resource Allocation
- **Memory**: 2GB (suitable for image processing)
- **CPU**: 2 vCPUs (for faster processing)
- **Max Instances**: 10 (auto-scaling)
- **Timeout**: 300 seconds (for large image processing)

### Environment Variables
You can add environment variables in `cloudbuild.yaml`:
```yaml
- '--set-env-vars'
- 'MAX_UPLOAD_SIZE=50MB,DEBUG=false'
```

## 🛠️ Customization

### Changing Region
Edit `cloudbuild.yaml` and deployment scripts to change the region:
```yaml
- '--region'
- 'europe-west1'  # Change to your preferred region
```

### Scaling Configuration
Modify the Cloud Build configuration:
```yaml
- '--min-instances'
- '0'
- '--max-instances'
- '20'
- '--concurrency'
- '80'
```

## 📊 Monitoring

After deployment, you can monitor your service:

1. **Cloud Console**: Visit the Cloud Run section in Google Cloud Console
2. **Logs**: View logs with `gcloud logs tail`
3. **Metrics**: Monitor CPU, memory, and request metrics

## 🔒 Security

### Authentication (Optional)
To require authentication:
```bash
gcloud run services update coloranalyzer --region=us-central1 --no-allow-unauthenticated
```

### Custom Domain (Optional)
1. Verify domain ownership in Google Cloud Console
2. Map your domain to the Cloud Run service

## 💰 Cost Estimation

Cloud Run pricing (as of 2024):
- **CPU**: $0.00002400 per vCPU-second
- **Memory**: $0.00000250 per GB-second
- **Requests**: $0.40 per million requests
- **Free Tier**: 2 million requests, 400,000 GB-seconds, 200,000 vCPU-seconds per month

Estimated cost for moderate usage: $5-20/month

## 🐛 Troubleshooting

### Common Issues

1. **Build Fails**:
   - Check that all dependencies are in `requirements.txt`
   - Verify Dockerfile syntax

2. **Service Won't Start**:
   - Check logs: `gcloud logs tail --service=coloranalyzer`
   - Verify port configuration (must use PORT environment variable)

3. **Memory Issues**:
   - Increase memory allocation in `cloudbuild.yaml`
   - Optimize image processing for large files

4. **Timeout Issues**:
   - Increase timeout in `cloudbuild.yaml`
   - Optimize processing algorithms

### Debug Commands
```bash
# View service details
gcloud run services describe coloranalyzer --region=us-central1

# View logs
gcloud logs tail --service=coloranalyzer --region=us-central1

# Update service configuration
gcloud run services update coloranalyzer --region=us-central1 --memory=4Gi
```

## 🔄 Updates

To update your deployed application:
1. Make changes to your code
2. Run the deployment script again
3. Cloud Run will automatically create a new revision

## 📞 Support

- **Google Cloud Documentation**: [cloud.google.com/run/docs](https://cloud.google.com/run/docs)
- **Community Support**: [stackoverflow.com/questions/tagged/google-cloud-run](https://stackoverflow.com/questions/tagged/google-cloud-run)
