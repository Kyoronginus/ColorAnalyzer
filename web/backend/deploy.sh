#!/bin/bash

# ColorAnalyzer Google Cloud Run Deployment Script

# Set your project ID here
PROJECT_ID="your-project-id"
SERVICE_NAME="coloranalyzer"
REGION="us-central1"

echo "🚀 Deploying ColorAnalyzer to Google Cloud Run..."

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "❌ gcloud CLI is not installed. Please install it first:"
    echo "https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Check if user is authenticated
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    echo "❌ Not authenticated with gcloud. Please run:"
    echo "gcloud auth login"
    exit 1
fi

# Set the project
echo "📋 Setting project to $PROJECT_ID..."
gcloud config set project $PROJECT_ID

# Enable required APIs
echo "🔧 Enabling required APIs..."
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com

# Build and deploy using Cloud Build
echo "🏗️ Building and deploying with Cloud Build..."
gcloud builds submit --config cloudbuild.yaml

echo "✅ Deployment complete!"
echo "🌐 Your ColorAnalyzer app should be available at:"
echo "https://$SERVICE_NAME-[hash]-$REGION.a.run.app"

# Get the actual URL
echo "🔍 Getting service URL..."
gcloud run services describe $SERVICE_NAME --region=$REGION --format="value(status.url)"
