@echo off
REM ColorAnalyzer Google Cloud Run Deployment Script for Windows

REM Set your project ID here
set PROJECT_ID=coloranalyzer-471116
set SERVICE_NAME=coloranalyzer
set REGION=us-central1

echo 🚀 Deploying ColorAnalyzer to Google Cloud Run...

REM Check if gcloud is installed
gcloud --version >nul 2>&1
if errorlevel 1 (
    echo ❌ gcloud CLI is not installed. Please install it first:
    echo https://cloud.google.com/sdk/docs/install
    pause
    exit /b 1
)

REM Check if user is authenticated
gcloud auth list --filter=status:ACTIVE --format="value(account)" | findstr "@" >nul
if errorlevel 1 (
    echo ❌ Not authenticated with gcloud. Please run:
    echo gcloud auth login
    pause
    exit /b 1
)

REM Set the project
echo 📋 Setting project to %PROJECT_ID%...
gcloud config set project %PROJECT_ID%

REM Enable required APIs
echo 🔧 Enabling required APIs...
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com

REM Build and deploy using Cloud Build
echo 🏗️ Building and deploying with Cloud Build...
gcloud builds submit --config cloudbuild.yaml

echo ✅ Deployment complete!
echo 🌐 Your ColorAnalyzer app should be available at:
echo https://%SERVICE_NAME%-[hash]-%REGION%.a.run.app

REM Get the actual URL
echo 🔍 Getting service URL...
gcloud run services describe %SERVICE_NAME% --region=%REGION% --format="value(status.url)"

pause
