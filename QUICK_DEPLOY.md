# 🚀 Quick Deploy to Google Cloud Run

## Step 1: Setup Google Cloud

1. **Create Google Cloud Project**:

   - Go to [console.cloud.google.com](https://console.cloud.google.com)
   - Create a new project
   - Note your PROJECT_ID

2. **Install gcloud CLI**:
   - Download from [cloud.google.com/sdk](https://cloud.google.com/sdk/docs/install)
   - Run: `gcloud auth login`

## Step 2: Configure Project

1. **Edit deployment files**:

   - Open `deploy.sh` (Linux/Mac) or `deploy.bat` (Windows)
   - Replace `your-project-id` with your actual PROJECT_ID

2. **Set up billing** (required for Cloud Run):
   - Go to Google Cloud Console → Billing
   - Link a billing account to your project

## Step 3: Deploy

### Windows:

```cmd
deploy.bat
```

### Linux/Mac:

```bash
chmod +x deploy.sh
./deploy.sh
```

## Step 4: Access Your App

After deployment completes, you'll get a URL like:

```
https://coloranalyzer-sadqrvcttsa-uc.a.run.app
```

## 💰 Cost

- **Free Tier**: 2M requests/month
- **Typical Usage**: $5-20/month
- **Pay-per-use**: Only charged when app is running

## 🔧 Troubleshooting

**Build fails?**

```bash
gcloud logs tail --service=coloranalyzer
```

**Need more memory?**
Edit `cloudbuild.yaml` and change `--memory` to `4Gi`

**Custom domain?**
Use Google Cloud Console → Cloud Run → Manage Custom Domains
