#!/bin/bash
set -e

# ── CONFIG ───────────────────────────────────────────────────
# Set these in your environment before running this script:
#   export PROJECT_ID="your-gcp-project-id"
#   export GEMINI_API_KEY="your-gemini-api-key"
#   export CHAT_SPACE_ID="spaces/XXXXXXXXX"
PROJECT_ID="${PROJECT_ID:?PROJECT_ID env var is required}"
GEMINI_API_KEY="${GEMINI_API_KEY:?GEMINI_API_KEY env var is required}"
CHAT_SPACE_ID="${CHAT_SPACE_ID:?CHAT_SPACE_ID env var is required}"
REGION="us-central1"
SERVICE_NAME="momo"
# ─────────────────────────────────────────────────────────────

echo "Deploying Momo to Cloud Run..."

# Ensure we're using the right project
gcloud config set project $PROJECT_ID

# Read token.json into a variable for passing as env var
GOOGLE_TOKEN_JSON=$(cat token.json)

# Build and deploy in one step
gcloud run deploy $SERVICE_NAME \
  --source . \
  --region $REGION \
  --platform managed \
  --allow-unauthenticated \
  --set-env-vars="GEMINI_API_KEY=${GEMINI_API_KEY},GCP_PROJECT_ID=$PROJECT_ID,CHAT_SPACE_ID=${CHAT_SPACE_ID}" \
  --set-env-vars="^##^GOOGLE_TOKEN_JSON=${GOOGLE_TOKEN_JSON}" \
  --memory=1Gi \
  --timeout=120 \
  --min-instances=0 \
  --max-instances=3

# Get the URL
URL=$(gcloud run services describe $SERVICE_NAME --region=$REGION --format='value(status.url)')

echo ""
echo "Momo deployed successfully!"
echo "Momo's URL: $URL"
echo ""
echo "Next steps:"
echo "  1. Set Google Chat App HTTP endpoint to: ${URL}/chat"
echo "  2. Cloud Scheduler job should point to: ${URL}/briefing"
echo "  3. Create another Cloud Scheduler job to call: ${URL}/email-alerts (e.g. every 5 minutes)"