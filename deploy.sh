#!/bin/bash
set -e

# ── CONFIG ───────────────────────────────────────────────────
# Set these in your environment before running this script:
#   export PROJECT_ID="your-gcp-project-id"
#   export GEMINI_API_KEY="your-gemini-api-key"
#   export CHAT_SPACE_ID="spaces/XXXXXXXXX"
PROJECT_ID="${PROJECT_ID:?PROJECT_ID env var is required}"
GEMINI_API_KEY="${GEMINI_API_KEY:?GEMINI_API_KEY env var is required}"
CHAT_SPACE_ID="${CHAT_SPACE_ID:-$(grep '^CHAT_SPACE_ID=' .env 2>/dev/null | cut -d= -f2)}"
if [ -z "$CHAT_SPACE_ID" ]; then
  echo "WARNING: CHAT_SPACE_ID is not set. Briefings will print to console only."
fi
REGION="us-central1"
SERVICE_NAME="momo"
# ─────────────────────────────────────────────────────────────

echo "Deploying Momo to Cloud Run..."

# Ensure we're using the right project
gcloud config set project $PROJECT_ID

# Sync Granola token to Firestore so Cloud Run has the latest (with _client_id etc.)
if [ -f granola_token.json ]; then
    echo "Syncing Granola token to Firestore..."
    python3 -c "
from granola_service import _write_token_to_firestore
import json, time
with open('granola_token.json') as f:
    token = json.load(f)
if '_expires_at' not in token:
    import os
    token['_expires_at'] = os.path.getmtime('granola_token.json') + token.get('expires_in', 21600)
_write_token_to_firestore(token)
print('  Granola token synced to Firestore')
" 2>/dev/null || echo "  (Firestore sync skipped — will use env var fallback)"
fi

# Read token files into variables for passing as env vars
GOOGLE_TOKEN_JSON=$(cat token.json)
GRANOLA_TOKEN_JSON=$(cat granola_token.json 2>/dev/null || echo "")

# Build and deploy in one step
gcloud run deploy $SERVICE_NAME \
  --source . \
  --region $REGION \
  --platform managed \
  --allow-unauthenticated \
  --set-env-vars="GEMINI_API_KEY=${GEMINI_API_KEY},GCP_PROJECT_ID=$PROJECT_ID,CHAT_SPACE_ID=${CHAT_SPACE_ID},GRANOLA_ENABLED=true" \
  --set-env-vars="^##^GOOGLE_TOKEN_JSON=${GOOGLE_TOKEN_JSON}##GRANOLA_TOKEN_JSON=${GRANOLA_TOKEN_JSON}" \
  --memory=1Gi \
  --timeout=300 \
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
echo "  4. Create Cloud Scheduler job for: ${URL}/meeting-debrief (e.g. */10 9-18 * * 1-5)"
echo "  5. (One-time) Backfill knowledge graph: curl -X POST ${URL}/knowledge-backfill"