# Momo

A personal AI assistant that lives in Google Chat. Every morning it delivers a briefing of your emails, calendar, and tasks. Throughout the day you can ask it anything about your inbox, schedule, or to-do list — and it can create, update, complete, and delete tasks on your behalf.

Built with FastAPI + Gemini 2.5 Flash, deployed on Google Cloud Run.

---

## What it does

- **Daily briefing** — at 8 AM, Momo pushes a formatted summary of your unread emails, today's meetings, and open tasks directly into your Google Chat space
- **Proactive email alerts** — runs on a 5-minute interval, uses Gemini to triage your inbox and pings you in Chat if something genuinely needs your attention (clients, deadlines, escalations)
- **Conversational assistant** — ask anything: *"what's on my calendar today?"*, *"any urgent emails?"*, *"draft a reply to [person]"*, *"push all my tasks to Friday"*
- **Task management** — full CRUD over Google Tasks via natural language; Momo emits structured action tags in its response that the backend executes automatically

---

## Architecture

```
Google Chat
    │
    ▼ POST /chat (webhook)
┌──────────────────────────────────────────────┐
│                FastAPI (main.py)             │
│                                              │
│  ┌─────────────────────────────────────────┐ │
│  │           Event Parser                  │ │
│  │  handles standard Chat + Workspace      │ │
│  │  Add-on event formats                   │ │
│  └──────────────┬──────────────────────────┘ │
│                 │                             │
│         ┌───────▼────────┐                   │
│         │ _build_context │  fetches live data │
│         └───────┬────────┘  (always: meetings│
│                 │            + tasks; emails  │
│    ┌────────────┼────────────┐  if relevant)  │
│    ▼            ▼            ▼                │
│  Gmail      Calendar       Tasks              │
│  Service    Service        Service            │
│    │            │            │                │
│    └────────────┴────────────┘                │
│                 │                             │
│         ┌───────▼────────┐                   │
│         │ Gemini Service │  chat_response()   │
│         │  (2.5 Flash)   │  with conversation │
│         │                │  history + context │
│         └───────┬────────┘                   │
│                 │                             │
│   ┌─────────────▼──────────────┐             │
│   │   Task Action Extractor    │             │
│   │  parses [CREATE_TASK],     │             │
│   │  [UPDATE_TASK], etc. tags  │             │
│   │  from Gemini's response    │             │
│   └─────────────┬──────────────┘             │
│                 │ executes actions            │
│         ┌───────▼────────┐                   │
│         │  Tasks Service │  create/update/   │
│         │   (write ops)  │  complete/delete  │
│         └───────┬────────┘                   │
│                 │                             │
│         ┌───────▼────────┐                   │
│         │ Conversation   │  Firestore         │
│         │ Store          │  (per-user turns)  │
│         └───────┬────────┘                   │
└─────────────────┼────────────────────────────┘
                  │ formatted reply
                  ▼
           Google Chat
```

### Scheduled jobs (Cloud Scheduler)

```
Cloud Scheduler ──► POST /briefing      (daily at 8 AM)
Cloud Scheduler ──► POST /email-alerts  (every 5 minutes)
```

**`/briefing`** — orchestrated by `briefing.py`:
1. Fetches unread client emails, today's meetings, all open tasks
2. Sends all three as context to Gemini → formatted morning briefing
3. Posts the result to the configured Google Chat space

**`/email-alerts`** — also in `briefing.py`:
1. Fetches recent inbox emails matching the configured query
2. Filters out any already sent (checked against Firestore)
3. Batches up to 10 unseen emails → Gemini triage (returns JSON with `alert: true/false`, `priority`, `summary`)
4. Posts alerts to Chat for emails that pass triage; marks each as sent in Firestore to prevent duplicates

---

## File structure

```
momo/
├── main.py               # FastAPI app, endpoints, message handling, task action execution
├── gemini_service.py     # Gemini API wrapper; system prompt; chat_response(); generate_morning_briefing()
├── briefing.py           # Morning briefing + proactive email alert pipelines
├── gmail_service.py      # Gmail API: fetch unread emails, search, format for context
├── calendar_service.py   # Google Calendar API: fetch today's events, format for context
├── tasks_service.py      # Google Tasks API: fetch, create, update, complete, delete tasks
├── chat_service.py       # Google Chat API: send_chat_message(), format_for_google_chat()
├── conversation_store.py # Firestore-backed per-user conversation history + email alert dedup
├── google_auth.py        # OAuth credential loading (file for local dev, env var for Cloud Run)
├── auth_setup.py         # One-time local script to generate token.json via browser OAuth flow
├── config.py             # All config loaded from environment variables
├── Dockerfile            # python:3.12-slim, uvicorn entrypoint
├── deploy.sh             # gcloud run deploy wrapper (reads secrets from env vars)
├── requirements.txt
├── .env.example          # Template for local environment variables
└── .gitignore
```

---

## How task management works

Gemini is instructed to emit structured tags at the end of its response when a task action is needed:

```
[CREATE_TASK] title="Call Sarah" due="2026-02-18"
[UPDATE_TASK] find="Review proposal" due="2026-02-21"
[COMPLETE_TASK] find="ClientB analysis"
[DELETE_TASK] find="test task"
```

`main.py` parses these tags with regex, executes each action against the Google Tasks API, strips the tags from the user-visible response, and appends a summary of what was done. A prose fallback parser (`_fallback_parse_prose`) catches cases where Gemini describes an action in natural language without emitting a tag.

---

## Data flow for a chat message

```
User message
    │
    ▼
Parse event format (standard Chat vs. Workspace Add-on)
    │
    ▼
Load conversation history from Firestore
    │
    ▼
_build_context():
  - always fetches: today's meetings + open tasks
  - fetches emails if message contains email-related keywords or is a general query
    │
    ▼
Gemini chat_response():
  - injects date reference + context as first turn in history
  - appends stored conversation history
  - sends user message
    │
    ▼
Parse task action tags from response
    │
    ├─► Execute task actions (Google Tasks API)
    │
    ▼
Save turn to Firestore
    │
    ▼
format_for_google_chat() → send reply
```

---

## Google services used

| Service | Usage | Scopes |
|---|---|---|
| Gmail API | Read inbox, search emails | `gmail.readonly` |
| Google Calendar API | Read today's events | `calendar.readonly` |
| Google Tasks API | Read + write tasks | `tasks` |
| Google Chat API | Receive webhooks, send messages | N/A (bot config) |
| Cloud Firestore | Conversation history, email alert deduplication | via ADC / service account |
| Cloud Run | Hosts the FastAPI server | N/A |
| Cloud Scheduler | Triggers `/briefing` and `/email-alerts` | N/A |

---

## Setup

### Prerequisites

- Python 3.12+
- A Google Cloud project with the following APIs enabled:
  - Gmail API
  - Google Calendar API
  - Google Tasks API
  - Google Chat API
  - Cloud Firestore
- A Google Chat App (bot) configured with your Cloud Run URL as the HTTP endpoint
- A Gemini API key from [Google AI Studio](https://aistudio.google.com/)

### 1. Clone and install

```bash
git clone https://github.com/icekarim/momo-assistant.git
cd momo-assistant
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Google OAuth credentials

In the [Google Cloud Console](https://console.cloud.google.com/):

1. Go to **APIs & Services → Credentials**
2. Create an **OAuth 2.0 Client ID** (Application type: Web application)
3. Add `http://localhost:8080/` to Authorized redirect URIs
4. Download the JSON and save it as `client_secret.json` in the project root

### 3. Generate token.json (one-time local auth)

```bash
python auth_setup.py
```

This opens a browser OAuth flow and writes `token.json`. Both files are gitignored and never committed.

### 4. Environment variables

Copy the example and fill in your values:

```bash
cp .env.example .env
```

```env
GEMINI_API_KEY=your-gemini-api-key
GCP_PROJECT_ID=your-gcp-project-id

# Paths to local credential files
GOOGLE_CLIENT_SECRET_FILE=client_secret.json
GOOGLE_TOKEN_FILE=token.json

# Google Chat space to post briefings to
# Format: spaces/XXXXXXXXX — you'll see this logged on the first message Momo receives
CHAT_SPACE_ID=

# Proactive email alerts
EMAIL_ALERTS_ENABLED=true
EMAIL_ALERT_GMAIL_QUERY=is:unread in:inbox newer_than:2d
EMAIL_ALERTS_MAX_PER_RUN=5
IMPORTANT_EMAIL_KEYWORDS=urgent,asap,important,action required,deadline,escalation,blocker
CLIENT_EMAIL_KEYWORDS=client,customer
CLIENT_DOMAINS=                          # e.g. acme.com,clientco.io
```

### 5. Run locally

```bash
uvicorn main:app --host 0.0.0.0 --port 8080 --reload
```

Use [ngrok](https://ngrok.com/) to expose the local server and point your Google Chat App HTTP endpoint to `https://<your-ngrok-url>/chat`.

---

## Deployment (Google Cloud Run)

Set your secrets in the environment, then run:

```bash
export PROJECT_ID="your-gcp-project-id"
export GEMINI_API_KEY="your-gemini-api-key"
export CHAT_SPACE_ID="spaces/XXXXXXXXX"

./deploy.sh
```

The script passes `GOOGLE_TOKEN_JSON` (the contents of `token.json`) as an environment variable on Cloud Run so no credential files need to be bundled in the image.

After deploying:
1. Copy the Cloud Run URL
2. Set it as the HTTP endpoint in your Google Chat App config: `https://<url>/chat`
3. Create two Cloud Scheduler jobs:
   - `POST https://<url>/briefing` — daily at 8 AM (or your preferred time)
   - `POST https://<url>/email-alerts` — every 5 minutes

---

## Firestore collections

| Collection | Document key | Purpose |
|---|---|---|
| `conversations` | sanitized user ID | Stores up to 50 conversation turns per user |
| `email_alerts` | Gmail message ID | Tracks which emails have already triggered an alert |

---

## Environment variable reference

| Variable | Required | Default | Description |
|---|---|---|---|
| `GEMINI_API_KEY` | Yes | — | Gemini API key from Google AI Studio |
| `GCP_PROJECT_ID` | Yes | — | GCP project ID (used for Firestore) |
| `GOOGLE_CLIENT_SECRET_FILE` | Local only | `client_secret.json` | Path to OAuth client secret |
| `GOOGLE_TOKEN_FILE` | Local only | `token.json` | Path to OAuth token |
| `GOOGLE_TOKEN_JSON` | Cloud Run | — | Full token JSON as a string (replaces token file) |
| `CHAT_SPACE_ID` | Yes | — | Google Chat space to post to (`spaces/XXX`) |
| `EMAIL_ALERTS_ENABLED` | No | `true` | Toggle proactive email alerts |
| `EMAIL_ALERT_GMAIL_QUERY` | No | `is:unread in:inbox newer_than:2d` | Gmail search query for alert candidates |
| `EMAIL_ALERTS_MAX_PER_RUN` | No | `5` | Max alerts to send per scheduler run |
| `IMPORTANT_EMAIL_KEYWORDS` | No | `urgent,asap,...` | Keywords that flag an email as important |
| `CLIENT_EMAIL_KEYWORDS` | No | `client,customer` | Keywords that identify client emails |
| `CLIENT_DOMAINS` | No | — | Comma-separated domains to treat as clients |
| `PORT` | No | `8080` | Server port |
