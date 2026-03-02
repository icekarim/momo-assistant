# Momo

Momo is a personal AI chief of staff that lives in Google Chat. It knows your inbox, your calendar, your tasks, and your meeting history — and it keeps you on top of all of it without you having to ask.

Every morning it drops a briefing in Chat: what needs your attention in your inbox, what's on your calendar, what's open in your task list, and what came out of yesterday's meetings. Throughout the day you can message it like a colleague — ask questions, get summaries, delegate tasks, or dig into what was decided in a meeting last week.

Momo also builds a **cross-meeting knowledge graph** — it extracts decisions, commitments, action items, blockers, and topics from every meeting and email, then connects dots across them over time. Ask it *"what commitments have I made this week?"* or *"what's the full history of the pricing discussion?"* and it pulls from institutional memory, not just today's data.

Built with **FastAPI** + **Gemini 3 Flash / Pro** (tiered routing), deployed on **Google Cloud Run**.

---

## What it does

### Morning briefing
At 8 AM, Momo sends a structured daily briefing covering:
- **Inbox** — prioritized summary of unread emails with triage (🔴 urgent / 🟡 needs attention / 🟢 FYI)
- **Calendar** — today's meetings and schedule
- **Tasks** — all open items from Google Tasks
- **Yesterday's meetings** — key decisions, action items, and follow-ups pulled from Granola notes

### Proactive email alerts
Runs every 5 minutes. Momo triages your inbox with Gemini and pings you in Chat the moment something genuinely needs your attention — client emails, escalations, deadlines, blockers. Each alert fires once and is never duplicated.

### Post-meeting debriefs
Within minutes of a calendar meeting ending, Momo pulls the Granola notes for that meeting and sends a short debrief to Chat: what was decided, what action items came out, and who owns them. No manual note-taking required.

### Conversational assistant
Ask Momo anything about your work context in plain language:
- *"what's on my calendar today?"*
- *"any urgent emails from clients?"*
- *"what did we decide in the product standup?"*
- *"pull up the action items from yesterday's investor call"*
- *"push all my tasks to Friday"*
- *"draft a reply to [person] about [topic]"*

Momo fetches live data (emails, calendar, tasks, meeting notes) on every message and answers with full context.

### Task management
Full CRUD over Google Tasks via natural language. Momo emits structured action tags in its response (`[CREATE_TASK]`, `[UPDATE_TASK]`, `[COMPLETE_TASK]`, `[DELETE_TASK]`) that the backend parses and executes automatically — no confirmation dialogs, just done.

### Meeting intelligence (via Granola)
Momo is connected to [Granola](https://granola.ai) via MCP and can answer questions about any recorded meeting:
- *"what were the action items from Monday's sync?"*
- *"who was in the Q1 planning meeting?"*
- *"pull the full transcript from yesterday's call"*
- *"what decisions have we made about [topic] this month?"*

The Granola token is managed automatically — it's stored in Firestore and refreshed in the background. You authenticate once with `python granola_auth_setup.py` and never think about it again.

### Cross-meeting knowledge graph

Momo builds persistent institutional memory by extracting structured entities from every meeting debrief and important email:

- **Decisions** — what was decided and when
- **Commitments** — who promised what to whom, with status tracking
- **Action items** — tasks that need to happen, with owners
- **Blockers** — what's blocking progress
- **Topics** — subjects discussed without a clear outcome
- **Updates** — status reports on ongoing work

Each entity is stored in Firestore with full provenance (source meeting/email, date, related people, related projects, tags). When you ask Momo a question that touches on history, commitments, or cross-meeting context, the knowledge graph is queried and injected alongside live data.

Example queries:
- *"what commitments have I made this week?"*
- *"what's the full history of the pricing discussion with BJ's?"*
- *"what blockers are outstanding?"*
- *"what's changed since I last met with Sarah?"*
- *"what decisions have we made about the Q2 launch?"*

Extraction happens in the background (daemon threads) so it never blocks debrief or alert delivery. You can bootstrap the graph from existing data with `POST /knowledge-backfill`.

### Tiered model routing

Momo routes requests to the right Gemini model based on complexity:

| Tier | Model | Used for |
|---|---|---|
| Light | Flash | Quick extractions, triage |
| Standard | Flash | Normal chat, briefings, debriefs |
| Deep | Pro | Knowledge graph queries requiring cross-meeting reasoning |

If Pro fails, Momo automatically falls back to Flash so you always get a response.

---

## Architecture

```
Google Chat
    │
    ▼ POST /chat (webhook)
┌──────────────────────────────────────────────────────────────┐
│                      FastAPI (main.py)                        │
│                                                              │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │                     Event Parser                        │ │
│  │         handles standard Chat + Workspace               │ │
│  │         Add-on event formats                            │ │
│  └──────────────────────────┬──────────────────────────────┘ │
│                              │                               │
│                    ┌─────────▼─────────┐                     │
│                    │   _build_context  │  fetches live data   │
│                    └─────────┬─────────┘  (7 workers)        │
│                              │                               │
│    ┌──────────┬──────────┬───┴───┬──────────┬─────────┐      │
│    ▼          ▼          ▼       ▼          ▼         ▼      │
│  Gmail    Calendar     Tasks  Granola   Knowledge  Targeted  │
│  Service  Service      Svc     MCP      Graph      Emails   │
│  (read) (today+ended) (r/w) (notes)  (Firestore)  (search)  │
│    └──────────┴──────────┴───────┴──────────┴─────────┘      │
│                              │                               │
│                    ┌─────────▼─────────┐                     │
│                    │   Gemini Service  │  tiered routing:     │
│                    │  Flash ◄──► Pro   │  Standard → Flash    │
│                    │                   │  Deep → Pro          │
│                    └─────────┬─────────┘                     │
│                              │                               │
│         ┌────────────────────▼──────────────────┐            │
│         │          Task Action Extractor         │            │
│         │    parses [CREATE_TASK],               │            │
│         │    [UPDATE_TASK], etc. tags            │            │
│         └────────────────────┬──────────────────┘            │
│                              │ executes actions              │
│                    ┌─────────▼─────────┐                     │
│                    │   Tasks Service   │  create/update/      │
│                    │    (write ops)    │  complete/delete     │
│                    └─────────┬─────────┘                     │
│                              │                               │
│                    ┌─────────▼─────────┐                     │
│                    │   Conversation    │  Firestore           │
│                    │   Store           │  (per-user turns)    │
│                    └─────────┬─────────┘                     │
└──────────────────────────────┼───────────────────────────────┘
                               │ formatted reply
                               ▼
                         Google Chat


Cloud Scheduler ──► POST /briefing
                         │
                ┌────────┴────────────────┐
                ▼                         ▼
           Gmail/Calendar/Tasks       Granola MCP
           (morning context)       (yesterday's notes)
                └────────┬────────────────┘
                         ▼
                   Gemini (briefing)
                         ▼
                   Google Chat

Cloud Scheduler ──► POST /meeting-debrief
                         │
                ┌────────┴────────────────┐
                ▼                         ▼
        Calendar Service             Granola MCP
      (recently ended mtgs)        (notes for mtg)
                └────────┬────────────────┘
                         │
                   ┌─────┴──────┐
                   ▼            ▼
             Gemini         Knowledge
            (debrief)     Graph Extract
                │         (background)
                ▼            ▼
          Google Chat    Firestore
```

### Scheduled jobs (Cloud Scheduler)

```
Cloud Scheduler ──► POST /briefing           (daily at 8 AM)
Cloud Scheduler ──► POST /email-alerts       (every 5 minutes)
Cloud Scheduler ──► POST /meeting-debrief    (every 10 minutes, work hours)
One-time         ──► POST /knowledge-backfill (bootstrap knowledge graph)
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
5. Triggers background knowledge graph extraction for each alerted email

**`/meeting-debrief`** — also in `briefing.py`:
1. Fetches today's calendar meetings whose end time falls within the lookback window (default 120 min)
2. Builds a title-to-ID map from Granola (single API call), then batch-fetches notes
3. Skips any already debriefed (checked against Firestore `meeting_debriefs` collection)
4. Sends a short debrief to Chat with key decisions and action items; marks meeting as debriefed
5. Triggers background knowledge graph extraction for meetings with notes

**`/knowledge-backfill`** — in `main.py`:
1. Fetches the last 30 days of Granola meetings and recent inbox emails
2. Batch-extracts entities (decisions, commitments, blockers, etc.) via Gemini Flash
3. Stores everything in the `knowledge_graph` Firestore collection
4. Idempotent — skips sources that have already been processed
5. Run once after first deploy to populate the graph with existing data

---

## File structure

```
momo/
├── main.py               # FastAPI app, endpoints, message handling, task action execution
├── gemini_service.py     # Gemini API wrapper; tiered model routing; system prompt; chat/briefing/debrief
├── knowledge_graph.py    # Entity extraction, Firestore storage, and query functions for institutional memory
├── briefing.py           # Morning briefing, proactive email alerts, post-meeting debrief pipelines
├── gmail_service.py      # Gmail API: fetch unread emails, search, format for context
├── calendar_service.py   # Google Calendar API: fetch today's events, recently ended meetings
├── tasks_service.py      # Google Tasks API: fetch, create, update, complete, delete tasks
├── chat_service.py       # Google Chat API: send_chat_message(), format_for_google_chat()
├── conversation_store.py # Firestore-backed conversation history, email alert + debrief dedup
├── granola_service.py    # Granola MCP client: fetch meeting notes, transcripts, auto token refresh
├── granola_auth_setup.py # One-time local OAuth flow to authenticate with Granola (saves token)
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
[COMPLETE_TASK] find="Lowes analysis"
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
_build_context() — 7 concurrent workers:
  - always: today's meetings + open tasks
  - if email keywords: inbox emails + targeted search
  - if meeting/notes keywords: Granola MCP query
  - if history/commitment/decision keywords: knowledge graph query
    │
    ▼
Gemini chat_response() — tiered model routing:
  - Standard (Flash): normal queries
  - Deep (Pro): when knowledge graph context is present
  - Auto-fallback: Pro → Flash on failure
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
| Cloud Firestore | Conversation history, email/debrief dedup, knowledge graph, Granola token | via ADC / service account |
| Cloud Run | Hosts the FastAPI server | N/A |
| Cloud Scheduler | Triggers `/briefing`, `/email-alerts`, and `/meeting-debrief` | N/A |
| Granola MCP | Fetch meeting notes, transcripts, action items | OAuth 2.0 (PKCE) |
| Gemini Flash | Chat, briefings, debriefs, entity extraction, email triage | API key |
| Gemini Pro | Deep reasoning for knowledge graph queries | API key |

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

### 5. Granola integration (optional)

Momo can pull meeting notes, decisions, and action items from [Granola](https://granola.ai).

**Authenticate once:**

```bash
python granola_auth_setup.py
```

This opens a browser OAuth flow, saves the token to `granola_token.json` (gitignored), and syncs it to Firestore so Cloud Run can use it automatically. The token auto-refreshes — you only need to run this once.

**Enable it in `.env`:**

```env
GRANOLA_ENABLED=true
```

If Granola is disabled, all meeting-note features are silently skipped.

### 6. Run locally

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
3. Create Cloud Scheduler jobs:
   - `POST https://<url>/briefing` — daily at 8 AM (or your preferred time)
   - `POST https://<url>/email-alerts` — every 5 minutes
   - `POST https://<url>/meeting-debrief` — every 10 minutes during work hours (optional, requires Granola)
4. (One-time) Bootstrap the knowledge graph: `curl -X POST https://<url>/knowledge-backfill`

---

## Firestore collections

| Collection | Document key | Purpose |
|---|---|---|
| `conversations` | sanitized user ID | Stores up to 50 conversation turns per user |
| `email_alerts` | Gmail message ID | Tracks which emails have already triggered an alert |
| `meeting_debriefs` | Google Calendar event ID | Tracks which meetings have already been debriefed |
| `knowledge_graph` | auto-generated | Extracted entities (decisions, commitments, blockers, etc.) with provenance |
| `granola_auth` | `token` | Stores the Granola OAuth token for Cloud Run (auto-refreshed) |

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
| `KNOWLEDGE_GRAPH_ENABLED` | No | `true` | Toggle knowledge graph extraction and querying |
| `GEMINI_MODEL_FLASH` | No | `gemini-3-flash-preview` | Model for standard/light tasks |
| `GEMINI_MODEL_PRO` | No | `gemini-3.1-pro-preview` | Model for deep reasoning (knowledge graph queries) |
| `GRANOLA_ENABLED` | No | `false` | Enable Granola meeting notes integration |
| `GRANOLA_MCP_URL` | No | `https://mcp.granola.ai/mcp` | Granola MCP server URL |
| `GRANOLA_TOKEN_JSON` | Cloud Run | — | Full Granola token JSON as a string (seeds Firestore on first boot) |
| `MEETING_DEBRIEF_LOOKBACK_MINUTES` | No | `120` | How far back to look for recently ended meetings |
| `MEETING_DEBRIEF_GRACE_MINUTES` | No | `45` | Grace period before sending debrief without notes |
| `PORT` | No | `8080` | Server port |
