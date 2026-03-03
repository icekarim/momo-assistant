# Momo

Momo is a personal AI chief of staff that lives in Google Chat. It knows your inbox, your calendar, your tasks, and your meeting history — and it keeps you on top of all of it without you having to ask.

Every morning it drops a briefing in Chat: what needs your attention in your inbox, what's on your calendar, what's open in your task list, and what came out of yesterday's meetings. Throughout the day you can message it like a colleague — ask questions, get summaries, delegate tasks, or dig into what was decided in a meeting last week.

Momo builds a **cross-meeting knowledge graph** — it extracts decisions, commitments, action items, blockers, and topics from every meeting, email, and chat conversation, then connects dots across them over time. The knowledge graph is queried on every message, so Momo always has institutional memory alongside live data.

On top of that, Momo runs a **proactive intelligence engine** — it surfaces insights before you ask. Pre-meeting prep briefs, commitment follow-ups with auto-resolution, pattern detection across meetings, and drift alerts for projects that have gone quiet. All delivered as nudges in your morning briefing or as standalone Chat messages for urgent items.

Built with **FastAPI** + **Gemini Flash / Pro** (tiered routing), deployed on **Google Cloud Run**.

---

## What it does

### Morning briefing
At 8 AM, Momo sends a structured daily briefing covering:
- **Inbox** — prioritized summary of unread emails with triage (urgent / needs attention / FYI)
- **Calendar** — today's meetings and schedule
- **Tasks** — all open items from Google Tasks
- **Yesterday's meetings** — key decisions, action items, and follow-ups pulled from Granola notes
- **Momo's nudges** — proactive insights from the intelligence engine: overdue commitments, patterns, stale items

### Proactive email alerts
Runs every 5 minutes. Momo triages your inbox with Gemini and pings you in Chat the moment something genuinely needs your attention — client emails, escalations, deadlines, blockers. Each alert fires once and is never duplicated. Alerted emails are also extracted into the knowledge graph for future cross-referencing.

### Post-meeting debriefs
Within minutes of a calendar meeting ending, Momo pulls the Granola notes for that meeting and sends a short debrief to Chat: what was decided, what action items came out, and who owns them. If Granola hasn't finished processing the notes yet, Momo defers and retries until a configurable grace period expires (default 45 min), at which point it sends whatever is available. Debriefed meetings are extracted into the knowledge graph automatically.

### Pre-meeting prep briefs
Before each meeting (configurable, default 1 hour ahead), Momo queries the knowledge graph for context about the attendees — past interactions, open commitments involving them, related project history, and outstanding blockers — then sends a short prep brief to Chat. Each prep fires once per calendar event.

### Conversational assistant
Ask Momo anything about your work context in plain language:
- *"what's on my calendar today?"*
- *"any urgent emails from clients?"*
- *"what did we decide in the product standup?"*
- *"pull up the action items from yesterday's investor call"*
- *"push all my tasks to Friday"*
- *"draft a reply to [person] about [topic]"*
- *"what commitments have I made this week?"*
- *"what's the full history of the pricing discussion?"*

Momo fetches live data on every message using up to 7 concurrent workers (Gmail, Calendar, Tasks, Granola, knowledge graph, targeted email search) with a 90-second timeout per source. Context is assembled and sent to Gemini alongside conversation history (up to 50 turns per user, stored in Firestore).

Chat messages are also extracted into the knowledge graph in the background, so context from conversations is captured alongside meetings and emails.

### Personality
Momo is casual and concise — lowercase by default, minimal formatting, no corporate filler. It uses structured sections with emoji headers for briefings and priority colors for items. Responses are designed to be scannable, not verbose.

### Task management
Full CRUD over Google Tasks via natural language. Momo emits structured action tags in its response that the backend parses and executes automatically:

```
[CREATE_TASK] title="Call Sarah" due="2026-02-18" notes="Re: Q2 plan"
[UPDATE_TASK] find="Review proposal" due="2026-02-21"
[COMPLETE_TASK] find="Send deck"
[DELETE_TASK] find="test task"
```

Tags are parsed with regex, executed against the Google Tasks API, stripped from the user-visible response, and a summary of what was done is appended. A prose fallback parser catches cases where Gemini describes an action in natural language without emitting a tag (e.g. *"I moved your tasks to Friday"* is detected and executed). Bulk operations like *"push all my tasks to Friday"* are also supported.

### Meeting intelligence (via Granola)
Momo is connected to [Granola](https://granola.ai) via the Model Context Protocol (MCP) and can answer questions about any recorded meeting:
- *"what were the action items from Monday's sync?"*
- *"who was in the planning meeting?"*
- *"pull the full transcript from yesterday's call"*
- *"what decisions have we made about [topic] this month?"*

The Granola token is managed automatically — it's stored in Firestore and refreshed in the background using OAuth 2.0 with PKCE. Token resolution order: in-memory cache → local file → Firestore → environment variable. You authenticate once with `python granola_auth_setup.py` and never think about it again.

Calendar meetings are matched to Granola notes using fuzzy title matching, and notes are batch-fetched (up to 10 at a time) to minimize API calls.

### Cross-meeting knowledge graph

Momo builds persistent institutional memory by extracting structured entities from every meeting debrief, important email, and chat conversation:

- **Decisions** — what was decided and when
- **Commitments** — who promised what to whom, with status tracking
- **Action items** — tasks that need to happen, with owners
- **Blockers** — what's blocking progress
- **Topics** — subjects discussed without a clear outcome
- **Updates** — status reports on ongoing work

Each entity is stored in Firestore with full provenance (source type, source ID, source title, date, related people, related projects, tags). The knowledge graph is queried on every chat message — no keyword gating — so Momo always has access to institutional memory. Queries are routed intelligently: by person, by project, by entity type (commitments, blockers, decisions), or by tag-based search.

Example queries:
- *"what commitments have I made this week?"*
- *"what's the full history of the pricing discussion?"*
- *"what blockers are outstanding?"*
- *"what's changed since I last met with [person]?"*
- *"what decisions have we made about the launch?"*

Extraction happens in the background (daemon threads) so it never blocks debrief or alert delivery. You can bootstrap the graph from existing data with `POST /knowledge-backfill`.

### Proactive intelligence

Momo doesn't just answer questions — it surfaces insights before you think to ask. Four engines run daily (during the morning briefing) and throughout the day:

**1. Pre-meeting prep** — Before each meeting, Momo pulls relevant context from the knowledge graph: past interactions with attendees, open commitments involving them, related project history, and outstanding blockers. Delivered as a brief to Chat before the meeting starts (configurable lookahead, default 1 hour).

**2. Commitment follow-up** — Scans the knowledge graph for open commitments older than 3 days (configurable). Cross-references Gmail and Google Tasks for evidence of completion — if a matching email was sent or a task was completed, the commitment is auto-resolved in the knowledge graph. Otherwise, it's flagged as a nudge with days elapsed and source context. High-priority nudges (commitments overdue by 2x the threshold) are sent as standalone Chat messages immediately.

**3. Pattern detection** — Analyzes the last 30 days of knowledge graph entries to spot recurring people, projects, and tags. Feeds the patterns into Gemini for interpretation — surfaces things like recurring topics that need a dedicated discussion or frequent collaborators worth noting.

**4. Drift detection** — Flags open items (commitments, action items) and projects that haven't been mentioned in any meeting or email for 14+ days (configurable). Helps catch things that silently fell off the radar.

All nudges are deduplicated via Firestore with a configurable cooldown (default 7 days) so you're never spammed with the same reminder. Nudges are delivered in two modes:
- **Briefing** — included as a section in the morning briefing (default for medium/low priority)
- **Standalone** — sent as immediate Chat messages (for high-priority items like severely overdue commitments)

### Tiered model routing

Momo routes requests to the right Gemini model based on complexity:

| Tier | Model | Used for |
|---|---|---|
| Light | Flash | Quick extractions, triage, entity extraction |
| Standard | Flash | Normal chat, briefings, debriefs, meeting prep |
| Deep | Pro | Knowledge graph queries requiring cross-meeting reasoning |

Each tier has its own timeout (Light: 30s, Standard: 60s, Deep: 120s). If Pro fails, Momo automatically falls back to Flash so you always get a response.

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
│                    └─────────┬─────────┘  (7 workers, 90s)   │
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
│         │    regex + prose fallback parser        │            │
│         └────────────────────┬──────────────────┘            │
│                              │ executes actions              │
│                    ┌─────────▼─────────┐                     │
│                    │   Tasks Service   │  create/update/      │
│                    │    (write ops)    │  complete/delete     │
│                    └─────────┬─────────┘                     │
│                              │                               │
│              ┌───────────────┼───────────────┐               │
│              ▼               ▼               ▼               │
│        Conversation      Knowledge       Chat API            │
│        Store (turns)   Graph Extract   (send reply)          │
│        (Firestore)     (background)                          │
└──────────────────────────────────────────────────────────────┘


Cloud Scheduler ──► POST /briefing
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
    Gmail/Calendar   Granola MCP   Proactive
    /Tasks           (yesterday)   Intelligence
          └──────────────┼──────────────┘
                         ▼
                   Gemini (briefing)
                         ▼
                   Google Chat

Cloud Scheduler ──► POST /email-alerts
                         │
                    Gmail Service
                   (alert candidates)
                         │
                    Gemini (triage)
                         │
                   ┌─────┴──────┐
                   ▼            ▼
             Google Chat    Knowledge
            (send alert)  Graph Extract
                          (background)

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

Cloud Scheduler ──► POST /meeting-prep
                         │
                ┌────────┴────────────────┐
                ▼                         ▼
        Calendar Service           Knowledge Graph
      (upcoming meetings)         (people, projects,
                │                  commitments)
                └────────┬────────────────┘
                         ▼
                   Gemini (prep brief)
                         ▼
                   Google Chat
```

### Endpoints

| Endpoint | Method | Trigger | Description |
|---|---|---|---|
| `/` | GET | On-demand | Index / status check |
| `/health` | GET | Load balancer | Health check |
| `/chat` | GET/POST | Google Chat webhook | Receive and respond to user messages |
| `/briefing` | POST | Cloud Scheduler (daily) | Generate and send morning briefing |
| `/email-alerts` | POST | Cloud Scheduler (every 5 min) | Triage inbox and send urgent alerts |
| `/meeting-debrief` | POST | Cloud Scheduler (every 10 min) | Post-meeting debriefs with Granola notes |
| `/meeting-prep` | POST | Cloud Scheduler (every 10 min) | Pre-meeting prep briefs with KG context |
| `/knowledge-backfill` | POST | One-time | Bootstrap knowledge graph from existing data |

### Scheduled jobs (Cloud Scheduler)

```
Cloud Scheduler ──► POST /briefing           (daily at 8 AM)
Cloud Scheduler ──► POST /email-alerts       (every 5 minutes)
Cloud Scheduler ──► POST /meeting-debrief    (every 10 minutes, work hours)
Cloud Scheduler ──► POST /meeting-prep       (every 10 minutes, work hours)
One-time         ──► POST /knowledge-backfill (bootstrap knowledge graph)
```

**`/briefing`** — orchestrated by `briefing.py`:
1. Fetches unread emails, today's meetings, all open tasks (in parallel)
2. Optionally fetches yesterday's meeting notes from Granola
3. Runs proactive intelligence engines (commitment follow-up, pattern detection, drift detection)
4. Sends all context + nudges to Gemini → formatted morning briefing
5. Posts the result to the configured Google Chat space

**`/email-alerts`** — also in `briefing.py`:
1. Fetches recent inbox emails matching the configured query
2. Filters out any already alerted (checked against Firestore)
3. Batches up to 10 unseen emails → Gemini triage (returns JSON with `alert: true/false`, `priority`, `summary`)
4. Posts alerts to Chat for emails that pass triage; marks each as sent in Firestore to prevent duplicates
5. Triggers background knowledge graph extraction for each alerted email

**`/meeting-debrief`** — also in `briefing.py`:
1. Fetches today's calendar meetings whose end time falls within the lookback window (default 120 min)
2. Builds a title-to-ID map from Granola (fuzzy matching), then batch-fetches notes
3. Skips any already debriefed (checked against Firestore)
4. If notes aren't ready yet and grace period hasn't expired, defers to next run
5. Sends a short debrief to Chat with key decisions and action items; marks meeting as debriefed
6. Triggers background knowledge graph extraction for meetings with notes

**`/meeting-prep`** — orchestrated by `proactive_intelligence.py`:
1. Fetches upcoming meetings within the lookahead window (default 1 hour)
2. For each unsent meeting, queries the knowledge graph for attendee history, open commitments, and related projects
3. Generates a short prep brief via Gemini Flash with key context, open items, and talking points
4. Posts the brief to Chat and marks it as sent in Firestore to prevent duplicates

**`/knowledge-backfill`** — in `main.py`:
1. Fetches the last 30 days of Granola meetings and recent inbox emails
2. Batch-extracts entities (decisions, commitments, blockers, etc.) via Gemini Flash
3. Stores everything in the `knowledge_graph` Firestore collection
4. Idempotent — skips sources that have already been processed
5. Runs in a background thread so the endpoint returns immediately

---

## File structure

```
momo/
├── main.py                    # FastAPI app, endpoints, event parsing, context building, task execution
├── gemini_service.py          # Gemini API wrapper, tiered model routing, system prompt, chat/briefing/debrief
├── proactive_intelligence.py  # Proactive intelligence: pre-meeting prep, commitment follow-up, patterns, drift
├── knowledge_graph.py         # Entity extraction, Firestore storage, querying, smart routing by intent
├── briefing.py                # Morning briefing, proactive email alerts, post-meeting debrief pipelines
├── gmail_service.py           # Gmail API: fetch unread, search, alert candidates, format for context
├── calendar_service.py        # Calendar API: today's events, upcoming, recently ended, format for context
├── tasks_service.py           # Tasks API: fetch, create, update, complete, delete, find completed
├── chat_service.py            # Chat API: send messages (auto-split at 4000 chars), format markdown
├── conversation_store.py      # Firestore: conversation history, email/debrief/prep/nudge dedup
├── granola_service.py         # Granola MCP client: meeting notes, transcripts, token lifecycle
├── granola_auth_setup.py      # One-time local OAuth2 + PKCE flow for Granola authentication
├── google_auth.py             # Thread-safe OAuth credential loading with auto-refresh
├── auth_setup.py              # One-time local script to generate token.json via browser OAuth flow
├── config.py                  # All configuration loaded from environment variables
├── Dockerfile                 # python:3.12-slim, uvicorn entrypoint
├── deploy.sh                  # gcloud run deploy wrapper with .env fallback for CHAT_SPACE_ID
├── requirements.txt
├── .env.example               # Template for local environment variables
└── .gitignore
```

---

## Data flow for a chat message

```
User message in Google Chat
    │
    ▼
Parse event format (standard Chat vs. Workspace Add-on)
    │
    ▼
Load conversation history from Firestore (up to 50 turns)
    │
    ▼
_build_context() — up to 7 concurrent workers (90s timeout):
  - always: today's meetings + open tasks + knowledge graph
  - if email keywords detected: inbox emails + targeted email search
  - if meeting/notes keywords or specific entities detected: Granola MCP query
    │
    ▼
Gemini chat_response() — tiered model routing:
  - Standard (Flash): normal queries
  - Deep (Pro): when knowledge graph context is present
  - Auto-fallback: Pro → Flash on failure
    │
    ▼
Parse task action tags from response (regex + prose fallback)
    │
    ├─► Execute task actions against Google Tasks API
    │
    ▼
Save turn to Firestore conversation history
    │
    ├─► Extract entities to knowledge graph (background thread)
    │
    ▼
format_for_google_chat() → send reply (auto-split at 4000 chars)
```

---

## How task management works

Gemini is instructed to emit structured tags at the end of its response when a task action is needed:

```
[CREATE_TASK] title="Call Sarah" due="2026-02-18" notes="Re: Q2 plan"
[UPDATE_TASK] find="Review proposal" due="2026-02-21"
[COMPLETE_TASK] find="Send deck"
[DELETE_TASK] find="test task"
```

The backend:
1. **Regex extraction** — parses `[ACTION_TAG]` patterns with key-value pairs
2. **Prose fallback** — catches natural language task descriptions (e.g. *"I moved your tasks to Friday"*) when Gemini doesn't emit tags
3. **Bulk detection** — handles batch operations like *"push all my tasks to Friday"*
4. **Fuzzy matching** — finds tasks by substring match across all task lists when executing updates, completions, or deletions
5. **Execution** — runs each action against the Google Tasks API
6. **Cleanup** — strips tags from the user-visible response and appends a summary of what was done

---

## Google services used

| Service | Usage | Auth |
|---|---|---|
| Gmail API | Read inbox, search emails | OAuth 2.0 (`gmail.readonly`) |
| Google Calendar API | Read events, upcoming and recently ended meetings | OAuth 2.0 (`calendar.readonly`) |
| Google Tasks API | Read + write tasks (create, update, complete, delete) | OAuth 2.0 (`tasks`) |
| Google Chat API | Receive webhook events, send messages | Application Default Credentials (service account) |
| Cloud Firestore | Conversations, dedup (alerts/debriefs/preps/nudges), knowledge graph, Granola token | ADC / service account |
| Cloud Run | Hosts the FastAPI server | N/A |
| Cloud Scheduler | Triggers `/briefing`, `/email-alerts`, `/meeting-debrief`, `/meeting-prep` | N/A |
| Granola MCP | Fetch meeting notes, transcripts, action items | OAuth 2.0 (PKCE, auto-refresh) |
| Gemini Flash | Chat, briefings, debriefs, meeting prep, entity extraction, email triage | API key |
| Gemini Pro | Deep reasoning for knowledge graph queries (auto-fallback to Flash) | API key |

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

1. Go to **APIs & Services > Credentials**
2. Create an **OAuth 2.0 Client ID** (Application type: Web application)
3. Add `http://localhost:8080/` to Authorized redirect URIs
4. Download the JSON and save it as `client_secret.json` in the project root

### 3. Generate token.json (one-time local auth)

```bash
python auth_setup.py
```

This opens a browser OAuth flow and writes `token.json`. Both files are gitignored and never committed. The token is thread-safe cached in memory and auto-refreshed when expired.

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
CLIENT_DOMAINS=
```

### 5. Granola integration (optional)

Momo can pull meeting notes, decisions, and action items from [Granola](https://granola.ai).

**Authenticate once:**

```bash
python granola_auth_setup.py
```

This runs a full OAuth 2.0 + PKCE flow: discovers OAuth endpoints automatically, opens a browser for sign-in, captures the auth code via local callback, exchanges it for tokens, and saves them to `granola_token.json` (gitignored) + Firestore. The token auto-refreshes — you only need to run this once.

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

The deploy script:
- Reads `token.json` and `granola_token.json` into environment variables so no credential files are bundled in the image
- Syncs the Granola token to Firestore (if present)
- Falls back to reading `CHAT_SPACE_ID` from `.env` if not in the environment, and warns if still unset
- Deploys to Cloud Run with 1Gi memory, 300s timeout, 0-3 instances, unauthenticated access

After deploying:
1. Copy the Cloud Run URL
2. Set it as the HTTP endpoint in your Google Chat App config: `https://<url>/chat`
3. Create Cloud Scheduler jobs:
   - `POST https://<url>/briefing` — daily at 8 AM (or your preferred time)
   - `POST https://<url>/email-alerts` — every 5 minutes
   - `POST https://<url>/meeting-debrief` — every 10 minutes during work hours (e.g. `*/10 9-18 * * 1-5`)
   - `POST https://<url>/meeting-prep` — every 10 minutes during work hours
4. (One-time) Bootstrap the knowledge graph: `curl -X POST https://<url>/knowledge-backfill`

---

## Firestore collections

| Collection | Document key | Purpose |
|---|---|---|
| `conversations` | sanitized user ID | Stores up to 50 conversation turns per user |
| `email_alerts` | Gmail message ID | Tracks which emails have already triggered an alert |
| `meeting_debriefs` | Google Calendar event ID | Tracks which meetings have already been debriefed |
| `meeting_prep_sent` | Google Calendar event ID | Tracks which meetings have had prep briefs sent |
| `proactive_nudges_sent` | hashed nudge key (MD5) | Deduplicates nudges with configurable cooldown window |
| `knowledge_graph` | auto-generated | Extracted entities with provenance (source type/ID/title/date, related people/projects, tags) |
| `granola_auth` | `token` | Stores the Granola OAuth token for Cloud Run (auto-refreshed) |

---

## Environment variable reference

| Variable | Required | Default | Description |
|---|---|---|---|
| **Core** | | | |
| `GEMINI_API_KEY` | Yes | — | Gemini API key from Google AI Studio |
| `GCP_PROJECT_ID` | Yes | — | GCP project ID (used for Firestore) |
| `CHAT_SPACE_ID` | No | — | Google Chat space to post to (`spaces/XXX`). If unset, briefings print to console |
| `PORT` | No | `8080` | Server port |
| **Google Auth** | | | |
| `GOOGLE_CLIENT_SECRET_FILE` | Local only | `client_secret.json` | Path to OAuth client secret |
| `GOOGLE_TOKEN_FILE` | Local only | `token.json` | Path to OAuth token |
| `GOOGLE_TOKEN_JSON` | Cloud Run | — | Full token JSON as a string (replaces token file) |
| **Gemini Models** | | | |
| `GEMINI_MODEL_FLASH` | No | `gemini-3-flash-preview` | Model for standard/light tasks |
| `GEMINI_MODEL_PRO` | No | `gemini-3.1-pro-preview` | Model for deep reasoning (auto-fallback to Flash) |
| **Email** | | | |
| `EMAIL_ALERTS_ENABLED` | No | `true` | Toggle proactive email alerts |
| `EMAIL_ALERT_GMAIL_QUERY` | No | `is:unread in:inbox newer_than:2d` | Gmail search query for alert candidates |
| `EMAIL_ALERTS_MAX_PER_RUN` | No | `5` | Max alerts to send per scheduler run |
| `IMPORTANT_EMAIL_KEYWORDS` | No | `urgent,asap,...` | Keywords that flag an email as important |
| `CLIENT_EMAIL_KEYWORDS` | No | `client,customer` | Keywords that identify client emails |
| `CLIENT_DOMAINS` | No | — | Comma-separated domains to treat as clients |
| `BRIEFING_LOOKBACK_HOURS` | No | `24` | How far back to look for unread emails in briefing |
| `SEARCH_LOOKBACK_DAYS` | No | `90` | How far back to search when looking up specific emails |
| **Knowledge Graph** | | | |
| `KNOWLEDGE_GRAPH_ENABLED` | No | `true` | Toggle knowledge graph extraction and querying |
| **Proactive Intelligence** | | | |
| `PROACTIVE_INTELLIGENCE_ENABLED` | No | `true` | Toggle all proactive intelligence engines |
| `MEETING_PREP_ENABLED` | No | `true` | Toggle pre-meeting prep briefs |
| `MEETING_PREP_LOOKAHEAD_HOURS` | No | `1` | How far ahead to look for upcoming meetings to prep |
| `COMMITMENT_FOLLOWUP_DAYS` | No | `3` | Days before an open commitment triggers a follow-up nudge |
| `DRIFT_THRESHOLD_DAYS` | No | `14` | Days of inactivity before flagging a project/item as stale |
| `NUDGE_COOLDOWN_DAYS` | No | `7` | Minimum days between repeat nudges for the same item |
| **Granola** | | | |
| `GRANOLA_ENABLED` | No | `false` | Enable Granola meeting notes integration |
| `GRANOLA_MCP_URL` | No | `https://mcp.granola.ai/mcp` | Granola MCP server URL |
| `GRANOLA_TOKEN_JSON` | Cloud Run | — | Full Granola token JSON as a string (seeds Firestore on first boot) |
| `MEETING_DEBRIEF_LOOKBACK_MINUTES` | No | `120` | How far back to look for recently ended meetings |
| `MEETING_DEBRIEF_GRACE_MINUTES` | No | `45` | Grace period before sending debrief without notes |
| **Firestore** | | | |
| `FIRESTORE_DATABASE` | No | `testing` | Firestore database name |
| `MAX_CONVERSATION_TURNS` | No | `50` | Max conversation turns stored per user |
