# Momo

Momo is a personal AI chief of staff that lives in Google Chat. It knows your inbox, your calendar, your tasks, and your meeting history — and it keeps you on top of all of it without you having to ask.

Every morning it drops a briefing in Chat: what needs your attention in your inbox, what's on your calendar, what's open in your task list, and what came out of yesterday's meetings. Throughout the day you can message it like a colleague — ask questions, get summaries, delegate tasks, or dig into what was decided in a meeting last week.

Momo is **agentic** — when you send a message, Gemini decides which tools to call (Gmail, Calendar, Tasks, Knowledge Graph, Granola, Jira), executes them, observes the results, and iterates until it has enough information to respond. No pre-fetching everything on every message. No keyword gating. Momo pulls exactly the data it needs, when it needs it.

Momo builds a **cross-meeting knowledge graph** with **vector embeddings** — it extracts decisions, commitments, action items, blockers, and topics from every meeting, email, calendar event, task, and chat conversation, then embeds them for semantic search. Ask about anything in natural language and Momo finds the relevant context across all sources.

On top of that, Momo runs a **proactive intelligence engine** — it surfaces insights before you ask. Pre-meeting prep briefs, commitment follow-ups with auto-resolution, pattern detection across meetings, and drift alerts for projects that have gone quiet. All delivered as nudges in your morning briefing or as standalone Chat messages for urgent items.

Built with **FastAPI** + **Gemini Flash** (agentic tool-use loop), deployed on **Google Cloud Run**.

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

### Agentic conversational assistant
Momo uses a **Gemini-powered agent loop with tool calling**. When you send a message, Gemini autonomously decides which tools to invoke — Gmail, Calendar, Tasks, Knowledge Graph, Granola, Jira — executes them, observes the results, and iterates (up to 6 rounds) until it has enough data to respond. Each tool has its own timeout (10-15s) and runs in a thread pool.

Ask Momo anything about your work context in plain language:
- *"what's on my calendar today?"* → calls `get_todays_calendar`
- *"any urgent emails from clients?"* → calls `get_recent_emails`
- *"what did we decide in the product standup?"* → calls `search_knowledge_graph`
- *"pull up the action items from yesterday's call"* → calls `get_meeting_notes` + `search_knowledge_graph`
- *"push all my tasks to Friday"* → calls `get_open_tasks` then `update_task` for each
- *"what commitments have I made this week?"* → calls `search_knowledge_graph`
- *"create a task to follow up with Sarah by Friday"* → calls `create_task`

The agent has access to **13 tools**:

| Tool | Description |
|---|---|
| `get_todays_calendar` | Today's meetings with times, attendees, details |
| `get_calendar_for_date` | Meetings for a specific date |
| `get_open_tasks` | All open tasks across task lists |
| `create_task` | Create a new task (with dedup) |
| `update_task` | Update task title, due date, or notes |
| `complete_task` | Mark a task as done |
| `delete_task` | Delete a task |
| `get_recent_emails` | Recent unread inbox emails |
| `search_emails` | Search emails by query, person, or topic |
| `search_knowledge_graph` | Semantic search across all institutional memory |
| `get_meeting_notes` | Search Granola meeting notes and transcripts |
| `get_jira_tickets` | Active Jira tickets for the user |
| `search_jira_tickets` | Search Jira by text query |

Chat messages are also extracted into the knowledge graph in the background, so context from conversations is captured alongside meetings and emails.

### Personality
Momo is casual and concise — lowercase by default, minimal formatting, no corporate filler. It uses structured sections with emoji headers for briefings and priority colors for items. Responses are designed to be scannable, not verbose.

### Task management
Full CRUD over Google Tasks via the agent's tool-use loop. Gemini calls task tools directly — `create_task`, `update_task`, `complete_task`, `delete_task` — with automatic dedup checking and fuzzy title matching. Bulk operations like *"push all my tasks to Friday"* work naturally since the agent can call `get_open_tasks` and then `update_task` for each one in a single loop.

### Meeting intelligence (via Granola)
Momo is connected to [Granola](https://granola.ai) via the Model Context Protocol (MCP) and can answer questions about any recorded meeting:
- *"what were the action items from Monday's sync?"*
- *"who was in the planning meeting?"*
- *"pull the full transcript from yesterday's call"*
- *"what decisions have we made about [topic] this month?"*

The Granola token is managed automatically — it's stored in Firestore and refreshed in the background using OAuth 2.0 with PKCE. Token resolution order: in-memory cache → local file → Firestore → environment variable. You authenticate once with `python granola_auth_setup.py`, and a **Cloud Scheduler job refreshes the token every 4 hours** (`POST /granola-token-refresh`) to prevent the refresh_token from expiring due to inactivity.

Calendar meetings are matched to Granola notes using fuzzy title matching, and notes are batch-fetched (up to 10 at a time) to minimize API calls.

### Cross-meeting knowledge graph with semantic search

Momo builds persistent institutional memory by extracting structured entities from **every meeting debrief, email, calendar event, task, Granola meeting note, and chat conversation**:

- **Decisions** — what was decided and when
- **Commitments** — who promised what to whom, with status tracking
- **Action items** — tasks that need to happen, with owners
- **Blockers** — what's blocking progress
- **Topics** — subjects discussed without a clear outcome
- **Updates** — status reports on ongoing work

Each entity is stored in Firestore with full provenance (source type, source ID, source title, date, related people, related projects, tags) **plus a vector embedding** generated by Gemini's gemini-embedding-001 model.

**Semantic search** — The knowledge graph is searched using vector cosine similarity, not keyword matching. The agent embeds the user's query and compares it against all stored entity embeddings using a **numpy-vectorized matrix operation** — all 1500+ similarities are computed in a single dot product (~0.14s). Results above a configurable similarity threshold (default 0.60) are returned. Queries like *"what happened with the partner rollout?"* find relevant entities even when the exact words don't match.

The embedding matrix is **pre-loaded on startup** in a background thread and cached for 30 minutes, eliminating cold-start latency on the first query.

**Six source types feed the knowledge graph:**

| Source | Trigger | source_type |
|---|---|---|
| Emails | Proactive email alerts | `email` |
| Meetings | Post-meeting debriefs via Granola | `meeting` |
| Calendar events | Morning briefing + backfill | `calendar` |
| Tasks | Morning briefing + backfill (daily snapshot) | `tasks` |
| Granola notes | Morning briefing (catches notes debrief missed) | `meeting_notes` |
| Chat messages | Every user message | `chat` |

Extraction happens in the background (daemon threads) so it never blocks debrief or alert delivery. Bootstrap the graph with `POST /knowledge-backfill`, then add embeddings to existing entities with `POST /knowledge-embed-backfill`.

### Proactive intelligence

Momo doesn't just answer questions — it surfaces insights before you think to ask. Four engines run daily (during the morning briefing) and throughout the day:

**1. Pre-meeting prep** — Before each meeting, Momo pulls relevant context from the knowledge graph: past interactions with attendees, open commitments involving them, related project history, and outstanding blockers. Delivered as a brief to Chat before the meeting starts (configurable lookahead, default 1 hour).

**2. Commitment follow-up** — Scans the knowledge graph for open commitments older than 3 days (configurable). Cross-references Gmail and Google Tasks for evidence of completion — if a matching email was sent or a task was completed, the commitment is auto-resolved in the knowledge graph. Otherwise, it's flagged as a nudge with days elapsed and source context. High-priority nudges (commitments overdue by 2x the threshold) are sent as standalone Chat messages immediately.

**3. Pattern detection** — Analyzes the last 30 days of knowledge graph entries to spot recurring people, projects, and tags. Feeds the patterns into Gemini for interpretation — surfaces things like recurring topics that need a dedicated discussion or frequent collaborators worth noting.

**4. Drift detection** — Flags open items (commitments, action items) and projects that haven't been mentioned in any meeting or email for 14+ days (configurable). Helps catch things that silently fell off the radar.

All nudges are deduplicated via Firestore with a configurable cooldown (default 7 days) so you're never spammed with the same reminder. Nudges are delivered in two modes:
- **Briefing** — included as a section in the morning briefing (default for medium/low priority)
- **Standalone** — sent as immediate Chat messages (for high-priority items like severely overdue commitments)

### Model usage

| Use case | Model |
|---|---|
| Agentic chat (tool-use loop) | Gemini Flash |
| Entity extraction | Gemini Flash |
| Email triage | Gemini Flash |
| Briefings, debriefs, meeting prep | Gemini Flash |
| Vector embeddings | gemini-embedding-001 (numpy-vectorized cosine similarity) |

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
│                    │   Agent Loop      │  Gemini decides      │
│                    │   (agent.py)      │  which tools to call │
│                    └─────────┬─────────┘                     │
│                              │                               │
│              Gemini calls tools iteratively:                  │
│    ┌──────────┬──────────┬───┴───┬──────────┬─────────┐      │
│    ▼          ▼          ▼       ▼          ▼         ▼      │
│  Gmail    Calendar     Tasks  Granola   Knowledge   Jira     │
│ (read/   (today/      (CRUD)  (notes)   Graph      (read)   │
│  search)  by date)                    (semantic)             │
│    └──────────┴──────────┴───────┴──────────┴─────────┘      │
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
                         │
              ┌──────────┼──────────┐
              ▼          ▼          ▼
        Gemini       Knowledge   Knowledge
       (briefing)   Graph Feed   Graph Feed
                   (cal+tasks)   (granola)
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
                          (+ embedding)

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
                │         (+ embedding)
                ▼
          Google Chat

Cloud Scheduler ──► POST /meeting-prep
                         │
                ┌────────┴────────────────┐
                ▼                         ▼
        Calendar Service           Knowledge Graph
      (upcoming meetings)         (semantic search)
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
| `/knowledge-backfill` | POST | One-time | Bootstrap knowledge graph from meetings, emails, calendar, tasks |
| `/knowledge-embed-backfill` | POST | One-time | Add vector embeddings to existing KG entities |
| `/granola-token-refresh` | POST | Cloud Scheduler (every 4h) | Keep Granola OAuth token alive |

### Scheduled jobs (Cloud Scheduler)

```
Cloud Scheduler ──► POST /briefing                (daily at 8 AM)
Cloud Scheduler ──► POST /email-alerts            (every 5 minutes)
Cloud Scheduler ──► POST /meeting-debrief         (every 10 minutes, work hours)
Cloud Scheduler ──► POST /meeting-prep            (every 10 minutes, work hours)
One-time         ──► POST /knowledge-backfill      (bootstrap knowledge graph)
One-time         ──► POST /knowledge-embed-backfill (add embeddings to existing entities)
Cloud Scheduler ──► POST /granola-token-refresh    (every 4 hours — keeps token alive)
```

**`/briefing`** — orchestrated by `briefing.py`:
1. Fetches unread emails, today's meetings, all open tasks (in parallel)
2. Optionally fetches yesterday's meeting notes from Granola
3. Runs proactive intelligence engines (commitment follow-up, pattern detection, drift detection)
4. Sends all context + nudges to Gemini → formatted morning briefing
5. Posts the result to the configured Google Chat space
6. Feeds calendar events, tasks, and Granola notes into the knowledge graph (background)

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
2. Fetches today's calendar events and current open tasks
3. Batch-extracts entities (decisions, commitments, blockers, etc.) via Gemini Flash
4. Generates vector embeddings for each entity using gemini-embedding-001
5. Stores everything in the `knowledge_graph` Firestore collection
6. Idempotent — skips sources that have already been processed
7. Runs in a background thread so the endpoint returns immediately

**`/knowledge-embed-backfill`** — in `main.py`:
1. Reads all existing KG entities from Firestore
2. Generates vector embeddings for entities that don't have one yet
3. Updates each document in-place with the embedding
4. Idempotent — skips entities that already have embeddings

**`/granola-token-refresh`** — in `main.py`:
1. Forces a proactive refresh of the Granola OAuth token
2. Scheduled every 4 hours via Cloud Scheduler to prevent the refresh_token from expiring due to inactivity
3. Granola access tokens last 6 hours — refreshing at 4h intervals keeps the token chain alive permanently

---

## File structure

```
momo/
├── main.py                    # FastAPI app, endpoints, event parsing, chat routing
├── agent.py                   # Agentic tool-use loop: tool declarations, executor, agent system prompt
├── gemini_service.py          # Gemini API wrapper, system prompt, chat/briefing/debrief generation
├── proactive_intelligence.py  # Proactive intelligence: pre-meeting prep, commitment follow-up, patterns, drift
├── knowledge_graph.py         # Entity extraction, vector embeddings, semantic search, Firestore storage
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
Agent loop (agent.py) — Gemini with tool declarations:
    │
    ├─► Gemini decides which tools to call
    │     │
    │     ▼
    │   Execute tools in thread pool (per-tool timeouts):
    │     - get_todays_calendar, get_open_tasks, get_recent_emails
    │     - search_emails, search_knowledge_graph (semantic)
    │     - create_task, update_task, complete_task, delete_task
    │     - get_meeting_notes, get_jira_tickets, search_jira_tickets
    │     │
    │     ▼
    │   Return results to Gemini → may call more tools
    │     │
    │     (repeats up to 6 iterations)
    │
    ▼
Final text response from Gemini
    │
    ▼
Save turn to Firestore conversation history
    │
    ├─► Extract entities to knowledge graph (background, with embedding)
    │
    ▼
format_for_google_chat() → send reply (auto-split at 4000 chars)
```

---

## How task management works

The agent calls task tools directly during the tool-use loop — no regex parsing or tag extraction needed:

1. **Gemini decides** — based on the user's message, the agent calls `create_task`, `update_task`, `complete_task`, or `delete_task`
2. **Dedup checking** — `create_task` automatically checks for existing tasks with matching titles before creating
3. **Fuzzy matching** — `update_task`, `complete_task`, and `delete_task` find tasks by fuzzy title match across all task lists
4. **Bulk operations** — for requests like *"push all my tasks to Friday"*, the agent calls `get_open_tasks` first, then `update_task` for each one in a multi-iteration loop
5. **Results in context** — tool results are returned to Gemini, which incorporates them naturally into its response

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
| Gemini Flash | Agentic chat (tool-use loop), briefings, debriefs, meeting prep, entity extraction, email triage | API key |
| Gemini gemini-embedding-001 | Vector embeddings for knowledge graph semantic search (numpy-accelerated) | API key |

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
5. (One-time) Add embeddings to existing entities: `curl -X POST https://<url>/knowledge-embed-backfill`
6. Create Cloud Scheduler job for Granola token keepalive: `POST https://<url>/granola-token-refresh` — every 4 hours

---

## Firestore collections

| Collection | Document key | Purpose |
|---|---|---|
| `conversations` | sanitized user ID | Stores up to 50 conversation turns per user |
| `email_alerts` | Gmail message ID | Tracks which emails have already triggered an alert |
| `meeting_debriefs` | Google Calendar event ID | Tracks which meetings have already been debriefed |
| `meeting_prep_sent` | Google Calendar event ID | Tracks which meetings have had prep briefs sent |
| `proactive_nudges_sent` | hashed nudge key (MD5) | Deduplicates nudges with configurable cooldown window |
| `knowledge_graph` | auto-generated | Extracted entities with provenance, vector embeddings, and metadata (source type/ID/title/date, related people/projects, tags) |
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
| `GEMINI_EMBEDDING_MODEL` | No | `models/gemini-embedding-001` | Model for generating entity embeddings (must support embedContent API) |
| `SEMANTIC_SEARCH_THRESHOLD` | No | `0.60` | Minimum cosine similarity for semantic search results |
| `SEMANTIC_SEARCH_LIMIT` | No | `15` | Maximum number of semantic search results |
| **Agentic Mode** | | | |
| `AGENTIC_MODE_ENABLED` | No | `true` | Use agent tool-use loop for chat (falls back to legacy context mode if disabled) |
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
