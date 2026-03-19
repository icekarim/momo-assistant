import os
from dotenv import load_dotenv

load_dotenv()

# ── Gemini ───────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-3-flash-preview"
GEMINI_MODEL_FLASH = os.getenv("GEMINI_MODEL_FLASH", "gemini-3-flash-preview")
GEMINI_MODEL_PRO = os.getenv("GEMINI_MODEL_PRO", "gemini-3.1-pro-preview")

# ── Gmail ────────────────────────────────────────────────────
GMAIL_QUERY = "is:unread in:inbox"
BRIEFING_LOOKBACK_HOURS = 24
SEARCH_LOOKBACK_DAYS = 90
MAX_EMAILS = 50
MAX_CHAT_EMAILS = 15
EMAIL_ALERTS_ENABLED = os.getenv("EMAIL_ALERTS_ENABLED", "true").lower() == "true"
EMAIL_ALERT_GMAIL_QUERY = os.getenv(
    "EMAIL_ALERT_GMAIL_QUERY",
    "is:unread in:inbox newer_than:2d",
)
EMAIL_ALERTS_MAX_PER_RUN = int(os.getenv("EMAIL_ALERTS_MAX_PER_RUN", "5"))
IMPORTANT_EMAIL_KEYWORDS = [
    kw.strip().lower()
    for kw in os.getenv(
        "IMPORTANT_EMAIL_KEYWORDS",
        "urgent,asap,important,action required,deadline,escalation,blocker",
    ).split(",")
    if kw.strip()
]
CLIENT_EMAIL_KEYWORDS = [
    kw.strip().lower()
    for kw in os.getenv("CLIENT_EMAIL_KEYWORDS", "client,customer").split(",")
    if kw.strip()
]
CLIENT_DOMAINS = [
    domain.strip().lower()
    for domain in os.getenv("CLIENT_DOMAINS", "").split(",")
    if domain.strip()
]

# ── Google Auth ──────────────────────────────────────────────
GOOGLE_CLIENT_SECRET_FILE = os.getenv("GOOGLE_CLIENT_SECRET_FILE", "client_secret.json")
GOOGLE_TOKEN_FILE = os.getenv("GOOGLE_TOKEN_FILE", "token.json")

GOOGLE_SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/calendar.readonly",
    "https://www.googleapis.com/auth/tasks",
]

# ── Google Chat ──────────────────────────────────────────────
CHAT_SPACE_ID = os.getenv("CHAT_SPACE_ID", "")

# ── Firestore ────────────────────────────────────────────────
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "")
FIRESTORE_DATABASE = os.getenv("FIRESTORE_DATABASE", "testing")
FIRESTORE_COLLECTION = "conversations"
FIRESTORE_EMAIL_ALERTS_COLLECTION = "email_alerts"
FIRESTORE_MEETING_DEBRIEFS_COLLECTION = "meeting_debriefs"
FIRESTORE_KNOWLEDGE_GRAPH_COLLECTION = "knowledge_graph"
MAX_CONVERSATION_TURNS = 50

# ── Knowledge Graph ──────────────────────────────────────────
KNOWLEDGE_GRAPH_ENABLED = os.getenv("KNOWLEDGE_GRAPH_ENABLED", "true").lower() == "true"
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "models/text-embedding-004")
SEMANTIC_SEARCH_THRESHOLD = float(os.getenv("SEMANTIC_SEARCH_THRESHOLD", "0.60"))
SEMANTIC_SEARCH_LIMIT = int(os.getenv("SEMANTIC_SEARCH_LIMIT", "15"))

# ── Agentic Mode ─────────────────────────────────────────────
AGENTIC_MODE_ENABLED = os.getenv("AGENTIC_MODE_ENABLED", "true").lower() == "true"

# ── Proactive Intelligence ───────────────────────────────────
PROACTIVE_INTELLIGENCE_ENABLED = os.getenv("PROACTIVE_INTELLIGENCE_ENABLED", "true").lower() == "true"
MEETING_PREP_ENABLED = os.getenv("MEETING_PREP_ENABLED", "true").lower() == "true"
MEETING_PREP_LOOKAHEAD_HOURS = int(os.getenv("MEETING_PREP_LOOKAHEAD_HOURS", "1"))
COMMITMENT_FOLLOWUP_DAYS = int(os.getenv("COMMITMENT_FOLLOWUP_DAYS", "3"))
DRIFT_THRESHOLD_DAYS = int(os.getenv("DRIFT_THRESHOLD_DAYS", "14"))
NUDGE_COOLDOWN_DAYS = int(os.getenv("NUDGE_COOLDOWN_DAYS", "7"))
FIRESTORE_MEETING_PREP_COLLECTION = "meeting_prep_sent"
FIRESTORE_NUDGES_COLLECTION = "proactive_nudges_sent"
FIRESTORE_PENDING_TASKS_COLLECTION = "pending_task_proposals"

# ── Granola MCP ──────────────────────────────────────────────
GRANOLA_ENABLED = os.getenv("GRANOLA_ENABLED", "false").lower() == "true"
GRANOLA_MCP_URL = os.getenv("GRANOLA_MCP_URL", "https://mcp.granola.ai/mcp")
GRANOLA_TOKEN = os.getenv("GRANOLA_TOKEN", "")
MEETING_DEBRIEF_LOOKBACK_MINUTES = int(os.getenv("MEETING_DEBRIEF_LOOKBACK_MINUTES", "120"))
MEETING_DEBRIEF_GRACE_MINUTES = int(os.getenv("MEETING_DEBRIEF_GRACE_MINUTES", "45"))
MEETING_DEBRIEF_MIN_WAIT_MINUTES = int(os.getenv("MEETING_DEBRIEF_MIN_WAIT_MINUTES", "15"))
MEETING_DEBRIEF_MIN_NOTE_WORDS = int(os.getenv("MEETING_DEBRIEF_MIN_NOTE_WORDS", "50"))

# ── Jira ─────────────────────────────────────────────────────
JIRA_ENABLED = os.getenv("JIRA_ENABLED", "false").lower() == "true"
JIRA_SITE_URL = os.getenv("JIRA_SITE_URL", "")          # e.g. yourcompany.atlassian.net
JIRA_USER_EMAIL = os.getenv("JIRA_USER_EMAIL", "")
JIRA_API_TOKEN = os.getenv("JIRA_API_TOKEN", "")
JIRA_JQL_FILTER = os.getenv(
    "JIRA_JQL_FILTER",
    "(assignee = currentUser() OR reporter = currentUser() OR watcher = currentUser()) "
    "AND statusCategory != Done ORDER BY updated DESC",
)

# ── Server ───────────────────────────────────────────────────
PORT = int(os.getenv("PORT", "8080"))
