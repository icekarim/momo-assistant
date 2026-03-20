import base64
import re
from datetime import datetime, timedelta, timezone
from googleapiclient.discovery import build
from google_auth import get_credentials
import config


def get_gmail_service():
    creds = get_credentials()
    return build("gmail", "v1", credentials=creds)


def fetch_unread_client_emails(lookback_hours=None, max_results=None):
    """Fetch unread client emails using the configured Gmail filter."""
    hours = lookback_hours or config.BRIEFING_LOOKBACK_HOURS
    after = datetime.now(timezone.utc) - timedelta(hours=hours)
    after_str = after.strftime("%Y/%m/%d")

    query = f"{config.GMAIL_QUERY} after:{after_str}"
    return _search_emails(query, max_results or config.MAX_EMAILS)


def search_emails(search_query, days_back=None, max_results=None):
    """Search emails with a custom query for conversational lookups."""
    days = days_back or config.SEARCH_LOOKBACK_DAYS
    after = datetime.now(timezone.utc) - timedelta(days=days)
    after_str = after.strftime("%Y/%m/%d")

    query = f"{search_query} after:{after_str}"
    return _search_emails(query, max_results or config.MAX_EMAILS)


def fetch_email_alert_candidates():
    """Fetch unread emails to evaluate for proactive alerts."""
    return _search_emails(config.EMAIL_ALERT_GMAIL_QUERY, config.EMAIL_ALERTS_MAX_PER_RUN * 5)


def _search_emails(query, max_results):
    """Internal: execute a Gmail search and return parsed emails.
    Fetches individual messages in parallel to avoid N+1 latency."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    svc = get_gmail_service()
    msg_refs = []
    page_token = None

    while len(msg_refs) < max_results:
        resp = svc.users().messages().list(
            userId="me",
            q=query,
            maxResults=min(max_results - len(msg_refs), 50),
            pageToken=page_token,
        ).execute()

        messages = resp.get("messages", [])
        if not messages:
            break

        msg_refs.extend(messages)
        page_token = resp.get("nextPageToken")
        if not page_token:
            break

    msg_refs = msg_refs[:max_results]

    def _fetch_one(msg_id):
        thread_svc = get_gmail_service()
        msg = thread_svc.users().messages().get(
            userId="me", id=msg_id, format="full"
        ).execute()
        return msg_id, _parse_message(msg)

    results_by_id = {}
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = {pool.submit(_fetch_one, ref["id"]): ref["id"] for ref in msg_refs}
        for future in as_completed(futures):
            try:
                msg_id, parsed = future.result()
                results_by_id[msg_id] = parsed
            except Exception as e:
                print(f"Failed to fetch message {futures[future]}: {e}")

    return [results_by_id[ref["id"]] for ref in msg_refs if ref["id"] in results_by_id]


def _parse_message(msg):
    """Parse a Gmail API message into a clean dict."""
    headers = {h["name"].lower(): h["value"] for h in msg["payload"]["headers"]}
    labels = msg.get("labelIds", [])

    body = _extract_body(msg["payload"])

    max_len = 1500
    if len(body) > max_len:
        body = body[:max_len] + "\n[... truncated]"

    # Parse date
    internal_date = int(msg.get("internalDate", 0))
    date = datetime.fromtimestamp(internal_date / 1000, tz=timezone.utc)

    return {
        "id": msg["id"],
        "thread_id": msg["threadId"],
        "from": headers.get("from", "Unknown"),
        "to": headers.get("to", ""),
        "subject": headers.get("subject", "(no subject)"),
        "date": date.isoformat(),
        "date_ymd": date.strftime("%Y-%m-%d"),
        "date_human": date.strftime("%b %d, %I:%M %p"),
        "body": body,
        "snippet": msg.get("snippet", ""),
        "labels": labels,
    }


def _extract_body(payload):
    """Extract plain text body from a Gmail message payload."""
    if payload.get("body", {}).get("data"):
        return _decode_base64(payload["body"]["data"])

    parts = payload.get("parts", [])
    plain = ""
    html = ""

    for part in parts:
        mime = part.get("mimeType", "")
        if mime == "text/plain" and part.get("body", {}).get("data"):
            plain = _decode_base64(part["body"]["data"])
        elif mime == "text/html" and part.get("body", {}).get("data"):
            html = _decode_base64(part["body"]["data"])
        elif "parts" in part:
            nested = _extract_body(part)
            if nested:
                plain = plain or nested

    if plain:
        return plain

    if html:
        text = re.sub(r"<style[^>]*>[\s\S]*?</style>", "", html)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"&nbsp;", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    return ""


def _decode_base64(data):
    """Decode Gmail's URL-safe base64."""
    return base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")


def format_emails_for_context(emails):
    """Format a list of emails into a text block for Gemini."""
    if not emails:
        return "No emails found."

    lines = []
    for i, e in enumerate(emails, 1):
        lines.append(
            f"--- Email {i} ---\n"
            f"From: {e['from']}\n"
            f"Subject: {e['subject']}\n"
            f"Date: {e['date_human']}\n"
            f"\n{e['body']}"
        )
    return "\n\n".join(lines)
