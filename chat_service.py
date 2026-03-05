import threading
import time

import google.auth
from google.auth.transport.requests import AuthorizedSession
import config

_CHAT_SCOPES = ["https://www.googleapis.com/auth/chat.bot"]
_MAX_ATTACHMENT_BYTES = 10 * 1024 * 1024  # 10 MB
_SUPPORTED_AUDIO_TYPES = frozenset([
    "audio/ogg", "audio/mp4", "audio/mpeg", "audio/wav",
    "audio/webm", "audio/aac", "audio/x-m4a", "audio/mp3",
])

_chat_session = None
_chat_session_lock = threading.Lock()


def _get_chat_session():
    global _chat_session
    if _chat_session is not None:
        return _chat_session
    with _chat_session_lock:
        if _chat_session is not None:
            return _chat_session
        creds, _ = google.auth.default(scopes=_CHAT_SCOPES)
        _chat_session = AuthorizedSession(creds)
        return _chat_session


def download_attachment(resource_name: str) -> tuple[bytes, str] | None:
    """Download an attachment from Google Chat.

    Returns (raw_bytes, content_type) on success, None on failure.
    """
    session = _get_chat_session()
    url = f"https://chat.googleapis.com/v1/media/{resource_name}?alt=media"
    try:
        resp = session.get(url)
        if resp.status_code != 200:
            print(f"Attachment download failed ({resp.status_code}): {resp.text}")
            return None

        content_type = resp.headers.get("Content-Type", "application/octet-stream")
        data = resp.content

        if len(data) > _MAX_ATTACHMENT_BYTES:
            print(f"Attachment too large ({len(data)} bytes), skipping")
            return None

        print(f"Downloaded attachment: {len(data)} bytes, type={content_type}")
        return data, content_type
    except Exception as e:
        print(f"Attachment download error: {e}")
        return None


def send_chat_message(space_id, text):
    """Send a message to a Google Chat space as the bot (app credentials).
    Retries up to 3 times on transient network/SSL errors."""
    session = _get_chat_session()
    url = f"https://chat.googleapis.com/v1/{space_id}/messages"

    chunks = _split_message(text, max_len=4000)

    for chunk in chunks:
        _send_with_retry(session, url, chunk, space_id)


def _send_with_retry(session, url, text, space_id, max_retries=3):
    for attempt in range(max_retries):
        try:
            resp = session.post(url, json={"text": text})
            if resp.status_code != 200:
                print(f"Chat API error ({resp.status_code}): {resp.text}")
            else:
                print(f"Momo sent message to {space_id}")
            return
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 1.0 * (attempt + 1)
                print(f"Chat API send failed (attempt {attempt + 1}/{max_retries}): {e} — retrying in {wait}s")
                time.sleep(wait)
            else:
                print(f"Chat API send failed after {max_retries} attempts: {e}")
                raise


def format_for_google_chat(markdown_text):
    """Convert standard markdown to Google Chat's supported format."""
    import re
    text = markdown_text
    text = re.sub(r"\*\*(.+?)\*\*", r"*\1*", text)
    text = re.sub(r"^#{1,3} (.+)$", r"\n*\1*", text, flags=re.MULTILINE)
    return text.strip()


def _split_message(text, max_len=4000):
    """Split a long message into chunks, breaking at newlines."""
    if len(text) <= max_len:
        return [text]

    chunks = []
    current = ""

    for line in text.split("\n"):
        if len(current) + len(line) + 1 > max_len:
            if current:
                chunks.append(current)
            current = line
        else:
            current = f"{current}\n{line}" if current else line

    if current:
        chunks.append(current)

    return chunks
