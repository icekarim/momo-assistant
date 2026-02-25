import google.auth
from google.auth.transport.requests import AuthorizedSession
import config


def send_chat_message(space_id, text):
    """Send a message to a Google Chat space as the bot (app credentials)."""
    creds, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/chat.bot"]
    )
    session = AuthorizedSession(creds)

    url = f"https://chat.googleapis.com/v1/{space_id}/messages"

    chunks = _split_message(text, max_len=4000)

    for chunk in chunks:
        resp = session.post(url, json={"text": chunk})
        if resp.status_code != 200:
            print(f"Chat API error ({resp.status_code}): {resp.text}")
        else:
            print(f"Momo sent message to {space_id}")


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
