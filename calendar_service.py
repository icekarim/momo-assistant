from datetime import datetime, timedelta, timezone
from googleapiclient.discovery import build
from google_auth import get_credentials


def get_calendar_service():
    creds = get_credentials()
    return build("calendar", "v3", credentials=creds)


def fetch_todays_meetings():
    """Fetch all events for today from the primary calendar."""
    now = datetime.now().astimezone()
    start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
    end_of_day = start_of_day + timedelta(days=1)

    return _fetch_events(start_of_day, end_of_day)


def fetch_meetings_for_date(date_str):
    """Fetch meetings for a specific date (YYYY-MM-DD)."""
    try:
        day = datetime.strptime(date_str, "%Y-%m-%d").astimezone()
    except ValueError:
        return []
    end = day + timedelta(days=1)
    return _fetch_events(day, end)


def fetch_upcoming_meetings(hours=4):
    """Fetch meetings in the next N hours."""
    now = datetime.now().astimezone()
    end = now + timedelta(hours=hours)
    return _fetch_events(now, end)


def _fetch_events(time_min, time_max):
    """Fetch events between two datetimes."""
    svc = get_calendar_service()

    resp = svc.events().list(
        calendarId="primary",
        timeMin=time_min.isoformat(),
        timeMax=time_max.isoformat(),
        singleEvents=True,
        orderBy="startTime",
        maxResults=50,
    ).execute()

    events = []
    for ev in resp.get("items", []):
        if ev.get("status") == "cancelled":
            continue

        start = ev.get("start", {})
        end = ev.get("end", {})
        is_all_day = "date" in start

        if is_all_day:
            start_str = start["date"]
            end_str = end["date"]
            start_time = "All Day"
            end_time = ""
        else:
            start_dt = datetime.fromisoformat(start["dateTime"])
            end_dt = datetime.fromisoformat(end["dateTime"])
            start_str = start_dt.isoformat()
            end_str = end_dt.isoformat()
            start_time = start_dt.strftime("%I:%M %p").lstrip("0")
            end_time = end_dt.strftime("%I:%M %p").lstrip("0")

        attendees = []
        for att in ev.get("attendees", []):
            name = att.get("displayName", att.get("email", ""))
            status = att.get("responseStatus", "")
            if not att.get("self", False):
                attendees.append({"name": name, "status": status})

        events.append({
            "id": ev.get("id"),
            "title": ev.get("summary", "(No title)"),
            "start_time": start_time,
            "end_time": end_time,
            "start_iso": start_str,
            "end_iso": end_str,
            "is_all_day": is_all_day,
            "location": ev.get("location", ""),
            "description": (ev.get("description", "") or "")[:500],
            "attendees": attendees,
            "meeting_link": ev.get("hangoutLink", ""),
            "organizer": ev.get("organizer", {}).get("email", ""),
        })

    return events


def format_meetings_for_context(meetings):
    """Format meetings into a text block for Gemini."""
    if not meetings:
        return "No meetings today."

    lines = []
    for i, m in enumerate(meetings, 1):
        if m["is_all_day"]:
            time_str = "All Day"
        else:
            time_str = f"{m['start_time']} – {m['end_time']}"

        line = f"{i}. [{time_str}] {m['title']}"
        if m["location"]:
            line += f"\n   Location: {m['location']}"
        if m["meeting_link"]:
            line += f"\n   Meet: {m['meeting_link']}"
        if m["attendees"]:
            names = ", ".join(a["name"] for a in m["attendees"][:10])
            line += f"\n   With: {names}"
        if m["description"]:
            line += f"\n   Notes: {m['description'][:200]}"
        lines.append(line)

    return "\n\n".join(lines)
