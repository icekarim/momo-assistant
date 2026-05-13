"""Accuracy helpers for proactive meeting-prep retrieval and generation."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any


_GENERIC_TITLE_PHRASES = {
    "sync",
    "last sync",
    "weekly sync",
    "daily sync",
    "touchbase",
    "touch base",
    "catchup",
    "catch up",
    "follow up",
    "1:1",
    "one on one",
    "meeting",
    "part",
}
_GENERIC_TITLE_TOKENS = {
    token
    for phrase in _GENERIC_TITLE_PHRASES
    for token in re.findall(r"[a-z0-9]+", phrase.lower())
}
MEETING_PREP_LARGE_MEETING_THRESHOLD = 8
MEETING_PREP_MAX_PERSON_QUERIES = 4


@dataclass
class PrepEvidence:
    evidence_id: str
    entry: dict[str, Any]
    score: int
    reasons: list[str]


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def is_generic_meeting_title(title: str) -> bool:
    normalized = " ".join(_tokens(title))
    if not normalized:
        return True
    tokens = normalized.split()
    meaningful = [
        token for token in tokens
        if token not in {"the", "a", "an", "and", "or", "with", "for", "to"}
    ]
    if len(meaningful) <= 2 and any(token in _GENERIC_TITLE_PHRASES for token in meaningful):
        return True
    if normalized in _GENERIC_TITLE_PHRASES:
        return True
    generic_count = sum(1 for token in meaningful if token in _GENERIC_TITLE_TOKENS or token.isdigit())
    return bool(meaningful) and generic_count == len(meaningful)


def _text_contains_person(text: str, person: str) -> bool:
    text_tokens = set(_tokens(text))
    person_tokens = [
        token for token in _tokens(person)
        if len(token) >= 3 and token != "rokt"
    ]
    return bool(person_tokens and any(token in text_tokens for token in person_tokens))


def plan_prep_queries(meeting: dict[str, Any]) -> dict[str, Any]:
    attendees = [
        attendee.get("name", "")
        for attendee in meeting.get("attendees", [])
        if attendee.get("name")
    ]
    organizer = meeting.get("organizer", "")
    title = meeting.get("title", "")
    description = meeting.get("description", "")
    meeting_text = f"{title} {description}"

    people: list[str] = []
    if organizer:
        people.append(organizer)

    explicit_people = [
        name for name in attendees
        if _text_contains_person(meeting_text, name)
    ]
    for name in explicit_people:
        if name not in people:
            people.append(name)

    if len(attendees) <= MEETING_PREP_LARGE_MEETING_THRESHOLD:
        for name in attendees:
            if name not in people:
                people.append(name)

    people = people[:MEETING_PREP_MAX_PERSON_QUERIES]

    return {
        "people": people,
        "include_title_semantic_search": not is_generic_meeting_title(title),
        "is_large_meeting": len(attendees) > MEETING_PREP_LARGE_MEETING_THRESHOLD,
    }


def build_prep_diagnostics(
    meeting: dict[str, Any],
    included: list[PrepEvidence],
    excluded: list[PrepEvidence],
    query_labels: list[str],
) -> str:
    title = meeting.get("title", "")
    attendees = meeting.get("attendees", []) or []
    lines = [
        f"Meeting prep retrieval: meeting='{title}' attendees={len(attendees)} queries={query_labels}",
    ]
    for item in included:
        entry = item.entry
        lines.append(
            "  included "
            f"{item.evidence_id} score={item.score} source='{entry.get('source_title', '?')}' "
            f"date={entry.get('source_date', '?')} reasons={item.reasons}"
        )
    for item in excluded:
        entry = item.entry
        lines.append(
            "  excluded "
            f"{item.evidence_id} score={item.score} source='{entry.get('source_title', '?')}' "
            f"date={entry.get('source_date', '?')} reasons={item.reasons}"
        )
    return "\n".join(lines)
