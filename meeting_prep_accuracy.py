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


def _norm(text: str) -> str:
    return " ".join(_tokens(text))


def _entry_people(entry: dict[str, Any]) -> set[str]:
    people = set()
    for key in ("mentioned_people", "owners", "related_people"):
        value = entry.get(key) or []
        if isinstance(value, str):
            value = [value]
        for person in value:
            people.update(_tokens(person))
    owner = entry.get("owner")
    if owner:
        people.update(_tokens(owner))
    return {token for token in people if len(token) >= 3 and token != "rokt"}


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


def score_prep_entry(meeting: dict[str, Any], entry: dict[str, Any]) -> tuple[int, list[str]]:
    score = 0
    reasons: list[str] = []
    title = meeting.get("title", "")
    query_label = entry.get("_query_label", "")

    if _norm(entry.get("source_title", "")) == _norm(title):
        score += 80
        reasons.append("exact source title")

    meeting_text = f"{meeting.get('title', '')} {meeting.get('description', '')}"
    meeting_tokens = set(_tokens(meeting_text))
    if meeting_tokens & _entry_people(entry):
        score += 35
        reasons.append("person mentioned in meeting text")

    projects = entry.get("related_projects") or []
    project_tokens = {
        token
        for project in projects
        for token in _tokens(project)
        if len(token) >= 3
    }
    if meeting_tokens & project_tokens:
        score += 30
        reasons.append("project appears in meeting text")

    if query_label.startswith("person:"):
        score += 15
        reasons.append("person query match")

    if query_label.startswith("title:") and is_generic_meeting_title(title):
        score -= 70
        reasons.append("generic title search")

    source_type = entry.get("source_type", "")
    if source_type == "calendar" and not reasons:
        score -= 15
        reasons.append("calendar-only weak context")

    name_and_content = f"{entry.get('name', '')} {entry.get('content', '')}".lower()
    if any(word in name_and_content for word in ("tekken", "farewell social", "social")):
        score -= 25
        reasons.append("social context")

    return score, reasons or ["weak relevance"]


def select_prep_evidence(
    meeting: dict[str, Any],
    entries: list[dict[str, Any]],
    max_items: int = 12,
    min_score: int = 25,
) -> tuple[list[PrepEvidence], list[PrepEvidence]]:
    included: list[PrepEvidence] = []
    excluded: list[PrepEvidence] = []
    for idx, entry in enumerate(entries, start=1):
        score, reasons = score_prep_entry(meeting, entry)
        evidence = PrepEvidence(f"E{idx}", entry, score, reasons)
        if score >= min_score:
            included.append(evidence)
        else:
            excluded.append(evidence)

    included.sort(key=lambda item: item.score, reverse=True)
    return included[:max_items], excluded + included[max_items:]


def format_prep_evidence_context(evidence: list[PrepEvidence]) -> str:
    if not evidence:
        return "(No strong prior context found for this meeting.)"
    lines = []
    for item in evidence:
        e = item.entry
        lines.append(
            f"[{item.evidence_id}] date={e.get('source_date', '?')} "
            f"type={e.get('entity_type', '?')} source={e.get('source_title', '?')} "
            f"name={e.get('name', '')} content={e.get('content', '')}"
        )
    return "\n".join(lines)


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
