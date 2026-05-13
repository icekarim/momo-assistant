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


def _entry_query_labels(entry: dict[str, Any]) -> list[str]:
    labels: list[str] = []
    legacy_label = entry.get("_query_label")
    if legacy_label:
        labels.append(str(legacy_label))

    value = entry.get("_query_labels") or []
    if isinstance(value, str):
        value = [value]
    for label in value:
        if label:
            labels.append(str(label))
    return labels


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
    query_labels = _entry_query_labels(entry)

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

    if any(label.startswith("person:") for label in query_labels):
        score += 15
        reasons.append("person query match")

    if any(label.startswith("title:") for label in query_labels) and is_generic_meeting_title(title):
        score -= 70
        reasons.append("generic title search")

    source_type = entry.get("source_type", "")
    if source_type == "calendar" and not reasons:
        score -= 15
        reasons.append("calendar-only weak context")

    return score, reasons or ["weak relevance"]


def select_prep_evidence(
    meeting: dict[str, Any],
    entries: list[dict[str, Any]],
    max_items: int = 12,
    min_score: int = 25,
) -> tuple[list[PrepEvidence], list[PrepEvidence]]:
    scored: list[tuple[int, list[str], dict[str, Any]]] = []
    for entry in entries:
        score, reasons = score_prep_entry(meeting, entry)
        scored.append((score, reasons, entry))

    scored.sort(
        key=lambda item: (
            -item[0],
            str(item[2].get("id", "")),
            str(item[2].get("source_date", "")),
            str(item[2].get("source_title", "")),
            str(item[2].get("name", "")),
            str(item[2].get("content", "")),
        )
    )

    evidence_items = [
        PrepEvidence(f"E{idx}", entry, score, reasons)
        for idx, (score, reasons, entry) in enumerate(scored, start=1)
    ]
    included = [item for item in evidence_items if item.score >= min_score]
    excluded = [item for item in evidence_items if item.score < min_score]
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


def finalize_evidence_gated_prep(
    title: str,
    raw_text: str,
    evidence: list[PrepEvidence],
) -> str:
    header = f"📋 *meeting prep — {title}*"
    evidence_by_id = {item.evidence_id: item for item in evidence}
    valid_ids = set(evidence_by_id)
    kept_lines: list[str] = []

    for line in (raw_text or "").splitlines():
        stripped = line.strip()
        if not stripped or stripped == header or stripped.lower().startswith("📋 *meeting prep"):
            continue

        cited_ids: list[str] = []
        for match in re.finditer(r"(?<![A-Za-z0-9])\[?(E\d+)\]?(?![A-Za-z0-9])", stripped):
            evidence_id = match.group(1)
            if evidence_id in valid_ids and evidence_id not in cited_ids:
                cited_ids.append(evidence_id)

        if not cited_ids:
            continue

        cleaned = re.sub(r"\s*_?\(source:[^)]+\)_?\s*$", "", stripped, flags=re.IGNORECASE).strip()
        cleaned = re.sub(
            r"\s*(?<![A-Za-z0-9])\[?E\d+\]?(?![A-Za-z0-9])",
            "",
            cleaned,
        ).strip()
        cleaned = re.sub(r"^#{1,6}\s*", "", cleaned).strip()
        cleaned = re.sub(r"^(?:[-*•]\s+|\d+[.)]\s+)", "", cleaned).strip()
        if not cleaned:
            continue

        sources = []
        for evidence_id in cited_ids:
            entry = evidence_by_id[evidence_id].entry
            source_title = entry.get("source_title") or "unknown source"
            source_date = entry.get("source_date") or "unknown date"
            source = f"{source_title}, {source_date}"
            if source not in sources:
                sources.append(source)

        kept_lines.append(f"- {cleaned} _(source: {'; '.join(sources)})_")
        if len(kept_lines) >= 6:
            break

    if not kept_lines:
        return f"{header}\nI don't have strong prep context for this one yet."

    return "\n".join([header, *kept_lines])


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
