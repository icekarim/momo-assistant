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
