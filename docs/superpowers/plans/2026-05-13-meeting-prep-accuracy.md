# Meeting Prep Accuracy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent Momo meeting prep from mixing people, tasks, and projects by making prep retrieval evidence-scored, source-cited, and conservative.

**Architecture:** Add a focused meeting-prep accuracy layer that sits between the knowledge graph and Gemini. Retrieval produces candidate KG entries, scoring ranks and filters them, prompt context includes stable evidence IDs, and a post-processor drops unsupported generated bullets. Knowledge graph extraction gains explicit person relationship fields so future prep can distinguish attendees from people actually mentioned or owning work.

**Tech Stack:** Python stdlib, FastAPI service code, Firestore-backed KG helpers, Gemini Flash generation, existing unittest/pytest patterns.

---

## File Structure

- Create `meeting_prep_accuracy.py`: title genericity checks, attendee selection, relevance scoring, evidence context formatting, prompt post-processing, and diagnostics.
- Modify `proactive_intelligence.py`: replace broad meeting-prep retrieval with the new accuracy layer and stricter prompt.
- Modify `knowledge_graph.py`: store `mentioned_people`, `attendees`, and `_search_mentioned_people` for new KG entries while preserving legacy compatibility.
- Create `test_meeting_prep_accuracy.py`: unit tests for generic titles, large meetings, relevance scoring, evidence formatting, and post-processing.
- Create or modify `test_knowledge_graph_people_fields.py`: unit tests for KG person relationship storage helpers.

## Baseline Context

The concrete production failure was `last sync part 2`. Calendar showed 24 attendees. The current code queried KG by every attendee plus semantic search on the generic title. That pulled unrelated “sync” entries and weak attendee/social context, then the prompt synthesized it as if it were strongly relevant.

Known baseline issue: full `python3 -m pytest -q` currently fails because generated/mock-heavy tests mutate `sys.modules` and older tests call `main.handle_message` without `background_tasks`. Focused tests must be run directly or with narrow pytest selection.

---

### Task 1: Add Prep Accuracy Utilities And Diagnostics

**Files:**
- Create: `meeting_prep_accuracy.py`
- Test: `test_meeting_prep_accuracy.py`

- [ ] **Step 1: Write failing tests for generic title detection and diagnostics**

Create `test_meeting_prep_accuracy.py` with:

```python
import unittest

from meeting_prep_accuracy import (
    PrepEvidence,
    build_prep_diagnostics,
    is_generic_meeting_title,
)


class TestMeetingPrepAccuracyUtilities(unittest.TestCase):
    def test_generic_titles_are_suppressed(self):
        generic = [
            "last sync part 2",
            "weekly sync",
            "touchbase",
            "catch up",
            "1:1",
            "follow up sync",
        ]
        for title in generic:
            with self.subTest(title=title):
                self.assertTrue(is_generic_meeting_title(title))

    def test_specific_titles_are_not_generic(self):
        specific = [
            "PacSun integration launch review",
            "Saks Global decision makers",
            "Paymenttype escalation readout",
        ]
        for title in specific:
            with self.subTest(title=title):
                self.assertFalse(is_generic_meeting_title(title))

    def test_diagnostics_include_included_and_excluded_reasons(self):
        included = [
            PrepEvidence(
                evidence_id="E1",
                entry={"id": "kg1", "source_title": "last sync part 2", "source_date": "2026-05-13"},
                score=90,
                reasons=["exact source title"],
            )
        ]
        excluded = [
            PrepEvidence(
                evidence_id="X1",
                entry={"id": "kg2", "source_title": "QVC sync", "source_date": "2026-03-24"},
                score=-20,
                reasons=["generic title search"],
            )
        ]

        text = build_prep_diagnostics(
            meeting={"title": "last sync part 2", "attendees": [{"name": "a@example.com"}]},
            included=included,
            excluded=excluded,
            query_labels=["person:a@example.com", "title:last sync part 2"],
        )

        self.assertIn("meeting='last sync part 2'", text)
        self.assertIn("included E1 score=90", text)
        self.assertIn("excluded X1 score=-20", text)
        self.assertIn("generic title search", text)
```

- [ ] **Step 2: Run the tests and verify they fail**

Run:

```bash
python3 -m unittest test_meeting_prep_accuracy.py
```

Expected: import failure because `meeting_prep_accuracy.py` does not exist.

- [ ] **Step 3: Implement the utility module**

Create `meeting_prep_accuracy.py`:

```python
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
    generic_count = sum(1 for token in meaningful if token in _GENERIC_TITLE_PHRASES or token.isdigit())
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
```

- [ ] **Step 4: Run tests and verify they pass**

Run:

```bash
python3 -m unittest test_meeting_prep_accuracy.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add meeting_prep_accuracy.py test_meeting_prep_accuracy.py
git commit -m "feat: add meeting prep accuracy utilities"
```

---

### Task 2: Gate Noisy Retrieval Before It Reaches Gemini

**Files:**
- Modify: `meeting_prep_accuracy.py`
- Modify: `proactive_intelligence.py`
- Test: `test_meeting_prep_accuracy.py`

- [ ] **Step 1: Write failing tests for retrieval planning**

Append tests:

```python
from meeting_prep_accuracy import plan_prep_queries


class TestMeetingPrepRetrievalPlanning(unittest.TestCase):
    def test_large_generic_meeting_skips_title_search_and_limits_people(self):
        meeting = {
            "title": "last sync part 2",
            "description": "Agenda: final handovers for Agnes Jang",
            "organizer": "user@example.com",
            "attendees": [
                {"name": "alex.rivera@example.com"},
                {"name": "pat.miller@example.com"},
                {"name": "jamie.fox@example.com"},
                {"name": "taylor.singh@example.com"},
                {"name": "morgan.reed@example.com"},
                {"name": "riley.chen@example.com"},
                {"name": "daniel.price@example.com"},
                {"name": "casey.morgan@example.com"},
                {"name": "person5@example.com"},
            ],
        }

        plan = plan_prep_queries(meeting)

        self.assertFalse(plan["include_title_semantic_search"])
        self.assertLessEqual(len(plan["people"]), 4)
        self.assertIn("alex.rivera@example.com", plan["people"])
        self.assertIn("user@example.com", plan["people"])

    def test_specific_meeting_allows_title_search(self):
        meeting = {
            "title": "PacSun integration launch review",
            "description": "",
            "organizer": "user@example.com",
            "attendees": [{"name": "jamie.fox@example.com"}],
        }

        plan = plan_prep_queries(meeting)

        self.assertTrue(plan["include_title_semantic_search"])
        self.assertIn("jamie.fox@example.com", plan["people"])
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
python3 -m unittest test_meeting_prep_accuracy.py
```

Expected: import failure for `plan_prep_queries`.

- [ ] **Step 3: Implement query planning**

Add to `meeting_prep_accuracy.py`:

```python
MEETING_PREP_LARGE_MEETING_THRESHOLD = 8
MEETING_PREP_MAX_PERSON_QUERIES = 4


def _text_contains_person(text: str, person: str) -> bool:
    text_tokens = set(_tokens(text))
    person_tokens = [token for token in _tokens(person) if len(token) >= 3 and token != "rokt"]
    return bool(person_tokens and any(token in text_tokens for token in person_tokens))


def plan_prep_queries(meeting: dict[str, Any]) -> dict[str, Any]:
    attendees = [a.get("name", "") for a in meeting.get("attendees", []) if a.get("name")]
    organizer = meeting.get("organizer", "")
    title = meeting.get("title", "")
    description = meeting.get("description", "")
    meeting_text = f"{title} {description}"

    people: list[str] = []
    if organizer:
        people.append(organizer)

    explicit_people = [name for name in attendees if _text_contains_person(meeting_text, name)]
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
```

- [ ] **Step 4: Wire query planning into `_build_meeting_prep`**

In `proactive_intelligence.py`, import `plan_prep_queries` inside `_build_meeting_prep`. Replace person future construction with the planned people list. Only submit title semantic search when `include_title_semantic_search` is true.

- [ ] **Step 5: Run focused tests**

Run:

```bash
python3 -m unittest test_meeting_prep_accuracy.py
python3 -m py_compile proactive_intelligence.py meeting_prep_accuracy.py
```

Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add meeting_prep_accuracy.py proactive_intelligence.py test_meeting_prep_accuracy.py
git commit -m "fix: gate noisy meeting prep retrieval"
```

---

### Task 3: Add Relevance Scoring And Evidence Context Formatting

**Files:**
- Modify: `meeting_prep_accuracy.py`
- Modify: `proactive_intelligence.py`
- Test: `test_meeting_prep_accuracy.py`

- [ ] **Step 1: Write failing tests for scoring**

Append tests:

```python
from meeting_prep_accuracy import select_prep_evidence


class TestMeetingPrepEvidenceScoring(unittest.TestCase):
    def test_selects_exact_source_and_rejects_generic_sync_noise(self):
        meeting = {
            "title": "last sync part 2",
            "description": "Agenda: final handovers for Agnes Jang",
            "attendees": [{"name": "alex.rivera@example.com"}],
        }
        entries = [
            {
                "id": "good",
                "entity_type": "update",
                "name": "Agnes handover",
                "content": "Agnes Jang has handover items to finalize.",
                "source_title": "last sync part 2",
                "source_date": "2026-05-13",
                "mentioned_people": ["Agnes Jang"],
                "related_people": ["Agnes Jang"],
                "related_projects": [],
            },
            {
                "id": "bad",
                "entity_type": "topic",
                "name": "QVC sync review",
                "content": "QVC follow-up sync needs rescheduling.",
                "source_title": "Re: QVC x Rokt follow up",
                "source_date": "2026-03-24",
                "mentioned_people": ["Stephanie"],
                "related_people": ["Stephanie"],
                "related_projects": ["QVC"],
                "_query_label": "title:last sync part 2",
            },
        ]

        included, excluded = select_prep_evidence(meeting, entries, max_items=5)

        self.assertEqual([item.entry["id"] for item in included], ["good"])
        self.assertEqual([item.entry["id"] for item in excluded], ["bad"])
        self.assertIn("exact source title", included[0].reasons)
        self.assertIn("generic title search", excluded[0].reasons)
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
python3 -m unittest test_meeting_prep_accuracy.py
```

Expected: import failure for `select_prep_evidence`.

- [ ] **Step 3: Implement scoring and context formatting**

Add:

```python
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
    project_tokens = {token for project in projects for token in _tokens(project) if len(token) >= 3}
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
```

- [ ] **Step 4: Use scoring in meeting prep**

In `_build_meeting_prep`, after collecting and deduping entries, call `select_prep_evidence()`. Pass only `format_prep_evidence_context(included)` to Gemini. Print `build_prep_diagnostics(...)`.

- [ ] **Step 5: Run tests**

```bash
python3 -m unittest test_meeting_prep_accuracy.py
python3 -m py_compile proactive_intelligence.py meeting_prep_accuracy.py
```

- [ ] **Step 6: Commit**

```bash
git add meeting_prep_accuracy.py proactive_intelligence.py test_meeting_prep_accuracy.py
git commit -m "feat: score meeting prep evidence"
```

---

### Task 4: Evidence-Gate Generated Prep Bullets

**Files:**
- Modify: `meeting_prep_accuracy.py`
- Modify: `proactive_intelligence.py`
- Test: `test_meeting_prep_accuracy.py`

- [ ] **Step 1: Write failing tests for post-processing**

Append tests:

```python
from meeting_prep_accuracy import finalize_evidence_gated_prep


class TestEvidenceGatedPrepOutput(unittest.TestCase):
    def test_drops_uncited_bullets_and_expands_sources(self):
        evidence = [
            PrepEvidence(
                "E1",
                {"source_title": "last sync part 2", "source_date": "2026-05-13"},
                90,
                ["exact source title"],
            )
        ]
        raw = "\n".join([
            "📋 *meeting prep — last sync part 2*",
            "*Agnes handover:* finalize handover items. [E1]",
            "*PacSun:* ask about latency spikes.",
        ])

        final = finalize_evidence_gated_prep("last sync part 2", raw, evidence)

        self.assertIn("Agnes handover", final)
        self.assertIn("source: last sync part 2, 2026-05-13", final)
        self.assertNotIn("PacSun", final)

    def test_no_valid_bullets_returns_low_context_message(self):
        final = finalize_evidence_gated_prep("last sync part 2", "Random uncited bullet", [])

        self.assertIn("meeting prep — last sync part 2", final)
        self.assertIn("I don't have strong prep context", final)
```

- [ ] **Step 2: Run tests and verify they fail**

```bash
python3 -m unittest test_meeting_prep_accuracy.py
```

Expected: import failure for `finalize_evidence_gated_prep`.

- [ ] **Step 3: Implement post-processing**

Add:

```python
def finalize_evidence_gated_prep(title: str, raw_text: str, evidence: list[PrepEvidence]) -> str:
    evidence_by_id = {item.evidence_id: item for item in evidence}
    header = f"📋 *meeting prep — {title}*"
    kept = []
    for line in (raw_text or "").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("📋"):
            continue
        ids = re.findall(r"\bE\d+\b", stripped)
        valid = [evidence_by_id[item_id] for item_id in ids if item_id in evidence_by_id]
        if not valid:
            continue
        source_bits = [
            f"{item.entry.get('source_title', '?')}, {item.entry.get('source_date', '?')}"
            for item in valid
        ]
        cleaned = re.sub(r"\s*\[?E\d+\]?", "", stripped).strip()
        kept.append(f"- {cleaned} _(source: {'; '.join(source_bits)})_")

    if not kept:
        return f"{header}\n\nI don't have strong prep context for this one yet."
    return header + "\n\n" + "\n".join(kept[:6])
```

- [ ] **Step 4: Tighten the prep prompt**

Replace `_PREP_PROMPT` with language requiring evidence IDs:

```python
_PREP_PROMPT = """You are Momo, preparing a quick pre-meeting intel brief. Be casual, concise, and useful.

Meeting: {title}
Attendees: {attendees}
Starts: {start_time}

Use ONLY the evidence items below. Each item has an ID like [E1].

{knowledge_context}

Rules:
- Every bullet MUST cite at least one evidence ID, like [E1].
- Do NOT mention a person, task, project, blocker, departure, deadline, or decision unless an evidence item explicitly supports it.
- Do NOT combine facts across evidence items unless they share the same person or project explicitly.
- If evidence is weak or empty, say you don't have strong prep context.
- Keep 3-6 bullets max.

Start with: 📋 *meeting prep — {title}*"""
```

After Gemini returns text, call `finalize_evidence_gated_prep(title, resp.text.strip(), included_evidence)`.

- [ ] **Step 5: Run tests**

```bash
python3 -m unittest test_meeting_prep_accuracy.py
python3 -m py_compile proactive_intelligence.py meeting_prep_accuracy.py
```

- [ ] **Step 6: Commit**

```bash
git add meeting_prep_accuracy.py proactive_intelligence.py test_meeting_prep_accuracy.py
git commit -m "fix: evidence gate meeting prep output"
```

---

### Task 5: Store Direct People Separately In The Knowledge Graph

**Files:**
- Modify: `knowledge_graph.py`
- Test: `test_knowledge_graph_people_fields.py`

- [ ] **Step 1: Write failing storage tests**

Create `test_knowledge_graph_people_fields.py`:

```python
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.modules["google"] = MagicMock()
sys.modules["google.cloud"] = MagicMock()
sys.modules["google.cloud.firestore"] = MagicMock()
sys.modules["google.cloud.firestore_v1"] = MagicMock()
sys.modules["google.generativeai"] = MagicMock()

import knowledge_graph


class TestKnowledgeGraphPeopleFields(unittest.TestCase):
    def test_prepare_entry_document_separates_attendees_and_mentions(self):
        entry = {
            "entity_type": "commitment",
            "name": "Finalize mapping",
            "content": "Agnes Jang will finalize mapping tables.",
            "owner": "Agnes Jang",
            "mentioned_people": ["Agnes Jang", "Scott"],
            "related_people": ["Agnes Jang", "Scott"],
            "related_projects": ["Carbon"],
            "tags": ["mapping"],
        }

        with patch.object(knowledge_graph, "_get_embedding", return_value=[0.1, 0.2]):
            doc = knowledge_graph._prepare_entry_document(
                entry,
                source_type="meeting",
                source_title="last sync part 2",
                source_date="2026-05-13",
                attendees=["Jessica Francis", "Patrick Tsui"],
            )

        self.assertEqual(doc["mentioned_people"], ["Agnes Jang", "Scott"])
        self.assertEqual(doc["attendees"], ["Jessica Francis", "Patrick Tsui"])
        self.assertIn("agnes", doc["_search_mentioned_people"])
        self.assertIn("jessica", doc["_search_attendees"])
```

- [ ] **Step 2: Run test and verify it fails**

```bash
python3 -m unittest test_knowledge_graph_people_fields.py
```

Expected: `_prepare_entry_document` does not exist.

- [ ] **Step 3: Extract document preparation helper**

In `knowledge_graph.py`, add:

```python
def _prepare_entry_document(
    entry: dict,
    source_type: str,
    source_title: str,
    source_date: str,
    attendees: list[str],
) -> dict:
    people = entry.get("related_people", [])
    owner = entry.get("owner")
    projects = entry.get("related_projects", [])
    mentioned_people = entry.get("mentioned_people") or people
    now_iso = datetime.now(timezone.utc).isoformat()
    doc = {
        "entity_type": entry.get("entity_type", "topic"),
        "name": entry.get("name", ""),
        "content": entry.get("content", ""),
        "status": entry.get("status"),
        "owner": owner,
        "mentioned_people": mentioned_people,
        "attendees": attendees or [],
        "related_people": people,
        "related_projects": projects,
        "tags": entry.get("tags", []),
        "_search_people": _search_tokens_for_people(people, owner),
        "_search_mentioned_people": _search_tokens_for_people(mentioned_people, owner),
        "_search_attendees": _search_tokens_for_people(attendees or []),
        "_search_projects": _search_tokens_for_projects(projects),
        "source_type": source_type,
        "source_id": "",
        "source_title": source_title,
        "source_date": _normalize_source_date(source_date),
        "extracted_at": now_iso,
    }
    try:
        text = _build_embedding_text(entry, source_type=source_type)
        doc["embedding"] = Vector(_get_embedding(text))
        doc["embedding_model"] = config.GEMINI_EMBEDDING_MODEL
    except Exception as exc:
        print(f"  Knowledge graph: embedding generation failed ({exc}), storing without")
    return doc
```

Update `_store_entries()` to call `_prepare_entry_document(...)`, then set `doc["source_id"] = source_id` before `collection.add(doc)`.

- [ ] **Step 4: Update extraction prompt**

In `_EXTRACTION_PROMPT`, add `mentioned_people` to the JSON example and rules:

```text
"mentioned_people": ["people explicitly named in the content"],
```

Rule:

```text
- "mentioned_people" should include only people explicitly named in CONTENT; attendees who are not mentioned should not be included there.
```

- [ ] **Step 5: Run tests**

```bash
python3 -m unittest test_knowledge_graph_people_fields.py
python3 -m py_compile knowledge_graph.py
```

- [ ] **Step 6: Commit**

```bash
git add knowledge_graph.py test_knowledge_graph_people_fields.py
git commit -m "feat: separate KG mentioned people from attendees"
```

---

### Task 6: Integrate Direct-People Signals And Add End-To-End Regression

**Files:**
- Modify: `meeting_prep_accuracy.py`
- Modify: `proactive_intelligence.py`
- Test: `test_meeting_prep_accuracy.py`

- [ ] **Step 1: Write regression for `last sync part 2` pollution**

Append:

```python
class TestLastSyncRegression(unittest.TestCase):
    def test_last_sync_context_excludes_unrelated_sync_and_social_items(self):
        meeting = {
            "title": "last sync part 2",
            "description": "Agenda: final handovers for Agnes Jang",
            "attendees": [{"name": "alex.rivera@example.com"}, {"name": "morgan.reed@example.com"}],
        }
        entries = [
            {
                "id": "handover",
                "entity_type": "commitment",
                "name": "Agnes handover",
                "content": "Agnes Jang needs to finalize data mapping table handover.",
                "source_title": "last sync part 2",
                "source_date": "2026-05-13",
                "mentioned_people": ["Agnes Jang"],
                "related_projects": ["Carbon"],
            },
            {
                "id": "pacsun",
                "entity_type": "blocker",
                "name": "PacSun latency",
                "content": "Jessica Francis is dealing with 2-second latency spikes.",
                "source_title": "Internal Sync: Aftersell 1P for PacSun",
                "source_date": "2026-04-21",
                "mentioned_people": ["Jessica Francis"],
                "related_projects": ["PacSun"],
                "_query_label": "title:last sync part 2",
            },
            {
                "id": "tekken",
                "entity_type": "topic",
                "name": "Tekken 3 Exhibition Matches",
                "content": "Matthew Monjarrez and Daniel Piet played Tekken 3.",
                "source_title": "Tekken 3 Exhibition Matches",
                "source_date": "2026-05-11",
                "mentioned_people": ["Matthew Monjarrez", "Daniel Piet"],
                "related_projects": ["Tekken 3"],
                "_query_label": "person:morgan.reed@example.com",
            },
        ]

        included, excluded = select_prep_evidence(meeting, entries, max_items=10)
        context = format_prep_evidence_context(included)

        self.assertIn("Agnes handover", context)
        self.assertNotIn("PacSun latency", context)
        self.assertNotIn("Tekken", context)
        self.assertEqual({item.entry["id"] for item in excluded}, {"pacsun", "tekken"})
```

- [ ] **Step 2: Run regression and verify it fails if current scoring is too permissive**

```bash
python3 -m unittest test_meeting_prep_accuracy.py
```

Expected: if Task 3 scoring already excludes these, this test may pass. If it passes, record that it locks the regression. If it fails, tune scoring.

- [ ] **Step 3: Tune scoring for social and unrelated project leakage**

If needed, update `score_prep_entry()` so:
- social items require exact source title or project/person overlap in meeting text
- generic title semantic matches are excluded unless exact source title also matches
- unrelated projects receive a negative score when no meeting text overlap exists

- [ ] **Step 4: Run complete focused validation**

```bash
python3 -m unittest test_meeting_prep_accuracy.py test_knowledge_graph_people_fields.py
python3 -m py_compile proactive_intelligence.py knowledge_graph.py meeting_prep_accuracy.py
python3 -m pytest test_meeting_prep_accuracy.py test_knowledge_graph_people_fields.py test_post_meeting_debrief.py -q
```

Expected: all focused tests pass.

- [ ] **Step 5: Commit**

```bash
git add meeting_prep_accuracy.py proactive_intelligence.py test_meeting_prep_accuracy.py
git commit -m "test: lock meeting prep relevance regression"
```

---

### Task 7: Final Verification And PR Prep

**Files:**
- Modify: none unless verification reveals defects.

- [ ] **Step 1: Run focused verification**

```bash
python3 -m unittest test_meeting_prep_accuracy.py test_knowledge_graph_people_fields.py test_post_meeting_debrief.py
python3 -m py_compile proactive_intelligence.py knowledge_graph.py meeting_prep_accuracy.py
```

Expected: pass.

- [ ] **Step 2: Run broader pytest and capture baseline failures**

```bash
python3 -m pytest -q
```

Expected: likely fails with existing generated/mock-heavy test issues. Record failure count and first few categories in the PR summary.

- [ ] **Step 3: Inspect diff**

```bash
git status -sb
git diff --stat origin/main...HEAD
git log --oneline origin/main..HEAD
```

- [ ] **Step 4: Update tracker**

```bash
python3 scripts/update_notion_tracker.py update "Improve meeting prep accuracy" --status "Done"
```

Only run after all focused checks pass.

---

## Self-Review

- Spec coverage: diagnostics, generic title suppression, large-meeting attendee limits, relevance scoring, evidence-gated generation, direct people fields, source-cited output, and regression tests are all covered.
- Placeholder scan: no unfinished placeholder text; all tasks include exact files, commands, and expected outcomes.
- Scope check: one cohesive subsystem, meeting-prep accuracy. KG changes are included only where necessary to distinguish attendee-only context from direct mentions.
- Type consistency: `PrepEvidence`, `plan_prep_queries`, `select_prep_evidence`, `format_prep_evidence_context`, and `finalize_evidence_gated_prep` are introduced before use.
