"""Google Chat Cards v2 builders — PURE and dependency-free.

This module builds plain dicts/lists for the interactive task-tray UX
(PLAN_card_task_ux.md §5.1, §4.2, §4.3). It imports nothing heavy (no
google, no config, no firestore) so it can be unit-tested with a bare
``import cards``.

Public API:
    build_task_tray_card(batch_id, source, rows) -> list[dict]

Row schema (design §6.5):
    {
        "taskId":   str,
        "title":    str,
        "due":      str | None,
        "owner":    str | None,
        "priority": str | None,   # "high" -> 🔴, else 🟡
        "state":    str,          # pending | added | already_exists | dismissed
    }
"""

# --------------------------------------------------------------------------
# constants
# --------------------------------------------------------------------------
PENDING = "pending"
ADDED = "added"
ALREADY_EXISTS = "already_exists"
ALREADY_COMPLETED = "already_completed"
DISMISSED = "dismissed"
FAILED = "failed"

# A failed create stays actionable (Add/Edit/Dismiss) so the user can retry.
_FAILED_NOTE = "⚠️ Couldn't add — tap Add to retry"

# Cards v2 hard limit is 100 widgets/card (§4.3). A pending row costs
# 2 widgets (textParagraph + buttonList); footer/note cost 1 each. We cap
# well below the hard limit so we never silently exceed it.
MAX_WIDGETS = 100
MAX_ROWS = 20  # design §4.3: a debrief rarely exceeds 5; ~20 is the safe cap

_HIGH_PRIORITY = "high"

# Terminal-state display text (§4.2)
_TERMINAL_TEXT = {
    ADDED: "✅ Added to Google Tasks",
    ALREADY_EXISTS: "↩︎ Already in your tasks",
    ALREADY_COMPLETED: "✅ Already completed",
    DISMISSED: "✕ Dismissed",
}

# circled-number glyphs ①..⑳ (U+2460..U+2473) for indices 1..20
_CIRCLED_BASE = 0x2460
_CIRCLED_MAX = 20


# --------------------------------------------------------------------------
# small pure helpers
# --------------------------------------------------------------------------
def _index_glyph(n):
    """1-based index -> ①②③… ; falls back to 'n.' beyond the glyph range."""
    if 1 <= n <= _CIRCLED_MAX:
        return chr(_CIRCLED_BASE + n - 1)
    return f"{n}."


def _priority_emoji(priority):
    return "🔴" if (priority or "").lower() == _HIGH_PRIORITY else "🟡"


def _param(key, value):
    return {"key": key, "value": value}


def _row_text(index, row):
    """Render the textParagraph body for a single row."""
    glyph = _index_glyph(index)
    title = row.get("title", "")
    line1 = f"{glyph} <b>{title}</b>"

    meta_parts = [_priority_emoji(row.get("priority"))]
    owner = row.get("owner")
    if owner:
        meta_parts.append(owner)
    due = row.get("due")
    if due:
        meta_parts.append(f"due {due}")

    return f"{line1}<br>{' · '.join(meta_parts)}"


def _row_buttons(batch_id, row, chat_url):
    """The three action buttons for a pending row (§5.1).

    ADD-ON CONTRACT (HANDOFF §4/§7.1): ``function`` is the FULL ``/chat`` endpoint
    URL (``chat_url``) — a bare function name cannot be routed by the add-on
    framework. The real action travels in ``parameters`` under ``actionName``.
    """
    task_id = row.get("taskId")

    def _action_params(action_name):
        return [
            _param("actionName", action_name),
            _param("batchId", batch_id),
            _param("taskId", task_id),
        ]

    return {
        "buttonList": {
            "buttons": [
                {
                    "text": "✅ Add",
                    "onClick": {
                        "action": {
                            "function": chat_url,
                            "loadIndicator": "SPINNER",
                            "parameters": _action_params("task_add"),
                        }
                    },
                },
                {
                    "text": "✏️ Edit",
                    "onClick": {
                        "action": {
                            "function": chat_url,
                            "interaction": "OPEN_DIALOG",
                            "parameters": _action_params("task_edit"),
                        }
                    },
                },
                {
                    "text": "✕ Dismiss",
                    "onClick": {
                        "action": {
                            "function": chat_url,
                            "parameters": _action_params("task_dismiss"),
                        }
                    },
                },
            ]
        }
    }


def _row_section(batch_id, index, row, chat_url):
    """One section per row: text + (buttons if actionable, else collapsed text).

    A FAILED row is rendered like a pending row (keeps Add/Edit/Dismiss so the
    user can retry) but with a warning note prefixed to its text."""
    state = row.get("state")
    text = _row_text(index, row)
    if state == FAILED:
        text = f"{_FAILED_NOTE}<br>{text}"
    widgets = [{"textParagraph": {"text": text}}]
    if state in (PENDING, FAILED):
        widgets.append(_row_buttons(batch_id, row, chat_url))
    else:
        terminal = _TERMINAL_TEXT.get(state)
        if terminal:
            widgets.append({"textParagraph": {"text": terminal}})
    return {"widgets": widgets}


def _footer_section(batch_id, chat_url):
    """Add-all / Dismiss-all footer (only rendered when pending rows remain).

    Like the row buttons, footer buttons route to the full ``/chat`` URL with the
    action carried in ``parameters.actionName`` (HANDOFF §4/§7.1)."""

    def _action_params(action_name):
        return [_param("actionName", action_name), _param("batchId", batch_id)]

    return {
        "widgets": [
            {
                "buttonList": {
                    "buttons": [
                        {
                            "text": "✅ Add all",
                            "onClick": {
                                "action": {
                                    "function": chat_url,
                                    "loadIndicator": "SPINNER",
                                    "parameters": _action_params("task_add_all"),
                                }
                            },
                        },
                        {
                            "text": "✕ Dismiss all",
                            "onClick": {
                                "action": {
                                    "function": chat_url,
                                    "parameters": _action_params("task_dismiss_all"),
                                }
                            },
                        },
                    ]
                }
            }
        ]
    }


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------
def build_task_tray_card(batch_id, source, rows, chat_url=""):
    """Build the cardsV2 task-tray payload for a batch of suggested tasks.

    Returns a cardsV2 LIST with a single card:
        [{"cardId": "tasktray-<batch_id>", "card": {...}}]

    ``chat_url`` is the FULL ``/chat`` endpoint URL the add-on must call back
    (production: ``config.MOMO_SERVICE_URL + "/chat"``). Every button emits
    ``onClick.action.function == chat_url`` with the real action in
    ``parameters.actionName`` — a bare function name cannot be routed by the
    add-on framework (HANDOFF §4/§7.1). The default ``""`` keeps this module pure
    and bare-importable; production call sites always pass the URL.

    See PLAN_card_task_ux.md §5.1/§4.2/§4.3 for the contract.
    """
    rows = list(rows or [])

    # widget-budget guard (§4.3): cap rows so we never exceed 100 widgets.
    truncated = max(0, len(rows) - MAX_ROWS)
    visible_rows = rows[:MAX_ROWS] if truncated else rows

    sections = []
    for i, row in enumerate(visible_rows, start=1):
        sections.append(_row_section(batch_id, i, row, chat_url))

    if truncated:
        sections.append(
            {
                "widgets": [
                    {
                        "textParagraph": {
                            "text": f"… +{truncated} more not shown (widget limit)"
                        }
                    }
                ]
            }
        )

    pending_count = sum(1 for r in visible_rows if r.get("state") == PENDING)

    # footer only when at least one pending row remains (§4.2)
    if pending_count >= 1:
        sections.append(_footer_section(batch_id, chat_url))

    noun = "task" if pending_count == 1 else "tasks"
    subtitle = f"{pending_count} suggested {noun}"

    card = {
        "header": {"title": source, "subtitle": subtitle},
        "sections": sections,
    }

    return [{"cardId": f"tasktray-{batch_id}", "card": card}]
