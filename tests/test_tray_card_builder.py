"""Tests for cards.build_task_tray_card — Google Chat Cards v2 task tray.

TDD: the cards module is required to be PURE and dependency-free, so this imports
it directly with no stubbing.

ADD-ON CONTRACT (HANDOFF_addon_cards.md §4/§7.1): for a Google Workspace add-on a
button's ``onClick.action.function`` MUST be the FULL ``/chat`` endpoint URL, with
the real action passed in ``parameters`` under the key ``actionName``. A bare
function name ("task_add") cannot be routed by the add-on framework — that was the
core bug. So ``build_task_tray_card`` takes a ``chat_url`` and every button emits
``function == chat_url`` + ``{"key": "actionName", "value": "task_add"|...}``.
"""
import cards


# The full /chat endpoint URL the add-on must call back. In production this is
# config.MOMO_SERVICE_URL + "/chat"; here it is a fixed test value.
CHAT_URL = "https://momo.example/chat"


# --------------------------------------------------------------------------
# helpers — generic recursive traversal of the cardsV2 structure
# --------------------------------------------------------------------------
def _build(batch_id, source, rows):
    """Always render with the add-on chat_url so buttons carry the full URL."""
    return cards.build_task_tray_card(batch_id, source, rows, chat_url=CHAT_URL)


def _card(result):
    """Unwrap the single card dict from the cardsV2 list."""
    assert isinstance(result, list)
    assert len(result) == 1
    return result[0]["card"]


def _sections(result):
    return _card(result)["sections"]


def _iter_widgets(result):
    for section in _sections(result):
        for widget in section.get("widgets", []):
            yield widget


def _all_buttons(result):
    """Every button across every buttonList in the card."""
    buttons = []
    for widget in _iter_widgets(result):
        bl = widget.get("buttonList")
        if bl:
            buttons.extend(bl.get("buttons", []))
    return buttons


def _button_functions(result):
    """The raw onClick.action.function values (now full URLs)."""
    fns = []
    for btn in _all_buttons(result):
        fn = btn.get("onClick", {}).get("action", {}).get("function")
        if fn:
            fns.append(fn)
    return fns


def _params_dict(button):
    """Flatten action.parameters [{key,value}] into a plain dict."""
    params = button.get("onClick", {}).get("action", {}).get("parameters", [])
    return {p["key"]: p["value"] for p in params}


def _button_action_names(result):
    """The parameters.actionName values across every button — the real dispatch
    keys now that ``function`` carries the URL."""
    names = []
    for btn in _all_buttons(result):
        name = _params_dict(btn).get("actionName")
        if name:
            names.append(name)
    return names


def _button_by_action(result, action_name):
    """Find the single button whose parameters.actionName == action_name."""
    return next(
        b for b in _all_buttons(result)
        if _params_dict(b).get("actionName") == action_name
    )


def _all_text(result):
    texts = []
    for widget in _iter_widgets(result):
        tp = widget.get("textParagraph")
        if tp:
            texts.append(tp.get("text", ""))
    return texts


def _count_widgets_recursive(obj):
    """Count every 'widgets' list entry anywhere in the structure."""
    total = 0
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "widgets" and isinstance(value, list):
                total += len(value)
            total += _count_widgets_recursive(value)
    elif isinstance(obj, list):
        for item in obj:
            total += _count_widgets_recursive(item)
    return total


def _pending_row(task_id="t1", title="Confirm OneTrust status", **kw):
    row = {
        "taskId": task_id,
        "title": title,
        "due": "Jun 17",
        "owner": "Karim",
        "priority": "high",
        "state": "pending",
    }
    row.update(kw)
    return row


# --------------------------------------------------------------------------
# ADD-ON button contract: function == full URL, action in parameters.actionName
# --------------------------------------------------------------------------
def test_every_button_function_is_the_full_chat_url():
    """The core fix: NO button may carry a bare function name; every one routes
    to the full /chat URL."""
    rows = [_pending_row("t1")]
    result = _build("b1", "Petco SDK+ Sync", rows)
    fns = _button_functions(result)
    assert fns, "expected buttons"
    for fn in fns:
        assert fn == CHAT_URL, f"button function must be the full URL, got {fn!r}"
    # the bare names must NOT appear as the function value anymore
    for bare in ("task_add", "task_edit", "task_dismiss",
                 "task_add_all", "task_dismiss_all"):
        assert bare not in fns


def test_pending_row_has_three_buttons():
    rows = [_pending_row("t1")]
    result = _build("b1", "Petco SDK+ Sync", rows)
    # three row actions present via parameters.actionName
    row_actions = {"task_add", "task_edit", "task_dismiss"}
    names = _button_action_names(result)
    for name in row_actions:
        assert name in names, f"missing row action {name}"
    # each row button carries the URL function + actionName + batchId + taskId
    for btn in _all_buttons(result):
        params = _params_dict(btn)
        action_name = params.get("actionName")
        if action_name in row_actions:
            assert btn["onClick"]["action"]["function"] == CHAT_URL
            assert params.get("batchId") == "b1", f"{action_name} missing batchId"
            assert params.get("taskId") == "t1", f"{action_name} missing taskId"


def test_add_button_has_spinner_and_url_and_actionname():
    rows = [_pending_row("t1")]
    result = _build("b1", "src", rows)
    add = _button_by_action(result, "task_add")
    action = add["onClick"]["action"]
    assert action["function"] == CHAT_URL
    assert action["loadIndicator"] == "SPINNER"
    assert _params_dict(add)["actionName"] == "task_add"


def test_edit_button_opens_dialog_with_url_and_actionname():
    rows = [_pending_row("t1")]
    result = _build("b1", "src", rows)
    edit = _button_by_action(result, "task_edit")
    action = edit["onClick"]["action"]
    assert action["function"] == CHAT_URL
    assert action["interaction"] == "OPEN_DIALOG"
    assert _params_dict(edit)["actionName"] == "task_edit"


def test_dismiss_button_url_and_actionname():
    rows = [_pending_row("t1")]
    result = _build("b1", "src", rows)
    dismiss = _button_by_action(result, "task_dismiss")
    assert dismiss["onClick"]["action"]["function"] == CHAT_URL
    assert _params_dict(dismiss)["actionName"] == "task_dismiss"


# --------------------------------------------------------------------------
# row-state rendering (unchanged behavior)
# --------------------------------------------------------------------------
def test_added_row_collapses_no_buttons():
    rows = [_pending_row("t1", state="added")]
    result = _build("b1", "src", rows)
    # no per-row buttons at all; footer also hidden (no pending)
    assert _all_buttons(result) == []
    assert any("Added to Google Tasks" in t for t in _all_text(result))


def test_already_exists_row_text():
    rows = [_pending_row("t1", state="already_exists")]
    result = _build("b1", "src", rows)
    assert any("↩︎ Already in your tasks" in t for t in _all_text(result))
    assert "task_add" not in _button_action_names(result)


def test_dismissed_row_text():
    rows = [_pending_row("t1", state="dismissed")]
    result = _build("b1", "src", rows)
    assert any("✕ Dismissed" in t for t in _all_text(result))
    assert _all_buttons(result) == []


def test_failed_row_keeps_buttons_with_warning():
    rows = [_pending_row("t1", state="failed")]
    result = _build("b1", "src", rows)
    assert any("Couldn't add" in t for t in _all_text(result))
    names = _button_action_names(result)
    for name in ("task_add", "task_edit", "task_dismiss"):
        assert name in names, f"failed row must keep {name} button for retry"
    # and those retry buttons still carry the full URL
    for fn in _button_functions(result):
        assert fn == CHAT_URL


def test_already_completed_row_terminal():
    rows = [_pending_row("t1", state="already_completed")]
    result = _build("b1", "src", rows)
    assert any("✅ Already completed" in t for t in _all_text(result))
    assert _all_buttons(result) == []


def test_footer_present_with_pending():
    rows = [_pending_row("t1"), _pending_row("t2", state="added")]
    result = _build("b1", "src", rows)
    names = _button_action_names(result)
    assert "task_add_all" in names
    assert "task_dismiss_all" in names
    # footer buttons carry the URL function + actionName + batchId (and only batchId)
    add_all = _button_by_action(result, "task_add_all")
    assert add_all["onClick"]["action"]["function"] == CHAT_URL
    params = _params_dict(add_all)
    assert params.get("actionName") == "task_add_all"
    assert params.get("batchId") == "b1"
    assert "taskId" not in params


def test_footer_hidden_when_no_pending():
    rows = [
        _pending_row("t1", state="added"),
        _pending_row("t2", state="dismissed"),
        _pending_row("t3", state="already_exists"),
    ]
    result = _build("b1", "src", rows)
    names = _button_action_names(result)
    assert "task_add_all" not in names
    assert "task_dismiss_all" not in names
    assert _all_buttons(result) == []


def test_cardid_format():
    result = _build("abc123", "src", [_pending_row("t1")])
    assert result[0]["cardId"] == "tasktray-abc123"


def test_widget_count_under_100():
    rows = [_pending_row(f"t{i}") for i in range(5)]
    result = _build("b1", "src", rows)
    assert _count_widgets_recursive(result) <= 100


# --------------------------------------------------------------------------
# supporting tests (header, owner/due rendering, priority emoji, budget cap)
# --------------------------------------------------------------------------
def test_header_title_and_subtitle():
    rows = [_pending_row("t1"), _pending_row("t2")]
    result = _build("b1", "Petco SDK+ Sync", rows)
    header = _card(result)["header"]
    assert header["title"] == "Petco SDK+ Sync"
    assert "2" in header["subtitle"]


def test_priority_emoji_high_vs_normal():
    rows = [
        _pending_row("t1", priority="high"),
        _pending_row("t2", priority="medium"),
    ]
    result = _build("b1", "src", rows)
    texts = " ".join(_all_text(result))
    assert "🔴" in texts
    assert "🟡" in texts


def test_due_omitted_when_absent():
    rows = [_pending_row("t1", due=None)]
    result = _build("b1", "src", rows)
    assert not any("due" in t for t in _all_text(result))


def test_widget_budget_cap_does_not_exceed_100():
    rows = [_pending_row(f"t{i}") for i in range(40)]
    result = _build("b1", "src", rows)
    assert _count_widgets_recursive(result) <= 100


# --------------------------------------------------------------------------
# default chat_url="" stays a harmless pure render (no URL/actionName required
# for the dormant-import contract; callers always pass chat_url in production)
# --------------------------------------------------------------------------
def test_default_chat_url_is_empty_string():
    """Without a chat_url the builder still renders (pure module contract); the
    function value is empty rather than a bare name."""
    rows = [_pending_row("t1")]
    result = cards.build_task_tray_card("b1", "src", rows)
    for btn in _all_buttons(result):
        assert btn["onClick"]["action"]["function"] == ""
        # actionName is still present so dispatch works regardless of URL
        assert _params_dict(btn).get("actionName") in {
            "task_add", "task_edit", "task_dismiss",
            "task_add_all", "task_dismiss_all",
        }
