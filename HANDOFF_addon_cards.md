# HANDOFF — Build add-on-correct interactive task-creation cards

Status: ready to implement
Audience: a fresh agent with NO prior context on this work
Owner: Karim
Last verified live: 2026-06-18, deployed revision `momo-00150-5fx`

---

## 0. TL;DR (read this first)

momo is a **Google Workspace add-on** Chat app (confirmed in the Chat API Configuration: "Build this Chat app as a Workspace add-on" = CHECKED). We tried to add interactive Cards v2 (Add/Edit/Dismiss buttons) for task suggestions and **every click failed** with "momo is unable to process your request" — the click never reached the `/chat` endpoint.

**Root cause (verified against Google's official add-on samples):** for a Workspace add-on, a card button's `onClick.action.function` **must be the FULL HTTPS endpoint URL** (e.g. `https://momo-ia4bhvubwa-uc.a.run.app/chat`), with the action name passed in `parameters`. momo was sending a **bare function name** (`"function": "task_add"`), which an add-on cannot route — so the click was silently dropped. It was **never** a sync-vs-async problem in the way we first thought; it was the button action format + the response envelope.

**What is achievable:** interactive cards returned **synchronously** in reply to a user MESSAGE (i.e. conversational "add a task …"). Their buttons WILL call back if you use the full-URL function + the add-on response envelopes.

**What is NOT achievable (hard platform limit — do not attempt):** interactive buttons on **asynchronously posted** cards (cron-triggered debriefs, anything sent via `spaces.messages.create`). Add-ons cannot route button clicks for unsolicited/async cards. **Debrief suggestions stay as plain text** (already implemented).

**Your job:** re-implement the **conversational task-CREATE** path to emit an interactive tray card the add-on-correct way, verify with a real live click, and deploy. Do NOT touch the debrief text flow.

---

## 1. Objective & scope

IN SCOPE:
- When a user types "add a task to X" (and the agent decides to create a task), reply **synchronously** with an interactive tray card (✅ Add / ✏️ Edit / ✕ Dismiss) whose buttons actually work on the add-on framework.
- Handle the button click (`buttonClickedPayload`) → perform the create via `tasks_service.create_task` → update the card in place ("✅ Added").

OUT OF SCOPE (do NOT change):
- **Debrief / proactive cards** → they stay PLAIN TEXT. Async cards can't have working buttons. `briefing._process_debrief_tasks` is already text — leave it.
- `update` / `complete` / `delete` conversational actions → keep on the existing text "reply yes" approval flow (only `create` becomes a card).
- The conversational-CRUD text approval machinery (it's load-bearing — see §6).

---

## 2. Why the previous attempts failed (do not repeat these)

1. **Bare function name** in `onClick.action.function` (`"task_add"`). Add-ons require the **full endpoint URL**. THIS is the core bug.
2. **Async REST-posted cards** (`send_chat_message(space, cards=...)` via service account) — buttons on these can NEVER call back for an add-on. Cards must be **returned synchronously** as the HTTP response to a MESSAGE event.
3. **Unverified response envelopes** — we guessed the add-on `hostAppDataAction` shapes (the old "§10 UNVERIFIED" markers). The correct shapes are now confirmed by Google's official samples (§4).

---

## 3. Authoritative references (USE THESE — they are the source of truth)

- Convert guide: https://developers.google.com/workspace/add-ons/chat/convert
- Add-on build/interactions: https://developers.google.com/workspace/add-ons/chat/build
- Send messages (add-on, response envelopes): https://developers.google.com/workspace/add-ons/chat/send-messages
- Dialogs (add-on): https://developers.google.com/workspace/add-ons/chat/dialogs
- Event objects (`chat.buttonClickedPayload`, `commonEventObject`): https://developers.google.com/workspace/add-ons/concepts/event-objects
- **Official working Python samples (COPY THESE PATTERNS):**
  - `googleworkspace/add-ons-samples` → `python/chat/contact-form-app/main.py` (button click handler + `onClick.action.function` = full URL + `parameters`, dialog navigation)
  - `googleworkspace/add-ons-samples` → `python/chat/avatar-app/main.py` (synchronous MESSAGE → card response via `hostAppDataAction.chatDataAction.createMessageAction`)

VERIFY the exact payloads against these before coding. Do NOT trust this doc's payloads over the live official samples if they ever differ.

---

## 4. The correct add-on payloads (verified against official samples)

### 4a. Outbound: synchronous card reply to a MESSAGE (NEW message with a card)
```json
{
  "hostAppDataAction": {
    "chatDataAction": {
      "createMessageAction": {
        "message": {
          "text": "got it — here's the task to add:",
          "cardsV2": [{
            "cardId": "tasktray-<batchId>",
            "card": {
              "header": {"title": "New task"},
              "sections": [{
                "widgets": [
                  {"textParagraph": {"text": "<b>① test the card fix</b><br>🟡 due 2026-06-18"}},
                  {"buttonList": {"buttons": [
                    {"text": "✅ Add", "onClick": {"action": {
                      "function": "https://momo-ia4bhvubwa-uc.a.run.app/chat",
                      "parameters": [
                        {"key": "actionName", "value": "task_add"},
                        {"key": "batchId", "value": "<batchId>"},
                        {"key": "taskId", "value": "t1"}
                      ]
                    }}},
                    {"text": "✏️ Edit", "onClick": {"action": {
                      "function": "https://momo-ia4bhvubwa-uc.a.run.app/chat",
                      "interaction": "OPEN_DIALOG",
                      "parameters": [{"key":"actionName","value":"task_edit"},{"key":"batchId","value":"<batchId>"},{"key":"taskId","value":"t1"}]
                    }}},
                    {"text": "✕ Dismiss", "onClick": {"action": {
                      "function": "https://momo-ia4bhvubwa-uc.a.run.app/chat",
                      "parameters": [{"key":"actionName","value":"task_dismiss"},{"key":"batchId","value":"<batchId>"},{"key":"taskId","value":"t1"}]
                    }}}
                  ]}}
                ]
              }]
            }
          }]
        }
      }
    }
  }
}
```
**The critical line is `"function": "https://…/chat"` (full URL), with the real action in `parameters.actionName`.** This is what momo got wrong.

### 4b. Inbound: the button-click event (add-on format)
```jsonc
{
  "chat": { "buttonClickedPayload": { "message": {"name": "spaces/.../messages/..."}, ... } },
  "commonEventObject": {
    "parameters": { "actionName": "task_add", "batchId": "...", "taskId": "t1" }
  }
}
```
NOTE: `_parse_event` in main.py ALREADY detects `buttonClickedPayload` → sets `event_type="CARD_CLICKED"`, `is_addon=True`, and extracts `parameters`. **But the dispatch key for add-ons is `parameters["actionName"]`, NOT `invoked_function`** (because `function` is now the URL). Adjust `handle_card_click` to dispatch on `parameters.get("actionName")`.

### 4c. Outbound: response to a click (update the card in place)
```json
{
  "hostAppDataAction": {
    "chatDataAction": {
      "updateMessageAction": {
        "message": {
          "cardsV2": [ /* re-rendered tray with the clicked row collapsed to "✅ Added" */ ]
        }
      }
    }
  }
}
```

### 4d. Edit dialog (OPEN_DIALOG) — add-on navigation model
Per the contact-form sample, dialogs use `{"action": {"navigations": [{"pushCard": {...}}]}}` (the add-on `RenderActions` model), NOT the Chat-API `actionResponse.type=DIALOG`. If you implement Edit, follow `contact-form-app/main.py` exactly. (Edit is OPTIONAL for v1 — Add + Dismiss are the priority.)

---

## 5. Current codebase state (what exists already)

The card infrastructure is present but **dormant** (built earlier, then conversational creates were reverted to text):

| File | Symbol | State | Notes |
|---|---|---|---|
| `cards.py` | `build_task_tray_card(batch_id, source, rows)` | EXISTS | Renders the tray. **Currently emits bare `function` names — must change to full-URL + `parameters.actionName`.** Pure module, no deps. |
| `main.py` | `_parse_event` | EXISTS | Already detects `CARD_CLICKED`/`buttonClickedPayload`, extracts `invoked_function`/`parameters`/`form_inputs`/`message_name`/`dialog_event_type`. |
| `main.py` | `handle_card_click(ev)` + `CARD_CLICKED` branch in `chat_webhook` | EXISTS (dormant) | Dispatches add/dismiss/add_all/dismiss_all/edit. **Change dispatch to read `parameters["actionName"]`.** |
| `main.py` | `_make_card_response(cards, is_addon, update, text)` | EXISTS | Builds standard + add-on (`hostAppDataAction`) envelopes. Verify the add-on `createMessageAction`/`updateMessageAction` shapes against §4 / official samples. |
| `conversation_store.py` | `store_task_batch` / `get_task_batch` / `update_task_batch` | EXISTS | batchId-keyed row state (`pending/added/already_exists/dismissed/failed/already_completed`). Reuse as-is. |
| `conversation_store.py` | `claim_message_once` / `release_message_claim` | EXISTS | Idempotency guard (Firestore create-if-absent). Reuse if you go synchronous (see §7). |
| `tasks_service.py` | `create_task(title, notes, due_date)` | EXISTS | Atomic; dedup identity = **title + due** (`_task_identity_match` + `_normalize_due`, handles ISO/RFC3339/"%b %d, %Y"). Returns status `created`/`already_exists`/`already_completed` or `{"error":...}`. |
| `briefing.py` | `_process_debrief_tasks` | TEXT (do not touch) | Debrief renders a plain-text "📋 Suggested follow-ups" list. KEEP. |
| `main.py` | `handle_message` / `_process_message_background` | TEXT/ASYNC (reverted) | Currently: empty ack + background processing; ALL queued actions → text "reply yes" approval. You will re-route `create` to a synchronous card. |

What was deliberately removed in the revert: `_process_message_sync`, `_build_conversational_create_card`, `_extract_chat_to_kg`, `import asyncio`. You will re-introduce a synchronous path (see §7).

Genuine fixes to KEEP (do not regress): `tasks_service` title+due dedup; the "yes-dead-end" fix (bare "yes" answering a clarifying question falls through to the agent when the last assistant turn ends with "?"); latency caps (`claude_client` TIER_TIMEOUTS wired + `max_retries=1`, `config.MCP_DEFAULT_TIMEOUT=25`, `agent.py` tool timeouts 20s).

---

## 6. Hard constraints (violating these = regression)

- **Async cards can't be interactive.** Do NOT try to make debrief/cron cards clickable. Debrief stays text.
- **Conversational create card must be returned SYNCHRONOUSLY** as the HTTP response to the MESSAGE (not `send_chat_message(cards=)`).
- **Button `function` = full `/chat` URL**, action in `parameters.actionName`. (The URL is the deployed Cloud Run URL; read it from config / the existing `MOMO_SERVICE_URL` env or hardcode-via-config — do NOT hardcode a stale URL. Prefer `config.MOMO_SERVICE_URL + "/chat"`.)
- **Preserve the conversational-CRUD text flow** for update/complete/delete and the pending-reply branch (`_parse_pending_task_reply`, `_apply_pending_task_actions_background`, `store_pending_task_actions_if_empty`, `_append_task_approval_block`, `_user_task_scope`). Only `create` becomes a card.
- **Preserve** voice/audio async path, clear/reset + briefing commands, the yes-dead-end fall-through.

---

## 7. Implementation plan (synchronous create-card on the add-on framework)

1. **`cards.py` `build_task_tray_card`**: change every button `onClick.action.function` from the bare name to `config.MOMO_SERVICE_URL + "/chat"`, and add `{"key":"actionName","value":"task_add"|"task_dismiss"|"task_edit"|...}` to `parameters` (alongside `batchId`/`taskId`). Keep row-state rendering.
2. **`handle_card_click`**: dispatch on `ev["parameters"].get("actionName")` (not `invoked_function`). Keep the existing add/dismiss/add_all/dismiss_all logic. Return via `_make_card_response(tray, is_addon=True, update=True)` → must produce the `hostAppDataAction.chatDataAction.updateMessageAction` shape (§4c). Verify against the avatar/contact-form samples.
3. **Synchronous message path** for creates: when the agent yields `create` action(s) for a TEXT message, process in-request and RETURN the card (§4a) as the HTTP response (use `hostAppDataAction.chatDataAction.createMessageAction`). Run the blocking agent loop off the event loop (`await asyncio.to_thread(...)`) — handle_message is `async`. Build a batch (`store_task_batch`) + tray, return it synchronously. NON-create actions and plain replies keep the existing async/text path.
   - **30s deadline + retry**: synchronous processing risks exceeding Chat's 30s limit → Chat retries → double-processing. REUSE the existing `claim_message_once(ev["message_name"])` idempotency guard at the top of the synchronous path (release on exception). The latency caps are already in place. (This was Oracle-mandated last time — keep it.)
4. **Debrief**: untouched (text).

---

## 8. Verification (MANDATORY — the live click is the only real proof)

- **TDD**: write failing tests first. Test the card builder emits full-URL `function` + `parameters.actionName`; test `handle_card_click` dispatches on `actionName`; test the synchronous create returns a `hostAppDataAction…createMessageAction` with the tray; test non-create still text; test debrief still text.
- **Test isolation gotcha (IMPORTANT):** these tests stub modules via `sys.modules[...] = MagicMock()`. For any test that imports `main`, copy the proven pattern from `test_handle_card_click.py` lines 1–97: an **identity-decorator `fastapi` stub** (so the real async `chat_webhook` survives import) + `sys.modules.pop("main", None)` immediately before `import main`. Without this, co-running test files cross-pollute and the round-trip test fails spuriously.
- **Full feature suite** must stay green and **order-independent** (run forward AND reversed). Pre-existing baseline: `test_task_approval_safety.py` has exactly **3 pre-existing failures** (`handle_message() missing background_tasks` — a test-harness signature drift, NOT yours); do not "fix" them, just confirm no NEW failures. Note: the full repo suite has ~18 pre-existing `ModuleNotFoundError` collection errors (anthropic/httpx not installed locally) in KG/eval files — unrelated; run the feature test files explicitly.
- **`py_compile`** changed files.
- **Oracle review** (read-only) of the synchronous-path change before deploy — last time Oracle caught a real double-processing defect. Do this.
- **Deploy**: `cd ~/Documents/momo && export PROJECT_ID=$(grep '^GCP_PROJECT_ID=' .env | cut -d= -f2) && export GEMINI_API_KEY=$(grep '^GEMINI_API_KEY=' .env | cut -d= -f2) && ./deploy.sh` (Cloud Build takes >120s — run with a long timeout or check `gcloud run revisions list` after). Confirm new revision + `/health` 200 + `/chat` 200.
- **LIVE CLICK TEST (the real acceptance — only the user can do this):** ask Karim to type "add a task to test cards", wait for the card, click **✅ Add**, and report. Then check logs:
  `gcloud logging read 'resource.type="cloud_run_revision" AND resource.labels.service_name="momo" AND textPayload:"POST /chat"' --project=operations-api-455512 --freshness=10m --order=desc`
  **SUCCESS = a SECOND `POST /chat` appears for the click** (the `CARD_CLICKED`/`buttonClickedPayload`) and the card flips to "✅ Added". If no second POST appears, the add-on still isn't delivering clicks → stop, report, and momo stays on text. Synthetic `curl` of a CARD_CLICKED only proves the handler doesn't crash; it does NOT prove Chat delivers real clicks.

---

## 9. Definition of done

- A real user click on a conversational create card produces a `POST /chat` (verified in logs) and the card updates in place to "✅ Added".
- Non-create conversational actions still use text approval; debrief still text; voice, pending-reply, yes-dead-end all intact.
- Full feature suite green + order-independent; baseline unchanged (3 pre-existing fails); py_compile clean; Oracle approved.
- Deployed; Notion tracker updated per `AGENTS.md`.
- If the live click still fails after the full-URL + add-on-envelope fix: that is a definitive negative result — revert the create path to text and document that interactive cards are not feasible for momo without a deeper add-on re-architecture.

---

## 10. Project facts (for the fresh agent)
- Stack: Python/FastAPI/Gemini+Claude/Google Workspace/Firestore on Cloud Run. GCP project `operations-api-455512`, service `momo`, region `us-central1`, URL `https://momo-ia4bhvubwa-uc.a.run.app`.
- Config loads from `.env` (dotenv). Deploy via `deploy.sh` (`gcloud run deploy --source .`, `--no-cpu-throttling`, no git commit needed).
- `AGENTS.md`: update the Notion tracker via `python3 scripts/update_notion_tracker.py` on start/finish (descriptions must be < 2000 chars).
- Full prior design doc for the card UX: `PLAN_card_task_ux.md` (note: its §10 add-on-envelope concern is exactly what this handoff resolves; its async-card assumptions are superseded by the §6 hard limit here).
