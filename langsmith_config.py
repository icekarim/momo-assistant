"""LangSmith observability for Momo agents.

Configure via environment variables:
  LANGSMITH_TRACING=true
  LANGSMITH_API_KEY=lsv2_pt_...
  LANGSMITH_PROJECT=momo           (optional, defaults to "default")
  LANGSMITH_ENDPOINT=https://api.smith.langchain.com  (optional)

When LANGSMITH_TRACING is not "true" or langsmith is not installed,
all decorators and wrappers become no-ops with zero overhead.
"""

from __future__ import annotations

import os

_TRACING_ENABLED = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"

try:
    if not _TRACING_ENABLED:
        raise ImportError("tracing disabled")
    from langsmith import traceable
    from langsmith import get_current_run_tree as _get_run_tree
    print("[langsmith] tracing enabled — sending to project:",
          os.getenv("LANGSMITH_PROJECT", "default"))
except ImportError:
    # No-op decorator when langsmith is not installed or tracing is off.
    def traceable(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda fn: fn

    def _get_run_tree():
        return None


def set_trace_metadata(**kwargs):
    """Inject runtime metadata/tags into the current trace.

    Usage:
        set_trace_metadata(thread_id="spaces/abc", user_id="u123",
                           tags=["chat", "user-initiated"])
    """
    rt = _get_run_tree()
    if rt is None:
        return
    tags = kwargs.pop("tags", None)
    if tags:
        existing = rt.tags or []
        rt.tags = list(set(existing + tags))
    for key, value in kwargs.items():
        if rt.metadata is None:
            rt.metadata = {}
        rt.metadata[key] = value


def add_trace_tags(*new_tags):
    """Append tags to the current trace span."""
    rt = _get_run_tree()
    if rt is None:
        return
    existing = rt.tags or []
    rt.tags = list(set(existing + list(new_tags)))


def _get_current_trace_url() -> str | None:
    """Build a LangSmith URL for the current trace, or None."""
    rt = _get_run_tree()
    if rt is None:
        return None
    project = os.getenv("LANGSMITH_PROJECT", "default")
    return f"https://smith.langchain.com/o/default/projects/p/{project}/r/{rt.id}"


def log_eval_failure(user_message: str, expected_behavior: str,
                     actual_behavior: str, category: str = "regression"):
    """Stage a production failure in Firestore for later promotion to the eval dataset.

    Failures are stored with status "pending_review" so they can be reviewed
    and optionally promoted to the momo-eval-golden dataset via
    scripts/promote_failures_to_evals.py.
    """
    if not _TRACING_ENABLED:
        return
    try:
        from google.cloud import firestore as _firestore
        import config
        db = _firestore.Client(
            project=config.GCP_PROJECT_ID,
            database=config.FIRESTORE_DATABASE,
        )
        db.collection("eval_failures").add({
            "user_message": user_message,
            "expected_behavior": expected_behavior,
            "actual_behavior": actual_behavior,
            "category": category,
            "status": "pending_review",
            "created_at": _firestore.SERVER_TIMESTAMP,
            "trace_url": _get_current_trace_url(),
        })
    except Exception as e:
        print(f"[langsmith] failed to log eval failure: {e}")


# ── Traced Gemini wrappers ───────────────────────────────────────
# These thin wrappers let LangSmith capture LLM calls as proper
# "llm" spans with model name, inputs, outputs, and timing.


@traceable(run_type="llm", name="gemini-generate")
def traced_generate_content(model, content, *, model_name="unknown"):
    """Traced wrapper around model.generate_content()."""
    response = model.generate_content(content)
    return response


@traceable(run_type="llm", name="gemini-chat-send")
def traced_chat_send(chat, message, *, model_name="unknown", iteration=None):
    """Traced wrapper around chat.send_message()."""
    response = chat.send_message(message)
    return response
