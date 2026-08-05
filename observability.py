"""Langfuse observability for Momo (successor to langsmith_config.py).

Configure via environment variables (read through config.py, which loads .env):
  LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY
  LANGFUSE_BASE_URL              (default https://us.cloud.langfuse.com)
  LANGFUSE_TRACING_ENABLED       (default true)
  LANGFUSE_TRACING_ENVIRONMENT   (default "production")

When keys are absent or LANGFUSE_TRACING_ENABLED=false, everything in this
module is a zero-overhead no-op: `observe` is an identity decorator,
`propagate_attributes` / `start_as_current_observation` are inert context
managers, and flush()/shutdown() do nothing. Langfuse is only imported when
tracing is actually enabled.

CRITICAL ordering: init_tracing() must run AFTER dotenv is loaded (config.py
does load_dotenv at import — this module imports config first) and BEFORE the
anthropic client object is constructed (claude_client.py calls init_tracing()
at its top, before `anthropic.Anthropic(...)`). AnthropicInstrumentor
auto-captures messages.create calls as generations (model + tokens), and
ThreadingInstrumentor propagates OTel context through ThreadPoolExecutor so
tool spans nest correctly under the agent root.

init_tracing() also runs at module import (idempotent) so `from observability
import observe` always hands out the final binding.
"""

from contextlib import contextmanager

import config

_client = None
_enabled = False
_initialized = False


def tracing_enabled() -> bool:
    return _enabled


# ── No-op fallbacks (bound by default; replaced by init_tracing) ────────────


def _noop_observe(func=None, *, name=None, as_type=None, capture_input=None,
                  capture_output=None, **kwargs):
    """Identity decorator matching langfuse.observe's calling conventions."""
    if func is not None and callable(func):
        return func

    def decorator(fn):
        return fn
    return decorator


class _NoopSpan:
    """Stands in for a Langfuse span/observation when tracing is off."""

    def update(self, **kwargs):
        return self

    def update_trace(self, **kwargs):
        return self

    def score(self, **kwargs):
        return self


@contextmanager
def _noop_context(*args, **kwargs):
    yield _NoopSpan()


# Public bindings — rebound to the real Langfuse callables by init_tracing().
observe = _noop_observe
propagate_attributes = _noop_context


def init_tracing():
    """Initialize Langfuse + OTel instrumentation exactly once (idempotent).

    Must be called before the anthropic client object is created so
    AnthropicInstrumentor can patch the SDK first.
    """
    global _client, _enabled, _initialized, observe, propagate_attributes
    if _initialized:
        return
    _initialized = True

    if not (config.LANGFUSE_TRACING_ENABLED
            and config.LANGFUSE_PUBLIC_KEY and config.LANGFUSE_SECRET_KEY):
        print("[observability] Langfuse tracing disabled — running in no-op mode")
        return

    try:
        import os
        # The v4 SDK reads these env vars at client creation.
        os.environ.setdefault("LANGFUSE_PUBLIC_KEY", config.LANGFUSE_PUBLIC_KEY)
        os.environ.setdefault("LANGFUSE_SECRET_KEY", config.LANGFUSE_SECRET_KEY)
        os.environ.setdefault("LANGFUSE_BASE_URL", config.LANGFUSE_BASE_URL)
        os.environ.setdefault("LANGFUSE_TRACING_ENVIRONMENT", config.LANGFUSE_TRACING_ENVIRONMENT)

        from langfuse import get_client
        from langfuse import observe as _lf_observe
        from langfuse import propagate_attributes as _lf_propagate

        _client = get_client()
        observe = _lf_observe
        propagate_attributes = _lf_propagate

        try:
            from opentelemetry.instrumentation.anthropic import AnthropicInstrumentor
            AnthropicInstrumentor().instrument()
        except Exception as exc:
            print(f"[observability] Anthropic instrumentation failed: {exc}")
        try:
            from opentelemetry.instrumentation.threading import ThreadingInstrumentor
            ThreadingInstrumentor().instrument()
        except Exception as exc:
            print(f"[observability] Threading instrumentation failed: {exc}")

        _enabled = True
        print(f"[observability] Langfuse tracing enabled → {config.LANGFUSE_BASE_URL} "
              f"(environment={config.LANGFUSE_TRACING_ENVIRONMENT})")
    except Exception as exc:
        print(f"[observability] Langfuse init failed ({exc}) — running in no-op mode")


@contextmanager
def start_as_current_observation(*, name: str, as_type: str = "span", input=None):
    """Open a Langfuse observation as the current OTel span (no-op safe).

    EXPLICIT INPUT ONLY (skill rule): callers must pass the user-relevant
    payload explicitly — never full function args, configs, or histories.
    """
    if _enabled and _client is not None:
        with _client.start_as_current_observation(
            name=name, as_type=as_type, input=input,
        ) as span:
            yield span
    else:
        yield _NoopSpan()


def flush():
    """Flush pending spans (call at the end of scheduled jobs). No-op safe."""
    if _enabled and _client is not None:
        try:
            _client.flush()
        except Exception as exc:
            print(f"[observability] flush failed: {exc}")


def shutdown():
    """Flush + shutdown the exporter (FastAPI lifespan teardown; Cloud Run
    gives ~10s between SIGTERM and SIGKILL). No-op safe."""
    if _enabled and _client is not None:
        try:
            _client.shutdown()
        except Exception as exc:
            print(f"[observability] shutdown failed: {exc}")


def log_eval_failure(user_message: str, expected_behavior: str,
                     actual_behavior: str, category: str = "regression"):
    """Stage a production failure in Firestore for later promotion to the eval
    dataset (scripts/promote_failures_to_evals.py → momo-eval-golden).

    Failures are stored with status "pending_review". Preserved verbatim from
    the langsmith_config era — only the trace URL now points at Langfuse.
    """
    if not _enabled:
        return
    try:
        from google.cloud import firestore as _firestore
        trace_url = None
        try:
            trace_id = _client.get_current_trace_id() if _client else None
            if trace_id:
                trace_url = f"{config.LANGFUSE_BASE_URL.rstrip('/')}/trace/{trace_id}"
        except Exception:
            pass
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
            "trace_url": trace_url,
        })
    except Exception as e:
        print(f"[observability] failed to log eval failure: {e}")


# Self-initialize at import so decorators pick up the final `observe` binding
# no matter which module imports observability first. config.py has already
# run load_dotenv by this point (imported above).
init_tracing()
