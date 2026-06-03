from enum import Enum

import anthropic
from langsmith.wrappers import wrap_anthropic

import config


class TaskComplexity(Enum):
    LIGHT = "light"
    STANDARD = "standard"
    DEEP = "deep"


TASK_MODEL_MAP = {
    TaskComplexity.LIGHT: config.CLAUDE_MODEL_HAIKU,
    TaskComplexity.STANDARD: config.CLAUDE_MODEL_SONNET,
    TaskComplexity.DEEP: config.CLAUDE_MODEL_OPUS,
}

TASK_MAX_TOKENS = {
    TaskComplexity.LIGHT: config.CLAUDE_MAX_TOKENS_LIGHT,
    TaskComplexity.STANDARD: config.CLAUDE_MAX_TOKENS_STANDARD,
    TaskComplexity.DEEP: config.CLAUDE_MAX_TOKENS_DEEP,
}

TIER_TIMEOUTS = {
    TaskComplexity.LIGHT: 30,
    TaskComplexity.STANDARD: 60,
    TaskComplexity.DEEP: 120,
}

_DEEP_FALLBACK = TaskComplexity.STANDARD
_MAX_FALLBACK_ATTEMPTS = 1

_client = wrap_anthropic(
    anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY, max_retries=2)
)


def get_client():
    return _client


def extract_text(message) -> str:
    return "".join(
        b.text for b in message.content if getattr(b, "type", None) == "text"
    ).strip()


def _is_downshiftable(exc: Exception) -> bool:
    # Transient/capacity failures justify a one-step tier downshift.
    # Auth/bad-request (4xx except 429) must NOT downshift — a cheaper
    # model won't fix a malformed request and only doubles spend.
    if isinstance(exc, (anthropic.RateLimitError, anthropic.APITimeoutError,
                        anthropic.InternalServerError, anthropic.APIConnectionError)):
        return True
    if isinstance(exc, anthropic.APIStatusError):
        return exc.status_code in (429, 500, 502, 503, 529)
    return False


def generate(prompt=None, *, tier=TaskComplexity.STANDARD, system=None,
             tools=None, messages=None, max_tokens=None, model=None,
             allow_fallback=True):
    if messages is None:
        if prompt is None:
            raise ValueError("generate requires either prompt or messages")
        messages = [{"role": "user", "content": prompt}]

    resolved_model = model or TASK_MODEL_MAP[tier]
    resolved_max = max_tokens or TASK_MAX_TOKENS[tier]
    kwargs = {"model": resolved_model, "max_tokens": resolved_max, "messages": messages}
    if system is not None:
        kwargs["system"] = system
    if tools is not None:
        kwargs["tools"] = tools

    try:
        return _client.messages.create(**kwargs)
    except Exception as exc:
        if (allow_fallback and tier == TaskComplexity.DEEP
                and _is_downshiftable(exc)):
            return _fallback(messages, system, tools, max_tokens)
        raise


def _fallback(messages, system, tools, max_tokens):
    attempts = 0
    tier = _DEEP_FALLBACK
    last_exc = None
    while attempts < _MAX_FALLBACK_ATTEMPTS:
        attempts += 1
        kwargs = {
            "model": TASK_MODEL_MAP[tier],
            "max_tokens": max_tokens or TASK_MAX_TOKENS[tier],
            "messages": messages,
        }
        if system is not None:
            kwargs["system"] = system
        if tools is not None:
            kwargs["tools"] = tools
        try:
            return _client.messages.create(**kwargs)
        except Exception as exc:
            last_exc = exc
            if not _is_downshiftable(exc):
                raise
    raise last_exc
