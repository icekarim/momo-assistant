import json
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


def extract_json(text):
    if not text:
        return None
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t[3:]
    if t.endswith("```"):
        t = t[: t.rfind("```")]
    t = t.strip()
    try:
        parsed = json.loads(t)
    except Exception:
        # Tolerate prose-prefixed output: slice from the first JSON bracket.
        start = min((i for i in (t.find("["), t.find("{")) if i != -1), default=-1)
        if start == -1:
            print(f"claude_client.extract_json: no JSON found in output")
            return None
        try:
            parsed = json.loads(t[start:])
        except Exception as exc:
            print(f"claude_client.extract_json: parse failed — {exc}")
            return None
    if isinstance(parsed, dict):
        parsed = [parsed]
    return parsed


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


def gemini_tool_to_claude(decl: dict) -> dict:
    schema = decl.get("parameters") or {"type": "object", "properties": {}}
    return {
        "name": decl["name"],
        "description": decl.get("description", ""),
        "input_schema": schema,
    }


def run_tool_loop(*, messages, tools, system, dispatch, max_iterations=6,
                  tier=TaskComplexity.STANDARD, on_tool=None):
    """Drive a Claude tool-use conversation to a final text answer.

    dispatch(name, input_dict) -> str. Returns (final_text, stop_reason).
    Handles parallel tool_use blocks, malformed input, tool errors, and
    max_tokens truncation mid-loop without silently dropping it.
    """
    convo = list(messages)
    last_stop = None
    for _ in range(max_iterations):
        msg = generate(messages=convo, tools=tools, system=system, tier=tier)
        last_stop = msg.stop_reason

        if msg.stop_reason == "max_tokens":
            text = extract_text(msg)
            return (text or "[response truncated: max_tokens reached]"), "max_tokens"

        tool_uses = [b for b in msg.content if getattr(b, "type", None) == "tool_use"]
        if not tool_uses:
            return extract_text(msg), msg.stop_reason

        convo.append({"role": "assistant", "content": msg.content})
        results = []
        for tu in tool_uses:
            name = getattr(tu, "name", None)
            tool_input = getattr(tu, "input", None)
            is_error = False
            if not name or not isinstance(tool_input, dict):
                result_str = f"Tool call malformed (name={name!r})"
                is_error = True
            else:
                try:
                    result_str = dispatch(name, tool_input)
                    if isinstance(result_str, str) and result_str.startswith(("Tool '", "Error calling", "Unknown tool")):
                        is_error = True
                except Exception as exc:
                    result_str = f"Tool '{name}' failed: {exc}"
                    is_error = True
            if on_tool:
                on_tool(name, result_str, is_error)
            results.append({
                "type": "tool_result",
                "tool_use_id": tu.id,
                "content": str(result_str),
                "is_error": is_error,
            })
        convo.append({"role": "user", "content": results})

    return "", last_stop


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


_RERANK_SYSTEM = (
    "You are a search reranker. Given a user query and a numbered list of candidate "
    "results, return the candidate numbers ordered from MOST to LEAST relevant to the "
    "query. Only include candidates that are genuinely relevant; drop irrelevant ones. "
    "Respond with ONLY a JSON array of integers, e.g. [3,1,7]. No prose."
)


def rerank(query: str, candidates: list[str], top_k=None):
    """Rerank candidate strings by relevance to query using Claude (Haiku).

    Returns a list of 0-based indices into `candidates`, ordered most->least
    relevant. On any failure returns the original order (graceful fallback),
    so callers can always rely on getting a usable ordering.
    """
    if not candidates:
        return []
    n = len(candidates)
    numbered = "\n".join(f"{i}: {c[:500]}" for i, c in enumerate(candidates))
    prompt = f"Query: {query}\n\nCandidates:\n{numbered}"
    try:
        msg = generate(prompt=prompt, tier=TaskComplexity.LIGHT, system=_RERANK_SYSTEM)
        order = extract_json(extract_text(msg))
        if not isinstance(order, list):
            return list(range(n))
        seen = set()
        cleaned = []
        for x in order:
            if isinstance(x, int) and 0 <= x < n and x not in seen:
                cleaned.append(x)
                seen.add(x)
        if not cleaned:
            return list(range(n))
        if top_k is not None:
            cleaned = cleaned[:top_k]
        return cleaned
    except Exception as exc:
        print(f"claude_client.rerank failed, using original order — {exc}")
        return list(range(n))
