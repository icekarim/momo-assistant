"""Run Langfuse experiments against the momo-eval-golden dataset (v4 SDK).

The task runs the LIVE agent for every dataset item (the old LangSmith harness
defaulted to a stored-output passthrough, so the nightly job never actually
exercised the agent — fixed here: live agent is the ONLY mode).

Evaluators (LLM-as-judge on Gemini Flash + deterministic trajectory checks):
  1. correctness            — binary PASS/FAIL against correctness criteria
  2. trajectory metrics     — step_ratio, tool_call_ratio, required/forbidden tools
  3. hallucination          — did Momo fabricate data?
  4. response_quality       — 1-5 scaled to 0-1

Evaluators read the task output dict directly ({"response", "trajectory"}) —
they never query the just-created trace, because ingestion is async.

Usage:
    python scripts/run_langfuse_evals.py                       # all golden examples
    python scripts/run_langfuse_evals.py --category calendar   # one category
    python scripts/run_langfuse_evals.py --limit 5             # first N items
    python scripts/run_langfuse_evals.py --prefix v2-prompt    # experiment name prefix
"""

import argparse
import os
import re
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()

import google.generativeai as genai

import config

genai.configure(api_key=config.GEMINI_API_KEY)

GOLDEN_DATASET = "momo-eval-golden"
JUDGE_MODEL = config.GEMINI_MODEL_FLASH


# ── Evaluator prompts ────────────────────────────────────────────

CORRECTNESS_PROMPT = """\
You are evaluating an AI assistant called Momo. Given the user's message, Momo's response, and the correctness criteria below, determine if the response PASSES or FAILS.

<trace>
User message: {input}

Momo's response: {output}
</trace>

<criteria>
{criteria}
</criteria>

A response PASSES if it meets ALL the criteria. It FAILS if it violates ANY criterion.

Respond with ONLY one of: "PASS" or "FAIL" followed by a brief reason on the next line.
"""

HALLUCINATION_PROMPT = """\
You are evaluating an AI assistant called Momo that has access to tools (Gmail, Calendar, Tasks, Knowledge Graph, Jira).

Momo MUST only reference data that came from tool calls. It must NEVER fabricate emails, meetings, tasks, people, or events.

Given the trace below, check whether the final response contains any hallucinated information — data that was NOT present in the tool call results or the user's message.

<trace>
Input: {input}

Output: {output}
</trace>

Scoring:
- Return "no hallucination" if every claim in the output can be traced back to tool results, the user's message, or is generic commentary.
- Return "hallucination" if the output references specific emails, meetings, tasks, people, dates, or decisions that do not appear in the inputs.

Respond with ONLY one of: "no hallucination" or "hallucination"
"""

RESPONSE_QUALITY_PROMPT = """\
You are evaluating the response quality of an AI assistant called Momo that lives in Google Chat. Momo should be casual, helpful, accurate, and scannable.

<trace>
User message: {input}

Momo's response: {output}
</trace>

Evaluate the response on these criteria:
1. Accuracy — does it answer what was asked? Does it avoid making things up?
2. Formatting — is it scannable? Does it use section headers and priority colors for multi-topic responses?
3. Tone — is it casual and natural without being unprofessional?
4. Completeness — did it address all parts of the user's request?

Score on a scale of 1-5:
- 5: Excellent across all criteria
- 4: Good with minor issues in one area
- 3: Acceptable but noticeable issues
- 2: Poor — significant problems with accuracy, formatting, or completeness
- 1: Bad — wrong answer, bad tone, or missed the point entirely

Respond with ONLY a number from 1 to 5.
"""


# ── Judge + extraction helpers ───────────────────────────────────

def _judge(prompt: str, **kwargs) -> str:
    """Call Gemini to judge a trace."""
    filled = prompt.format(**kwargs)
    model = genai.GenerativeModel(model_name=JUDGE_MODEL)
    resp = model.generate_content(filled)
    return resp.text.strip()


def _extract_input(inputs) -> str:
    if isinstance(inputs, dict):
        return inputs.get("user_message") or inputs.get("message") or str(inputs)
    return str(inputs)


def _extract_output(output) -> str:
    if output is None:
        return "(no output)"
    if isinstance(output, dict):
        for key in ("response", "output", "text"):
            if key in output:
                return str(output[key])
    return str(output)


# ── Evaluators (langfuse v4 signature) ───────────────────────────
# Each returns Evaluation(name=..., value=..., comment=...) or a list of them.
# They read the task output directly — never the just-created trace
# (ingestion is async).

def correctness_check(*, input, output, expected_output, metadata=None, **kwargs):
    """Binary correctness: does the response satisfy the correctness criteria?"""
    from langfuse import Evaluation
    criteria = (expected_output or {}).get("correctness_criteria", "")
    if not criteria:
        return Evaluation(name="correctness", value=1.0, comment="no criteria defined")
    result = _judge(CORRECTNESS_PROMPT, input=_extract_input(input),
                    output=_extract_output(output), criteria=criteria)
    passed = result.strip().upper().startswith("PASS")
    return Evaluation(name="correctness", value=1.0 if passed else 0.0, comment=result)


def trajectory_metrics(*, input, output, expected_output, metadata=None, **kwargs):
    """Trajectory efficiency vs the example's ideal trajectory (deterministic).

    Reads the actual trajectory from the task output ({"trajectory": ...}) —
    the run_agent_loop metrics_sink — not from trace metadata."""
    from langfuse import Evaluation
    ideal = (expected_output or {}).get("ideal_trajectory", {})
    if not ideal:
        return Evaluation(name="trajectory", value=1.0,
                          comment="no ideal trajectory defined — skipped")

    trajectory = (output or {}).get("trajectory", {}) if isinstance(output, dict) else {}
    actual_steps = trajectory.get("iteration_count", 0)
    actual_tool_count = trajectory.get("total_tool_calls", 0)
    actual_tools = set(trajectory.get("tool_sequence", []) or [])

    ideal_steps = ideal.get("ideal_step_count", 1)
    ideal_tool_count = ideal.get("ideal_tool_count", 1)
    required_tools = set(ideal.get("required_tools", []))
    forbidden_tools = set(ideal.get("forbidden_tools", []))

    results = []

    if ideal_steps > 0 and actual_steps > 0:
        step_ratio = actual_steps / ideal_steps
        results.append(Evaluation(
            name="step_ratio",
            value=min(1.0, 1.0 / step_ratio) if step_ratio > 0 else 0.0,
            comment=f"actual={actual_steps} ideal={ideal_steps} ratio={step_ratio:.2f}",
        ))

    if ideal_tool_count > 0 and actual_tool_count > 0:
        tool_ratio = actual_tool_count / ideal_tool_count
        results.append(Evaluation(
            name="tool_call_ratio",
            value=min(1.0, 1.0 / tool_ratio) if tool_ratio > 0 else 0.0,
            comment=f"actual={actual_tool_count} ideal={ideal_tool_count} ratio={tool_ratio:.2f}",
        ))

    if required_tools:
        missing = required_tools - actual_tools
        results.append(Evaluation(
            name="required_tools_hit",
            value=1.0 if not missing else 0.0,
            comment=f"missing={sorted(missing)}" if missing else "all required tools called",
        ))

    if forbidden_tools:
        violated = forbidden_tools & actual_tools
        results.append(Evaluation(
            name="forbidden_tools_clean",
            value=1.0 if not violated else 0.0,
            comment=f"violated={sorted(violated)}" if violated else "no forbidden tools called",
        ))

    if not results:
        return Evaluation(name="trajectory", value=1.0, comment="no trajectory data — skipped")
    return results


def hallucination_check(*, input, output, expected_output=None, metadata=None, **kwargs):
    """Score whether the response hallucinates data."""
    from langfuse import Evaluation
    result = _judge(HALLUCINATION_PROMPT, input=_extract_input(input),
                    output=_extract_output(output))
    is_hallucination = ("hallucination" in result.lower()
                        and "no hallucination" not in result.lower())
    return Evaluation(name="hallucination", value=0.0 if is_hallucination else 1.0,
                      comment=result)


def response_quality(*, input, output, expected_output=None, metadata=None, **kwargs):
    """Score response quality 1-5 (stored scaled to 0-1)."""
    from langfuse import Evaluation
    result = _judge(RESPONSE_QUALITY_PROMPT, input=_extract_input(input),
                    output=_extract_output(output))
    match = re.search(r"[1-5]", result)
    score = int(match.group()) if match else 3
    return Evaluation(name="response_quality", value=score / 5.0,
                      comment=f"Score: {score}/5")


EVALUATORS = [correctness_check, trajectory_metrics, hallucination_check, response_quality]


# ── Task: the LIVE agent ─────────────────────────────────────────

def live_agent_task(*, item, **kwargs):
    """Run the live agent loop for one dataset item.

    Returns {"response", "trajectory"} — trajectory comes from the loop's
    metrics_sink so evaluators never have to query the (async-ingested) trace."""
    from agent import run_agent_loop

    raw_input = getattr(item, "input", None) or {}
    user_message = _extract_input(raw_input)
    metrics: dict = {}
    response, _pending = run_agent_loop(user_message, [], metrics_sink=metrics)
    return {
        "response": response,
        "trajectory": {
            "iteration_count": metrics.get("iteration_count", 0),
            "total_tool_calls": metrics.get("total_tool_calls", 0),
            "tool_sequence": metrics.get("tool_sequence", []),
        },
    }


# ── Main ─────────────────────────────────────────────────────────

def _print_summary(result):
    """Aggregate per-evaluator scores from the experiment result."""
    scores: dict[str, list] = {}
    for item_result in getattr(result, "item_results", []) or []:
        for ev in getattr(item_result, "evaluations", []) or []:
            name = getattr(ev, "name", "")
            value = getattr(ev, "value", None)
            if isinstance(value, (int, float)):
                scores.setdefault(name, []).append(float(value))

    print("\n── Summary ─────────────────────────────────")
    for key in ("correctness", "hallucination", "step_ratio", "tool_call_ratio",
                "required_tools_hit", "forbidden_tools_clean"):
        vals = scores.get(key)
        if vals:
            print(f"  {key:<22} {sum(vals)/len(vals):.2f} avg over {len(vals)}")
    if scores.get("response_quality"):
        vals = scores["response_quality"]
        print(f"  {'response_quality':<22} {sum(vals)*5/len(vals):.1f}/5 avg")
    if scores.get("correctness"):
        vals = scores["correctness"]
        print(f"\n  Solve Rate:            {sum(vals)/len(vals)*100:.0f}%")


def run_evals(prefix: str = "momo-eval", limit: int | None = None,
              category: str | None = None):
    """Run the live agent + all evaluators against the golden dataset.

    Keeps the old harness's (prefix, limit) call signature so the /run-evals
    endpoint is a drop-in."""
    from langfuse import get_client

    langfuse = get_client()

    try:
        dataset = langfuse.get_dataset(GOLDEN_DATASET)
    except Exception as exc:
        print(f"Dataset '{GOLDEN_DATASET}' not found ({exc}).")
        print("Run: python scripts/seed_langfuse_dataset.py")
        raise SystemExit(1)

    items = list(dataset.items)
    if category:
        items = [i for i in items
                 if (getattr(i, "metadata", None) or {}).get("category") == category]
    if limit:
        items = items[:limit]

    if not items:
        print(f"No examples found in '{GOLDEN_DATASET}'"
              + (f" for category '{category}'" if category else "") + ".")
        raise SystemExit(1)

    name = prefix if prefix.startswith("momo-eval") else f"momo-eval-{prefix}"
    print(f"Dataset:    {GOLDEN_DATASET} ({len(items)} examples)")
    if category:
        print(f"Category:   {category}")
    print(f"Judge:      {JUDGE_MODEL}")
    print(f"Experiment: {name}")
    print("Target:     LIVE agent (run_agent_loop)")
    print()

    if category or limit:
        # Subset run: hand the filtered items to the client-level runner.
        result = langfuse.run_experiment(
            name=name, data=items, task=live_agent_task,
            evaluators=EVALUATORS, max_concurrency=4,
        )
    else:
        result = dataset.run_experiment(
            name=name, task=live_agent_task,
            evaluators=EVALUATORS, max_concurrency=4,
        )

    print(result.format())
    _print_summary(result)
    langfuse.flush()
    return {"experiment": name, "items": len(items)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Langfuse evals on Momo (live agent)")
    parser.add_argument("--prefix", default=f"momo-eval-{datetime.now().strftime('%Y%m%d')}",
                        help="Experiment name (prefix)")
    parser.add_argument("--limit", type=int, default=None, help="Max examples to evaluate")
    parser.add_argument("--category", default=None,
                        help="Filter by category (calendar, retrieval, tool_use, memory, conversation, multi_tool)")
    args = parser.parse_args()
    try:
        run_evals(prefix=args.prefix, limit=args.limit, category=args.category)
    finally:
        from langfuse import get_client
        get_client().shutdown()
