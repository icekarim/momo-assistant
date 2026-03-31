"""Run LangSmith evaluations against Momo datasets.

Supports two datasets:
  - momo-eval-golden: curated examples with ideal trajectories (default)
  - momo-prod-traces: production trace samples

Evaluators:
  1. Correctness       — binary PASS/FAIL against correctness criteria
  2. Trajectory Metrics — step_ratio, tool_call_ratio, required/forbidden tools
  3. Hallucination     — did Momo fabricate data?
  4. Response Quality   — accuracy, tone, formatting, completeness

Usage:
    python scripts/run_langsmith_evals.py                           # golden dataset, all
    python scripts/run_langsmith_evals.py --category calendar       # only calendar evals
    python scripts/run_langsmith_evals.py --dataset prod --limit 20 # prod traces
    python scripts/run_langsmith_evals.py --prefix v2-prompt        # custom experiment name
"""

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()

import google.generativeai as genai
from langsmith import Client
from langsmith.schemas import Run, Example

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

client = Client()

GOLDEN_DATASET = "momo-eval-golden"
PROD_DATASET = "momo-prod-traces"
JUDGE_MODEL = "gemini-2.0-flash"


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


# ── Judge function ───────────────────────────────────────────────

def _judge(prompt: str, **kwargs) -> str:
    """Call Gemini to judge a trace."""
    filled = prompt.format(**kwargs)
    model = genai.GenerativeModel(model_name=JUDGE_MODEL)
    resp = model.generate_content(filled)
    return resp.text.strip()


def _extract_input(inputs: dict) -> str:
    """Extract a readable input string from the dataset example."""
    if "user_message" in inputs:
        return inputs["user_message"]
    if "message" in inputs:
        return inputs["message"]
    return str(inputs)


def _extract_output(outputs: dict) -> str:
    """Extract a readable output string from the dataset example."""
    if outputs is None:
        return "(no output)"
    if "output" in outputs:
        return str(outputs["output"])
    if "response" in outputs:
        return str(outputs["response"])
    if "text" in outputs:
        return str(outputs["text"])
    return str(outputs)


# ── Evaluators ───────────────────────────────────────────────────

def correctness_check(run: Run, example: Example) -> dict:
    """Binary correctness: does the response satisfy the correctness criteria?"""
    criteria = (example.outputs or {}).get("correctness_criteria", "")
    if not criteria:
        return {"key": "correctness", "score": 1, "comment": "no criteria defined"}

    input_text = _extract_input(run.inputs or {})
    output_text = _extract_output(run.outputs or {})
    result = _judge(CORRECTNESS_PROMPT, input=input_text, output=output_text, criteria=criteria)
    passed = result.strip().upper().startswith("PASS")
    return {
        "key": "correctness",
        "score": 1 if passed else 0,
        "comment": result,
    }


def trajectory_metrics(run: Run, example: Example) -> list[dict]:
    """Compute trajectory efficiency metrics from trace metadata and example ideal trajectory."""
    ideal = (example.outputs or {}).get("ideal_trajectory", {})
    if not ideal:
        return {"key": "trajectory", "score": True, "comment": "no ideal trajectory defined — skipped"}

    # Extract actual metrics from run metadata (set by Phase 1 instrumentation)
    metadata = {}
    if run.extra:
        metadata = run.extra.get("metadata", {})

    actual_steps = metadata.get("iteration_count", 0)
    actual_tool_count = metadata.get("total_tool_calls", 0)
    actual_tools = set(metadata.get("tool_sequence", []))

    ideal_steps = ideal.get("ideal_step_count", 1)
    ideal_tool_count = ideal.get("ideal_tool_count", 1)
    required_tools = set(ideal.get("required_tools", []))
    forbidden_tools = set(ideal.get("forbidden_tools", []))

    results = []

    # step_ratio: 1.0 = perfect, <1.0 means took more steps than ideal
    if ideal_steps > 0 and actual_steps > 0:
        step_ratio = actual_steps / ideal_steps
        results.append({
            "key": "step_ratio",
            "score": min(1.0, 1.0 / step_ratio) if step_ratio > 0 else 0,
            "comment": f"actual={actual_steps} ideal={ideal_steps} ratio={step_ratio:.2f}",
        })

    # tool_call_ratio: 1.0 = perfect, <1.0 means used more tools than ideal
    if ideal_tool_count > 0 and actual_tool_count > 0:
        tool_ratio = actual_tool_count / ideal_tool_count
        results.append({
            "key": "tool_call_ratio",
            "score": min(1.0, 1.0 / tool_ratio) if tool_ratio > 0 else 0,
            "comment": f"actual={actual_tool_count} ideal={ideal_tool_count} ratio={tool_ratio:.2f}",
        })

    # required_tools_hit: did the agent call all required tools?
    if required_tools:
        missing = required_tools - actual_tools
        results.append({
            "key": "required_tools_hit",
            "score": 1.0 if not missing else 0.0,
            "comment": f"missing={sorted(missing)}" if missing else "all required tools called",
        })

    # forbidden_tools_clean: did the agent avoid forbidden tools?
    if forbidden_tools:
        violated = forbidden_tools & actual_tools
        results.append({
            "key": "forbidden_tools_clean",
            "score": 1.0 if not violated else 0.0,
            "comment": f"violated={sorted(violated)}" if violated else "no forbidden tools called",
        })

    if not results:
        return {"key": "trajectory", "score": True, "comment": "no trajectory data — skipped"}
    return results


def hallucination_check(run: Run, example: Example) -> dict:
    """Score whether the response hallucinates data."""
    input_text = _extract_input(run.inputs or {})
    output_text = _extract_output(run.outputs or {})
    result = _judge(HALLUCINATION_PROMPT, input=input_text, output=output_text)
    is_hallucination = "hallucination" in result.lower() and "no hallucination" not in result.lower()
    return {
        "key": "hallucination",
        "score": 0 if is_hallucination else 1,
        "comment": result,
    }


def response_quality(run: Run, example: Example) -> dict:
    """Score response quality 1-5."""
    input_text = _extract_input(run.inputs or {})
    output_text = _extract_output(run.outputs or {})
    result = _judge(RESPONSE_QUALITY_PROMPT, input=input_text, output=output_text)
    match = re.search(r"[1-5]", result)
    score = int(match.group()) if match else 3
    return {
        "key": "response_quality",
        "score": score / 5.0,
        "comment": f"Score: {score}/5",
    }


# ── Main ─────────────────────────────────────────────────────────

def run_evals(prefix: str = "momo-eval", limit: int | None = None,
              dataset: str = "golden", category: str | None = None):
    """Run all evaluators against the dataset."""

    dataset_name = GOLDEN_DATASET if dataset == "golden" else PROD_DATASET

    # Verify dataset exists
    try:
        ds = client.read_dataset(dataset_name=dataset_name)
    except Exception:
        print(f"Dataset '{dataset_name}' not found.")
        if dataset == "golden":
            print("Run: python scripts/seed_eval_dataset.py")
        else:
            print("Send some messages to Momo first — the automation will populate the dataset.")
        raise SystemExit(1)

    # Fetch examples, optionally filtered by category split
    if category:
        examples = list(client.list_examples(dataset_id=ds.id, splits=[category]))
    else:
        examples = list(client.list_examples(dataset_id=ds.id))

    if limit:
        examples = examples[:limit]
    total = len(examples)

    if total == 0:
        print(f"No examples found in '{dataset_name}'"
              + (f" for category '{category}'" if category else "") + ".")
        raise SystemExit(1)

    # Determine which evaluators to run based on dataset type
    evaluators = [hallucination_check, response_quality]
    evaluator_names = ["hallucination", "response_quality"]

    if dataset == "golden":
        evaluators = [correctness_check, trajectory_metrics, hallucination_check, response_quality]
        evaluator_names = ["correctness", "trajectory", "hallucination", "response_quality"]

    print(f"Dataset:    {dataset_name} ({total} examples)")
    if category:
        print(f"Category:   {category}")
    print(f"Judge:      {JUDGE_MODEL}")
    print(f"Prefix:     {prefix}")
    print(f"Evaluators: {', '.join(evaluator_names)}")
    print()

    # Passthrough target — returns existing outputs (we're evaluating stored data)
    def passthrough(inputs: dict) -> dict:
        for ex in examples:
            if ex.inputs == inputs:
                return ex.outputs or {}
        return {}

    results = client.evaluate(
        passthrough,
        data=dataset_name if not category else examples,
        evaluators=evaluators,
        experiment_prefix=prefix,
        max_concurrency=4,
    )

    collected_results = list(results)

    print(f"\nDone! View results at: https://smith.langchain.com")
    print(f"Experiment: {prefix}")

    # ── Summary ──────────────────────────────────────────────
    print("\n── Summary ─────────────────────────────────")

    scores = {
        "correctness": [],
        "hallucination": [],
        "response_quality": [],
        "step_ratio": [],
        "tool_call_ratio": [],
        "required_tools_hit": [],
        "forbidden_tools_clean": [],
    }

    for result in collected_results:
        eval_results = result.get("evaluation_results") if isinstance(result, dict) else getattr(result, "evaluation_results", None)
        if eval_results is None:
            continue
        result_list = eval_results.get("results") if isinstance(eval_results, dict) else getattr(eval_results, "results", None)
        for ev_result in (result_list or []):
            key = ev_result.key if hasattr(ev_result, "key") else ev_result.get("key", "")
            score = ev_result.score if hasattr(ev_result, "score") else ev_result.get("score")
            if score is not None and key in scores:
                scores[key].append(score)

    # Correctness (binary)
    if scores["correctness"]:
        passed = sum(1 for s in scores["correctness"] if s == 1)
        total_c = len(scores["correctness"])
        print(f"  Correctness:         {passed}/{total_c} pass ({passed/total_c*100:.0f}%)")

    # Hallucination (binary)
    if scores["hallucination"]:
        clean = sum(1 for s in scores["hallucination"] if s == 1)
        total_h = len(scores["hallucination"])
        print(f"  Hallucination:       {clean}/{total_h} clean ({clean/total_h*100:.0f}%)")

    # Trajectory metrics
    if scores["step_ratio"]:
        avg = sum(scores["step_ratio"]) / len(scores["step_ratio"])
        print(f"  Step Ratio:          {avg:.2f} avg (1.0 = ideal)")

    if scores["tool_call_ratio"]:
        avg = sum(scores["tool_call_ratio"]) / len(scores["tool_call_ratio"])
        print(f"  Tool Call Ratio:     {avg:.2f} avg (1.0 = ideal)")

    if scores["required_tools_hit"]:
        hit = sum(1 for s in scores["required_tools_hit"] if s == 1)
        total_r = len(scores["required_tools_hit"])
        print(f"  Required Tools Hit:  {hit}/{total_r} ({hit/total_r*100:.0f}%)")

    if scores["forbidden_tools_clean"]:
        clean = sum(1 for s in scores["forbidden_tools_clean"] if s == 1)
        total_f = len(scores["forbidden_tools_clean"])
        print(f"  Forbidden Avoided:   {clean}/{total_f} ({clean/total_f*100:.0f}%)")

    # Response quality (1-5)
    if scores["response_quality"]:
        avg = sum(scores["response_quality"]) * 5 / len(scores["response_quality"])
        print(f"  Response Quality:    {avg:.1f}/5 avg")

    # Solve rate: correctness-weighted across all examples
    if scores["correctness"]:
        solve_rate = sum(scores["correctness"]) / len(scores["correctness"])
        print(f"\n  Solve Rate:          {solve_rate*100:.0f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LangSmith evals on Momo")
    parser.add_argument("--prefix", default="momo-eval", help="Experiment name prefix")
    parser.add_argument("--limit", type=int, default=None, help="Max examples to evaluate")
    parser.add_argument("--dataset", choices=["golden", "prod"], default="golden",
                        help="Dataset to evaluate: golden (curated) or prod (production traces)")
    parser.add_argument("--category", default=None,
                        help="Filter by category split (calendar, retrieval, tool_use, memory, conversation, multi_tool)")
    args = parser.parse_args()
    run_evals(prefix=args.prefix, limit=args.limit, dataset=args.dataset, category=args.category)
