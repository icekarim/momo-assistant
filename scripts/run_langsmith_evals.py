"""Run LangSmith evaluations against the momo-prod-traces dataset.

Scores every example with three LLM-as-judge evaluators:
  1. Hallucination Check — did Momo fabricate data?
  2. Tool Efficiency     — were the right tools called?
  3. Response Quality    — accuracy, tone, formatting, completeness

Usage:
    python scripts/run_langsmith_evals.py                    # eval all examples
    python scripts/run_langsmith_evals.py --limit 20         # eval latest 20
    python scripts/run_langsmith_evals.py --prefix v2-prompt # custom experiment name
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

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

client = Client()

DATASET_NAME = "momo-prod-traces"
JUDGE_MODEL = "gemini-2.0-flash"


# ── Evaluator prompts ────────────────────────────────────────────

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

TOOL_EFFICIENCY_PROMPT = """\
You are evaluating the tool usage of an AI agent called Momo. Momo has these tools:
- get_todays_calendar, get_calendar_for_date
- get_open_tasks, create_task, update_task, complete_task, delete_task
- get_recent_emails, search_emails
- search_knowledge_graph
- get_meeting_notes (Granola)
- get_jira_tickets, get_jira_issue, search_jira_tickets

Given the user's message and Momo's response, evaluate whether the agent likely used tools efficiently. Consider what data would be needed to produce the response.

<trace>
User message: {input}

Momo's response: {output}
</trace>

Score on a scale of 1-5:
- 5: Perfect — response shows exactly the right data was fetched, nothing unnecessary
- 4: Good — minor inefficiency but got the job done well
- 3: Acceptable — got the answer but likely fetched too much or missed something
- 2: Poor — response suggests wrong tools were used or critical data is missing
- 1: Bad — response clearly lacks data that should have been fetched, or is completely off

Respond with ONLY a number from 1 to 5.
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

def _judge(prompt: str, input_text: str, output_text: str) -> str:
    """Call Gemini to judge a trace."""
    filled = prompt.format(input=input_text, output=output_text)
    model = genai.GenerativeModel(model_name=JUDGE_MODEL)
    resp = model.generate_content(filled)
    return resp.text.strip()


def _extract_input(inputs: dict) -> str:
    """Extract a readable input string from the dataset example."""
    if "user_message" in inputs:
        return inputs["user_message"]
    if "message" in inputs:
        return inputs["message"]
    # Fall back to stringifying the whole input
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


# ── Evaluator functions for client.evaluate() ────────────────────

def hallucination_check(inputs: dict, outputs: dict, **kwargs) -> dict:
    """Score whether the response hallucinates data."""
    input_text = _extract_input(inputs)
    output_text = _extract_output(outputs)
    result = _judge(HALLUCINATION_PROMPT, input_text, output_text)
    is_hallucination = "hallucination" in result.lower() and "no hallucination" not in result.lower()
    return {
        "key": "hallucination",
        "score": 0 if is_hallucination else 1,
        "comment": result,
    }


def tool_efficiency(inputs: dict, outputs: dict, **kwargs) -> dict:
    """Score tool usage efficiency 1-5."""
    input_text = _extract_input(inputs)
    output_text = _extract_output(outputs)
    result = _judge(TOOL_EFFICIENCY_PROMPT, input_text, output_text)
    match = re.search(r"[1-5]", result)
    score = int(match.group()) if match else 3
    return {
        "key": "tool_efficiency",
        "score": score / 5.0,  # normalize to 0-1 for LangSmith
        "comment": f"Score: {score}/5",
    }


def response_quality(inputs: dict, outputs: dict, **kwargs) -> dict:
    """Score response quality 1-5."""
    input_text = _extract_input(inputs)
    output_text = _extract_output(outputs)
    result = _judge(RESPONSE_QUALITY_PROMPT, input_text, output_text)
    match = re.search(r"[1-5]", result)
    score = int(match.group()) if match else 3
    return {
        "key": "response_quality",
        "score": score / 5.0,  # normalize to 0-1 for LangSmith
        "comment": f"Score: {score}/5",
    }


# ── Main ─────────────────────────────────────────────────────────

def run_evals(prefix: str = "momo-eval", limit: int | None = None):
    """Run all evaluators against the dataset."""

    # Verify dataset exists
    try:
        dataset = client.read_dataset(dataset_name=DATASET_NAME)
    except Exception:
        print(f"Dataset '{DATASET_NAME}' not found.")
        print("Send some messages to Momo first — the automation will populate the dataset.")
        raise SystemExit("dataset not found or empty")

    example_count = client.list_examples(dataset_id=dataset.id)
    examples = list(example_count)
    total = len(examples)

    if total == 0:
        print(f"Dataset '{DATASET_NAME}' is empty. Send some messages to Momo first.")
        raise SystemExit("dataset not found or empty")

    eval_count = min(limit, total) if limit else total
    print(f"Dataset: {DATASET_NAME} ({total} examples, evaluating {eval_count})")
    print(f"Judge model: {JUDGE_MODEL}")
    print(f"Experiment prefix: {prefix}")
    print(f"Evaluators: hallucination, tool_efficiency, response_quality")
    print()

    # Define the target function — just returns the existing outputs
    # (we're evaluating existing data, not re-running the agent)
    def passthrough(inputs: dict) -> dict:
        # Find the matching example to return its stored output
        for ex in examples:
            if ex.inputs == inputs:
                return ex.outputs or {}
        return {}

    results = client.evaluate(
        passthrough,
        data=DATASET_NAME,
        evaluators=[hallucination_check, tool_efficiency, response_quality],
        experiment_prefix=prefix,
        max_concurrency=4,
    )

    # Collect results before they're consumed by the iterator
    collected_results = list(results)

    print("\nDone! View results at: https://smith.langchain.com")
    print(f"Look for experiment: {prefix}")

    # Print summary
    print("\n── Summary ─────────────────────────────────")
    hallucination_scores = []
    efficiency_scores = []
    quality_scores = []

    for result in collected_results:
        eval_results = getattr(result, "evaluation_results", None)
        if eval_results is None:
            continue
        result_list = getattr(eval_results, "results", None) or []
        for ev_result in result_list:
            key = getattr(ev_result, "key", "")
            score = getattr(ev_result, "score", None)
            if score is None:
                continue
            if key == "hallucination":
                hallucination_scores.append(score)
            elif key == "tool_efficiency":
                efficiency_scores.append(score * 5)
            elif key == "response_quality":
                quality_scores.append(score * 5)

    if hallucination_scores:
        clean = sum(1 for s in hallucination_scores if s == 1)
        print(f"  Hallucination:    {clean}/{len(hallucination_scores)} clean ({clean/len(hallucination_scores)*100:.0f}%)")
    if efficiency_scores:
        avg = sum(efficiency_scores) / len(efficiency_scores)
        print(f"  Tool Efficiency:  {avg:.1f}/5 avg")
    if quality_scores:
        avg = sum(quality_scores) / len(quality_scores)
        print(f"  Response Quality: {avg:.1f}/5 avg")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LangSmith evals on Momo traces")
    parser.add_argument("--prefix", default="momo-eval", help="Experiment name prefix")
    parser.add_argument("--limit", type=int, default=None, help="Max examples to evaluate")
    args = parser.parse_args()
    run_evals(prefix=args.prefix, limit=args.limit)
