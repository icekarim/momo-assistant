"""Set up LangSmith online evaluators for Momo.

Run once to configure automated LLM-as-judge scoring on production traces.
These evaluators sample a percentage of traces and score them automatically.

Usage:
    python scripts/setup_langsmith_evaluators.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()

from langsmith import Client

client = Client()

PROJECT_NAME = os.getenv("LANGSMITH_PROJECT", "momo")

# ── Evaluator prompts ────────────────────────────────────────────

HALLUCINATION_PROMPT = """\
You are evaluating an AI assistant called Momo that has access to tools (Gmail, Calendar, Tasks, Knowledge Graph, Jira).

Momo MUST only reference data that came from tool calls. It must NEVER fabricate emails, meetings, tasks, people, or events.

Given the full trace below, check whether the final response contains any hallucinated information — data that was NOT present in the tool results.

<trace>
Input: {input}

Output: {output}
</trace>

Scoring:
- Return "no hallucination" if every claim in the output can be traced back to tool results or is generic commentary.
- Return "hallucination" if the output references specific emails, meetings, tasks, people, dates, or decisions that do not appear in the tool results.

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

Given the user's message and the tools that were called, evaluate whether the agent's tool usage was efficient and appropriate.

<trace>
User message: {input}

Output: {output}
</trace>

Score on a scale of 1-5:
- 5: Perfect — called exactly the right tools, no unnecessary calls, no missed tools
- 4: Good — minor inefficiency (e.g., one extra call) but got the job done
- 3: Acceptable — got the answer but with notable waste or a missed obvious tool
- 2: Poor — called many unnecessary tools or missed critical ones, degrading response quality
- 1: Bad — completely wrong tool selection, or failed to call tools when clearly needed

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


def setup_evaluators():
    """Create online evaluators in LangSmith."""

    evaluators = [
        {
            "name": "Hallucination Check",
            "prompt": HALLUCINATION_PROMPT,
            "scoring": "categorical",
            "categories": ["no hallucination", "hallucination"],
            "filter_tags": ["chat"],
            "sample_rate": 0.2,
        },
        {
            "name": "Tool Efficiency",
            "prompt": TOOL_EFFICIENCY_PROMPT,
            "scoring": "continuous",
            "min_score": 1,
            "max_score": 5,
            "filter_tags": ["chat"],
            "sample_rate": 0.15,
        },
        {
            "name": "Response Quality",
            "prompt": RESPONSE_QUALITY_PROMPT,
            "scoring": "continuous",
            "min_score": 1,
            "max_score": 5,
            "filter_tags": ["chat", "briefing"],
            "sample_rate": 0.15,
        },
    ]

    print(f"Setting up online evaluators for project: {PROJECT_NAME}\n")

    for ev in evaluators:
        print(f"  Creating: {ev['name']}")
        print(f"    Sample rate: {ev['sample_rate'] * 100:.0f}%")
        print(f"    Filter tags: {ev['filter_tags']}")
        print(f"    Scoring: {ev['scoring']}")

    print(f"""
Setup complete. To activate these evaluators:

1. Go to https://smith.langchain.com/o/default/projects
2. Open the '{PROJECT_NAME}' project
3. Click the 'Automations' tab → '+ New Automation'
4. For each evaluator below, create an automation:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📋 Hallucination Check
   Action: Online Evaluation
   Filter: tag = "chat" AND is_root = true
   Sampling: 20%
   Model: Claude Haiku or GPT-4o-mini (cheap + fast)
   Prompt:
{HALLUCINATION_PROMPT}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📋 Tool Efficiency
   Action: Online Evaluation
   Filter: tag = "chat" AND is_root = true
   Sampling: 15%
   Model: Claude Haiku or GPT-4o-mini
   Prompt:
{TOOL_EFFICIENCY_PROMPT}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📋 Response Quality
   Action: Online Evaluation
   Filter: is_root = true
   Sampling: 15%
   Model: Claude Haiku or GPT-4o-mini
   Prompt:
{RESPONSE_QUALITY_PROMPT}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TIP: Start with these sampling rates and increase once you're
comfortable with the cost. Each eval costs ~1 cheap LLM call.
""")


if __name__ == "__main__":
    setup_evaluators()
