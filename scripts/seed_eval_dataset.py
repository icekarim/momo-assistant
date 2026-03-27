"""Seed the momo-eval-golden dataset in LangSmith.

Creates a curated evaluation dataset with ideal trajectories for each example.
Examples are organized by behavior category (calendar, retrieval, tool_use,
memory, conversation, multi_tool) and include expected tool sequences, step
counts, and correctness criteria.

Usage:
    python scripts/seed_eval_dataset.py           # create if doesn't exist
    python scripts/seed_eval_dataset.py --force    # recreate from scratch
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()

from langsmith import Client

client = Client()

DATASET_NAME = "momo-eval-golden"

# ── Golden examples ─────────────────────────────────────────────
# Each example defines inputs, expected outputs with ideal trajectory,
# and metadata for categorization.

GOLDEN_EXAMPLES = [
    # ── Calendar (3) ────────────────────────────────────────────
    {
        "inputs": {"user_message": "What's on my calendar today?", "category": "calendar"},
        "outputs": {
            "ideal_trajectory": {
                "tool_sequence": ["get_todays_calendar"],
                "ideal_step_count": 1,
                "ideal_tool_count": 1,
                "required_tools": ["get_todays_calendar"],
                "forbidden_tools": ["get_calendar_for_date"],
            },
            "correctness_criteria": "Must list today's meetings with times. Must NOT fabricate any meetings.",
        },
        "metadata": {"category": "calendar", "source": "artisanal", "difficulty": "easy"},
        "split": "calendar",
    },
    {
        "inputs": {"user_message": "What meetings do I have next Monday?", "category": "calendar"},
        "outputs": {
            "ideal_trajectory": {
                "tool_sequence": ["get_calendar_for_date"],
                "ideal_step_count": 1,
                "ideal_tool_count": 1,
                "required_tools": ["get_calendar_for_date"],
                "forbidden_tools": ["get_todays_calendar"],
            },
            "correctness_criteria": "Must call get_calendar_for_date with a Monday date. Must NOT call get_todays_calendar.",
        },
        "metadata": {"category": "calendar", "source": "artisanal", "difficulty": "easy"},
        "split": "calendar",
    },
    {
        "inputs": {"user_message": "Am I free tomorrow afternoon?", "category": "calendar"},
        "outputs": {
            "ideal_trajectory": {
                "tool_sequence": ["get_calendar_for_date"],
                "ideal_step_count": 1,
                "ideal_tool_count": 1,
                "required_tools": ["get_calendar_for_date"],
                "forbidden_tools": ["get_todays_calendar"],
            },
            "correctness_criteria": "Must check tomorrow's calendar and indicate afternoon availability. Must use tomorrow's date, not today's.",
        },
        "metadata": {"category": "calendar", "source": "artisanal", "difficulty": "easy"},
        "split": "calendar",
    },

    # ── Retrieval (3) ───────────────────────────────────────────
    {
        "inputs": {"user_message": "What did Sarah email me about the project last week?", "category": "retrieval"},
        "outputs": {
            "ideal_trajectory": {
                "tool_sequence": ["search_emails"],
                "ideal_step_count": 1,
                "ideal_tool_count": 1,
                "required_tools": ["search_emails"],
                "forbidden_tools": ["get_recent_emails"],
            },
            "correctness_criteria": "Must use search_emails with a query about Sarah. Must reference only actual email content from the tool result.",
        },
        "metadata": {"category": "retrieval", "source": "artisanal", "difficulty": "easy"},
        "split": "retrieval",
    },
    {
        "inputs": {"user_message": "Show me my recent unread emails", "category": "retrieval"},
        "outputs": {
            "ideal_trajectory": {
                "tool_sequence": ["get_recent_emails"],
                "ideal_step_count": 1,
                "ideal_tool_count": 1,
                "required_tools": ["get_recent_emails"],
                "forbidden_tools": ["search_emails"],
            },
            "correctness_criteria": "Must call get_recent_emails and list actual emails. Must NOT fabricate senders or subjects.",
        },
        "metadata": {"category": "retrieval", "source": "artisanal", "difficulty": "easy"},
        "split": "retrieval",
    },
    {
        "inputs": {"user_message": "Find any emails from the engineering team about the deployment", "category": "retrieval"},
        "outputs": {
            "ideal_trajectory": {
                "tool_sequence": ["search_emails"],
                "ideal_step_count": 1,
                "ideal_tool_count": 1,
                "required_tools": ["search_emails"],
                "forbidden_tools": [],
            },
            "correctness_criteria": "Must search for emails about deployment. Must reference only data from the tool result.",
        },
        "metadata": {"category": "retrieval", "source": "artisanal", "difficulty": "medium"},
        "split": "retrieval",
    },

    # ── Tool Use / Task CRUD (3) ────────────────────────────────
    {
        "inputs": {"user_message": "Create a task to review the Q1 report by Friday", "category": "tool_use"},
        "outputs": {
            "ideal_trajectory": {
                "tool_sequence": ["get_open_tasks", "create_task"],
                "ideal_step_count": 2,
                "ideal_tool_count": 2,
                "required_tools": ["get_open_tasks", "create_task"],
                "forbidden_tools": [],
            },
            "correctness_criteria": "Must check for duplicates via get_open_tasks first, then call create_task. Must mention pending approval.",
        },
        "metadata": {"category": "tool_use", "source": "artisanal", "difficulty": "medium"},
        "split": "tool_use",
    },
    {
        "inputs": {"user_message": "Mark the 'update docs' task as done", "category": "tool_use"},
        "outputs": {
            "ideal_trajectory": {
                "tool_sequence": ["complete_task"],
                "ideal_step_count": 1,
                "ideal_tool_count": 1,
                "required_tools": ["complete_task"],
                "forbidden_tools": ["delete_task"],
            },
            "correctness_criteria": "Must call complete_task with find='update docs'. Must mention pending approval.",
        },
        "metadata": {"category": "tool_use", "source": "artisanal", "difficulty": "easy"},
        "split": "tool_use",
    },
    {
        "inputs": {"user_message": "What tasks do I have open?", "category": "tool_use"},
        "outputs": {
            "ideal_trajectory": {
                "tool_sequence": ["get_open_tasks"],
                "ideal_step_count": 1,
                "ideal_tool_count": 1,
                "required_tools": ["get_open_tasks"],
                "forbidden_tools": ["create_task", "complete_task", "delete_task", "update_task"],
            },
            "correctness_criteria": "Must list open tasks from the tool result. Must NOT fabricate task names.",
        },
        "metadata": {"category": "tool_use", "source": "artisanal", "difficulty": "easy"},
        "split": "tool_use",
    },

    # ── Memory / Knowledge Graph (2) ───────────────────────────
    {
        "inputs": {"user_message": "What did we decide about the pricing change last month?", "category": "memory"},
        "outputs": {
            "ideal_trajectory": {
                "tool_sequence": ["search_knowledge_graph"],
                "ideal_step_count": 1,
                "ideal_tool_count": 1,
                "required_tools": ["search_knowledge_graph"],
                "forbidden_tools": [],
            },
            "correctness_criteria": "Must query the knowledge graph about pricing decisions. Must only reference data from the tool result.",
        },
        "metadata": {"category": "memory", "source": "artisanal", "difficulty": "medium"},
        "split": "memory",
    },
    {
        "inputs": {"user_message": "Who committed to delivering the API spec?", "category": "memory"},
        "outputs": {
            "ideal_trajectory": {
                "tool_sequence": ["search_knowledge_graph"],
                "ideal_step_count": 1,
                "ideal_tool_count": 1,
                "required_tools": ["search_knowledge_graph"],
                "forbidden_tools": [],
            },
            "correctness_criteria": "Must search knowledge graph for commitment about API spec. Must attribute the commitment to the correct person from tool results only.",
        },
        "metadata": {"category": "memory", "source": "artisanal", "difficulty": "medium"},
        "split": "memory",
    },

    # ── Conversation / No Tools (2) ─────────────────────────────
    {
        "inputs": {"user_message": "Hey what's up?", "category": "conversation"},
        "outputs": {
            "ideal_trajectory": {
                "tool_sequence": [],
                "ideal_step_count": 0,
                "ideal_tool_count": 0,
                "required_tools": [],
                "forbidden_tools": [],
            },
            "correctness_criteria": "Must respond conversationally without calling any tools. Casual greeting only.",
        },
        "metadata": {"category": "conversation", "source": "artisanal", "difficulty": "easy"},
        "split": "conversation",
    },
    {
        "inputs": {"user_message": "Thanks for the help!", "category": "conversation"},
        "outputs": {
            "ideal_trajectory": {
                "tool_sequence": [],
                "ideal_step_count": 0,
                "ideal_tool_count": 0,
                "required_tools": [],
                "forbidden_tools": [],
            },
            "correctness_criteria": "Must respond casually without calling any tools. Should not fetch data for a simple thank-you.",
        },
        "metadata": {"category": "conversation", "source": "artisanal", "difficulty": "easy"},
        "split": "conversation",
    },

    # ── Multi-Tool (3) ──────────────────────────────────────────
    {
        "inputs": {"user_message": "Give me a quick rundown of today — meetings, tasks, and emails", "category": "multi_tool"},
        "outputs": {
            "ideal_trajectory": {
                "tool_sequence": ["get_todays_calendar", "get_open_tasks", "get_recent_emails"],
                "ideal_step_count": 1,
                "ideal_tool_count": 3,
                "required_tools": ["get_todays_calendar", "get_open_tasks", "get_recent_emails"],
                "forbidden_tools": [],
            },
            "correctness_criteria": "Must call all three tools (calendar, tasks, emails) and present a consolidated summary. Must NOT fabricate data.",
        },
        "metadata": {"category": "multi_tool", "source": "artisanal", "difficulty": "medium"},
        "split": "multi_tool",
    },
    {
        "inputs": {"user_message": "What's the context on ProjectX? Check emails and meeting notes", "category": "multi_tool"},
        "outputs": {
            "ideal_trajectory": {
                "tool_sequence": ["search_emails", "search_knowledge_graph"],
                "ideal_step_count": 1,
                "ideal_tool_count": 2,
                "required_tools": ["search_emails", "search_knowledge_graph"],
                "forbidden_tools": [],
            },
            "correctness_criteria": "Must search both emails and knowledge graph for ProjectX. Must synthesize findings from both sources without fabrication.",
        },
        "metadata": {"category": "multi_tool", "source": "artisanal", "difficulty": "medium"},
        "split": "multi_tool",
    },
    {
        "inputs": {
            "user_message": "Create a task to follow up on the pricing discussion from last week's meeting, and find any related emails",
            "category": "multi_tool",
        },
        "outputs": {
            "ideal_trajectory": {
                "tool_sequence": ["search_knowledge_graph", "search_emails", "get_open_tasks", "create_task"],
                "ideal_step_count": 3,
                "ideal_tool_count": 4,
                "required_tools": ["search_knowledge_graph", "get_open_tasks", "create_task"],
                "forbidden_tools": [],
            },
            "correctness_criteria": "Must search for pricing context, check for duplicate tasks, create a task, and find related emails. Must mention pending approval for the task.",
        },
        "metadata": {"category": "multi_tool", "source": "artisanal", "difficulty": "hard"},
        "split": "multi_tool",
    },
]


def seed_dataset(force: bool = False):
    """Create the golden eval dataset in LangSmith."""

    # Check if dataset already exists
    existing = list(client.list_datasets(dataset_name=DATASET_NAME))
    if existing and not force:
        print(f"Dataset '{DATASET_NAME}' already exists ({len(list(client.list_examples(dataset_id=existing[0].id)))} examples).")
        print("Use --force to delete and recreate.")
        return

    if existing and force:
        print(f"Deleting existing dataset '{DATASET_NAME}'...")
        client.delete_dataset(dataset_name=DATASET_NAME)

    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description="Curated golden evaluation dataset for Momo agent with ideal trajectories and correctness criteria.",
    )

    # Group examples by split for batch creation
    for example in GOLDEN_EXAMPLES:
        split = example.pop("split", None)
        client.create_example(
            dataset_id=dataset.id,
            inputs=example["inputs"],
            outputs=example["outputs"],
            metadata=example["metadata"],
            split=split,
        )

    print(f"\nCreated dataset '{DATASET_NAME}' with {len(GOLDEN_EXAMPLES)} examples.")
    print(f"\nCategory breakdown:")

    from collections import Counter
    cats = Counter(ex["metadata"]["category"] for ex in GOLDEN_EXAMPLES)
    for cat, count in sorted(cats.items()):
        print(f"  {cat}: {count}")

    print(f"\nView at: https://smith.langchain.com")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed the momo-eval-golden dataset")
    parser.add_argument("--force", action="store_true", help="Delete and recreate the dataset")
    args = parser.parse_args()
    seed_dataset(force=args.force)
