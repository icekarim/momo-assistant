"""Add a single regression eval case to the momo-eval-golden dataset.

Use this when a production bug is identified and you want to add a targeted
regression test to prevent recurrence.

Usage:
    python scripts/add_regression_eval.py \
        --message "What's on my calendar for next Monday?" \
        --expected-tools get_calendar_for_date \
        --category calendar \
        --bug-description "Agent called get_todays_calendar instead of get_calendar_for_date"

    python scripts/add_regression_eval.py \
        --message "Create a task to review docs" \
        --expected-tools get_open_tasks create_task \
        --forbidden-tools delete_task \
        --category tool_use \
        --correctness "Must check for duplicates then create task with pending approval" \
        --steps 2
"""

import argparse
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()

from langsmith import Client

client = Client()

DATASET_NAME = "momo-eval-golden"


def add_regression(args):
    """Add a regression eval example to the golden dataset."""

    # Verify dataset exists
    existing = list(client.list_datasets(dataset_name=DATASET_NAME))
    if not existing:
        print(f"Dataset '{DATASET_NAME}' not found. Run seed_eval_dataset.py first.")
        raise SystemExit(1)

    dataset = existing[0]
    required_tools = args.expected_tools or []
    forbidden_tools = args.forbidden_tools or []
    ideal_step_count = args.steps or len(required_tools)
    ideal_tool_count = args.tools or len(required_tools)

    example = {
        "inputs": {
            "user_message": args.message,
            "category": args.category,
        },
        "outputs": {
            "ideal_trajectory": {
                "tool_sequence": required_tools,
                "ideal_step_count": ideal_step_count,
                "ideal_tool_count": ideal_tool_count,
                "required_tools": required_tools,
                "forbidden_tools": forbidden_tools,
            },
            "correctness_criteria": args.correctness or args.bug_description,
        },
    }

    metadata = {
        "category": args.category,
        "source": "regression",
        "difficulty": args.difficulty or "medium",
        "added_date": datetime.now().strftime("%Y-%m-%d"),
        "bug_description": args.bug_description,
    }

    client.create_example(
        dataset_id=dataset.id,
        inputs=example["inputs"],
        outputs=example["outputs"],
        metadata=metadata,
        split=args.category,
    )

    print(f"Added regression eval to '{DATASET_NAME}':")
    print(f"  Message:        {args.message}")
    print(f"  Category:       {args.category}")
    print(f"  Required tools: {required_tools}")
    print(f"  Forbidden tools:{forbidden_tools}")
    print(f"  Bug:            {args.bug_description}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add a regression eval to the golden dataset")
    parser.add_argument("--message", required=True, help="The user message to test")
    parser.add_argument("--expected-tools", nargs="+", default=[], help="Tools that must be called")
    parser.add_argument("--forbidden-tools", nargs="+", default=[], help="Tools that must NOT be called")
    parser.add_argument("--category", required=True,
                        choices=["calendar", "retrieval", "tool_use", "memory",
                                 "conversation", "multi_tool", "regression"],
                        help="Behavior category")
    parser.add_argument("--bug-description", required=True, help="What went wrong in production")
    parser.add_argument("--correctness", default=None, help="Correctness criteria (defaults to bug description)")
    parser.add_argument("--steps", type=int, default=None, help="Ideal step count (defaults to len(expected-tools))")
    parser.add_argument("--tools", type=int, default=None, help="Ideal tool count (defaults to len(expected-tools))")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="medium")
    args = parser.parse_args()
    add_regression(args)
