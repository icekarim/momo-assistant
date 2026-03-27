"""Promote production failures from Firestore to the momo-eval-golden dataset.

Reads eval_failures documents with status "pending_review" from Firestore
and promotes them to the golden evaluation dataset as regression test cases.

Usage:
    python scripts/promote_failures_to_evals.py            # interactive review
    python scripts/promote_failures_to_evals.py --auto      # auto-promote all pending
    python scripts/promote_failures_to_evals.py --discard   # discard all pending
"""

import argparse
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()

from google.cloud import firestore
from langsmith import Client

import config

DATASET_NAME = "momo-eval-golden"
COLLECTION = "eval_failures"


def _get_firestore_client():
    return firestore.Client(
        project=config.GCP_PROJECT_ID,
        database=config.FIRESTORE_DATABASE,
    )


def _get_langsmith_client():
    return Client()


def _fetch_pending(db):
    """Fetch all pending_review failures from Firestore."""
    docs = db.collection(COLLECTION).where("status", "==", "pending_review").stream()
    return [(doc.id, doc.to_dict()) for doc in docs]


def _promote_to_dataset(ls_client, failure: dict):
    """Add a failure as a regression eval in the golden dataset."""
    existing = list(ls_client.list_datasets(dataset_name=DATASET_NAME))
    if not existing:
        print(f"Dataset '{DATASET_NAME}' not found. Run seed_eval_dataset.py first.")
        raise SystemExit(1)

    dataset = existing[0]
    category = failure.get("category", "regression")

    ls_client.create_example(
        dataset_id=dataset.id,
        inputs={
            "user_message": failure["user_message"],
            "category": category,
        },
        outputs={
            "ideal_trajectory": {
                "tool_sequence": [],
                "ideal_step_count": 1,
                "ideal_tool_count": 1,
                "required_tools": [],
                "forbidden_tools": [],
            },
            "correctness_criteria": failure.get("expected_behavior", ""),
        },
        metadata={
            "category": category,
            "source": "regression",
            "difficulty": "medium",
            "added_date": datetime.now().strftime("%Y-%m-%d"),
            "bug_description": failure.get("actual_behavior", ""),
            "trace_url": failure.get("trace_url", ""),
        },
        split=category if category != "regression" else "regression",
    )


def _update_status(db, doc_id: str, new_status: str):
    """Update a failure document's status in Firestore."""
    db.collection(COLLECTION).document(doc_id).update({
        "status": new_status,
        "reviewed_at": firestore.SERVER_TIMESTAMP,
    })


def auto_promote():
    """Auto-promote all pending failures to the eval dataset."""
    db = _get_firestore_client()
    ls = _get_langsmith_client()
    pending = _fetch_pending(db)

    if not pending:
        print("No pending failures to promote.")
        return {"promoted": 0, "total": 0}

    promoted = 0
    for doc_id, failure in pending:
        try:
            _promote_to_dataset(ls, failure)
            _update_status(db, doc_id, "promoted")
            promoted += 1
            print(f"  Promoted: {failure.get('user_message', '')[:60]}...")
        except Exception as e:
            print(f"  Failed to promote {doc_id}: {e}")

    print(f"\nPromoted {promoted}/{len(pending)} failures to '{DATASET_NAME}'.")
    return {"promoted": promoted, "total": len(pending)}


def interactive_review():
    """Interactively review each pending failure."""
    db = _get_firestore_client()
    ls = _get_langsmith_client()
    pending = _fetch_pending(db)

    if not pending:
        print("No pending failures to review.")
        return

    print(f"Found {len(pending)} pending failure(s) to review.\n")

    promoted = 0
    discarded = 0
    skipped = 0

    for i, (doc_id, failure) in enumerate(pending, 1):
        print(f"── Failure {i}/{len(pending)} ──────────────────────────")
        print(f"  Message:  {failure.get('user_message', 'N/A')}")
        print(f"  Expected: {failure.get('expected_behavior', 'N/A')}")
        print(f"  Actual:   {failure.get('actual_behavior', 'N/A')}")
        print(f"  Category: {failure.get('category', 'N/A')}")
        if failure.get("trace_url"):
            print(f"  Trace:    {failure['trace_url']}")
        print()

        while True:
            choice = input("  [p]romote / [d]iscard / [s]kip? ").strip().lower()
            if choice in ("p", "promote"):
                try:
                    _promote_to_dataset(ls, failure)
                    _update_status(db, doc_id, "promoted")
                    promoted += 1
                    print("  -> Promoted to eval dataset.\n")
                except Exception as e:
                    print(f"  -> Error: {e}\n")
                break
            elif choice in ("d", "discard"):
                _update_status(db, doc_id, "discarded")
                discarded += 1
                print("  -> Discarded.\n")
                break
            elif choice in ("s", "skip"):
                skipped += 1
                print("  -> Skipped (still pending).\n")
                break
            else:
                print("  Invalid choice. Enter p, d, or s.")

    print(f"\nDone: {promoted} promoted, {discarded} discarded, {skipped} skipped.")


def discard_all():
    """Discard all pending failures."""
    db = _get_firestore_client()
    pending = _fetch_pending(db)

    if not pending:
        print("No pending failures to discard.")
        return

    for doc_id, _ in pending:
        _update_status(db, doc_id, "discarded")

    print(f"Discarded {len(pending)} pending failure(s).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Review and promote eval failures")
    parser.add_argument("--auto", action="store_true", help="Auto-promote all pending failures")
    parser.add_argument("--discard", action="store_true", help="Discard all pending failures")
    args = parser.parse_args()

    if args.auto:
        auto_promote()
    elif args.discard:
        discard_all()
    else:
        interactive_review()
