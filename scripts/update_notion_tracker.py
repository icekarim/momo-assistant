#!/usr/bin/env python3
"""
Update the Momo Notion project tracker.

Usage:
  # Add a new task
  python scripts/update_notion_tracker.py add "Task name" --status "To Do" --priority "High" --component "Gmail" --type "Feature" --effort "Small"

  # Update a task's status
  python scripts/update_notion_tracker.py update "Task name" --status "Done"

  # List tasks by status
  python scripts/update_notion_tracker.py list --status "In Progress"
"""

import argparse
import json
import os
import sys
import urllib.request
import urllib.error
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

NOTION_TOKEN = os.environ.get("NOTION_API_KEY", "")
DATABASE_ID = "***REMOVED-NOTION-DB-ID***"
NOTION_VERSION = "2022-06-28"

if not NOTION_TOKEN:
    print("Error: NOTION_API_KEY not found. Set it in .env or as an environment variable.", file=sys.stderr)
    sys.exit(1)

VALID_STATUSES = ["Backlog", "To Do", "In Progress", "Done", "Blocked"]
VALID_PRIORITIES = ["Critical", "High", "Medium", "Low"]
VALID_COMPONENTS = ["Briefing", "Gmail", "Calendar", "Tasks", "Knowledge Graph", "Granola", "Gemini", "Chat", "Proactive Intelligence", "Infrastructure"]
VALID_TYPES = ["Feature", "Bug Fix", "Performance", "Refactor", "DevOps"]
VALID_EFFORTS = ["Small", "Medium", "Large"]


def notion_request(method, endpoint, data=None):
    url = f"https://api.notion.com/v1/{endpoint}"
    headers = {
        "Authorization": f"Bearer {NOTION_TOKEN}",
        "Notion-Version": NOTION_VERSION,
        "Content-Type": "application/json",
    }
    body = json.dumps(data).encode() if data else None
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        error_body = e.read().decode()
        print(f"Error {e.code}: {error_body}", file=sys.stderr)
        sys.exit(1)


def find_task(name):
    """Find a task by name (case-insensitive substring match)."""
    data = notion_request("POST", f"databases/{DATABASE_ID}/query", {
        "filter": {
            "property": "Task",
            "title": {"contains": name}
        }
    })
    results = data.get("results", [])
    if not results:
        return None
    # Prefer exact match
    for r in results:
        title = r["properties"]["Task"]["title"]
        if title and title[0]["plain_text"].lower() == name.lower():
            return r
    return results[0]


def add_page_content(page_id, text):
    """Add a description as content blocks inside a Notion page."""
    blocks = []
    for paragraph in text.split("\n\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if paragraph.startswith("# "):
            blocks.append({"object": "block", "type": "heading_2", "heading_2": {
                "rich_text": [{"type": "text", "text": {"content": paragraph[2:]}}]
            }})
        elif paragraph.startswith("- "):
            for line in paragraph.split("\n"):
                line = line.strip().lstrip("- ")
                if line:
                    blocks.append({"object": "block", "type": "bulleted_list_item",
                        "bulleted_list_item": {"rich_text": [{"type": "text", "text": {"content": line}}]}
                    })
        else:
            blocks.append({"object": "block", "type": "paragraph", "paragraph": {
                "rich_text": [{"type": "text", "text": {"content": paragraph}}]
            }})
    if blocks:
        notion_request("PATCH", f"blocks/{page_id}/children", {"children": blocks})


def add_task(args):
    props = {
        "Task": {"title": [{"text": {"content": args.name}}]},
        "Status": {"select": {"name": args.status or "To Do"}},
    }
    if args.priority:
        props["Priority"] = {"select": {"name": args.priority}}
    if args.component:
        props["Component"] = {"select": {"name": args.component}}
    if args.type:
        props["Type"] = {"select": {"name": args.type}}
    if args.effort:
        props["Effort"] = {"select": {"name": args.effort}}

    result = notion_request("POST", "pages", {
        "parent": {"database_id": DATABASE_ID},
        "properties": props,
    })

    if result and hasattr(args, "description") and args.description:
        add_page_content(result["id"], args.description)

    print(f"Created: {args.name} [{args.status or 'To Do'}]")
    return result


def update_task(args):
    task = find_task(args.name)
    if not task:
        print(f"Task not found: {args.name}", file=sys.stderr)
        sys.exit(1)

    props = {}
    if args.status:
        props["Status"] = {"select": {"name": args.status}}
    if args.priority:
        props["Priority"] = {"select": {"name": args.priority}}
    if args.component:
        props["Component"] = {"select": {"name": args.component}}
    if args.type:
        props["Type"] = {"select": {"name": args.type}}
    if args.effort:
        props["Effort"] = {"select": {"name": args.effort}}

    if not props:
        print("Nothing to update", file=sys.stderr)
        sys.exit(1)

    result = notion_request("PATCH", f"pages/{task['id']}", {"properties": props})
    title = task["properties"]["Task"]["title"][0]["plain_text"]
    print(f"Updated: {title} -> {args.status or 'no status change'}")
    return result


def list_tasks(args):
    query = {}
    if args.status:
        query["filter"] = {"property": "Status", "select": {"equals": args.status}}

    data = notion_request("POST", f"databases/{DATABASE_ID}/query", query)
    for r in data.get("results", []):
        title = r["properties"]["Task"]["title"]
        status = r["properties"]["Status"]["select"]
        priority = r["properties"]["Priority"]["select"]
        component = r["properties"]["Component"]["select"]
        name = title[0]["plain_text"] if title else "Untitled"
        st = status["name"] if status else "-"
        pri = priority["name"] if priority else "-"
        comp = component["name"] if component else "-"
        print(f"  [{st}] {name} ({pri}, {comp})")


def main():
    parser = argparse.ArgumentParser(description="Manage Momo Notion project tracker")
    sub = parser.add_subparsers(dest="command")

    # Add
    add_p = sub.add_parser("add", help="Add a new task")
    add_p.add_argument("name", help="Task name")
    add_p.add_argument("--status", default="To Do", choices=VALID_STATUSES)
    add_p.add_argument("--priority", choices=VALID_PRIORITIES)
    add_p.add_argument("--component", choices=VALID_COMPONENTS)
    add_p.add_argument("--type", choices=VALID_TYPES)
    add_p.add_argument("--effort", choices=VALID_EFFORTS)
    add_p.add_argument("--description", help="Description to add inside the task page")

    # Update
    upd_p = sub.add_parser("update", help="Update a task")
    upd_p.add_argument("name", help="Task name (substring match)")
    upd_p.add_argument("--status", choices=VALID_STATUSES)
    upd_p.add_argument("--priority", choices=VALID_PRIORITIES)
    upd_p.add_argument("--component", choices=VALID_COMPONENTS)
    upd_p.add_argument("--type", choices=VALID_TYPES)
    upd_p.add_argument("--effort", choices=VALID_EFFORTS)

    # List
    list_p = sub.add_parser("list", help="List tasks")
    list_p.add_argument("--status", choices=VALID_STATUSES)

    args = parser.parse_args()
    if args.command == "add":
        add_task(args)
    elif args.command == "update":
        update_task(args)
    elif args.command == "list":
        list_tasks(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
