#!/usr/bin/env python3
"""
Git post-commit hook: detects which files changed and creates/updates
tasks in the Notion project tracker based on the components affected.
"""

import json
import os
import subprocess
import sys
import urllib.request
import urllib.error
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

NOTION_TOKEN = os.environ.get("NOTION_API_KEY", "")
DATABASE_ID = "***REMOVED-NOTION-DB-ID***"
NOTION_VERSION = "2022-06-28"

# Map files to components
FILE_TO_COMPONENT = {
    "briefing.py": "Briefing",
    "gmail_service.py": "Gmail",
    "calendar_service.py": "Calendar",
    "tasks_service.py": "Tasks",
    "knowledge_graph.py": "Knowledge Graph",
    "granola_service.py": "Granola",
    "granola_auth_setup.py": "Granola",
    "gemini_service.py": "Gemini",
    "chat_service.py": "Chat",
    "proactive_intelligence.py": "Proactive Intelligence",
    "conversation_store.py": "Chat",
    "main.py": "Infrastructure",
    "config.py": "Infrastructure",
    "Dockerfile": "Infrastructure",
    "deploy.sh": "Infrastructure",
    "requirements.txt": "Infrastructure",
}


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
    except urllib.error.HTTPError:
        return None


def get_commit_info():
    """Get the latest commit message and changed files."""
    msg = subprocess.check_output(
        ["git", "log", "-1", "--pretty=%s"], text=True
    ).strip()
    files = subprocess.check_output(
        ["git", "diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD"], text=True
    ).strip().split("\n")
    return msg, [f for f in files if f]


def detect_type(msg):
    """Guess task type from commit message."""
    msg_lower = msg.lower()
    if any(w in msg_lower for w in ["fix", "bug", "patch", "hotfix"]):
        return "Bug Fix"
    if any(w in msg_lower for w in ["perf", "optim", "speed", "cache", "parallel", "batch"]):
        return "Performance"
    if any(w in msg_lower for w in ["refactor", "clean", "reorganize", "rename"]):
        return "Refactor"
    if any(w in msg_lower for w in ["deploy", "ci", "cd", "docker", "infra"]):
        return "DevOps"
    return "Feature"


def detect_priority(msg, num_files):
    """Estimate priority from commit message and scope."""
    msg_lower = msg.lower()
    if any(w in msg_lower for w in ["critical", "urgent", "hotfix", "breaking"]):
        return "Critical"
    if any(w in msg_lower for w in ["fix", "bug", "security"]) or num_files >= 5:
        return "High"
    if num_files >= 3:
        return "Medium"
    return "Low"


def detect_effort(num_files):
    """Estimate effort from number of files changed."""
    if num_files >= 6:
        return "Large"
    if num_files >= 3:
        return "Medium"
    return "Small"


def find_existing_task(name):
    """Check if a task with this name already exists."""
    data = notion_request("POST", f"databases/{DATABASE_ID}/query", {
        "filter": {"property": "Task", "title": {"contains": name}}
    })
    if data and data.get("results"):
        return data["results"][0]
    return None


def create_task(name, component, task_type, priority, effort):
    """Create a task marked as Done (it was just committed)."""
    existing = find_existing_task(name)
    if existing:
        notion_request("PATCH", f"pages/{existing['id']}", {
            "properties": {"Status": {"select": {"name": "Done"}}}
        })
        print(f"  Notion: Updated '{name}' -> Done")
        return

    notion_request("POST", "pages", {
        "parent": {"database_id": DATABASE_ID},
        "properties": {
            "Task": {"title": [{"text": {"content": name}}]},
            "Status": {"select": {"name": "Done"}},
            "Component": {"select": {"name": component}},
            "Type": {"select": {"name": task_type}},
            "Priority": {"select": {"name": priority}},
            "Effort": {"select": {"name": effort}},
        }
    })
    print(f"  Notion: Created '{name}' [Done, {priority}, {effort}]")


def main():
    msg, files = get_commit_info()
    if not files:
        return

    # Detect affected components
    components = set()
    for f in files:
        basename = f.split("/")[-1]
        if basename in FILE_TO_COMPONENT:
            components.add(FILE_TO_COMPONENT[basename])

    if not components:
        return

    task_type = detect_type(msg)
    priority = detect_priority(msg, len(files))
    effort = detect_effort(len(files))
    comp_str = ", ".join(sorted(components))

    task_name = msg[:100]
    primary_component = sorted(components)[0]

    print(f"Post-commit: {task_name}")
    print(f"  Components: {comp_str}")
    create_task(task_name, primary_component, task_type, priority, effort)


if __name__ == "__main__":
    main()
