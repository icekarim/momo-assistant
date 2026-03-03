from datetime import datetime, timezone
from googleapiclient.discovery import build
from google_auth import get_credentials


def get_tasks_service():
    creds = get_credentials()
    return build("tasks", "v1", credentials=creds)


def fetch_open_tasks():
    """Fetch all incomplete tasks across all task lists."""
    svc = get_tasks_service()
    all_tasks = []

    lists_resp = svc.tasklists().list(maxResults=100).execute()
    task_lists = lists_resp.get("items", [])

    for tl in task_lists:
        tasks_resp = svc.tasks().list(
            tasklist=tl["id"],
            showCompleted=False,
            showHidden=False,
            maxResults=100,
        ).execute()

        for task in tasks_resp.get("items", []):
            if not task.get("title", "").strip():
                continue
            if task.get("status") == "completed":
                continue

            due = None
            is_overdue = False
            if task.get("due"):
                due_dt = datetime.fromisoformat(task["due"].replace("Z", "+00:00"))
                due = due_dt.strftime("%b %d, %Y")
                is_overdue = due_dt < datetime.now(timezone.utc)

            all_tasks.append({
                "id": task["id"],
                "title": task["title"],
                "notes": (task.get("notes", "") or "")[:300],
                "due": due,
                "is_overdue": is_overdue,
                "list_name": tl["title"],
                "status": task.get("status", ""),
            })

    all_tasks.sort(key=lambda t: (
        not t["is_overdue"],
        t["due"] is None,
        t["due"] or "",
    ))

    return all_tasks


def create_task(title, notes="", due_date=None, task_list_name=None):
    """Create a new task in Google Tasks.
    Args:
        title: Task title (required)
        notes: Optional notes/description
        due_date: Optional due date as 'YYYY-MM-DD' string
        task_list_name: Optional task list name (defaults to first list)
    Returns:
        dict with created task info, or error message
    """
    svc = get_tasks_service()

    # Find the target task list
    lists_resp = svc.tasklists().list(maxResults=100).execute()
    task_lists = lists_resp.get("items", [])
    if not task_lists:
        return {"error": "No task lists found"}

    target_list = task_lists[0]  # default to first list
    if task_list_name:
        for tl in task_lists:
            if tl["title"].lower() == task_list_name.lower():
                target_list = tl
                break

    body = {"title": title, "status": "needsAction"}
    if notes:
        body["notes"] = notes
    if due_date:
        body["due"] = f"{due_date}T00:00:00.000Z"

    result = svc.tasks().insert(tasklist=target_list["id"], body=body).execute()

    return {
        "id": result["id"],
        "title": result["title"],
        "list_name": target_list["title"],
        "status": "created",
    }


def update_task(task_title, new_title=None, new_notes=None, new_due=None):
    """Update an existing task found by title match.
    Returns dict with updated task info, or error message.
    """
    svc = get_tasks_service()
    task_id, tasklist_id, list_name = _find_task_by_title(svc, task_title)
    if not task_id:
        return {"error": f"Couldn't find a task matching '{task_title}'"}

    task = svc.tasks().get(tasklist=tasklist_id, task=task_id).execute()
    if new_title:
        task["title"] = new_title
    if new_notes is not None:
        task["notes"] = new_notes
    if new_due:
        task["due"] = f"{new_due}T00:00:00.000Z"

    result = svc.tasks().update(tasklist=tasklist_id, task=task_id, body=task).execute()
    return {"id": result["id"], "title": result["title"], "list_name": list_name, "status": "updated"}


def complete_task(task_title):
    """Mark a task as completed by title match."""
    svc = get_tasks_service()
    task_id, tasklist_id, list_name = _find_task_by_title(svc, task_title)
    if not task_id:
        return {"error": f"Couldn't find a task matching '{task_title}'"}

    task = svc.tasks().get(tasklist=tasklist_id, task=task_id).execute()
    task["status"] = "completed"

    result = svc.tasks().update(tasklist=tasklist_id, task=task_id, body=task).execute()
    return {"id": result["id"], "title": result["title"], "list_name": list_name, "status": "completed"}


def delete_task(task_title):
    """Delete a task by title match."""
    svc = get_tasks_service()
    task_id, tasklist_id, list_name = _find_task_by_title(svc, task_title)
    if not task_id:
        return {"error": f"Couldn't find a task matching '{task_title}'"}

    title = svc.tasks().get(tasklist=tasklist_id, task=task_id).execute().get("title", task_title)
    svc.tasks().delete(tasklist=tasklist_id, task=task_id).execute()
    return {"title": title, "list_name": list_name, "status": "deleted"}


def find_completed_task(title_query, days_back=30):
    """Check if a task matching the title was recently completed.
    Returns the task dict if found, else None."""
    svc = get_tasks_service()
    title_lower = title_query.lower().strip()

    from datetime import timedelta
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days_back)).isoformat()

    lists_resp = svc.tasklists().list(maxResults=100).execute()
    for tl in lists_resp.get("items", []):
        tasks_resp = svc.tasks().list(
            tasklist=tl["id"],
            showCompleted=True,
            showHidden=True,
            completedMin=cutoff,
            maxResults=100,
        ).execute()

        for task in tasks_resp.get("items", []):
            task_title = (task.get("title", "") or "").strip()
            if not task_title or task.get("status") != "completed":
                continue
            if title_lower in task_title.lower() or task_title.lower() in title_lower:
                return {
                    "id": task["id"],
                    "title": task_title,
                    "list_name": tl["title"],
                    "status": "completed",
                }
    return None


def _find_task_by_title(svc, title_query):
    """Find a task by fuzzy title match. Returns (task_id, tasklist_id, list_name) or (None, None, None)."""
    title_lower = title_query.lower().strip()

    lists_resp = svc.tasklists().list(maxResults=100).execute()
    task_lists = lists_resp.get("items", [])

    best_match = None
    for tl in task_lists:
        tasks_resp = svc.tasks().list(
            tasklist=tl["id"],
            showCompleted=False,
            showHidden=False,
            maxResults=100,
        ).execute()

        for task in tasks_resp.get("items", []):
            task_title = (task.get("title", "") or "").strip()
            if not task_title:
                continue
            if task.get("status") == "completed":
                continue
            if task_title.lower() == title_lower:
                return task["id"], tl["id"], tl["title"]
            if title_lower in task_title.lower() or task_title.lower() in title_lower:
                best_match = (task["id"], tl["id"], tl["title"])

    if best_match:
        return best_match
    return None, None, None


def format_tasks_for_context(tasks):
    """Format tasks into a text block for Gemini."""
    if not tasks:
        return "No open tasks."

    lines = []
    for i, t in enumerate(tasks, 1):
        overdue = " ⚠️ OVERDUE" if t["is_overdue"] else ""
        due = f" (Due: {t['due']}{overdue})" if t["due"] else " (No due date)"
        line = f"{i}. [{t['list_name']}] {t['title']}{due}"
        if t["notes"]:
            line += f"\n   Notes: {t['notes']}"
        lines.append(line)

    return "\n\n".join(lines)
