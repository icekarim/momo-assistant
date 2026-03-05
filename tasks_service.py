from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

from googleapiclient.discovery import build
from google_auth import get_credentials


def get_tasks_service():
    creds = get_credentials()
    return build("tasks", "v1", credentials=creds)


def fetch_open_tasks():
    """Fetch all incomplete tasks across all task lists (parallel per list)."""
    svc = get_tasks_service()

    lists_resp = svc.tasklists().list(maxResults=100).execute()
    task_lists = lists_resp.get("items", [])

    def _fetch_list(tl):
        list_svc = get_tasks_service()
        tasks_resp = list_svc.tasks().list(
            tasklist=tl["id"],
            showCompleted=False,
            showHidden=False,
            maxResults=100,
        ).execute()

        results = []
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

            results.append({
                "id": task["id"],
                "title": task["title"],
                "notes": (task.get("notes", "") or "")[:300],
                "due": due,
                "is_overdue": is_overdue,
                "list_name": tl["title"],
                "status": task.get("status", ""),
            })
        return results

    all_tasks = []
    with ThreadPoolExecutor(max_workers=min(len(task_lists), 5)) as pool:
        futures = {pool.submit(_fetch_list, tl): tl["title"] for tl in task_lists}
        for future in as_completed(futures):
            try:
                all_tasks.extend(future.result())
            except Exception as e:
                print(f"Error fetching task list '{futures[future]}': {e}")

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

    lists_resp = svc.tasklists().list(maxResults=100).execute()
    task_lists = lists_resp.get("items", [])
    if not task_lists:
        return {"error": "No task lists found"}

    target_list = task_lists[0]
    if task_list_name:
        for tl in task_lists:
            if tl["title"].lower() == task_list_name.lower():
                target_list = tl
                break

    title_lower = title.lower().strip()
    for tl in task_lists:
        tasks_resp = svc.tasks().list(
            tasklist=tl["id"],
            showCompleted=False,
            showHidden=False,
            maxResults=100,
        ).execute()
        for existing in tasks_resp.get("items", []):
            existing_title = (existing.get("title", "") or "").strip()
            if not existing_title:
                continue
            if existing_title.lower() == title_lower or _titles_match(title_lower, existing_title.lower()):
                return {
                    "id": existing["id"],
                    "title": existing_title,
                    "list_name": tl["title"],
                    "status": "already_exists",
                }

    completed = find_completed_task(title, days_back=14)
    if completed:
        return {
            "id": completed["id"],
            "title": completed["title"],
            "list_name": completed["list_name"],
            "status": "already_completed",
        }

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


def _titles_match(a: str, b: str) -> bool:
    """Fuzzy match: true if one title is a substantial substring of the other,
    filtering out trivial word overlaps."""
    if a == b:
        return True
    if len(a) < 5 or len(b) < 5:
        return a == b
    if a in b or b in a:
        return True
    a_words = set(a.split())
    b_words = set(b.split())
    if not a_words or not b_words:
        return False
    overlap = a_words & b_words
    smaller = min(len(a_words), len(b_words))
    return smaller >= 2 and len(overlap) / smaller >= 0.7


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
    """Find a task by fuzzy title match (parallel per list).
    Returns (task_id, tasklist_id, list_name) or (None, None, None)."""
    title_lower = title_query.lower().strip()

    lists_resp = svc.tasklists().list(maxResults=100).execute()
    task_lists = lists_resp.get("items", [])

    def _search_list(tl):
        list_svc = get_tasks_service()
        tasks_resp = list_svc.tasks().list(
            tasklist=tl["id"],
            showCompleted=False,
            showHidden=False,
            maxResults=100,
        ).execute()

        exact = None
        partial = None
        for task in tasks_resp.get("items", []):
            task_title = (task.get("title", "") or "").strip()
            if not task_title or task.get("status") == "completed":
                continue
            if task_title.lower() == title_lower:
                exact = (task["id"], tl["id"], tl["title"])
                break
            if title_lower in task_title.lower() or task_title.lower() in title_lower:
                partial = (task["id"], tl["id"], tl["title"])
        return exact, partial

    exact_match = None
    best_partial = None
    with ThreadPoolExecutor(max_workers=min(len(task_lists), 5)) as pool:
        futures = [pool.submit(_search_list, tl) for tl in task_lists]
        for future in as_completed(futures):
            try:
                exact, partial = future.result()
                if exact:
                    exact_match = exact
                if partial and not best_partial:
                    best_partial = partial
            except Exception:
                pass

    if exact_match:
        return exact_match
    if best_partial:
        return best_partial
    return None, None, None


def fetch_recently_completed_tasks(days_back=14):
    """Fetch tasks completed in the last N days so Gemini knows what's already done."""
    svc = get_tasks_service()
    from datetime import timedelta
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days_back)).isoformat()

    completed = []
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
            title = (task.get("title", "") or "").strip()
            if not title or task.get("status") != "completed":
                continue
            completed.append({"title": title, "list_name": tl["title"]})
    return completed


def format_tasks_for_context(tasks, include_completed=True):
    """Format tasks into a text block for Gemini."""
    if not tasks:
        lines_text = "No open tasks."
    else:
        lines = []
        for i, t in enumerate(tasks, 1):
            overdue = " ⚠️ OVERDUE" if t["is_overdue"] else ""
            due = f" (Due: {t['due']}{overdue})" if t["due"] else " (No due date)"
            line = f"{i}. [{t['list_name']}] {t['title']}{due}"
            if t["notes"]:
                line += f"\n   Notes: {t['notes']}"
            lines.append(line)
        lines_text = "\n\n".join(lines)

    if include_completed:
        try:
            completed = fetch_recently_completed_tasks(days_back=14)
            if completed:
                lines_text += "\n\n--- Recently completed (do NOT re-create these) ---\n"
                for c in completed:
                    lines_text += f"- [DONE] {c['title']}\n"
        except Exception as e:
            print(f"Warning: couldn't fetch completed tasks: {e}")

    return lines_text
