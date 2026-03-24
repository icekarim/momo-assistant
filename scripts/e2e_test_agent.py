#!/usr/bin/env python3
"""
E2E Test Agent — generates and runs tests, verifies live GCloud health,
and posts results to Notion.

Usage:
    # Test uncommitted changes (default) — unit tests + live checks + Notion
    python scripts/e2e_test_agent.py

    # Test a specific commit
    python scripts/e2e_test_agent.py --commit HEAD

    # Test changes between two refs
    python scripts/e2e_test_agent.py --range main..feature-branch

    # Dry run — print generated tests without executing
    python scripts/e2e_test_agent.py --dry-run

    # Skip live GCloud checks
    python scripts/e2e_test_agent.py --skip-live

    # Run only live checks (skip unit test generation)
    python scripts/e2e_test_agent.py --live-only

    # Skip Notion reporting
    python scripts/e2e_test_agent.py --skip-notion

Requires GEMINI_API_KEY in .env or environment.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import textwrap
import urllib.request
import urllib.error
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv
import google.generativeai as genai

# ── Config ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

MODEL = os.getenv("E2E_TEST_MODEL", "gemini-3.1-pro-preview")
VENV_PYTHON = str(PROJECT_ROOT / ".venv" / "bin" / "python")

GCP_PROJECT = os.getenv("GCP_PROJECT_ID", "operations-api-455512")
CLOUD_RUN_SERVICE = os.getenv("CLOUD_RUN_SERVICE", "momo")
CLOUD_RUN_REGION = os.getenv("CLOUD_RUN_REGION", "us-central1")
CLOUD_RUN_URL = os.getenv("CLOUD_RUN_URL", "https://momo-ia4bhvubwa-uc.a.run.app")
FIRESTORE_DB = os.getenv("FIRESTORE_DATABASE", "testing")

NOTION_TOKEN = os.getenv("NOTION_API_KEY", "")
NOTION_TEST_DB_ID = "32dd79c1-41fc-81ca-97d1-eea0212f8ca2"
NOTION_VERSION = "2022-06-28"

SYSTEM_PROMPT = textwrap.dedent("""\
    You are a senior QA engineer. You will receive:
    1. A git diff showing what changed
    2. The full content of every changed file

    Your job: write a self-contained Python test script that validates the changes.

    Rules:
    - Use unittest and unittest.mock (no pytest, no extra deps).
    - Mock ALL external services: Firestore, Google APIs, Gemini, Chat, Gmail, Calendar.
    - Do NOT make real network calls or require credentials.
    - Import only from the project's own modules and the standard library.
    - Test the actual logic introduced in the diff — not just that files parse.
    - Cover: normal paths, edge cases, error handling.
    - Print a clear PASS/FAIL summary at the end.
    - The script must be runnable as: python test_generated.py
    - Output ONLY the Python code inside a single ```python ... ``` block. No explanation.
""")


# ── Helpers ─────────────────────────────────────────────────────────
def run_cmd(cmd: list[str], cwd: str | None = None, timeout: int = 30) -> str:
    result = subprocess.run(
        cmd, capture_output=True, text=True,
        cwd=cwd or str(PROJECT_ROOT), timeout=timeout,
    )
    return result.stdout.strip()


def run_cmd_full(cmd: list[str], timeout: int = 30) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd, capture_output=True, text=True,
        cwd=str(PROJECT_ROOT), timeout=timeout,
    )


def get_diff(args: argparse.Namespace) -> str:
    if args.range:
        return run_cmd(["git", "diff", args.range])
    if args.commit:
        return run_cmd(["git", "diff", f"{args.commit}~1", args.commit])
    return run_cmd(["git", "diff", "HEAD"])


def get_changed_files(args: argparse.Namespace) -> list[str]:
    if args.range:
        raw = run_cmd(["git", "diff", "--name-only", args.range])
    elif args.commit:
        raw = run_cmd(["git", "diff", "--name-only", f"{args.commit}~1", args.commit])
    else:
        raw = run_cmd(["git", "diff", "--name-only", "HEAD"])
    return [f for f in raw.splitlines() if f.endswith(".py")]


def read_file_contents(files: list[str]) -> dict[str, str]:
    contents = {}
    for f in files:
        path = PROJECT_ROOT / f
        if path.exists():
            contents[f] = path.read_text()
    return contents


def extract_code_block(response_text: str) -> str:
    match = re.search(r"```(?:python)?\s*\n(.*?)```", response_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"```(?:python)?\s*\n(.*)", response_text, re.DOTALL)
    if match:
        return match.group(1).strip().rstrip("`")
    lines = response_text.strip().splitlines()
    lines = [l for l in lines if not l.strip().startswith("```")]
    cleaned = "\n".join(lines).strip()
    if "import " in cleaned and "def test" in cleaned:
        return cleaned
    return ""


def _minutes_ago(n: int) -> str:
    dt = datetime.now(timezone.utc) - timedelta(minutes=n)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _get_commit_hash() -> str:
    return run_cmd(["git", "rev-parse", "--short", "HEAD"])


# ── Live GCloud Checks ─────────────────────────────────────────────
def check_cloud_run_health() -> tuple[bool, str]:
    """Hit the /health endpoint on the live Cloud Run service."""
    print("[live] Checking Cloud Run health endpoint...")
    try:
        result = run_cmd_full([
            "curl", "-s", "-o", "/dev/null", "-w", "%{http_code}",
            "--max-time", "10",
            f"{CLOUD_RUN_URL}/health",
        ], timeout=15)
        status_code = result.stdout.strip()
        if status_code == "200":
            return True, "/health → 200 OK"
        return False, f"/health → {status_code}"
    except Exception as e:
        return False, f"/health → error: {e}"


def check_cloud_run_revision() -> tuple[bool, str]:
    """Check the latest Cloud Run revision is serving and healthy."""
    print("[live] Checking Cloud Run revision status...")
    try:
        raw = run_cmd([
            "gcloud", "run", "revisions", "list",
            f"--service={CLOUD_RUN_SERVICE}",
            f"--region={CLOUD_RUN_REGION}",
            f"--project={GCP_PROJECT}",
            "--format=json",
            "--limit=1",
        ], timeout=15)
        if not raw:
            return False, "No revisions found"
        revisions = json.loads(raw)
        if not revisions:
            return False, "No revisions found"
        rev = revisions[0]
        name = rev.get("metadata", {}).get("name", "unknown")
        ready = any(
            c.get("type") == "Ready" and c.get("status") == "True"
            for c in rev.get("status", {}).get("conditions", [])
        )
        if ready:
            return True, f"Revision {name} — Ready"
        return False, f"Revision {name} — NOT ready"
    except Exception as e:
        return False, f"Revision check error: {e}"


def check_cloud_run_logs() -> tuple[bool, str]:
    """Scan recent Cloud Run logs for errors (last 5 minutes)."""
    print("[live] Scanning Cloud Run logs for errors (last 5m)...")
    try:
        log_filter = (
            f'resource.type="cloud_run_revision" '
            f'resource.labels.service_name="{CLOUD_RUN_SERVICE}" '
            f'severity>=ERROR '
            f'timestamp>="{_minutes_ago(5)}"'
        )
        raw = run_cmd([
            "gcloud", "logging", "read", log_filter,
            f"--project={GCP_PROJECT}",
            "--format=json",
            "--limit=10",
        ], timeout=20)
        if not raw or raw == "[]":
            return True, "No errors in last 5 minutes"
        errors = json.loads(raw)
        count = len(errors)
        samples = []
        for e in errors[:3]:
            msg = (
                e.get("textPayload", "")
                or e.get("jsonPayload", {}).get("message", "")
                or str(e.get("jsonPayload", ""))[:120]
            )
            samples.append(f"  - {msg[:120]}")
        detail = "\n".join(samples)
        return False, f"{count} error(s) in last 5 minutes:\n{detail}"
    except Exception as e:
        return False, f"Log check error: {e}"


def check_firestore_connectivity() -> tuple[bool, str]:
    """Verify Firestore is reachable by reading a known collection."""
    print("[live] Checking Firestore connectivity...")
    try:
        code = (
            "from google.cloud import firestore; "
            f"db = firestore.Client(project='{GCP_PROJECT}', database='{FIRESTORE_DB}'); "
            "docs = list(db.collection('conversations').limit(1).stream()); "
            "print(f'OK — conversations collection reachable ({len(docs)} doc(s) sampled)')"
        )
        result = run_cmd_full([VENV_PYTHON, "-c", code], timeout=15)
        if result.returncode == 0:
            return True, result.stdout.strip()
        return False, f"Firestore error: {result.stderr.strip()[:200]}"
    except Exception as e:
        return False, f"Firestore error: {e}"


def run_live_checks() -> tuple[bool, list[tuple[str, bool, str]]]:
    """Run all live GCloud health checks. Returns (all_passed, results)."""
    print("\n" + "=" * 50)
    print("LIVE GCLOUD HEALTH CHECKS")
    print("=" * 50)

    checks = [
        ("Cloud Run Health", check_cloud_run_health),
        ("Cloud Run Revision", check_cloud_run_revision),
        ("Cloud Run Logs", check_cloud_run_logs),
        ("Firestore", check_firestore_connectivity),
    ]

    results = []
    for name, fn in checks:
        try:
            passed, detail = fn()
        except Exception as e:
            passed, detail = False, f"Unexpected error: {e}"
        status = "PASS" if passed else "FAIL"
        results.append((name, passed, detail))
        print(f"  [{status}] {name}: {detail}")

    all_passed = all(r[1] for r in results)
    print()
    if all_passed:
        print("[live] ALL LIVE CHECKS PASSED")
    else:
        failed = [r[0] for r in results if not r[1]]
        print(f"[live] LIVE CHECKS FAILED: {', '.join(failed)}")

    return all_passed, results


# ── Unit Test Generation ───────────────────────────────────────────
def run_unit_tests(args: argparse.Namespace) -> tuple[int, str]:
    """Generate and run unit tests via Gemini. Returns (exit_code, output_summary)."""
    model = args.model or MODEL

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not set in .env or environment")
        return 1, "No API key"

    genai.configure(api_key=api_key)

    print(f"[agent] Analyzing changes with {model}...")
    diff = get_diff(args)
    if not diff:
        print("[agent] No changes detected. Nothing to test.")
        return 0, "No changes"

    changed_files = get_changed_files(args)
    if not changed_files:
        print("[agent] No Python files changed. Nothing to test.")
        return 0, "No Python changes"

    file_contents = read_file_contents(changed_files)
    print(f"[agent] Changed files: {', '.join(changed_files)}")

    user_msg = "## Git Diff\n```\n" + diff + "\n```\n\n"
    for fname, content in file_contents.items():
        user_msg += f"## Full file: {fname}\n```python\n{content}\n```\n\n"

    print("[agent] Generating tests...")
    gemini_model = genai.GenerativeModel(
        model_name=model,
        system_instruction=SYSTEM_PROMPT,
    )
    response = gemini_model.generate_content(
        user_msg,
        generation_config=genai.types.GenerationConfig(
            temperature=0.2,
            max_output_tokens=65536,
        ),
    )

    raw_response = response.text or ""
    print(f"[agent] Response length: {len(raw_response)} chars")
    test_code = extract_code_block(raw_response)

    if not test_code:
        print("[agent] ERROR: Gemini did not return valid test code.")
        return 1, "Gemini returned no valid code"

    test_file = PROJECT_ROOT / "test_generated_e2e.py"
    test_file.write_text(test_code)
    print(f"[agent] Tests written to {test_file}")

    if args.dry_run:
        print("\n── Generated Tests ──")
        print(test_code)
        return 0, "Dry run"

    print("[agent] Running tests...\n")
    result = subprocess.run(
        [VENV_PYTHON, str(test_file)],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=60,
    )

    # Parse test output for counts
    output = result.stdout + result.stderr
    print(output)

    # Extract "Ran N tests" and failure count
    ran_match = re.search(r"Ran (\d+) test", output)
    fail_match = re.search(r"failures=(\d+)", output)
    error_match = re.search(r"errors=(\d+)", output)

    total = int(ran_match.group(1)) if ran_match else 0
    failures = int(fail_match.group(1)) if fail_match else 0
    errors = int(error_match.group(1)) if error_match else 0
    passed = total - failures - errors

    summary = f"{passed}/{total} passed"
    if failures:
        summary += f", {failures} failed"
    if errors:
        summary += f", {errors} errors"

    print()
    if result.returncode == 0:
        print("[agent] ALL UNIT TESTS PASSED")
    else:
        print(f"[agent] UNIT TESTS FAILED (exit code {result.returncode})")
        print(f"[agent] Review: {test_file}")

    return result.returncode, summary


# ── Notion Reporting ───────────────────────────────────────────────
def _notion_request(method: str, endpoint: str, data: dict | None = None):
    """Make a request to the Notion API."""
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
        print(f"[notion] API error {e.code}: {error_body[:300]}", file=sys.stderr)
        return None


def post_to_notion(
    changed_files: list[str],
    unit_exit_code: int | None,
    unit_summary: str,
    live_passed: bool | None,
    live_results: list[tuple[str, bool, str]] | None,
):
    """Post test results as a new row in the Notion E2E Test Results database."""
    if not NOTION_TOKEN:
        print("[notion] Skipped — NOTION_API_KEY not set")
        return

    print("[notion] Posting results to Notion...")
    now = datetime.now(timezone.utc)
    commit = _get_commit_hash()

    # Determine overall result
    if unit_exit_code is not None and unit_exit_code != 0:
        if live_passed is True:
            overall = "Partial"
        else:
            overall = "Fail"
    elif live_passed is False:
        overall = "Partial"
    else:
        overall = "Pass"

    # Determine live check status
    if live_passed is None:
        live_status = "Skipped"
    elif live_passed:
        live_status = "Pass"
    else:
        live_status = "Fail"

    run_title = f"{now.strftime('%Y-%m-%d %H:%M')} — {commit}"

    properties = {
        "Run": {"title": [{"text": {"content": run_title}}]},
        "Result": {"select": {"name": overall}},
        "Unit Tests": {"rich_text": [{"text": {"content": unit_summary or "Skipped"}}]},
        "Live Checks": {"select": {"name": live_status}},
        "Changed Files": {"rich_text": [{"text": {"content": ", ".join(changed_files)[:2000]}}]},
        "Commit": {"rich_text": [{"text": {"content": commit}}]},
        "Date": {"date": {"start": now.isoformat()}},
    }

    page = _notion_request("POST", "pages", {
        "parent": {"database_id": NOTION_TEST_DB_ID},
        "properties": properties,
    })

    if not page:
        print("[notion] Failed to create page")
        return

    # Add detailed results as page content
    blocks = []

    # Unit test details
    blocks.append({"object": "block", "type": "heading_2", "heading_2": {
        "rich_text": [{"type": "text", "text": {"content": "Unit Tests"}}]
    }})
    blocks.append({"object": "block", "type": "paragraph", "paragraph": {
        "rich_text": [{"type": "text", "text": {"content": unit_summary or "Skipped"}}]
    }})

    # Live check details
    if live_results:
        blocks.append({"object": "block", "type": "heading_2", "heading_2": {
            "rich_text": [{"type": "text", "text": {"content": "Live GCloud Checks"}}]
        }})
        for name, passed, detail in live_results:
            icon = "PASS" if passed else "FAIL"
            blocks.append({"object": "block", "type": "bulleted_list_item",
                "bulleted_list_item": {"rich_text": [{"type": "text", "text": {
                    "content": f"[{icon}] {name}: {detail}"
                }}]}
            })

    # Changed files
    if changed_files:
        blocks.append({"object": "block", "type": "heading_2", "heading_2": {
            "rich_text": [{"type": "text", "text": {"content": "Changed Files"}}]
        }})
        for f in changed_files:
            blocks.append({"object": "block", "type": "bulleted_list_item",
                "bulleted_list_item": {"rich_text": [{"type": "text", "text": {"content": f}}]}
            })

    if blocks:
        _notion_request("PATCH", f"blocks/{page['id']}/children", {"children": blocks})

    print(f"[notion] Results posted: {page.get('url', 'unknown')}")


# ── Main ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="E2E test agent with live GCloud checks + Notion")
    parser.add_argument("--commit", help="Test a specific commit (e.g. HEAD, abc1234)")
    parser.add_argument("--range", help="Test a commit range (e.g. main..feature)")
    parser.add_argument("--dry-run", action="store_true", help="Print tests without running")
    parser.add_argument("--model", help=f"Override model (default: {MODEL})")
    parser.add_argument("--skip-live", action="store_true", help="Skip live GCloud checks")
    parser.add_argument("--live-only", action="store_true", help="Run only live checks")
    parser.add_argument("--skip-notion", action="store_true", help="Skip Notion reporting")
    args = parser.parse_args()

    exit_code = 0
    unit_exit_code = None
    unit_summary = ""
    live_passed = None
    live_results = None
    changed_files = get_changed_files(args) if not args.live_only else []

    # Phase 1: Unit tests (unless --live-only)
    if not args.live_only:
        unit_exit_code, unit_summary = run_unit_tests(args)
        exit_code = unit_exit_code
        if unit_exit_code != 0 and not args.skip_live:
            print("\n[agent] Skipping live checks — unit tests failed.")
            if not args.skip_notion and not args.dry_run:
                post_to_notion(changed_files, unit_exit_code, unit_summary, None, None)
            sys.exit(exit_code)

    # Phase 2: Live GCloud checks (unless --skip-live or --dry-run)
    if not args.skip_live and not args.dry_run:
        live_passed, live_results = run_live_checks()
        if not live_passed:
            exit_code = 1

    # Phase 3: Post to Notion (unless --skip-notion or --dry-run)
    if not args.skip_notion and not args.dry_run:
        post_to_notion(changed_files, unit_exit_code, unit_summary, live_passed, live_results)

    # Final summary
    print("\n" + "=" * 50)
    if exit_code == 0:
        print("E2E AGENT: ALL CHECKS PASSED")
    else:
        print("E2E AGENT: SOME CHECKS FAILED")
    print("=" * 50)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
