#!/usr/bin/env python3
"""
Auto PR Agent - Detects changes, creates branches, pushes, and triggers auto-merge.

This agent monitors your codebase for changes, automatically commits them,
creates pull requests, and merges them without manual intervention.
"""

import subprocess
import hashlib
import json
import time
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


class AutoPRAgent:
    """Agent that automatically detects changes, creates PRs, and merges them."""

    def __init__(
        self,
        repo_path: str = ".",
        branch_prefix: str = "auto",
        main_branch: str = "main",
        auto_merge: bool = True,
        poll_interval: int = 30,
    ):
        self.repo_path = Path(repo_path).resolve()
        self.branch_prefix = branch_prefix
        self.main_branch = main_branch
        self.auto_merge = auto_merge
        self.poll_interval = poll_interval
        self.state_file = self.repo_path / ".auto_pr_state.json"
        self._last_state: dict = {}

    def _run_git(self, *args: str, check: bool = True) -> subprocess.CompletedProcess:
        """Execute a git command in the repository."""
        result = subprocess.run(
            ["git", *args],
            cwd=self.repo_path,
            capture_output=True,
            text=True,
        )
        if check and result.returncode != 0:
            raise RuntimeError(f"Git command failed: {result.stderr}")
        return result

    def _run_gh(self, *args: str, check: bool = True) -> subprocess.CompletedProcess:
        """Execute a GitHub CLI command."""
        result = subprocess.run(
            ["gh", *args],
            cwd=self.repo_path,
            capture_output=True,
            text=True,
        )
        if check and result.returncode != 0:
            raise RuntimeError(f"GitHub CLI command failed: {result.stderr}")
        return result

    def get_file_hash(self, file_path: Path) -> str:
        """Calculate MD5 hash of a file."""
        try:
            with open(file_path, "rb") as f:
                return hashlib.md5(f.read()).hexdigest()
        except (IOError, OSError):
            return ""

    def get_tracked_files(self) -> dict[str, str]:
        """Get all tracked files and their hashes."""
        result = self._run_git("ls-files", check=False)
        if result.returncode != 0:
            return {}

        files = {}
        for file_name in result.stdout.strip().split("\n"):
            if file_name:
                file_path = self.repo_path / file_name
                if file_path.exists():
                    files[file_name] = self.get_file_hash(file_path)
        return files

    def detect_changes(self) -> tuple[list[str], list[str], list[str]]:
        """Detect modified, added, and deleted files."""
        result = self._run_git("status", "--porcelain", check=False)
        if result.returncode != 0:
            return [], [], []

        modified = []
        added = []
        deleted = []

        for line in result.stdout.strip().split("\n"):
            if not line:
                continue

            status = line[:2]
            file_name = line[3:].strip()

            if file_name.startswith('"') and file_name.endswith('"'):
                file_name = file_name[1:-1]

            if "M" in status:
                modified.append(file_name)
            elif "A" in status or "?" in status:
                added.append(file_name)
            elif "D" in status:
                deleted.append(file_name)

        return modified, added, deleted

    def has_changes(self) -> bool:
        """Check if there are any uncommitted changes."""
        modified, added, deleted = self.detect_changes()
        return bool(modified or added or deleted)

    def generate_branch_name(self) -> str:
        """Generate a unique branch name based on timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return f"{self.branch_prefix}/changes-{timestamp}"

    def generate_commit_message(
        self, modified: list[str], added: list[str], deleted: list[str]
    ) -> str:
        """Generate a descriptive commit message."""
        parts = []

        if modified:
            parts.append(f"Modified {len(modified)} file(s)")
        if added:
            parts.append(f"Added {len(added)} file(s)")
        if deleted:
            parts.append(f"Deleted {len(deleted)} file(s)")

        summary = ", ".join(parts) if parts else "Update files"

        details = []
        for f in modified[:5]:
            details.append(f"  - Modified: {f}")
        for f in added[:5]:
            details.append(f"  - Added: {f}")
        for f in deleted[:5]:
            details.append(f"  - Deleted: {f}")

        total = len(modified) + len(added) + len(deleted)
        if total > 15:
            details.append(f"  - ... and {total - 15} more files")

        return f"{summary}\n\n" + "\n".join(details) if details else summary

    def ensure_on_main(self) -> None:
        """Ensure we're on the main branch."""
        result = self._run_git("rev-parse", "--abbrev-ref", "HEAD")
        current_branch = result.stdout.strip()

        if current_branch != self.main_branch:
            print(f"Switching to {self.main_branch} branch...")
            self._run_git("checkout", self.main_branch)
            self._run_git("pull", "--rebase", "origin", self.main_branch, check=False)

    def create_branch_and_commit(
        self, branch_name: str, commit_message: str
    ) -> bool:
        """Create a new branch and commit all changes."""
        try:
            self._run_git("checkout", "-b", branch_name)
            self._run_git("add", "-A")
            self._run_git("commit", "-m", commit_message)
            return True
        except RuntimeError as e:
            print(f"Failed to create branch and commit: {e}")
            self._run_git("checkout", self.main_branch, check=False)
            self._run_git("branch", "-D", branch_name, check=False)
            return False

    def push_branch(self, branch_name: str) -> bool:
        """Push branch to remote."""
        try:
            self._run_git("push", "-u", "origin", branch_name)
            return True
        except RuntimeError as e:
            print(f"Failed to push branch: {e}")
            return False

    def create_pr(self, branch_name: str, title: str, body: str) -> Optional[str]:
        """Create a pull request using GitHub CLI."""
        try:
            result = self._run_gh(
                "pr", "create",
                "--title", title,
                "--body", body,
                "--base", self.main_branch,
                "--head", branch_name,
            )
            pr_url = result.stdout.strip()
            print(f"Created PR: {pr_url}")
            return pr_url
        except RuntimeError as e:
            print(f"Failed to create PR: {e}")
            return None

    def merge_pr(self, branch_name: str) -> bool:
        """Merge the PR using GitHub CLI."""
        try:
            self._run_gh(
                "pr", "merge", branch_name,
                "--squash",
                "--delete-branch",
                "--auto",
            )
            print(f"Enabled auto-merge for branch: {branch_name}")
            return True
        except RuntimeError:
            try:
                self._run_gh(
                    "pr", "merge", branch_name,
                    "--squash",
                    "--delete-branch",
                )
                print(f"Merged branch: {branch_name}")
                return True
            except RuntimeError as e:
                print(f"Failed to merge PR: {e}")
                return False

    def process_changes(self) -> bool:
        """Process any detected changes: commit, push, create PR, and merge."""
        if not self.has_changes():
            return False

        modified, added, deleted = self.detect_changes()
        print(f"\nDetected changes:")
        print(f"  Modified: {len(modified)}")
        print(f"  Added: {len(added)}")
        print(f"  Deleted: {len(deleted)}")

        branch_name = self.generate_branch_name()
        commit_message = self.generate_commit_message(modified, added, deleted)

        print(f"\nCreating branch: {branch_name}")
        if not self.create_branch_and_commit(branch_name, commit_message):
            return False

        print(f"Pushing to remote...")
        if not self.push_branch(branch_name):
            return False

        pr_title = f"Auto: {commit_message.split(chr(10))[0]}"
        pr_body = f"""## Automated Changes

{commit_message}

---
*This PR was automatically created by Auto PR Agent*
*Timestamp: {datetime.now().isoformat()}*
"""

        print(f"Creating PR...")
        pr_url = self.create_pr(branch_name, pr_title, pr_body)

        if pr_url and self.auto_merge:
            print(f"Initiating auto-merge...")
            self.merge_pr(branch_name)

        self._run_git("checkout", self.main_branch, check=False)
        self._run_git("pull", "--rebase", "origin", self.main_branch, check=False)

        return True

    def run_once(self) -> bool:
        """Run the agent once to check for and process changes."""
        print(f"\n{'='*50}")
        print(f"Auto PR Agent - {datetime.now().isoformat()}")
        print(f"{'='*50}")

        self.ensure_on_main()
        return self.process_changes()

    def run_daemon(self) -> None:
        """Run the agent as a daemon, continuously monitoring for changes."""
        print(f"Starting Auto PR Agent daemon...")
        print(f"  Repository: {self.repo_path}")
        print(f"  Poll interval: {self.poll_interval}s")
        print(f"  Auto-merge: {self.auto_merge}")

        while True:
            try:
                self.run_once()
            except KeyboardInterrupt:
                print("\nStopping daemon...")
                break
            except Exception as e:
                print(f"Error: {e}")

            time.sleep(self.poll_interval)


def main():
    """Main entry point for the Auto PR Agent."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Auto PR Agent - Automatically detect changes, create PRs, and merge them"
    )
    parser.add_argument(
        "--repo", "-r",
        default=".",
        help="Path to the git repository (default: current directory)"
    )
    parser.add_argument(
        "--branch-prefix", "-p",
        default="auto",
        help="Prefix for auto-generated branch names (default: auto)"
    )
    parser.add_argument(
        "--main-branch", "-m",
        default="main",
        help="Name of the main branch (default: main)"
    )
    parser.add_argument(
        "--no-merge",
        action="store_true",
        help="Don't automatically merge PRs"
    )
    parser.add_argument(
        "--daemon", "-d",
        action="store_true",
        help="Run as a daemon, continuously monitoring for changes"
    )
    parser.add_argument(
        "--interval", "-i",
        type=int,
        default=30,
        help="Poll interval in seconds for daemon mode (default: 30)"
    )

    args = parser.parse_args()

    agent = AutoPRAgent(
        repo_path=args.repo,
        branch_prefix=args.branch_prefix,
        main_branch=args.main_branch,
        auto_merge=not args.no_merge,
        poll_interval=args.interval,
    )

    if args.daemon:
        agent.run_daemon()
    else:
        success = agent.run_once()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
