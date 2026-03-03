#!/bin/bash
#
# Auto Commit & Push Script
# Detects changes on any branch (including main), creates a new branch,
# and pushes to trigger the auto-PR workflow.
#

set -e

# Configuration
BRANCH_PREFIX="${BRANCH_PREFIX:-auto}"
MAIN_BRANCH="${MAIN_BRANCH:-main}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    log_error "Not in a git repository"
    exit 1
fi

# Get current branch
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
log_info "Current branch: ${CURRENT_BRANCH}"

# Check for changes
if [ -z "$(git status --porcelain)" ]; then
    log_info "No changes to commit"
    exit 0
fi

# Show current changes
log_info "Detected changes:"
git status --short

# Generate branch name
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
if [ "$CURRENT_BRANCH" = "$MAIN_BRANCH" ]; then
    BRANCH_NAME="${BRANCH_PREFIX}/main-changes-${TIMESTAMP}"
else
    BRANCH_NAME="${BRANCH_PREFIX}/changes-${TIMESTAMP}"
fi

# Generate commit message
MODIFIED=$(git status --porcelain | grep -c "^ M\|^M " || echo "0")
ADDED=$(git status --porcelain | grep -c "^A \|^??" || echo "0")
DELETED=$(git status --porcelain | grep -c "^ D\|^D " || echo "0")

COMMIT_MSG="Auto: Modified ${MODIFIED}, Added ${ADDED}, Deleted ${DELETED} file(s)"

# If a custom message was provided, use it
if [ -n "$1" ]; then
    COMMIT_MSG="$1"
fi

log_info "Creating branch: ${BRANCH_NAME}"

# If on main, stash changes, ensure we're up to date, then create branch
if [ "$CURRENT_BRANCH" = "$MAIN_BRANCH" ]; then
    log_info "Detected changes on main branch, creating PR branch..."
    
    # Stash current changes
    git stash push -m "auto-pr-agent-temp-stash"
    
    # Pull latest from main
    git pull --rebase origin "$MAIN_BRANCH" || true
    
    # Create new branch from main
    git checkout -b "$BRANCH_NAME"
    
    # Pop stashed changes
    git stash pop || true
else
    # If not on main, first stash, switch to main, create branch
    log_warn "Currently on branch ${CURRENT_BRANCH}, switching to ${MAIN_BRANCH} first"
    git stash push -m "auto-pr-agent-temp-stash"
    git checkout "$MAIN_BRANCH"
    git pull --rebase origin "$MAIN_BRANCH" || true
    git checkout -b "$BRANCH_NAME"
    git stash pop || true
fi

# Stage and commit all changes
git add -A
git commit -m "$COMMIT_MSG"

log_info "Pushing to remote..."
git push -u origin "$BRANCH_NAME"

log_info "Branch pushed! The auto-PR workflow will now create and merge the PR."

# Switch back to main and pull
git checkout "$MAIN_BRANCH"
git pull --rebase origin "$MAIN_BRANCH" || true

log_info "Done! Check GitHub Actions for PR status."
log_info "PR will be auto-merged after checks pass."
