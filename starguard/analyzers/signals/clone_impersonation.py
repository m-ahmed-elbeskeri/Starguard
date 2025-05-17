"""
Detects repositories that are clones of other repositories without being marked as forks,
using GitHub’s REST commit‐search preview endpoint.
"""

import logging
import os
import requests
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# -----------------------------
# Configuration for REST search
# -----------------------------
COMMIT_SEARCH_URL = "https://api.github.com/search/commits"
PREVIEW_ACCEPT_HEADER = "application/vnd.github.cloak-preview+json"
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")  # make sure this is set in your environment


def search_commit_rest(sha: str, per_page: int = 3) -> Dict[str, Any]:
    """
    Search for a commit SHA across all repositories via the REST API.
    Requires the 'cloak-preview' media type for commit search.
    """
    headers = {"Accept": PREVIEW_ACCEPT_HEADER}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"
    params = {"q": sha, "per_page": per_page}
    resp = requests.get(COMMIT_SEARCH_URL, headers=headers, params=params)
    resp.raise_for_status()
    return resp.json()


def check_clone_impersonation(api: Any, owner: str, repo: str) -> bool:
    """
    Synchronous version to detect if a repository is a clone impersonation.

    Args:
        api: GitHubAPI-like instance, with methods:
             - get_repo(owner, repo) -> dict
             - request(path: str, params: dict) -> list of commits
        owner: repository owner
        repo: repository name

    Returns:
        True if repo appears to be a clone impersonation (>=80% of earliest commits
        were pushed to other repos earlier than this one), False otherwise.
    """
    try:
        # 1) Skip if it's already marked as a fork
        repo_data = api.get_repo(owner, repo)
        if repo_data.get("fork", False):
            return False

        # 2) Fetch the most recent commits (newest first)
        commits: List[Dict[str, Any]] = api.request(
            f"/repos/{owner}/{repo}/commits", params={"per_page": 20}
        )
        if not isinstance(commits, list) or len(commits) < 5:
            logger.debug(f"Not enough commits to analyze for {owner}/{repo}")
            return False

        # 3) Reverse to get chronological order (oldest first), then take up to 20
        chronological = list(reversed(commits))
        earliest_commits = chronological[:20]
        total_checks = len(earliest_commits)
        if total_checks == 0:
            return False

        match_hits = 0

        for idx, commit in enumerate(earliest_commits):
            sha = commit.get("sha")
            if not sha:
                continue

            # 4) REST‐based commit search
            search_result = search_commit_rest(sha, per_page=3)
            for item in search_result.get("items", []):
                repo_info = item.get("repository", {})
                name_with_owner = repo_info.get("full_name", "")

                # skip self
                if name_with_owner.lower() == f"{owner}/{repo}".lower():
                    continue

                pushed_at = repo_info.get("pushed_at")
                commit_date = commit.get("commit", {}).get("author", {}).get("date")
                if pushed_at and commit_date and pushed_at < commit_date:
                    match_hits += 1
                    break  # move to next commit

            # early positive
            if match_hits >= 0.8 * total_checks:
                return True

            # early negative: not enough remaining to reach 80%
            remaining = total_checks - idx - 1
            if match_hits + remaining < 0.8 * total_checks:
                return False

        # final check
        return (match_hits / total_checks) >= 0.8

    except Exception as e:
        logger.warning(f"Error in clone impersonation check for {owner}/{repo}: {e}")
        return False  # safe default on error


async def is_clone_impersonation(api: Any, owner: str, repo: str) -> bool:
    """
    Async wrapper for the synchronous implementation.
    """
    return check_clone_impersonation(api, owner, repo)
