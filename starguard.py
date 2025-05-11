
"""
StarGuard - Advanced GitHub Repository Analysis Tool with Robust Fake Star Detection

This script performs comprehensive security and trust analysis on GitHub repositories,
using a repository-only approach to detect fake stars, making it more efficient and
practical for individual repository analysis.
"""

import os
import sys
import argparse
import json
import datetime
import time
from collections import Counter, defaultdict
import re
from urllib.parse import urlparse, quote_plus
import logging
from typing import Dict, List, Tuple, Any, Optional, Union
import traceback
from operator import itemgetter
import random
import math

import requests
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
import networkx as nx
from dateutil.parser import parse as parse_date
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("starguard")

# Constants
GITHUB_API_BASE = "https://api.github.com"
GITHUB_GRAPHQL_URL = "https://api.github.com/graphql"
LICENSE_RISK_LEVELS = {
    "mit": "low",
    "apache-2.0": "low",
    "bsd-2-clause": "low",
    "bsd-3-clause": "low",
    "isc": "low",
    "cc0-1.0": "low",
    "unlicense": "low",
    "gpl-2.0": "medium",
    "gpl-3.0": "medium",
    "lgpl-2.1": "medium",
    "lgpl-3.0": "medium",
    "agpl-3.0": "high",
    "cc-by-nc-sa-4.0": "high",
    "proprietary": "high",
    "unknown": "high"
}
PACKAGE_MANAGERS = {
    "javascript": {
        "file": "package.json",
        "dependencies_key": ["dependencies", "devDependencies"]
    },
    "python": {
        "file": "requirements.txt",
        "pattern": r"^([\w\-\._]+).*?(?:==|>=|<=|~=|!=|>|<)?\s*([\d\.\w\-]+)?.*$"
    },
    "python_pipfile": {
        "file": "Pipfile",
        "section_marker": "[packages]"
    },
    "python_poetry": {
        "file": "pyproject.toml",
        "section_marker": "[tool.poetry.dependencies]"
    },
    "ruby": {
        "file": "Gemfile",
        "pattern": r"gem\s+['\"]([^'\"]+)['\"](?:,\s*['\"](.+)['\"])?"
    },
    "java": {
        "file": "pom.xml",
        "tag_pattern": r"<dependency>\s*<groupId>([^<]+)</groupId>\s*<artifactId>([^<]+)</artifactId>\s*(?:<version>([^<]+)</version>)?"
    },
    "go": {
        "file": "go.mod",
        "pattern": r"^\s*require\s+([^\s]+)\s+v?([^\s]+)"
    }
}

# Constants for fake star detection
MAD_THRESHOLD = 3.0 * 1.48  # MAD threshold for spike detection
WINDOW_SIZE = 28  # Days for sliding window in MAD calculation
MIN_STAR_COUNT = 30  # Minimum star count before using MAD detection
MIN_STARS_GROWTH_PERCENT = 300  # Alternative % growth threshold for small repos

# Fake star user scoring weights
USER_SCORE_THRESHOLDS = {
    "account_age_days": (30, 2.0),  # (threshold, score if below threshold)
    "followers": (5, 1.0),
    "public_repos": (2, 1.0),
    "total_stars": (3, 1.0),
    "prior_interaction": (0, 1.0),  # 0 = no prior interaction
    "default_avatar": (True, 0.5)  # True = has default avatar
}
FAKE_USER_THRESHOLD = 4.0  # Score threshold to flag a user as likely fake

# Burst scoring weights
FAKE_RATIO_WEIGHT = 0.7
RULE_HITS_WEIGHT = 0.3

# Burst classification thresholds
BURST_ORGANIC_THRESHOLD = 0.3
BURST_FAKE_THRESHOLD = 0.6


def make_naive_datetime(dt: Optional[datetime.datetime]) -> Optional[datetime.datetime]:
    """Convert a datetime to naive (remove timezone info)."""
    if dt is None:
        return None
    if dt.tzinfo is not None:
        return dt.replace(tzinfo=None)
    return dt


class GitHubAPI:
    """GitHub API client for StarGuard."""

    def __init__(self, token: Optional[str] = None, rate_limit_pause: bool = True):
        """Initialize the GitHub API client.

        Args:
            token: GitHub personal access token
            rate_limit_pause: Whether to pause when rate limit is hit
        """
        self.token = token
        self.rate_limit_pause = rate_limit_pause
        self.headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28"
        }
        if token:
            self.headers["Authorization"] = f"token {token}"
            self.graphql_headers = {"Authorization": f"bearer {token}"}
        else:
            self.graphql_headers = {}

        self.session = requests.Session()
        self.session.headers.update(self.headers)

        # Cache for user profile data to avoid repeated API calls
        self.user_profile_cache = {}
        self.remaining_rate_limit = 5000  # Default GitHub API rate limit
        self.rate_limit_reset = 0

    def _handle_rate_limit(self, response: requests.Response) -> bool:
        """Handle GitHub API rate limiting.

        Returns:
            bool: True if we hit rate limit and had to wait, False otherwise
        """
        # Update rate limit info
        if 'X-RateLimit-Remaining' in response.headers:
            self.remaining_rate_limit = int(response.headers['X-RateLimit-Remaining'])
            self.rate_limit_reset = int(response.headers.get('X-RateLimit-Reset', 0))

        if response.status_code == 403 and self.remaining_rate_limit <= 1 and self.rate_limit_pause: # Check if <= 1 to be safe
            sleep_time = self.rate_limit_reset - time.time() + 5 # Add 5s buffer
            if sleep_time > 0:
                logger.warning(f"Rate limit hit. Sleeping for {sleep_time:.0f} seconds until {datetime.datetime.fromtimestamp(self.rate_limit_reset)}")
                time.sleep(sleep_time)
                return True
        return False

    def request(self, endpoint: str, method: str = "GET", params: Dict = None, data: Dict = None) -> Dict:
        """Make a request to the GitHub API."""
        url = f"{GITHUB_API_BASE}{endpoint}"
        max_retries = 3
        retry_count = 0

        while retry_count < max_retries:
            try:
                response = self.session.request(method, url, params=params, json=data, timeout=30)

                if self._handle_rate_limit(response):
                    continue

                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 204:  # No content
                    return {}
                elif response.status_code == 404:
                    # For some resources, 404 is a valid non-error state (e.g., license not found)
                    # The caller should handle this specific to the endpoint.
                    # Raising ValueError here might be too general.
                    # Let's return a dict indicating the 404.
                    logger.debug(f"Resource not found (404): {url}")
                    return {"error": "Not Found", "status_code": 404}
                else:
                    if response.status_code >= 500:  # Server error, retry
                        retry_count += 1
                        logger.warning(f"Server error ({response.status_code}) from {url}. Retrying {retry_count}/{max_retries}...")
                        time.sleep(2 * (retry_count +1 ))  # Exponential backoff
                        continue
                    # For other client errors (4xx), log and return error structure
                    logger.error(f"GitHub API client error: {response.status_code} - {response.text} for URL {url}")
                    return {"error": response.text, "status_code": response.status_code}
            except requests.exceptions.RequestException as e:
                retry_count += 1
                logger.warning(f"Network error for {url}: {str(e)}. Retrying {retry_count}/{max_retries}...")
                time.sleep(2 * (retry_count+1))
                continue

        logger.error(f"Failed to make request to {url} after {max_retries} attempts")
        return {"error": f"Failed after {max_retries} retries", "status_code": 0}


    def graphql_request(self, query: str, variables: Dict = None) -> Dict:
        """Make a GraphQL request to the GitHub API."""
        if variables is None:
            variables = {}

        json_data = {
            "query": query,
            "variables": variables
        }

        max_retries = 3
        retry_count = 0

        while retry_count < max_retries:
            try:
                response = requests.post(
                    GITHUB_GRAPHQL_URL,
                    json=json_data,
                    headers=self.graphql_headers,
                    timeout=30
                )

                if self._handle_rate_limit(response):
                    continue

                if response.status_code == 200:
                    result = response.json()

                    if "errors" in result:
                        error_message = result.get("errors", [{}])[0].get("message", "Unknown GraphQL error")
                        # Certain GraphQL errors are not worth retrying, e.g., 'NOT_FOUND'
                        if "type': 'NOT_FOUND'" in str(result.get("errors")):
                             logger.warning(f"GraphQL query target not found: {error_message}")
                             return {"data": {}, "errors": result.get("errors")}

                        if retry_count < max_retries - 1:
                            retry_count += 1
                            logger.warning(f"GraphQL error: {error_message}. Retrying {retry_count}/{max_retries}...")
                            time.sleep(2 * (retry_count+1)) # Exponential backoff
                            continue
                        else:
                            logger.error(f"GraphQL API error after retries: {error_message}")
                            return {"data": {}, "errors": result.get("errors")}
                    return result
                else:
                    if response.status_code >= 500:
                        retry_count += 1
                        logger.warning(f"GraphQL server error ({response.status_code}). Retrying {retry_count}/{max_retries}...")
                        time.sleep(2 * (retry_count+1))
                        continue
                    error_message = f"GraphQL API HTTP error: {response.status_code} - {response.text}"
                    logger.error(error_message)
                    return {"data": {}, "errors": [{"message": error_message, "status_code": response.status_code}]}
            except requests.exceptions.RequestException as e:
                retry_count += 1
                logger.warning(f"GraphQL network error: {str(e)}. Retrying {retry_count}/{max_retries}...")
                time.sleep(2 * (retry_count+1))
                continue

        logger.error(f"Failed to make GraphQL request after {max_retries} attempts")
        return {"data": {}, "errors": [{"message": f"Failed after {max_retries} retries"}]}

    def paginate(self, endpoint: str, params: Dict = None) -> List[Dict]:
        """Paginate through GitHub API results."""
        if params is None:
            params = {}

        params["per_page"] = 100
        results = []
        page = 1

        while True:
            params["page"] = page
            try:
                page_data = self.request(endpoint, params=params)
                
                if not page_data or "error" in page_data: # Check for empty response or error dict
                    if "error" in page_data:
                        logger.warning(f"Error during pagination for {endpoint} (page {page}): {page_data.get('error')}")
                    break
                
                # Ensure page_data is a list before extending
                if isinstance(page_data, list):
                    results.extend(page_data)
                    if len(page_data) < 100:
                        break
                else: # Should not happen if API behaves, but good to guard
                    logger.warning(f"Unexpected data type from {endpoint} (page {page}): {type(page_data)}")
                    break

                page += 1
                
            except Exception as e: # Catch broader exceptions from request logic
                logger.error(f"Unhandled error while paginating {endpoint} (page {page}): {str(e)}")
                break
        return results

    def get_repo(self, owner: str, repo: str) -> Dict:
        """Get repository information."""
        data = self.request(f"/repos/{owner}/{repo}")
        if "error" in data and data.get("status_code") == 404:
            raise ValueError(f"Repository {owner}/{repo} not found.")
        if "error" in data:
            raise ValueError(f"Failed to fetch repo {owner}/{repo}: {data.get('error')}")
        return data

    def get_stargazers(self, owner: str, repo: str, get_timestamps: bool = True, days_limit: int = 0) -> List[Dict]:
        """Get repository stargazers with timestamps using REST API.

        Args:
            owner: Repository owner
            repo: Repository name
            get_timestamps: Whether to get timestamps
            days_limit: Limit to stars from the last X days.
                        Note: The REST API fetches all stargazers; filtering by days_limit
                        is applied post-fetch if days_limit > 0. This can be inefficient for
                        repositories with a very large number of stars.

        Returns:
            List of stargazer data with timestamps (if requested)
        """
        logger.info("Fetching stargazers using REST API.")
        try:
            all_stars = self._get_stargazers_rest(owner, repo, get_timestamps)

            if days_limit > 0 and all_stars:
                # Filter stars by days_limit *after* fetching them.
                cutoff_date_dt = make_naive_datetime(datetime.datetime.now() - datetime.timedelta(days=days_limit))

                filtered_stars = []
                for star in all_stars:
                    starred_at_str = star.get("starred_at")
                    if starred_at_str:
                        try:
                            starred_at_dt = make_naive_datetime(parse_date(starred_at_str))
                            if starred_at_dt and cutoff_date_dt and starred_at_dt >= cutoff_date_dt: # Ensure not None
                                filtered_stars.append(star)
                        except Exception as e:
                            logger.debug(f"Could not parse or compare date for star: {starred_at_str}, error: {e}")
                logger.info(f"Fetched {len(all_stars)} total stars, filtered to {len(filtered_stars)} within {days_limit} days.")
                return filtered_stars

            return all_stars
        except Exception as e:
            logger.warning(f"Error fetching stargazers via REST API: {str(e)}")
            return []

    def _get_stargazers_graphql(self, owner: str, repo: str, days_limit: int = 0) -> List[Dict]:
        """Get stargazers using GraphQL API. (Currently bypassed by get_stargazers)"""
        logger.debug("Attempting to fetch stargazers using GraphQL (bypassed in get_stargazers method).")
        # stars = []
        # cursor = "null"
        # cutoff_date = (datetime.datetime.now() - datetime.timedelta(days=days_limit)).isoformat() if days_limit > 0 else None
        #
        # while True:
        #     query = f"""
        #     query {{
        #       repository(owner: "{owner}", name: "{repo}") {{
        #         stargazers(first: 100, after: {cursor}, orderBy: {{field: STARRED_AT, direction: DESC}}) {{ # Added orderBy
        #           pageInfo {{
        #             hasNextPage
        #             endCursor
        #           }}
        #           edges {{
        #             starredAt
        #             node {{
        #               login
        #               avatarUrl
        #               createdAt
        #               followers {{
        #                 totalCount
        #               }}
        #               repositories {{
        #                 totalCount
        #               }}
        #               starredRepositories {{
        #                 totalCount
        #               }}
        #             }}
        #           }}
        #         }}
        #       }}
        #     }}
        #     """
        #     result = self.graphql_request(query)
        #     if "errors" in result or not result.get("data"):
        #         logger.warning(f"GraphQL query for stargazers failed or returned no data. Errors: {result.get('errors')}")
        #         return stars if stars else [] # Return what we have or empty list
        #
        #     data = result.get("data", {}).get("repository", {}).get("stargazers", {})
        #     if not data:
        #         break
        #
        #     page_info = data.get("pageInfo", {})
        #     edges = data.get("edges", [])
        #
        #     for edge in edges:
        #         starred_at = edge.get("starredAt")
        #         if cutoff_date and starred_at < cutoff_date:
        #             logger.info(f"Reached cutoff date {cutoff_date} for stargazers via GraphQL.")
        #             return stars # Stop fetching if stars are older than limit
        #
        #         node = edge.get("node", {})
        #         stars.append({
        #             "starred_at": starred_at,
        #             "user": {
        #                 "login": node.get("login"),
        #                 "avatar_url": node.get("avatarUrl"),
        #                 "created_at": node.get("createdAt"),
        #                 "followers_count": node.get("followers", {}).get("totalCount", 0),
        #                 "public_repos": node.get("repositories", {}).get("totalCount", 0),
        #                 "starred_count": node.get("starredRepositories", {}).get("totalCount", 0)
        #             }
        #         })
        #
        #     if not page_info.get("hasNextPage", False) or not page_info.get("endCursor"):
        #         break
        #     cursor = f'"{page_info.get("endCursor")}"'
        # return stars
        return [] # Bypassed

    def _get_stargazers_rest(self, owner: str, repo: str, get_timestamps: bool = True) -> List[Dict]:
        """Get repository stargazers using REST API (fallback)."""
        headers = self.headers.copy()
        if get_timestamps:
            headers["Accept"] = "application/vnd.github.v3.star+json"

        endpoint = f"/repos/{owner}/{repo}/stargazers"
        # For REST API, pagination for stargazers is typically done with a custom session
        # but self.paginate should handle it.
        
        raw_stars = self.paginate(endpoint) # self.paginate already uses self.session which has headers

        processed_stars = []
        for star_entry in raw_stars:
            if get_timestamps and "starred_at" in star_entry and "user" in star_entry:
                # Format to match structure expected by callers (similar to GraphQL output)
                user_data = star_entry["user"]
                processed_star = {
                    "starred_at": star_entry["starred_at"],
                    "user": {
                        "login": user_data.get("login"),
                        "avatar_url": user_data.get("avatar_url"),
                        # These will be filled by get_user if needed later
                        "created_at": None, 
                        "followers_count": None,
                        "public_repos": None,
                        "starred_count": None 
                    }
                }
            elif "login" in star_entry: # Fallback for non-timestamped structure (user objects directly in list)
                 processed_star = {
                    "starred_at": "2020-01-01T00:00:00Z",  # Placeholder if not using star+json
                    "user": {
                        "login": star_entry.get("login"),
                        "avatar_url": star_entry.get("avatar_url"),
                        "created_at": None,
                        "followers_count": None,
                        "public_repos": None,
                        "starred_count": None
                    }
                }
            else:
                logger.debug(f"Skipping malformed star entry: {star_entry}")
                continue # Skip malformed entries
            
            processed_stars.append(processed_star)

        # Augment with detailed user profile info (this part is expensive)
        # This is done by BurstDetector.score_stargazers if needed for specific users.
        # To keep this method focused, we don't do it here for all users.
        # Callers like BurstDetector will fetch full profiles for users they analyze.
        
        return processed_stars

    def get_forks(self, owner: str, repo: str) -> List[Dict]:
        """Get repository forks."""
        endpoint = f"/repos/{owner}/{repo}/forks"
        return self.paginate(endpoint)

    def get_issues(self, owner: str, repo: str, state: str = "all") -> List[Dict]:
        """Get repository issues."""
        endpoint = f"/repos/{owner}/{repo}/issues"
        return self.paginate(endpoint, {"state": state})

    def get_pulls(self, owner: str, repo: str, state: str = "all") -> List[Dict]:
        """Get repository pull requests."""
        endpoint = f"/repos/{owner}/{repo}/pulls"
        return self.paginate(endpoint, {"state": state})

    def get_contributors(self, owner: str, repo: str) -> List[Dict]:
        """Get repository contributors."""
        endpoint = f"/repos/{owner}/{repo}/contributors"
        return self.paginate(endpoint)

    def get_commits(self, owner: str, repo: str, since: str = None, until: str = None) -> List[Dict]:
        """Get repository commits."""
        endpoint = f"/repos/{owner}/{repo}/commits"
        params = {}
        if since:
            params["since"] = since
        if until:
            params["until"] = until
        return self.paginate(endpoint, params=params)

    def get_traffic_views(self, owner: str, repo: str) -> Dict:
        """Get repository traffic views (requires push access)."""
        endpoint = f"/repos/{owner}/{repo}/traffic/views"
        data = self.request(endpoint)
        if "error" in data: # Handles 403 if no push access, or other errors
            logger.debug(f"Could not fetch traffic views for {owner}/{repo}: {data.get('error')}")
            return {"count": 0, "uniques": 0, "views": []}
        return data

    def get_traffic_clones(self, owner: str, repo: str) -> Dict:
        """Get repository traffic clones (requires push access)."""
        endpoint = f"/repos/{owner}/{repo}/traffic/clones"
        data = self.request(endpoint)
        if "error" in data:
            logger.debug(f"Could not fetch traffic clones for {owner}/{repo}: {data.get('error')}")
            return {"count": 0, "uniques": 0, "clones": []}
        return data
        
    def get_releases(self, owner: str, repo: str) -> List[Dict]:
        """Get repository releases."""
        endpoint = f"/repos/{owner}/{repo}/releases"
        return self.paginate(endpoint)

    def get_user(self, username: str) -> Dict:
        """Get a user's profile information."""
        if username in self.user_profile_cache:
            return self.user_profile_cache[username]

        endpoint = f"/users/{username}"
        user_data = self.request(endpoint)
        
        if "error" in user_data: # Handles 404 or other errors
            logger.debug(f"Error fetching user {username}: {user_data.get('error')}")
            return {} # Return empty if error or not found
            
        self.user_profile_cache[username] = user_data
        return user_data

    def get_user_events(self, username: str, limit: int = 10) -> List[Dict]:
        """Get a user's public events (limited to conserve API quota)."""
        endpoint = f"/users/{username}/events/public"
        params = {"per_page": min(limit, 100)}
        events_data = self.request(endpoint, params=params)

        if "error" in events_data or not isinstance(events_data, list):
            logger.debug(f"Error fetching events for user {username}: {events_data.get('error', 'Not a list')}")
            return []
        return events_data[:limit]

    def check_user_repo_interaction(self, owner: str, repo: str, username: str) -> Dict:
        """Check if a user has interacted with a repository (issues, PRs, commits)."""
        interactions = {
            "has_issues": False,
            "has_prs": False,
            "has_commits": False,
            "has_any_interaction": False
        }

        try:
            # Check for issues
            issues_endpoint = f"/search/issues"
            params_issues = {"q": f"repo:{owner}/{repo} author:{username} type:issue", "per_page": 1}
            issues_result = self.request(issues_endpoint, params=params_issues)
            if "total_count" in issues_result:
                interactions["has_issues"] = issues_result["total_count"] > 0

            # Check for PRs
            params_prs = {"q": f"repo:{owner}/{repo} author:{username} type:pr", "per_page": 1}
            prs_result = self.request(issues_endpoint, params=params_prs) # Re-use issues_endpoint for search
            if "total_count" in prs_result:
                interactions["has_prs"] = prs_result["total_count"] > 0

            # Check for commits (recent commits)
            commits_endpoint = f"/repos/{owner}/{repo}/commits"
            params_commits = {"author": username, "per_page": 1}
            commits_result = self.request(commits_endpoint, params=params_commits)
            if isinstance(commits_result, list): # Check if it's a list (successful) vs error dict
                 interactions["has_commits"] = len(commits_result) > 0
        except Exception as e:
            logger.debug(f"Error checking user interaction for {username} on {owner}/{repo}: {e}")
        
        interactions["has_any_interaction"] = (
            interactions["has_issues"] or
            interactions["has_prs"] or
            interactions["has_commits"]
        )
        return interactions

    def get_file_content(self, owner: str, repo: str, path: str, ref: str = None) -> Optional[str]:
        """Get file content from a repository."""
        endpoint = f"/repos/{owner}/{repo}/contents/{path}"
        params = {}
        if ref:
            params["ref"] = ref

        response_data = self.request(endpoint, params=params)
        if "error" in response_data or "content" not in response_data:
            logger.debug(f"Could not fetch file content for {path} in {owner}/{repo}: {response_data.get('error', 'No content field')}")
            return None
        
        try:
            import base64
            return base64.b64decode(response_data["content"]).decode("utf-8")
        except (UnicodeDecodeError, base64.binascii.Error) as e:
            logger.debug(f"Error decoding file content for {path}: {e}")
            return None


    def get_dependencies(self, owner: str, repo: str) -> Dict:
        """Get repository dependencies using the dependency graph API or file parsing."""
        endpoint = f"/repos/{owner}/{repo}/dependency-graph/sbom"
        try:
            # Dependency graph API requires specific permissions and may not always be enabled/available.
            response = self.request(endpoint)
            if "error" not in response and "sbom" in response: # Check for valid response
                return response
            logger.warning(f"Could not fetch dependencies via API for {owner}/{repo} (Reason: {response.get('error', 'No SBOM field')}). Falling back to file parsing.")
        except Exception as e: # Catch any exception during the API call itself
            logger.warning(f"Exception fetching dependencies via API for {owner}/{repo}: {str(e)}. Falling back to file parsing.")
        
        # Fallback to parsing manifest files
        return self._parse_dependencies_from_files(owner, repo)


    def _parse_dependencies_from_files(self, owner: str, repo: str) -> Dict:
        """Parse dependencies from manifest files."""
        dependencies = {}

        for lang, config in PACKAGE_MANAGERS.items():
            try:
                content = self.get_file_content(owner, repo, config["file"])
                if content:
                    deps = self._parse_dependency_file(content, config, lang)
                    if deps:
                        dependencies[lang] = deps
            except Exception as e:
                logger.debug(f"Could not parse {config['file']} for {owner}/{repo}: {str(e)}")
        return dependencies

    def _parse_dependency_file(self, content: str, config: Dict, lang: str) -> List[Dict]:
        """Parse a dependency file based on its format."""
        deps = []
        if not content: return deps

        if lang == "javascript":
            try:
                package_json = json.loads(content)
                for key in config["dependencies_key"]:
                    if key in package_json and isinstance(package_json[key], dict):
                        for name, version in package_json[key].items():
                            deps.append({"name": name, "version": str(version), "type": "runtime" if key == "dependencies" else "development"})
            except json.JSONDecodeError as e:
                logger.debug(f"JSONDecodeError parsing package.json: {e}")
                pass

        elif lang == "python": # requirements.txt
            pattern = re.compile(config["pattern"])
            for line in content.splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    match = pattern.match(line)
                    if match:
                        name = match.group(1)
                        version = match.group(2) if match.group(2) else "latest"
                        deps.append({"name": name, "version": version, "type": "runtime"})

        elif lang == "python_pipfile": # Pipfile
            in_section = False
            for line in content.splitlines():
                line = line.strip()
                if config["section_marker"] in line:
                    in_section = True
                    continue
                if line.startswith("[") and in_section: # Reached another section
                    in_section = False
                    continue
                if in_section and not line.startswith("#") and "=" in line:
                    parts = line.split("=", 1)
                    name = parts[0].strip()
                    version = parts[1].strip().strip('"\'')
                    deps.append({"name": name, "version": version, "type": "runtime"})
        
        elif lang == "python_poetry": # pyproject.toml
            in_section = False
            for line in content.splitlines():
                line = line.strip()
                if config["section_marker"] in line:
                    in_section = True
                    continue
                if line.startswith("[") and in_section: # Reached another section
                    in_section = False
                    continue
                if in_section and not line.startswith("#") and "=" in line:
                    parts = line.split("=", 1)
                    name = parts[0].strip().strip('"\'') # Name can be quoted
                    version_spec = parts[1].strip()
                    # Extract version, handling complex dicts like {version = "^1.0", optional = true}
                    if version_spec.startswith("{"):
                        try:
                            # This is a simplified TOML dict parsing. A full TOML parser would be better.
                            if "version" in version_spec:
                                v_match = re.search(r"version\s*=\s*[\"']([^\"']+)[\"']", version_spec)
                                version = v_match.group(1) if v_match else "latest"
                            else: # Or just take the whole string as a specifier
                                version = version_spec.strip('"\'')
                        except Exception:
                            version = version_spec.strip('"\'') # Fallback
                    else:
                        version = version_spec.strip('"\'')
                    deps.append({"name": name, "version": version, "type": "runtime"})


        elif lang == "ruby": # Gemfile
            pattern = re.compile(config["pattern"])
            for line in content.splitlines():
                line = line.strip()
                if line.startswith("gem"):
                    match = pattern.search(line)
                    if match:
                        name = match.group(1)
                        version = match.group(2) if len(match.groups()) > 1 and match.group(2) else "latest"
                        deps.append({"name": name, "version": version, "type": "runtime"})

        elif lang == "java": # pom.xml
            # This requires XML parsing, regex is fragile. Using it for simplicity.
            pattern = re.compile(config["tag_pattern"], re.DOTALL | re.IGNORECASE)
            matches = pattern.findall(content)
            for match_tuple in matches:
                # Ensure match_tuple has enough elements, handling optional version
                group_id = match_tuple[0]
                artifact_id = match_tuple[1]
                version = match_tuple[2] if len(match_tuple) > 2 and match_tuple[2] else "latest"
                deps.append({"name": f"{group_id}:{artifact_id}", "version": version, "type": "runtime"})

        elif lang == "go": # go.mod
            pattern = re.compile(config["pattern"])
            in_require_block = False
            for line in content.splitlines():
                line = line.strip()
                if line.startswith("require ("):
                    in_require_block = True
                    continue
                if line.startswith(")") and in_require_block:
                    in_require_block = False
                    continue
                
                if line.startswith("require") and not in_require_block: # single line require
                    match = pattern.match(line)
                    if match:
                        name, version = match.group(1), match.group(2)
                        deps.append({"name": name, "version": version, "type": "runtime"})
                elif in_require_block: # inside multi-line require block
                    parts = line.split()
                    if len(parts) == 2: # e.g., "github.com/user/repo v1.2.3"
                        name, version = parts[0], parts[1]
                        if version.startswith("v"): version = version[1:] # Strip leading 'v'
                        deps.append({"name": name, "version": version, "type": "runtime"})

        return deps

    def get_license(self, owner: str, repo: str) -> Dict:
        """Get repository license information."""
        endpoint = f"/repos/{owner}/{repo}/license"
        # This request often returns 404 if no license file is detected by GitHub.
        # The request() method returns an error dict for 404.
        license_data = self.request(endpoint)
        
        if "error" in license_data and license_data.get("status_code") == 404:
            logger.info(f"No license file found by GitHub API for {owner}/{repo}.")
            return {"license": {"spdx_id": "unknown", "key": "unknown", "name": "Unknown"}}
        elif "error" in license_data:
            logger.warning(f"Error fetching license for {owner}/{repo}: {license_data.get('error')}")
            return {"license": {"spdx_id": "unknown", "key": "unknown", "name": "Unknown"}}
        
        # Ensure the "license" key itself exists and is a dict, and "key" exists within it.
        if "license" not in license_data or not isinstance(license_data["license"], dict):
            logger.warning(f"Malformed license data for {owner}/{repo}: {license_data}")
            return {"license": {"spdx_id": "unknown", "key": "unknown", "name": "Unknown"}}
        
        # Ensure spdx_id and key are present, default to "unknown"
        if "spdx_id" not in license_data["license"]:
            license_data["license"]["spdx_id"] = "unknown"
        if "key" not in license_data["license"]:
            license_data["license"]["key"] = "unknown"
        if "name" not in license_data["license"]:
            license_data["license"]["name"] = "Unknown"
            
        return license_data


class BurstDetector:
    """
    Detects star burst patterns using robust statistical methods.
    Uses Median Absolute Deviation (MAD) which is more robust to outliers than Z-score.
    """

    def __init__(self, owner: str, repo: str, github_api: GitHubAPI, window_size: int = WINDOW_SIZE):
        self.owner = owner
        self.repo = repo
        self.github_api = github_api
        self.window_size = window_size
        self.stars_df = None
        self.bursts = []

    def build_daily_timeseries(self, days_limit: int = 0) -> pd.DataFrame:
        """Build a daily time series of stars from the GitHub API."""
        try:
            stars_data = self.github_api.get_stargazers(self.owner, self.repo, get_timestamps=True, days_limit=days_limit)
            if not stars_data:
                logger.warning(f"No star data found for {self.owner}/{self.repo} within {days_limit} days.")
                return pd.DataFrame(columns=["date", "stars", "users", "day"]) # Add "users" and "day"

            records = []
            for star in stars_data:
                try:
                    if "starred_at" not in star or not star["starred_at"]:
                        continue
                    username = star.get("user", {}).get("login")
                    
                    # Ensure starred_at is parsed correctly and made naive for consistency
                    starred_at_dt = make_naive_datetime(parse_date(star["starred_at"]))
                    if starred_at_dt: # Check if parse_date was successful
                         records.append({
                            "date": starred_at_dt.date(), # Group by date object
                            "datetime": starred_at_dt,
                            "username": username
                        })
                except Exception as e:
                    logger.debug(f"Error processing star data entry: {star}. Error: {str(e)}")
            
            if not records:
                logger.warning(f"No valid star records after parsing for {self.owner}/{self.repo}.")
                return pd.DataFrame(columns=["date", "stars", "users", "day"])

            df = pd.DataFrame(records)
            df["day"] = pd.to_datetime(df["date"]) # This converts date objects to Timestamps (datetime64[ns])

            # Group by original date objects for daily_counts and daily_users
            daily_counts = df.groupby("date").size().reset_index(name="stars")
            daily_users = df.groupby("date")["username"].apply(list).reset_index(name="users")
            
            result_df = pd.merge(daily_counts, daily_users, on="date", how="outer")
            
            # Ensure 'date' column in result_df is also datetime64[ns] for merge with all_dates
            result_df["date"] = pd.to_datetime(result_df["date"]) 

            # Fill date gaps
            if not df.empty:
                min_date_ts = df["day"].min() # Use 'day' which is Timestamp
                max_date_ts = df["day"].max() # Use 'day' which is Timestamp
                
                # Create a full date range of Timestamps
                all_dates_df = pd.DataFrame({
                    "date": pd.date_range(start=min_date_ts, end=max_date_ts, freq="D")
                })
                # Merge with all_dates_df. 'date' column is now Timestamp in both.
                merged_df = pd.merge(all_dates_df, result_df, on="date", how="left")
            else: # df was empty, so result_df is also empty or near empty
                merged_df = result_df # or an empty df with correct columns

            merged_df["stars"] = merged_df["stars"].fillna(0).astype(int) # Fixed FutureWarning & ensure int
            merged_df["users"] = merged_df["users"].apply(lambda x: x if isinstance(x, list) else [])
            
            merged_df.sort_values("date", inplace=True)
            
            # Convert 'date' column back to date objects for consistency if other parts expect it
            # Or ensure all date handling uses Timestamps. For self.stars_df, Timestamp is fine.
            # The plotting function uses self.stars_df["date"], which will be Timestamp.
            # Burst detection logic uses row["date"], which will be Timestamp.
            # Cross-check logic converts date objects to strings, so Timestamp.date() or Timestamp.isoformat() works.
            
            self.stars_df = merged_df
            return merged_df

        except Exception as e:
            logger.error(f"Error building timeseries for {self.owner}/{self.repo}: {str(e)}", exc_info=True)
            return pd.DataFrame(columns=["date", "stars", "users", "day"])


    def detect_bursts(self) -> List[Dict]:
        """Detect star bursts using MAD."""
        try:
            if self.stars_df is None or self.stars_df.empty:
                # Try building timeseries with a default limit if not already built
                self.stars_df = self.build_daily_timeseries() 
            if self.stars_df.empty:
                logger.info(f"No star timeseries data to detect bursts for {self.owner}/{self.repo}.")
                return []

            df_analysis = self.stars_df.copy()
            if 'stars' not in df_analysis.columns or df_analysis['stars'].isnull().all():
                logger.warning(f"Stars column missing or all null in timeseries for {self.owner}/{self.repo}")
                return []

            df_analysis["median"] = np.nan
            df_analysis["mad"] = np.nan
            df_analysis["is_anomaly"] = False
            
            # Date column in df_analysis is Timestamp objects. Access .date() for date objects if needed.
            # Loop from self.window_size implies index-based access.
            for i in range(self.window_size, len(df_analysis)):
                window_data = df_analysis.iloc[i - self.window_size:i]["stars"]
                if not window_data.empty:
                    median_val = window_data.median()
                    df_analysis.loc[df_analysis.index[i], "median"] = median_val
                    df_analysis.loc[df_analysis.index[i], "mad"] = (window_data - median_val).abs().median()
            
            # Fill NaNs that might remain if window was too short at start
            df_analysis["median"] = df_analysis["median"].bfill()  
            df_analysis["mad"] = df_analysis["mad"].bfill()
            df_analysis["mad"] = df_analysis["mad"].fillna(0) 

            for i in range(len(df_analysis)): # Check all days, not just starting from window_size
                # Use .loc with index for setting values
                idx = df_analysis.index[i]
                median = df_analysis.loc[idx, "median"]
                mad = df_analysis.loc[idx, "mad"]
                stars_today = df_analysis.loc[idx, "stars"]
                
                # total_stars up to *before* current day
                total_stars_before_today = df_analysis.iloc[:i]["stars"].sum() if i > 0 else 0

                is_anomaly_flag = False
                if pd.isna(median) or pd.isna(mad): # Handle cases where median/mad couldn't be computed
                    # For early days before full window, or if data was sparse
                    if stars_today > 0 and total_stars_before_today < MIN_STAR_COUNT: # Small repo, early phase
                         percent_increase = (stars_today / max(1, total_stars_before_today)) * 100
                         if percent_increase > MIN_STARS_GROWTH_PERCENT and stars_today > 5: # Min 5 stars for a spike
                             is_anomaly_flag = True
                elif total_stars_before_today >= MIN_STAR_COUNT and mad > 0.001: # MAD > 0 (avoid MAD=0 issues)
                    threshold = median + MAD_THRESHOLD * mad
                    if stars_today > threshold and stars_today > median + 1: # Ensure it's meaningfully above median
                        is_anomaly_flag = True
                elif stars_today > 0: # Small repo or MAD is zero (low variance period)
                    # Use percentage growth relative to historical or absolute jump
                    percent_increase = (stars_today / max(1, total_stars_before_today)) * 100
                    # Check if stars_today is significantly more than recent median
                    significant_jump = stars_today > max(5, median * 2) # e.g. >5 stars and double the median
                    if (percent_increase > MIN_STARS_GROWTH_PERCENT and stars_today > 5) or significant_jump :
                        is_anomaly_flag = True
                
                df_analysis.loc[idx, "is_anomaly"] = is_anomaly_flag

            bursts = []
            in_burst = False
            current_burst_start_date = None
            current_burst_users = []
            current_burst_star_count = 0

            for _, row in df_analysis.iterrows():
                # row["date"] is a Timestamp object from self.stars_df
                current_date_obj = row["date"].to_pydatetime().date() # Convert to datetime.date for consistency in burst dict

                if row["is_anomaly"] and not in_burst:
                    in_burst = True
                    current_burst_start_date = current_date_obj
                    current_burst_users = list(row["users"]) if isinstance(row["users"], list) else []
                    current_burst_star_count = row["stars"]
                elif row["is_anomaly"] and in_burst:
                    if isinstance(row["users"], list):
                        current_burst_users.extend(row["users"])
                    current_burst_star_count += row["stars"]
                elif not row["is_anomaly"] and in_burst:
                    in_burst = False
                    # Use previous day's date as burst_end_date
                    # Find the date of previous row in df_analysis
                    prev_row_date_obj = df_analysis.loc[df_analysis[df_analysis['date'] < row['date']].index[-1], 'date'].to_pydatetime().date()

                    bursts.append({
                        "start_date": current_burst_start_date,
                        "end_date": prev_row_date_obj, # End date is the last anomalous day
                        "days": (prev_row_date_obj - current_burst_start_date).days + 1,
                        "stars": int(current_burst_star_count),
                        "users": list(set(current_burst_users))
                    })
                    # Reset for next potential burst
                    current_burst_users = []
                    current_burst_star_count = 0
                    current_burst_start_date = None


            if in_burst: # If loop ends while in a burst
                last_date_obj = df_analysis.iloc[-1]["date"].to_pydatetime().date()
                bursts.append({
                    "start_date": current_burst_start_date,
                    "end_date": last_date_obj,
                    "days": (last_date_obj - current_burst_start_date).days + 1,
                    "stars": int(current_burst_star_count),
                    "users": list(set(current_burst_users))
                })

            self.bursts = bursts
            return bursts

        except Exception as e:
            logger.error(f"Error detecting bursts for {self.owner}/{self.repo}: {str(e)}", exc_info=True)
            if self.stars_df is not None:
                logger.debug(f"Stars dataframe for {self.owner}/{self.repo} has {len(self.stars_df)} rows. Columns: {self.stars_df.columns}")
            return []

    def cross_check_bursts(self) -> List[Dict]:
        """Perform consistency cross-checks on detected bursts."""
        try:
            if not self.bursts: # If empty, try to detect them first
                self.detect_bursts()
            if not self.bursts: # If still empty, return empty
                return []

            enhanced_bursts = []
            for burst in self.bursts:
                # Ensure start_date and end_date are datetime.date objects
                burst_start_date_obj = burst["start_date"]
                burst_end_date_obj = burst["end_date"]

                # Cross-check 1: Fork delta
                fork_delta, fork_ratio, flag_forks = 0, 0, True
                try:
                    forks = self.github_api.get_forks(self.owner, self.repo)
                    forks_in_burst_period = [
                        f for f in forks 
                        if f.get("created_at") and 
                           burst_start_date_obj <= make_naive_datetime(parse_date(f["created_at"])).date() <= burst_end_date_obj
                    ]
                    fork_delta = len(forks_in_burst_period)
                    fork_ratio = fork_delta / burst["stars"] if burst["stars"] > 0 else 0
                    flag_forks = fork_ratio < 0.01 # Suspicious if very few forks per star
                except Exception as e:
                    logger.debug(f"Error checking forks for burst {burst_start_date_obj}-{burst_end_date_obj}: {str(e)}")

                # Cross-check 2: Issue + PR delta
                issue_delta, pr_delta, issue_pr_delta, flag_issues_prs = 0,0,0,True
                try:
                    issues = self.github_api.get_issues(self.owner, self.repo, state="all") # Get all states
                    issues_in_burst = [
                        i for i in issues 
                        if i.get("created_at") and
                           burst_start_date_obj <= make_naive_datetime(parse_date(i["created_at"])).date() <= burst_end_date_obj
                    ]
                    issue_delta = len(issues_in_burst)

                    prs = self.github_api.get_pulls(self.owner, self.repo, state="all")
                    prs_in_burst = [
                        p for p in prs
                        if p.get("created_at") and
                           burst_start_date_obj <= make_naive_datetime(parse_date(p["created_at"])).date() <= burst_end_date_obj
                    ]
                    pr_delta = len(prs_in_burst)
                    issue_pr_delta = issue_delta + pr_delta
                    flag_issues_prs = issue_pr_delta == 0 and burst["stars"] > 10 # No interaction during a sizable burst
                except Exception as e:
                    logger.debug(f"Error checking issues/PRs for burst {burst_start_date_obj}-{burst_end_date_obj}: {str(e)}")
                
                # Cross-check 3: Traffic views (if available)
                views_delta, traffic_ratio, flag_traffic = 0,0,True # Default to suspicious if no data
                try:
                    # Traffic data is for past 14 days. May not align with older bursts.
                    # For simplicity, we just check if *any* traffic data is available from the API.
                    traffic_views_data = self.github_api.get_traffic_views(self.owner, self.repo)
                    if traffic_views_data and traffic_views_data.get("views"):
                        for view_entry in traffic_views_data["views"]:
                            view_date = make_naive_datetime(parse_date(view_entry["timestamp"])).date()
                            if burst_start_date_obj <= view_date <= burst_end_date_obj:
                                views_delta += view_entry["count"]
                        
                        if views_delta > 0 : # Only if we have view data for the period
                            traffic_ratio = views_delta / burst["stars"] if burst["stars"] > 0 else 0
                            # Flag if views are less than stars (e.g. 1 view per star is low for organic)
                            flag_traffic = traffic_ratio < 1.0 and burst["stars"] > 10 
                        # If no views in burst period but traffic API worked, it's suspicious.
                        # If traffic API failed, it remains flag_traffic = True (suspicious by default)
                    else: # No traffic data from API (e.g. permissions)
                        logger.debug(f"No traffic view data available for {self.owner}/{self.repo}")

                except Exception as e:
                    logger.debug(f"Error checking traffic for burst {burst_start_date_obj}-{burst_end_date_obj}: {str(e)}")

                # Cross-check 4: Commits / releases around the burst
                has_commits, has_release, flag_activity = False, False, True
                try:
                    # Check commits/releases in a window around the burst
                    window_start = burst_start_date_obj - datetime.timedelta(days=7)
                    window_end = burst_end_date_obj + datetime.timedelta(days=7)
                    
                    commits = self.github_api.get_commits(self.owner, self.repo, 
                                                          since=window_start.isoformat(), 
                                                          until=window_end.isoformat())
                    has_commits = len(commits) > 0

                    releases = self.github_api.get_releases(self.owner, self.repo)
                    has_release = any(
                        r.get("published_at") and
                        window_start <= make_naive_datetime(parse_date(r["published_at"])).date() <= window_end 
                        for r in releases
                    )
                    flag_activity = not (has_commits or has_release) and burst["stars"] > 20 # No activity around a significant burst
                except Exception as e:
                    logger.debug(f"Error checking activity for burst {burst_start_date_obj}-{burst_end_date_obj}: {str(e)}")

                rule_hits = sum([
                    1 if flag_forks else 0,
                    1 if flag_issues_prs else 0,
                    1 if flag_traffic else 0, # Traffic data might not be available
                    1 if flag_activity else 0
                ])

                enhanced_burst = burst.copy()
                enhanced_burst.update({
                    "cross_checks": {
                        "fork_delta": fork_delta, "fork_ratio": fork_ratio, "flag_forks": flag_forks,
                        "issue_delta": issue_delta, "pr_delta": pr_delta, "issue_pr_delta": issue_pr_delta, "flag_issues_prs": flag_issues_prs,
                        "views_delta": views_delta, "traffic_ratio": traffic_ratio, "flag_traffic": flag_traffic,
                        "has_commits_around_burst": has_commits, "has_release_around_burst": has_release, "flag_activity": flag_activity
                    },
                    "rule_hits": rule_hits,
                    "inorganic_heuristic": rule_hits >= 2 # Adjusted heuristic based on available data
                })
                enhanced_bursts.append(enhanced_burst)
            
            self.bursts = enhanced_bursts
            return enhanced_bursts

        except Exception as e:
            logger.error(f"Error cross-checking bursts for {self.owner}/{self.repo}: {str(e)}", exc_info=True)
            return self.bursts # Return original if error

    def score_stargazers(self, max_users_to_score_per_burst: int = 10000, max_total_users_to_score: int = 10000) -> Dict:
        """Score stargazers in burst windows to identify likely fake accounts."""
        try:
            if not self.bursts: self.cross_check_bursts()
            if not self.bursts: return {"bursts": [], "user_scores": {}}

            all_user_scores = {}
            scored_bursts_output = []
            total_users_scored_so_far = 0

            # Prioritize scoring users from more suspicious bursts
            # Sort bursts by a suspicion score (e.g., rule_hits desc, then stars desc)
            sorted_bursts_for_scoring = sorted(
                self.bursts, 
                key=lambda b: (b.get("inorganic_heuristic", False), b.get("rule_hits", 0), b.get("stars", 0)),
                reverse=True
            )

            for burst_idx, burst in enumerate(sorted_bursts_for_scoring):
                if total_users_scored_so_far >= max_total_users_to_score:
                    logger.info(f"Reached max total users to score ({max_total_users_to_score}). Skipping further user scoring.")
                    # Add remaining bursts without user scores, or with minimal info
                    burst_copy = burst.copy()
                    burst_copy.update({
                        "user_scores": {}, "likely_fake_users": [], "likely_fake_count": 0,
                        "sampled_users_count": 0, "fake_ratio": 0,
                        "scoring_skipped": True
                    })
                    scored_bursts_output.append(burst_copy)
                    continue

                users_in_burst = burst.get("users", [])
                if not users_in_burst:
                    scored_bursts_output.append(burst) # Add as is if no users
                    continue

                # Sample users from this burst
                users_to_score_this_burst = users_in_burst
                if len(users_in_burst) > max_users_to_score_per_burst:
                    users_to_score_this_burst = random.sample(users_in_burst, max_users_to_score_per_burst)
                
                # Further limit if close to max_total_users_to_score
                remaining_total_slots = max_total_users_to_score - total_users_scored_so_far
                if len(users_to_score_this_burst) > remaining_total_slots:
                    users_to_score_this_burst = users_to_score_this_burst[:remaining_total_slots]


                burst_user_evals = {} # Renamed from burst_user_scores to avoid confusion with final score
                likely_fake_usernames_in_burst = []

                desc = f"Scoring users in burst {burst_idx+1}/{len(sorted_bursts_for_scoring)} ({burst['start_date']})"
                for username in tqdm(users_to_score_this_burst, desc=desc, disable=len(users_to_score_this_burst) < 10):
                    if not username: continue # Skip if username is None or empty

                    if username in all_user_scores: # Already scored globally
                        user_eval = all_user_scores[username]
                    else:
                        user_profile = self.github_api.get_user(username)
                        if not user_profile or "login" not in user_profile: # Ensure profile is valid
                            logger.debug(f"Skipping scoring for {username}, profile not found or invalid.")
                            continue 
                        
                        score_breakdown = {}
                        # 1. Account age
                        account_age_days_val = None
                        if user_profile.get("created_at"):
                            created_at_dt = make_naive_datetime(parse_date(user_profile["created_at"]))
                            if created_at_dt:
                                account_age_days_val = (make_naive_datetime(datetime.datetime.now()) - created_at_dt).days
                                age_thresh, age_score = USER_SCORE_THRESHOLDS["account_age_days"]
                                score_breakdown["account_age"] = age_score if account_age_days_val < age_thresh else 0
                        
                        # 2. Followers
                        followers_val = user_profile.get("followers", 0)
                        foll_thresh, foll_score = USER_SCORE_THRESHOLDS["followers"]
                        score_breakdown["followers"] = foll_score if followers_val < foll_thresh else 0
                        
                        # 3. Public repos
                        pub_repos_val = user_profile.get("public_repos", 0)
                        repo_thresh, repo_score = USER_SCORE_THRESHOLDS["public_repos"]
                        score_breakdown["public_repos"] = repo_score if pub_repos_val < repo_thresh else 0

                        # 4. User's total starred repos (expensive, uses GraphQL)
                        user_total_stars_val = None
                        # Check if other scores already push this over the threshold to save API call
                        current_score_sum = sum(score_breakdown.values())
                        stars_component_max_score = USER_SCORE_THRESHOLDS["total_stars"][1]

                        if current_score_sum + stars_component_max_score < FAKE_USER_THRESHOLD : # Only query if it can make a difference
                            gql_query_user_stars = f"""query {{ user(login: "{username}") {{ starredRepositories {{ totalCount }} }} }}"""
                            gql_result = self.github_api.graphql_request(gql_query_user_stars)
                            if gql_result and not gql_result.get("errors") and gql_result.get("data", {}).get("user"):
                                user_total_stars_val = gql_result["data"]["user"].get("starredRepositories", {}).get("totalCount")
                        
                        if user_total_stars_val is not None:
                            star_thresh, star_score_val = USER_SCORE_THRESHOLDS["total_stars"]
                            score_breakdown["total_stars"] = star_score_val if user_total_stars_val < star_thresh else 0
                        else: # If GraphQL failed or skipped
                             score_breakdown["total_stars"] = 0 # Neutral or slightly suspicious if cannot fetch

                        # 5. Prior interaction with THIS repo
                        interaction_data = self.github_api.check_user_repo_interaction(self.owner, self.repo, username)
                        has_prior_interaction = interaction_data.get("has_any_interaction", False)
                        _, interact_score = USER_SCORE_THRESHOLDS["prior_interaction"]
                        score_breakdown["prior_interaction"] = interact_score if not has_prior_interaction else 0
                        
                        # 6. Default avatar
                        has_default_avatar_flag = "avatar_url" in user_profile and \
                                                  ("gravatar.com/avatar/00000000000000000000000000000000" in user_profile["avatar_url"] or \
                                                   "avatars.githubusercontent.com/u/0?" in user_profile["avatar_url"] or \
                                                   "identicons" in user_profile["avatar_url"] or # Common for default
                                                   "no-avatar" in user_profile["avatar_url"]) # Check for common default patterns
                        _, avatar_score = USER_SCORE_THRESHOLDS["default_avatar"]
                        score_breakdown["default_avatar"] = avatar_score if has_default_avatar_flag else 0

                        final_total_score = sum(score_breakdown.values())
                        user_eval = {
                            "username": username,
                            "account_age_days": account_age_days_val,
                            "followers": followers_val,
                            "public_repos": pub_repos_val,
                            "total_stars_by_user": user_total_stars_val, # Renamed for clarity
                            "has_interaction_with_repo": has_prior_interaction,
                            "has_default_avatar": has_default_avatar_flag,
                            "score_components": score_breakdown,
                            "total_score": final_total_score,
                            "likely_fake_profile": final_total_score >= FAKE_USER_THRESHOLD
                        }
                        all_user_scores[username] = user_eval
                    
                    burst_user_evals[username] = user_eval
                    if user_eval["likely_fake_profile"]:
                        likely_fake_usernames_in_burst.append(username)
                    total_users_scored_so_far +=1
                    if total_users_scored_so_far >= max_total_users_to_score: break # Break inner loop too

                burst_copy = burst.copy() # Work on a copy
                burst_copy.update({
                    "user_evaluations": burst_user_evals, # Store evaluations for this burst
                    "likely_fake_users_in_burst": likely_fake_usernames_in_burst,
                    "likely_fake_count_in_burst": len(likely_fake_usernames_in_burst),
                    "sampled_users_in_burst_count": len(burst_user_evals),
                    "fake_ratio_in_burst": len(likely_fake_usernames_in_burst) / len(burst_user_evals) if burst_user_evals else 0,
                    "scoring_skipped": False
                })
                scored_bursts_output.append(burst_copy)

            return {"bursts": scored_bursts_output, "user_scores_cache": all_user_scores}

        except Exception as e:
            logger.error(f"Error scoring stargazers for {self.owner}/{self.repo}: {str(e)}", exc_info=True)
            # Return original bursts, so subsequent steps don't fail on missing keys
            return {"bursts": self.bursts, "user_scores_cache": {}}


    def calculate_fake_star_index(self) -> Dict:
        """Calculate the Fake Star Index for the repository."""
        default_return = {
            "has_fake_stars": False, "fake_star_index": 0.0, "risk_level": "low",
            "bursts": [], "total_stars_analyzed": 0, "total_likely_fake": 0,
            "fake_percentage": 0.0, "worst_burst": None
        }
        try:
            scoring_result = self.score_stargazers() # This returns {"bursts": ..., "user_scores_cache": ...}
            processed_bursts = scoring_result.get("bursts", [])

            if not processed_bursts:
                logger.info(f"No bursts to analyze for fake star index for {self.owner}/{self.repo}.")
                default_return["bursts"] = self.bursts # Return original bursts if any
                return default_return

            final_bursts_with_scores = []
            for burst in processed_bursts:
                # Use the new keys from score_stargazers
                fake_ratio = burst.get("fake_ratio_in_burst", 0)
                # Normalize rule_hits (0-4 range) to 0-1. Max 4 rules.
                normalized_rule_hits = burst.get("rule_hits", 0) / 4.0 
                
                burst_score = (FAKE_RATIO_WEIGHT * fake_ratio) + (RULE_HITS_WEIGHT * normalized_rule_hits)
                burst_score = min(max(burst_score, 0.0), 1.0) # Clamp to 0-1

                verdict = "organic"
                if burst_score >= BURST_FAKE_THRESHOLD: verdict = "fake"
                elif burst_score >= BURST_ORGANIC_THRESHOLD: verdict = "suspicious"
                
                burst_copy = burst.copy()
                burst_copy.update({"burst_score": burst_score, "verdict": verdict})
                final_bursts_with_scores.append(burst_copy)

            # Repo-level metrics based on *all* stars in *all* detected bursts (not just scored ones)
            # total_stars_in_bursts is sum of stars in all bursts from self.bursts
            # total_likely_fake_estimate is based on scored bursts and extrapolated if needed
            
            total_stars_in_all_bursts = sum(b["stars"] for b in self.bursts if "stars" in b) # from original self.bursts
            
            # Calculate weighted sum of likely fake stars from *scored* bursts
            # Extrapolate if some bursts were not scored due to limits.
            estimated_total_fake_in_bursts = 0
            stars_in_scored_bursts = 0

            for b_scored in final_bursts_with_scores:
                if not b_scored.get("scoring_skipped", True) and "stars" in b_scored : # if it was scored
                    stars_in_scored_bursts += b_scored["stars"]
                    # Estimate fakes for this burst: ratio * stars_in_this_burst
                    estimated_total_fake_in_bursts += b_scored.get("fake_ratio_in_burst",0) * b_scored["stars"]

            # Overall fake percentage across scored bursts
            avg_fake_ratio_in_scored_bursts = (estimated_total_fake_in_bursts / stars_in_scored_bursts) \
                                              if stars_in_scored_bursts > 0 else 0
            
            # Extrapolate to all bursts if some were skipped
            if total_stars_in_all_bursts > stars_in_scored_bursts and stars_in_scored_bursts > 0:
                 stars_in_unscored_bursts = total_stars_in_all_bursts - stars_in_scored_bursts
                 estimated_total_fake_in_bursts += avg_fake_ratio_in_scored_bursts * stars_in_unscored_bursts
            
            # Ensure it's an integer
            estimated_total_fake_in_bursts = int(round(estimated_total_fake_in_bursts))


            repo_index_val = 0.0
            if total_stars_in_all_bursts > 0:
                # Weighted average of burst_score by stars in burst
                weighted_score_sum = sum(b.get("burst_score", 0) * b.get("stars",0) for b in final_bursts_with_scores if "stars" in b)
                repo_index_val = weighted_score_sum / total_stars_in_all_bursts
            repo_index_val = min(max(repo_index_val, 0.0), 1.0) # Clamp

            repo_risk_level = "low"
            if repo_index_val >= BURST_FAKE_THRESHOLD: repo_risk_level = "high"
            elif repo_index_val >= BURST_ORGANIC_THRESHOLD: repo_risk_level = "medium"
            
            worst_burst_obj = None
            if final_bursts_with_scores:
                worst_burst_obj = max(final_bursts_with_scores, key=lambda b: b.get("burst_score", 0), default=None)

            return {
                "has_fake_stars": repo_index_val > BURST_ORGANIC_THRESHOLD,
                "fake_star_index": repo_index_val,
                "risk_level": repo_risk_level,
                "total_stars_analyzed": total_stars_in_all_bursts, # Stars in all detected bursts
                "total_likely_fake": estimated_total_fake_in_bursts, # Estimated fakes across all bursts
                "fake_percentage": (estimated_total_fake_in_bursts / total_stars_in_all_bursts * 100) if total_stars_in_all_bursts > 0 else 0.0,
                "bursts": final_bursts_with_scores, # these are the processed ones
                "worst_burst": worst_burst_obj
            }

        except Exception as e:
            logger.error(f"Error calculating fake star index for {self.owner}/{self.repo}: {str(e)}", exc_info=True)
            default_return["error"] = str(e)
            default_return["bursts"] = self.bursts # Return original bursts if error during processing
            return default_return


    def plot_star_history(self, save_path: str = None) -> None:
        """Plot star history with burst windows and anomalies highlighted."""
        try:
            if self.stars_df is None or self.stars_df.empty:
                logger.info(f"Attempting to build timeseries for plotting {self.owner}/{self.repo}")
                self.stars_df = self.build_daily_timeseries() # Ensure data is loaded

            if self.stars_df.empty:
                logger.warning(f"No star data to plot for {self.owner}/{self.repo}")
                return

            plt.figure(figsize=(15, 8))
            ax = plt.gca()
            
            # Ensure 'date' is plottable (datetime-like) and 'stars' are numeric
            plot_dates = pd.to_datetime(self.stars_df["date"]) # Ensure datetime for plotting
            plot_stars = pd.to_numeric(self.stars_df["stars"], errors='coerce').fillna(0)

            ax.bar(plot_dates, plot_stars, color="lightblue", width=0.9, alpha=0.7, label="Daily Stars")

            # Plot median and threshold if available and not all NaN
            median_data = self.stars_df[self.stars_df["median"].notna()] if "median" in self.stars_df.columns else pd.DataFrame()
            mad_data = self.stars_df[self.stars_df["mad"].notna()] if "mad" in self.stars_df.columns else pd.DataFrame()

            if not median_data.empty:
                ax.plot(pd.to_datetime(median_data["date"]), median_data["median"], color="blue",
                         linestyle="--", linewidth=1.5, label="Median (sliding window)")
            
            if not median_data.empty and not mad_data.empty and "mad" in median_data.columns: # check mad column exists
                # Align median and mad data before calculating threshold
                aligned_data = pd.merge(median_data[['date', 'median']], mad_data[['date', 'mad']], on='date', how='inner')
                if not aligned_data.empty:
                    threshold_values = aligned_data["median"] + MAD_THRESHOLD * aligned_data["mad"]
                    ax.plot(pd.to_datetime(aligned_data["date"]), threshold_values, color="red",
                             linestyle=":", linewidth=1.5, label=f"Anomaly Threshold (Median + {MAD_THRESHOLD:.1f}*MAD)")

            y_max_val = plot_stars.max()
            plot_y_max = max(10, y_max_val * 1.1) if y_max_val > 0 else 10 # Ensure y_max is at least 10

            # Highlight burst windows from self.bursts (which should be populated by calculate_fake_star_index)
            if hasattr(self, 'bursts') and self.bursts:
                # Deduplicate legend entries for bursts
                legend_labels_done = set()

                for burst in self.bursts:
                    # Ensure dates are datetime objects for axvspan
                    start_dt = datetime.datetime.combine(burst["start_date"], datetime.time.min)
                    end_dt = datetime.datetime.combine(burst["end_date"], datetime.time.max) # Cover full end day

                    burst_verdict = burst.get("verdict", "unknown")
                    burst_score_val = burst.get("burst_score", 0)
                    
                    color, alpha, label_base = "lightgray", 0.3, "Burst"
                    if burst_verdict == "fake": color, alpha, label_base = "red", 0.5, f"Fake Burst"
                    elif burst_verdict == "suspicious": color, alpha, label_base = "orange", 0.4, f"Suspicious Burst"
                    elif burst_verdict == "organic": color, alpha, label_base = "lightgreen", 0.3, f"Organic Burst"
                    
                    # Add score to label if not 'unknown'
                    full_label = f"{label_base} ({burst_score_val:.2f})" if burst_verdict != "unknown" else label_base
                    
                    if full_label not in legend_labels_done:
                        ax.axvspan(start_dt, end_dt, alpha=alpha, color=color, label=full_label)
                        legend_labels_done.add(full_label)
                    else: # Plot without adding to legend again
                        ax.axvspan(start_dt, end_dt, alpha=alpha, color=color)

                    mid_date_plot = start_dt + (end_dt - start_dt) / 2
                    ax.text(mid_date_plot, plot_y_max * 0.95, f"+{int(burst['stars'])} stars",
                             ha='center', va='top', fontsize=8, color='black',
                             bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.7))

            ax.set_xlabel("Date", fontsize=12)
            ax.set_ylabel("Stars per Day", fontsize=12)
            ax.set_title(f"Star History & Burst Analysis for {self.owner}/{self.repo}", fontsize=14)
            
            ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=12))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=30, ha='right', fontsize=10)
            plt.yticks(fontsize=10)
            
            ax.yaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=5))
            ax.grid(True, linestyle='--', alpha=0.5)
            
            handles, labels = ax.get_legend_handles_labels()
            unique_labels = dict(zip(labels, handles)) # Use dict to store unique labels
            ax.legend(unique_labels.values(), unique_labels.keys(), loc='upper left', fontsize=10)
            
            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Star history plot saved to {save_path}")
            else:
                plt.show()
            plt.close() # Close plot to free memory

        except Exception as e:
            logger.error(f"Error plotting star history for {self.owner}/{self.repo}: {str(e)}", exc_info=True)

# ... (StarAnalyzer, DependencyAnalyzer, LicenseAnalyzer, MaintainerAnalyzer, TrustScoreCalculator, CodeAnalyzer remain largely the same) ...
# Minor change in StarAnalyzer constructor regarding github_api optionality
class StarAnalyzer:
    """Analyzes star patterns to detect anomalies."""
    
    def __init__(self, owner: str, repo: str, stars_data: List[Dict], repo_created_date: str, github_api: GitHubAPI): # Made github_api non-optional
        self.owner = owner
        self.repo = repo
        self.stars_data = stars_data # This is raw data from get_stargazers
        try:
            self.repo_created_date = make_naive_datetime(parse_date(repo_created_date))
        except (TypeError, ValueError) as e:
            logger.warning(f"Invalid repo_created_date '{repo_created_date}': {e}. Defaulting to distant past.")
            self.repo_created_date = make_naive_datetime(datetime.datetime(2000,1,1)) # A sensible default

        self.github_api = github_api # Now required
        self.df = self._prepare_dataframe() # Uses self.stars_data
        
        # Initialize BurstDetector here if StarAnalyzer is the primary user of it for anomaly scores
        # Or, it can be passed in if already created by StarGuard main logic
        self.burst_detector = BurstDetector(owner, repo, github_api) # StarAnalyzer can use its own instance

    def _prepare_dataframe(self) -> Optional[pd.DataFrame]:
        """Prepare a dataframe with star data for analysis."""
        if not self.stars_data:
            logger.warning(f"No star data provided to StarAnalyzer for {self.owner}/{self.repo}")
            return pd.DataFrame() # Return empty DataFrame

        records = []
        for star_entry in self.stars_data:
            try:
                if "starred_at" in star_entry and star_entry["starred_at"]:
                    dt_obj = make_naive_datetime(parse_date(star_entry["starred_at"]))
                    if dt_obj:
                        records.append({
                            "date": dt_obj, # Keep as datetime objects
                            "user": star_entry.get("user", {}).get("login")
                        })
            except Exception as e:
                logger.debug(f"Error processing star data entry in StarAnalyzer: {str(e)}")

        if not records:
            logger.warning(f"No valid timestamp data after parsing in StarAnalyzer for {self.owner}/{self.repo}")
            return pd.DataFrame()

        df = pd.DataFrame(records)
        df.sort_values("date", inplace=True)

        # Group by day and count for timeseries analysis
        # Ensure 'date' column used for grouping is just the date part if original 'date' has time
        daily_df = df.groupby(df["date"].dt.date).size().reset_index(name="stars")
        daily_df["date"] = pd.to_datetime(daily_df["date"]) # Convert date objects back to Timestamps for consistency

        # Fill in missing dates
        if self.repo_created_date and not daily_df.empty:
            # Ensure repo_created_date is Timestamp for pd.date_range
            start_date_ts = pd.Timestamp(self.repo_created_date)
            end_date_ts = max(pd.Timestamp(make_naive_datetime(datetime.datetime.now())), daily_df["date"].max())

            all_dates_df = pd.DataFrame({"date": pd.date_range(start=start_date_ts, end=end_date_ts, freq='D')})
            
            result_df = pd.merge(all_dates_df, daily_df, on="date", how="left")
            result_df["stars"] = result_df["stars"].fillna(0).astype(int)
        elif not daily_df.empty: # If repo_created_date is not available, use data range
            result_df = daily_df
        else: # No data at all
            return pd.DataFrame()


        # Add features
        result_df["day_of_week"] = result_df["date"].dt.dayofweek
        result_df["is_weekend"] = result_df["day_of_week"].isin([5, 6]).astype(int)
        
        # Rolling windows - ensure enough data points for window, else NaNs are fine.
        if len(result_df) >= 3:
            result_df["rolling_3d"] = result_df["stars"].rolling(window=3, min_periods=1).mean()
        else: result_df["rolling_3d"] = np.nan
        if len(result_df) >= 7:
            result_df["rolling_7d"] = result_df["stars"].rolling(window=7, min_periods=1).mean()
        else: result_df["rolling_7d"] = np.nan
        if len(result_df) >= 30:
            result_df["rolling_30d"] = result_df["stars"].rolling(window=30, min_periods=1).mean()
        else: result_df["rolling_30d"] = np.nan
        
        return result_df

    def detect_anomalies(self) -> Dict:
        """Detect anomalies in star patterns using multiple methods."""
        if self.df is None or self.df.empty:
             return {"anomalies": [], "score": 50, "error": "No star data for anomaly detection."} # Neutral score

        # Initialize with an empty list or from previous calculation
        mad_anomalies_list = []
        # Use BurstDetector for MAD-based anomalies
        if self.burst_detector:
            # Ensure bursts are detected if not already
            if not self.burst_detector.bursts:
                 self.burst_detector.detect_bursts() # This populates self.burst_detector.bursts

            for burst in self.burst_detector.bursts:
                # burst["start_date"] and burst["end_date"] are datetime.date objects
                current_d = burst["start_date"]
                while current_d <= burst["end_date"]:
                    # Find stars for this day from self.df (StarAnalyzer's daily timeseries)
                    # self.df['date'] is Timestamp. current_d is datetime.date.
                    day_data = self.df[self.df["date"].dt.date == current_d]
                    stars_on_day = day_data["stars"].values[0] if not day_data.empty else 0
                    
                    mad_anomalies_list.append({
                        "date": datetime.datetime.combine(current_d, datetime.time.min), # Store as datetime
                        "stars": int(stars_on_day),
                        "z_score": np.nan, # Not applicable for MAD method here
                        "method": "mad_burst_day"
                    })
                    current_d += datetime.timedelta(days=1)
        
        # Other anomaly detection methods (Z-score, Isolation Forest, Spikes)
        z_score_anomalies_list = self._detect_with_zscore()
        isolation_forest_anomalies_list = self._detect_with_isolation_forest()
        spike_anomalies_list = self._detect_spikes()

        all_found_anomalies = z_score_anomalies_list + isolation_forest_anomalies_list + \
                              spike_anomalies_list + mad_anomalies_list
        
        # Deduplicate anomalies by date
        unique_anomalies_by_date = {}
        for anomaly in all_found_anomalies:
            # Anomaly "date" should be datetime object. Use .date() part for key.
            anomaly_date_key = anomaly["date"].date() 
            if anomaly_date_key not in unique_anomalies_by_date:
                unique_anomalies_by_date[anomaly_date_key] = anomaly
            else: # If date exists, merge methods or keep the one with higher severity indication
                if anomaly["stars"] > unique_anomalies_by_date[anomaly_date_key]["stars"]:
                     unique_anomalies_by_date[anomaly_date_key] = anomaly # Keep higher star count for that day
                # Append method if different
                existing_method = unique_anomalies_by_date[anomaly_date_key]["method"]
                if anomaly["method"] not in existing_method:
                    unique_anomalies_by_date[anomaly_date_key]["method"] += f", {anomaly['method']}"


        unique_anomalies_list = sorted(list(unique_anomalies_by_date.values()), key=lambda x: x["date"], reverse=True)

        # Score: 0-50. Higher is better.
        # Max 10 anomalies reduce score by 5 each. More than 10 anomalies -> score 0.
        anomaly_penalty = len(unique_anomalies_list) * 5
        final_score = max(0, 50 - anomaly_penalty)

        return {"anomalies": unique_anomalies_list, "score": final_score}

    def _detect_with_zscore(self, threshold: float = 3.0) -> List[Dict]:
        """Detect anomalies using Z-score method."""
        if self.df is None or self.df.empty or "stars" not in self.df.columns: return []
        df_copy = self.df.copy()
        
        mean_stars = df_copy["stars"].mean()
        std_stars = df_copy["stars"].std()

        if std_stars == 0 or pd.isna(std_stars): return [] # Avoid division by zero or NaN std

        df_copy["z_score"] = (df_copy["stars"] - mean_stars) / std_stars
        anomalies_df = df_copy[df_copy["z_score"].abs() > threshold]
        
        return [
            {"date": row["date"].to_pydatetime(), "stars": int(row["stars"]), 
             "z_score": float(row["z_score"]), "method": "z-score"}
            for _, row in anomalies_df.iterrows()
        ]

    def _detect_with_isolation_forest(self, contamination: Union[str, float] = 'auto') -> List[Dict]:
        """Detect anomalies using Isolation Forest algorithm."""
        if self.df is None or self.df.empty or len(self.df) < 10: return [] # Need enough data
        
        # Use features like 'stars' and 'rolling_7d'. Ensure 'rolling_7d' exists and is filled.
        features_to_use = ["stars"]
        if "rolling_7d" in self.df.columns and self.df["rolling_7d"].notna().any():
            features_to_use.append("rolling_7d")
        
        df_analysis = self.df[features_to_use].copy()
        df_analysis.fillna(0, inplace=True) # Fill NaNs in features (e.g. initial rolling mean)

        # Check for variance
        if any(df_analysis[col].std() == 0 for col in features_to_use):
            return [] 

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_analysis)

        try:
            # IsolationForest can be sensitive to contamination, 'auto' is often a good start.
            # If contamination is too high for sparse anomalies, it might flag too many.
            clf = IsolationForest(contamination=contamination, random_state=42, n_estimators=100)
            anomaly_predictions = clf.fit_predict(X_scaled)
        except ValueError as e: # e.g. contamination not in range
            logger.warning(f"IsolationForest ValueError: {e}. Trying with default contamination 0.1")
            try:
                clf = IsolationForest(contamination=0.1, random_state=42, n_estimators=100)
                anomaly_predictions = clf.fit_predict(X_scaled)
            except Exception as e_inner:
                logger.error(f"IsolationForest failed even with default contamination: {e_inner}")
                return []


        # Anomalies are -1
        anomaly_indices = self.df.index[anomaly_predictions == -1]
        anomalies_df = self.df.loc[anomaly_indices]

        # Calculate z-score for context, even if not used for detection by this method
        mean_stars_all = self.df["stars"].mean()
        std_stars_all = self.df["stars"].std()
        z_score_calc = lambda stars: (stars - mean_stars_all) / std_stars_all if std_stars_all > 0 else 0.0

        return [
            {"date": row["date"].to_pydatetime(), "stars": int(row["stars"]),
             "z_score": float(z_score_calc(row["stars"])), "method": "isolation_forest"}
            for _, row in anomalies_df.iterrows()
        ]


    def _detect_spikes(self, threshold_multiplier: float = 5.0, min_stars_for_spike: int = 5) -> List[Dict]:
        """Detect sudden spikes in star activity."""
        if self.df is None or self.df.empty or "stars" not in self.df.columns: return []
        df_copy = self.df.copy()

        # Average stars on days with >0 stars, excluding current day being checked
        # This is complex. Simpler: spike if stars > N AND stars > M * historical_avg
        
        historical_avg_stars = df_copy["stars"].rolling(window=30, min_periods=7).mean().shift(1) # Avg of past 30 days, lagged
        df_copy["historical_avg"] = historical_avg_stars.fillna(df_copy["stars"].expanding().mean().shift(1)) # Fallback for early data

        spikes_df = df_copy[
            (df_copy["stars"] > min_stars_for_spike) & 
            (df_copy["stars"] > df_copy["historical_avg"] * threshold_multiplier)
        ]
        
        # Calculate z-score for context
        mean_stars_all = self.df["stars"].mean()
        std_stars_all = self.df["stars"].std()
        z_score_calc = lambda stars: (stars - mean_stars_all) / std_stars_all if std_stars_all > 0 else 0.0

        return [
            {"date": row["date"].to_pydatetime(), "stars": int(row["stars"]),
             "z_score": float(z_score_calc(row["stars"])), "method": "spike_detection"}
            for _, row in spikes_df.iterrows()
        ]


    def detect_fake_stars(self) -> Dict:
        """Detect fake stars using the repository-only approach with BurstDetector."""
        if not self.github_api: # Should not happen if constructor enforces it
            return {"has_fake_stars": False, "bursts": [], "error": "GitHub API instance not provided"}
        
        try:
            # BurstDetector is already initialized in StarAnalyzer's __init__
            return self.burst_detector.calculate_fake_star_index()
        except Exception as e:
            logger.error(f"Error detecting fake stars for {self.owner}/{self.repo}: {str(e)}", exc_info=True)
            return {
                "has_fake_stars": False, "fake_star_index": 0.0, "risk_level": "low",
                "error": str(e), "bursts": [],
                "total_stars_analyzed": 0, "total_likely_fake": 0, "fake_percentage": 0.0, "worst_burst": None
            }

    def plot_star_history(self, save_path: str = None) -> None:
        """Plot star history with anomalies and bursts."""
        try:
            # Use BurstDetector's plot if available, as it's more detailed for fake star context
            if self.burst_detector:
                # Ensure burst_detector has up-to-date data based on StarAnalyzer's view
                if self.burst_detector.stars_df is None or self.burst_detector.stars_df.empty:
                    self.burst_detector.stars_df = self.df # Share the prepared DataFrame
                if not self.burst_detector.bursts: # If bursts aren't calculated yet for plotting
                    self.burst_detector.calculate_fake_star_index() # This populates bursts with scores
                
                self.burst_detector.plot_star_history(save_path)
                return

            # Fallback basic plot (should ideally not be reached if BurstDetector is always used)
            if self.df is None or self.df.empty:
                logger.warning(f"No star data to plot for {self.owner}/{self.repo} (StarAnalyzer fallback plot)")
                return

            plt.figure(figsize=(14, 7))
            ax = plt.gca()
            ax.plot(pd.to_datetime(self.df["date"]), self.df["stars"], marker='.', linestyle='-', alpha=0.6, label="Daily Stars")
            if "rolling_7d" in self.df.columns and self.df["rolling_7d"].notna().any():
                ax.plot(pd.to_datetime(self.df["date"]), self.df["rolling_7d"], color="tomato", linestyle='--', label="7-day Rolling Avg")

            anomaly_data = self.detect_anomalies() # This will use BurstDetector internally if configured
            if anomaly_data.get("anomalies"):
                anomaly_dates = [pd.to_datetime(a["date"]) for a in anomaly_data["anomalies"]]
                anomaly_stars = [a["stars"] for a in anomaly_data["anomalies"]]
                ax.scatter(anomaly_dates, anomaly_stars, color="red", s=80, label="Detected Anomalies", zorder=5)
            
            ax.set_title(f"Star History for {self.owner}/{self.repo}", fontsize=14)
            ax.set_xlabel("Date", fontsize=12)
            ax.set_ylabel("Stars per Day", fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=30, ha='right')
            plt.tight_layout()

            if save_path: plt.savefig(save_path, dpi=300)
            else: plt.show()
            plt.close()

        except Exception as e:
            logger.error(f"Error plotting star history (StarAnalyzer fallback): {str(e)}", exc_info=True)


class DependencyAnalyzer:
    """Analyzes repository dependencies for security and maintenance risks."""
    
    def __init__(self, dependencies: Dict, github_api: GitHubAPI):
        self.dependencies = dependencies # This is the raw dict from get_dependencies or _parse_dependencies
        self.github_api = github_api # Potentially used for deeper analysis later
        self.flat_dependencies = self._flatten_dependencies()
        
    def _flatten_dependencies(self) -> List[Dict]:
        flat_deps = []
        try:
            # Handle structure from GitHub Dependency Graph API (SBOM)
            if "sbom" in self.dependencies and isinstance(self.dependencies["sbom"], dict):
                sbom_data = self.dependencies["sbom"]
                if "packages" in sbom_data and isinstance(sbom_data["packages"], list):
                    for pkg_info in sbom_data["packages"]:
                        if isinstance(pkg_info, dict) and pkg_info.get("relationship") == "direct": # Process only direct dependencies
                            flat_deps.append({
                                "name": pkg_info.get("name", "unknown_dep"),
                                "version": pkg_info.get("versionInfo", "unknown_ver"),
                                "ecosystem": pkg_info.get("packageSupplier", {}).get("name", # Heuristic for ecosystem
                                                             pkg_info.get("externalRefs", [{}])[0].get("referenceCategory", "unknown_eco")),
                                "type": "runtime" # Default, SBOM might provide more
                            })
                return flat_deps # If SBOM format, assume it's complete and don't process other formats

            # Handle structure from manual file parsing (PACKAGE_MANAGERS)
            for lang_eco, dep_list in self.dependencies.items():
                if isinstance(dep_list, list):
                    for dep_item in dep_list:
                        if isinstance(dep_item, dict):
                            # Ensure basic keys exist
                            flat_deps.append({
                                "name": dep_item.get("name", f"unknown_{lang_eco}_dep"),
                                "version": dep_item.get("version", "unknown_ver"),
                                "ecosystem": lang_eco, # lang from PACKAGE_MANAGERS is the ecosystem
                                "type": dep_item.get("type", "runtime")
                            })
        except Exception as e:
            logger.error(f"Error flattening dependencies: {str(e)}", exc_info=True)
        return flat_deps

    def analyze(self) -> Dict:
        """Perform comprehensive analysis of dependencies."""
        if not self.flat_dependencies:
            return {"dependencies": [], "score": 25, "error": "No dependencies found/parsed."} # Neutral-low score

        analyzed_deps_list = []
        # Max 50 deps to analyze in detail to save time/resources, prioritize by some heuristic if needed
        deps_to_analyze = self.flat_dependencies[:50] if len(self.flat_dependencies) > 50 else self.flat_dependencies

        for dep_data in deps_to_analyze:
            single_analysis = self._analyze_single_dependency(dep_data)
            analyzed_deps_list.append(single_analysis)
        
        if not analyzed_deps_list: # Should not happen if flat_dependencies was not empty
            return {"dependencies": [], "score": 25, "error": "No dependencies were analyzed."}

        total_analyzed_count = len(analyzed_deps_list)
        high_risk_ct = sum(1 for d in analyzed_deps_list if d["risk_level"] == "high")
        medium_risk_ct = sum(1 for d in analyzed_deps_list if d["risk_level"] == "medium")
        
        # Score based on proportion of risky dependencies. Max score 30.
        # risk_score_penalty = (high_risk_ct * 2 + medium_risk_ct * 1) # Max penalty if all high_risk = total_analyzed_count*2
        # Max possible penalty is total_analyzed_count * 2 (if all are high risk)
        # Score = 30 * (1 - (penalty / (total_analyzed_count * 2) ))
        # Simplified: each high risk -2, med risk -1 from 30.
        score_val = 30 - (high_risk_ct * 2) - (medium_risk_ct * 1)
        score_val = max(0, score_val) # Ensure score is not negative

        return {
            "dependencies": analyzed_deps_list,
            "stats": {
                "total_found": len(self.flat_dependencies), # Total found before capping
                "total_analyzed": total_analyzed_count, # Number actually analyzed
                "high_risk": high_risk_ct,
                "medium_risk": medium_risk_ct,
                "low_risk": total_analyzed_count - high_risk_ct - medium_risk_ct
            },
            "score": score_val
        }

    def _analyze_single_dependency(self, dep: Dict) -> Dict:
        """Analyze a single dependency for various risk factors."""
        # Basic result structure
        analysis_result = {
            "name": dep.get("name", "N/A"),
            "version": dep.get("version", "N/A"),
            "ecosystem": dep.get("ecosystem", "N/A"),
            "type": dep.get("type", "N/A"),
            "risk_level": "low", # Default
            "risk_factors": []
        }

        version_str = str(dep.get("version", "")).lower()
        # 1. Unpinned version (simplified check)
        if version_str in ["latest", "*", "", "unknwon_ver"] or \
           any(c in version_str for c in ['^', '~', '>', '<']) and not any(op in version_str for op in ['==', '=']): # Loose check
            analysis_result["risk_factors"].append({
                "type": "version_specifier",
                "description": "Dependency version might not be pinned, potentially allowing auto-updates to risky versions."
            })
        
        # 2. Known vulnerable (placeholder - requires vulnerability DB)
        # if dep.get("name") == "known-vulnerable-package":
        #    analysis_result["risk_factors"].append({"type": "vulnerability", "description": "Known vulnerability CVE-XXXX-YYYY associated."})

        # 3. Package from non-standard source (e.g. git URL in version for npm)
        if dep.get("ecosystem") == "javascript" and isinstance(dep.get("version"), str) and \
           ("git#" in dep.get("version") or dep.get("version").startswith("file:")) :
            analysis_result["risk_factors"].append({
                "type": "source",
                "description": "Dependency sourced directly from Git or local file, bypassing standard registry vetting."
            })
        
        # Determine overall risk level based on number/severity of factors
        if len(analysis_result["risk_factors"]) >= 2: # Example: 2+ factors = high
            analysis_result["risk_level"] = "high"
        elif len(analysis_result["risk_factors"]) == 1:
            analysis_result["risk_level"] = "medium"
            
        return analysis_result
    
    def build_dependency_graph(self) -> nx.DiGraph: # Not used by main scoring, but available
        G = nx.DiGraph()
        for dep in self.flat_dependencies:
            G.add_node(dep["name"], **dep)
        # Edges would require transitive dependency info, not easily available from simple parsing.
        return G

    def check_in_package_registries(self) -> Dict: # Informational, not directly scored
        registry_presence = defaultdict(list)
        for dep in self.flat_dependencies:
            eco = str(dep.get("ecosystem", "")).lower()
            dep_name = dep.get("name")
            if not dep_name: continue

            # Map internal ecosystem names to common registry names if needed
            if "javascript" in eco or "npm" in eco: registry_presence["npm"].append(dep_name)
            elif "python" in eco or "pypi" in eco: registry_presence["pypi"].append(dep_name)
            elif "java" in eco or "maven" in eco: registry_presence["maven"].append(dep_name)
            elif "ruby" in eco or "gem" in eco: registry_presence["rubygems"].append(dep_name)
            elif "go" in eco : registry_presence["go_modules"].append(dep_name)
            else: registry_presence[eco if eco else "other"].append(dep_name)
        return dict(registry_presence)


class LicenseAnalyzer:
    """Analyzes repository licenses for compliance risks."""
    
    def __init__(self, license_data: Dict, dependencies_analyzed: List[Dict]): # Takes analyzed deps
        self.license_data = license_data # From get_license
        self.dependencies_analyzed = dependencies_analyzed # List of dicts from DependencyAnalyzer.analyze()

    def analyze(self) -> Dict:
        """Analyze licenses for risks."""
        repo_license_key = "unknown"
        repo_license_spdx = "unknown"
        
        if isinstance(self.license_data, dict) and isinstance(self.license_data.get("license"), dict):
            license_info = self.license_data["license"]
            repo_license_key = str(license_info.get("key", "unknown")).lower()
            repo_license_spdx = str(license_info.get("spdx_id", "unknown"))
            if repo_license_spdx == "NOASSERTION": repo_license_spdx = "unknown" # Treat NOASSERTION as unknown
            if repo_license_key == "other" and repo_license_spdx != "unknown": # Prefer SPDX if key is 'other'
                 pass # Use spdx_id as is
            elif repo_license_key == "unknown" and repo_license_spdx != "unknown":
                repo_license_key = repo_license_spdx.lower() # Try to use SPDX as key if main key is unknown


        repo_risk_str = LICENSE_RISK_LEVELS.get(repo_license_key, "high") # Default to high if key not in map

        # Analyze dependency licenses (placeholder - would need license info per dependency)
        # For now, assume no incompatible dependency licenses found.
        # A real version would check for GPL in MIT project, etc.
        dependency_license_issues = [] 
        # Example: if any dep has 'agpl-3.0' and repo is 'mit', it's a high risk.

        # Score 0-20. Higher is better.
        # 'high' risk license (AGPL, proprietary, unknown) -> low score
        # 'medium' risk (GPLs) -> medium score
        # 'low' risk (permissive) -> high score
        score_val = 5 # Default for high risk / unknown
        if repo_risk_str == "medium": score_val = 10
        elif repo_risk_str == "low": score_val = 20
        
        # Penalize if dependency license issues were found (not implemented yet)
        # if dependency_license_issues: score_val = max(0, score_val - 10)

        return {
            "repo_license_spdx": repo_license_spdx,
            "repo_license_key": repo_license_key,
            "repo_license_risk": repo_risk_str,
            "dependency_license_issues_found": len(dependency_license_issues), # Placeholder
            "score": score_val
        }


class MaintainerAnalyzer:
    """Analyzes repository maintainers for reputation and activity."""
    
    def __init__(self, contributors: List[Dict], commits: List[Dict], github_api: GitHubAPI): # Added github_api
        self.contributors = contributors # From get_contributors
        self.commits = commits # From get_commits (recent)
        self.github_api = github_api # For fetching more maintainer details if needed

    def analyze(self) -> Dict:
        """Analyze maintainer activity and reputation."""
        if not self.contributors:
            return {"maintainers": [], "recent_activity_summary": {}, "score": 5, "error": "No contributor data."} # Low score

        processed_maintainers = self._process_contributors() # Top N contributors
        
        # Recent commit activity from self.commits (already fetched for last 90 days)
        commits_last_90d = len(self.commits)
        
        # Active maintainers: defined as top contributors with recent commits (more robust)
        # For simplicity, use processed_maintainers' own contribution count as proxy for "active"
        active_maintainer_count = sum(1 for m in processed_maintainers if m["activity_level"] == "high")
        total_processed_maintainers = len(processed_maintainers)

        # Score 0-20. Higher is better.
        # Based on number of active maintainers and recent commit cadence.
        score_val = 0
        if total_processed_maintainers > 0:
            # Base score on having active maintainers
            if active_maintainer_count >= 3: score_val += 10
            elif active_maintainer_count >= 1: score_val += 5
            
            # Add points for commit frequency
            if commits_last_90d > 50: score_val += 10 # Very active
            elif commits_last_90d > 10: score_val += 7 # Moderately active
            elif commits_last_90d > 0: score_val += 3  # Some activity
        
        score_val = min(20, score_val) # Cap at 20

        return {
            "maintainers_analyzed": processed_maintainers,
            "recent_activity_summary": {
                "commits_last_90d": commits_last_90d,
                "active_maintainers_heuristic": active_maintainer_count,
                "total_top_contributors_analyzed": total_processed_maintainers
            },
            "score": score_val
        }

    def _process_contributors(self) -> List[Dict]:
        """Process contributor data to extract maintainer information."""
        if not self.contributors: return []
        
        # Sort by contributions, ensure 'contributions' key exists
        valid_contributors = [c for c in self.contributors if isinstance(c, dict) and "contributions" in c]
        sorted_contribs = sorted(valid_contributors, key=itemgetter("contributions"), reverse=True)
        
        top_n_maintainers = []
        for contrib_data in sorted_contribs[:5]: # Analyze top 5 contributors
            contributions_count = contrib_data.get("contributions", 0)
            activity_lvl = "low"
            if contributions_count > 100: activity_lvl = "high" # Arbitrary thresholds
            elif contributions_count > 20: activity_lvl = "medium"
            
            top_n_maintainers.append({
                "login": contrib_data.get("login", "N/A"),
                "contributions": contributions_count,
                "activity_level": activity_lvl, # Based on total contributions
                "profile_url": contrib_data.get("html_url", "")
                # Could add: fetch user profile for age, followers for deeper analysis (API heavy)
            })
        return top_n_maintainers

    def check_recent_activity(self) -> Dict: # Primarily informational
        """Check repository activity based on recent commits (from self.commits)."""
        if not self.commits: # self.commits are already for last 90 days
            return {
                "activity_counts_by_period": {"last_90_days": 0},
                "overall_activity_level": "inactive",
                "days_since_last_commit": 999 # Indicates very old or no commits
            }

        # All commits in self.commits are within the last 90 days.
        # Calculate days since the very last commit among these.
        last_commit_date_obj = None
        if self.commits:
            try:
                # Assuming commits are sorted by date by API, but re-sort to be sure
                # Commits are usually returned most recent first from API.
                # Find the most recent commit date.
                most_recent_commit_dt = None
                for c_data in self.commits:
                    if isinstance(c_data, dict) and "commit" in c_data and \
                       isinstance(c_data["commit"], dict) and "author" in c_data["commit"] and \
                       isinstance(c_data["commit"]["author"], dict) and "date" in c_data["commit"]["author"]:
                        
                        commit_dt = make_naive_datetime(parse_date(c_data["commit"]["author"]["date"]))
                        if commit_dt and (most_recent_commit_dt is None or commit_dt > most_recent_commit_dt):
                            most_recent_commit_dt = commit_dt
                
                if most_recent_commit_dt:
                    last_commit_date_obj = most_recent_commit_dt

            except Exception as e:
                logger.debug(f"Error parsing commit dates for activity check: {e}")

        days_lapsed = 999
        if last_commit_date_obj:
            days_lapsed = (make_naive_datetime(datetime.datetime.now()) - last_commit_date_obj).days
            days_lapsed = max(0, days_lapsed) # Ensure non-negative

        # Determine activity level based on commits in last 90 days and recency
        activity_lvl_str = "inactive"
        commits_90d_count = len(self.commits)
        if commits_90d_count > 20 and days_lapsed < 14: activity_lvl_str = "high"
        elif commits_90d_count > 5 and days_lapsed < 30: activity_lvl_str = "medium"
        elif commits_90d_count > 0 and days_lapsed < 90 : activity_lvl_str = "low"
        
        return {
            "activity_counts_by_period": {"last_90_days": commits_90d_count},
            "overall_activity_level": activity_lvl_str,
            "days_since_last_commit": days_lapsed
        }


class TrustScoreCalculator:
    """Calculates overall trust score from individual analysis components."""
    
    def __init__(
        self, 
        star_analysis_res: Dict, 
        dependency_analysis_res: Dict,
        license_analysis_res: Dict,
        maintainer_analysis_res: Dict,
        fake_star_analysis_res: Optional[Dict] = None # This is from BurstDetector
    ):
        self.star_analysis = star_analysis_res
        self.dependency_analysis = dependency_analysis_res
        self.license_analysis = license_analysis_res
        self.maintainer_analysis = maintainer_analysis_res
        self.fake_star_analysis = fake_star_analysis_res # Results from BurstDetector.calculate_fake_star_index

    def calculate(self) -> Dict:
        """Calculate the overall trust score."""
        # Scores from components (higher is better for these)
        # Star pattern: 0-50
        # Dependencies: 0-30
        # License: 0-20
        # Maintainers: 0-20
        # Total ideal sum = 50 + 30 + 20 + 20 = 120 (oops, components sum to 120, need to re-scale or adjust max points)
        # Let's assume the component scores are correctly scaled to their max as per their classes.
        # Star: 50, Deps: 30, License: 20, Maintainer: 20. Total = 120.
        # We want final score 0-100.

        star_score_raw = self.star_analysis.get("score", 25) # Default to neutral if missing
        dep_score_raw = self.dependency_analysis.get("score", 15)
        lic_score_raw = self.license_analysis.get("score", 10)
        maint_score_raw = self.maintainer_analysis.get("score", 10)

        # Sum of raw scores from components. Max possible is 120.
        base_total_score = star_score_raw + dep_score_raw + lic_score_raw + maint_score_raw
        
        # Scale this base_total_score (max 120) to a 0-100 range.
        # scaled_base_score = (base_total_score / 120) * 100
        # For now, let's keep the sum and apply penalty, then cap at 100.
        # This means some aspects might "max out" the score even if others are low.
        # Alternative: weighted average. For now, simple sum then penalty.

        penalty_for_fake_stars = 0
        if self.fake_star_analysis and self.fake_star_analysis.get("has_fake_stars", False):
            # fake_star_index is 0 (good) to 1 (bad)
            fsi = self.fake_star_analysis.get("fake_star_index", 0.0)
            # Penalty: up to 50 points based on FSI.
            # If FSI is 1.0 (max fake), penalty is 50.
            # If FSI is 0.3 (organic threshold), penalty is low.
            penalty_for_fake_stars = int(fsi * 50) 

            # Extra penalty if risk is 'high' from FSI
            if self.fake_star_analysis.get("risk_level") == "high":
                penalty_for_fake_stars += 20 # Hefty additional penalty
            elif self.fake_star_analysis.get("risk_level") == "medium":
                penalty_for_fake_stars +=10

        # Total score after penalty, capped 0-100
        # Current max base_total_score = 120. Max penalty can be 50+20=70.
        # If base is 120, penalty 70 -> final 50.
        # If base is 60, penalty 70 -> final -10 (becomes 0).
        # This seems reasonable.
        final_score_val = base_total_score - penalty_for_fake_stars
        final_score_val = max(0, min(100, final_score_val)) # Cap at 0 and 100


        risk_lvl_str = "high" # Default for low scores
        if final_score_val >= 80: risk_lvl_str = "low"
        elif final_score_val >= 60: risk_lvl_str = "medium"
        
        return {
            "total_score": int(round(final_score_val)),
            "risk_level": risk_lvl_str,
            "score_components": {
                "star_pattern_score": star_score_raw,       # max 50
                "dependencies_score": dep_score_raw,         # max 30
                "license_score": lic_score_raw,              # max 20
                "maintainer_activity_score": maint_score_raw # max 20
            },
            "fake_star_penalty_applied": penalty_for_fake_stars
        }
        
    def generate_badge_url(self, owner: str, repo: str) -> str: # Changed 'self' to 'cls' or make static if no self needed
        """Generate a badge URL for the repository."""
        # This method is called on an instance, so 'self' is fine.
        # It needs the calculated score, so it should be called after `calculate()`.
        # Let's assume it's called on an instance that has just run `calculate()`.
        # Or, it should take the score as argument.

        # For simplicity, let's make it require the score dict from calculate().
        # Modifying signature: def generate_badge_url(self, score_data: Dict, owner: str, repo: str) -> str:
        # However, the current call in StarGuard.analyze_repo is `trust_calculator.generate_badge_url(owner, repo)`
        # This implies it re-calculates or uses stored state. Let's assume it re-calculates.
        
        score_data = self.calculate() # Recalculate if called independently, or use stored if available.
        calculated_total_score = score_data["total_score"]
        
        # Determine badge color based on score
        color_str = "red" # Default for high risk / low score
        if calculated_total_score >= 80: color_str = "success" # Green
        elif calculated_total_score >= 60: color_str = "yellow"    # Yellow
        
        # URL encode label and message
        label_enc = quote_plus("StarGuard Score")
        message_enc = quote_plus(f"{calculated_total_score}/100")
        
        # Shield.io URL
        badge_url = f"https://img.shields.io/badge/{label_enc}-{message_enc}-{color_str}.svg?style=flat-square&logo=github"
        return badge_url


class CodeAnalyzer:
    """Analyzes repository code for potential malware or suspicious patterns."""
    
    def __init__(self, owner: str, repo: str, github_api: GitHubAPI):
        self.owner = owner
        self.repo = repo
        self.github_api = github_api
        
    def check_for_suspicious_patterns(self) -> Dict:
        """Check repository for suspicious code patterns."""
        # Define suspicious patterns (simplified)
        patterns_by_category = {
            "obfuscation": [r"eval\s*\(", r"fromCharCode", r"\\x[0-9a-f]{2}", r"document\.write\s*\(unescape\("],
            "remote_execution": [r"new\s+Function\s*\(", r"setTimeout\s*\(\s*[\"'].*eval\("],
            "data_exfiltration": [r"fetch\s*\(\s*[\"']https?:\/\/[^\"']+", r"navigator\.sendBeacon"],
            "crypto_jacking": [r"CryptoJS", r"miner", r"coinhive", r"cryptonight"],
            "suspicious_imports": [r"require\s*\(\s*[\"'](http|https|net|child_process)[\"']"],
        }

        # Files to check (prioritize common script/config files)
        files_to_scan = [
            "package.json", "Gruntfile.js", "gulpfile.js", # JS
            "setup.py", "__init__.py", # Python
            # Add more based on repo language if known, e.g. from repo_data["language"]
        ]
        # Heuristic: try to get a few .js or .py files from root if above are not found
        # This is complex, for now, stick to predefined list.

        found_indicators = defaultdict(list)
        total_hits = 0

        for file_path_str in files_to_scan:
            try:
                file_content_str = self.github_api.get_file_content(self.owner, self.repo, file_path_str)
                if not file_content_str: continue

                for category, regex_list in patterns_by_category.items():
                    for regex_pattern in regex_list:
                        try:
                            # Using re.finditer to get match objects for more context if needed
                            for match_obj in re.finditer(regex_pattern, file_content_str, re.IGNORECASE):
                                found_indicators[category].append({
                                    "file": file_path_str,
                                    "pattern": regex_pattern,
                                    "matched_text": match_obj.group(0)[:100], # First 100 chars of match
                                    # "line_number": file_content_str[:match_obj.start()].count('\n') + 1 # Can be slow
                                })
                                total_hits +=1
                        except re.error as re_e:
                             logger.debug(f"Regex error for pattern '{regex_pattern}': {re_e}")
            except Exception as e:
                logger.debug(f"Error scanning file {file_path_str} in {self.owner}/{self.repo}: {e}")
        
        # Analyze package.json for suspicious scripts or dependencies (if JS project)
        suspicious_deps_or_scripts = []
        if "package.json" in files_to_scan: # or if repo language is JavaScript
            pkg_json_content = self.github_api.get_file_content(self.owner, self.repo, "package.json")
            if pkg_json_content:
                try:
                    pkg_data = json.loads(pkg_json_content)
                    # Check scripts for obfuscated commands or downloads from weird URLs
                    for script_name, script_cmd in pkg_data.get("scripts", {}).items():
                        if isinstance(script_cmd, str) and ("curl" in script_cmd or "wget" in script_cmd or "node -e" in script_cmd):
                             if "npmjs.org" not in script_cmd and "github.com" not in script_cmd: # If not from known good sources
                                suspicious_deps_or_scripts.append(f"Suspicious script '{script_name}': {script_cmd[:100]}")
                                total_hits += 2 # Higher weight for suspicious scripts
                    # Check dependencies for typosquatting (very basic)
                    for dep_name in list(pkg_data.get("dependencies", {}).keys()) + list(pkg_data.get("devDependencies", {}).keys()):
                        if "rpel" in dep_name or "ajv-" in dep_name and dep_name != "ajv-keywords": # Example typos
                            suspicious_deps_or_scripts.append(f"Potentially typosquatted dependency: {dep_name}")
                            total_hits += 3
                except json.JSONDecodeError:
                    logger.debug(f"Could not parse package.json for code analysis in {self.owner}/{self.repo}")


        # Calculate suspicion score (0-100). Higher means more suspicious.
        # Each hit +5, max score from hits is 50.
        # Suspicious deps/scripts add more.
        suspicion_score_val = min(50, total_hits * 5) 
        suspicion_score_val += min(50, len(suspicious_deps_or_scripts) * 10) # Max 50 from this
        suspicion_score_val = min(100, suspicion_score_val) # Cap total at 100

        return {
            "findings_by_category": dict(found_indicators),
            "suspicious_package_elements": suspicious_deps_or_scripts,
            "total_suspicious_hits": total_hits,
            "calculated_suspicion_score": suspicion_score_val, # 0-100, higher is more suspicious
            "is_potentially_suspicious": suspicion_score_val > 40 # Threshold for flagging
        }


class StarGuard:
    """Main StarGuard analysis engine."""

    def __init__(self, token: Optional[str] = None):
        self.github_api = GitHubAPI(token)

    def analyze_repo(self, owner: str, repo: str, analyze_fake_stars: bool = True) -> Dict:
        """Perform comprehensive analysis on a GitHub repository."""
        logger.info(f"Starting analysis of {owner}/{repo}")

        try:
            repo_data = self.github_api.get_repo(owner, repo) # Raises ValueError if not found or API fails
            logger.info(f"Fetched repository data for: {repo_data.get('full_name', f'{owner}/{repo}')}")
        except ValueError as e: # Catch specific error from get_repo
            logger.error(f"Failed to fetch repository {owner}/{repo}: {e}")
            return {"error": str(e)}

        # Fetch data concurrently or sequentially
        # For simplicity, sequential fetching:
        stars_raw_data = self.github_api.get_stargazers(owner, repo, get_timestamps=True, days_limit=0) # Get all for StarAnalyzer
        logger.info(f"Fetched {len(stars_raw_data)} stargazers total.")

        contributors_data = self.github_api.get_contributors(owner, repo)
        logger.info(f"Fetched {len(contributors_data)} contributors.")
        
        # Fetch recent commits (last 90 days)
        since_90d_iso = (make_naive_datetime(datetime.datetime.now()) - datetime.timedelta(days=90)).isoformat()
        commits_recent_data = self.github_api.get_commits(owner, repo, since=since_90d_iso)
        logger.info(f"Fetched {len(commits_recent_data)} commits from last 90 days.")

        dependencies_raw_data = self.github_api.get_dependencies(owner, repo)
        logger.info(f"Fetched raw dependency data (Source: {'API' if 'sbom' in dependencies_raw_data else 'File Parsing'}).")

        license_api_data = self.github_api.get_license(owner, repo)
        logger.info(f"Fetched license info: {license_api_data.get('license',{}).get('spdx_id', 'N/A')}")

        # Initialize Analyzers
        # BurstDetector for fake star analysis (central component for this)
        burst_detector_inst = BurstDetector(owner, repo, self.github_api)
        
        # StarAnalyzer uses raw star data and its own BurstDetector instance, or can share one
        star_analyzer_inst = StarAnalyzer(owner, repo, stars_raw_data, repo_data["created_at"], self.github_api)
        # If StarAnalyzer should use the main burst_detector_inst:
        # star_analyzer_inst.burst_detector = burst_detector_inst # Share instance

        dependency_analyzer_inst = DependencyAnalyzer(dependencies_raw_data, self.github_api)
        # LicenseAnalyzer needs analyzed dependency list if it were to check compatibility
        # For now, it doesn't use it deeply, so pass empty or basic dep list.
        # Let's pass the flat_dependencies from dependency_analyzer_inst.
        license_analyzer_inst = LicenseAnalyzer(license_api_data, dependency_analyzer_inst.flat_dependencies)
        
        maintainer_analyzer_inst = MaintainerAnalyzer(contributors_data, commits_recent_data, self.github_api)
        code_analyzer_inst = CodeAnalyzer(owner, repo, self.github_api)

        # Perform Analyses
        logger.info("Running star pattern analysis...")
        star_analysis_results = star_analyzer_inst.detect_anomalies()
        
        fake_star_analysis_results = None
        if analyze_fake_stars:
            logger.info("Running fake star detection (BurstDetector)...")
            # Use star_analyzer_inst's burst_detector as it has timeseries possibly
            fake_star_analysis_results = star_analyzer_inst.detect_fake_stars() 
            fsi_val = fake_star_analysis_results.get('fake_star_index',0.0)
            fsi_risk = fake_star_analysis_results.get('risk_level','low')
            logger.info(f"Fake Star Index: {fsi_val:.2f} ({fsi_risk.upper()} RISK)")
        else: # Create a default structure if skipped
             fake_star_analysis_results = {
                "has_fake_stars": False, "fake_star_index": 0.0, "risk_level": "low", "bursts": [],
                "total_stars_analyzed": 0, "total_likely_fake": 0, "fake_percentage": 0.0, "worst_burst": None,
                "message": "Fake star analysis skipped by user."
            }


        logger.info("Running dependency analysis...")
        dependency_analysis_results = dependency_analyzer_inst.analyze()
        
        logger.info("Running license analysis...")
        license_analysis_results = license_analyzer_inst.analyze()
        
        logger.info("Running maintainer analysis...")
        maintainer_analysis_results = maintainer_analyzer_inst.analyze()
        
        logger.info("Running suspicious code pattern check...")
        code_analysis_results = code_analyzer_inst.check_for_suspicious_patterns()


        # Calculate Trust Score
        trust_calculator_inst = TrustScoreCalculator(
            star_analysis_results, dependency_analysis_results,
            license_analysis_results, maintainer_analysis_results,
            fake_star_analysis_results # Pass the FSI results here
        )
        trust_score_final = trust_calculator_inst.calculate()
        badge_url_str = trust_calculator_inst.generate_badge_url(owner, repo) # Uses the score from calculate()

        # Additional informational outputs
        activity_info_detailed = maintainer_analyzer_inst.check_recent_activity()
        package_registry_info = dependency_analyzer_inst.check_in_package_registries()

        # Prepare final result dictionary
        final_result_dict = {
            "repository_info": { # Renamed from "repository" to avoid clash with analysis result key "repo"
                "name": repo_data.get("name"), "full_name": repo_data.get("full_name"),
                "description": repo_data.get("description", ""), "stars_count": repo_data.get("stargazers_count"),
                "created_at": repo_data.get("created_at"), "html_url": repo_data.get("html_url"),
                "language": repo_data.get("language", "N/A")
            },
            "trust_score_summary": trust_score_final,
            "star_pattern_analysis": star_analysis_results,
            "fake_star_detection_analysis": fake_star_analysis_results, # FSI specific results
            "dependency_health_analysis": dependency_analysis_results,
            "license_compliance_analysis": license_analysis_results,
            "maintainer_activity_analysis": maintainer_analysis_results, # Contains score
            "code_suspicion_analysis": code_analysis_results,
            "detailed_activity_metrics": activity_info_detailed, # Extra info
            "package_registry_check": package_registry_info,   # Extra info
            "generated_badge": {
                "url": badge_url_str,
                "markdown": f"[![StarGuard Score]({badge_url_str})](https://starguard.example.com/report/{owner}/{repo})" # Example link
            }
        }
        logger.info(f"Analysis for {owner}/{repo} complete. Trust Score: {trust_score_final['total_score']}")
        return final_result_dict

    def generate_report(self, analysis_result: Dict, format_str: str = "text") -> str:
        """Generate a formatted report from analysis results."""
        if "error" in analysis_result: # Top-level error from analyze_repo
            return f"Error: {analysis_result['error']}"
        
        if format_str == "json":
            # Custom default handler for datetime objects if any slip through
            def dt_handler(o):
                if isinstance(o, (datetime.datetime, datetime.date)): return o.isoformat()
                raise TypeError (f"Type {type(o)} not serializable")
            return json.dumps(analysis_result, indent=2, default=dt_handler)

        # Text & Markdown reports
        repo_info = analysis_result.get("repository_info", {})
        trust_summary = analysis_result.get("trust_score_summary", {})
        fake_star_info = analysis_result.get("fake_star_detection_analysis", {})
        code_sus_info = analysis_result.get("code_suspicion_analysis", {})

        md_report_lines = []
        text_report_lines = []

        # --- Header ---
        title = f"StarGuard Analysis: {repo_info.get('full_name', 'N/A')}"
        md_report_lines.extend([f"# {title}", ""])
        text_report_lines.extend([title, "=" * len(title), ""])

        # --- Overview ---
        overview_md = [
            "##  Overview Section",
            f"- **Repository**: [{repo_info.get('full_name','N/A')}]({repo_info.get('html_url','')})",
            f"- **Description**: {repo_info.get('description','N/A')}",
            f"- **Created**: {repo_info.get('created_at','N/A')}",
            f"- **Stars**: {repo_info.get('stars_count','N/A')}",
            f"- **Primary Language**: {repo_info.get('language','N/A')}", ""
        ]
        overview_text = [
            "Overview:",
            f"  Repository: {repo_info.get('full_name','N/A')} ({repo_info.get('html_url','')})",
            f"  Description: {repo_info.get('description','N/A')}",
            f"  Created: {repo_info.get('created_at','N/A')}",
            f"  Stars: {repo_info.get('stars_count','N/A')}",
            f"  Primary Language: {repo_info.get('language','N/A')}", ""
        ]
        md_report_lines.extend(overview_md)
        text_report_lines.extend(overview_text)
        
        # --- Trust Score ---
        trust_score_val = trust_summary.get('total_score', 'N/A')
        trust_risk_lvl = trust_summary.get('risk_level', 'N/A').upper()
        md_report_lines.extend([f"## Trust Score: {trust_score_val}/100 ({trust_risk_lvl} RISK)", ""])
        text_report_lines.extend([f"TRUST SCORE: {trust_score_val}/100 ({trust_risk_lvl} RISK)", ""])

        # Fake Star Penalty in Trust Score Breakdown
        penalty = trust_summary.get('fake_star_penalty_applied',0)
        if penalty > 0:
            md_report_lines.append(f"**Fake Star Penalty Applied**: -{penalty} points from base score.")
            text_report_lines.append(f"  Fake Star Penalty Applied: -{penalty} points from base score.")
        md_report_lines.append("")
        text_report_lines.append("")

        # --- Fake Star Detection ---
        if fake_star_info and fake_star_info.get("has_fake_stars"):
            fsi = fake_star_info.get('fake_star_index',0.0)
            fsi_risk = fake_star_info.get('risk_level','N/A').upper()
            fsi_likely_fake = fake_star_info.get('total_likely_fake',0)
            fsi_perc = fake_star_info.get('fake_percentage',0.0)
            fsi_burst_count = len(fake_star_info.get('bursts',[]))

            alert_emoji = "🚨" if fsi_risk == "HIGH" else "⚠️"
            md_report_lines.extend([f"## {alert_emoji} Fake Star Detection {alert_emoji}", ""])
            text_report_lines.extend([f"!!! {alert_emoji} Fake Star Detection {alert_emoji} !!!", "-"*30, ""])
            
            md_report_lines.extend([
                f"**Fake Star Index**: {fsi:.2f} ({fsi_risk} RISK)",
                f"**Likely Fake Stars**: {fsi_likely_fake} ({fsi_perc:.1f}% of those analyzed in bursts)",
                f"**Suspicious Bursts Detected**: {fsi_burst_count}", ""
            ])
            text_report_lines.extend([
                f"  Fake Star Index: {fsi:.2f} ({fsi_risk} RISK)",
                f"  Likely Fake Stars: {fsi_likely_fake} ({fsi_perc:.1f}% of those analyzed in bursts)",
                f"  Suspicious Bursts Detected: {fsi_burst_count}", ""
            ])

            if fake_star_info.get('bursts'):
                md_report_lines.append("### Suspicious Star Burst Details:")
                text_report_lines.append("  Suspicious Star Burst Details:")
                for idx, burst_item in enumerate(fake_star_info['bursts'][:3]): # Show top 3
                    b_verdict = burst_item.get('verdict','N/A').upper()
                    b_score = burst_item.get('burst_score',0.0)
                    b_start = burst_item.get('start_date','N/A')
                    b_end = burst_item.get('end_date','N/A')
                    b_stars = burst_item.get('stars',0)
                    b_fake_ratio = burst_item.get('fake_ratio_in_burst',0.0) * 100

                    md_report_lines.append(f"- **Burst {idx+1}**: {b_verdict} (Score: {b_score:.2f}), Period: {b_start} to {b_end}, Stars: +{b_stars},  Estimated Fake Ratio: {b_fake_ratio:.1f}%")
                    text_report_lines.append(f"    Burst {idx+1}: {b_verdict} (Score: {b_score:.2f})")
                    text_report_lines.append(f"      Period: {b_start} a {b_end}, Stars: +{b_stars}, Estimated Fake Ratio: {b_fake_ratio:.1f}%")
                if len(fake_star_info['bursts']) > 3:
                    md_report_lines.append("- ...and other bursts.")
                    text_report_lines.append("    ...and other bursts.")
                md_report_lines.append("")
                text_report_lines.append("")
        elif "message" in fake_star_info : # E.g. analysis skipped
             md_report_lines.extend(["## Fake Star Detection", f"_{fake_star_info['message']}_", ""])
             text_report_lines.extend(["Fake Star Detection:", f"  {fake_star_info['message']}", ""])


        # --- Code Suspicion ---
        if code_sus_info and code_sus_info.get("is_potentially_suspicious"):
            cs_score = code_sus_info.get('calculated_suspicion_score',0)
            md_report_lines.extend([f"## ⚠️ Suspicious Code Detection", ""])
            text_report_lines.extend(["!!! ⚠️ SUSPICIOUS CODE DETECTION !!!", "-"*30, ""])
            
            md_report_lines.extend([f"**Code Suspicion Score**: {cs_score}/100", ""])
            text_report_lines.extend([f"  Code Suspicion Score: {cs_score}/100", ""])

            if code_sus_info.get("findings_by_category"):
                md_report_lines.append("Suspicious patterns found:")
                text_report_lines.append("  Suspicious patterns found:")
                for cat, finds in code_sus_info["findings_by_category"].items():
                    if finds:
                        md_report_lines.append(f"- **{cat.replace('_',' ').title()}**: {len(finds)} istanze")
                        text_report_lines.append(f"    - {cat.replace('_',' ').title()}: {len(finds)} istanze")
            if code_sus_info.get("suspicious_package_elements"):
                 md_report_lines.append("Suspicious elements in package manifest:")
                 text_report_lines.append("  Suspicious elements in package manifest:")
                 for elem_desc in code_sus_info["suspicious_package_elements"][:3]:
                     md_report_lines.append(f"- {elem_desc}")
                     text_report_lines.append(f"    - {elem_desc}")
            md_report_lines.append("")
            text_report_lines.append("")

        # --- Individual Analysis Sections (Simplified) ---
        sections = {
            "Star Pattern Analysis": analysis_result.get("star_pattern_analysis"),
            "Dependency Health": analysis_result.get("dependency_health_analysis"),
            "License Compliance": analysis_result.get("license_compliance_analysis"),
            "Maintainer Activity": analysis_result.get("maintainer_activity_analysis")
        }
        score_comp_map = trust_summary.get("score_components", {})
        score_key_map = { # Map section title to score component key
            "Star Pattern Analysis": "star_pattern_score",
            "Dependency Health": "dependencies_score",
            "License Compliance": "license_score",
            "Maintainer Activity": "maintainer_activity_score"
        }

        for section_title, data in sections.items():
            if not data: continue
            score_val = score_comp_map.get(score_key_map.get(section_title), "N/A")
            max_score = 50 if "Sttars" in section_title else \
                        30 if "Dependancy" in section_title else \
                        20 # Licenza e Manutentori
            
            md_report_lines.extend([f"## {section_title}", f"**Component Score**: {score_val}/{max_score}", ""])
            text_report_lines.extend([f"{section_title.upper()}:", f"  Component Score: {score_val}/{max_score}", ""])
            
            # Add 1-2 key details from each section
            if section_title == "Star Pattern Analysis" and data.get("anomalies"):
                num_anom = len(data["anomalies"])
                md_report_lines.append(f"- Detected {num_anom} anomalies in star pattern.")
                text_report_lines.append(f"  - Detected {num_anom} anomalies in star pattern.")
            elif section_title == "Dependency Health" and data.get("stats"):
                stats = data["stats"]
                md_report_lines.append(f"- Analyzed {stats.get('total_analyzed',0)} dependencies: {stats.get('high_risk',0)} high risk, {stats.get('medium_risk',0)} medium risk.")
                text_report_lines.append(f"  - Analyzed {stats.get('total_analyzed',0)} dependencies: {stats.get('high_risk',0)} high risk, {stats.get('medium_risk',0)} medium risk.")
            elif section_title == "License Compliance":
                lic_key = data.get('repo_license_key','N/A')
                lic_risk = data.get('repo_license_risk','N/A').upper()
                md_report_lines.append(f"- Repository License: `{lic_key}` (Risk: {lic_risk}).")
                text_report_lines.append(f"  - Repository License: {lic_key} (Risk: {lic_risk}).")
            elif section_title == "Maintainer activity" and data.get("recent_activity_summary"):
                act_sum = data["recent_activity_summary"]
                md_report_lines.append(f"- {act_sum.get('commits_last_90d',0)} commit negli ultimi 90 giorni; {act_sum.get('active_maintainers_heuristic',0)} active maintainers (among top contributors).")
                text_report_lines.append(f"  - {act_sum.get('commits_last_90d',0)} commit negli ultimi 90 giorni; {act_sum.get('active_maintainers_heuristic',0)} active maintainers (among top contributors).")

            md_report_lines.append("")
            text_report_lines.append("")


        # --- Badge ---
        badge_info = analysis_result.get("generated_badge", {})
        if badge_info.get("url"):
            md_report_lines.extend(["## Badge", "", badge_info["markdown"], ""])
            text_report_lines.extend(["BADGE", "", badge_info["markdown"], ""])

        if format_str == "markdown":
            return "\n".join(md_report_lines)
        else: # Default to text
            return "\n".join(text_report_lines)


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="StarGuard - GitHub Repository Analysis Tool with Advanced Fake Star Detection")
    parser.add_argument("owner_repo", help="GitHub repository in format 'owner/repo' or full URL")
    parser.add_argument("-t", "--token", help="GitHub personal access token (or set GITHUB_TOKEN env var)")
    parser.add_argument("-f", "--format", choices=["text", "json", "markdown"], default="text", help="Output format")
    parser.add_argument("-o", "--output", help="Output file (default: stdout)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose DEBUG logging")
    parser.add_argument("--plot", help="Save star history plot to specified file path (e.g., plot.png)")
    parser.add_argument("--no-fake-stars", action="store_true", help="Skip fake star detection component (faster, less comprehensive)")
    parser.add_argument("--burst-only", action="store_true", help="Only run fake star burst detection and basic report (fastest)")
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logging.getLogger("urllib3").setLevel(logging.INFO) # Quieten noisy library if needed
    
    owner_str, repo_str = "", ""
    if args.owner_repo.startswith(("http://", "https://")):
        try:
            parsed_url = urlparse(args.owner_repo)
            path_parts = parsed_url.path.strip('/').split('/')
            if len(path_parts) >= 2 and parsed_url.netloc.lower() == "github.com":
                owner_str, repo_str = path_parts[0], path_parts[1]
                if repo_str.endswith(".git"): repo_str = repo_str[:-4] # Remove .git suffix
            else:
                raise ValueError("Invalid GitHub URL structure.")
        except ValueError as e:
            logger.error(f"Invalid GitHub URL format: {e}. Expected: https://github.com/owner/repo")
            sys.exit(1)
    else:
        try:
            owner_str, repo_str = args.owner_repo.split("/")
        except ValueError:
            logger.error("Invalid repository format. Use 'owner/repo' or a full GitHub URL.")
            sys.exit(1)

    github_token = args.token or os.environ.get("GITHUB_TOKEN")
    if not github_token:
        logger.warning("No GitHub token provided via --token or GITHUB_TOKEN env var. API rate limits will be significantly lower.")

    try:
        final_report_str = ""
        if args.burst_only:
            logger.info(f"Running burst detection ONLY for {owner_str}/{repo_str}")
            github_api_inst = GitHubAPI(github_token)
            burst_detector_inst = BurstDetector(owner_str, repo_str, github_api_inst)
            # calculate_fake_star_index now returns a dict with expected keys
            burst_result_dict = burst_detector_inst.calculate_fake_star_index() 

            if args.format == "json":
                final_report_str = json.dumps(burst_result_dict, indent=2, default=str)
            else: # Text/Markdown for burst_only is simplified
                lines = [
                    f"StarGuard Burst-Only Detection for {owner_str}/{repo_str}",
                    "=" * (35 + len(owner_str) + len(repo_str)), "",
                    f"Fake Star Index: {burst_result_dict.get('fake_star_index', 0.0):.2f} ({burst_result_dict.get('risk_level', 'N/A').upper()} RISK)",
                    f"Detected {len(burst_result_dict.get('bursts',[]))} suspicious bursts.",
                    f"Total Likely Fake Stars in Bursts: {burst_result_dict.get('total_likely_fake', 0)} ({burst_result_dict.get('fake_percentage', 0.0):.1f}%)", ""
                ]
                if burst_result_dict.get('bursts'):
                    lines.append("Top Suspicious Bursts (max 3 shown):")
                    for idx, burst_item in enumerate(burst_result_dict['bursts'][:3]):
                        lines.append(
                            f"  Burst {idx+1}: {burst_item.get('verdict','N/A').upper()} (Score: {burst_item.get('burst_score',0.0):.2f}), "
                            f"{burst_item.get('start_date','N/A')} to {burst_item.get('end_date','N/A')}, "
                            f"+{burst_item.get('stars',0)} stars"
                        )
                final_report_str = "\n".join(lines)

            if args.plot:
                # Ensure burst_detector_inst has its data populated for plotting
                if burst_detector_inst.stars_df is None: burst_detector_inst.build_daily_timeseries()
                if not burst_detector_inst.bursts: burst_detector_inst.detect_bursts() # Needed if calculate_fake_star_index wasn't run or failed early
                burst_detector_inst.plot_star_history(args.plot)
        
        else: # Full analysis
            starguard_engine = StarGuard(github_token)
            analysis_results_dict = starguard_engine.analyze_repo(owner_str, repo_str, analyze_fake_stars=not args.no_fake_stars)

            if "error" in analysis_results_dict: # Handle error from analyze_repo itself
                logger.error(f"Analysis failed: {analysis_results_dict['error']}")
                sys.exit(1)
            
            final_report_str = starguard_engine.generate_report(analysis_results_dict, format_str=args.format)

            if args.plot:
                # For full analysis, plot needs access to the StarAnalyzer's BurstDetector instance or similar data.
                # The StarGuard.analyze_repo would need to return the relevant analyzer instance or data for plotting.
                # This is a bit complex to pass around. For simplicity, instantiate a new one for plot if needed.
                # Or, could modify StarGuard.analyze_repo to return the StarAnalyzer instance.
                # Quick solution: recreate for plot.
                logger.info(f"Generating plot for {owner_str}/{repo_str}...")
                plot_api_inst = GitHubAPI(github_token) # New API instance for plot to be safe
                
                # Use BurstDetector directly for plotting as it's the most comprehensive.
                # Data fetching for plot is separate from main analysis to ensure plot has what it needs.
                plot_burst_detector = BurstDetector(owner_str, repo_str, plot_api_inst)
                # Populate data needed for plot_star_history
                if plot_burst_detector.stars_df is None: plot_burst_detector.build_daily_timeseries()
                # plot_star_history will call calculate_fake_star_index if bursts are not populated
                plot_burst_detector.plot_star_history(args.plot)


        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(final_report_str)
            logger.info(f"Report saved to {args.output}")
        else:
            sys.stdout.write(final_report_str + "\n") # Ensure newline at end

    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user.")
        sys.exit(130) # Standard exit code for Ctrl+C
    except Exception as e:
        logger.error(f"An critical error occurred: {str(e)}", exc_info=args.verbose)
        if not args.verbose:
            logger.error("Run with -v or --verbose for detailed traceback.")
        sys.exit(1)

if __name__ == "__main__":
    main()
