"""GitHub API client for StarGuard."""

import os
import sys
import json
import datetime
import time
import re
import logging
import base64
from typing import Dict, List, Tuple, Any, Optional, Union
from collections import Counter, defaultdict
from urllib.parse import urlparse, quote_plus

import requests
from dateutil.parser import parse as parse_date

from starguard.utils.date_utils import make_naive_datetime
from starguard.core.constants import (
    GITHUB_API_BASE,
    GITHUB_GRAPHQL_URL,
    LICENSE_RISK_LEVELS,
    PACKAGE_MANAGERS
)

# Configure logging
logger = logging.getLogger(__name__)


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

