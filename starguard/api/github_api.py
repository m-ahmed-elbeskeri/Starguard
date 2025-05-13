"""
GitHub API client for StarGuard.

This module provides a comprehensive interface to the GitHub API for retrieving
repository metadata, stars, contributors, and other information needed
for repository analysis.
"""

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

# Configure module logger
logger = logging.getLogger(__name__)


class GitHubAPI:
    """
    GitHub API client for StarGuard.
    
    This class handles all direct interactions with the GitHub REST and GraphQL APIs,
    providing methods to fetch repository data, contributors, stars, and other
    information needed for repository analysis.
    
    Attributes:
        token (Optional[str]): GitHub personal access token for authentication
        rate_limit_pause (bool): Whether to pause execution when rate limits are hit
        headers (Dict[str, str]): HTTP headers for REST API requests
        graphql_headers (Dict[str, str]): HTTP headers for GraphQL API requests
        session (requests.Session): Persistent session for making HTTP requests
        user_profile_cache (Dict[str, Dict]): Cache for user profile data
        remaining_rate_limit (int): Current remaining API requests
        rate_limit_reset (int): Timestamp when rate limit resets
    """

    def __init__(self, token: Optional[str] = None, rate_limit_pause: bool = True):
        """
        Initialize the GitHub API client.

        Args:
            token: GitHub personal access token for authentication.
                  Using a token increases rate limits and enables access to private repos.
            rate_limit_pause: If True, pause execution when rate limits are hit
                             instead of returning an error.
        """
        self.token = token
        self.rate_limit_pause = rate_limit_pause
        
        # Set up headers for REST API requests
        self.headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28"
        }
        
        # Configure authentication if token is provided
        if token:
            self.headers["Authorization"] = f"token {token}"
            self.graphql_headers = {"Authorization": f"bearer {token}"}
        else:
            self.graphql_headers = {}

        # Create persistent session for better performance
        self.session = requests.Session()
        self.session.headers.update(self.headers)

        # Cache for user profile data to avoid repeated API calls
        self.user_profile_cache = {}
        
        # Rate limit tracking
        self.remaining_rate_limit = 5000  # Default GitHub API rate limit with token
        self.rate_limit_reset = 0
        
        logger.debug("GitHub API client initialized%s", 
                    " with authentication token" if token else " without authentication")

    def _handle_rate_limit(self, response: requests.Response) -> bool:
        """
        Handle GitHub API rate limiting with adaptive backoff strategy.
        
        Updates internal rate limit counters and implements a graduated
        waiting strategy when limits are hit, instead of waiting for
        the full reset time.

        Args:
            response: The HTTP response from a GitHub API request

        Returns:
            bool: True if we hit rate limit and had to wait, False otherwise
        """
        # Update rate limit info if headers are present
        if 'X-RateLimit-Remaining' in response.headers:
            self.remaining_rate_limit = int(response.headers['X-RateLimit-Remaining'])
            self.rate_limit_reset = int(response.headers.get('X-RateLimit-Reset', 0))
            
            # Log rate limit status if very low
            if self.remaining_rate_limit < 10:
                reset_time = datetime.datetime.fromtimestamp(self.rate_limit_reset)
                logger.debug("Rate limit low: %d remaining, resets at %s", 
                           self.remaining_rate_limit, reset_time)

        # Check if we've hit the rate limit (403 status with 0-1 remaining requests)
        if response.status_code == 403 and self.remaining_rate_limit <= 1 and self.rate_limit_pause:
            # Calculate time until reset
            now = time.time()
            full_wait_time = max(0, self.rate_limit_reset - now)
            
            # If reset time is in the past or very close, use a short wait
            if full_wait_time <= 5:
                logger.warning("Rate limit reached but reset time is imminent. Waiting 5 seconds.")
                time.sleep(5)
                return True
                
            # Implement graduated backoff strategy:
            # 1. For short waits (<2 min), wait 1/3 of the time
            # 2. For medium waits (2-10 min), wait up to 60 seconds
            # 3. For long waits (>10 min), wait proportionally (1/60 of the time)
            # This avoids extremely long pauses while respecting GitHub's limits
            
            if full_wait_time < 120:  # Less than 2 minutes
                wait_time = full_wait_time / 3
            elif full_wait_time < 600:  # Less than 10 minutes
                wait_time = min(60, full_wait_time / 10)
            else:  # Long wait
                wait_time = min(120, full_wait_time / 60)
            
            reset_datetime = datetime.datetime.fromtimestamp(self.rate_limit_reset)
            logger.warning(
                f"Rate limit hit. Remaining: {self.remaining_rate_limit}, "
                f"Reset: {reset_datetime.strftime('%Y-%m-%d %H:%M:%S')} "
                f"(in {full_wait_time:.0f}s). Waiting {wait_time:.0f}s before retry."
            )
            
            time.sleep(wait_time)
            return True
        
        return False

    def request(self, endpoint: str, method: str = "GET", params: Optional[Dict] = None, 
               data: Optional[Dict] = None) -> Dict:
        """
        Make a request to the GitHub REST API with retries and error handling.
        
        This is the core method for all REST API interactions, handling:
        - Rate limiting (with backoff)
        - Retries for network errors and server errors
        - Error response parsing
        
        Args:
            endpoint: API endpoint path (will be appended to the API base URL)
            method: HTTP method to use (GET, POST, etc.)
            params: URL query parameters
            data: Request body data (for POST, PUT, etc.)

        Returns:
            Dict: API response parsed from JSON, or error information
        """
        url = f"{GITHUB_API_BASE}{endpoint}"
        max_retries = 3
        retry_count = 0

        while retry_count < max_retries:
            try:
                # Make the request with the session (maintains headers and cookies)
                response = self.session.request(
                    method, url, 
                    params=params, 
                    json=data, 
                    timeout=30
                )

                # Handle rate limiting - may pause execution
                if self._handle_rate_limit(response):
                    continue  # Try the request again after waiting

                # Process response based on status code
                if response.status_code == 200:
                    return response.json()
                    
                elif response.status_code == 204:  # No content
                    return {}
                    
                elif response.status_code == 404:
                    # For some resources, 404 is a valid non-error state (e.g., license not found)
                    # Return structured data with error flag for caller to handle
                    logger.debug(f"Resource not found (404): {url}")
                    return {"error": "Not Found", "status_code": 404}
                    
                else:
                    # Handle server errors with retry
                    if response.status_code >= 500:
                        retry_count += 1
                        wait_time = 2 * (retry_count + 1)  # Exponential backoff
                        logger.warning(
                            f"Server error ({response.status_code}) from {url}. "
                            f"Retrying {retry_count}/{max_retries} in {wait_time}s..."
                        )
                        time.sleep(wait_time)
                        continue
                        
                    # For other client errors (4xx), log and return error structure
                    logger.error(
                        f"GitHub API client error: {response.status_code} - {response.text} "
                        f"for URL {url}"
                    )
                    return {
                        "error": response.text,
                        "status_code": response.status_code
                    }
                    
            except requests.exceptions.RequestException as e:
                # Handle network errors, timeouts, etc.
                retry_count += 1
                wait_time = 2 * (retry_count + 1)
                logger.warning(
                    f"Network error for {url}: {str(e)}. "
                    f"Retrying {retry_count}/{max_retries} in {wait_time}s..."
                )
                time.sleep(wait_time)
                continue

        # If we get here, all retries failed
        logger.error(f"Failed to make request to {url} after {max_retries} attempts")
        return {
            "error": f"Failed after {max_retries} retries",
            "status_code": 0
        }

    def graphql_request(self, query: str, variables: Optional[Dict] = None) -> Dict:
        """
        Make a GraphQL request to the GitHub API.
        
        Handles GraphQL-specific error formats and retries.
        
        Args:
            query: GraphQL query string
            variables: Variables for the GraphQL query

        Returns:
            Dict: Parsed response data or error information
        """
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
                # GraphQL uses POST method with the query in request body
                response = requests.post(
                    GITHUB_GRAPHQL_URL,
                    json=json_data,
                    headers=self.graphql_headers,
                    timeout=30
                )

                # Handle rate limiting
                if self._handle_rate_limit(response):
                    continue

                if response.status_code == 200:
                    result = response.json()

                    # GraphQL may return 200 status but have errors in the response
                    if "errors" in result:
                        error_message = result.get("errors", [{}])[0].get("message", "Unknown GraphQL error")
                        
                        # Special case: NOT_FOUND errors are not retryable
                        if "type': 'NOT_FOUND'" in str(result.get("errors")):
                            logger.warning(f"GraphQL query target not found: {error_message}")
                            return {"data": {}, "errors": result.get("errors")}

                        # For other errors, retry if we haven't reached max retries
                        if retry_count < max_retries - 1:
                            retry_count += 1
                            wait_time = 2 * (retry_count + 1)
                            logger.warning(
                                f"GraphQL error: {error_message}. "
                                f"Retrying {retry_count}/{max_retries} in {wait_time}s..."
                            )
                            time.sleep(wait_time)
                            continue
                        else:
                            logger.error(f"GraphQL API error after retries: {error_message}")
                            return {"data": {}, "errors": result.get("errors")}
                            
                    # Successful response with no errors
                    return result
                    
                else:
                    # Handle HTTP errors
                    if response.status_code >= 500:
                        # Server error, retry
                        retry_count += 1
                        wait_time = 2 * (retry_count + 1)
                        logger.warning(
                            f"GraphQL server error ({response.status_code}). "
                            f"Retrying {retry_count}/{max_retries} in {wait_time}s..."
                        )
                        time.sleep(wait_time)
                        continue
                        
                    # Other HTTP errors
                    error_message = f"GraphQL API HTTP error: {response.status_code} - {response.text}"
                    logger.error(error_message)
                    return {
                        "data": {},
                        "errors": [{
                            "message": error_message,
                            "status_code": response.status_code
                        }]
                    }
                    
            except requests.exceptions.RequestException as e:
                # Handle network errors
                retry_count += 1
                wait_time = 2 * (retry_count + 1)
                logger.warning(
                    f"GraphQL network error: {str(e)}. "
                    f"Retrying {retry_count}/{max_retries} in {wait_time}s..."
                )
                time.sleep(wait_time)
                continue

        # All retries failed
        logger.error(f"Failed to make GraphQL request after {max_retries} attempts")
        return {
            "data": {},
            "errors": [{"message": f"Failed after {max_retries} retries"}]
        }

    def paginate(self, endpoint: str, params: Optional[Dict] = None) -> List[Dict]:
        """
        Paginate through GitHub API results.
        
        Handles GitHub's pagination for endpoints that return list data.
        Continues fetching pages until there are no more results
        or until an error occurs.
        
        Args:
            endpoint: API endpoint path
            params: Base query parameters

        Returns:
            List[Dict]: Combined results from all pages
        """
        if params is None:
            params = {}

        # GitHub's pagination uses per_page and page parameters
        params["per_page"] = 100  # Maximum allowed by GitHub
        results = []
        page = 1

        while True:
            params["page"] = page
            try:
                page_data = self.request(endpoint, params=params)
                
                # Check for errors or empty response
                if not page_data or "error" in page_data:
                    if "error" in page_data:
                        logger.warning(
                            f"Error during pagination for {endpoint} "
                            f"(page {page}): {page_data.get('error')}"
                        )
                    break
                
                # Ensure page_data is a list before extending results
                if isinstance(page_data, list):
                    results.extend(page_data)
                    
                    # If we got fewer items than requested, we've reached the end
                    if len(page_data) < 100:
                        break
                        
                else:
                    # API returned unexpected data type
                    logger.warning(
                        f"Unexpected data type from {endpoint} "
                        f"(page {page}): {type(page_data)}"
                    )
                    break

                # Move to next page
                page += 1
                
            except Exception as e:
                # Catch any exceptions that weren't handled in request()
                logger.error(
                    f"Unhandled error while paginating {endpoint} "
                    f"(page {page}): {str(e)}"
                )
                break
                
        logger.debug(f"Pagination complete for {endpoint}: retrieved {len(results)} items")
        return results

    def get_repo(self, owner: str, repo: str) -> Dict:
        """
        Get comprehensive repository information.
        
        This is typically the first method called to analyze a repository,
        and returns basic metadata like stars, forks, creation date, etc.
        
        Args:
            owner: Repository owner (user or organization)
            repo: Repository name

        Returns:
            Dict: Repository metadata

        Raises:
            ValueError: If the repository doesn't exist or can't be accessed
        """
        logger.info(f"Fetching repository data for {owner}/{repo}")
        data = self.request(f"/repos/{owner}/{repo}")
        
        # Handle common error cases with clear exceptions
        if "error" in data and data.get("status_code") == 404:
            raise ValueError(f"Repository {owner}/{repo} not found.")
        if "error" in data:
            raise ValueError(f"Failed to fetch repo {owner}/{repo}: {data.get('error')}")
            
        logger.debug(f"Retrieved repo data for {owner}/{repo}: {len(data)} fields")
        return data

    def get_stargazers(self, owner: str, repo: str, get_timestamps: bool = True, 
                      days_limit: int = 0) -> List[Dict]:
        """
        Get repository stargazers with timestamps using REST API.
        
        This is a core method for fake star detection. It retrieves a list
        of users who have starred the repository, along with when they 
        starred it (if get_timestamps is True).

        Args:
            owner: Repository owner
            repo: Repository name
            get_timestamps: Whether to get timestamps (requires special Accept header)
            days_limit: Limit to stars from the last X days.
                        Note: The REST API fetches all stargazers; filtering by days_limit
                        is applied post-fetch if days_limit > 0. This can be inefficient for
                        repositories with a very large number of stars.

        Returns:
            List[Dict]: List of stargazer data with timestamps (if requested)
        """
        logger.info(f"Fetching stargazers for {owner}/{repo} with timestamps={get_timestamps}")
        try:
            # Use REST API to get stargazers (GraphQL implementation is commented out)
            all_stars = self._get_stargazers_rest(owner, repo, get_timestamps)

            # Apply days filter if requested
            if days_limit > 0 and all_stars:
                # Calculate cutoff date
                cutoff_date_dt = make_naive_datetime(
                    datetime.datetime.now() - datetime.timedelta(days=days_limit)
                )

                # Filter stars by date
                filtered_stars = []
                for star in all_stars:
                    starred_at_str = star.get("starred_at")
                    if starred_at_str:
                        try:
                            starred_at_dt = make_naive_datetime(parse_date(starred_at_str))
                            if starred_at_dt and cutoff_date_dt and starred_at_dt >= cutoff_date_dt:
                                filtered_stars.append(star)
                        except Exception as e:
                            logger.debug(
                                f"Could not parse or compare date for star: {starred_at_str}, "
                                f"error: {e}"
                            )
                
                logger.info(
                    f"Fetched {len(all_stars)} total stars, filtered to "
                    f"{len(filtered_stars)} within {days_limit} days."
                )
                return filtered_stars

            logger.info(f"Fetched {len(all_stars)} stars for {owner}/{repo}")
            return all_stars
            
        except Exception as e:
            logger.warning(f"Error fetching stargazers via REST API: {str(e)}")
            return []

    def _get_stargazers_graphql(self, owner: str, repo: str, days_limit: int = 0) -> List[Dict]:
        """
        Get stargazers using GraphQL API.
        
        Note: This method is currently not used, as the implementation is
        commented out. It's a placeholder for future use. The REST API
        implementation in _get_stargazers_rest is used instead.
        
        Args:
            owner: Repository owner
            repo: Repository name
            days_limit: Limit to stars from the last X days

        Returns:
            List[Dict]: Empty list (currently) or stargazer data when implemented
        """
        logger.debug("Attempting to fetch stargazers using GraphQL (bypassed in get_stargazers method).")
        # GraphQL implementation is commented out in the original code
        return []  # Currently bypassed

    def _get_stargazers_rest(self, owner: str, repo: str, get_timestamps: bool = True) -> List[Dict]:
        """
        Get repository stargazers using REST API.
        
        This method fetches users who have starred the repository, optionally
        with timestamps of when they starred it.
        
        Args:
            owner: Repository owner
            repo: Repository name
            get_timestamps: Whether to get timestamps (requires special Accept header)

        Returns:
            List[Dict]: List of stargazer data with timestamps (if requested)
        """
        # For star timestamps, we need a special Accept header
        headers = self.headers.copy()
        if get_timestamps:
            headers["Accept"] = "application/vnd.github.v3.star+json"
            logger.debug("Using star+json Accept header for timestamp data")

        # Endpoint for stargazers
        endpoint = f"/repos/{owner}/{repo}/stargazers"
        
        # Fetch all pages of stargazers
        raw_stars = self.paginate(endpoint)  # Uses self.session which has headers

        # Process raw star data into a consistent format
        processed_stars = []
        for star_entry in raw_stars:
            try:
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
                elif "login" in star_entry:
                    # Fallback for non-timestamped structure (user objects directly in list)
                    processed_star = {
                        "starred_at": "2020-01-01T00:00:00Z",  # Placeholder
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
                    continue  # Skip malformed entries
                
                processed_stars.append(processed_star)
            except Exception as e:
                logger.debug(f"Error processing star entry: {e}")
                
        logger.debug(f"Processed {len(processed_stars)} star entries")
        return processed_stars

    def get_forks(self, owner: str, repo: str) -> List[Dict]:
        """
        Get repository forks.
        
        Retrieves a list of repositories that are forks of the specified repository.
        
        Args:
            owner: Repository owner
            repo: Repository name

        Returns:
            List[Dict]: List of fork data
        """
        logger.debug(f"Fetching forks for {owner}/{repo}")
        endpoint = f"/repos/{owner}/{repo}/forks"
        forks = self.paginate(endpoint)
        logger.debug(f"Retrieved {len(forks)} forks for {owner}/{repo}")
        return forks

    def get_issues(self, owner: str, repo: str, state: str = "all") -> List[Dict]:
        """
        Get repository issues.
        
        Retrieves a list of issues for the specified repository.
        
        Args:
            owner: Repository owner
            repo: Repository name
            state: Issue state: "open", "closed", or "all" (default)

        Returns:
            List[Dict]: List of issue data
        """
        logger.debug(f"Fetching {state} issues for {owner}/{repo}")
        endpoint = f"/repos/{owner}/{repo}/issues"
        issues = self.paginate(endpoint, {"state": state})
        logger.debug(f"Retrieved {len(issues)} {state} issues for {owner}/{repo}")
        return issues

    def get_pulls(self, owner: str, repo: str, state: str = "all") -> List[Dict]:
        """
        Get repository pull requests.
        
        Retrieves a list of pull requests for the specified repository.
        
        Args:
            owner: Repository owner
            repo: Repository name
            state: PR state: "open", "closed", or "all" (default)

        Returns:
            List[Dict]: List of pull request data
        """
        logger.debug(f"Fetching {state} pull requests for {owner}/{repo}")
        endpoint = f"/repos/{owner}/{repo}/pulls"
        pulls = self.paginate(endpoint, {"state": state})
        logger.debug(f"Retrieved {len(pulls)} {state} pull requests for {owner}/{repo}")
        return pulls

    def get_contributors(self, owner: str, repo: str) -> List[Dict]:
        """
        Get repository contributors.
        
        Retrieves a list of contributors to the specified repository,
        including their contribution counts.
        
        Args:
            owner: Repository owner
            repo: Repository name

        Returns:
            List[Dict]: List of contributor data
        """
        logger.debug(f"Fetching contributors for {owner}/{repo}")
        endpoint = f"/repos/{owner}/{repo}/contributors"
        contributors = self.paginate(endpoint)
        logger.debug(f"Retrieved {len(contributors)} contributors for {owner}/{repo}")
        return contributors

    def get_commits(self, owner: str, repo: str, since: Optional[str] = None, 
                   until: Optional[str] = None) -> List[Dict]:
        """
        Get repository commits.
        
        Retrieves a list of commits for the specified repository,
        optionally filtered by date range.
        
        Args:
            owner: Repository owner
            repo: Repository name
            since: ISO 8601 date string to filter commits after this date
            until: ISO 8601 date string to filter commits before this date

        Returns:
            List[Dict]: List of commit data
        """
        logger.debug(f"Fetching commits for {owner}/{repo} (since={since}, until={until})")
        endpoint = f"/repos/{owner}/{repo}/commits"
        params = {}
        if since:
            params["since"] = since
        if until:
            params["until"] = until
            
        commits = self.paginate(endpoint, params=params)
        logger.debug(f"Retrieved {len(commits)} commits for {owner}/{repo}")
        return commits

    def get_traffic_views(self, owner: str, repo: str) -> Dict:
        """
        Get repository traffic views (requires push access).
        
        Note: This endpoint requires push access to the repository.
        For repositories the token doesn't have push access to,
        a default structure is returned.
        
        Args:
            owner: Repository owner
            repo: Repository name

        Returns:
            Dict: Traffic view data or default structure if unauthorized
        """
        logger.debug(f"Fetching traffic views for {owner}/{repo}")
        endpoint = f"/repos/{owner}/{repo}/traffic/views"
        data = self.request(endpoint)
        
        if "error" in data:  # Handles 403 if no push access, or other errors
            logger.debug(f"Could not fetch traffic views for {owner}/{repo}: {data.get('error')}")
            return {"count": 0, "uniques": 0, "views": []}
            
        return data

    def get_traffic_clones(self, owner: str, repo: str) -> Dict:
        """
        Get repository traffic clones (requires push access).
        
        Note: This endpoint requires push access to the repository.
        For repositories the token doesn't have push access to,
        a default structure is returned.
        
        Args:
            owner: Repository owner
            repo: Repository name

        Returns:
            Dict: Traffic clone data or default structure if unauthorized
        """
        logger.debug(f"Fetching traffic clones for {owner}/{repo}")
        endpoint = f"/repos/{owner}/{repo}/traffic/clones"
        data = self.request(endpoint)
        
        if "error" in data:
            logger.debug(f"Could not fetch traffic clones for {owner}/{repo}: {data.get('error')}")
            return {"count": 0, "uniques": 0, "clones": []}
            
        return data
        
    def get_releases(self, owner: str, repo: str) -> List[Dict]:
        """
        Get repository releases.
        
        Retrieves a list of releases for the specified repository.
        
        Args:
            owner: Repository owner
            repo: Repository name

        Returns:
            List[Dict]: List of release data
        """
        logger.debug(f"Fetching releases for {owner}/{repo}")
        endpoint = f"/repos/{owner}/{repo}/releases"
        releases = self.paginate(endpoint)
        logger.debug(f"Retrieved {len(releases)} releases for {owner}/{repo}")
        return releases

    def get_user(self, username: str) -> Dict:
        """
        Get a user's profile information.
        
        Retrieves profile data for a GitHub user, including
        creation date, followers, and public repositories.
        Uses an internal cache to avoid repeated API calls.
        
        Args:
            username: GitHub username

        Returns:
            Dict: User profile data or empty dict if not found
        """
        # Check cache first
        if username in self.user_profile_cache:
            logger.debug(f"Using cached user profile for {username}")
            return self.user_profile_cache[username]

        logger.debug(f"Fetching user profile for {username}")
        endpoint = f"/users/{username}"
        user_data = self.request(endpoint)
        
        if "error" in user_data:  # Handles 404 or other errors
            logger.debug(f"Error fetching user {username}: {user_data.get('error')}")
            return {}  # Return empty if error or not found
            
        # Cache the result for future use
        self.user_profile_cache[username] = user_data
        return user_data

    def get_user_events(self, username: str, limit: int = 10) -> List[Dict]:
        """
        Get a user's public events (limited to conserve API quota).
        
        Retrieves recent public activities for a GitHub user.
        
        Args:
            username: GitHub username
            limit: Maximum number of events to retrieve

        Returns:
            List[Dict]: List of user event data
        """
        logger.debug(f"Fetching up to {limit} recent events for user {username}")
        endpoint = f"/users/{username}/events/public"
        params = {"per_page": min(limit, 100)}
        events_data = self.request(endpoint, params=params)

        if "error" in events_data or not isinstance(events_data, list):
            logger.debug(
                f"Error fetching events for user {username}: "
                f"{events_data.get('error', 'Not a list')}"
            )
            return []
            
        # Return limited number of events
        return events_data[:limit]

    def check_user_repo_interaction(self, owner: str, repo: str, username: str) -> Dict:
        """
        Check if a user has interacted with a repository.
        
        Determines if a user has opened issues, created PRs, or made commits
        to the specified repository. Used to evaluate the relationship between
        users and repositories they star.
        
        Args:
            owner: Repository owner
            repo: Repository name
            username: GitHub username to check

        Returns:
            Dict: Interaction flags for issues, PRs, and commits
        """
        logger.debug(f"Checking interactions between user {username} and repo {owner}/{repo}")
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

            # Check for PRs (using same search endpoint with different query)
            params_prs = {"q": f"repo:{owner}/{repo} author:{username} type:pr", "per_page": 1}
            prs_result = self.request(issues_endpoint, params=params_prs)
            if "total_count" in prs_result:
                interactions["has_prs"] = prs_result["total_count"] > 0

            # Check for commits
            commits_endpoint = f"/repos/{owner}/{repo}/commits"
            params_commits = {"author": username, "per_page": 1}
            commits_result = self.request(commits_endpoint, params=params_commits)
            if isinstance(commits_result, list):  # Check if it's a list (successful) vs error dict
                 interactions["has_commits"] = len(commits_result) > 0
                 
        except Exception as e:
            logger.debug(f"Error checking user interaction for {username} on {owner}/{repo}: {e}")
        
        # Summary flag for any interaction
        interactions["has_any_interaction"] = (
            interactions["has_issues"] or
            interactions["has_prs"] or
            interactions["has_commits"]
        )
        
        logger.debug(f"Interaction check results for {username} on {owner}/{repo}: {interactions}")
        return interactions

    def get_file_content(self, owner: str, repo: str, path: str, 
                        ref: Optional[str] = None) -> Optional[str]:
        """
        Get file content from a repository.
        
        Retrieves and decodes the contents of a file from a GitHub repository.
        
        Args:
            owner: Repository owner
            repo: Repository name
            path: Path to the file within the repository
            ref: Git reference (branch, tag, or commit SHA) to get the file from

        Returns:
            Optional[str]: Decoded file content or None if not found/error
        """
        logger.debug(f"Fetching file content for {path} in {owner}/{repo}")
        endpoint = f"/repos/{owner}/{repo}/contents/{path}"
        params = {}
        if ref:
            params["ref"] = ref

        response_data = self.request(endpoint, params=params)
        if "error" in response_data or "content" not in response_data:
            logger.debug(
                f"Could not fetch file content for {path} in {owner}/{repo}: "
                f"{response_data.get('error', 'No content field')}"
            )
            return None
        
        try:
            # GitHub returns base64-encoded content
            content = base64.b64decode(response_data["content"]).decode("utf-8")
            logger.debug(f"Successfully decoded {len(content)} bytes of content from {path}")
            return content
        except (UnicodeDecodeError, base64.binascii.Error) as e:
            logger.debug(f"Error decoding file content for {path}: {e}")
            return None

    def get_dependencies(self, owner: str, repo: str) -> Dict:
        """
        Get repository dependencies using the dependency graph API or file parsing.
        
        Attempts to use GitHub's dependency graph API first, then falls back
        to parsing manifest files (package.json, requirements.txt, etc.)
        if the API isn't available.
        
        Args:
            owner: Repository owner
            repo: Repository name

        Returns:
            Dict: Dependency information either from API or file parsing
        """
        logger.info(f"Fetching dependencies for {owner}/{repo}")
        endpoint = f"/repos/{owner}/{repo}/dependency-graph/sbom"
        try:
            # Try the dependency graph API first
            # Note: This API requires specific permissions and may not always be available
            logger.debug(f"Attempting to use dependency graph API for {owner}/{repo}")
            response = self.request(endpoint)
            
            if "error" not in response and "sbom" in response:
                logger.info(f"Successfully fetched dependencies via API for {owner}/{repo}")
                return response
                
            logger.warning(
                f"Could not fetch dependencies via API for {owner}/{repo} "
                f"(Reason: {response.get('error', 'No SBOM field')}). "
                f"Falling back to file parsing."
            )
            
        except Exception as e:
            logger.warning(
                f"Exception fetching dependencies via API for {owner}/{repo}: "
                f"{str(e)}. Falling back to file parsing."
            )
        
        # Fallback: Parse manifest files
        return self._parse_dependencies_from_files(owner, repo)

    def _parse_dependencies_from_files(self, owner: str, repo: str) -> Dict:
        """
        Parse dependencies from manifest files.
        
        Checks for package manager manifest files (package.json, requirements.txt, etc.)
        and parses them to extract dependency information.
        
        Args:
            owner: Repository owner
            repo: Repository name

        Returns:
            Dict: Parsed dependency information grouped by language
        """
        logger.info(f"Parsing dependency files for {owner}/{repo}")
        dependencies = {}

        for lang, config in PACKAGE_MANAGERS.items():
            try:
                logger.debug(f"Checking for {lang} dependency file: {config['file']}")
                content = self.get_file_content(owner, repo, config["file"])
                
                if content:
                    logger.debug(f"Found {lang} dependency file, parsing...")
                    deps = self._parse_dependency_file(content, config, lang)
                    
                    if deps:
                        dependencies[lang] = deps
                        logger.debug(f"Parsed {len(deps)} {lang} dependencies")
                        
            except Exception as e:
                logger.debug(f"Could not parse {config['file']} for {owner}/{repo}: {str(e)}")
                
        logger.info(f"Parsed dependencies for {len(dependencies)} languages in {owner}/{repo}")
        return dependencies

    def _parse_dependency_file(self, content: str, config: Dict, lang: str) -> List[Dict]:
        """
        Parse a dependency file based on its format.
        
        Uses language-specific parsing logic to extract dependencies
        from manifest files.
        
        Args:
            content: File content to parse
            config: Language-specific configuration from PACKAGE_MANAGERS
            lang: Language identifier

        Returns:
            List[Dict]: Parsed dependency information
        """
        deps = []
        if not content:
            return deps

        # JavaScript (package.json)
        if lang == "javascript":
            try:
                package_json = json.loads(content)
                for key in config["dependencies_key"]:
                    if key in package_json and isinstance(package_json[key], dict):
                        for name, version in package_json[key].items():
                            deps.append({
                                "name": name,
                                "version": str(version),
                                "type": "runtime" if key == "dependencies" else "development"
                            })
            except json.JSONDecodeError as e:
                logger.debug(f"JSONDecodeError parsing package.json: {e}")

        # Python (requirements.txt)
        elif lang == "python":
            pattern = re.compile(config["pattern"])
            for line in content.splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    match = pattern.match(line)
                    if match:
                        name = match.group(1)
                        version = match.group(2) if match.group(2) else "latest"
                        deps.append({"name": name, "version": version, "type": "runtime"})

        # Python (Pipfile)
        elif lang == "python_pipfile":
            in_section = False
            for line in content.splitlines():
                line = line.strip()
                if config["section_marker"] in line:
                    in_section = True
                    continue
                if line.startswith("[") and in_section:  # Reached another section
                    in_section = False
                    continue
                if in_section and not line.startswith("#") and "=" in line:
                    parts = line.split("=", 1)
                    name = parts[0].strip()
                    version = parts[1].strip().strip('"\'')
                    deps.append({"name": name, "version": version, "type": "runtime"})
        
        # Python (pyproject.toml) - Poetry
        elif lang == "python_poetry":
            in_section = False
            for line in content.splitlines():
                line = line.strip()
                if config["section_marker"] in line:
                    in_section = True
                    continue
                if line.startswith("[") and in_section:  # Reached another section
                    in_section = False
                    continue
                if in_section and not line.startswith("#") and "=" in line:
                    parts = line.split("=", 1)
                    name = parts[0].strip().strip('"\'')  # Name can be quoted
                    version_spec = parts[1].strip()
                    
                    # Extract version, handling complex dicts like {version = "^1.0", optional = true}
                    if version_spec.startswith("{"):
                        try:
                            # Simplified TOML dict parsing
                            if "version" in version_spec:
                                v_match = re.search(r"version\s*=\s*[\"']([^\"']+)[\"']", version_spec)
                                version = v_match.group(1) if v_match else "latest"
                            else:
                                # Just take the whole string as a specifier
                                version = version_spec.strip('"\'')
                        except Exception:
                            version = version_spec.strip('"\'')  # Fallback
                    else:
                        version = version_spec.strip('"\'')
                        
                    deps.append({"name": name, "version": version, "type": "runtime"})

        # Ruby (Gemfile)
        elif lang == "ruby":
            pattern = re.compile(config["pattern"])
            for line in content.splitlines():
                line = line.strip()
                if line.startswith("gem"):
                    match = pattern.search(line)
                    if match:
                        name = match.group(1)
                        version = match.group(2) if len(match.groups()) > 1 and match.group(2) else "latest"
                        deps.append({"name": name, "version": version, "type": "runtime"})

        # Java (pom.xml)
        elif lang == "java":
            # Note: regex parsing for XML is fragile, a proper XML parser would be better
            pattern = re.compile(config["tag_pattern"], re.DOTALL | re.IGNORECASE)
            matches = pattern.findall(content)
            for match_tuple in matches:
                # Handle optional version
                group_id = match_tuple[0]
                artifact_id = match_tuple[1]
                version = match_tuple[2] if len(match_tuple) > 2 and match_tuple[2] else "latest"
                deps.append({"name": f"{group_id}:{artifact_id}", "version": version, "type": "runtime"})

        # Go (go.mod)
        elif lang == "go":
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
                
                if line.startswith("require") and not in_require_block:  # Single line require
                    match = pattern.match(line)
                    if match:
                        name, version = match.group(1), match.group(2)
                        deps.append({"name": name, "version": version, "type": "runtime"})
                elif in_require_block:  # Inside multi-line require block
                    parts = line.split()
                    if len(parts) == 2:  # e.g., "github.com/user/repo v1.2.3"
                        name, version = parts[0], parts[1]
                        if version.startswith("v"):
                            version = version[1:]  # Strip leading 'v'
                        deps.append({"name": name, "version": version, "type": "runtime"})

        return deps

    def get_license(self, owner: str, repo: str) -> Dict:
        """
        Get repository license information.
        
        Retrieves license details for a repository, normalizing
        response formats and handling cases where no license is found.
        
        Args:
            owner: Repository owner
            repo: Repository name

        Returns:
            Dict: License information with consistent structure
        """
        logger.info(f"Fetching license information for {owner}/{repo}")
        endpoint = f"/repos/{owner}/{repo}/license"
        
        # This request often returns 404 if no license file is detected by GitHub.
        license_data = self.request(endpoint)
        
        # Handle 404 - no license found
        if "error" in license_data and license_data.get("status_code") == 404:
            logger.info(f"No license file found by GitHub API for {owner}/{repo}.")
            return {"license": {"spdx_id": "unknown", "key": "unknown", "name": "Unknown"}}
            
        # Handle other errors
        elif "error" in license_data:
            logger.warning(f"Error fetching license for {owner}/{repo}: {license_data.get('error')}")
            return {"license": {"spdx_id": "unknown", "key": "unknown", "name": "Unknown"}}
        
        # Ensure the expected structure exists
        if "license" not in license_data or not isinstance(license_data["license"], dict):
            logger.warning(f"Malformed license data for {owner}/{repo}: {license_data}")
            return {"license": {"spdx_id": "unknown", "key": "unknown", "name": "Unknown"}}
        
        # Ensure required fields exist
        if "spdx_id" not in license_data["license"]:
            license_data["license"]["spdx_id"] = "unknown"
        if "key" not in license_data["license"]:
            license_data["license"]["key"] = "unknown"
        if "name" not in license_data["license"]:
            license_data["license"]["name"] = "Unknown"
            
        logger.debug(
            f"License data retrieved for {owner}/{repo}: "
            f"{license_data['license'].get('spdx_id')} ({license_data['license'].get('key')})"
        )
        return license_data
