"""
GitHub API client for StarGuard.

This module provides a comprehensive interface to the GitHub API for retrieving
repository metadata, stars, contributors, and other information needed
for repository analysis.
"""

import time
import logging
from typing import Dict, List, Optional, Union

import requests

from starguard.core.constants import (
    GITHUB_API_BASE,
    GITHUB_GRAPHQL_URL,
)
from starguard.api.token_manager import TokenManager
from starguard.api.api_cache import ApiCache

# Configure module logger
logger = logging.getLogger(__name__)


class GitHubAPI:
    """
    GitHub API client for StarGuard.

    This class handles all direct interactions with the GitHub REST and GraphQL APIs,
    providing methods to fetch repository data, contributors, stars, and other
    information needed for repository analysis.

    Attributes:
        token_manager: Manager for multiple GitHub tokens and rate limits
        cache: Cache for API responses
        headers: HTTP headers for REST API requests
        graphql_headers: HTTP headers for GraphQL API requests
        session: Persistent session for making HTTP requests
        user_profile_cache: Cache for user profile data
    """

    def __init__(
        self, tokens: Optional[Union[str, List[str]]] = None, rate_limit_pause: bool = True
    ):
        """
        Initialize the GitHub API client.

        Args:
            tokens: GitHub personal access token(s) for authentication.
                  Single token as string or list of multiple tokens.
            rate_limit_pause: If True, pause execution when rate limits are hit
                             instead of returning an error.
        """
        # Initialize token manager
        self.token_manager = TokenManager(tokens)
        self.rate_limit_pause = rate_limit_pause

        # Initialize API cache
        self.cache = ApiCache()

        # Set up headers for REST API requests
        self.headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

        # Create persistent session for better performance
        self.session = requests.Session()
        self.session.headers.update(self.headers)

        # Cache for user profile data to avoid repeated API calls
        self.user_profile_cache = {}

        logger.debug(
            "GitHub API client initialized with %d token(s)", self.token_manager.token_count
        )

    def _update_auth_header(self, headers: Dict[str, str]) -> Dict[str, str]:
        """Update headers with current token's authorization."""
        current_token = self.token_manager.get_current_token()
        if current_token:
            headers["Authorization"] = f"token {current_token}"
        return headers

    def request(
        self,
        endpoint: str,
        method: str = "GET",
        params: Optional[Dict] = None,
        data: Optional[Dict] = None,
        use_cache: bool = True,
    ) -> Dict:
        """
        Make a request to the GitHub REST API with retries and error handling.

        This is the core method for all REST API interactions, handling:
        - Rate limiting (with backoff)
        - Token rotation
        - Caching
        - Retries for network errors and server errors
        - Error response parsing

        Args:
            endpoint: API endpoint path (will be appended to the API base URL)
            method: HTTP method to use (GET, POST, etc.)
            params: URL query parameters
            data: Request body data (for POST, PUT, etc.)
            use_cache: Whether to use cached response if available

        Returns:
            Dict: API response parsed from JSON, or error information
        """
        url = f"{GITHUB_API_BASE}{endpoint}"
        max_retries = 3
        retry_count = 0

        # Try to get from cache for GET requests
        if use_cache and method == "GET":
            cached_data = self.cache.get(endpoint, params)
            if cached_data:
                return cached_data

        while retry_count < max_retries:
            try:
                # Update headers with current token
                request_headers = self._update_auth_header(self.headers.copy())

                # Make the request with the session
                response = self.session.request(
                    method, url, params=params, json=data, headers=request_headers, timeout=30
                )

                # Handle rate limiting
                if response.status_code == 403 and "X-RateLimit-Remaining" in response.headers:
                    remaining = int(response.headers.get("X-RateLimit-Remaining", 0))
                    reset_time = int(response.headers.get("X-RateLimit-Reset", 0))

                    if remaining <= 1:
                        # Try rotating to next token
                        if self.token_manager.rotate_token():
                            logger.debug("Rotated to next token due to rate limit")
                            continue

                        # If no more tokens, wait if rate_limit_pause is True
                        if self.rate_limit_pause:
                            wait_time = self._calculate_wait_time(reset_time)
                            logger.warning(
                                f"Rate limit hit on all tokens. Waiting {wait_time:.0f}s before retry."
                            )
                            time.sleep(wait_time)
                            continue

                # Process response based on status code
                if response.status_code == 200:
                    response_data = response.json()

                    # Cache successful GET responses
                    if use_cache and method == "GET":
                        self.cache.put(endpoint, params, response_data)

                    return response_data

                elif response.status_code == 204:  # No content
                    return {}

                elif response.status_code == 404:
                    logger.debug(f"Resource not found (404): {url}")
                    return {"error": "Not Found", "status_code": 404}

                else:
                    # Handle server errors with retry
                    if response.status_code >= 500:
                        retry_count += 1
                        wait_time = 2 * (retry_count + 1)
                        logger.warning(
                            f"Server error ({response.status_code}) from {url}. "
                            f"Retrying {retry_count}/{max_retries} in {wait_time}s..."
                        )
                        time.sleep(wait_time)
                        continue

                    # For other client errors
                    logger.error(
                        f"GitHub API client error: {response.status_code} - {response.text}"
                    )
                    return {"error": response.text, "status_code": response.status_code}

            except requests.exceptions.RequestException as e:
                # Handle network errors, timeouts, etc.
                retry_count += 1
                wait_time = 2 * (retry_count + 1)
                logger.warning(f"Network error for {url}: {str(e)}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
                continue

        # If we get here, all retries failed
        logger.error(f"Failed to make request to {url} after {max_retries} attempts")
        return {"error": f"Failed after {max_retries} retries", "status_code": 0}

    def graphql_request(
        self, query: str, variables: Optional[Dict] = None, use_cache: bool = True
    ) -> Dict:
        """
        Make a GraphQL request to the GitHub API.

        Handles GraphQL-specific error formats, token rotation, and caching.

        Args:
            query: GraphQL query string
            variables: Variables for the GraphQL query
            use_cache: Whether to use cached response if available

        Returns:
            Dict: Parsed response data or error information
        """
        variables = variables or {}

        # Try to get from cache
        if use_cache:
            cache_key = f"gql:{hash(query)}:{hash(str(variables))}"
            cached_data = self.cache.get_raw(cache_key)
            if cached_data:
                return cached_data

        json_data = {"query": query, "variables": variables}
        max_retries = 3
        retry_count = 0

        while retry_count < max_retries:
            try:
                # Get current token for authorization header
                current_token = self.token_manager.get_current_token()
                headers = {"Authorization": f"bearer {current_token}"} if current_token else {}

                # Make the GraphQL request
                response = requests.post(
                    GITHUB_GRAPHQL_URL, json=json_data, headers=headers, timeout=30
                )

                # Check for rate limiting in GraphQL response
                if response.status_code == 200:
                    result = response.json()

                    # Look for rate limit error in GraphQL response
                    rate_limit_error = False
                    if "errors" in result:
                        for error in result.get("errors", []):
                            if "RATE_LIMITED" in str(error) or "rate limit" in str(error).lower():
                                rate_limit_error = True
                                break

                    if rate_limit_error:
                        # Try rotating to next token
                        if self.token_manager.rotate_token():
                            logger.debug("Rotated to next token due to GraphQL rate limit")
                            continue

                        # If no more tokens, wait if rate_limit_pause is True
                        if self.rate_limit_pause:
                            wait_time = 60  # Default wait for GraphQL rate limit
                            logger.warning(
                                f"GraphQL rate limit hit on all tokens. Waiting {wait_time}s before retry."
                            )
                            time.sleep(wait_time)
                            continue

                if response.status_code == 200:
                    result = response.json()

                    # Cache successful responses
                    if use_cache and "errors" not in result:
                        self.cache.put_raw(cache_key, result)

                    return result

                else:
                    # Handle HTTP errors
                    if response.status_code >= 500:
                        # Server error, retry
                        retry_count += 1
                        wait_time = 2 * (retry_count + 1)
                        logger.warning(
                            f"GraphQL server error ({response.status_code}). Retrying in {wait_time}s..."
                        )
                        time.sleep(wait_time)
                        continue

                    # Other HTTP errors
                    error_message = (
                        f"GraphQL API HTTP error: {response.status_code} - {response.text}"
                    )
                    logger.error(error_message)
                    return {"data": {}, "errors": [{"message": error_message}]}

            except requests.exceptions.RequestException as e:
                # Handle network errors
                retry_count += 1
                wait_time = 2 * (retry_count + 1)
                logger.warning(f"GraphQL network error: {str(e)}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
                continue

        # All retries failed
        logger.error(f"Failed to make GraphQL request after {max_retries} attempts")
        return {"data": {}, "errors": [{"message": f"Failed after {max_retries} retries"}]}

    def _calculate_wait_time(self, reset_time: int) -> float:
        """
        Calculate how long to wait for rate limit reset with adaptive backoff.

        Args:
            reset_time: UNIX timestamp when rate limit resets

        Returns:
            float: Number of seconds to wait
        """
        # Calculate time until reset
        now = time.time()
        full_wait_time = max(0, reset_time - now)

        # If reset time is in the past or very close, use a short wait
        if full_wait_time <= 5:
            return 5

        # Implement graduated backoff strategy
        if full_wait_time < 120:  # Less than 2 minutes
            wait_time = full_wait_time / 3
        elif full_wait_time < 600:  # Less than 10 minutes
            wait_time = min(60, full_wait_time / 10)
        else:  # Long wait
            wait_time = min(120, full_wait_time / 60)

        return wait_time
