"""Additional GitHub API methods for specific data types."""

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class GitHubApiMethods:
    """
    Implementation of GitHub API methods for specific data types.

    This class provides methods for fetching various types of repository data
    like contributors, commits, issues, etc.
    """

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

    def get_commits(
        self, owner: str, repo: str, since: Optional[str] = None, until: Optional[str] = None
    ) -> List[Dict]:
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
            logger.warning(
                f"Error fetching license for {owner}/{repo}: {license_data.get('error')}"
            )
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

    def check_user_repo_interaction(self, owner: str, repo: str, username: str) -> Dict[str, int]:
        """
        Check if a user has any prior interaction with a repository:
        issues, pull requests, or commits authored by them.

        Returns a summary dict with counts and a boolean flag.
        """
        # Initialize counts
        commit_count = 0
        issue_count = 0
        pr_count = 0

        # Check commits authored by the user
        try:
            commits = self.get_commits(owner, repo)
            commit_count = sum(1 for c in commits if c.get("author", {}).get("login") == username)
        except Exception:
            pass

        # Check issues opened by the user
        try:
            issues = self.get_issues(owner, repo, state="all")
            issue_count = sum(1 for i in issues if i.get("user", {}).get("login") == username)
        except Exception:
            pass

        # Check pull requests opened by the user
        try:
            prs = self.get_pulls(owner, repo, state="all")
            pr_count = sum(1 for p in prs if p.get("user", {}).get("login") == username)
        except Exception:
            pass

        total_interactions = commit_count + issue_count + pr_count
        return {
            "has_any_interaction": total_interactions > 0,
            "commit_count": commit_count,
            "issue_count": issue_count,
            "pr_count": pr_count,
        }

    def list_user_events(self, username: str, per_page: int = 30) -> List[Dict]:
        """
        Get a list of events performed by a user.
        
        Retrieves the most recent events performed by the specified user.
        This is used to analyze user activity patterns.
        
        Args:
            username: GitHub username
            per_page: Number of events per page (max 100)
            
        Returns:
            List[Dict]: List of event data
        """
        logger.debug(f"Fetching events for user {username}")
        endpoint = f"/users/{username}/events"
        params = {"per_page": min(per_page, 100)}
        
        try:
            events = self.paginate(endpoint, params=params)
            logger.debug(f"Retrieved {len(events)} events for user {username}")
            return events
        except Exception as e:
            logger.warning(f"Error fetching events for user {username}: {e}")
            return []

    def list_starred_repos(self, username: str, limit: int = 100) -> List[Dict]:
        """
        Get repositories starred by a user.
        
        Retrieves a list of repositories that have been starred by the
        specified user. Used to analyze star patterns and detect coordinated activity.
        
        Args:
            username: GitHub username
            limit: Maximum number of repos to return
            
        Returns:
            List[Dict]: List of starred repository data
        """
        logger.debug(f"Fetching starred repos for user {username}")
        endpoint = f"/users/{username}/starred"
        params = {"per_page": min(limit, 100)}
        
        try:
            starred = self.paginate(endpoint, params=params, max_items=limit)
            logger.debug(f"Retrieved {len(starred)} starred repos for user {username}")
            return starred
        except Exception as e:
            logger.warning(f"Error fetching starred repos for user {username}: {e}")
            return []

    def has_user_interacted(self, username: str, owner: str, repo: str) -> bool:
        """
        Check if a user has interacted with a repository before starring it.
        
        This is a simplified version of check_user_repo_interaction that returns
        just a boolean value. Used to identify users who star without any prior interaction.
        
        Args:
            username: GitHub username
            owner: Repository owner
            repo: Repository name
            
        Returns:
            bool: True if user has interacted with the repository, False otherwise
        """
        interaction_data = self.check_user_repo_interaction(owner, repo, username)
        return interaction_data.get("has_any_interaction", False)

    def get_traffic_views(self, owner: str, repo: str) -> Dict:
        """
        Get repository traffic view data.
        
        Retrieves view statistics for the specified repository.
        Requires push access to the repository.
        
        Args:
            owner: Repository owner
            repo: Repository name
            
        Returns:
            Dict: Repository traffic view data
        """
        logger.debug(f"Fetching traffic views for {owner}/{repo}")
        endpoint = f"/repos/{owner}/{repo}/traffic/views"
        
        try:
            view_data = self.request(endpoint)
            if "error" in view_data:
                logger.warning(f"Error fetching traffic views for {owner}/{repo}: {view_data.get('error')}")
                return {"views": []}
            return view_data
        except Exception as e:
            logger.warning(f"Error fetching traffic views for {owner}/{repo}: {e}")
            return {"views": []}

    def get_users_bulk(self, usernames: List[str]) -> Dict[str, Dict]:
        """
        Get profile information for multiple users efficiently.
        
        Retrieves profile data for multiple GitHub users in bulk,
        using cached results when available to minimize API calls.
        
        Args:
            usernames: List of GitHub usernames
            
        Returns:
            Dict[str, Dict]: Mapping of username to profile data
        """
        result = {}
        for username in usernames:
            result[username] = self.get_user(username)
        return result

    def paginate(self, endpoint: str, params: Optional[Dict] = None, max_items: Optional[int] = None) -> List[Dict]:
        """
        Retrieve all pages of results for a paginated API endpoint.
        
        Automatically handles GitHub's pagination to retrieve all results,
        or up to max_items if specified.
        
        Args:
            endpoint: API endpoint path
            params: URL query parameters
            max_items: Maximum number of items to retrieve across all pages
            
        Returns:
            List[Dict]: Combined results from all retrieved pages
        """
        # Initialize parameters for pagination
        params = params or {}
        if "per_page" not in params:
            params["per_page"] = 100  # Max items per page in GitHub API
        
        results = []
        page = 1
        
        while True:
            # Update params for current page
            page_params = params.copy()
            page_params["page"] = page
            
            # Make the request for this page
            response = self.request(endpoint, params=page_params)
            
            # Handle errors
            if isinstance(response, dict) and "error" in response:
                if response.get("status_code") == 404:
                    # Resource not found, return empty list
                    return []
                # Other error, log and return what we have so far
                logger.warning(f"Error during pagination for {endpoint}: {response.get('error')}")
                break
            
            # No items returned, we've reached the end
            if not response or (isinstance(response, list) and len(response) == 0):
                break
            
            # Add this page's results
            if isinstance(response, list):
                results.extend(response)
            else:
                # If response is not a list, it's likely an error or unexpected format
                logger.warning(f"Unexpected response format during pagination for {endpoint}")
                break
            
            # Check if we've reached max_items
            if max_items and len(results) >= max_items:
                results = results[:max_items]  # Truncate to max_items
                break
            
            # Check if we have a next page
            if len(response) < params["per_page"]:
                # Received fewer items than requested per page, must be the last page
                break
            
            # Move to next page
            page += 1
        
        return results