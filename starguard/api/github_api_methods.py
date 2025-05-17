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
