"""GitHub REST API specific methods."""

import logging
from typing import Dict, List, Optional

from starguard.utils.date_utils import make_naive_datetime
from dateutil.parser import parse as parse_date

logger = logging.getLogger(__name__)


class GitHubRestMethods:
    """
    Implementation of GitHub REST API methods.

    This class extends the core GitHubAPI with specific methods for
    the REST API. The methods are imported into the main GitHubAPI class.
    """

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
                        f"Unexpected data type from {endpoint} " f"(page {page}): {type(page_data)}"
                    )
                    break

                # Move to next page
                page += 1

            except Exception as e:
                # Catch any exceptions that weren't handled in request()
                logger.error(
                    f"Unhandled error while paginating {endpoint} " f"(page {page}): {str(e)}"
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

    def get_stargazers(
        self, owner: str, repo: str, get_timestamps: bool = True, days_limit: int = 0
    ) -> List[Dict]:
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
                        is applied post-fetch if days_limit > 0.

        Returns:
            List[Dict]: List of stargazer data with timestamps (if requested)
        """
        logger.info(f"Fetching stargazers for {owner}/{repo} with timestamps={get_timestamps}")
        try:
            # For star timestamps, we need a special Accept header
            headers_backup = self.session.headers.copy()

            if get_timestamps:
                self.session.headers["Accept"] = "application/vnd.github.v3.star+json"
                logger.debug("Using star+json Accept header for timestamp data")

            # Endpoint for stargazers
            endpoint = f"/repos/{owner}/{repo}/stargazers"

            # Fetch all pages of stargazers
            raw_stars = self.paginate(endpoint)

            # Restore original headers
            self.session.headers = headers_backup

            # Process raw star data into a consistent format
            processed_stars = []
            for star_entry in raw_stars:
                try:
                    if get_timestamps and "starred_at" in star_entry and "user" in star_entry:
                        # Format to match structure expected by callers
                        user_data = star_entry["user"]
                        processed_star = {
                            "starred_at": star_entry["starred_at"],
                            "user": {
                                "login": user_data.get("login"),
                                "avatar_url": user_data.get("avatar_url"),
                                "created_at": None,
                                "followers_count": None,
                                "public_repos": None,
                                "starred_count": None,
                            },
                        }
                    elif "login" in star_entry:
                        # Fallback for non-timestamped structure
                        processed_star = {
                            "starred_at": "2020-01-01T00:00:00Z",  # Placeholder
                            "user": {
                                "login": star_entry.get("login"),
                                "avatar_url": star_entry.get("avatar_url"),
                                "created_at": None,
                                "followers_count": None,
                                "public_repos": None,
                                "starred_count": None,
                            },
                        }
                    else:
                        logger.debug(f"Skipping malformed star entry: {star_entry}")
                        continue  # Skip malformed entries

                    processed_stars.append(processed_star)
                except Exception as e:
                    logger.debug(f"Error processing star entry: {e}")

            # Apply days filter if requested
            if days_limit > 0 and processed_stars:
                # Calculate cutoff date
                import datetime

                cutoff_date = make_naive_datetime(
                    datetime.datetime.now() - datetime.timedelta(days=days_limit)
                )

                # Filter stars by date
                filtered_stars = []
                for star in processed_stars:
                    starred_at_str = star.get("starred_at")
                    if starred_at_str:
                        try:
                            starred_at = make_naive_datetime(parse_date(starred_at_str))
                            if starred_at and starred_at >= cutoff_date:
                                filtered_stars.append(star)
                        except Exception as e:
                            logger.debug(
                                f"Could not parse date for star: {starred_at_str}, error: {e}"
                            )

                logger.info(
                    f"Fetched {len(processed_stars)} total stars, filtered to {len(filtered_stars)} within {days_limit} days."
                )
                return filtered_stars

            logger.debug(f"Processed {len(processed_stars)} stargazer entries")
            return processed_stars

        except Exception as e:
            logger.warning(f"Error fetching stargazers: {str(e)}")
            return []
