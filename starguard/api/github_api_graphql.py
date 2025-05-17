"""GitHub GraphQL API specific methods."""

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


class GitHubGraphQLMethods:
    """
    Implementation of GitHub GraphQL API methods.

    This class provides methods that leverage GraphQL for more efficient data retrieval
    than what's possible with the REST API alone.
    """

    def get_user_graphql(self, username: str) -> Dict:
        """
        Get detailed user information using GraphQL.

        This method retrieves more comprehensive user data than the REST API
        in a single request, including contribution statistics.

        Args:
            username: GitHub username

        Returns:
            Dict: Enhanced user profile data
        """
        query = """
        query($username: String!) {
          user(login: $username) {
            login
            name
            createdAt
            updatedAt
            followers {
              totalCount
            }
            following {
              totalCount
            }
            repositories(first: 0) {
              totalCount
            }
            contributionsCollection {
              totalCommitContributions
              totalIssueContributions
              totalPullRequestContributions
              totalPullRequestReviewContributions
              totalRepositoriesWithContributedCommits
              contributionCalendar {
                totalContributions
              }
            }
            starredRepositories {
              totalCount
            }
          }
        }
        """

        variables = {"username": username}
        result = self.graphql_request(query, variables)

        if "errors" in result or "data" not in result or not result["data"].get("user"):
            logger.debug(f"Error or no data from GraphQL for user {username}")
            return {}

        # Convert the GraphQL response to a flattened dict similar to REST API format
        user_data = result["data"]["user"]
        processed_data = {
            "login": user_data.get("login"),
            "name": user_data.get("name"),
            "created_at": user_data.get("createdAt"),
            "updated_at": user_data.get("updatedAt"),
            "followers": user_data.get("followers", {}).get("totalCount", 0),
            "following": user_data.get("following", {}).get("totalCount", 0),
            "public_repos": user_data.get("repositories", {}).get("totalCount", 0),
            "starred_count": user_data.get("starredRepositories", {}).get("totalCount", 0),
            # Add contribution data
            "contributions": {
                "total": user_data.get("contributionsCollection", {})
                .get("contributionCalendar", {})
                .get("totalContributions", 0),
                "commits": user_data.get("contributionsCollection", {}).get(
                    "totalCommitContributions", 0
                ),
                "issues": user_data.get("contributionsCollection", {}).get(
                    "totalIssueContributions", 0
                ),
                "pull_requests": user_data.get("contributionsCollection", {}).get(
                    "totalPullRequestContributions", 0
                ),
                "reviews": user_data.get("contributionsCollection", {}).get(
                    "totalPullRequestReviewContributions", 0
                ),
                "repos_contributed_to": user_data.get("contributionsCollection", {}).get(
                    "totalRepositoriesWithContributedCommits", 0
                ),
            },
        }

        # Store in cache
        self.user_profile_cache[username] = processed_data

        return processed_data

    def get_stargazers_graphql(self, owner: str, repo: str, limit: int = 100) -> List[Dict]:
        """
        Get repository stargazers using GraphQL.

        More efficient than REST API for retrieving star timestamps and basic user info.

        Args:
            owner: Repository owner
            repo: Repository name
            limit: Maximum number of stargazers to retrieve

        Returns:
            List[Dict]: List of stargazer data with timestamps
        """
        query = """
        query($owner: String!, $repo: String!, $cursor: String) {
          repository(owner: $owner, name: $repo) {
            stargazers(first: 100, after: $cursor) {
              totalCount
              pageInfo {
                hasNextPage
                endCursor
              }
              edges {
                starredAt
                node {
                  login
                  avatarUrl
                  createdAt
                }
              }
            }
          }
        }
        """

        variables = {"owner": owner, "repo": repo, "cursor": None}
        stars = []
        page_count = 0

        # GraphQL pagination
        while True:
            result = self.graphql_request(query, variables)

            if "errors" in result or "data" not in result:
                logger.warning(f"Error fetching stargazers via GraphQL for {owner}/{repo}")
                break

            repo_data = result["data"].get("repository", {})
            stargazers_data = repo_data.get("stargazers", {})

            if not stargazers_data:
                break

            # Process edges
            for edge in stargazers_data.get("edges", []):
                user_node = edge.get("node", {})
                stars.append(
                    {
                        "starred_at": edge.get("starredAt"),
                        "user": {
                            "login": user_node.get("login"),
                            "avatar_url": user_node.get("avatarUrl"),
                            "created_at": user_node.get("createdAt"),
                        },
                    }
                )

            # Check pagination
            page_info = stargazers_data.get("pageInfo", {})
            if not page_info.get("hasNextPage") or len(stars) >= limit:
                break

            # Update cursor for next page
            variables["cursor"] = page_info.get("endCursor")
            page_count += 1

            # Safety limit for pagination
            if page_count >= 10 or len(stars) >= limit:
                break

        logger.debug(f"Retrieved {len(stars)} stargazers via GraphQL for {owner}/{repo}")
        return stars[:limit]

    def get_yearly_contributions(self, username: str, year: int) -> Dict:
        """
        Get a user's contributions for a specific year.

        Args:
            username: GitHub username
            year: Year to get contributions for

        Returns:
            Dict: Yearly contribution data
        """
        query = """
        query($username: String!, $from: DateTime!, $to: DateTime!) {
          user(login: $username) {
            contributionsCollection(from: $from, to: $to) {
              contributionCalendar {
                totalContributions
              }
              totalCommitContributions
              totalIssueContributions
              totalPullRequestContributions
              totalPullRequestReviewContributions
              restrictedContributionsCount
            }
          }
        }
        """

        from_date = f"{year}-01-01T00:00:00Z"
        to_date = f"{year}-12-31T23:59:59Z"

        variables = {"username": username, "from": from_date, "to": to_date}

        result = self.graphql_request(query, variables)

        if "errors" in result or "data" not in result or not result["data"].get("user"):
            logger.debug(
                f"Error or no data from GraphQL for user {username} contributions in {year}"
            )
            return {
                "year": year,
                "total": 0,
                "commits": 0,
                "issues": 0,
                "pull_requests": 0,
                "reviews": 0,
                "private": 0,
            }

        contributions = result["data"]["user"]["contributionsCollection"]

        return {
            "year": year,
            "total": contributions.get("contributionCalendar", {}).get("totalContributions", 0),
            "commits": contributions.get("totalCommitContributions", 0),
            "issues": contributions.get("totalIssueContributions", 0),
            "pull_requests": contributions.get("totalPullRequestContributions", 0),
            "reviews": contributions.get("totalPullRequestReviewContributions", 0),
            "private": contributions.get("restrictedContributionsCount", 0),
        }
