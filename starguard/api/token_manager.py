"""Token manager for GitHub API requests."""

import logging
from typing import List, Optional, Union

logger = logging.getLogger(__name__)


class TokenManager:
    """
    Manages multiple GitHub API tokens with rotation and rate limit tracking.

    This class keeps track of multiple tokens, their rate limits, and
    handles token rotation when limits are reached.

    Attributes:
        tokens: List of GitHub API tokens
        current_index: Index of the currently active token
        rate_limits: Dictionary tracking remaining API calls and reset times
    """

    def __init__(self, tokens: Optional[Union[str, List[str]]] = None):
        """
        Initialize the token manager.

        Args:
            tokens: Single token as string or list of multiple tokens
        """
        self.tokens = []
        self.current_index = 0
        self.rate_limits = {}

        # Handle token input
        if tokens:
            if isinstance(tokens, str):
                if tokens.strip():  # Non-empty string
                    self.tokens = [tokens.strip()]
            elif isinstance(tokens, list):
                self.tokens = [t for t in tokens if isinstance(t, str) and t.strip()]

        # Add environment token if available and not already included
        env_token = self._get_token_from_env()
        if env_token and env_token not in self.tokens:
            self.tokens.append(env_token)

        # Initialize rate limit tracking for each token
        for token in self.tokens:
            self.rate_limits[token] = {
                "remaining": 5000,  # Default GitHub API rate limit
                "reset_time": 0,
                "graphql_limited": False,
                "graphql_reset_time": 0,
            }

        logger.debug(f"Token manager initialized with {len(self.tokens)} tokens")

    def _get_token_from_env(self) -> Optional[str]:
        """Get GitHub token from environment variables."""
        import os

        token = os.environ.get("GITHUB_TOKEN")
        if token and token.strip():
            return token.strip()
        return None

    @property
    def token_count(self) -> int:
        """Get the number of available tokens."""
        return len(self.tokens)

    def get_current_token(self) -> Optional[str]:
        """Get the currently active token."""
        if not self.tokens:
            return None
        return self.tokens[self.current_index]

    def rotate_token(self) -> bool:
        """
        Rotate to the next available token.

        Returns:
            bool: True if successfully rotated to a new token, False if no more tokens
        """
        if len(self.tokens) <= 1:
            return False

        # Save the original index to detect full rotation
        original_index = self.current_index

        # Try each token in sequence
        for _ in range(len(self.tokens)):
            self.current_index = (self.current_index + 1) % len(self.tokens)

            # Check if this token has available rate limit
            token = self.tokens[self.current_index]
            rate_info = self.rate_limits.get(token, {})

            if rate_info.get("remaining", 0) > 10:
                logger.debug(f"Rotated to token {self.current_index+1}/{len(self.tokens)}")
                return True

            # If we've tried all tokens and come back to the original
            if self.current_index == original_index:
                break

        # If we get here, all tokens are at/near rate limit
        logger.warning("All tokens are at or near rate limit")
        return False

    def update_rate_limit(self, token: str, remaining: int, reset_time: int) -> None:
        """
        Update rate limit information for a specific token.

        Args:
            token: The token to update
            remaining: Number of API calls remaining
            reset_time: UNIX timestamp when the rate limit resets
        """
        if token in self.rate_limits:
            self.rate_limits[token]["remaining"] = remaining
            self.rate_limits[token]["reset_time"] = reset_time

            if remaining <= 10:
                logger.debug(
                    f"Token {self.tokens.index(token)+1} is low on rate limit: {remaining} remaining"
                )
