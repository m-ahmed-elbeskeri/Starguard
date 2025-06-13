"""API response caching for GitHub API calls."""

import os
import json
import time
import hashlib
from typing import Dict, Optional


class ApiCache:
    """
    Caches API responses to reduce API calls and improve performance.

    This class provides file-based caching for both REST and GraphQL API responses,
    with configurable expiration times.

    Attributes:
        cache_dir: Directory where cache files are stored
        default_ttl: Default time-to-live for cache entries in seconds
    """

    def __init__(self, cache_dir: str = "cache", default_ttl: int = 86400):
        """
        Initialize the API cache.

        Args:
            cache_dir: Directory to store cache files
            default_ttl: Default cache expiration time in seconds (24 hours)
        """
        self.cache_dir = os.path.join(cache_dir, "github_api")
        self.default_ttl = default_ttl

        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_cache_key(self, endpoint: str, params: Optional[Dict] = None) -> str:
        """
        Generate a unique cache key for an API request.

        Args:
            endpoint: API endpoint path
            params: URL query parameters

        Returns:
            str: Unique cache key
        """
        # Create a deterministic string representation of params
        params_str = ""
        if params:
            # Sort to ensure consistent ordering
            keys = sorted(params.keys())
            params_str = "_".join(f"{k}={params[k]}" for k in keys)

        # Create a cache key with endpoint and params
        cache_key = f"{endpoint.replace('/', '_')}"
        if params_str:
            cache_key += f"_{params_str}"

        return cache_key

    def _get_cache_path(self, cache_key: str) -> str:
        """
        Get the file path for a cache item.

        Args:
            cache_key: Unique cache key

        Returns:
            str: Path to cache file
        """
        # Hash long keys to avoid filename length issues
        if len(cache_key) > 100:
            hashed_key = hashlib.md5(cache_key.encode()).hexdigest()
            filename = f"{hashed_key}.json"
        else:
            # Make filename safe
            safe_key = "".join(c if c.isalnum() or c in "_-." else "_" for c in cache_key)
            filename = f"{safe_key}.json"

        return os.path.join(self.cache_dir, filename)

    def get(
        self, endpoint: str, params: Optional[Dict] = None, ttl: Optional[int] = None
    ) -> Optional[Dict]:
        """
        Retrieve response from cache if available and not expired.

        Args:
            endpoint: API endpoint path
            params: URL query parameters
            ttl: Time-to-live in seconds, defaults to self.default_ttl

        Returns:
            Optional[Dict]: Cached response or None if not found/expired
        """
        cache_key = self._get_cache_key(endpoint, params)
        return self.get_raw(cache_key, ttl)

    def get_raw(self, cache_key: str, ttl: Optional[int] = None) -> Optional[Dict]:
        """
        Retrieve response from cache by raw key.

        Args:
            cache_key: Raw cache key
            ttl: Time-to-live in seconds, defaults to self.default_ttl

        Returns:
            Optional[Dict]: Cached response or None if not found/expired
        """
        cache_path = self._get_cache_path(cache_key)

        if not os.path.exists(cache_path):
            return None

        # Check if cache is expired
        max_age = ttl if ttl is not None else self.default_ttl
        cache_age = time.time() - os.path.getmtime(cache_path)

        if cache_age > max_age:
            return None

        try:
            with open(cache_path, "r") as f:
                return json.load(f)
        except Exception as e:
            import logging

            logging.getLogger(__name__).debug(f"Error reading cache: {e}")
            return None

    def put(self, endpoint: str, params: Optional[Dict], data: Dict) -> None:
        """
        Save API response to cache.

        Args:
            endpoint: API endpoint path
            params: URL query parameters
            data: Response data to cache
        """
        cache_key = self._get_cache_key(endpoint, params)
        self.put_raw(cache_key, data)

    def put_raw(self, cache_key: str, data: Dict) -> None:
        """
        Save API response to cache with raw key.

        Args:
            cache_key: Raw cache key
            data: Response data to cache
        """
        cache_path = self._get_cache_path(cache_key)

        try:
            with open(cache_path, "w") as f:
                json.dump(data, f)
        except Exception as e:
            import logging

            logging.getLogger(__name__).debug(f"Error writing to cache: {e}")
