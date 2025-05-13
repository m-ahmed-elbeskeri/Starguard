"""Date and time utilities for StarGuard."""

import datetime
from typing import Optional

def make_naive_datetime(dt: Optional[datetime.datetime]) -> Optional[datetime.datetime]:
    """Convert a datetime to naive (remove timezone info)."""
    if dt is None:
        return None
    if dt.tzinfo is not None:
        return dt.replace(tzinfo=None)
    return dt