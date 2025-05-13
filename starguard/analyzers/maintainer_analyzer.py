"""Analyzes repository maintainers for reputation and activity."""

import datetime
import logging
from typing import Dict, List
from operator import itemgetter

from starguard.api.github_api import GitHubAPI
from starguard.utils.date_utils import make_naive_datetime

logger = logging.getLogger(__name__)


class MaintainerAnalyzer:
    """Analyzes repository maintainers for reputation and activity."""
    
    def __init__(self, contributors: List[Dict], commits: List[Dict], github_api: GitHubAPI): # Added github_api
        self.contributors = contributors # From get_contributors
        self.commits = commits # From get_commits (recent)
        self.github_api = github_api # For fetching more maintainer details if needed

    def analyze(self) -> Dict:
        """Analyze maintainer activity and reputation."""
        if not self.contributors:
            return {"maintainers": [], "recent_activity_summary": {}, "score": 5, "error": "No contributor data."} # Low score

        processed_maintainers = self._process_contributors() # Top N contributors
        
        # Recent commit activity from self.commits (already fetched for last 90 days)
        commits_last_90d = len(self.commits)
        
        # Active maintainers: defined as top contributors with recent commits (more robust)
        # For simplicity, use processed_maintainers' own contribution count as proxy for "active"
        active_maintainer_count = sum(1 for m in processed_maintainers if m["activity_level"] == "high")
        total_processed_maintainers = len(processed_maintainers)

        # Score 0-20. Higher is better.
        # Based on number of active maintainers and recent commit cadence.
        score_val = 0
        if total_processed_maintainers > 0:
            # Base score on having active maintainers
            if active_maintainer_count >= 3: score_val += 10
            elif active_maintainer_count >= 1: score_val += 5
            
            # Add points for commit frequency
            if commits_last_90d > 50: score_val += 10 # Very active
            elif commits_last_90d > 10: score_val += 7 # Moderately active
            elif commits_last_90d > 0: score_val += 3  # Some activity
        
        score_val = min(20, score_val) # Cap at 20

        return {
            "maintainers_analyzed": processed_maintainers,
            "recent_activity_summary": {
                "commits_last_90d": commits_last_90d,
                "active_maintainers_heuristic": active_maintainer_count,
                "total_top_contributors_analyzed": total_processed_maintainers
            },
            "score": score_val
        }

    def _process_contributors(self) -> List[Dict]:
        """Process contributor data to extract maintainer information."""
        if not self.contributors: return []
        
        # Sort by contributions, ensure 'contributions' key exists
        valid_contributors = [c for c in self.contributors if isinstance(c, dict) and "contributions" in c]
        sorted_contribs = sorted(valid_contributors, key=itemgetter("contributions"), reverse=True)
        
        top_n_maintainers = []
        for contrib_data in sorted_contribs[:5]: # Analyze top 5 contributors
            contributions_count = contrib_data.get("contributions", 0)
            activity_lvl = "low"
            if contributions_count > 100: activity_lvl = "high" # Arbitrary thresholds
            elif contributions_count > 20: activity_lvl = "medium"
            
            top_n_maintainers.append({
                "login": contrib_data.get("login", "N/A"),
                "contributions": contributions_count,
                "activity_level": activity_lvl, # Based on total contributions
                "profile_url": contrib_data.get("html_url", "")
                # Could add: fetch user profile for age, followers for deeper analysis (API heavy)
            })
        return top_n_maintainers

    def check_recent_activity(self) -> Dict: # Primarily informational
        """Check repository activity based on recent commits (from self.commits)."""
        if not self.commits: # self.commits are already for last 90 days
            return {
                "activity_counts_by_period": {"last_90_days": 0},
                "overall_activity_level": "inactive",
                "days_since_last_commit": 999 # Indicates very old or no commits
            }

        # All commits in self.commits are within the last 90 days.
        # Calculate days since the very last commit among these.
        last_commit_date_obj = None
        if self.commits:
            try:
                # Assuming commits are sorted by date by API, but re-sort to be sure
                # Commits are usually returned most recent first from API.
                # Find the most recent commit date.
                most_recent_commit_dt = None
                for c_data in self.commits:
                    if isinstance(c_data, dict) and "commit" in c_data and \
                       isinstance(c_data["commit"], dict) and "author" in c_data["commit"] and \
                       isinstance(c_data["commit"]["author"], dict) and "date" in c_data["commit"]["author"]:
                        
                        commit_dt = make_naive_datetime(parse_date(c_data["commit"]["author"]["date"]))
                        if commit_dt and (most_recent_commit_dt is None or commit_dt > most_recent_commit_dt):
                            most_recent_commit_dt = commit_dt
                
                if most_recent_commit_dt:
                    last_commit_date_obj = most_recent_commit_dt

            except Exception as e:
                logger.debug(f"Error parsing commit dates for activity check: {e}")

        days_lapsed = 999
        if last_commit_date_obj:
            days_lapsed = (make_naive_datetime(datetime.datetime.now()) - last_commit_date_obj).days
            days_lapsed = max(0, days_lapsed) # Ensure non-negative

        # Determine activity level based on commits in last 90 days and recency
        activity_lvl_str = "inactive"
        commits_90d_count = len(self.commits)
        if commits_90d_count > 20 and days_lapsed < 14: activity_lvl_str = "high"
        elif commits_90d_count > 5 and days_lapsed < 30: activity_lvl_str = "medium"
        elif commits_90d_count > 0 and days_lapsed < 90 : activity_lvl_str = "low"
        
        return {
            "activity_counts_by_period": {"last_90_days": commits_90d_count},
            "overall_activity_level": activity_lvl_str,
            "days_since_last_commit": days_lapsed
        }

