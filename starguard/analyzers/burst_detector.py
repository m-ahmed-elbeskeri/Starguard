"""Detects star burst patterns using robust statistical methods."""

import datetime
import logging
import random
import math
from typing import Dict, List, Optional, Set, Tuple
from collections import Counter, defaultdict

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
from dateutil.parser import parse as parse_date
from tqdm import tqdm

from starguard.api.github_api import GitHubAPI
from starguard.utils.date_utils import make_naive_datetime
from starguard.core.constants import (
    MAD_THRESHOLD,
    WINDOW_SIZE,
    MIN_STAR_COUNT,
    MIN_STARS_GROWTH_PERCENT,
    USER_SCORE_THRESHOLDS,
    FAKE_USER_THRESHOLD,
    FAKE_RATIO_WEIGHT,
    RULE_HITS_WEIGHT,
    BURST_ORGANIC_THRESHOLD,
    BURST_FAKE_THRESHOLD,
    LOCKSTEP_DETECTION,
    AVATAR_HASH_DISTANCE,
    AVATAR_MATCH_SCORE
)

logger = logging.getLogger(__name__)


# Helper functions
def _gini(values: List[int | float]) -> float:
    """Calculate the Gini coefficient (0=perfectly even, 1=all mass on one item)."""
    if len(values) == 0:
        return 0.0
    arr = np.array(values, dtype=float)
    if np.all(arr == 0):
        return 0.0
    arr += 1e-12  # avoid zero division
    arr.sort()
    index = np.arange(1, arr.size + 1)
    return (np.sum((2 * index - arr.size - 1) * arr) /
            (arr.size * np.sum(arr)))


def _entropy(values: List[int]) -> float:
    """Calculate Shannon entropy of a list of values."""
    if not values:
        return 0.0
    counts = Counter(values)
    total = len(values)
    probs = [count/total for count in counts.values()]
    return -sum(p * math.log2(p) for p in probs)


def _logistic_score(x: float,
                   mid: float,
                   steepness: float = 0.3,
                   max_score: float = 2.0) -> float:
    """Smooth monotone mapping → (0, max_score]."""
    return max_score / (1 + math.exp(steepness * (x - mid)))


class LockStepDetector:
    """Detect clusters of stargazers that star many of the same repos."""

    def __init__(
        self,
        user_to_repos: Dict[str, Set[str]],
        eps: float = LOCKSTEP_DETECTION["eps"],
        min_size: int = LOCKSTEP_DETECTION["min_samples"]
    ) -> None:
        self.user_to_repos = user_to_repos
        self.eps = eps
        self.min_size = min_size
        self._cluster_scores: Dict[str, float] | None = None

    def cluster_scores(self) -> Dict[str, float]:
        """Return {user: density} where density ∈ [0, 1].
        
        0 means the user is singleton/noisy;
        >0.05 means likely part of a coordinated cluster.
        """
        if self._cluster_scores is None:
            self._cluster_scores = self._build_scores()
        return self._cluster_scores

    def _build_scores(self) -> Dict[str, float]:
        users = list(self.user_to_repos)
        idx = {u: i for i, u in enumerate(users)}
        m = len(users)

        # Build sparse Jaccard distance matrix
        # d(i,j) = 1 - |A∩B| / |A∪B|
        rows, cols, dists = [], [], []
        for i, u in enumerate(users):
            repos_u = self.user_to_repos[u]
            for v in users[i + 1:]:
                inter = repos_u & self.user_to_repos[v]
                if not inter:
                    continue
                union = repos_u | self.user_to_repos[v]
                dist = 1.0 - len(inter) / len(union)
                rows.append(i)
                cols.append(idx[v])
                dists.append(dist)

        if not rows:
            # All users share nothing ⇒ no clusters
            return {u: 0.0 for u in users}

        # Build symmetric condensed distance matrix for DBSCAN
        from scipy.sparse import coo_matrix
        dm = coo_matrix((dists + dists,  # mirror upper triangle
                         (rows + cols, cols + rows)),
                        shape=(m, m)).toarray()
        np.fill_diagonal(dm, 0.0)

        clustering = DBSCAN(
            eps=self.eps,
            min_samples=self.min_size,
            metric="precomputed"
        ).fit(dm)

        labels = clustering.labels_  # −1 = noise
        label_counts = Counter(labels[labels >= 0])

        density = {}
        total_users = len(users)
        for u, lbl in zip(users, labels, strict=False):
            if lbl == -1:
                density[u] = 0.0
            else:
                density[u] = (label_counts[lbl] - 1) / total_users
        return density


class BurstDetector:
    """
    Detects star burst patterns using robust statistical methods.
    Uses Median Absolute Deviation (MAD) which is more robust to outliers than Z-score.
    """

    def __init__(self, owner: str, repo: str, github_api: GitHubAPI, window_size: int = WINDOW_SIZE):
        self.owner = owner
        self.repo = repo
        self.github_api = github_api
        self.window_size = window_size
        self.stars_df = None
        self.bursts = []
        self.stars_data = []  # Store raw star data
        self.avatar_hashes = {}  # Cache for avatar perceptual hashes

    def build_daily_timeseries(self, days_limit: int = 0) -> pd.DataFrame:
        """Build a daily time series of stars from the GitHub API."""
        try:
            stars_data = self.github_api.get_stargazers(self.owner, self.repo, get_timestamps=True, days_limit=days_limit)
            self.stars_data = stars_data  # Store for checking estimated dates later
            
            if not stars_data:
                logger.warning(f"No star data found for {self.owner}/{self.repo} within {days_limit} days.")
                return pd.DataFrame(columns=["date", "stars", "users", "day"]) # Add "users" and "day"

            # Check if we're using estimated dates
            has_estimated_dates = any(star.get("date_is_estimated", False) for star in stars_data)
            if has_estimated_dates:
                logger.warning(f"Using estimated star timestamps for {self.owner}/{self.repo} - this may affect burst detection")

            records = []
            for star in stars_data:
                try:
                    if "starred_at" not in star or not star["starred_at"]:
                        continue
                    username = star.get("user", {}).get("login")
                    
                    # Ensure starred_at is parsed correctly and made naive for consistency
                    starred_at_dt = make_naive_datetime(parse_date(star["starred_at"]))
                    if starred_at_dt: # Check if parse_date was successful
                         records.append({
                            "date": starred_at_dt.date(), # Group by date object
                            "datetime": starred_at_dt,
                            "username": username,
                            "date_is_estimated": star.get("date_is_estimated", False)
                        })
                except Exception as e:
                    logger.debug(f"Error processing star data entry: {star}. Error: {str(e)}")
            
            if not records:
                logger.warning(f"No valid star records after parsing for {self.owner}/{self.repo}.")
                return pd.DataFrame(columns=["date", "stars", "users", "day"])

            df = pd.DataFrame(records)
            df["day"] = pd.to_datetime(df["date"]) # This converts date objects to Timestamps (datetime64[ns])

            # Group by original date objects for daily_counts and daily_users
            daily_counts = df.groupby("date").size().reset_index(name="stars")
            daily_users = df.groupby("date")["username"].apply(list).reset_index(name="users")
            
            # Track estimated dates
            estimated_count = df.groupby("date")["date_is_estimated"].sum().reset_index(name="estimated_count")
            
            # Merge the grouped data
            result_df = pd.merge(daily_counts, daily_users, on="date", how="outer")
            result_df = pd.merge(result_df, estimated_count, on="date", how="outer")
            
            # Calculate what percent of stars that day had estimated dates
            result_df["estimated_ratio"] = result_df["estimated_count"] / result_df["stars"]
            result_df["has_estimated_dates"] = result_df["estimated_ratio"] > 0.5  # Flag if >50% are estimated
            
            # Ensure 'date' column in result_df is also datetime64[ns] for merge with all_dates
            result_df["date"] = pd.to_datetime(result_df["date"]) 

            # Fill date gaps
            if not df.empty:
                min_date_ts = df["day"].min() # Use 'day' which is Timestamp
                max_date_ts = df["day"].max() # Use 'day' which is Timestamp
                
                # Create a full date range of Timestamps
                all_dates_df = pd.DataFrame({
                    "date": pd.date_range(start=min_date_ts, end=max_date_ts, freq="D")
                })
                # Merge with all_dates_df. 'date' column is now Timestamp in both.
                merged_df = pd.merge(all_dates_df, result_df, on="date", how="left")
            else: # df was empty, so result_df is also empty or near empty
                merged_df = result_df # or an empty df with correct columns

            merged_df["stars"] = merged_df["stars"].fillna(0).astype(int) # Fixed FutureWarning & ensure int
            merged_df["users"] = merged_df["users"].apply(lambda x: x if isinstance(x, list) else [])
            merged_df["estimated_count"] = merged_df["estimated_count"].fillna(0).astype(int)
            merged_df["estimated_ratio"] = merged_df["estimated_ratio"].fillna(0.0)
            merged_df["has_estimated_dates"] = merged_df["has_estimated_dates"].fillna(False)
            
            merged_df.sort_values("date", inplace=True)
            
            # Calculate time-of-day entropy
            if not df.empty:
                hours = [dt.hour for dt in df["datetime"]]
                merged_df["tod_entropy"] = _entropy(hours)
            else:
                merged_df["tod_entropy"] = 0.0
                
            self.stars_df = merged_df
            return merged_df

        except Exception as e:
            logger.error(f"Error building timeseries for {self.owner}/{self.repo}: {str(e)}", exc_info=True)
            return pd.DataFrame(columns=["date", "stars", "users", "day"])

    def detect_bursts(self) -> List[Dict]:
        """Detect star bursts using MAD."""
        try:
            if self.stars_df is None or self.stars_df.empty:
                # Try building timeseries with a default limit if not already built
                self.stars_df = self.build_daily_timeseries() 
            if self.stars_df.empty:
                logger.info(f"No star timeseries data to detect bursts for {self.owner}/{self.repo}.")
                return []

            df_analysis = self.stars_df.copy()
            if 'stars' not in df_analysis.columns or df_analysis['stars'].isnull().all():
                logger.warning(f"Stars column missing or all null in timeseries for {self.owner}/{self.repo}")
                return []
            
            # Check if we're using estimated dates
            has_estimated_dates = False
            estimated_dates_count = df_analysis["estimated_count"].sum() if "estimated_count" in df_analysis.columns else 0
            total_stars = df_analysis["stars"].sum()
            
            if estimated_dates_count > 0 and total_stars > 0:
                estimated_ratio = estimated_dates_count / total_stars
                has_estimated_dates = estimated_ratio > 0.5  # If more than 50% dates are estimated
                
                if has_estimated_dates:
                    logger.warning(f"Using estimated dates for {self.owner}/{self.repo} star burst detection ({estimated_ratio:.1%} estimated)")
            
            # Adjust MAD threshold if we're using estimated dates
            mad_threshold_to_use = MAD_THRESHOLD
            if has_estimated_dates:
                # Use a more conservative threshold to avoid false positives with estimated dates
                mad_threshold_to_use = MAD_THRESHOLD * 1.5  # 50% higher threshold for estimated dates
                logger.info(f"Using adjusted MAD threshold of {mad_threshold_to_use} for estimated dates")

            df_analysis["median"] = np.nan
            df_analysis["mad"] = np.nan
            df_analysis["is_anomaly"] = False
            
            # Date column in df_analysis is Timestamp objects. Access .date() for date objects if needed.
            # Loop from self.window_size implies index-based access.
            for i in range(self.window_size, len(df_analysis)):
                window_data = df_analysis.iloc[i - self.window_size:i]["stars"]
                if not window_data.empty:
                    median_val = window_data.median()
                    df_analysis.loc[df_analysis.index[i], "median"] = median_val
                    df_analysis.loc[df_analysis.index[i], "mad"] = (window_data - median_val).abs().median()
            
            # Fill NaNs that might remain if window was too short at start
            df_analysis["median"] = df_analysis["median"].bfill()  
            df_analysis["mad"] = df_analysis["mad"].bfill()
            df_analysis["mad"] = df_analysis["mad"].fillna(0) 

            for i in range(len(df_analysis)): # Check all days, not just starting from window_size
                # Use .loc with index for setting values
                idx = df_analysis.index[i]
                median = df_analysis.loc[idx, "median"]
                mad = df_analysis.loc[idx, "mad"]
                stars_today = df_analysis.loc[idx, "stars"]
                
                # total_stars up to *before* current day
                total_stars_before_today = df_analysis.iloc[:i]["stars"].sum() if i > 0 else 0

                is_anomaly_flag = False
                if pd.isna(median) or pd.isna(mad): # Handle cases where median/mad couldn't be computed
                    # For early days before full window, or if data was sparse
                    if stars_today > 0 and total_stars_before_today < MIN_STAR_COUNT: # Small repo, early phase
                         percent_increase = (stars_today / max(1, total_stars_before_today)) * 100
                         if percent_increase > MIN_STARS_GROWTH_PERCENT and stars_today > 5: # Min 5 stars for a spike
                             is_anomaly_flag = True
                elif total_stars_before_today >= MIN_STAR_COUNT and mad > 0.001: # MAD > 0 (avoid MAD=0 issues)
                    threshold = median + mad_threshold_to_use * mad  # Use the potentially adjusted threshold
                    if stars_today > threshold and stars_today > median + 1: # Ensure it's meaningfully above median
                        is_anomaly_flag = True
                elif stars_today > 0: # Small repo or MAD is zero (low variance period)
                    # Use percentage growth relative to historical or absolute jump
                    percent_increase = (stars_today / max(1, total_stars_before_today)) * 100
                    # Check if stars_today is significantly more than recent median
                    significant_jump = stars_today > max(5, median * 2) # e.g. >5 stars and double the median
                    if (percent_increase > MIN_STARS_GROWTH_PERCENT and stars_today > 5) or significant_jump :
                        is_anomaly_flag = True
                
                df_analysis.loc[idx, "is_anomaly"] = is_anomaly_flag

            bursts = []
            in_burst = False
            current_burst_start_date = None
            current_burst_users = []
            current_burst_star_count = 0

            for _, row in df_analysis.iterrows():
                # row["date"] is a Timestamp object from self.stars_df
                current_date_obj = row["date"].to_pydatetime().date() # Convert to datetime.date for consistency in burst dict

                if row["is_anomaly"] and not in_burst:
                    in_burst = True
                    current_burst_start_date = current_date_obj
                    current_burst_users = list(row["users"]) if isinstance(row["users"], list) else []
                    current_burst_star_count = row["stars"]
                elif row["is_anomaly"] and in_burst:
                    if isinstance(row["users"], list):
                        current_burst_users.extend(row["users"])
                    current_burst_star_count += row["stars"]
                elif not row["is_anomaly"] and in_burst:
                    in_burst = False
                    # Use previous day's date as burst_end_date
                    # Find the date of previous row in df_analysis
                    prev_row_date_obj = df_analysis.loc[df_analysis[df_analysis['date'] < row['date']].index[-1], 'date'].to_pydatetime().date()

                    bursts.append({
                        "start_date": current_burst_start_date,
                        "end_date": prev_row_date_obj, # End date is the last anomalous day
                        "days": (prev_row_date_obj - current_burst_start_date).days + 1,
                        "stars": int(current_burst_star_count),
                        "users": list(set(current_burst_users)),
                        "has_estimated_dates": has_estimated_dates  # Flag if using estimated dates
                    })
                    # Reset for next potential burst
                    current_burst_users = []
                    current_burst_star_count = 0
                    current_burst_start_date = None

            if in_burst: # If loop ends while in a burst
                last_date_obj = df_analysis.iloc[-1]["date"].to_pydatetime().date()
                bursts.append({
                    "start_date": current_burst_start_date,
                    "end_date": last_date_obj,
                    "days": (last_date_obj - current_burst_start_date).days + 1,
                    "stars": int(current_burst_star_count),
                    "users": list(set(current_burst_users)),
                    "has_estimated_dates": has_estimated_dates  # Flag if using estimated dates
                })

            self.bursts = bursts
            return bursts

        except Exception as e:
            logger.error(f"Error detecting bursts for {self.owner}/{self.repo}: {str(e)}", exc_info=True)
            if self.stars_df is not None:
                logger.debug(f"Stars dataframe for {self.owner}/{self.repo} has {len(self.stars_df)} rows. Columns: {self.stars_df.columns}")
            return []

    def cross_check_bursts(self) -> List[Dict]:
        """Perform consistency cross-checks on detected bursts."""
        try:
            if not self.bursts: # If empty, try to detect them first
                self.detect_bursts()
            if not self.bursts: # If still empty, return empty
                return []

            enhanced_bursts = []
            for burst in self.bursts:
                # Ensure start_date and end_date are datetime.date objects
                burst_start_date_obj = burst["start_date"]
                burst_end_date_obj = burst["end_date"]

                # Check if this burst has estimated dates
                has_estimated_dates = burst.get("has_estimated_dates", False)

                # Cross-check 1: Fork delta
                fork_delta, fork_ratio, flag_forks = 0, 0, True
                try:
                    forks = self.github_api.get_forks(self.owner, self.repo)
                    forks_in_burst_period = [
                        f for f in forks 
                        if f.get("created_at") and 
                           burst_start_date_obj <= make_naive_datetime(parse_date(f["created_at"])).date() <= burst_end_date_obj
                    ]
                    fork_delta = len(forks_in_burst_period)
                    fork_ratio = fork_delta / burst["stars"] if burst["stars"] > 0 else 0
                    flag_forks = fork_ratio < 0.01 # Suspicious if very few forks per star
                except Exception as e:
                    logger.debug(f"Error checking forks for burst {burst_start_date_obj}-{burst_end_date_obj}: {str(e)}")

                # Cross-check 2: Issue + PR delta
                issue_delta, pr_delta, issue_pr_delta, flag_issues_prs = 0,0,0,True
                try:
                    issues = self.github_api.get_issues(self.owner, self.repo, state="all") # Get all states
                    issues_in_burst = [
                        i for i in issues 
                        if i.get("created_at") and
                           burst_start_date_obj <= make_naive_datetime(parse_date(i["created_at"])).date() <= burst_end_date_obj
                    ]
                    issue_delta = len(issues_in_burst)

                    prs = self.github_api.get_pulls(self.owner, self.repo, state="all")
                    prs_in_burst = [
                        p for p in prs
                        if p.get("created_at") and
                           burst_start_date_obj <= make_naive_datetime(parse_date(p["created_at"])).date() <= burst_end_date_obj
                    ]
                    pr_delta = len(prs_in_burst)
                    issue_pr_delta = issue_delta + pr_delta
                    flag_issues_prs = issue_pr_delta == 0 and burst["stars"] > 10 # No interaction during a sizable burst
                except Exception as e:
                    logger.debug(f"Error checking issues/PRs for burst {burst_start_date_obj}-{burst_end_date_obj}: {str(e)}")
                
                # Cross-check 3: Traffic views (if available)
                views_delta, traffic_ratio, flag_traffic = 0,0,True # Default to suspicious if no data
                try:
                    # Traffic data is for past 14 days. May not align with older bursts.
                    # For simplicity, we just check if *any* traffic data is available from the API.
                    traffic_views_data = self.github_api.get_traffic_views(self.owner, self.repo)
                    if traffic_views_data and traffic_views_data.get("views"):
                        for view_entry in traffic_views_data["views"]:
                            view_date = make_naive_datetime(parse_date(view_entry["timestamp"])).date()
                            if burst_start_date_obj <= view_date <= burst_end_date_obj:
                                views_delta += view_entry["count"]
                        
                        if views_delta > 0 : # Only if we have view data for the period
                            traffic_ratio = views_delta / burst["stars"] if burst["stars"] > 0 else 0
                            # Flag if views are less than stars (e.g. 1 view per star is low for organic)
                            flag_traffic = traffic_ratio < 1.0 and burst["stars"] > 10 
                        # If no views in burst period but traffic API worked, it's suspicious.
                        # If traffic API failed, it remains flag_traffic = True (suspicious by default)
                    else: # No traffic data from API (e.g. permissions)
                        logger.debug(f"No traffic view data available for {self.owner}/{self.repo}")

                except Exception as e:
                    logger.debug(f"Error checking traffic for burst {burst_start_date_obj}-{burst_end_date_obj}: {str(e)}")

                # Cross-check 4: Commits / releases around the burst
                has_commits, has_release, flag_activity = False, False, True
                try:
                    # Check commits/releases in a window around the burst
                    window_start = burst_start_date_obj - datetime.timedelta(days=7)
                    window_end = burst_end_date_obj + datetime.timedelta(days=7)
                    
                    commits = self.github_api.get_commits(self.owner, self.repo, 
                                                          since=window_start.isoformat(), 
                                                          until=window_end.isoformat())
                    has_commits = len(commits) > 0

                    releases = self.github_api.get_releases(self.owner, self.repo)
                    has_release = any(
                        r.get("published_at") and
                        window_start <= make_naive_datetime(parse_date(r["published_at"])).date() <= window_end 
                        for r in releases
                    )
                    flag_activity = not (has_commits or has_release) and burst["stars"] > 20 # No activity around a significant burst
                except Exception as e:
                    logger.debug(f"Error checking activity for burst {burst_start_date_obj}-{burst_end_date_obj}: {str(e)}")

                # Check time-of-day entropy for this burst
                burst_tod_entropy = 0.0
                try:
                    # Get all star timestamps in this burst
                    stars_data = self.github_api.get_stargazers(self.owner, self.repo, get_timestamps=True)
                    burst_star_dates = [
                        make_naive_datetime(parse_date(s["starred_at"]))
                        for s in stars_data
                        if s.get("starred_at") and
                           burst_start_date_obj <= make_naive_datetime(parse_date(s["starred_at"])).date() <= burst_end_date_obj
                    ]
                    if burst_star_dates:
                        hours = [d.hour for d in burst_star_dates]
                        burst_tod_entropy = _entropy(hours)
                except Exception as e:
                    logger.debug(f"Error calculating time-of-day entropy for burst: {str(e)}")

                # Calculate rule hits, but reduce weight if dates are estimated
                rule_hits = sum([
                    1 if flag_forks else 0,
                    1 if flag_issues_prs else 0,
                    1 if flag_traffic else 0, # Traffic data might not be available
                    1 if flag_activity else 0,
                    1 if burst_tod_entropy < 1.5 else 0  # Check for tod entropy
                ])

                # If we're using estimated dates, reduce the rule hits to prevent false positives
                if has_estimated_dates:
                    # Apply a penalty reduction for estimated dates
                    rule_hits = max(0, rule_hits - 1)  # Reduce by 1 to be more conservative
                
                enhanced_burst = burst.copy()
                enhanced_burst.update({
                    "cross_checks": {
                        "fork_delta": fork_delta, "fork_ratio": fork_ratio, "flag_forks": flag_forks,
                        "issue_delta": issue_delta, "pr_delta": pr_delta, "issue_pr_delta": issue_pr_delta, "flag_issues_prs": flag_issues_prs,
                        "views_delta": views_delta, "traffic_ratio": traffic_ratio, "flag_traffic": flag_traffic,
                        "has_commits_around_burst": has_commits, "has_release_around_burst": has_release, "flag_activity": flag_activity,
                        "tod_entropy": burst_tod_entropy, "flag_tod_entropy": burst_tod_entropy < 1.5
                    },
                    "rule_hits": rule_hits,
                    "tod_entropy": burst_tod_entropy,
                    "has_estimated_dates": has_estimated_dates,
                    "inorganic_heuristic": rule_hits >= 2 # Adjusted heuristic based on available data
                })
                enhanced_bursts.append(enhanced_burst)
            
            self.bursts = enhanced_bursts
            return enhanced_bursts

        except Exception as e:
            logger.error(f"Error cross-checking bursts for {self.owner}/{self.repo}: {str(e)}", exc_info=True)
            return self.bursts # Return original if error

    def score_stargazers(self, max_users_to_score_per_burst: int = 10000, max_total_users_to_score: int = 10000) -> Dict:
        """Score stargazers in burst windows to identify likely fake accounts."""
        try:
            if not self.bursts: self.cross_check_bursts()
            if not self.bursts: return {"bursts": [], "user_scores": {}}

            all_user_scores = {}
            scored_bursts_output = []
            total_users_scored_so_far = 0

            # Prioritize scoring users from more suspicious bursts
            # Sort bursts by a suspicion score (e.g., rule_hits desc, then stars desc)
            sorted_bursts_for_scoring = sorted(
                self.bursts, 
                key=lambda b: (b.get("inorganic_heuristic", False), b.get("rule_hits", 0), b.get("stars", 0)),
                reverse=True
            )

            # Get all stargazers once to build a lock-step detector
            all_stargazers = set()
            for burst in self.bursts:
                all_stargazers.update(burst.get("users", []))
                
            # Limit total users for performance
            if len(all_stargazers) > LOCKSTEP_DETECTION["max_users"]:
                all_stargazers = set(random.sample(list(all_stargazers), LOCKSTEP_DETECTION["max_users"]))
                
            # Build user_to_repos mapping for lock-step detection
            user_to_repos = {}
            for username in all_stargazers:
                try:
                    starred_repos = self.github_api.list_starred_repos(username, limit=200)
                    user_to_repos[username] = {r["full_name"] for r in starred_repos}
                except Exception as e:
                    logger.debug(f"Error getting starred repos for {username}: {str(e)}")
                    user_to_repos[username] = set()
                    
            # Create lock-step detector
            lockstep_detector = LockStepDetector(user_to_repos)
            cluster_scores = lockstep_detector.cluster_scores()

            # Process each burst
            for burst_idx, burst in enumerate(sorted_bursts_for_scoring):
                if total_users_scored_so_far >= max_total_users_to_score:
                    logger.info(f"Reached max total users to score ({max_total_users_to_score}). Skipping further user scoring.")
                    # Add remaining bursts without user scores, or with minimal info
                    burst_copy = burst.copy()
                    burst_copy.update({
                        "user_scores": {}, "likely_fake_users": [], "likely_fake_count": 0,
                        "sampled_users_count": 0, "fake_ratio": 0,
                        "scoring_skipped": True
                    })
                    scored_bursts_output.append(burst_copy)
                    continue

                users_in_burst = burst.get("users", [])
                if not users_in_burst:
                    scored_bursts_output.append(burst) # Add as is if no users
                    continue

                # Sample users from this burst
                users_to_score_this_burst = users_in_burst
                if len(users_in_burst) > max_users_to_score_per_burst:
                    users_to_score_this_burst = random.sample(users_in_burst, max_users_to_score_per_burst)
                
                # Further limit if close to max_total_users_to_score
                remaining_total_slots = max_total_users_to_score - total_users_scored_so_far
                if len(users_to_score_this_burst) > remaining_total_slots:
                    users_to_score_this_burst = users_to_score_this_burst[:remaining_total_slots]

                burst_user_evals = {} # Renamed from burst_user_scores to avoid confusion with final score
                likely_fake_usernames_in_burst = []

                desc = f"Scoring users in burst {burst_idx+1}/{len(sorted_bursts_for_scoring)} ({burst['start_date']})"
                for username in tqdm(users_to_score_this_burst, desc=desc, disable=len(users_to_score_this_burst) < 10):
                    if not username: continue # Skip if username is None or empty

                    if username in all_user_scores: # Already scored globally
                        user_eval = all_user_scores[username]
                    else:
                        user_profile = self.github_api.get_user(username)
                        if not user_profile or "login" not in user_profile: # Ensure profile is valid
                            logger.debug(f"Skipping scoring for {username}, profile not found or invalid.")
                            continue 
                        
                        score_breakdown = {}
                        # 1. Account age
                        account_age_days_val = None
                        if user_profile.get("created_at"):
                            created_at_dt = make_naive_datetime(parse_date(user_profile["created_at"]))
                            if created_at_dt:
                                account_age_days_val = (make_naive_datetime(datetime.datetime.now()) - created_at_dt).days
                                age_thresh, age_score = USER_SCORE_THRESHOLDS["account_age_days"]
                                score_breakdown["account_age"] = _logistic_score(age_thresh - account_age_days_val, mid=0, max_score=age_score)
                        
                        # 2. Followers
                        followers_val = user_profile.get("followers", 0)
                        foll_thresh, foll_score = USER_SCORE_THRESHOLDS["followers"]
                        score_breakdown["followers"] = _logistic_score(foll_thresh - followers_val, mid=0, max_score=foll_score)
                        
                        # 3. Public repos
                        pub_repos_val = user_profile.get("public_repos", 0)
                        repo_thresh, repo_score = USER_SCORE_THRESHOLDS["public_repos"]
                        score_breakdown["public_repos"] = _logistic_score(repo_thresh - pub_repos_val, mid=0, max_score=repo_score)

                        # 4. User's total starred repos (expensive, uses GraphQL)
                        user_total_stars_val = None
                        # Check if other scores already push this over the threshold to save API call
                        current_score_sum = sum(score_breakdown.values())
                        stars_component_max_score = USER_SCORE_THRESHOLDS["total_stars"][1]

                        if current_score_sum + stars_component_max_score < FAKE_USER_THRESHOLD : # Only query if it can make a difference
                            gql_query_user_stars = f"""query {{ user(login: "{username}") {{ starredRepositories {{ totalCount }} }} }}"""
                            gql_result = self.github_api.graphql_request(gql_query_user_stars)
                            if gql_result and not gql_result.get("errors") and gql_result.get("data", {}).get("user"):
                                user_total_stars_val = gql_result["data"]["user"].get("starredRepositories", {}).get("totalCount")
                        
                        if user_total_stars_val is not None:
                            star_thresh, star_score_val = USER_SCORE_THRESHOLDS["total_stars"]
                            score_breakdown["total_stars"] = _logistic_score(star_thresh - user_total_stars_val, mid=0, max_score=star_score_val)
                        else: # If GraphQL failed or skipped
                             score_breakdown["total_stars"] = 0 # Neutral or slightly suspicious if cannot fetch

                        # 5. Prior interaction with THIS repo
                        interaction_data = self.github_api.check_user_repo_interaction(self.owner, self.repo, username)
                        has_prior_interaction = interaction_data.get("has_any_interaction", False)
                        _, interact_score = USER_SCORE_THRESHOLDS["prior_interaction"]
                        score_breakdown["prior_interaction"] = interact_score if not has_prior_interaction else 0
                        
                        # 6. Default avatar
                        has_default_avatar_flag = "avatar_url" in user_profile and \
                                                  ("gravatar.com/avatar/00000000000000000000000000000000" in user_profile["avatar_url"] or \
                                                   "avatars.githubusercontent.com/u/0?" in user_profile["avatar_url"] or \
                                                   "identicons" in user_profile["avatar_url"] or # Common for default
                                                   "no-avatar" in user_profile["avatar_url"]) # Check for common default patterns
                        _, avatar_score = USER_SCORE_THRESHOLDS["default_avatar"]
                        score_breakdown["default_avatar"] = avatar_score if has_default_avatar_flag else 0

                        # 7. Get user events for activity pattern analysis
                        events = None
                        if hasattr(self.github_api, 'list_user_events'):
                            events = self.github_api.list_user_events(username, per_page=300)
                        
                        if events:
                            # 7.1 Longest inactivity period
                            longest_inactivity_val = 0
                            if len(events) >= 2:
                                event_dates = sorted([make_naive_datetime(parse_date(e["created_at"])) for e in events])
                                gaps = [(event_dates[i+1] - event_dates[i]).days for i in range(len(event_dates)-1)]
                                longest_inactivity_val = max(gaps) if gaps else 0
                                
                            inactivity_thresh, inactivity_score = USER_SCORE_THRESHOLDS["longest_inactivity"]
                            score_breakdown["longest_inactivity"] = _logistic_score(longest_inactivity_val - inactivity_thresh, mid=0, max_score=inactivity_score) if longest_inactivity_val > inactivity_thresh else 0
                            
                            # 7.2 Contribution dispersion (Gini coefficient)
                            contribution_gini_val = 0.0
                            if events:
                                weeks = [make_naive_datetime(parse_date(e["created_at"])).isocalendar()[1] for e in events]
                                week_counts = Counter(weeks).values()
                                contribution_gini_val = _gini(list(week_counts))
                                
                            gini_thresh, gini_score = USER_SCORE_THRESHOLDS["contribution_gini"]
                            score_breakdown["contribution_gini"] = _logistic_score(contribution_gini_val - gini_thresh, mid=0, max_score=gini_score) if contribution_gini_val > gini_thresh else 0
                            
                            # 7.3 Time-of-day entropy
                            tod_entropy_val = 0.0
                            if events:
                                hours = [make_naive_datetime(parse_date(e["created_at"])).hour for e in events]
                                tod_entropy_val = _entropy(hours)
                                
                            entropy_thresh, entropy_score = USER_SCORE_THRESHOLDS["tod_entropy"]
                            score_breakdown["tod_entropy"] = entropy_score if tod_entropy_val < entropy_thresh else 0
                        
                        # 8. Lock-step behavior
                        lockstep_score_val = cluster_scores.get(username, 0.0)
                        lockstep_thresh, lockstep_score = USER_SCORE_THRESHOLDS["lockstep_score"]
                        score_breakdown["lockstep"] = _logistic_score(lockstep_score_val - lockstep_thresh, mid=0, max_score=lockstep_score) if lockstep_score_val > lockstep_thresh else 0
                        
                        # 9. Avatar hash comparison - commented out as in original

                        final_total_score = sum(score_breakdown.values())
                        
                        # Adjust score if we have estimated dates to be more conservative
                        if burst.get("has_estimated_dates", False):
                            # Reduce the final score by 20% to avoid false positives with estimated dates
                            final_total_score = final_total_score * 0.8
                        
                        user_eval = {
                            "username": username,
                            "account_age_days": account_age_days_val,
                            "followers": followers_val,
                            "public_repos": pub_repos_val,
                            "total_stars_by_user": user_total_stars_val, # Renamed for clarity
                            "has_interaction_with_repo": has_prior_interaction,
                            "has_default_avatar": has_default_avatar_flag,
                            "longest_inactivity": longest_inactivity_val if events else None,
                            "contribution_gini": contribution_gini_val if events else None,
                            "tod_entropy": tod_entropy_val if events else None,
                            "lockstep_score": lockstep_score_val,
                            "score_components": score_breakdown,
                            "total_score": final_total_score,
                            "has_estimated_dates": burst.get("has_estimated_dates", False),
                            "likely_fake_profile": final_total_score >= FAKE_USER_THRESHOLD
                        }
                        all_user_scores[username] = user_eval
                    
                    burst_user_evals[username] = user_eval
                    if user_eval["likely_fake_profile"]:
                        likely_fake_usernames_in_burst.append(username)
                    total_users_scored_so_far +=1
                    if total_users_scored_so_far >= max_total_users_to_score: break # Break inner loop too

                burst_copy = burst.copy() # Work on a copy
                burst_copy.update({
                    "user_evaluations": burst_user_evals, # Store evaluations for this burst
                    "likely_fake_users_in_burst": likely_fake_usernames_in_burst,
                    "likely_fake_count_in_burst": len(likely_fake_usernames_in_burst),
                    "sampled_users_in_burst_count": len(burst_user_evals),
                    "fake_ratio_in_burst": len(likely_fake_usernames_in_burst) / len(burst_user_evals) if burst_user_evals else 0,
                    "scoring_skipped": False
                })
                scored_bursts_output.append(burst_copy)

            return {"bursts": scored_bursts_output, "user_scores_cache": all_user_scores}

        except Exception as e:
            logger.error(f"Error scoring stargazers for {self.owner}/{self.repo}: {str(e)}", exc_info=True)
            # Return original bursts, so subsequent steps don't fail on missing keys
            return {"bursts": self.bursts, "user_scores_cache": {}}

    def analyze_temporal_bursts(self) -> Dict:
        """Analyze temporal burst features at repo level."""
        # Get stargazers with timestamps
        stargazers = self.github_api.get_stargazers(self.owner, self.repo, get_timestamps=True)
        
        if not stargazers:
            return {
                "burst_duration": 0,
                "tod_entropy": 0.0,
                "suspicious_patterns": []
            }
            
        # Extract timestamps and convert to datetime
        star_dates = [make_naive_datetime(parse_date(s["starred_at"])) for s in stargazers]
        
        # Create daily counts
        date_counter = Counter([d.date() for d in star_dates])
        daily_stars = sorted(((d, c) for d, c in date_counter.items()), key=lambda x: x[0])
        
        if not daily_stars:
            return {
                "burst_duration": 0,
                "tod_entropy": 0.0,
                "suspicious_patterns": []
            }
            
        # Calculate 95th percentile threshold for burst detection
        counts = [c for _, c in daily_stars]
        p95 = np.percentile(counts, 95) if counts else 0
        
        # Find bursts (consecutive days above 95th percentile)
        bursts = []
        current_burst = []
        
        for date, count in daily_stars:
            if count > p95:
                current_burst.append((date, count))
            else:
                if current_burst:
                    bursts.append(current_burst)
                    current_burst = []
                    
        if current_burst:  # Add the last burst if exists
            bursts.append(current_burst)
            
        # Calculate burst durations
        burst_durations = []
        for burst in bursts:
            if len(burst) > 0:
                start_date = burst[0][0]
                end_date = burst[-1][0]
                duration = (end_date - start_date).days + 1
                burst_durations.append(duration)
        
        # Time of day entropy
        hours = [d.hour for d in star_dates]
        tod_entropy = _entropy(hours)
        
        # Check if we're using estimated dates
        has_estimated_dates = any(s.get("date_is_estimated", False) for s in stargazers)
        
        # Suspicious patterns detection
        suspicious_patterns = []
        
        # Skip some checks if we're using estimated dates to avoid false positives
        if not has_estimated_dates:
            # Short bursts with many stars
            for burst in bursts:
                total_stars = sum(count for _, count in burst)
                duration = (burst[-1][0] - burst[0][0]).days + 1 if len(burst) > 0 else 0
                
                if duration <= 2 and total_stars > 20:
                    suspicious_patterns.append({
                        "type": "short_intense_burst",
                        "start_date": burst[0][0].isoformat(),
                        "end_date": burst[-1][0].isoformat(),
                        "stars": total_stars,
                        "duration_days": duration
                    })
            
            # Low time-of-day entropy
            if tod_entropy < 1.5:
                suspicious_patterns.append({
                    "type": "low_tod_entropy",
                    "entropy": tod_entropy,
                    "message": "Stars concentrated in specific hours, suggesting automated activity"
                })
        
        return {
            "burst_durations": burst_durations,
            "avg_burst_duration": np.mean(burst_durations) if burst_durations else 0,
            "tod_entropy": tod_entropy,
            "suspicious_patterns": suspicious_patterns,
            "has_estimated_dates": has_estimated_dates
        }

    def calculate_fake_star_index(self) -> Dict:
        """Calculate the Fake Star Index for the repository."""
        default_return = {
            "has_fake_stars": False, "fake_star_index": 0.0, "risk_level": "low",
            "bursts": [], "total_stars_analyzed": 0, "total_likely_fake": 0,
            "fake_percentage": 0.0, "worst_burst": None
        }
        try:
            scoring_result = self.score_stargazers() # This returns {"bursts": ..., "user_scores_cache": ...}
            processed_bursts = scoring_result.get("bursts", [])

            if not processed_bursts:
                logger.info(f"No bursts to analyze for fake star index for {self.owner}/{self.repo}.")
                default_return["bursts"] = self.bursts # Return original bursts if any
                return default_return

            # Get temporal burst analysis
            temporal_analysis = self.analyze_temporal_bursts()
            has_estimated_dates = temporal_analysis.get("has_estimated_dates", False)

            final_bursts_with_scores = []
            for burst in processed_bursts:
                # Use the new keys from score_stargazers
                fake_ratio = burst.get("fake_ratio_in_burst", 0)
                # Normalize rule_hits (0-5 range now) to 0-1. Max 5 rules.
                normalized_rule_hits = burst.get("rule_hits", 0) / 5.0 
                
                # Apply temporal factors
                temporal_factor = 0.0
                tod_entropy = burst.get("tod_entropy", float('inf'))
                if tod_entropy < 1.5 and not has_estimated_dates:
                    temporal_factor = 0.2  # Add weight for suspicious time patterns
                
                burst_score = (FAKE_RATIO_WEIGHT * fake_ratio) + (RULE_HITS_WEIGHT * normalized_rule_hits) + temporal_factor
                
                # Adjust score if using estimated dates
                if burst.get("has_estimated_dates", False) or has_estimated_dates:
                    # Reduce burst score by 25% when using estimated dates
                    burst_score = burst_score * 0.75
                    
                burst_score = min(max(burst_score, 0.0), 1.0) # Clamp to 0-1

                verdict = "organic"
                if burst_score >= BURST_FAKE_THRESHOLD: verdict = "fake"
                elif burst_score >= BURST_ORGANIC_THRESHOLD: verdict = "suspicious"
                
                burst_copy = burst.copy()
                burst_copy.update({
                    "burst_score": burst_score, 
                    "verdict": verdict,
                    "temporal_factors": {
                        "tod_entropy": tod_entropy,
                        "suspicious_timing": tod_entropy < 1.5 and not has_estimated_dates
                    },
                    "has_estimated_dates": burst.get("has_estimated_dates", False) or has_estimated_dates
                })
                
                # Add warning if using estimated dates
                if burst.get("has_estimated_dates", False) or has_estimated_dates:
                    burst_copy["date_estimation_warning"] = "This burst analysis uses estimated dates and may be less accurate"
                    
                final_bursts_with_scores.append(burst_copy)

            # Repo-level metrics based on all stars in all detected bursts 
            total_stars_in_all_bursts = sum(b["stars"] for b in self.bursts if "stars" in b)
            
            # Calculate weighted sum of likely fake stars from scored bursts
            # Extrapolate if some bursts were not scored due to limits.
            estimated_total_fake_in_bursts = 0
            stars_in_scored_bursts = 0

            for b_scored in final_bursts_with_scores:
                if not b_scored.get("scoring_skipped", True) and "stars" in b_scored:
                    stars_in_scored_bursts += b_scored["stars"]
                    # Estimate fakes for this burst: ratio * stars_in_this_burst
                    estimated_total_fake_in_bursts += b_scored.get("fake_ratio_in_burst", 0) * b_scored["stars"]

            # Overall fake percentage across scored bursts
            avg_fake_ratio_in_scored_bursts = (estimated_total_fake_in_bursts / stars_in_scored_bursts) \
                                              if stars_in_scored_bursts > 0 else 0
            
            # Extrapolate to all bursts if some were skipped
            if total_stars_in_all_bursts > stars_in_scored_bursts and stars_in_scored_bursts > 0:
                 stars_in_unscored_bursts = total_stars_in_all_bursts - stars_in_scored_bursts
                 estimated_total_fake_in_bursts += avg_fake_ratio_in_scored_bursts * stars_in_unscored_bursts
            
            # Ensure it's an integer
            estimated_total_fake_in_bursts = int(round(estimated_total_fake_in_bursts))

            # Calculate repo index with temporal factors
            repo_index_val = 0.0
            if total_stars_in_all_bursts > 0:
                # Weighted average of burst_score by stars in burst
                weighted_score_sum = sum(b.get("burst_score", 0) * b.get("stars", 0) 
                                        for b in final_bursts_with_scores if "stars" in b)
                repo_index_val = weighted_score_sum / total_stars_in_all_bursts
                
                # Incorporate time-of-day entropy from temporal analysis if not using estimated dates
                if not has_estimated_dates and temporal_analysis.get("tod_entropy", float('inf')) < 1.5:
                    repo_index_val = min(1.0, repo_index_val + 0.2)  # Boost score for suspicious timing
                    
                # Add penalty for short bursts if not using estimated dates
                if not has_estimated_dates and temporal_analysis.get("avg_burst_duration", 0) <= 2 and temporal_analysis.get("avg_burst_duration", 0) > 0:
                    repo_index_val = min(1.0, repo_index_val + 0.15)  # Boost score for short bursts
                
            repo_index_val = min(max(repo_index_val, 0.0), 1.0) # Clamp
            
            # Reduce index if using estimated dates to be more conservative
            if has_estimated_dates:
                repo_index_val = repo_index_val * 0.75  # 25% reduction
                repo_index_val = min(max(repo_index_val, 0.0), 0.85)  # Cap at 0.85 to avoid definitive "fake" verdict

            repo_risk_level = "low"
            if repo_index_val >= BURST_FAKE_THRESHOLD: repo_risk_level = "high"
            elif repo_index_val >= BURST_ORGANIC_THRESHOLD: repo_risk_level = "medium"
            
            worst_burst_obj = None
            if final_bursts_with_scores:
                worst_burst_obj = max(final_bursts_with_scores, key=lambda b: b.get("burst_score", 0), default=None)

            result = {
                "has_fake_stars": repo_index_val > BURST_ORGANIC_THRESHOLD,
                "fake_star_index": repo_index_val,
                "risk_level": repo_risk_level,
                "total_stars_analyzed": total_stars_in_all_bursts, # Stars in all detected bursts
                "total_likely_fake": estimated_total_fake_in_bursts, # Estimated fakes across all bursts
                "fake_percentage": (estimated_total_fake_in_bursts / total_stars_in_all_bursts * 100) if total_stars_in_all_bursts > 0 else 0.0,
                "bursts": final_bursts_with_scores, # these are the processed ones
                "worst_burst": worst_burst_obj,
                "temporal_analysis": temporal_analysis
            }
            
            # Add warning if dates are estimated
            if has_estimated_dates:
                result["warning"] = "This analysis uses estimated dates because precise star timestamps weren't available. Results may be less reliable."
                result["dates_estimated"] = True

            return result

        except Exception as e:
            logger.error(f"Error calculating fake star index for {self.owner}/{self.repo}: {str(e)}", exc_info=True)
            default_return["error"] = str(e)
            default_return["bursts"] = self.bursts # Return original bursts if error during processing
            return default_return


    def plot_star_history(self, save_path: str = None) -> None:
        """Plot star history with burst windows and anomalies highlighted."""
        try:
            if self.stars_df is None or self.stars_df.empty:
                logger.info(f"Attempting to build timeseries for plotting {self.owner}/{self.repo}")
                self.stars_df = self.build_daily_timeseries() # Ensure data is loaded

            if self.stars_df.empty:
                logger.warning(f"No star data to plot for {self.owner}/{self.repo}")
                return

            plt.figure(figsize=(15, 8))
            ax = plt.gca()
            
            # Ensure 'date' is plottable (datetime-like) and 'stars' are numeric
            plot_dates = pd.to_datetime(self.stars_df["date"]) # Ensure datetime for plotting
            plot_stars = pd.to_numeric(self.stars_df["stars"], errors='coerce').fillna(0)

            ax.bar(plot_dates, plot_stars, color="lightblue", width=0.9, alpha=0.7, label="Daily Stars")

            # Plot median and threshold if available and not all NaN
            median_data = self.stars_df[self.stars_df["median"].notna()] if "median" in self.stars_df.columns else pd.DataFrame()
            mad_data = self.stars_df[self.stars_df["mad"].notna()] if "mad" in self.stars_df.columns else pd.DataFrame()

            if not median_data.empty:
                ax.plot(pd.to_datetime(median_data["date"]), median_data["median"], color="blue",
                         linestyle="--", linewidth=1.5, label="Median (sliding window)")
            
            if not median_data.empty and not mad_data.empty and "mad" in median_data.columns: # check mad column exists
                # Align median and mad data before calculating threshold
                aligned_data = pd.merge(median_data[['date', 'median']], mad_data[['date', 'mad']], on='date', how='inner')
                if not aligned_data.empty:
                    # Check if we're using estimated dates for threshold adjustment
                    has_estimated_dates = self.stars_df["has_estimated_dates"].any() if "has_estimated_dates" in self.stars_df.columns else False
                    mad_multiplier = MAD_THRESHOLD * 1.5 if has_estimated_dates else MAD_THRESHOLD
                    
                    threshold_values = aligned_data["median"] + mad_multiplier * aligned_data["mad"]
                    ax.plot(pd.to_datetime(aligned_data["date"]), threshold_values, color="red",
                             linestyle=":", linewidth=1.5, label=f"Anomaly Threshold (Median + {mad_multiplier:.1f}*MAD)")

            y_max_val = plot_stars.max()
            plot_y_max = max(10, y_max_val * 1.1) if y_max_val > 0 else 10 # Ensure y_max is at least 10

            # Highlight burst windows from self.bursts (which should be populated by calculate_fake_star_index)
            if hasattr(self, 'bursts') and self.bursts:
                # Deduplicate legend entries for bursts
                legend_labels_done = set()

                for burst in self.bursts:
                    # Ensure dates are datetime objects for axvspan
                    start_dt = datetime.datetime.combine(burst["start_date"], datetime.time.min)
                    end_dt = datetime.datetime.combine(burst["end_date"], datetime.time.max) # Cover full end day

                    burst_verdict = burst.get("verdict", "unknown")
                    burst_score_val = burst.get("burst_score", 0)
                    has_estimated_dates = burst.get("has_estimated_dates", False)
                    
                    color, alpha, label_base = "lightgray", 0.3, "Burst"
                    if burst_verdict == "fake": color, alpha, label_base = "red", 0.5, f"Fake Burst"
                    elif burst_verdict == "suspicious": color, alpha, label_base = "orange", 0.4, f"Suspicious Burst"
                    elif burst_verdict == "organic": color, alpha, label_base = "lightgreen", 0.3, f"Organic Burst"
                    
                    # Add score to label if not 'unknown' and estimated status if needed
                    if burst_verdict != "unknown":
                        full_label = f"{label_base} ({burst_score_val:.2f})"
                        if has_estimated_dates:
                            full_label += " [Est. Dates]"
                    else:
                        full_label = label_base
                    
                    if full_label not in legend_labels_done:
                        ax.axvspan(start_dt, end_dt, alpha=alpha, color=color, label=full_label)
                        legend_labels_done.add(full_label)
                    else: # Plot without adding to legend again
                        ax.axvspan(start_dt, end_dt, alpha=alpha, color=color)

                    mid_date_plot = start_dt + (end_dt - start_dt) / 2
                    ax.text(mid_date_plot, plot_y_max * 0.95, f"+{int(burst['stars'])} stars",
                             ha='center', va='top', fontsize=8, color='black',
                             bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.7))

            # Add time-of-day entropy annotation if available
            if "tod_entropy" in self.stars_df.columns and not self.stars_df.empty:
                tod_entropy_val = self.stars_df["tod_entropy"].iloc[0]
                tod_color = "green" if tod_entropy_val >= 1.5 else "red"
                ax.text(0.02, 0.02, f"Time-of-day entropy: {tod_entropy_val:.2f}", 
                        transform=ax.transAxes, color=tod_color, fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.7))
                
            # Add estimated dates warning if applicable
            has_estimated_dates = self.stars_df["has_estimated_dates"].any() if "has_estimated_dates" in self.stars_df.columns else False
            if has_estimated_dates:
                ax.text(0.5, 0.97, "⚠️ Using estimated dates - burst detection may be less reliable", 
                        transform=ax.transAxes, color="darkred", fontsize=10, ha='center',
                        bbox=dict(boxstyle="round,pad=0.2", fc="lightyellow", ec="orange", alpha=0.7))

            ax.set_xlabel("Date", fontsize=12)
            ax.set_ylabel("Stars per Day", fontsize=12)
            ax.set_title(f"Star History & Burst Analysis for {self.owner}/{self.repo}", fontsize=14)
            
            ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=12))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=30, ha='right', fontsize=10)
            plt.yticks(fontsize=10)
            
            ax.yaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=5))
            ax.grid(True, linestyle='--', alpha=0.5)
            
            handles, labels = ax.get_legend_handles_labels()
            unique_labels = dict(zip(labels, handles)) # Use dict to store unique labels
            ax.legend(unique_labels.values(), unique_labels.keys(), loc='upper left', fontsize=10)
            
            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Star history plot saved to {save_path}")
            else:
                plt.show()
            plt.close() # Close plot to free memory

        except Exception as e:
            logger.error(f"Error plotting star history for {self.owner}/{self.repo}: {str(e)}", exc_info=True)