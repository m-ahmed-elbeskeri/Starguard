"""Detects star burst patterns using robust statistical methods."""

import datetime
import logging
import random
import math
import re
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
    # Prevent overflow by clamping the exponent argument
    exponent_arg = steepness * (x - mid)
    
    # Clamp to prevent overflow: exp(700) ≈ 1e304, exp(-700) ≈ 0
    if exponent_arg > 700:
        return 0.0  # When exponent is very large, 1/(1+exp(large)) ≈ 0
    elif exponent_arg < -700:
        return max_score  # When exponent is very negative, 1/(1+exp(small)) ≈ max_score
    else:
        return max_score / (1 + math.exp(exponent_arg))


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
            merged_df["has_estimated_dates"] = merged_df["has_estimated_dates"].fillna(False).infer_objects(copy=False)

            
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

    def _analyze_username_patterns(self, username: str) -> float:
        """Detect bot-like username patterns"""
        if not username:
            return 1.0
        
        suspicion_score = 0.0
        
        # Pattern 1: Random character sequences
        if re.search(r'[a-z]{3,}\d{4,}$', username.lower()):  # like user1234
            suspicion_score += 1.5
        
        # Pattern 2: Word + numbers pattern
        if re.search(r'^[a-zA-Z]+\d+$', username) and len(username) > 8:
            suspicion_score += 1.0
        
        # Pattern 3: Repeating patterns
        if len(set(username.lower())) < len(username) * 0.6:  # Low character diversity
            suspicion_score += 0.8
        
        # Pattern 4: GitHub default patterns
        if re.search(r'github|user|account', username.lower()):
            suspicion_score += 0.5
        
        return min(suspicion_score, 3.0)

    def _analyze_interaction_quality(self, starred_repos: List[Dict]) -> float:
        """Analyze quality of repository interactions"""
        if not starred_repos:
            return 2.0  # Suspicious if no starred repos
        
        suspicion_score = 0.0
        
        # Check for mass-starring patterns
        repo_languages = [repo.get("language") for repo in starred_repos if repo.get("language")]
        language_diversity = len(set(repo_languages)) / max(len(repo_languages), 1)
        
        if language_diversity < 0.3:  # Low diversity suggests bot behavior
            suspicion_score += 1.5
        
        # Check for popular-only repositories (trend following)
        high_star_repos = [r for r in starred_repos if r.get("stargazers_count", 0) > 1000]
        if len(high_star_repos) / len(starred_repos) > 0.8:  # Only stars popular repos
            suspicion_score += 1.0
        
        return min(suspicion_score, 3.0)

    def _analyze_account_evolution(self, user_data: Dict) -> float:
        """Analyze if account shows natural evolution"""
        suspicion_score = 0.0
        
        followers = user_data.get("followers", 0)
        following = user_data.get("following", 0)
        public_repos = user_data.get("public_repos", 0)
        
        # Unnatural ratios
        if followers > 0 and following / max(followers, 1) > 10:  # Following way more than followers
            suspicion_score += 1.0
        
        if followers > 100 and public_repos == 0:  # Many followers but no repos
            suspicion_score += 1.5
        
        # Check account age vs activity
        try:
            created_at = user_data.get("created_at")
            if created_at:
                account_age = (make_naive_datetime(datetime.datetime.now()) - 
                              make_naive_datetime(parse_date(created_at))).days
                
                # Very new accounts with lots of activity
                if account_age < 30 and (followers > 50 or public_repos > 10):
                    suspicion_score += 2.0
                elif account_age < 7:  # Very new accounts are suspicious
                    suspicion_score += 1.0
        except:
            pass
        
        return min(suspicion_score, 3.0)

    def _analyze_single_user_enhanced(self, user_data: Dict, starred_repos: List[Dict]) -> Dict:
        """Enhanced user analysis with better heuristics"""
        
        # Current basic scoring
        basic_score = self._calculate_basic_user_score(user_data)
        
        # Enhanced username pattern analysis
        username_suspicion = self._analyze_username_patterns(user_data.get("login", ""))
        
        # Repository interaction quality
        interaction_quality = self._analyze_interaction_quality(starred_repos)
        
        # Account evolution patterns
        evolution_score = self._analyze_account_evolution(user_data)
        
        # Combine scores with weights
        total_score = (
            basic_score * 0.4 +
            username_suspicion * 0.2 +
            interaction_quality * 0.3 +
            evolution_score * 0.1
        )
        
        return {
            "total_score": total_score,
            "components": {
                "basic": basic_score,
                "username_pattern": username_suspicion,
                "interaction_quality": interaction_quality,
                "evolution": evolution_score
            },
            "likely_fake": total_score >= FAKE_USER_THRESHOLD * 0.8  # More conservative
        }

    def _calculate_basic_user_score(self, user_data: Dict) -> float:
        """Calculate basic user score (FIXED LOGIC)"""
        score_breakdown = {}
        
        # 1. Account age - FIXED: older accounts should get lower scores (less suspicious)
        account_age_days_val = None
        if user_data.get("created_at"):
            created_at_dt = make_naive_datetime(parse_date(user_data["created_at"]))
            if created_at_dt:
                account_age_days_val = (make_naive_datetime(datetime.datetime.now()) - created_at_dt).days
                age_thresh, age_score = USER_SCORE_THRESHOLDS["account_age_days"]
                # FIXED: older accounts (higher age) should get lower suspicion scores
                score_breakdown["account_age"] = _logistic_score(account_age_days_val - age_thresh, mid=0, max_score=age_score)
        
        # 2. Followers - FIXED: more followers should get lower scores (less suspicious)
        followers_val = user_data.get("followers", 0)
        foll_thresh, foll_score = USER_SCORE_THRESHOLDS["followers"]
        # FIXED: more followers should get lower suspicion scores
        score_breakdown["followers"] = _logistic_score(followers_val - foll_thresh, mid=0, max_score=foll_score)
        
        # 3. Public repos - FIXED: more repos should get lower scores (less suspicious)
        pub_repos_val = user_data.get("public_repos", 0)
        repo_thresh, repo_score = USER_SCORE_THRESHOLDS["public_repos"]
        # FIXED: more repos should get lower suspicion scores
        score_breakdown["public_repos"] = _logistic_score(pub_repos_val - repo_thresh, mid=0, max_score=repo_score)
        
        return sum(score_breakdown.values())

    def score_stargazers(self,
                        max_users_to_score_per_burst: int = 10_000,
                        max_total_users_to_score: int = 10_000
                        ) -> Dict:
        """Score stargazers in burst windows to identify likely fake accounts."""
        try:
            # Early exit if there are no bursts
            if not self.bursts:
                self.cross_check_bursts()  # Try to detect and cross-check bursts first
            if not self.bursts:
                return {"bursts": [], "user_scores": {}}

            all_user_scores = {}
            scored_bursts_output = []
            total_users_scored = 0

            # 1) Sort bursts by suspicion
            sorted_bursts = sorted(
                self.bursts,
                key=lambda b: (
                    b.get("inorganic_heuristic", False),
                    b.get("rule_hits", 0),
                    b.get("stars", 0),
                ),
                reverse=True
            )

            # 2) Gather unique stargazers (cap for lock-step)
            all_stargazers = {u for b in self.bursts for u in b.get("users", [])}
            max_ls = LOCKSTEP_DETECTION["max_users"]
            if len(all_stargazers) > max_ls:
                all_stargazers = set(random.sample(list(all_stargazers), max_ls))

            # 3) Build lock-step detector map
            user_to_repos = {}
            for username in all_stargazers:
                try:
                    starred = self.github_api.list_starred_repos(username, limit=200)
                    user_to_repos[username] = {r["full_name"] for r in starred}
                except Exception:
                    user_to_repos[username] = set()
            cluster_scores = LockStepDetector(user_to_repos).cluster_scores()

            # 4) Score each burst
            for burst_idx, burst in enumerate(sorted_bursts):
                # Enforce total-users cap
                if total_users_scored >= max_total_users_to_score:
                    skipped = burst.copy()
                    skipped.update({
                        "user_evaluations": {},
                        "likely_fake_users_in_burst": [],
                        "likely_fake_count_in_burst": 0,
                        "sampled_users_in_burst_count": 0,
                        "fake_ratio_in_burst": 0.0,
                        "scoring_skipped": True,
                    })
                    scored_bursts_output.append(skipped)
                    continue

                users = burst.get("users", [])
                if not users:
                    scored_bursts_output.append(burst)
                    continue

                # Sample per-burst if too many
                to_score = users
                if len(to_score) > max_users_to_score_per_burst:
                    to_score = random.sample(to_score, max_users_to_score_per_burst)
                # Don't exceed global cap
                remaining = max_total_users_to_score - total_users_scored
                if len(to_score) > remaining:
                    to_score = to_score[:remaining]

                burst_evals = {}
                likely_fakes = []
                desc = f"Scoring burst {burst_idx+1}/{len(sorted_bursts)} ({burst.get('start_date')})"
                for username in tqdm(to_score, desc=desc, disable=len(to_score) < 10):
                    if not username:
                        continue

                    if username in all_user_scores:
                        user_eval = all_user_scores[username]
                    else:
                        profile = self.github_api.get_user(username)
                        if not profile or "login" not in profile:
                            continue

                        score_breakdown = {}

                        # 1. Account age - FIXED: older accounts should get lower scores (less suspicious)
                        age_days = None
                        if profile.get("created_at"):
                            dt = make_naive_datetime(parse_date(profile["created_at"]))
                            if dt:
                                age_days = (make_naive_datetime(datetime.datetime.now()) - dt).days
                                thresh, max_s = USER_SCORE_THRESHOLDS["account_age_days"]
                                # FIXED: older accounts get lower suspicion scores
                                score_breakdown["account_age"] = _logistic_score(
                                    age_days - thresh, mid=0, max_score=max_s
                                )

                        # 2. Followers - FIXED: more followers should get lower scores (less suspicious)
                        foll = profile.get("followers", 0)
                        thresh, max_s = USER_SCORE_THRESHOLDS["followers"]
                        # FIXED: more followers get lower suspicion scores
                        score_breakdown["followers"] = _logistic_score(
                            foll - thresh, mid=0, max_score=max_s
                        )

                        # 3. Public repos - FIXED: more repos should get lower scores (less suspicious)
                        repos = profile.get("public_repos", 0)
                        thresh, max_s = USER_SCORE_THRESHOLDS["public_repos"]
                        # FIXED: more repos get lower suspicion scores
                        score_breakdown["public_repos"] = _logistic_score(
                            repos - thresh, mid=0, max_score=max_s
                        )

                        # 4. Total stars - FIXED: more stars should get lower scores (less suspicious)
                        total_stars = None
                        current_sum = sum(score_breakdown.values())
                        star_thresh, star_max = USER_SCORE_THRESHOLDS["total_stars"]
                        if current_sum + star_max >= FAKE_USER_THRESHOLD:
                            q = (
                                f'query {{ user(login: "{username}") '
                                '{ starredRepositories {{ totalCount }} } }}'
                            )
                            res = self.github_api.graphql_request(q)
                            if res and not res.get("errors") and res.get("data", {}).get("user"):
                                total_stars = (
                                    res["data"]["user"]["starredRepositories"]["totalCount"]
                                )
                        if total_stars is not None:
                            # FIXED: more stars get lower suspicion scores
                            score_breakdown["total_stars"] = _logistic_score(
                                total_stars - star_thresh, mid=0, max_score=star_max
                            )
                        else:
                            score_breakdown["total_stars"] = 0.0

                        # 5. Prior interaction (no interaction → penalty)
                        interaction = self.github_api.check_user_repo_interaction(
                            self.owner, self.repo, username
                        )
                        _, interact_pen = USER_SCORE_THRESHOLDS["prior_interaction"]
                        has_interaction = interaction.get("has_any_interaction", False)
                        score_breakdown["prior_interaction"] = 0.0 if has_interaction else interact_pen

                        # 6. Default avatar (default → penalty)
                        avatar_url = profile.get("avatar_url", "")
                        default_flag = any(tok in avatar_url for tok in (
                            "00000000000000000000000000000000",
                            "avatars.githubusercontent.com/u/0?",
                            "identicons",
                            "no-avatar",
                        ))
                        _, avatar_pen = USER_SCORE_THRESHOLDS["default_avatar"]
                        score_breakdown["default_avatar"] = avatar_pen if default_flag else 0.0

                        # 7. Activity patterns (these are correctly implemented - higher values = more suspicious)
                        events = []
                        if hasattr(self.github_api, "list_user_events"):
                            events = self.github_api.list_user_events(username, per_page=300)

                        longest_inactivity_val = None
                        gini_val = None
                        entropy_val = None

                        if events:
                            dates = sorted(
                                make_naive_datetime(parse_date(e["created_at"]))
                                for e in events
                            )
                            if len(dates) >= 2:
                                gaps = [
                                    (dates[i+1] - dates[i]).days
                                    for i in range(len(dates)-1)
                                ]
                                longest_inactivity_val = max(gaps)
                            else:
                                longest_inactivity_val = 0
                            in_thresh, in_max = USER_SCORE_THRESHOLDS["longest_inactivity"]
                            # CORRECT: longer inactivity = more suspicious = higher score
                            score_breakdown["longest_inactivity"] = (
                                _logistic_score(longest_inactivity_val - in_thresh,
                                                mid=0, max_score=in_max)
                                if longest_inactivity_val > in_thresh else 0.0
                            )

                            # Contribution Gini - CORRECT: higher Gini = more suspicious = higher score
                            from collections import Counter
                            weeks = [
                                make_naive_datetime(parse_date(e["created_at"])).isocalendar()[1]
                                for e in events
                            ]
                            counts = Counter(weeks).values()
                            gini_val = _gini(list(counts))
                            g_thresh, g_max = USER_SCORE_THRESHOLDS["contribution_gini"]
                            score_breakdown["contribution_gini"] = (
                                _logistic_score(gini_val - g_thresh,
                                                mid=0, max_score=g_max)
                                if gini_val > g_thresh else 0.0
                            )

                            # Time-of-day entropy - CORRECT: lower entropy = more suspicious = higher score
                            hours = [
                                make_naive_datetime(parse_date(e["created_at"])).hour
                                for e in events
                            ]
                            entropy_val = _entropy(hours)
                            ent_thresh, ent_max = USER_SCORE_THRESHOLDS["tod_entropy"]
                            score_breakdown["tod_entropy"] = (
                                ent_max if entropy_val < ent_thresh else 0.0
                            )

                        # 8. Lock-step behavior - CORRECT: higher lockstep score = more suspicious = higher score
                        ls_val = cluster_scores.get(username, 0.0)
                        ls_thresh, ls_max = USER_SCORE_THRESHOLDS["lockstep_score"]
                        score_breakdown["lockstep"] = (
                            _logistic_score(ls_val - ls_thresh, mid=0, max_score=ls_max)
                            if ls_val > ls_thresh else 0.0
                        )

                        # Enhanced analysis
                        starred100 = self.github_api.list_starred_repos(username, limit=100)
                        enhanced = self._analyze_single_user_enhanced(profile, starred100)

                        # Combine scores & adjust for estimated dates
                        total_score = sum(score_breakdown.values()) + enhanced["total_score"] * 0.3
                        if burst.get("has_estimated_dates", False):
                            total_score *= 0.8

                        user_eval = {
                            "username": username,
                            "account_age_days": age_days,
                            "followers": foll,
                            "public_repos": repos,
                            "total_stars_by_user": total_stars,
                            "has_interaction_with_repo": has_interaction,
                            "has_default_avatar": default_flag,
                            "longest_inactivity": longest_inactivity_val,
                            "contribution_gini": gini_val,
                            "tod_entropy": entropy_val,
                            "lockstep_score": ls_val,
                            "score_components": score_breakdown,
                            "enhanced_components": enhanced["components"],
                            "total_score": total_score,
                            "has_estimated_dates": burst.get("has_estimated_dates", False),
                            "likely_fake_profile": total_score >= FAKE_USER_THRESHOLD * 0.8
                        }

                        all_user_scores[username] = user_eval

                    # Record evaluation
                    burst_evals[username] = all_user_scores[username]
                    if burst_evals[username]["likely_fake_profile"]:
                        likely_fakes.append(username)

                    total_users_scored += 1
                    if total_users_scored >= max_total_users_to_score:
                        break

                # Summarize this burst
                out = burst.copy()
                out.update({
                    "user_evaluations":             burst_evals,
                    "likely_fake_users_in_burst":   likely_fakes,
                    "likely_fake_count_in_burst":   len(likely_fakes),
                    "sampled_users_in_burst_count": len(burst_evals),
                    "fake_ratio_in_burst":          (len(likely_fakes) / len(burst_evals)
                                                    if burst_evals else 0.0),
                    "scoring_skipped":              False
                })
                scored_bursts_output.append(out)

            # Return unified structure
            return {"bursts": scored_bursts_output, "user_scores": all_user_scores}

        except Exception as e:
            logger.error(f"Error in score_stargazers: {e}", exc_info=True)
            return {"bursts": self.bursts if hasattr(self, 'bursts') else [], "user_scores": {}}

    def analyze_temporal_bursts(self) -> Dict:
        """Analyze temporal burst features at repo level."""
        # Get stargazers with timestamps
        stargazers = self.github_api.get_stargazers(self.owner, self.repo, get_timestamps=True)
        
        if not stargazers:
            return {
                "burst_duration": 0,
                "tod_entropy": 0.0,
                "suspicious_patterns": [],
                "has_estimated_dates": False
            }
            
        # Extract timestamps and convert to datetime - FIXED: Handle None values
        star_dates = []
        has_estimated_dates = False
        
        for s in stargazers:
            starred_at = s.get("starred_at")
            if starred_at:  # Only process if starred_at is not None/empty
                try:
                    parsed_date = make_naive_datetime(parse_date(starred_at))
                    if parsed_date:
                        star_dates.append(parsed_date)
                        if s.get("date_is_estimated", False):
                            has_estimated_dates = True
                except Exception as e:
                    logger.debug(f"Error parsing starred_at date '{starred_at}': {e}")
                    continue
        
        if not star_dates:
            logger.warning(f"No valid star dates found for {self.owner}/{self.repo}")
            return {
                "burst_duration": 0,
                "tod_entropy": 0.0,
                "suspicious_patterns": [],
                "has_estimated_dates": has_estimated_dates
            }
            
        # Create daily counts
        date_counter = Counter([d.date() for d in star_dates])
        daily_stars = sorted(((d, c) for d, c in date_counter.items()), key=lambda x: x[0])
        
        if not daily_stars:
            return {
                "burst_duration": 0,
                "tod_entropy": 0.0,
                "suspicious_patterns": [],
                "has_estimated_dates": has_estimated_dates
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

    def _calculate_user_diversity(self, burst: Dict) -> float:
        """Calculate diversity of user patterns in a burst"""
        user_evals = burst.get("user_evaluations", {})
        if not user_evals:
            return 0.0
        
        # Analyze diversity of user characteristics
        characteristics = []
        for user_eval in user_evals.values():
            char_vector = [
                user_eval.get("account_age_days", 0) / 365.0,  # Normalize to years
                min(user_eval.get("followers", 0) / 100.0, 1.0),  # Cap at 100
                min(user_eval.get("public_repos", 0) / 10.0, 1.0),  # Cap at 10
            ]
            characteristics.append(char_vector)
        
        if len(characteristics) < 2:
            return 0.0
        
        # Calculate variance across dimensions
        char_array = np.array(characteristics)
        variance = np.mean(np.var(char_array, axis=0))
        
        return min(variance * 2, 1.0)  # Scale to 0-1

    def _cross_validate_burst(self, current_burst: Dict, all_bursts: List[Dict]) -> float:
        """Cross-validate burst characteristics against others"""
        if len(all_bursts) < 2:
            return 0.0
        
        current_fake_ratio = current_burst.get("fake_ratio_in_burst", 0)
        other_ratios = [b.get("fake_ratio_in_burst", 0) for b in all_bursts if b != current_burst]
        
        if not other_ratios:
            return 0.0
        
        # If this burst has much higher fake ratio than others, it's more suspicious
        avg_other_ratio = sum(other_ratios) / len(other_ratios)
        ratio_difference = current_fake_ratio - avg_other_ratio
        
        if ratio_difference > 0.3:  # Significantly higher
            return 0.2
        elif ratio_difference > 0.1:
            return 0.1
        
        return 0.0

    def calculate_fake_star_index(self) -> Dict:
        """Enhanced fake star index calculation with better confidence scoring"""
        default_return = {
            "has_fake_stars": False, "fake_star_index": 0.0, "risk_level": "low",
            "bursts": [], "total_stars_analyzed": 0, "total_likely_fake": 0,
            "fake_percentage": 0.0, "worst_burst": None, "confidence_score": 0.0
        }
        
        try:
            scoring_result = self.score_stargazers()
            processed_bursts = scoring_result.get("bursts", [])
            
            if not processed_bursts:
                logger.info(f"No bursts to analyze for fake star index for {self.owner}/{self.repo}.")
                return default_return
            
            # Enhanced temporal analysis
            temporal_analysis = self.analyze_temporal_bursts()
            has_estimated_dates = temporal_analysis.get("has_estimated_dates", False)
            
            # Calculate more sophisticated burst scores
            final_bursts_with_scores = []
            confidence_factors = []
            
            for burst in processed_bursts:
                fake_ratio = burst.get("fake_ratio_in_burst", 0)
                rule_hits = burst.get("rule_hits", 0)
                
                # Enhanced scoring with multiple factors
                base_score = (FAKE_RATIO_WEIGHT * fake_ratio) + (RULE_HITS_WEIGHT * (rule_hits / 5.0))
                
                # NEW: Time-of-day scoring
                tod_factor = 0.0
                tod_entropy = burst.get("tod_entropy", float('inf'))
                if tod_entropy < 1.0:  # Very suspicious
                    tod_factor = 0.3
                elif tod_entropy < 1.5:
                    tod_factor = 0.15
                
                # NEW: User diversity scoring
                diversity_factor = 0.0
                users_analyzed = burst.get("sampled_users_in_burst_count", 0)
                if users_analyzed > 0:
                    unique_patterns = self._calculate_user_diversity(burst)
                    if unique_patterns < 0.5:  # Low diversity = suspicious
                        diversity_factor = 0.2
                
                # NEW: Cross-validation with other bursts
                cross_validation_factor = self._cross_validate_burst(burst, processed_bursts)
                
                # Combine all factors
                burst_score = base_score + tod_factor + diversity_factor + cross_validation_factor
                
                # Adjust for estimated dates (more conservative)
                if has_estimated_dates:
                    burst_score *= 0.6  # Stronger reduction
                    confidence_factors.append(0.6)
                else:
                    confidence_factors.append(0.9)
                
                burst_score = min(max(burst_score, 0.0), 1.0)
                
                # Enhanced verdict logic
                verdict = "organic"
                if burst_score >= 0.7:  # Stricter threshold
                    verdict = "fake"
                elif burst_score >= 0.4:  # Adjusted threshold
                    verdict = "suspicious"
                
                burst_copy = burst.copy()
                burst_copy.update({
                    "burst_score": burst_score,
                    "verdict": verdict,
                    "confidence_factors": {
                        "tod_entropy": tod_entropy,
                        "user_diversity": diversity_factor,
                        "cross_validation": cross_validation_factor,
                        "estimated_dates_penalty": has_estimated_dates
                    }
                })
                
                final_bursts_with_scores.append(burst_copy)
            
            # Repository-level index calculation
            total_stars_in_bursts = sum(b["stars"] for b in self.bursts if "stars" in b)
            
            if total_stars_in_bursts == 0:
                return default_return
            
            # Weighted average by burst size and confidence
            weighted_score = 0.0
            total_weight = 0.0
            
            for burst in final_bursts_with_scores:
                burst_stars = burst.get("stars", 0)
                burst_score = burst.get("burst_score", 0)
                weight = burst_stars * max(confidence_factors)  # Weight by confidence
                
                weighted_score += burst_score * weight
                total_weight += weight
            
            repo_index = weighted_score / total_weight if total_weight > 0 else 0.0
            
            # Overall confidence calculation
            overall_confidence = sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.0
            
            # Adjust final index based on confidence
            if overall_confidence < 0.7:  # Low confidence
                repo_index *= 0.8  # More conservative
            
            # Risk level with confidence consideration
            risk_level = "low"
            if repo_index >= 0.6 and overall_confidence > 0.7:
                risk_level = "high"
            elif repo_index >= 0.35 and overall_confidence > 0.6:
                risk_level = "medium"
            elif repo_index >= 0.6:  # High score but low confidence
                risk_level = "medium"
            
            # Calculate total fake stars
            total_fake_stars = sum(
                b.get("fake_ratio_in_burst", 0) * b.get("stars", 0) 
                for b in final_bursts_with_scores
            )
            
            result = default_return.copy()
            result.update({
                "has_fake_stars": repo_index > 0.3,
                "fake_star_index": repo_index,
                "risk_level": risk_level,
                "confidence_score": overall_confidence,
                "bursts": final_bursts_with_scores,
                "total_stars_analyzed": total_stars_in_bursts,
                "total_likely_fake": int(total_fake_stars),
                "fake_percentage": (total_fake_stars / total_stars_in_bursts * 100) if total_stars_in_bursts > 0 else 0.0,
                "temporal_analysis": temporal_analysis,
                "worst_burst": max(final_bursts_with_scores, key=lambda b: b.get("burst_score", 0)) if final_bursts_with_scores else None
            })
            
            # Add warning if dates are estimated
            if has_estimated_dates:
                result["warning"] = "This analysis uses estimated dates because precise star timestamps weren't available. Results may be less reliable."
                result["dates_estimated"] = True
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating fake star index: {str(e)}", exc_info=True)
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