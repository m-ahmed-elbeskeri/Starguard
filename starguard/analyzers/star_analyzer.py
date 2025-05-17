"""Analyzes star patterns to detect anomalies."""

import datetime
import logging
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from dateutil.parser import parse as parse_date

from starguard.api.github_api import GitHubAPI
from starguard.analyzers.burst_detector import BurstDetector
from starguard.utils.date_utils import make_naive_datetime

logger = logging.getLogger(__name__)

class StarAnalyzer:
    """Analyzes star patterns to detect anomalies."""
    
    def __init__(self, owner: str, repo: str, stars_data: List[Dict], repo_created_date: str, github_api: GitHubAPI):
        self.owner = owner
        self.repo = repo
        self.stars_data = stars_data # This is raw data from get_stargazers
        try:
            self.repo_created_date = make_naive_datetime(parse_date(repo_created_date))
        except (TypeError, ValueError) as e:
            logger.warning(f"Invalid repo_created_date '{repo_created_date}': {e}. Defaulting to distant past.")
            self.repo_created_date = make_naive_datetime(datetime.datetime(2000,1,1)) # A sensible default

        self.github_api = github_api # Now required
        self.df = self._prepare_dataframe() # Uses self.stars_data
        
        # Initialize BurstDetector here if StarAnalyzer is the primary user of it for anomaly scores
        # Or, it can be passed in if already created by StarGuard main logic
        self.burst_detector = BurstDetector(owner, repo, github_api) # StarAnalyzer can use its own instance

    def _prepare_dataframe(self) -> Optional[pd.DataFrame]:
        """Prepare a dataframe with star data for analysis."""
        if not self.stars_data:
            logger.warning(f"No star data provided to StarAnalyzer for {self.owner}/{self.repo}")
            return pd.DataFrame() # Return empty DataFrame

        records = []
        for star_entry in self.stars_data:
            try:
                if "starred_at" in star_entry and star_entry["starred_at"]:
                    dt_obj = make_naive_datetime(parse_date(star_entry["starred_at"]))
                    if dt_obj:
                        records.append({
                            "date": dt_obj, # Keep as datetime objects
                            "user": star_entry.get("user", {}).get("login")
                        })
            except Exception as e:
                logger.debug(f"Error processing star data entry in StarAnalyzer: {str(e)}")

        if not records:
            logger.warning(f"No valid timestamp data after parsing in StarAnalyzer for {self.owner}/{self.repo}")
            return pd.DataFrame()

        df = pd.DataFrame(records)
        df.sort_values("date", inplace=True)

        # Group by day and count for timeseries analysis
        # Ensure 'date' column used for grouping is just the date part if original 'date' has time
        daily_df = df.groupby(df["date"].dt.date).size().reset_index(name="stars")
        daily_df["date"] = pd.to_datetime(daily_df["date"]) # Convert date objects back to Timestamps for consistency

        # Fill in missing dates
        if self.repo_created_date and not daily_df.empty:
            # Ensure repo_created_date is Timestamp for pd.date_range
            start_date_ts = pd.Timestamp(self.repo_created_date)
            end_date_ts = max(pd.Timestamp(make_naive_datetime(datetime.datetime.now())), daily_df["date"].max())

            all_dates_df = pd.DataFrame({"date": pd.date_range(start=start_date_ts, end=end_date_ts, freq='D')})
            
            result_df = pd.merge(all_dates_df, daily_df, on="date", how="left")
            result_df["stars"] = result_df["stars"].fillna(0).astype(int)
        elif not daily_df.empty: # If repo_created_date is not available, use data range
            result_df = daily_df
        else: # No data at all
            return pd.DataFrame()


        # Add features
        result_df["day_of_week"] = result_df["date"].dt.dayofweek
        result_df["is_weekend"] = result_df["day_of_week"].isin([5, 6]).astype(int)
        
        # Rolling windows - ensure enough data points for window, else NaNs are fine.
        if len(result_df) >= 3:
            result_df["rolling_3d"] = result_df["stars"].rolling(window=3, min_periods=1).mean()
        else: result_df["rolling_3d"] = np.nan
        if len(result_df) >= 7:
            result_df["rolling_7d"] = result_df["stars"].rolling(window=7, min_periods=1).mean()
        else: result_df["rolling_7d"] = np.nan
        if len(result_df) >= 30:
            result_df["rolling_30d"] = result_df["stars"].rolling(window=30, min_periods=1).mean()
        else: result_df["rolling_30d"] = np.nan
        
        return result_df

    def _detect_with_percentiles(self, percentiles=[5, 10, 25, 50, 75, 90, 95]) -> Dict:
        """Detect anomalies using percentile analysis of star patterns"""
        if self.df is None or self.df.empty or "stars" not in self.df.columns:
            return {"percentile_analysis": "insufficient_data"}
        
        # Calculate percentiles of daily star counts
        percentile_values = {}
        for p in percentiles:
            percentile_values[f"p{p}"] = float(np.percentile(self.df["stars"], p))
        
        # Detect unusual gaps between percentiles (sign of purchased stars)
        anomalies = []
        
        # Check if there's an unusually large gap between higher percentiles
        # This indicates a few days with extremely high star counts
        if percentile_values["p95"] > 3 * percentile_values["p75"] and percentile_values["p95"] > 20:
            anomalies.append({
                "type": "high_percentile_gap",
                "message": f"Large gap between 95th and 75th percentiles: {percentile_values['p95']} vs {percentile_values['p75']}",
                "severity": "high"
            })
        
        # Check if median is unusually low compared to mean (skewed distribution)
        mean_stars = float(self.df["stars"].mean())
        if mean_stars > 2 * percentile_values["p50"] and mean_stars > 5:
            anomalies.append({
                "type": "mean_median_gap",
                "message": f"Mean ({mean_stars:.1f}) much higher than median ({percentile_values['p50']:.1f})",
                "severity": "medium"
            })
        
        # Calculate skewness (concentration of stars in few days)
        if len(self.df) > 10 and percentile_values["p95"] > 0:
            days_with_stars = len(self.df[self.df["stars"] > 0])
            days_with_high_stars = len(self.df[self.df["stars"] > percentile_values["p75"]])
            concentration_ratio = days_with_high_stars / max(1, days_with_stars)
            
            if concentration_ratio < 0.1 and percentile_values["p95"] > 20:
                anomalies.append({
                    "type": "star_concentration",
                    "message": f"Stars concentrated in very few days ({days_with_high_stars} days with high activity)",
                    "severity": "high"
                })
        
        # Determine overall suspicion level from percentile analysis
        suspicion_level = "low"
        if any(a["severity"] == "high" for a in anomalies):
            suspicion_level = "high"
        elif any(a["severity"] == "medium" for a in anomalies):
            suspicion_level = "medium"
        
        return {
            "percentile_values": percentile_values,
            "anomalies": anomalies,
            "suspicion_level": suspicion_level
        }

    def detect_anomalies(self) -> Dict:
        """Detect anomalies in star patterns using multiple methods."""
        if self.df is None or self.df.empty:
             return {"anomalies": [], "score": 50, "error": "No star data for anomaly detection."} # Neutral score

        # Initialize with an empty list or from previous calculation
        mad_anomalies_list = []
        # Use BurstDetector for MAD-based anomalies
        if self.burst_detector:
            # Ensure bursts are detected if not already
            if not self.burst_detector.bursts:
                 self.burst_detector.detect_bursts() # This populates self.burst_detector.bursts

            for burst in self.burst_detector.bursts:
                # burst["start_date"] and burst["end_date"] are datetime.date objects
                current_d = burst["start_date"]
                while current_d <= burst["end_date"]:
                    # Find stars for this day from self.df (StarAnalyzer's daily timeseries)
                    # self.df['date'] is Timestamp. current_d is datetime.date.
                    day_data = self.df[self.df["date"].dt.date == current_d]
                    stars_on_day = day_data["stars"].values[0] if not day_data.empty else 0
                    
                    mad_anomalies_list.append({
                        "date": datetime.datetime.combine(current_d, datetime.time.min), # Store as datetime
                        "stars": int(stars_on_day),
                        "z_score": np.nan, # Not applicable for MAD method here
                        "method": "mad_burst_day"
                    })
                    current_d += datetime.timedelta(days=1)
        
        # Other anomaly detection methods (Z-score, Isolation Forest, Spikes)
        z_score_anomalies_list = self._detect_with_zscore()
        isolation_forest_anomalies_list = self._detect_with_isolation_forest()
        spike_anomalies_list = self._detect_spikes()

        # NEW: Add percentile-based analysis
        percentile_analysis = self._detect_with_percentiles()
        
        all_found_anomalies = z_score_anomalies_list + isolation_forest_anomalies_list + \
                              spike_anomalies_list + mad_anomalies_list
        
        # Deduplicate anomalies by date
        unique_anomalies_by_date = {}
        for anomaly in all_found_anomalies:
            # Anomaly "date" should be datetime object. Use .date() part for key.
            anomaly_date_key = anomaly["date"].date() 
            if anomaly_date_key not in unique_anomalies_by_date:
                unique_anomalies_by_date[anomaly_date_key] = anomaly
            else: # If date exists, merge methods or keep the one with higher severity indication
                if anomaly["stars"] > unique_anomalies_by_date[anomaly_date_key]["stars"]:
                     unique_anomalies_by_date[anomaly_date_key] = anomaly # Keep higher star count for that day
                # Append method if different
                existing_method = unique_anomalies_by_date[anomaly_date_key]["method"]
                if anomaly["method"] not in existing_method:
                    unique_anomalies_by_date[anomaly_date_key]["method"] += f", {anomaly['method']}"

        unique_anomalies_list = sorted(list(unique_anomalies_by_date.values()), key=lambda x: x["date"], reverse=True)

        # Score: 0-50. Higher is better.
        # Max 10 anomalies reduce score by 5 each. More than 10 anomalies -> score 0.
        anomaly_penalty = len(unique_anomalies_list) * 5
        
        # NEW: Adjust penalty based on percentile analysis results
        if percentile_analysis.get("suspicion_level") == "high":
            anomaly_penalty += 15  # Additional penalty for high suspicion from percentiles
        elif percentile_analysis.get("suspicion_level") == "medium":
            anomaly_penalty += 7   # Additional penalty for medium suspicion
            
        final_score = max(0, 50 - anomaly_penalty)

        result = {
            "anomalies": unique_anomalies_list, 
            "score": final_score,
            "percentile_analysis": percentile_analysis  # Include the percentile analysis in results
        }
        
        return result

    def _detect_with_zscore(self, threshold: float = 3.0) -> List[Dict]:
        """Detect anomalies using Z-score method."""
        if self.df is None or self.df.empty or "stars" not in self.df.columns: return []
        df_copy = self.df.copy()
        
        mean_stars = df_copy["stars"].mean()
        std_stars = df_copy["stars"].std()

        if std_stars == 0 or pd.isna(std_stars): return [] # Avoid division by zero or NaN std

        df_copy["z_score"] = (df_copy["stars"] - mean_stars) / std_stars
        anomalies_df = df_copy[df_copy["z_score"].abs() > threshold]
        
        return [
            {"date": row["date"].to_pydatetime(), "stars": int(row["stars"]), 
             "z_score": float(row["z_score"]), "method": "z-score"}
            for _, row in anomalies_df.iterrows()
        ]

    def _detect_with_isolation_forest(self, contamination: Union[str, float] = 'auto') -> List[Dict]:
        """Detect anomalies using Isolation Forest algorithm."""
        if self.df is None or self.df.empty or len(self.df) < 10: return [] # Need enough data
        
        # Use features like 'stars' and 'rolling_7d'. Ensure 'rolling_7d' exists and is filled.
        features_to_use = ["stars"]
        if "rolling_7d" in self.df.columns and self.df["rolling_7d"].notna().any():
            features_to_use.append("rolling_7d")
        
        df_analysis = self.df[features_to_use].copy()
        df_analysis.fillna(0, inplace=True) # Fill NaNs in features (e.g. initial rolling mean)

        # Check for variance
        if any(df_analysis[col].std() == 0 for col in features_to_use):
            return [] 

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_analysis)

        try:
            # IsolationForest can be sensitive to contamination, 'auto' is often a good start.
            # If contamination is too high for sparse anomalies, it might flag too many.
            clf = IsolationForest(contamination=contamination, random_state=42, n_estimators=100)
            anomaly_predictions = clf.fit_predict(X_scaled)
        except ValueError as e: # e.g. contamination not in range
            logger.warning(f"IsolationForest ValueError: {e}. Trying with default contamination 0.1")
            try:
                clf = IsolationForest(contamination=0.1, random_state=42, n_estimators=100)
                anomaly_predictions = clf.fit_predict(X_scaled)
            except Exception as e_inner:
                logger.error(f"IsolationForest failed even with default contamination: {e_inner}")
                return []


        # Anomalies are -1
        anomaly_indices = self.df.index[anomaly_predictions == -1]
        anomalies_df = self.df.loc[anomaly_indices]

        # Calculate z-score for context, even if not used for detection by this method
        mean_stars_all = self.df["stars"].mean()
        std_stars_all = self.df["stars"].std()
        z_score_calc = lambda stars: (stars - mean_stars_all) / std_stars_all if std_stars_all > 0 else 0.0

        return [
            {"date": row["date"].to_pydatetime(), "stars": int(row["stars"]),
             "z_score": float(z_score_calc(row["stars"])), "method": "isolation_forest"}
            for _, row in anomalies_df.iterrows()
        ]


    def _detect_spikes(self, threshold_multiplier: float = 5.0, min_stars_for_spike: int = 5) -> List[Dict]:
        """Detect sudden spikes in star activity."""
        if self.df is None or self.df.empty or "stars" not in self.df.columns: return []
        df_copy = self.df.copy()

        # Average stars on days with >0 stars, excluding current day being checked
        # This is complex. Simpler: spike if stars > N AND stars > M * historical_avg
        
        historical_avg_stars = df_copy["stars"].rolling(window=30, min_periods=7).mean().shift(1) # Avg of past 30 days, lagged
        df_copy["historical_avg"] = historical_avg_stars.fillna(df_copy["stars"].expanding().mean().shift(1)) # Fallback for early data

        spikes_df = df_copy[
            (df_copy["stars"] > min_stars_for_spike) & 
            (df_copy["stars"] > df_copy["historical_avg"] * threshold_multiplier)
        ]
        
        # Calculate z-score for context
        mean_stars_all = self.df["stars"].mean()
        std_stars_all = self.df["stars"].std()
        z_score_calc = lambda stars: (stars - mean_stars_all) / std_stars_all if std_stars_all > 0 else 0.0

        return [
            {"date": row["date"].to_pydatetime(), "stars": int(row["stars"]),
             "z_score": float(z_score_calc(row["stars"])), "method": "spike_detection"}
            for _, row in spikes_df.iterrows()
        ]


    def detect_fake_stars(self) -> Dict:
        """Detect fake stars using the repository-only approach with BurstDetector."""
        if not self.github_api: # Should not happen if constructor enforces it
            return {"has_fake_stars": False, "bursts": [], "error": "GitHub API instance not provided"}
        
        try:
            # BurstDetector is already initialized in StarAnalyzer's __init__
            result = self.burst_detector.calculate_fake_star_index()
            
            # NEW: Add percentile analysis to enhance fake star detection
            percentile_analysis = self._detect_with_percentiles()
            
            # Adjust fake star index based on percentile analysis
            if percentile_analysis.get("suspicion_level") == "high":
                base_fsi = result.get("fake_star_index", 0.0)
                result["fake_star_index"] = min(1.0, base_fsi + 0.3)
                
                # If risk level needs upgrade based on new FSI
                new_fsi = result["fake_star_index"]
                if new_fsi >= 0.7 and result.get("risk_level") != "high":
                    result["risk_level"] = "high"
                elif new_fsi >= 0.4 and result.get("risk_level") == "low":
                    result["risk_level"] = "medium"
                    
            elif percentile_analysis.get("suspicion_level") == "medium":
                base_fsi = result.get("fake_star_index", 0.0)
                result["fake_star_index"] = min(1.0, base_fsi + 0.15)
                
                # Potential risk level upgrade if near threshold
                if result["fake_star_index"] >= 0.4 and result.get("risk_level") == "low":
                    result["risk_level"] = "medium"
            
            # Add percentile analysis to the result
            result["percentile_analysis"] = percentile_analysis
            
            return result
            
        except Exception as e:
            logger.error(f"Error detecting fake stars for {self.owner}/{self.repo}: {str(e)}", exc_info=True)
            return {
                "has_fake_stars": False, "fake_star_index": 0.0, "risk_level": "low",
                "error": str(e), "bursts": [],
                "total_stars_analyzed": 0, "total_likely_fake": 0, "fake_percentage": 0.0, "worst_burst": None
            }

    def plot_star_history(self, save_path: str = None) -> None:
        """Plot star history with anomalies and bursts."""
        try:
            # Use BurstDetector's plot if available, as it's more detailed for fake star context
            if self.burst_detector:
                # Ensure burst_detector has up-to-date data based on StarAnalyzer's view
                if self.burst_detector.stars_df is None or self.burst_detector.stars_df.empty:
                    self.burst_detector.stars_df = self.df # Share the prepared DataFrame
                if not self.burst_detector.bursts: # If bursts aren't calculated yet for plotting
                    self.burst_detector.calculate_fake_star_index() # This populates bursts with scores
                
                self.burst_detector.plot_star_history(save_path)
                return

            # Fallback basic plot (should ideally not be reached if BurstDetector is always used)
            if self.df is None or self.df.empty:
                logger.warning(f"No star data to plot for {self.owner}/{self.repo} (StarAnalyzer fallback plot)")
                return

            plt.figure(figsize=(14, 7))
            ax = plt.gca()
            ax.plot(pd.to_datetime(self.df["date"]), self.df["stars"], marker='.', linestyle='-', alpha=0.6, label="Daily Stars")
            if "rolling_7d" in self.df.columns and self.df["rolling_7d"].notna().any():
                ax.plot(pd.to_datetime(self.df["date"]), self.df["rolling_7d"], color="tomato", linestyle='--', label="7-day Rolling Avg")

            anomaly_data = self.detect_anomalies() # This will use BurstDetector internally if configured
            if anomaly_data.get("anomalies"):
                anomaly_dates = [pd.to_datetime(a["date"]) for a in anomaly_data["anomalies"]]
                anomaly_stars = [a["stars"] for a in anomaly_data["anomalies"]]
                ax.scatter(anomaly_dates, anomaly_stars, color="red", s=80, label="Detected Anomalies", zorder=5)
            
            ax.set_title(f"Star History for {self.owner}/{self.repo}", fontsize=14)
            ax.set_xlabel("Date", fontsize=12)
            ax.set_ylabel("Stars per Day", fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=30, ha='right')
            plt.tight_layout()

            if save_path: plt.savefig(save_path, dpi=300)
            else: plt.show()
            plt.close()

        except Exception as e:
            logger.error(f"Error plotting star history for {self.owner}/{self.repo}: {str(e)}", exc_info=True)