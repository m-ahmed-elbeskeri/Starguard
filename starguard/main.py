"""Main StarGuard analysis engine."""

import datetime
import json
from typing import Dict, List, Optional
import logging

from starguard.analyzers.burst_detector import BurstDetector
from starguard.analyzers.code_analyzer import CodeAnalyzer
from starguard.analyzers.dependency_analyzer import DependencyAnalyzer
from starguard.analyzers.license_analyzer import LicenseAnalyzer
from starguard.analyzers.maintainer_analyzer import MaintainerAnalyzer
from starguard.analyzers.star_analyzer import StarAnalyzer
from starguard.api.github_api_imports import GitHubAPI
from starguard.core.trust_score import TrustScoreCalculator
from starguard.utils.date_utils import make_naive_datetime

logger = logging.getLogger(__name__)


class StarGuard:
    """
    Main StarGuard analysis engine.
    """

    def __init__(self, tokens: Optional[List[str]] = None):
        """
        Initialize the StarGuard engine.

        Args:
            tokens: One or more GitHub API tokens
        """
        self.github_api = GitHubAPI(tokens)
        
        # Warn about analysis limitations without tokens
        if not self.github_api.token_manager.tokens:
            logger.warning(
                "🔑 No GitHub token provided. This significantly limits analysis quality:\n"
                "   - Star timestamps will be estimated (less accurate fake star detection)\n" 
                "   - Dependency analysis may be incomplete\n"
                "   - User profile analysis will be limited\n"
                "   - Rate limits are much lower (60 vs 5000 requests/hour)\n"
                "   Get a token at: https://github.com/settings/tokens"
            )

    def analyze_repo(self, owner: str, repo: str, analyze_fake_stars: bool = True) -> Dict:
        """Perform comprehensive analysis on a GitHub repository."""
        logger.info(f"Starting analysis of {owner}/{repo}")

        # Fetch basic repository data
        try:
            repo_data = self.github_api.get_repo(owner, repo)
            logger.info(
                f"Fetched repository data for: {repo_data.get('full_name', f'{owner}/{repo}')}"
            )
        except ValueError as e:
            logger.error(f"Failed to fetch repository {owner}/{repo}: {e}")
            return {"error": str(e)}

        # Fetch stargazers data
        stars_raw_data = self.github_api.get_stargazers(
            owner, repo, get_timestamps=True, days_limit=0
        )
        logger.info(f"Fetched {len(stars_raw_data)} stargazers total.")
        
        # Check if we're using estimated dates and warn user
        estimated_dates_count = sum(1 for star in stars_raw_data if star.get("date_is_estimated", False))
        if estimated_dates_count > 0:
            estimated_ratio = estimated_dates_count / len(stars_raw_data) if stars_raw_data else 0
            if estimated_ratio > 0.5:
                logger.warning(
                    f"⚠️  Using estimated star dates for {estimated_ratio:.1%} of stars due to API limitations. "
                    f"Fake star detection may be less accurate. Consider using a GitHub token."
                )

        # Fetch contributors data
        contributors_data = self.github_api.get_contributors(owner, repo)
        logger.info(f"Fetched {len(contributors_data)} contributors.")

        # Fetch recent commits (last 90 days)
        since_90d_iso = (
            make_naive_datetime(datetime.datetime.now()) - datetime.timedelta(days=90)
        ).isoformat()
        commits_recent_data = self.github_api.get_commits(owner, repo, since=since_90d_iso)
        logger.info(f"Fetched {len(commits_recent_data)} commits from last 90 days.")

        # Fetch dependencies with improved error handling
        dependencies_raw_data = self.github_api.get_dependencies(owner, repo)
        dep_source = "API" if "sbom" in dependencies_raw_data else "File Parsing"
        if "error" in dependencies_raw_data:
            dep_source = "Failed"
            logger.warning(f"Dependency analysis failed: {dependencies_raw_data.get('error')}")
        else:
            logger.info(f"Fetched dependency data (Source: {dep_source}).")

        # Fetch license information
        license_api_data = self.github_api.get_license(owner, repo)
        license_info = license_api_data.get('license', {}).get('spdx_id', 'N/A')
        logger.info(f"Fetched license info: {license_info}")

        # Initialize analyzers
        burst_detector_inst = BurstDetector(owner, repo, self.github_api)
        star_analyzer_inst = StarAnalyzer(
            owner, repo, stars_raw_data, repo_data["created_at"], self.github_api
        )
        dependency_analyzer_inst = DependencyAnalyzer(dependencies_raw_data, self.github_api)
        license_analyzer_inst = LicenseAnalyzer(
            license_api_data, dependency_analyzer_inst.flat_dependencies
        )
        maintainer_analyzer_inst = MaintainerAnalyzer(
            contributors_data, commits_recent_data, self.github_api
        )
        code_analyzer_inst = CodeAnalyzer(owner, repo, self.github_api)

        # Perform analyses with error handling
        logger.info("Running star pattern analysis...")
        try:
            star_analysis_results = star_analyzer_inst.detect_anomalies()
        except Exception as e:
            logger.error(f"Star pattern analysis failed: {e}")
            star_analysis_results = {"anomalies": [], "score": 25, "error": str(e)}

        # Fake star detection
        fake_star_analysis_results = None
        if analyze_fake_stars:
            logger.info("Running fake star detection (BurstDetector)...")
            try:
                fake_star_analysis_results = star_analyzer_inst.detect_fake_stars()
                fsi_val = fake_star_analysis_results.get("fake_star_index", 0.0)
                fsi_risk = fake_star_analysis_results.get("risk_level", "low")
                
                # Add context about estimated dates if applicable
                if fake_star_analysis_results.get("dates_estimated", False):
                    logger.info(f"Fake Star Index: {fsi_val:.2f} ({fsi_risk.upper()} RISK) - Based on estimated dates")
                else:
                    logger.info(f"Fake Star Index: {fsi_val:.2f} ({fsi_risk.upper()} RISK)")
            except Exception as e:
                logger.error(f"Fake star detection failed: {e}")
                fake_star_analysis_results = {
                    "has_fake_stars": False,
                    "fake_star_index": 0.0,
                    "risk_level": "low",
                    "bursts": [],
                    "total_stars_analyzed": 0,
                    "total_likely_fake": 0,
                    "fake_percentage": 0.0,
                    "worst_burst": None,
                    "error": str(e),
                }
        else:
            fake_star_analysis_results = {
                "has_fake_stars": False,
                "fake_star_index": 0.0,
                "risk_level": "low",
                "bursts": [],
                "total_stars_analyzed": 0,
                "total_likely_fake": 0,
                "fake_percentage": 0.0,
                "worst_burst": None,
                "message": "Fake star analysis skipped by user.",
            }

        logger.info("Running dependency analysis...")
        try:
            dependency_analysis_results = dependency_analyzer_inst.analyze()
        except Exception as e:
            logger.error(f"Dependency analysis failed: {e}")
            dependency_analysis_results = {"dependencies": [], "score": 15, "error": str(e)}

        logger.info("Running license analysis...")
        try:
            license_analysis_results = license_analyzer_inst.analyze()
        except Exception as e:
            logger.error(f"License analysis failed: {e}")
            license_analysis_results = {"score": 10, "error": str(e)}

        logger.info("Running maintainer analysis...")
        try:
            maintainer_analysis_results = maintainer_analyzer_inst.analyze()
        except Exception as e:
            logger.error(f"Maintainer analysis failed: {e}")
            maintainer_analysis_results = {"score": 5, "error": str(e)}

        logger.info("Running suspicious code pattern check...")
        try:
            code_analysis_results = code_analyzer_inst.check_for_suspicious_patterns()
        except Exception as e:
            logger.error(f"Code analysis failed: {e}")
            code_analysis_results = {
                "findings_by_category": {},
                "suspicious_package_elements": [],
                "total_suspicious_hits": 0,
                "calculated_suspicion_score": 0,
                "is_potentially_suspicious": False,
                "error": str(e)
            }

        # Calculate trust score
        try:
            trust_calculator_inst = TrustScoreCalculator(
                star_analysis_results,
                dependency_analysis_results,
                license_analysis_results,
                maintainer_analysis_results,
                fake_star_analysis_results,
            )
            trust_score_final = trust_calculator_inst.calculate()
            badge_url_str = trust_calculator_inst.generate_badge_url(owner, repo)
        except Exception as e:
            logger.error(f"Trust score calculation failed: {e}")
            trust_score_final = {"total_score": 50, "risk_level": "medium", "error": str(e)}
            badge_url_str = "https://img.shields.io/badge/StarGuard+Score-Error-red.svg"

        # Additional informational outputs
        try:
            activity_info_detailed = maintainer_analyzer_inst.check_recent_activity()
        except Exception as e:
            logger.warning(f"Activity analysis failed: {e}")
            activity_info_detailed = {"error": str(e)}

        try:
            package_registry_info = dependency_analyzer_inst.check_in_package_registries()
        except Exception as e:
            logger.warning(f"Package registry check failed: {e}")
            package_registry_info = {"error": str(e)}

        # Prepare final result dictionary
        final_result_dict = {
            "repository_info": {
                "name": repo_data.get("name"),
                "full_name": repo_data.get("full_name"),
                "description": repo_data.get("description", ""),
                "stars_count": repo_data.get("stargazers_count"),
                "created_at": repo_data.get("created_at"),
                "html_url": repo_data.get("html_url"),
                "language": repo_data.get("language", "N/A"),
            },
            "trust_score_summary": trust_score_final,
            "star_pattern_analysis": star_analysis_results,
            "fake_star_detection_analysis": fake_star_analysis_results,
            "dependency_health_analysis": dependency_analysis_results,
            "license_compliance_analysis": license_analysis_results,
            "maintainer_activity_analysis": maintainer_analysis_results,
            "code_suspicion_analysis": code_analysis_results,
            "detailed_activity_metrics": activity_info_detailed,
            "package_registry_check": package_registry_info,
            "generated_badge": {
                "url": badge_url_str,
                "markdown": f"[![StarGuard Score]({badge_url_str})](https://starguard.example.com/report/{owner}/{repo})",
            },
            "analysis_metadata": {
                "estimated_star_dates_ratio": estimated_ratio if 'estimated_ratio' in locals() else 0.0,
                "has_github_token": bool(self.github_api.token_manager.tokens),
                "dependency_source": dep_source,
                "analysis_timestamp": datetime.datetime.now().isoformat(),
            }
        }
        
        logger.info(
            f"Analysis for {owner}/{repo} complete. Trust Score: {trust_score_final.get('total_score', 'N/A')}"
        )
        return final_result_dict

    def generate_report(self, analysis_result: Dict, format_str: str = "text") -> str:
        """Generate a formatted report from analysis results."""
        if "error" in analysis_result:
            return f"Error: {analysis_result['error']}"

        if format_str == "json":
            def dt_handler(o):
                if isinstance(o, (datetime.datetime, datetime.date)):
                    return o.isoformat()
                raise TypeError(f"Type {type(o)} not serializable")

            return json.dumps(analysis_result, indent=2, default=dt_handler)

        repo_info = analysis_result.get("repository_info", {})
        trust_summary = analysis_result.get("trust_score_summary", {})
        fake_star_info = analysis_result.get("fake_star_detection_analysis", {})
        code_sus_info = analysis_result.get("code_suspicion_analysis", {})
        analysis_meta = analysis_result.get("analysis_metadata", {})

        md_report_lines = []
        text_report_lines = []

        # Header
        title = f"StarGuard Analysis: {repo_info.get('full_name', 'N/A')}"
        md_report_lines.extend([f"# {title}", ""])
        text_report_lines.extend([title, "=" * len(title), ""])

        # Analysis quality warnings
        if not analysis_meta.get("has_github_token", True):
            warning_text = "⚠️ **Analysis Limitation**: No GitHub token provided. Results may be less accurate."
            md_report_lines.extend([f"> {warning_text}", ""])
            text_report_lines.extend([f"WARNING: {warning_text.replace('**', '').replace('⚠️ ', '')}", ""])

        estimated_ratio = analysis_meta.get("estimated_star_dates_ratio", 0.0)
        if estimated_ratio > 0.5:
            warning_text = f"⚠️ **Date Estimation**: {estimated_ratio:.1%} of star dates are estimated due to API limits."
            md_report_lines.extend([f"> {warning_text}", ""])
            text_report_lines.extend([f"WARNING: {warning_text.replace('**', '').replace('⚠️ ', '')}", ""])

        # Overview section
        overview_md = [
            "## Overview",
            f"- **Repository**: [{repo_info.get('full_name','N/A')}]({repo_info.get('html_url','')})",
            f"- **Description**: {repo_info.get('description','N/A')}",
            f"- **Created**: {repo_info.get('created_at','N/A')}",
            f"- **Stars**: {repo_info.get('stars_count','N/A')}",
            f"- **Primary Language**: {repo_info.get('language','N/A')}",
            "",
        ]
        overview_text = [
            "Overview:",
            f"  Repository: {repo_info.get('full_name','N/A')} ({repo_info.get('html_url','')})",
            f"  Description: {repo_info.get('description','N/A')}",
            f"  Created: {repo_info.get('created_at','N/A')}",
            f"  Stars: {repo_info.get('stars_count','N/A')}",
            f"  Primary Language: {repo_info.get('language','N/A')}",
            "",
        ]
        md_report_lines.extend(overview_md)
        text_report_lines.extend(overview_text)

        # Trust Score section
        trust_score_val = trust_summary.get("total_score", "N/A")
        trust_risk_lvl = trust_summary.get("risk_level", "N/A").upper()
        md_report_lines.extend(
            [f"## Trust Score: {trust_score_val}/100 ({trust_risk_lvl} RISK)", ""]
        )
        text_report_lines.extend(
            [f"TRUST SCORE: {trust_score_val}/100 ({trust_risk_lvl} RISK)", ""]
        )

        # Show fake star penalty if applied
        penalty = trust_summary.get("fake_star_penalty_applied", 0)
        if penalty > 0:
            md_report_lines.append(
                f"**Fake Star Penalty Applied**: -{penalty} points from base score."
            )
            text_report_lines.append(
                f"  Fake Star Penalty Applied: -{penalty} points from base score."
            )
        md_report_lines.append("")
        text_report_lines.append("")

        # Fake Star Detection section
        if fake_star_info and fake_star_info.get("has_fake_stars"):
            fsi = fake_star_info.get("fake_star_index", 0.0)
            fsi_risk = fake_star_info.get("risk_level", "N/A").upper()
            fsi_likely_fake = fake_star_info.get("total_likely_fake", 0)
            fsi_perc = fake_star_info.get("fake_percentage", 0.0)
            fsi_burst_count = len(fake_star_info.get("bursts", []))

            alert_emoji = "🚨" if fsi_risk == "HIGH" else "⚠️"
            md_report_lines.extend([f"## {alert_emoji} Fake Star Detection {alert_emoji}", ""])
            text_report_lines.extend(
                [f"!!! {alert_emoji} Fake Star Detection {alert_emoji} !!!", "-" * 30, ""]
            )

            md_report_lines.extend(
                [
                    f"**Fake Star Index**: {fsi:.2f} ({fsi_risk} RISK)",
                    f"**Likely Fake Stars**: {fsi_likely_fake} ({fsi_perc:.1f}% of those analyzed in bursts)",
                    f"**Suspicious Bursts Detected**: {fsi_burst_count}",
                    "",
                ]
            )
            text_report_lines.extend(
                [
                    f"  Fake Star Index: {fsi:.2f} ({fsi_risk} RISK)",
                    f"  Likely Fake Stars: {fsi_likely_fake} ({fsi_perc:.1f}% of those analyzed in bursts)",
                    f"  Suspicious Bursts Detected: {fsi_burst_count}",
                    "",
                ]
            )

            # Show burst details
            if fake_star_info.get("bursts"):
                md_report_lines.append("### Suspicious Star Burst Details:")
                text_report_lines.append("  Suspicious Star Burst Details:")
                for idx, burst_item in enumerate(fake_star_info["bursts"][:3]):
                    b_verdict = burst_item.get("verdict", "N/A").upper()
                    b_score = burst_item.get("burst_score", 0.0)
                    b_start = burst_item.get("start_date", "N/A")
                    b_end = burst_item.get("end_date", "N/A")
                    b_stars = burst_item.get("stars", 0)
                    b_fake_ratio = burst_item.get("fake_ratio_in_burst", 0.0) * 100

                    md_report_lines.append(
                        f"- **Burst {idx+1}**: {b_verdict} (Score: {b_score:.2f}), Period: {b_start} to {b_end}, Stars: +{b_stars}, Estimated Fake Ratio: {b_fake_ratio:.1f}%"
                    )
                    text_report_lines.append(
                        f"    Burst {idx+1}: {b_verdict} (Score: {b_score:.2f})"
                    )
                    text_report_lines.append(
                        f"      Period: {b_start} to {b_end}, Stars: +{b_stars}, Estimated Fake Ratio: {b_fake_ratio:.1f}%"
                    )
                if len(fake_star_info["bursts"]) > 3:
                    md_report_lines.append("- ...and other bursts.")
                    text_report_lines.append("    ...and other bursts.")
                md_report_lines.append("")
                text_report_lines.append("")
        elif "message" in fake_star_info:
            md_report_lines.extend(["## Fake Star Detection", f"_{fake_star_info['message']}_", ""])
            text_report_lines.extend(["Fake Star Detection:", f"  {fake_star_info['message']}", ""])

        # Suspicious Code Detection section
        if code_sus_info and code_sus_info.get("is_potentially_suspicious"):
            cs_score = code_sus_info.get("calculated_suspicion_score", 0)
            md_report_lines.extend(["## ⚠️ Suspicious Code Detection", ""])
            text_report_lines.extend(["!!! ⚠️ SUSPICIOUS CODE DETECTION !!!", "-" * 30, ""])

            md_report_lines.extend([f"**Code Suspicion Score**: {cs_score}/100", ""])
            text_report_lines.extend([f"  Code Suspicion Score: {cs_score}/100", ""])

            if code_sus_info.get("findings_by_category"):
                md_report_lines.append("Suspicious patterns found:")
                text_report_lines.append("  Suspicious patterns found:")
                for cat, finds in code_sus_info["findings_by_category"].items():
                    if finds:
                        md_report_lines.append(
                            f"- **{cat.replace('_',' ').title()}**: {len(finds)} instances"
                        )
                        text_report_lines.append(
                            f"    - {cat.replace('_',' ').title()}: {len(finds)} instances"
                        )
            
            if code_sus_info.get("suspicious_package_elements"):
                md_report_lines.append("Suspicious elements in package manifest:")
                text_report_lines.append("  Suspicious elements in package manifest:")
                for elem_desc in code_sus_info["suspicious_package_elements"][:3]:
                    md_report_lines.append(f"- {elem_desc}")
                    text_report_lines.append(f"    - {elem_desc}")
            md_report_lines.append("")
            text_report_lines.append("")

        # Component analysis sections
        sections = {
            "Star Pattern Analysis": analysis_result.get("star_pattern_analysis"),
            "Dependency Health": analysis_result.get("dependency_health_analysis"),
            "License Compliance": analysis_result.get("license_compliance_analysis"),
            "Maintainer Activity": analysis_result.get("maintainer_activity_analysis"),
        }
        
        score_comp_map = trust_summary.get("score_components", {})
        score_key_map = {
            "Star Pattern Analysis": "star_pattern_score",
            "Dependency Health": "dependencies_score",
            "License Compliance": "license_score",
            "Maintainer Activity": "maintainer_activity_score",
        }

        for section_title, data in sections.items():
            if not data:
                continue
            
            score_val = score_comp_map.get(score_key_map.get(section_title), "N/A")
            max_score = {
                "Star Pattern Analysis": 40,
                "Dependency Health": 25,  
                "License Compliance": 20,
                "Maintainer Activity": 15
            }.get(section_title, 20)

            md_report_lines.extend(
                [f"## {section_title}", f"**Component Score**: {score_val}/{max_score}", ""]
            )
            text_report_lines.extend(
                [f"{section_title.upper()}:", f"  Component Score: {score_val}/{max_score}", ""]
            )

            # Add specific details for each section
            if section_title == "Star Pattern Analysis" and data.get("anomalies"):
                num_anom = len(data["anomalies"])
                md_report_lines.append(f"- Detected {num_anom} anomalies in star pattern.")
                text_report_lines.append(f"  - Detected {num_anom} anomalies in star pattern.")
                
                # Show detailed anomaly breakdown in text reports
                if format_str != "json" and num_anom > 0:
                    md_report_lines.append("  - **Top Anomalies**:")
                    text_report_lines.append("    Anomaly Details:")
                    for i, anomaly in enumerate(data["anomalies"][:3]):  # Show top 3
                        date = anomaly.get("date", "N/A")
                        stars = anomaly.get("stars", 0)
                        method = anomaly.get("method", "N/A")
                        z_score = anomaly.get("z_score", 0)
                        
                        md_report_lines.append(f"    - {date}: {stars} stars (Z-score: {z_score:.2f}, Method: {method})")
                        text_report_lines.append(f"      {date}: {stars} stars (Z-score: {z_score:.2f}, Method: {method})")
                        
            elif section_title == "Dependency Health" and data.get("stats"):
                stats = data["stats"]
                md_report_lines.append(
                    f"- Analyzed {stats.get('total_analyzed',0)} dependencies: {stats.get('high_risk',0)} high risk, {stats.get('medium_risk',0)} medium risk."
                )
                text_report_lines.append(
                    f"  - Analyzed {stats.get('total_analyzed',0)} dependencies: {stats.get('high_risk',0)} high risk, {stats.get('medium_risk',0)} medium risk."
                )
            elif section_title == "License Compliance":
                lic_key = data.get("repo_license_key", "N/A")
                lic_risk = data.get("repo_license_risk", "N/A").upper()
                md_report_lines.append(f"- Repository License: `{lic_key}` (Risk: {lic_risk}).")
                text_report_lines.append(f"  - Repository License: {lic_key} (Risk: {lic_risk}).")
            elif section_title == "Maintainer Activity" and data.get("recent_activity_summary"):
                act_sum = data["recent_activity_summary"]
                md_report_lines.append(
                    f"- {act_sum.get('commits_last_90d',0)} commits in the last 90 days; {act_sum.get('active_maintainers_heuristic',0)} active maintainers (among top contributors)."
                )
                text_report_lines.append(
                    f"  - {act_sum.get('commits_last_90d',0)} commits in the last 90 days; {act_sum.get('active_maintainers_heuristic',0)} active maintainers (among top contributors)."
                )

            md_report_lines.append("")
            text_report_lines.append("")

        # Badge section
        badge_info = analysis_result.get("generated_badge", {})
        if badge_info.get("url"):
            md_report_lines.extend(["## Badge", "", badge_info["markdown"], ""])
            text_report_lines.extend(["BADGE", "", badge_info["markdown"], ""])

        if format_str == "markdown":
            return "\n".join(md_report_lines)
        else:
            return "\n".join(text_report_lines)