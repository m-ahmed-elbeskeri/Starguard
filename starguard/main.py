"""Main StarGuard analysis engine."""

import logging
import datetime
import json
from typing import Dict, Optional

from starguard.api.github_api import GitHubAPI
from starguard.analyzers.star_analyzer import StarAnalyzer
from starguard.analyzers.burst_detector import BurstDetector
from starguard.analyzers.dependency_analyzer import DependencyAnalyzer
from starguard.analyzers.license_analyzer import LicenseAnalyzer
from starguard.analyzers.maintainer_analyzer import MaintainerAnalyzer
from starguard.analyzers.code_analyzer import CodeAnalyzer
from starguard.core.trust_score import TrustScoreCalculator
from starguard.utils.date_utils import make_naive_datetime

logger = logging.getLogger(__name__)

class StarGuard:
    """Main StarGuard analysis engine."""

    def __init__(self, token: Optional[str] = None):
        self.github_api = GitHubAPI(token)

    def analyze_repo(self, owner: str, repo: str, analyze_fake_stars: bool = True) -> Dict:
        """Perform comprehensive analysis on a GitHub repository."""
        logger.info(f"Starting analysis of {owner}/{repo}")

        try:
            repo_data = self.github_api.get_repo(owner, repo) # Raises ValueError if not found or API fails
            logger.info(f"Fetched repository data for: {repo_data.get('full_name', f'{owner}/{repo}')}")
        except ValueError as e: # Catch specific error from get_repo
            logger.error(f"Failed to fetch repository {owner}/{repo}: {e}")
            return {"error": str(e)}

        # Fetch data concurrently or sequentially
        # For simplicity, sequential fetching:
        stars_raw_data = self.github_api.get_stargazers(owner, repo, get_timestamps=True, days_limit=0) # Get all for StarAnalyzer
        logger.info(f"Fetched {len(stars_raw_data)} stargazers total.")

        contributors_data = self.github_api.get_contributors(owner, repo)
        logger.info(f"Fetched {len(contributors_data)} contributors.")
        
        # Fetch recent commits (last 90 days)
        since_90d_iso = (make_naive_datetime(datetime.datetime.now()) - datetime.timedelta(days=90)).isoformat()
        commits_recent_data = self.github_api.get_commits(owner, repo, since=since_90d_iso)
        logger.info(f"Fetched {len(commits_recent_data)} commits from last 90 days.")

        dependencies_raw_data = self.github_api.get_dependencies(owner, repo)
        logger.info(f"Fetched raw dependency data (Source: {'API' if 'sbom' in dependencies_raw_data else 'File Parsing'}).")

        license_api_data = self.github_api.get_license(owner, repo)
        logger.info(f"Fetched license info: {license_api_data.get('license',{}).get('spdx_id', 'N/A')}")

        # Initialize Analyzers
        # BurstDetector for fake star analysis (central component for this)
        burst_detector_inst = BurstDetector(owner, repo, self.github_api)
        
        # StarAnalyzer uses raw star data and its own BurstDetector instance, or can share one
        star_analyzer_inst = StarAnalyzer(owner, repo, stars_raw_data, repo_data["created_at"], self.github_api)
        # If StarAnalyzer should use the main burst_detector_inst:
        # star_analyzer_inst.burst_detector = burst_detector_inst # Share instance

        dependency_analyzer_inst = DependencyAnalyzer(dependencies_raw_data, self.github_api)
        # LicenseAnalyzer needs analyzed dependency list if it were to check compatibility
        # For now, it doesn't use it deeply, so pass empty or basic dep list.
        # Let's pass the flat_dependencies from dependency_analyzer_inst.
        license_analyzer_inst = LicenseAnalyzer(license_api_data, dependency_analyzer_inst.flat_dependencies)
        
        maintainer_analyzer_inst = MaintainerAnalyzer(contributors_data, commits_recent_data, self.github_api)
        code_analyzer_inst = CodeAnalyzer(owner, repo, self.github_api)

        # Perform Analyses
        logger.info("Running star pattern analysis...")
        star_analysis_results = star_analyzer_inst.detect_anomalies()
        
        fake_star_analysis_results = None
        if analyze_fake_stars:
            logger.info("Running fake star detection (BurstDetector)...")
            # Use star_analyzer_inst's burst_detector as it has timeseries possibly
            fake_star_analysis_results = star_analyzer_inst.detect_fake_stars() 
            fsi_val = fake_star_analysis_results.get('fake_star_index',0.0)
            fsi_risk = fake_star_analysis_results.get('risk_level','low')
            logger.info(f"Fake Star Index: {fsi_val:.2f} ({fsi_risk.upper()} RISK)")
        else: # Create a default structure if skipped
             fake_star_analysis_results = {
                "has_fake_stars": False, "fake_star_index": 0.0, "risk_level": "low", "bursts": [],
                "total_stars_analyzed": 0, "total_likely_fake": 0, "fake_percentage": 0.0, "worst_burst": None,
                "message": "Fake star analysis skipped by user."
            }


        logger.info("Running dependency analysis...")
        dependency_analysis_results = dependency_analyzer_inst.analyze()
        
        logger.info("Running license analysis...")
        license_analysis_results = license_analyzer_inst.analyze()
        
        logger.info("Running maintainer analysis...")
        maintainer_analysis_results = maintainer_analyzer_inst.analyze()
        
        logger.info("Running suspicious code pattern check...")
        code_analysis_results = code_analyzer_inst.check_for_suspicious_patterns()


        # Calculate Trust Score
        trust_calculator_inst = TrustScoreCalculator(
            star_analysis_results, dependency_analysis_results,
            license_analysis_results, maintainer_analysis_results,
            fake_star_analysis_results # Pass the FSI results here
        )
        trust_score_final = trust_calculator_inst.calculate()
        badge_url_str = trust_calculator_inst.generate_badge_url(owner, repo) # Uses the score from calculate()

        # Additional informational outputs
        activity_info_detailed = maintainer_analyzer_inst.check_recent_activity()
        package_registry_info = dependency_analyzer_inst.check_in_package_registries()

        # Prepare final result dictionary
        final_result_dict = {
            "repository_info": { # Renamed from "repository" to avoid clash with analysis result key "repo"
                "name": repo_data.get("name"), "full_name": repo_data.get("full_name"),
                "description": repo_data.get("description", ""), "stars_count": repo_data.get("stargazers_count"),
                "created_at": repo_data.get("created_at"), "html_url": repo_data.get("html_url"),
                "language": repo_data.get("language", "N/A")
            },
            "trust_score_summary": trust_score_final,
            "star_pattern_analysis": star_analysis_results,
            "fake_star_detection_analysis": fake_star_analysis_results, # FSI specific results
            "dependency_health_analysis": dependency_analysis_results,
            "license_compliance_analysis": license_analysis_results,
            "maintainer_activity_analysis": maintainer_analysis_results, # Contains score
            "code_suspicion_analysis": code_analysis_results,
            "detailed_activity_metrics": activity_info_detailed, # Extra info
            "package_registry_check": package_registry_info,   # Extra info
            "generated_badge": {
                "url": badge_url_str,
                "markdown": f"[![StarGuard Score]({badge_url_str})](https://starguard.example.com/report/{owner}/{repo})" # Example link
            }
        }
        logger.info(f"Analysis for {owner}/{repo} complete. Trust Score: {trust_score_final['total_score']}")
        return final_result_dict

    def generate_report(self, analysis_result: Dict, format_str: str = "text") -> str:
        """Generate a formatted report from analysis results."""
        if "error" in analysis_result: # Top-level error from analyze_repo
            return f"Error: {analysis_result['error']}"
        
        if format_str == "json":
            # Custom default handler for datetime objects if any slip through
            def dt_handler(o):
                if isinstance(o, (datetime.datetime, datetime.date)): return o.isoformat()
                raise TypeError (f"Type {type(o)} not serializable")
            return json.dumps(analysis_result, indent=2, default=dt_handler)

        # Text & Markdown reports
        repo_info = analysis_result.get("repository_info", {})
        trust_summary = analysis_result.get("trust_score_summary", {})
        fake_star_info = analysis_result.get("fake_star_detection_analysis", {})
        code_sus_info = analysis_result.get("code_suspicion_analysis", {})

        md_report_lines = []
        text_report_lines = []

        # --- Header ---
        title = f"StarGuard Analysis: {repo_info.get('full_name', 'N/A')}"
        md_report_lines.extend([f"# {title}", ""])
        text_report_lines.extend([title, "=" * len(title), ""])

        # --- Overview ---
        overview_md = [
            "##  Overview Section",
            f"- **Repository**: [{repo_info.get('full_name','N/A')}]({repo_info.get('html_url','')})",
            f"- **Description**: {repo_info.get('description','N/A')}",
            f"- **Created**: {repo_info.get('created_at','N/A')}",
            f"- **Stars**: {repo_info.get('stars_count','N/A')}",
            f"- **Primary Language**: {repo_info.get('language','N/A')}", ""
        ]
        overview_text = [
            "Overview:",
            f"  Repository: {repo_info.get('full_name','N/A')} ({repo_info.get('html_url','')})",
            f"  Description: {repo_info.get('description','N/A')}",
            f"  Created: {repo_info.get('created_at','N/A')}",
            f"  Stars: {repo_info.get('stars_count','N/A')}",
            f"  Primary Language: {repo_info.get('language','N/A')}", ""
        ]
        md_report_lines.extend(overview_md)
        text_report_lines.extend(overview_text)
        
        # --- Trust Score ---
        trust_score_val = trust_summary.get('total_score', 'N/A')
        trust_risk_lvl = trust_summary.get('risk_level', 'N/A').upper()
        md_report_lines.extend([f"## Trust Score: {trust_score_val}/100 ({trust_risk_lvl} RISK)", ""])
        text_report_lines.extend([f"TRUST SCORE: {trust_score_val}/100 ({trust_risk_lvl} RISK)", ""])

        # Fake Star Penalty in Trust Score Breakdown
        penalty = trust_summary.get('fake_star_penalty_applied',0)
        if penalty > 0:
            md_report_lines.append(f"**Fake Star Penalty Applied**: -{penalty} points from base score.")
            text_report_lines.append(f"  Fake Star Penalty Applied: -{penalty} points from base score.")
        md_report_lines.append("")
        text_report_lines.append("")

        # --- Fake Star Detection ---
        if fake_star_info and fake_star_info.get("has_fake_stars"):
            fsi = fake_star_info.get('fake_star_index',0.0)
            fsi_risk = fake_star_info.get('risk_level','N/A').upper()
            fsi_likely_fake = fake_star_info.get('total_likely_fake',0)
            fsi_perc = fake_star_info.get('fake_percentage',0.0)
            fsi_burst_count = len(fake_star_info.get('bursts',[]))

            alert_emoji = "🚨" if fsi_risk == "HIGH" else "⚠️"
            md_report_lines.extend([f"## {alert_emoji} Fake Star Detection {alert_emoji}", ""])
            text_report_lines.extend([f"!!! {alert_emoji} Fake Star Detection {alert_emoji} !!!", "-"*30, ""])
            
            md_report_lines.extend([
                f"**Fake Star Index**: {fsi:.2f} ({fsi_risk} RISK)",
                f"**Likely Fake Stars**: {fsi_likely_fake} ({fsi_perc:.1f}% of those analyzed in bursts)",
                f"**Suspicious Bursts Detected**: {fsi_burst_count}", ""
            ])
            text_report_lines.extend([
                f"  Fake Star Index: {fsi:.2f} ({fsi_risk} RISK)",
                f"  Likely Fake Stars: {fsi_likely_fake} ({fsi_perc:.1f}% of those analyzed in bursts)",
                f"  Suspicious Bursts Detected: {fsi_burst_count}", ""
            ])

            if fake_star_info.get('bursts'):
                md_report_lines.append("### Suspicious Star Burst Details:")
                text_report_lines.append("  Suspicious Star Burst Details:")
                for idx, burst_item in enumerate(fake_star_info['bursts'][:3]): # Show top 3
                    b_verdict = burst_item.get('verdict','N/A').upper()
                    b_score = burst_item.get('burst_score',0.0)
                    b_start = burst_item.get('start_date','N/A')
                    b_end = burst_item.get('end_date','N/A')
                    b_stars = burst_item.get('stars',0)
                    b_fake_ratio = burst_item.get('fake_ratio_in_burst',0.0) * 100

                    md_report_lines.append(f"- **Burst {idx+1}**: {b_verdict} (Score: {b_score:.2f}), Period: {b_start} to {b_end}, Stars: +{b_stars},  Estimated Fake Ratio: {b_fake_ratio:.1f}%")
                    text_report_lines.append(f"    Burst {idx+1}: {b_verdict} (Score: {b_score:.2f})")
                    text_report_lines.append(f"      Period: {b_start} a {b_end}, Stars: +{b_stars}, Estimated Fake Ratio: {b_fake_ratio:.1f}%")
                if len(fake_star_info['bursts']) > 3:
                    md_report_lines.append("- ...and other bursts.")
                    text_report_lines.append("    ...and other bursts.")
                md_report_lines.append("")
                text_report_lines.append("")
        elif "message" in fake_star_info : # E.g. analysis skipped
             md_report_lines.extend(["## Fake Star Detection", f"_{fake_star_info['message']}_", ""])
             text_report_lines.extend(["Fake Star Detection:", f"  {fake_star_info['message']}", ""])


        # --- Code Suspicion ---
        if code_sus_info and code_sus_info.get("is_potentially_suspicious"):
            cs_score = code_sus_info.get('calculated_suspicion_score',0)
            md_report_lines.extend([f"## ⚠️ Suspicious Code Detection", ""])
            text_report_lines.extend(["!!! ⚠️ SUSPICIOUS CODE DETECTION !!!", "-"*30, ""])
            
            md_report_lines.extend([f"**Code Suspicion Score**: {cs_score}/100", ""])
            text_report_lines.extend([f"  Code Suspicion Score: {cs_score}/100", ""])

            if code_sus_info.get("findings_by_category"):
                md_report_lines.append("Suspicious patterns found:")
                text_report_lines.append("  Suspicious patterns found:")
                for cat, finds in code_sus_info["findings_by_category"].items():
                    if finds:
                        md_report_lines.append(f"- **{cat.replace('_',' ').title()}**: {len(finds)} istanze")
                        text_report_lines.append(f"    - {cat.replace('_',' ').title()}: {len(finds)} istanze")
            if code_sus_info.get("suspicious_package_elements"):
                 md_report_lines.append("Suspicious elements in package manifest:")
                 text_report_lines.append("  Suspicious elements in package manifest:")
                 for elem_desc in code_sus_info["suspicious_package_elements"][:3]:
                     md_report_lines.append(f"- {elem_desc}")
                     text_report_lines.append(f"    - {elem_desc}")
            md_report_lines.append("")
            text_report_lines.append("")

        # --- Individual Analysis Sections (Simplified) ---
        sections = {
            "Star Pattern Analysis": analysis_result.get("star_pattern_analysis"),
            "Dependency Health": analysis_result.get("dependency_health_analysis"),
            "License Compliance": analysis_result.get("license_compliance_analysis"),
            "Maintainer Activity": analysis_result.get("maintainer_activity_analysis")
        }
        score_comp_map = trust_summary.get("score_components", {})
        score_key_map = { # Map section title to score component key
            "Star Pattern Analysis": "star_pattern_score",
            "Dependency Health": "dependencies_score",
            "License Compliance": "license_score",
            "Maintainer Activity": "maintainer_activity_score"
        }

        for section_title, data in sections.items():
            if not data: continue
            score_val = score_comp_map.get(score_key_map.get(section_title), "N/A")
            max_score = 50 if "Sttars" in section_title else \
                        30 if "Dependancy" in section_title else \
                        20 # Licenza e Manutentori
            
            md_report_lines.extend([f"## {section_title}", f"**Component Score**: {score_val}/{max_score}", ""])
            text_report_lines.extend([f"{section_title.upper()}:", f"  Component Score: {score_val}/{max_score}", ""])
            
            # Add 1-2 key details from each section
            if section_title == "Star Pattern Analysis" and data.get("anomalies"):
                num_anom = len(data["anomalies"])
                md_report_lines.append(f"- Detected {num_anom} anomalies in star pattern.")
                text_report_lines.append(f"  - Detected {num_anom} anomalies in star pattern.")
            elif section_title == "Dependency Health" and data.get("stats"):
                stats = data["stats"]
                md_report_lines.append(f"- Analyzed {stats.get('total_analyzed',0)} dependencies: {stats.get('high_risk',0)} high risk, {stats.get('medium_risk',0)} medium risk.")
                text_report_lines.append(f"  - Analyzed {stats.get('total_analyzed',0)} dependencies: {stats.get('high_risk',0)} high risk, {stats.get('medium_risk',0)} medium risk.")
            elif section_title == "License Compliance":
                lic_key = data.get('repo_license_key','N/A')
                lic_risk = data.get('repo_license_risk','N/A').upper()
                md_report_lines.append(f"- Repository License: `{lic_key}` (Risk: {lic_risk}).")
                text_report_lines.append(f"  - Repository License: {lic_key} (Risk: {lic_risk}).")
            elif section_title == "Maintainer activity" and data.get("recent_activity_summary"):
                act_sum = data["recent_activity_summary"]
                md_report_lines.append(f"- {act_sum.get('commits_last_90d',0)} commit negli ultimi 90 giorni; {act_sum.get('active_maintainers_heuristic',0)} active maintainers (among top contributors).")
                text_report_lines.append(f"  - {act_sum.get('commits_last_90d',0)} commit negli ultimi 90 giorni; {act_sum.get('active_maintainers_heuristic',0)} active maintainers (among top contributors).")

            md_report_lines.append("")
            text_report_lines.append("")


        # --- Badge ---
        badge_info = analysis_result.get("generated_badge", {})
        if badge_info.get("url"):
            md_report_lines.extend(["## Badge", "", badge_info["markdown"], ""])
            text_report_lines.extend(["BADGE", "", badge_info["markdown"], ""])

        if format_str == "markdown":
            return "\n".join(md_report_lines)
        else: # Default to text
            return "\n".join(text_report_lines)

