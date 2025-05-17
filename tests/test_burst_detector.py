import json
import os
import sys

import argparse
import logging

from starguard.analyzers.burst_detector import BurstDetector
from starguard.api.github_api import GitHubAPI
from starguard.main import StarGuard
from urllib.parse import urlparse


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="StarGuard - GitHub Repository Analysis Tool with Advanced Fake Star Detection"
    )
    parser.add_argument("owner_repo", help="GitHub repository in format 'owner/repo' or full URL")
    parser.add_argument(
        "-t", "--token", help="GitHub personal access token (or set GITHUB_TOKEN env var)"
    )
    parser.add_argument(
        "-f", "--format", choices=["text", "json", "markdown"], default="text", help="Output format"
    )
    parser.add_argument("-o", "--output", help="Output file (default: stdout)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose DEBUG logging")
    parser.add_argument(
        "--plot", help="Save star history plot to specified file path (e.g., plot.png)"
    )
    parser.add_argument(
        "--no-fake-stars",
        action="store_true",
        help="Skip fake star detection component (faster, less comprehensive)",
    )
    parser.add_argument(
        "--burst-only",
        action="store_true",
        help="Only run fake star burst detection and basic report (fastest)",
    )

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logging.getLogger("urllib3").setLevel(logging.INFO)  # Quieten noisy library if needed

    owner_str, repo_str = "", ""
    if args.owner_repo.startswith(("http://", "https://")):
        try:
            parsed_url = urlparse(args.owner_repo)
            path_parts = parsed_url.path.strip("/").split("/")
            if len(path_parts) >= 2 and parsed_url.netloc.lower() == "github.com":
                owner_str, repo_str = path_parts[0], path_parts[1]
                if repo_str.endswith(".git"):
                    repo_str = repo_str[:-4]  # Remove .git suffix
            else:
                raise ValueError("Invalid GitHub URL structure.")
        except ValueError as e:
            logger.error(f"Invalid GitHub URL format: {e}. Expected: https://github.com/owner/repo")
            sys.exit(1)
    else:
        try:
            owner_str, repo_str = args.owner_repo.split("/")
        except ValueError:
            logger.error("Invalid repository format. Use 'owner/repo' or a full GitHub URL.")
            sys.exit(1)

    github_token = args.token or os.environ.get("GITHUB_TOKEN")
    if not github_token:
        logger.warning(
            "No GitHub token provided via --token or GITHUB_TOKEN env var. API rate limits will be significantly lower."
        )

    try:
        final_report_str = ""
        if args.burst_only:
            logger.info(f"Running burst detection ONLY for {owner_str}/{repo_str}")
            github_api_inst = GitHubAPI(github_token)
            burst_detector_inst = BurstDetector(owner_str, repo_str, github_api_inst)
            # calculate_fake_star_index now returns a dict with expected keys
            burst_result_dict = burst_detector_inst.calculate_fake_star_index()

            if args.format == "json":
                final_report_str = json.dumps(burst_result_dict, indent=2, default=str)
            else:  # Text/Markdown for burst_only is simplified
                lines = [
                    f"StarGuard Burst-Only Detection for {owner_str}/{repo_str}",
                    "=" * (35 + len(owner_str) + len(repo_str)),
                    "",
                    f"Fake Star Index: {burst_result_dict.get('fake_star_index', 0.0):.2f} ({burst_result_dict.get('risk_level', 'N/A').upper()} RISK)",
                    f"Detected {len(burst_result_dict.get('bursts',[]))} suspicious bursts.",
                    f"Total Likely Fake Stars in Bursts: {burst_result_dict.get('total_likely_fake', 0)} ({burst_result_dict.get('fake_percentage', 0.0):.1f}%)",
                    "",
                ]
                if burst_result_dict.get("bursts"):
                    lines.append("Top Suspicious Bursts (max 3 shown):")
                    for idx, burst_item in enumerate(burst_result_dict["bursts"][:3]):
                        lines.append(
                            f"  Burst {idx+1}: {burst_item.get('verdict','N/A').upper()} (Score: {burst_item.get('burst_score',0.0):.2f}), "
                            f"{burst_item.get('start_date','N/A')} to {burst_item.get('end_date','N/A')}, "
                            f"+{burst_item.get('stars',0)} stars"
                        )
                final_report_str = "\n".join(lines)

            if args.plot:
                # Ensure burst_detector_inst has its data populated for plotting
                if burst_detector_inst.stars_df is None:
                    burst_detector_inst.build_daily_timeseries()
                if not burst_detector_inst.bursts:
                    burst_detector_inst.detect_bursts()  # Needed if calculate_fake_star_index wasn't run or failed early
                burst_detector_inst.plot_star_history(args.plot)

        else:  # Full analysis
            starguard_engine = StarGuard(github_token)
            analysis_results_dict = starguard_engine.analyze_repo(
                owner_str, repo_str, analyze_fake_stars=not args.no_fake_stars
            )

            if "error" in analysis_results_dict:  # Handle error from analyze_repo itself
                logger.error(f"Analysis failed: {analysis_results_dict['error']}")
                sys.exit(1)

            final_report_str = starguard_engine.generate_report(
                analysis_results_dict, format_str=args.format
            )

            if args.plot:
                # For full analysis, plot needs access to the StarAnalyzer's BurstDetector instance or similar data.
                # The StarGuard.analyze_repo would need to return the relevant analyzer instance or data for plotting.
                # This is a bit complex to pass around. For simplicity, instantiate a new one for plot if needed.
                # Or, could modify StarGuard.analyze_repo to return the StarAnalyzer instance.
                # Quick solution: recreate for plot.
                logger.info(f"Generating plot for {owner_str}/{repo_str}...")
                plot_api_inst = GitHubAPI(github_token)  # New API instance for plot to be safe

                # Use BurstDetector directly for plotting as it's the most comprehensive.
                # Data fetching for plot is separate from main analysis to ensure plot has what it needs.
                plot_burst_detector = BurstDetector(owner_str, repo_str, plot_api_inst)
                # Populate data needed for plot_star_history
                if plot_burst_detector.stars_df is None:
                    plot_burst_detector.build_daily_timeseries()
                # plot_star_history will call calculate_fake_star_index if bursts are not populated
                plot_burst_detector.plot_star_history(args.plot)

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(final_report_str)
            logger.info(f"Report saved to {args.output}")
        else:
            sys.stdout.write(final_report_str + "\n")  # Ensure newline at end

    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user.")
        sys.exit(130)  # Standard exit code for Ctrl+C
    except Exception as e:
        logger.error(f"An critical error occurred: {str(e)}", exc_info=args.verbose)
        if not args.verbose:
            logger.error("Run with -v or --verbose for detailed traceback.")
        sys.exit(1)

    def analyze_repo(self, owner: str, repo: str, analyze_fake_stars: bool = True) -> Dict:
        """Perform comprehensive analysis on a GitHub repository."""
        logger.info(f"Starting analysis of {owner}/{repo}")

        try:
            repo_data = self.github_api.get_repo(
                owner, repo
            )  # Raises ValueError if not found or API fails
            logger.info(
                f"Fetched repository data for: {repo_data.get('full_name', f'{owner}/{repo}')}"
            )
        except ValueError as e:  # Catch specific error from get_repo
            logger.error(f"Failed to fetch repository {owner}/{repo}: {e}")
            return {"error": str(e)}

        # Fetch data concurrently or sequentially
        # For simplicity, sequential fetching:
        stars_raw_data = self.github_api.get_stargazers(
            owner, repo, get_timestamps=True, days_limit=0
        )  # Get all for StarAnalyzer
        logger.info(f"Fetched {len(stars_raw_data)} stargazers total.")

        contributors_data = self.github_api.get_contributors(owner, repo)
        logger.info(f"Fetched {len(contributors_data)} contributors.")

        # Fetch recent commits (last 90 days)
        since_90d_iso = (
            make_naive_datetime(datetime.datetime.now()) - datetime.timedelta(days=90)
        ).isoformat()
        commits_recent_data = self.github_api.get_commits(owner, repo, since=since_90d_iso)
        logger.info(f"Fetched {len(commits_recent_data)} commits from last 90 days.")

        dependencies_raw_data = self.github_api.get_dependencies(owner, repo)
        logger.info(
            f"Fetched raw dependency data (Source: {'API' if 'sbom' in dependencies_raw_data else 'File Parsing'})."
        )

        license_api_data = self.github_api.get_license(owner, repo)
        logger.info(
            f"Fetched license info: {license_api_data.get('license',{}).get('spdx_id', 'N/A')}"
        )

        # Initialize Analyzers
        # BurstDetector for fake star analysis (central component for this)
        burst_detector_inst = BurstDetector(owner, repo, self.github_api)

        # StarAnalyzer uses raw star data and its own BurstDetector instance, or can share one
        star_analyzer_inst = StarAnalyzer(
            owner, repo, stars_raw_data, repo_data["created_at"], self.github_api
        )
        # If StarAnalyzer should use the main burst_detector_inst:
        # star_analyzer_inst.burst_detector = burst_detector_inst # Share instance

        dependency_analyzer_inst = DependencyAnalyzer(dependencies_raw_data, self.github_api)
        # LicenseAnalyzer needs analyzed dependency list if it were to check compatibility
        # For now, it doesn't use it deeply, so pass empty or basic dep list.
        # Let's pass the flat_dependencies from dependency_analyzer_inst.
        license_analyzer_inst = LicenseAnalyzer(
            license_api_data, dependency_analyzer_inst.flat_dependencies
        )

        maintainer_analyzer_inst = MaintainerAnalyzer(
            contributors_data, commits_recent_data, self.github_api
        )
        code_analyzer_inst = CodeAnalyzer(owner, repo, self.github_api)

        # Perform Analyses
        logger.info("Running star pattern analysis...")
        star_analysis_results = star_analyzer_inst.detect_anomalies()

        fake_star_analysis_results = None
        if analyze_fake_stars:
            logger.info("Running fake star detection (BurstDetector)...")
            # Use star_analyzer_inst's burst_detector as it has timeseries possibly
            fake_star_analysis_results = star_analyzer_inst.detect_fake_stars()
            fsi_val = fake_star_analysis_results.get("fake_star_index", 0.0)
            fsi_risk = fake_star_analysis_results.get("risk_level", "low")
            logger.info(f"Fake Star Index: {fsi_val:.2f} ({fsi_risk.upper()} RISK)")
        else:  # Create a default structure if skipped
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
        dependency_analysis_results = dependency_analyzer_inst.analyze()

        logger.info("Running license analysis...")
        license_analysis_results = license_analyzer_inst.analyze()

        logger.info("Running maintainer analysis...")
        maintainer_analysis_results = maintainer_analyzer_inst.analyze()

        logger.info("Running suspicious code pattern check...")
        code_analysis_results = code_analyzer_inst.check_for_suspicious_patterns()

        # Calculate Trust Score
        trust_calculator_inst = TrustScoreCalculator(
            star_analysis_results,
            dependency_analysis_results,
            license_analysis_results,
            maintainer_analysis_results,
            fake_star_analysis_results,  # Pass the FSI results here
        )
        trust_score_final = trust_calculator_inst.calculate()
        badge_url_str = trust_calculator_inst.generate_badge_url(
            owner, repo
        )  # Uses the score from calculate()

        # Additional informational outputs
        activity_info_detailed = maintainer_analyzer_inst.check_recent_activity()
        package_registry_info = dependency_analyzer_inst.check_in_package_registries()

        # Prepare final result dictionary
        final_result_dict = {
            "repository_info": {  # Renamed from "repository" to avoid clash with analysis result key "repo"
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
            "fake_star_detection_analysis": fake_star_analysis_results,  # FSI specific results
            "dependency_health_analysis": dependency_analysis_results,
            "license_compliance_analysis": license_analysis_results,
            "maintainer_activity_analysis": maintainer_analysis_results,  # Contains score
            "code_suspicion_analysis": code_analysis_results,
            "detailed_activity_metrics": activity_info_detailed,  # Extra info
            "package_registry_check": package_registry_info,  # Extra info
            "generated_badge": {
                "url": badge_url_str,
                "markdown": f"[![StarGuard Score]({badge_url_str})](https://starguard.example.com/report/{owner}/{repo})",  # Example link
            },
        }
        logger.info(
            f"Analysis for {owner}/{repo} complete. Trust Score: {trust_score_final['total_score']}"
        )
        return final_result_dict

    def generate_report(self, analysis_result: Dict, format_str: str = "text") -> str:
        """Generate a formatted report from analysis results."""
        if "error" in analysis_result:  # Top-level error from analyze_repo
            return f"Error: {analysis_result['error']}"

        if format_str == "json":
            # Custom default handler for datetime objects if any slip through
            def dt_handler(o):
                if isinstance(o, (datetime.datetime, datetime.date)):
                    return o.isoformat()
                raise TypeError(f"Type {type(o)} not serializable")

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

        # --- Trust Score ---
        trust_score_val = trust_summary.get("total_score", "N/A")
        trust_risk_lvl = trust_summary.get("risk_level", "N/A").upper()
        md_report_lines.extend(
            [f"## Trust Score: {trust_score_val}/100 ({trust_risk_lvl} RISK)", ""]
        )
        text_report_lines.extend(
            [f"TRUST SCORE: {trust_score_val}/100 ({trust_risk_lvl} RISK)", ""]
        )

        # Fake Star Penalty in Trust Score Breakdown
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

        # --- Fake Star Detection ---
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

            if fake_star_info.get("bursts"):
                md_report_lines.append("### Suspicious Star Burst Details:")
                text_report_lines.append("  Suspicious Star Burst Details:")
                for idx, burst_item in enumerate(fake_star_info["bursts"][:3]):  # Show top 3
                    b_verdict = burst_item.get("verdict", "N/A").upper()
                    b_score = burst_item.get("burst_score", 0.0)
                    b_start = burst_item.get("start_date", "N/A")
                    b_end = burst_item.get("end_date", "N/A")
                    b_stars = burst_item.get("stars", 0)
                    b_fake_ratio = burst_item.get("fake_ratio_in_burst", 0.0) * 100

                    md_report_lines.append(
                        f"- **Burst {idx+1}**: {b_verdict} (Score: {b_score:.2f}), Period: {b_start} to {b_end}, Stars: +{b_stars},  Estimated Fake Ratio: {b_fake_ratio:.1f}%"
                    )
                    text_report_lines.append(
                        f"    Burst {idx+1}: {b_verdict} (Score: {b_score:.2f})"
                    )
                    text_report_lines.append(
                        f"      Period: {b_start} a {b_end}, Stars: +{b_stars}, Estimated Fake Ratio: {b_fake_ratio:.1f}%"
                    )
                if len(fake_star_info["bursts"]) > 3:
                    md_report_lines.append("- ...and other bursts.")
                    text_report_lines.append("    ...and other bursts.")
                md_report_lines.append("")
                text_report_lines.append("")
        elif "message" in fake_star_info:  # E.g. analysis skipped
            md_report_lines.extend(["## Fake Star Detection", f"_{fake_star_info['message']}_", ""])
            text_report_lines.extend(["Fake Star Detection:", f"  {fake_star_info['message']}", ""])

        # --- Code Suspicion ---
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
                            f"- **{cat.replace('_',' ').title()}**: {len(finds)} istanze"
                        )
                        text_report_lines.append(
                            f"    - {cat.replace('_',' ').title()}: {len(finds)} istanze"
                        )
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
            "Maintainer Activity": analysis_result.get("maintainer_activity_analysis"),
        }
        score_comp_map = trust_summary.get("score_components", {})
        score_key_map = {  # Map section title to score component key
            "Star Pattern Analysis": "star_pattern_score",
            "Dependency Health": "dependencies_score",
            "License Compliance": "license_score",
            "Maintainer Activity": "maintainer_activity_score",
        }

        for section_title, data in sections.items():
            if not data:
                continue
            score_val = score_comp_map.get(score_key_map.get(section_title), "N/A")
            max_score = (
                50 if "Sttars" in section_title else 30 if "Dependancy" in section_title else 20
            )  # Licenza e Manutentori

            md_report_lines.extend(
                [f"## {section_title}", f"**Component Score**: {score_val}/{max_score}", ""]
            )
            text_report_lines.extend(
                [f"{section_title.upper()}:", f"  Component Score: {score_val}/{max_score}", ""]
            )

            # Add 1-2 key details from each section
            if section_title == "Star Pattern Analysis" and data.get("anomalies"):
                num_anom = len(data["anomalies"])
                md_report_lines.append(f"- Detected {num_anom} anomalies in star pattern.")
                text_report_lines.append(f"  - Detected {num_anom} anomalies in star pattern.")
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
            elif section_title == "Maintainer activity" and data.get("recent_activity_summary"):
                act_sum = data["recent_activity_summary"]
                md_report_lines.append(
                    f"- {act_sum.get('commits_last_90d',0)} commit negli ultimi 90 giorni; {act_sum.get('active_maintainers_heuristic',0)} active maintainers (among top contributors)."
                )
                text_report_lines.append(
                    f"  - {act_sum.get('commits_last_90d',0)} commit negli ultimi 90 giorni; {act_sum.get('active_maintainers_heuristic',0)} active maintainers (among top contributors)."
                )

            md_report_lines.append("")
            text_report_lines.append("")

        # --- Badge ---
        badge_info = analysis_result.get("generated_badge", {})
        if badge_info.get("url"):
            md_report_lines.extend(["## Badge", "", badge_info["markdown"], ""])
            text_report_lines.extend(["BADGE", "", badge_info["markdown"], ""])

        if format_str == "markdown":
            return "\n".join(md_report_lines)
        else:  # Default to text
            return "\n".join(text_report_lines)
