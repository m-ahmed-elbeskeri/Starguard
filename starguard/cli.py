"""Command-line interface for StarGuard."""

import os
import sys
import argparse
import json
import logging
from urllib.parse import urlparse

from starguard.main import StarGuard
from starguard.analyzers.burst_detector import BurstDetector
from starguard.api.github_api_imports import GitHubAPI

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("starguard")


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="StarGuard - GitHub Repository Analysis Tool with Advanced Fake Star Detection"
    )
    parser.add_argument("owner_repo", help="GitHub repository in format 'owner/repo' or full URL")
    parser.add_argument(
        "-t",
        "--token",
        help="GitHub personal access token (or set GITHUB_TOKEN env var)",
        action="append",
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
        logging.getLogger("urllib3").setLevel(logging.INFO)

    owner_str, repo_str = "", ""
    if args.owner_repo.startswith(("http://", "https://")):
        try:
            parsed_url = urlparse(args.owner_repo)
            path_parts = parsed_url.path.strip("/").split("/")
            if len(path_parts) >= 2 and parsed_url.netloc.lower() == "github.com":
                owner_str, repo_str = path_parts[0], path_parts[1]
                if repo_str.endswith(".git"):
                    repo_str = repo_str[:-4]
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

    github_tokens = []

    if args.token:
        github_tokens.extend(args.token)

    env_token = os.environ.get("GITHUB_TOKEN")
    if env_token and env_token not in github_tokens:
        github_tokens.append(env_token)

    if not github_tokens:
        logger.warning(
            "No GitHub tokens provided via --token or GITHUB_TOKEN env var. API rate limits will be significantly lower."
        )

    try:
        final_report_str = ""
        if args.burst_only:
            logger.info(f"Running burst detection ONLY for {owner_str}/{repo_str}")
            github_api_inst = GitHubAPI(github_tokens)
            burst_detector_inst = BurstDetector(owner_str, repo_str, github_api_inst)
            burst_result_dict = burst_detector_inst.calculate_fake_star_index()

            if args.format == "json":
                final_report_str = json.dumps(burst_result_dict, indent=2, default=str)
            else:
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
                if burst_detector_inst.stars_df is None:
                    burst_detector_inst.build_daily_timeseries()
                if not burst_detector_inst.bursts:
                    burst_detector_inst.detect_bursts()
                burst_detector_inst.plot_star_history(args.plot)

        else:
            starguard_engine = StarGuard(github_tokens)
            analysis_results_dict = starguard_engine.analyze_repo(
                owner_str, repo_str, analyze_fake_stars=not args.no_fake_stars
            )

            if "error" in analysis_results_dict:
                logger.error(f"Analysis failed: {analysis_results_dict['error']}")
                sys.exit(1)

            final_report_str = starguard_engine.generate_report(
                analysis_results_dict, format_str=args.format
            )

            if args.plot:
                logger.info(f"Generating plot for {owner_str}/{repo_str}...")
                plot_api_inst = GitHubAPI(github_tokens)

                plot_burst_detector = BurstDetector(owner_str, repo_str, plot_api_inst)
                if plot_burst_detector.stars_df is None:
                    plot_burst_detector.build_daily_timeseries()
                plot_burst_detector.plot_star_history(args.plot)

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(final_report_str)
            logger.info(f"Report saved to {args.output}")
        else:
            sys.stdout.write(final_report_str + "\n")

    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user.")
        sys.exit(130)
    except Exception as e:
        logger.error(f"An critical error occurred: {str(e)}", exc_info=args.verbose)
        if not args.verbose:
            logger.error("Run with -v or --verbose for detailed traceback.")
        sys.exit(1)


if __name__ == "__main__":
    main()
