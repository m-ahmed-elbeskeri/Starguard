"""Analyzes repository code for potential malware or suspicious patterns."""

import logging
import re
import json
from typing import Dict
from collections import defaultdict

from starguard.api.github_api import GitHubAPI

logger = logging.getLogger(__name__)


class CodeAnalyzer:
    """Analyzes repository code for potential malware or suspicious patterns."""
    
    def __init__(self, owner: str, repo: str, github_api: GitHubAPI):
        self.owner = owner
        self.repo = repo
        self.github_api = github_api
        
    def check_for_suspicious_patterns(self) -> Dict:
        """Check repository for suspicious code patterns."""
        # Define suspicious patterns (simplified)
        patterns_by_category = {
            "obfuscation": [r"eval\s*\(", r"fromCharCode", r"\\x[0-9a-f]{2}", r"document\.write\s*\(unescape\("],
            "remote_execution": [r"new\s+Function\s*\(", r"setTimeout\s*\(\s*[\"'].*eval\("],
            "data_exfiltration": [r"fetch\s*\(\s*[\"']https?:\/\/[^\"']+", r"navigator\.sendBeacon"],
            "crypto_jacking": [r"CryptoJS", r"miner", r"coinhive", r"cryptonight"],
            "suspicious_imports": [r"require\s*\(\s*[\"'](http|https|net|child_process)[\"']"],
        }

        # Files to check (prioritize common script/config files)
        files_to_scan = [
            "package.json", "Gruntfile.js", "gulpfile.js", # JS
            "setup.py", "__init__.py", # Python
            # Add more based on repo language if known, e.g. from repo_data["language"]
        ]
        # Heuristic: try to get a few .js or .py files from root if above are not found
        # This is complex, for now, stick to predefined list.

        found_indicators = defaultdict(list)
        total_hits = 0

        for file_path_str in files_to_scan:
            try:
                file_content_str = self.github_api.get_file_content(self.owner, self.repo, file_path_str)
                if not file_content_str: continue

                for category, regex_list in patterns_by_category.items():
                    for regex_pattern in regex_list:
                        try:
                            # Using re.finditer to get match objects for more context if needed
                            for match_obj in re.finditer(regex_pattern, file_content_str, re.IGNORECASE):
                                found_indicators[category].append({
                                    "file": file_path_str,
                                    "pattern": regex_pattern,
                                    "matched_text": match_obj.group(0)[:100], # First 100 chars of match
                                    # "line_number": file_content_str[:match_obj.start()].count('\n') + 1 # Can be slow
                                })
                                total_hits +=1
                        except re.error as re_e:
                             logger.debug(f"Regex error for pattern '{regex_pattern}': {re_e}")
            except Exception as e:
                logger.debug(f"Error scanning file {file_path_str} in {self.owner}/{self.repo}: {e}")
        
        # Analyze package.json for suspicious scripts or dependencies (if JS project)
        suspicious_deps_or_scripts = []
        if "package.json" in files_to_scan: # or if repo language is JavaScript
            pkg_json_content = self.github_api.get_file_content(self.owner, self.repo, "package.json")
            if pkg_json_content:
                try:
                    pkg_data = json.loads(pkg_json_content)
                    # Check scripts for obfuscated commands or downloads from weird URLs
                    for script_name, script_cmd in pkg_data.get("scripts", {}).items():
                        if isinstance(script_cmd, str) and ("curl" in script_cmd or "wget" in script_cmd or "node -e" in script_cmd):
                             if "npmjs.org" not in script_cmd and "github.com" not in script_cmd: # If not from known good sources
                                suspicious_deps_or_scripts.append(f"Suspicious script '{script_name}': {script_cmd[:100]}")
                                total_hits += 2 # Higher weight for suspicious scripts
                    # Check dependencies for typosquatting (very basic)
                    for dep_name in list(pkg_data.get("dependencies", {}).keys()) + list(pkg_data.get("devDependencies", {}).keys()):
                        if "rpel" in dep_name or "ajv-" in dep_name and dep_name != "ajv-keywords": # Example typos
                            suspicious_deps_or_scripts.append(f"Potentially typosquatted dependency: {dep_name}")
                            total_hits += 3
                except json.JSONDecodeError:
                    logger.debug(f"Could not parse package.json for code analysis in {self.owner}/{self.repo}")


        # Calculate suspicion score (0-100). Higher means more suspicious.
        # Each hit +5, max score from hits is 50.
        # Suspicious deps/scripts add more.
        suspicion_score_val = min(50, total_hits * 5) 
        suspicion_score_val += min(50, len(suspicious_deps_or_scripts) * 10) # Max 50 from this
        suspicion_score_val = min(100, suspicion_score_val) # Cap total at 100

        return {
            "findings_by_category": dict(found_indicators),
            "suspicious_package_elements": suspicious_deps_or_scripts,
            "total_suspicious_hits": total_hits,
            "calculated_suspicion_score": suspicion_score_val, # 0-100, higher is more suspicious
            "is_potentially_suspicious": suspicion_score_val > 40 # Threshold for flagging
        }

