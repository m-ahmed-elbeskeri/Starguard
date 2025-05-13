"""Analyzes repository dependencies for security and maintenance risks."""

import logging
from typing import Dict, List
from collections import defaultdict
import re
import json

import networkx as nx

from starguard.api.github_api import GitHubAPI

logger = logging.getLogger(__name__)


class DependencyAnalyzer:
    """Analyzes repository dependencies for security and maintenance risks."""
    
    def __init__(self, dependencies: Dict, github_api: GitHubAPI):
        self.dependencies = dependencies # This is the raw dict from get_dependencies or _parse_dependencies
        self.github_api = github_api # Potentially used for deeper analysis later
        self.flat_dependencies = self._flatten_dependencies()
        
    def _flatten_dependencies(self) -> List[Dict]:
        flat_deps = []
        try:
            # Handle structure from GitHub Dependency Graph API (SBOM)
            if "sbom" in self.dependencies and isinstance(self.dependencies["sbom"], dict):
                sbom_data = self.dependencies["sbom"]
                if "packages" in sbom_data and isinstance(sbom_data["packages"], list):
                    for pkg_info in sbom_data["packages"]:
                        if isinstance(pkg_info, dict) and pkg_info.get("relationship") == "direct": # Process only direct dependencies
                            flat_deps.append({
                                "name": pkg_info.get("name", "unknown_dep"),
                                "version": pkg_info.get("versionInfo", "unknown_ver"),
                                "ecosystem": pkg_info.get("packageSupplier", {}).get("name", # Heuristic for ecosystem
                                                             pkg_info.get("externalRefs", [{}])[0].get("referenceCategory", "unknown_eco")),
                                "type": "runtime" # Default, SBOM might provide more
                            })
                return flat_deps # If SBOM format, assume it's complete and don't process other formats

            # Handle structure from manual file parsing (PACKAGE_MANAGERS)
            for lang_eco, dep_list in self.dependencies.items():
                if isinstance(dep_list, list):
                    for dep_item in dep_list:
                        if isinstance(dep_item, dict):
                            # Ensure basic keys exist
                            flat_deps.append({
                                "name": dep_item.get("name", f"unknown_{lang_eco}_dep"),
                                "version": dep_item.get("version", "unknown_ver"),
                                "ecosystem": lang_eco, # lang from PACKAGE_MANAGERS is the ecosystem
                                "type": dep_item.get("type", "runtime")
                            })
        except Exception as e:
            logger.error(f"Error flattening dependencies: {str(e)}", exc_info=True)
        return flat_deps

    def analyze(self) -> Dict:
        """Perform comprehensive analysis of dependencies."""
        if not self.flat_dependencies:
            return {"dependencies": [], "score": 25, "error": "No dependencies found/parsed."} # Neutral-low score

        analyzed_deps_list = []
        # Max 50 deps to analyze in detail to save time/resources, prioritize by some heuristic if needed
        deps_to_analyze = self.flat_dependencies[:50] if len(self.flat_dependencies) > 50 else self.flat_dependencies

        for dep_data in deps_to_analyze:
            single_analysis = self._analyze_single_dependency(dep_data)
            analyzed_deps_list.append(single_analysis)
        
        if not analyzed_deps_list: # Should not happen if flat_dependencies was not empty
            return {"dependencies": [], "score": 25, "error": "No dependencies were analyzed."}

        total_analyzed_count = len(analyzed_deps_list)
        high_risk_ct = sum(1 for d in analyzed_deps_list if d["risk_level"] == "high")
        medium_risk_ct = sum(1 for d in analyzed_deps_list if d["risk_level"] == "medium")
        
        # Score based on proportion of risky dependencies. Max score 30.
        # risk_score_penalty = (high_risk_ct * 2 + medium_risk_ct * 1) # Max penalty if all high_risk = total_analyzed_count*2
        # Max possible penalty is total_analyzed_count * 2 (if all are high risk)
        # Score = 30 * (1 - (penalty / (total_analyzed_count * 2) ))
        # Simplified: each high risk -2, med risk -1 from 30.
        score_val = 30 - (high_risk_ct * 2) - (medium_risk_ct * 1)
        score_val = max(0, score_val) # Ensure score is not negative

        return {
            "dependencies": analyzed_deps_list,
            "stats": {
                "total_found": len(self.flat_dependencies), # Total found before capping
                "total_analyzed": total_analyzed_count, # Number actually analyzed
                "high_risk": high_risk_ct,
                "medium_risk": medium_risk_ct,
                "low_risk": total_analyzed_count - high_risk_ct - medium_risk_ct
            },
            "score": score_val
        }

    def _analyze_single_dependency(self, dep: Dict) -> Dict:
        """Analyze a single dependency for various risk factors."""
        # Basic result structure
        analysis_result = {
            "name": dep.get("name", "N/A"),
            "version": dep.get("version", "N/A"),
            "ecosystem": dep.get("ecosystem", "N/A"),
            "type": dep.get("type", "N/A"),
            "risk_level": "low", # Default
            "risk_factors": []
        }

        version_str = str(dep.get("version", "")).lower()
        # 1. Unpinned version (simplified check)
        if version_str in ["latest", "*", "", "unknwon_ver"] or \
           any(c in version_str for c in ['^', '~', '>', '<']) and not any(op in version_str for op in ['==', '=']): # Loose check
            analysis_result["risk_factors"].append({
                "type": "version_specifier",
                "description": "Dependency version might not be pinned, potentially allowing auto-updates to risky versions."
            })
        
        # 2. Known vulnerable (placeholder - requires vulnerability DB)
        # if dep.get("name") == "known-vulnerable-package":
        #    analysis_result["risk_factors"].append({"type": "vulnerability", "description": "Known vulnerability CVE-XXXX-YYYY associated."})

        # 3. Package from non-standard source (e.g. git URL in version for npm)
        if dep.get("ecosystem") == "javascript" and isinstance(dep.get("version"), str) and \
           ("git#" in dep.get("version") or dep.get("version").startswith("file:")) :
            analysis_result["risk_factors"].append({
                "type": "source",
                "description": "Dependency sourced directly from Git or local file, bypassing standard registry vetting."
            })
        
        # Determine overall risk level based on number/severity of factors
        if len(analysis_result["risk_factors"]) >= 2: # Example: 2+ factors = high
            analysis_result["risk_level"] = "high"
        elif len(analysis_result["risk_factors"]) == 1:
            analysis_result["risk_level"] = "medium"
            
        return analysis_result
    
    def build_dependency_graph(self) -> nx.DiGraph: # Not used by main scoring, but available
        G = nx.DiGraph()
        for dep in self.flat_dependencies:
            G.add_node(dep["name"], **dep)
        # Edges would require transitive dependency info, not easily available from simple parsing.
        return G

    def check_in_package_registries(self) -> Dict: # Informational, not directly scored
        registry_presence = defaultdict(list)
        for dep in self.flat_dependencies:
            eco = str(dep.get("ecosystem", "")).lower()
            dep_name = dep.get("name")
            if not dep_name: continue

            # Map internal ecosystem names to common registry names if needed
            if "javascript" in eco or "npm" in eco: registry_presence["npm"].append(dep_name)
            elif "python" in eco or "pypi" in eco: registry_presence["pypi"].append(dep_name)
            elif "java" in eco or "maven" in eco: registry_presence["maven"].append(dep_name)
            elif "ruby" in eco or "gem" in eco: registry_presence["rubygems"].append(dep_name)
            elif "go" in eco : registry_presence["go_modules"].append(dep_name)
            else: registry_presence[eco if eco else "other"].append(dep_name)
        return dict(registry_presence)
