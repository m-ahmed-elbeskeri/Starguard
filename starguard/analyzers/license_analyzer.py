"""Analyzes repository licenses for compliance risks."""

import logging
from typing import Dict, List, Optional, Tuple

from starguard.core.constants import LICENSE_RISK_LEVELS, LICENSE_COMPATIBILITY

logger = logging.getLogger(__name__)


class LicenseAnalyzer:
    """
    Analyzes repository licenses for compliance risks.
    
    This class evaluates the repository's license risk level and checks for
    potential license incompatibilities with its dependencies.
    """
    
    def __init__(self, license_data: Dict, dependencies_analyzed: List[Dict]):
        """
        Initialize the license analyzer.
        
        Args:
            license_data: License information retrieved from GitHub API
            dependencies_analyzed: List of analyzed dependencies from DependencyAnalyzer
        """
        self.license_data = license_data 
        self.dependencies_analyzed = dependencies_analyzed

    def _get_normalized_license_key(self) -> Tuple[str, str]:
        """
        Extract and normalize the repository license key and SPDX ID.
        
        Returns:
            Tuple containing (license_key, license_spdx_id)
        """
        repo_license_key = "unknown"
        repo_license_spdx = "unknown"
        
        try:
            if isinstance(self.license_data, dict) and isinstance(self.license_data.get("license"), dict):
                license_info = self.license_data["license"]
                repo_license_key = str(license_info.get("key", "unknown")).lower()
                repo_license_spdx = str(license_info.get("spdx_id", "unknown"))
                
                # Normalize values
                if repo_license_spdx == "NOASSERTION":
                    repo_license_spdx = "unknown"
                
                # Prefer SPDX if key is 'other' or 'unknown'
                if repo_license_key == "other" and repo_license_spdx != "unknown":
                    repo_license_key = repo_license_spdx.lower()
                elif repo_license_key == "unknown" and repo_license_spdx != "unknown":
                    repo_license_key = repo_license_spdx.lower()
        except Exception as e:
            logger.warning(f"Error parsing license data: {e}")
            
        return repo_license_key, repo_license_spdx

    def _check_dependency_license_compatibility(self) -> List[Dict]:
        """
        Check for license compatibility issues between the repository and its dependencies.
        
        Returns:
            List of dependency license issues found
        """
        repo_license_key, _ = self._get_normalized_license_key()
        dependency_license_issues = []
        
        # Skip if no dependencies or unknown repo license
        if not self.dependencies_analyzed or repo_license_key == "unknown":
            return dependency_license_issues
        
        for dep in self.dependencies_analyzed:
            dep_name = dep.get("name", "unknown")
            # Check if this dependency has license info
            dep_license = dep.get("license", "unknown")
            
            if dep_license == "unknown":
                continue
                
            # Check compatibility using LICENSE_COMPATIBILITY constant
            # For example: LICENSE_COMPATIBILITY = {("mit", "gpl-3.0"): "incompatible"}
            compatibility_key = (repo_license_key, dep_license)
            reverse_key = (dep_license, repo_license_key)
            
            if (compatibility_key in LICENSE_COMPATIBILITY and 
                LICENSE_COMPATIBILITY[compatibility_key] == "incompatible"):
                dependency_license_issues.append({
                    "dependency": dep_name,
                    "dependency_license": dep_license,
                    "repo_license": repo_license_key,
                    "severity": "high",
                    "message": f"Incompatible license: {dep_license} dependency in {repo_license_key} project"
                })
            elif (reverse_key in LICENSE_COMPATIBILITY and 
                  LICENSE_COMPATIBILITY[reverse_key] == "incompatible"):
                dependency_license_issues.append({
                    "dependency": dep_name,
                    "dependency_license": dep_license,
                    "repo_license": repo_license_key,
                    "severity": "high",
                    "message": f"Incompatible license: {dep_license} dependency in {repo_license_key} project"
                })
            
            # Check for copyleft licenses that may be problematic
            if (repo_license_key in ["mit", "apache-2.0", "bsd-3-clause"] and 
                dep_license in ["gpl-3.0", "agpl-3.0"]):
                dependency_license_issues.append({
                    "dependency": dep_name,
                    "dependency_license": dep_license,
                    "repo_license": repo_license_key,
                    "severity": "medium",
                    "message": f"Copyleft license {dep_license} dependency in {repo_license_key} project may require special attention"
                })
                
        return dependency_license_issues

    def analyze(self) -> Dict:
        """
        Analyze licenses for risks and compliance issues.
        
        Returns:
            Dict containing license analysis results including risk level and score
        """
        # Get normalized license information
        repo_license_key, repo_license_spdx = self._get_normalized_license_key()
        
        # Determine risk level based on license type
        repo_risk_str = LICENSE_RISK_LEVELS.get(repo_license_key, "high")
        
        # Check for license compatibility issues with dependencies
        dependency_license_issues = self._check_dependency_license_compatibility()
        
        # Calculate score (0-20 scale, higher is better)
        # Base score depends on repo license risk level
        score_val = 5  # Default for high risk / unknown
        if repo_risk_str == "medium":
            score_val = 10
        elif repo_risk_str == "low":
            score_val = 20
        
        # Penalize for dependency license issues
        if dependency_license_issues:
            # -2 points per issue, up to a maximum of 10 points reduction
            penalty = min(10, len(dependency_license_issues) * 2)
            score_val = max(0, score_val - penalty)
        
        # Construct result dictionary
        result = {
            "repo_license_spdx": repo_license_spdx,
            "repo_license_key": repo_license_key,
            "repo_license_risk": repo_risk_str,
            "dependency_license_issues": dependency_license_issues,
            "dependency_license_issues_found": len(dependency_license_issues),
            "license_compliance_recommendations": self._generate_recommendations(repo_license_key, dependency_license_issues),
            "score": score_val
        }
        
        return result
    
    def _generate_recommendations(self, repo_license: str, issues: List[Dict]) -> List[str]:
        """
        Generate license compliance recommendations based on analysis.
        
        Args:
            repo_license: Repository license key
            issues: List of detected license issues
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        # Add default recommendation based on repo license
        if repo_license == "unknown":
            recommendations.append("Add a license file to the repository to clarify usage terms.")
        elif repo_license in ["gpl-3.0", "agpl-3.0"]:
            recommendations.append("Ensure all distributed code complies with the copyleft requirements.")
        
        # Add recommendations based on found issues
        if issues:
            recommendations.append("Review dependency licenses for compatibility with your project license.")
            
            # Check for specific issue patterns
            has_copyleft_issues = any(i["severity"] == "high" for i in issues)
            if has_copyleft_issues:
                recommendations.append("Consider isolating incompatible licensed dependencies or changing your project license.")
        
        return recommendations

    def get_license_details(self) -> Dict:
        """
        Get detailed information about the repository license.
        
        Returns:
            Dict containing detailed license information
        """
        repo_license_key, repo_license_spdx = self._get_normalized_license_key()
        
        # Additional license metadata could be provided here
        license_details = {
            "key": repo_license_key,
            "spdx_id": repo_license_spdx,
            "risk_level": LICENSE_RISK_LEVELS.get(repo_license_key, "high"),
            "description": self._get_license_description(repo_license_key),
            "is_osi_approved": self._is_osi_approved(repo_license_key),
        }
        
        return license_details
    
    def _get_license_description(self, license_key: str) -> str:
        """Return a brief description of the license."""
        descriptions = {
            "mit": "A permissive license that allows commercial use, modification, distribution, and private use.",
            "apache-2.0": "A permissive license that includes patent grants and specifies how to apply the license.",
            "gpl-3.0": "A copyleft license requiring that derivative works be distributed under the same license terms.",
            "agpl-3.0": "Similar to GPL-3.0 but also covers network use as a form of distribution.",
            "unknown": "No license or unrecognized license. All rights reserved by default."
        }
        return descriptions.get(license_key, "License details not available.")
    
    def _is_osi_approved(self, license_key: str) -> bool:
        """Check if the license is approved by the Open Source Initiative."""
        osi_approved = ["mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause", 
                        "gpl-2.0", "gpl-3.0", "lgpl-2.1", "lgpl-3.0", 
                        "mpl-2.0", "epl-2.0", "isc", "unlicense"]
        return license_key in osi_approved