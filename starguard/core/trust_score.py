"""Calculates overall trust score from individual analysis components."""

from typing import Dict, Optional
from urllib.parse import quote_plus


class TrustScoreCalculator:
    """
    Calculates overall trust score from individual analysis components.
    """

    def __init__(
        self,
        star_analysis_res: Dict,
        dependency_analysis_res: Dict,
        license_analysis_res: Dict,
        maintainer_analysis_res: Dict,
        fake_star_analysis_res: Optional[Dict] = None,  # This is from BurstDetector
    ):
        self.star_analysis = star_analysis_res
        self.dependency_analysis = dependency_analysis_res
        self.license_analysis = license_analysis_res
        self.maintainer_analysis = maintainer_analysis_res
        self.fake_star_analysis = fake_star_analysis_res

    def calculate(self) -> Dict:
        """Calculate the overall trust score."""
        # Scores from components (higher is better for these)
        # Star pattern: 0-40  (was 50)
        # Dependencies: 0-25  (was 30)
        # License: 0-20  (unchanged)
        # Maintainers: 0-15  (was 20)
        # Total ideal sum = 40 + 25 + 20 + 15 = 100

        star_score_raw = self.star_analysis.get("score", 20)  # Default to neutral if missing
        # Scale from 0-50 to 0-40
        star_score = star_score_raw * 0.8 if self.star_analysis else 20
        
        dep_score_raw = self.dependency_analysis.get("score", 12.5)  # Default to neutral if missing
        # Scale from 0-30 to 0-25
        dep_score = dep_score_raw * (25/30) if self.dependency_analysis else 12.5
        
        lic_score_raw = self.license_analysis.get("score", 10)  # Default to neutral if missing
        # Already 0-20, no scaling needed
        lic_score = lic_score_raw
        
        maint_score_raw = self.maintainer_analysis.get("score", 7.5)  # Default to neutral if missing
        # Scale from 0-20 to 0-15
        maint_score = maint_score_raw * 0.75 if self.maintainer_analysis else 7.5

        # Sum of scaled scores. Max possible is 100.
        base_total_score = star_score + dep_score + lic_score + maint_score

        penalty_for_fake_stars = 0
        fake_star_penalty_breakdown = {}
        
        if self.fake_star_analysis and self.fake_star_analysis.get("has_fake_stars", False):
            # fake_star_index is 0 (good) to 1 (bad)
            fsi = self.fake_star_analysis.get("fake_star_index", 0.0)
            # Base penalty: up to 40 points based on FSI.
            base_fsi_penalty = int(fsi * 40)
            penalty_for_fake_stars += base_fsi_penalty
            fake_star_penalty_breakdown["base_fsi_penalty"] = base_fsi_penalty

            # Risk level penalty
            risk_level_penalty = 0
            if self.fake_star_analysis.get("risk_level") == "high":
                risk_level_penalty = 15
            elif self.fake_star_analysis.get("risk_level") == "medium":
                risk_level_penalty = 8
            
            penalty_for_fake_stars += risk_level_penalty
            fake_star_penalty_breakdown["risk_level_penalty"] = risk_level_penalty

            # Time-of-day entropy penalty
            tod_entropy_penalty = 0
            tod_entropy = self.fake_star_analysis.get("tod_entropy", float('inf'))
            if tod_entropy < 1.5:
                tod_entropy_penalty = 10
                penalty_for_fake_stars += tod_entropy_penalty
            fake_star_penalty_breakdown["tod_entropy_penalty"] = tod_entropy_penalty
            
            # Contribution Gini coefficient penalty
            gini_penalty = 0
            gini = self.fake_star_analysis.get("contribution_gini", 0.0)
            if gini > 0.6:
                gini_penalty = 8
                penalty_for_fake_stars += gini_penalty
            fake_star_penalty_breakdown["gini_penalty"] = gini_penalty
            
            # Lock-step behavior penalty
            lockstep_penalty = 0
            if self.fake_star_analysis.get("clusters", []) and len(self.fake_star_analysis.get("clusters", [])) > 0:
                lockstep_penalty = 10
                penalty_for_fake_stars += lockstep_penalty
            fake_star_penalty_breakdown["lockstep_penalty"] = lockstep_penalty
            
            # Short burst duration penalty
            short_burst_penalty = 0
            temporal_analysis = self.fake_star_analysis.get("temporal_analysis", {})
            if temporal_analysis.get("avg_burst_duration", 0) <= 2 and temporal_analysis.get("avg_burst_duration", 0) > 0:
                short_burst_penalty = 8
                penalty_for_fake_stars += short_burst_penalty
            fake_star_penalty_breakdown["short_burst_penalty"] = short_burst_penalty
            
            # Cap the total penalty
            penalty_for_fake_stars = min(60, penalty_for_fake_stars)
            
            # Add fallback mode info if available
            if self.fake_star_analysis.get("fallback_mode", False):
                fake_star_penalty_breakdown["fallback_mode"] = True

        # Total score after penalty, capped 0-100
        final_score_val = base_total_score - penalty_for_fake_stars
        final_score_val = max(0, min(100, final_score_val))  # Cap at 0 and 100

        risk_lvl_str = "high"  # Default for low scores
        if final_score_val >= 80:
            risk_lvl_str = "low"
        elif final_score_val >= 60:
            risk_lvl_str = "medium"

        return {
            "total_score": int(round(final_score_val)),
            "risk_level": risk_lvl_str,
            "score_components": {
                "star_pattern_score": star_score,   # 0-40
                "dependencies_score": dep_score,    # 0-25
                "license_score": lic_score,         # 0-20
                "maintainer_activity_score": maint_score,  # 0-15
            },
            "fake_star_penalty_applied": penalty_for_fake_stars,
            "fake_star_penalty_breakdown": fake_star_penalty_breakdown,
            "fallback_mode_used": self.fake_star_analysis.get("fallback_mode", False) if self.fake_star_analysis else False
        }

    def generate_badge_url(
        self, owner: str, repo: str
    ) -> str:
        """Generate a badge URL for the repository."""
        score_data = self.calculate()
        calculated_total_score = score_data["total_score"]

        # Determine badge color based on score
        color_str = "red"  # Default for high risk / low score
        if calculated_total_score >= 80:
            color_str = "success"  # Green
        elif calculated_total_score >= 60:
            color_str = "yellow"  # Yellow

        # URL encode label and message
        label_enc = quote_plus("StarGuard Score")
        message_enc = quote_plus(f"{calculated_total_score}/100")

        # Shield.io URL
        badge_url = f"https://img.shields.io/badge/{label_enc}-{message_enc}-{color_str}.svg?style=flat-square&logo=github"
        return badge_url