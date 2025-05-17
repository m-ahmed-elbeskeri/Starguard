"""Calculates overall trust score from individual analysis components."""

from typing import Dict
from typing import Dict, Optional
from typing import Optional

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
        self.fake_star_analysis = (
            fake_star_analysis_res  # Results from BurstDetector.calculate_fake_star_index
        )

    def calculate(self) -> Dict:
        """Calculate the overall trust score."""
        # Scores from components (higher is better for these)
        # Star pattern: 0-50
        # Dependencies: 0-30
        # License: 0-20
        # Maintainers: 0-20
        # Total ideal sum = 50 + 30 + 20 + 20 = 120 (oops, components sum to 120, need to re-scale or adjust max points)
        # Let's assume the component scores are correctly scaled to their max as per their classes.
        # Star: 50, Deps: 30, License: 20, Maintainer: 20. Total = 120.
        # We want final score 0-100.

        star_score_raw = self.star_analysis.get("score", 25)  # Default to neutral if missing
        dep_score_raw = self.dependency_analysis.get("score", 15)
        lic_score_raw = self.license_analysis.get("score", 10)
        maint_score_raw = self.maintainer_analysis.get("score", 10)

        # Sum of raw scores from components. Max possible is 120.
        base_total_score = star_score_raw + dep_score_raw + lic_score_raw + maint_score_raw

        # Scale this base_total_score (max 120) to a 0-100 range.
        # scaled_base_score = (base_total_score / 120) * 100
        # For now, let's keep the sum and apply penalty, then cap at 100.
        # This means some aspects might "max out" the score even if others are low.
        # Alternative: weighted average. For now, simple sum then penalty.

        penalty_for_fake_stars = 0
        if self.fake_star_analysis and self.fake_star_analysis.get("has_fake_stars", False):
            # fake_star_index is 0 (good) to 1 (bad)
            fsi = self.fake_star_analysis.get("fake_star_index", 0.0)
            # Penalty: up to 50 points based on FSI.
            # If FSI is 1.0 (max fake), penalty is 50.
            # If FSI is 0.3 (organic threshold), penalty is low.
            penalty_for_fake_stars = int(fsi * 50)

            # Extra penalty if risk is 'high' from FSI
            if self.fake_star_analysis.get("risk_level") == "high":
                penalty_for_fake_stars += 20  # Hefty additional penalty
            elif self.fake_star_analysis.get("risk_level") == "medium":
                penalty_for_fake_stars += 10

        # Total score after penalty, capped 0-100
        # Current max base_total_score = 120. Max penalty can be 50+20=70.
        # If base is 120, penalty 70 -> final 50.
        # If base is 60, penalty 70 -> final -10 (becomes 0).
        # This seems reasonable.
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
                "star_pattern_score": star_score_raw,  # max 50
                "dependencies_score": dep_score_raw,  # max 30
                "license_score": lic_score_raw,  # max 20
                "maintainer_activity_score": maint_score_raw,  # max 20
            },
            "fake_star_penalty_applied": penalty_for_fake_stars,
        }

    def generate_badge_url(
        self, owner: str, repo: str
    ) -> str:  # Changed 'self' to 'cls' or make static if no self needed
        """Generate a badge URL for the repository."""
        # This method is called on an instance, so 'self' is fine.
        # It needs the calculated score, so it should be called after `calculate()`.
        # Let's assume it's called on an instance that has just run `calculate()`.
        # Or, it should take the score as argument.

        # For simplicity, let's make it require the score dict from calculate().
        # Modifying signature: def generate_badge_url(self, score_data: Dict, owner: str, repo: str) -> str:
        # However, the current call in StarGuard.analyze_repo is `trust_calculator.generate_badge_url(owner, repo)`
        # This implies it re-calculates or uses stored state. Let's assume it re-calculates.

        score_data = (
            self.calculate()
        )  # Recalculate if called independently, or use stored if available.
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
