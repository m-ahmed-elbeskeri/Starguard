"""Analyzes fork-to-star ratio to detect suspicious repositories."""


def fork_star_score(repo_json: dict) -> float:
    """
    Calculate trust score component based on fork-to-star ratio.

    Legitimate projects typically have fewer forks than stars.
    Suspicious patterns include forks ≈ stars or forks > stars.

    Args:
        repo_json: Repository data from GitHubAPI.get_repo()

    Returns:
        float: Score from 0.0 (suspicious) to 1.0 (normal ratio)
    """
    stars = max(1, repo_json.get("stargazers_count", 0))  # Avoid division by zero
    forks = repo_json.get("forks_count", 0)

    fsr = forks / stars

    if fsr >= 0.5:  # Very suspicious
        return 0.0
    elif fsr >= 0.25:  # Mild concern
        return 0.3
    else:  # Normal range
        return max(0.5, 1 - fsr * 2)  # Scales from 1.0 down to 0.5
