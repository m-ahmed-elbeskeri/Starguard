"""Constants used across the StarGuard package."""

# GitHub API constants
GITHUB_API_BASE = "https://api.github.com"
GITHUB_GRAPHQL_URL = "https://api.github.com/graphql"
LICENSE_RISK_LEVELS = {
    "mit": "low",
    "apache-2.0": "low",
    "bsd-2-clause": "low",
    "bsd-3-clause": "low",
    "isc": "low",
    "cc0-1.0": "low",
    "unlicense": "low",
    "gpl-2.0": "medium",
    "gpl-3.0": "medium",
    "lgpl-2.1": "medium",
    "lgpl-3.0": "medium",
    "agpl-3.0": "high",
    "cc-by-nc-sa-4.0": "high",
    "proprietary": "high",
    "unknown": "high"
}
PACKAGE_MANAGERS = {
    "javascript": {
        "file": "package.json",
        "dependencies_key": ["dependencies", "devDependencies"]
    },
    "python": {
        "file": "requirements.txt",
        "pattern": r"^([\w\-\._]+).*?(?:==|>=|<=|~=|!=|>|<)?\s*([\d\.\w\-]+)?.*$"
    },
    "python_pipfile": {
        "file": "Pipfile",
        "section_marker": "[packages]"
    },
    "python_poetry": {
        "file": "pyproject.toml",
        "section_marker": "[tool.poetry.dependencies]"
    },
    "ruby": {
        "file": "Gemfile",
        "pattern": r"gem\s+['\"]([^'\"]+)['\"](?:,\s*['\"](.+)['\"])?"
    },
    "java": {
        "file": "pom.xml",
        "tag_pattern": r"<dependency>\s*<groupId>([^<]+)</groupId>\s*<artifactId>([^<]+)</artifactId>\s*(?:<version>([^<]+)</version>)?"
    },
    "go": {
        "file": "go.mod",
        "pattern": r"^\s*require\s+([^\s]+)\s+v?([^\s]+)"
    }
}

# Constants for fake star detection
MAD_THRESHOLD = 3.0 * 1.48  # MAD threshold for spike detection
WINDOW_SIZE = 28  # Days for sliding window in MAD calculation
MIN_STAR_COUNT = 30  # Minimum star count before using MAD detection
MIN_STARS_GROWTH_PERCENT = 300  # Alternative % growth threshold for small repos

# Fake star user scoring weights
USER_SCORE_THRESHOLDS = {
    "account_age_days": (30, 2.0),  # (threshold, score if below threshold)
    "followers": (5, 1.0),
    "public_repos": (2, 1.0),
    "total_stars": (3, 1.0),
    "prior_interaction": (0, 1.0),  # 0 = no prior interaction
    "default_avatar": (True, 0.5)  # True = has default avatar
}
FAKE_USER_THRESHOLD = 4.0  # Score threshold to flag a user as likely fake

# Burst scoring weights
FAKE_RATIO_WEIGHT = 0.7
RULE_HITS_WEIGHT = 0.3

# Burst classification thresholds
BURST_ORGANIC_THRESHOLD = 0.3
BURST_FAKE_THRESHOLD = 0.6

# License compatibility matrix 
# 'compatible' - No license conflict
# 'conditional' - Compatible under certain conditions
# 'incompatible' - License conflict exists
LICENSE_COMPATIBILITY = {
    # Format: (license1, license2): compatibility
    #TODO: This is a simplified subset - a complete matrix would be much larger
    ("mit", "gpl-3.0"): "compatible",     # MIT code can be used in GPL projects
    ("gpl-3.0", "mit"): "incompatible",   # GPL code cannot be used in MIT projects
    ("apache-2.0", "gpl-2.0"): "incompatible",  # Apache 2.0 is incompatible with GPLv2
    ("apache-2.0", "gpl-3.0"): "compatible",    # Apache 2.0 is compatible with GPLv3
    ("mit", "apache-2.0"): "compatible",   # MIT can be used in Apache projects
    ("apache-2.0", "mit"): "compatible",   # Apache can be used in MIT projects
    ("mit", "agpl-3.0"): "compatible",     # MIT can be used in AGPL projects
    ("agpl-3.0", "mit"): "incompatible",   # AGPL cannot be used in MIT projects
}