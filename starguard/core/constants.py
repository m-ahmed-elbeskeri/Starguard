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
# TODO: Needs a proper check 

LICENSE_COMPATIBILITY = {
    # MIT License compatibility
    ("mit", "mit"): "compatible",
    ("mit", "apache-2.0"): "compatible",
    ("mit", "bsd-2-clause"): "compatible",
    ("mit", "bsd-3-clause"): "compatible",
    ("mit", "gpl-2.0"): "compatible",
    ("mit", "gpl-3.0"): "compatible",
    ("mit", "lgpl-2.1"): "compatible",
    ("mit", "lgpl-3.0"): "compatible",
    ("mit", "agpl-3.0"): "compatible",
    ("mit", "mpl-2.0"): "compatible",
    ("mit", "epl-2.0"): "compatible",
    ("mit", "cddl-1.0"): "compatible",
    ("mit", "unlicense"): "compatible",
    ("mit", "cc0-1.0"): "compatible",
    ("mit", "isc"): "compatible",
    
    # Apache 2.0 License compatibility
    ("apache-2.0", "mit"): "compatible",
    ("apache-2.0", "apache-2.0"): "compatible",
    ("apache-2.0", "bsd-2-clause"): "compatible",
    ("apache-2.0", "bsd-3-clause"): "compatible",
    ("apache-2.0", "gpl-2.0"): "incompatible",  # Apache 2.0 is incompatible with GPLv2
    ("apache-2.0", "gpl-3.0"): "compatible",    # Apache 2.0 is compatible with GPLv3
    ("apache-2.0", "lgpl-2.1"): "incompatible", # Same incompatibility with LGPLv2.1
    ("apache-2.0", "lgpl-3.0"): "compatible",   # Compatible with LGPLv3
    ("apache-2.0", "agpl-3.0"): "compatible",   # Compatible with AGPLv3
    ("apache-2.0", "mpl-2.0"): "compatible",
    ("apache-2.0", "epl-2.0"): "conditional",  # Potentially compatible but with conditions
    ("apache-2.0", "cddl-1.0"): "conditional",
    ("apache-2.0", "unlicense"): "compatible",
    ("apache-2.0", "cc0-1.0"): "compatible",
    ("apache-2.0", "isc"): "compatible",
    
    # BSD 2-Clause License compatibility
    ("bsd-2-clause", "mit"): "compatible",
    ("bsd-2-clause", "apache-2.0"): "compatible",
    ("bsd-2-clause", "bsd-2-clause"): "compatible",
    ("bsd-2-clause", "bsd-3-clause"): "compatible",
    ("bsd-2-clause", "gpl-2.0"): "compatible",
    ("bsd-2-clause", "gpl-3.0"): "compatible",
    ("bsd-2-clause", "lgpl-2.1"): "compatible",
    ("bsd-2-clause", "lgpl-3.0"): "compatible",
    ("bsd-2-clause", "agpl-3.0"): "compatible",
    ("bsd-2-clause", "mpl-2.0"): "compatible",
    ("bsd-2-clause", "epl-2.0"): "compatible",
    ("bsd-2-clause", "cddl-1.0"): "compatible",
    ("bsd-2-clause", "unlicense"): "compatible",
    ("bsd-2-clause", "cc0-1.0"): "compatible",
    ("bsd-2-clause", "isc"): "compatible",
    
    # BSD 3-Clause License compatibility
    ("bsd-3-clause", "mit"): "compatible",
    ("bsd-3-clause", "apache-2.0"): "compatible",
    ("bsd-3-clause", "bsd-2-clause"): "compatible",
    ("bsd-3-clause", "bsd-3-clause"): "compatible",
    ("bsd-3-clause", "gpl-2.0"): "compatible",
    ("bsd-3-clause", "gpl-3.0"): "compatible",
    ("bsd-3-clause", "lgpl-2.1"): "compatible",
    ("bsd-3-clause", "lgpl-3.0"): "compatible",
    ("bsd-3-clause", "agpl-3.0"): "compatible",
    ("bsd-3-clause", "mpl-2.0"): "compatible",
    ("bsd-3-clause", "epl-2.0"): "compatible",
    ("bsd-3-clause", "cddl-1.0"): "compatible",
    ("bsd-3-clause", "unlicense"): "compatible",
    ("bsd-3-clause", "cc0-1.0"): "compatible",
    ("bsd-3-clause", "isc"): "compatible",
    
    # GPL-2.0 License compatibility
    ("gpl-2.0", "mit"): "incompatible",     # GPL code cannot be included in MIT projects
    ("gpl-2.0", "apache-2.0"): "incompatible",
    ("gpl-2.0", "bsd-2-clause"): "incompatible",
    ("gpl-2.0", "bsd-3-clause"): "incompatible",
    ("gpl-2.0", "gpl-2.0"): "compatible",
    ("gpl-2.0", "gpl-3.0"): "incompatible", # GPLv2 code cannot be included in GPLv3 without explicit permission
    ("gpl-2.0", "lgpl-2.1"): "compatible",
    ("gpl-2.0", "lgpl-3.0"): "incompatible",
    ("gpl-2.0", "agpl-3.0"): "incompatible",
    ("gpl-2.0", "mpl-2.0"): "incompatible",
    ("gpl-2.0", "epl-2.0"): "incompatible",
    ("gpl-2.0", "cddl-1.0"): "incompatible",
    ("gpl-2.0", "unlicense"): "incompatible",
    ("gpl-2.0", "cc0-1.0"): "incompatible",
    ("gpl-2.0", "isc"): "incompatible",
    
    # GPL-3.0 License compatibility
    ("gpl-3.0", "mit"): "incompatible",     # GPL code cannot be included in MIT projects
    ("gpl-3.0", "apache-2.0"): "incompatible",
    ("gpl-3.0", "bsd-2-clause"): "incompatible",
    ("gpl-3.0", "bsd-3-clause"): "incompatible",
    ("gpl-3.0", "gpl-2.0"): "incompatible", # GPLv3 cannot be included in GPLv2 projects
    ("gpl-3.0", "gpl-3.0"): "compatible",
    ("gpl-3.0", "lgpl-2.1"): "incompatible",
    ("gpl-3.0", "lgpl-3.0"): "compatible",
    ("gpl-3.0", "agpl-3.0"): "compatible",  # GPL-3.0 code can be used in AGPL-3.0 projects
    ("gpl-3.0", "mpl-2.0"): "incompatible",
    ("gpl-3.0", "epl-2.0"): "incompatible",
    ("gpl-3.0", "cddl-1.0"): "incompatible",
    ("gpl-3.0", "unlicense"): "incompatible",
    ("gpl-3.0", "cc0-1.0"): "incompatible",
    ("gpl-3.0", "isc"): "incompatible",
    
    # LGPL-2.1 License compatibility
    ("lgpl-2.1", "mit"): "incompatible",
    ("lgpl-2.1", "apache-2.0"): "incompatible",
    ("lgpl-2.1", "bsd-2-clause"): "incompatible",
    ("lgpl-2.1", "bsd-3-clause"): "incompatible",
    ("lgpl-2.1", "gpl-2.0"): "compatible",  # LGPL can be upgraded to GPL
    ("lgpl-2.1", "gpl-3.0"): "incompatible",
    ("lgpl-2.1", "lgpl-2.1"): "compatible",
    ("lgpl-2.1", "lgpl-3.0"): "incompatible",
    ("lgpl-2.1", "agpl-3.0"): "incompatible",
    ("lgpl-2.1", "mpl-2.0"): "incompatible",
    ("lgpl-2.1", "epl-2.0"): "incompatible",
    ("lgpl-2.1", "cddl-1.0"): "incompatible",
    ("lgpl-2.1", "unlicense"): "incompatible",
    ("lgpl-2.1", "cc0-1.0"): "incompatible",
    ("lgpl-2.1", "isc"): "incompatible",
    
    # LGPL-3.0 License compatibility
    ("lgpl-3.0", "mit"): "incompatible",
    ("lgpl-3.0", "apache-2.0"): "incompatible",
    ("lgpl-3.0", "bsd-2-clause"): "incompatible",
    ("lgpl-3.0", "bsd-3-clause"): "incompatible",
    ("lgpl-3.0", "gpl-2.0"): "incompatible",
    ("lgpl-3.0", "gpl-3.0"): "compatible",  # LGPL can be upgraded to GPL
    ("lgpl-3.0", "lgpl-2.1"): "incompatible",
    ("lgpl-3.0", "lgpl-3.0"): "compatible",
    ("lgpl-3.0", "agpl-3.0"): "compatible",
    ("lgpl-3.0", "mpl-2.0"): "incompatible",
    ("lgpl-3.0", "epl-2.0"): "incompatible",
    ("lgpl-3.0", "cddl-1.0"): "incompatible",
    ("lgpl-3.0", "unlicense"): "incompatible",
    ("lgpl-3.0", "cc0-1.0"): "incompatible",
    ("lgpl-3.0", "isc"): "incompatible",
    
    # AGPL-3.0 License compatibility
    ("agpl-3.0", "mit"): "incompatible",
    ("agpl-3.0", "apache-2.0"): "incompatible",
    ("agpl-3.0", "bsd-2-clause"): "incompatible",
    ("agpl-3.0", "bsd-3-clause"): "incompatible",
    ("agpl-3.0", "gpl-2.0"): "incompatible",
    ("agpl-3.0", "gpl-3.0"): "incompatible", # AGPL is more restrictive than GPL
    ("agpl-3.0", "lgpl-2.1"): "incompatible",
    ("agpl-3.0", "lgpl-3.0"): "incompatible",
    ("agpl-3.0", "agpl-3.0"): "compatible",
    ("agpl-3.0", "mpl-2.0"): "incompatible",
    ("agpl-3.0", "epl-2.0"): "incompatible",
    ("agpl-3.0", "cddl-1.0"): "incompatible",
    ("agpl-3.0", "unlicense"): "incompatible",
    ("agpl-3.0", "cc0-1.0"): "incompatible",
    ("agpl-3.0", "isc"): "incompatible",
    
    # MPL-2.0 License compatibility
    ("mpl-2.0", "mit"): "compatible",
    ("mpl-2.0", "apache-2.0"): "compatible",
    ("mpl-2.0", "bsd-2-clause"): "compatible",
    ("mpl-2.0", "bsd-3-clause"): "compatible",
    ("mpl-2.0", "gpl-2.0"): "incompatible",
    ("mpl-2.0", "gpl-3.0"): "compatible",
    ("mpl-2.0", "lgpl-2.1"): "incompatible",
    ("mpl-2.0", "lgpl-3.0"): "compatible",
    ("mpl-2.0", "agpl-3.0"): "compatible",
    ("mpl-2.0", "mpl-2.0"): "compatible",
    ("mpl-2.0", "epl-2.0"): "conditional",
    ("mpl-2.0", "cddl-1.0"): "incompatible",
    ("mpl-2.0", "unlicense"): "compatible",
    ("mpl-2.0", "cc0-1.0"): "compatible",
    ("mpl-2.0", "isc"): "compatible",
    
    # EPL-2.0 License compatibility
    ("epl-2.0", "mit"): "conditional",
    ("epl-2.0", "apache-2.0"): "conditional",
    ("epl-2.0", "bsd-2-clause"): "conditional",
    ("epl-2.0", "bsd-3-clause"): "conditional",
    ("epl-2.0", "gpl-2.0"): "incompatible",
    ("epl-2.0", "gpl-3.0"): "incompatible",
    ("epl-2.0", "lgpl-2.1"): "incompatible",
    ("epl-2.0", "lgpl-3.0"): "incompatible",
    ("epl-2.0", "agpl-3.0"): "incompatible",
    ("epl-2.0", "mpl-2.0"): "conditional",
    ("epl-2.0", "epl-2.0"): "compatible",
    ("epl-2.0", "cddl-1.0"): "incompatible",
    ("epl-2.0", "unlicense"): "conditional",
    ("epl-2.0", "cc0-1.0"): "conditional",
    ("epl-2.0", "isc"): "conditional",
    
    # CDDL-1.0 License compatibility
    ("cddl-1.0", "mit"): "conditional",
    ("cddl-1.0", "apache-2.0"): "conditional",
    ("cddl-1.0", "bsd-2-clause"): "conditional",
    ("cddl-1.0", "bsd-3-clause"): "conditional",
    ("cddl-1.0", "gpl-2.0"): "incompatible",
    ("cddl-1.0", "gpl-3.0"): "incompatible",
    ("cddl-1.0", "lgpl-2.1"): "incompatible",
    ("cddl-1.0", "lgpl-3.0"): "incompatible",
    ("cddl-1.0", "agpl-3.0"): "incompatible",
    ("cddl-1.0", "mpl-2.0"): "incompatible",
    ("cddl-1.0", "epl-2.0"): "incompatible",
    ("cddl-1.0", "cddl-1.0"): "compatible",
    ("cddl-1.0", "unlicense"): "conditional",
    ("cddl-1.0", "cc0-1.0"): "conditional",
    ("cddl-1.0", "isc"): "conditional",
    
    # Unlicense compatibility
    ("unlicense", "mit"): "compatible",
    ("unlicense", "apache-2.0"): "compatible",
    ("unlicense", "bsd-2-clause"): "compatible",
    ("unlicense", "bsd-3-clause"): "compatible",
    ("unlicense", "gpl-2.0"): "compatible",
    ("unlicense", "gpl-3.0"): "compatible",
    ("unlicense", "lgpl-2.1"): "compatible",
    ("unlicense", "lgpl-3.0"): "compatible",
    ("unlicense", "agpl-3.0"): "compatible",
    ("unlicense", "mpl-2.0"): "compatible",
    ("unlicense", "epl-2.0"): "compatible",
    ("unlicense", "cddl-1.0"): "compatible",
    ("unlicense", "unlicense"): "compatible",
    ("unlicense", "cc0-1.0"): "compatible",
    ("unlicense", "isc"): "compatible",
    
    # CC0-1.0 License compatibility
    ("cc0-1.0", "mit"): "compatible",
    ("cc0-1.0", "apache-2.0"): "compatible",
    ("cc0-1.0", "bsd-2-clause"): "compatible",
    ("cc0-1.0", "bsd-3-clause"): "compatible",
    ("cc0-1.0", "gpl-2.0"): "compatible",
    ("cc0-1.0", "gpl-3.0"): "compatible",
    ("cc0-1.0", "lgpl-2.1"): "compatible",
    ("cc0-1.0", "lgpl-3.0"): "compatible",
    ("cc0-1.0", "agpl-3.0"): "compatible",
    ("cc0-1.0", "mpl-2.0"): "compatible",
    ("cc0-1.0", "epl-2.0"): "compatible",
    ("cc0-1.0", "cddl-1.0"): "compatible",
    ("cc0-1.0", "unlicense"): "compatible",
    ("cc0-1.0", "cc0-1.0"): "compatible",
    ("cc0-1.0", "isc"): "compatible",
    
    # ISC License compatibility
    ("isc", "mit"): "compatible",
    ("isc", "apache-2.0"): "compatible",
    ("isc", "bsd-2-clause"): "compatible",
    ("isc", "bsd-3-clause"): "compatible",
    ("isc", "gpl-2.0"): "compatible",
    ("isc", "gpl-3.0"): "compatible",
    ("isc", "lgpl-2.1"): "compatible",
    ("isc", "lgpl-3.0"): "compatible",
    ("isc", "agpl-3.0"): "compatible",
    ("isc", "mpl-2.0"): "compatible",
    ("isc", "epl-2.0"): "compatible",
    ("isc", "cddl-1.0"): "compatible",
    ("isc", "unlicense"): "compatible",
    ("isc", "cc0-1.0"): "compatible",
    ("isc", "isc"): "compatible",
    
    # Additional licenses
    ("artistic-2.0", "mit"): "incompatible",
    ("artistic-2.0", "gpl-2.0"): "compatible",
    ("artistic-2.0", "gpl-3.0"): "compatible",
    
    ("ms-pl", "mit"): "incompatible",
    ("ms-pl", "gpl-2.0"): "incompatible",
    ("ms-pl", "ms-pl"): "compatible",
    
    ("wtfpl", "mit"): "compatible",
    ("wtfpl", "apache-2.0"): "compatible",
    ("wtfpl", "gpl-2.0"): "compatible",
    ("wtfpl", "gpl-3.0"): "compatible",
    ("wtfpl", "wtfpl"): "compatible",
}