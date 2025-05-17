"""GitHub API class imports and initialization."""

# Import all the individual components
from starguard.api.github_api import GitHubAPI
from starguard.api.github_api_rest import GitHubRestMethods
from starguard.api.github_api_methods import GitHubApiMethods
from starguard.api.github_api_graphql import GitHubGraphQLMethods
from starguard.api.github_api_content import GitHubContentMethods

# Add REST API methods to GitHubAPI
for method_name in dir(GitHubRestMethods):
    if not method_name.startswith("_"):  # Skip private methods
        method = getattr(GitHubRestMethods, method_name)
        if callable(method):
            setattr(GitHubAPI, method_name, method)

# Add other API methods to GitHubAPI
for method_name in dir(GitHubApiMethods):
    if not method_name.startswith("_"):
        method = getattr(GitHubApiMethods, method_name)
        if callable(method):
            setattr(GitHubAPI, method_name, method)

# Add GraphQL methods to GitHubAPI
for method_name in dir(GitHubGraphQLMethods):
    if not method_name.startswith("_"):
        method = getattr(GitHubGraphQLMethods, method_name)
        if callable(method):
            setattr(GitHubAPI, method_name, method)

# Add content methods to GitHubAPI
for method_name in dir(GitHubContentMethods):
    if not method_name.startswith("_"):
        method = getattr(GitHubContentMethods, method_name)
        if callable(method):
            setattr(GitHubAPI, method_name, method)

# Export only GitHubAPI for use by external code
__all__ = ["GitHubAPI"]
