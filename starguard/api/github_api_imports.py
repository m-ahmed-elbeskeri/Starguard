"""GitHub API class imports and initialization."""

# Import all the individual components
from starguard.api.github_api import GitHubAPI
from starguard.api.github_api_rest import GitHubRestMethods
from starguard.api.github_api_methods import GitHubApiMethods
from starguard.api.github_api_graphql import GitHubGraphQLMethods
from starguard.api.github_api_content import GitHubContentMethods

# Add REST API methods to GitHubAPI
for method_name in dir(GitHubRestMethods):
<<<<<<< HEAD
    if not method_name.startswith("__"):  # Skip dunder methods only, include private methods
=======
    if not method_name.startswith("_"):  # Skip private methods
>>>>>>> d5db550d9587a3942a812e51e400250bb668badc
        method = getattr(GitHubRestMethods, method_name)
        if callable(method):
            setattr(GitHubAPI, method_name, method)

# Add other API methods to GitHubAPI
for method_name in dir(GitHubApiMethods):
<<<<<<< HEAD
    if not method_name.startswith("__"):  # Skip dunder methods only, include private methods
=======
    if not method_name.startswith("_"):
>>>>>>> d5db550d9587a3942a812e51e400250bb668badc
        method = getattr(GitHubApiMethods, method_name)
        if callable(method):
            setattr(GitHubAPI, method_name, method)

# Add GraphQL methods to GitHubAPI
for method_name in dir(GitHubGraphQLMethods):
<<<<<<< HEAD
    if not method_name.startswith("__"):  # Skip dunder methods only, include private methods
=======
    if not method_name.startswith("_"):
>>>>>>> d5db550d9587a3942a812e51e400250bb668badc
        method = getattr(GitHubGraphQLMethods, method_name)
        if callable(method):
            setattr(GitHubAPI, method_name, method)

<<<<<<< HEAD
# Add content methods to GitHubAPI (including private methods needed for dependency parsing)
for method_name in dir(GitHubContentMethods):
    if not method_name.startswith("__"):  # Skip dunder methods only, include private methods
=======
# Add content methods to GitHubAPI
for method_name in dir(GitHubContentMethods):
    if not method_name.startswith("_"):
>>>>>>> d5db550d9587a3942a812e51e400250bb668badc
        method = getattr(GitHubContentMethods, method_name)
        if callable(method):
            setattr(GitHubAPI, method_name, method)

# Export only GitHubAPI for use by external code
<<<<<<< HEAD
__all__ = ["GitHubAPI"]
=======
__all__ = ["GitHubAPI"]
>>>>>>> d5db550d9587a3942a812e51e400250bb668badc
