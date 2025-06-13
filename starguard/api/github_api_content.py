"""GitHub API methods for repository content and files."""

import logging
import base64
import json
import re
from typing import Dict, List, Optional

from starguard.core.constants import PACKAGE_MANAGERS

logger = logging.getLogger(__name__)


class GitHubContentMethods:
    """
    Implementation of GitHub API methods for repository content.

    This class provides methods for retrieving and analyzing repository files.
    """

    def get_file_content(
        self, owner: str, repo: str, path: str, ref: Optional[str] = None
    ) -> Optional[str]:
        """
        Get file content from a repository.

        Retrieves and decodes the contents of a file from a GitHub repository.

        Args:
            owner: Repository owner
            repo: Repository name
            path: Path to the file within the repository
            ref: Git reference (branch, tag, or commit SHA) to get the file from

        Returns:
            Optional[str]: Decoded file content or None if not found/error
        """
        logger.debug(f"Fetching file content for {path} in {owner}/{repo}")
        endpoint = f"/repos/{owner}/{repo}/contents/{path}"
        params = {}
        if ref:
            params["ref"] = ref

        response_data = self.request(endpoint, params=params)
        if "error" in response_data or "content" not in response_data:
            logger.debug(
                f"Could not fetch file content for {path} in {owner}/{repo}: "
                f"{response_data.get('error', 'No content field')}"
            )
            return None

        try:
            # GitHub returns base64-encoded content
            content = base64.b64decode(response_data["content"]).decode("utf-8")
            logger.debug(f"Successfully decoded {len(content)} bytes of content from {path}")
            return content
        except (UnicodeDecodeError, base64.binascii.Error) as e:
            logger.debug(f"Error decoding file content for {path}: {e}")
            return None

    def get_directory_contents(
        self, owner: str, repo: str, path: str = "", ref: Optional[str] = None
    ) -> List[Dict]:
        """
        Get contents of a directory in a repository.

        Args:
            owner: Repository owner
            repo: Repository name
            path: Path to the directory (empty for root)
            ref: Git reference (branch, tag, or commit SHA)

        Returns:
            List[Dict]: List of file/directory entries
        """
        logger.debug(f"Fetching directory contents for {path} in {owner}/{repo}")
        endpoint = f"/repos/{owner}/{repo}/contents/{path}"
        params = {}
        if ref:
            params["ref"] = ref

        response_data = self.request(endpoint, params=params)

        if "error" in response_data:
            logger.debug(f"Error fetching directory contents: {response_data.get('error')}")
            return []

        # If it's a single file, wrap in list for consistent return type
        if isinstance(response_data, dict) and "type" in response_data:
            return [response_data]

        if isinstance(response_data, list):
            return response_data

        logger.warning(f"Unexpected response type for directory contents: {type(response_data)}")
        return []

    def get_dependencies(self, owner: str, repo: str) -> Dict:
        """
        Get repository dependencies using the dependency graph API or file parsing.

        Attempts to use GitHub's dependency graph API first, then falls back
        to parsing manifest files (package.json, requirements.txt, etc.)
<<<<<<< HEAD
        if the API isn't available or returns empty results.
=======
        if the API isn't available.
>>>>>>> d5db550d9587a3942a812e51e400250bb668badc

        Args:
            owner: Repository owner
            repo: Repository name

        Returns:
            Dict: Dependency information either from API or file parsing
        """
        logger.info(f"Fetching dependencies for {owner}/{repo}")
        endpoint = f"/repos/{owner}/{repo}/dependency-graph/sbom"
<<<<<<< HEAD
        
        # Try the dependency graph API first
        try:
            response = self.request(endpoint)

            # Check if API call was successful and has actual dependency data
            if "error" not in response and "sbom" in response:
                sbom_data = response.get("sbom", {})
                packages = sbom_data.get("packages", [])
                
                # Only use API result if it actually contains packages
                if packages and len(packages) > 0:
                    # Filter to only direct dependencies for analysis
                    direct_packages = [
                        pkg for pkg in packages 
                        if isinstance(pkg, dict) and pkg.get("relationship") == "direct"
                    ]
                    
                    if direct_packages:
                        logger.info(f"Successfully fetched {len(direct_packages)} direct dependencies via API for {owner}/{repo}")
                        return response
                    else:
                        logger.info(f"API returned packages but no direct dependencies for {owner}/{repo}, trying file parsing")
                else:
                    logger.info(f"API returned empty dependency list for {owner}/{repo}, trying file parsing")
            else:
                api_error = response.get('error', 'No SBOM field in response')
                logger.info(f"Could not fetch dependencies via API for {owner}/{repo} (Reason: {api_error}), trying file parsing")

        except Exception as e:
            logger.warning(f"Exception fetching dependencies via API for {owner}/{repo}: {str(e)}, trying file parsing")

        # Fallback: Parse manifest files
        logger.info(f"Falling back to file parsing for dependencies in {owner}/{repo}")
=======
        try:
            # Try the dependency graph API first
            response = self.request(endpoint)

            if "error" not in response and "sbom" in response:
                logger.info(f"Successfully fetched dependencies via API for {owner}/{repo}")
                return response

            logger.warning(
                f"Could not fetch dependencies via API for {owner}/{repo} "
                f"(Reason: {response.get('error', 'No SBOM field')}). "
                f"Falling back to file parsing."
            )

        except Exception as e:
            logger.warning(
                f"Exception fetching dependencies via API for {owner}/{repo}: "
                f"{str(e)}. Falling back to file parsing."
            )

        # Fallback: Parse manifest files
>>>>>>> d5db550d9587a3942a812e51e400250bb668badc
        return self._parse_dependencies_from_files(owner, repo)

    def _parse_dependencies_from_files(self, owner: str, repo: str) -> Dict:
        """
        Parse dependencies from manifest files.

        Checks for package manager manifest files (package.json, requirements.txt, etc.)
        and parses them to extract dependency information.

        Args:
            owner: Repository owner
            repo: Repository name

        Returns:
            Dict: Parsed dependency information grouped by language
        """
        logger.info(f"Parsing dependency files for {owner}/{repo}")
        dependencies = {}
<<<<<<< HEAD
        files_checked = 0
        files_found = 0

        for lang, config in PACKAGE_MANAGERS.items():
            try:
                files_checked += 1
=======

        for lang, config in PACKAGE_MANAGERS.items():
            try:
>>>>>>> d5db550d9587a3942a812e51e400250bb668badc
                logger.debug(f"Checking for {lang} dependency file: {config['file']}")
                content = self.get_file_content(owner, repo, config["file"])

                if content:
<<<<<<< HEAD
                    files_found += 1
                    logger.debug(f"Found {lang} dependency file ({len(content)} bytes), parsing...")
=======
                    logger.debug(f"Found {lang} dependency file, parsing...")
>>>>>>> d5db550d9587a3942a812e51e400250bb668badc
                    deps = self._parse_dependency_file(content, config, lang)

                    if deps:
                        dependencies[lang] = deps
<<<<<<< HEAD
                        logger.info(f"Parsed {len(deps)} {lang} dependencies from {config['file']}")
                    else:
                        logger.debug(f"No dependencies found in {config['file']} for {lang}")
                else:
                    logger.debug(f"No {config['file']} file found for {lang}")
=======
                        logger.debug(f"Parsed {len(deps)} {lang} dependencies")
>>>>>>> d5db550d9587a3942a812e51e400250bb668badc

            except Exception as e:
                logger.debug(f"Could not parse {config['file']} for {owner}/{repo}: {str(e)}")

<<<<<<< HEAD
        if files_found == 0:
            logger.warning(f"No dependency files found in {owner}/{repo} (checked {files_checked} possible files)")
        else:
            logger.info(f"Found {files_found} dependency files out of {files_checked} checked in {owner}/{repo}")
            
        if not dependencies:
            logger.info(f"No dependencies parsed from any files in {owner}/{repo}")
            
=======
        logger.info(f"Parsed dependencies for {len(dependencies)} languages in {owner}/{repo}")
>>>>>>> d5db550d9587a3942a812e51e400250bb668badc
        return dependencies

    def _parse_dependency_file(self, content: str, config: Dict, lang: str) -> List[Dict]:
        """
        Parse a dependency file based on its format.

        Uses language-specific parsing logic to extract dependencies
        from manifest files.

        Args:
            content: File content to parse
            config: Language-specific configuration from PACKAGE_MANAGERS
            lang: Language identifier

        Returns:
            List[Dict]: Parsed dependency information
        """
        deps = []
        if not content:
            return deps

        # JavaScript (package.json)
        if lang == "javascript":
            try:
                package_json = json.loads(content)
                for key in config["dependencies_key"]:
                    if key in package_json and isinstance(package_json[key], dict):
                        for name, version in package_json[key].items():
                            deps.append(
                                {
                                    "name": name,
                                    "version": str(version),
                                    "type": "runtime" if key == "dependencies" else "development",
                                }
                            )
<<<<<<< HEAD
                logger.debug(f"Parsed {len(deps)} dependencies from package.json")
=======
>>>>>>> d5db550d9587a3942a812e51e400250bb668badc
            except json.JSONDecodeError as e:
                logger.debug(f"JSONDecodeError parsing package.json: {e}")

        # Python (requirements.txt)
        elif lang == "python":
            pattern = re.compile(config["pattern"])
<<<<<<< HEAD
            for line_num, line in enumerate(content.splitlines(), 1):
                line = line.strip()
                if line and not line.startswith("#") and not line.startswith("-"):
=======
            for line in content.splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
>>>>>>> d5db550d9587a3942a812e51e400250bb668badc
                    match = pattern.match(line)
                    if match:
                        name = match.group(1)
                        version = match.group(2) if match.group(2) else "latest"
                        deps.append({"name": name, "version": version, "type": "runtime"})
<<<<<<< HEAD
                    else:
                        logger.debug(f"Could not parse requirements.txt line {line_num}: {line}")
            logger.debug(f"Parsed {len(deps)} dependencies from requirements.txt")
=======
>>>>>>> d5db550d9587a3942a812e51e400250bb668badc

        # Python (Pipfile)
        elif lang == "python_pipfile":
            in_section = False
            section_type = "runtime"
            for line in content.splitlines():
                line = line.strip()
                if "[packages]" in line:
                    in_section = True
                    section_type = "runtime"
                    continue
                elif "[dev-packages]" in line:
                    in_section = True
                    section_type = "development"
                    continue
                elif line.startswith("[") and in_section:
                    in_section = False
                    continue

                if in_section and not line.startswith("#") and "=" in line:
                    parts = line.split("=", 1)
                    name = parts[0].strip().strip("\"'")
                    version = parts[1].strip().strip("\"'")
                    deps.append({"name": name, "version": version, "type": section_type})
<<<<<<< HEAD
            logger.debug(f"Parsed {len(deps)} dependencies from Pipfile")
=======
>>>>>>> d5db550d9587a3942a812e51e400250bb668badc

        # Python (pyproject.toml) - Poetry
        elif lang == "python_poetry":
            in_deps_section = False
            in_dev_deps_section = False
            section_type = "runtime"

            for line in content.splitlines():
                line = line.strip()
                if "[tool.poetry.dependencies]" in line:
                    in_deps_section = True
                    in_dev_deps_section = False
                    section_type = "runtime"
                    continue
<<<<<<< HEAD
                elif "[tool.poetry.dev-dependencies]" in line or "[tool.poetry.group.dev.dependencies]" in line:
=======
                elif "[tool.poetry.dev-dependencies]" in line:
>>>>>>> d5db550d9587a3942a812e51e400250bb668badc
                    in_deps_section = False
                    in_dev_deps_section = True
                    section_type = "development"
                    continue
                elif line.startswith("[") and (in_deps_section or in_dev_deps_section):
                    in_deps_section = False
                    in_dev_deps_section = False
                    continue

                if (
                    (in_deps_section or in_dev_deps_section)
                    and not line.startswith("#")
                    and "=" in line
<<<<<<< HEAD
                    and not line.startswith("python")  # Skip python version requirement
=======
>>>>>>> d5db550d9587a3942a812e51e400250bb668badc
                ):
                    # Extract package name (may be quoted)
                    name_match = re.match(r'^(["\']?)([a-zA-Z0-9_\-\.]+)\1\s*=\s*(.+)$', line)
                    if name_match:
                        name = name_match.group(2)
                        version_spec = name_match.group(3).strip()

                        # Handle different version formats
                        if version_spec.startswith('"') or version_spec.startswith("'"):
                            # Simple string version: package = "^1.0.0"
                            version = version_spec.strip("'\"")
                        elif version_spec.startswith("{"):
                            # Dict format: package = { version = "^1.0.0", optional = true }
                            version_match = re.search(
                                r'version\s*=\s*["\']([^"\']+)["\']', version_spec
                            )
                            version = version_match.group(1) if version_match else "latest"
                        else:
                            version = version_spec

                        deps.append({"name": name, "version": version, "type": section_type})
<<<<<<< HEAD
            logger.debug(f"Parsed {len(deps)} dependencies from pyproject.toml")
=======
>>>>>>> d5db550d9587a3942a812e51e400250bb668badc

        # Ruby (Gemfile)
        elif lang == "ruby":
            gem_pattern = re.compile(
                r"^\s*gem\s+['\"]([^'\"]+)['\"](?:[,\s]+['\"]([^'\"]+)['\"])?.*$"
            )
            for line in content.splitlines():
                line = line.strip()
                if line.startswith("gem"):
                    match = gem_pattern.match(line)
                    if match:
                        name = match.group(1)
                        version = match.group(2) if match.group(2) else "latest"
                        deps.append({"name": name, "version": version, "type": "runtime"})
<<<<<<< HEAD
            logger.debug(f"Parsed {len(deps)} dependencies from Gemfile")
=======
>>>>>>> d5db550d9587a3942a812e51e400250bb668badc

        # Java (pom.xml)
        elif lang == "java":
            # Simple regex for basic pom.xml parsing - for more complex projects, a proper XML parser would be better
            dependency_pattern = re.compile(
                r"<dependency>.*?<groupId>(.*?)</groupId>.*?<artifactId>(.*?)</artifactId>(?:.*?<version>(.*?)</version>)?.*?</dependency>",
                re.DOTALL,
            )

            matches = dependency_pattern.findall(content)
            for match in matches:
                group_id = match[0].strip()
                artifact_id = match[1].strip()
                version = match[2].strip() if len(match) > 2 and match[2] else "latest"

                deps.append(
                    {"name": f"{group_id}:{artifact_id}", "version": version, "type": "runtime"}
                )
<<<<<<< HEAD
            logger.debug(f"Parsed {len(deps)} dependencies from pom.xml")
=======
>>>>>>> d5db550d9587a3942a812e51e400250bb668badc

        # Go (go.mod)
        elif lang == "go":
            # Find require statements
            in_require_block = False
            for line in content.splitlines():
                line = line.strip()

                # Start of multi-line require block
                if line == "require (":
                    in_require_block = True
                    continue

                # End of multi-line require block
                if line == ")" and in_require_block:
                    in_require_block = False
                    continue

                # Single-line require outside a block
                if line.startswith("require ") and not in_require_block:
                    # Extract module and version
                    single_require_match = re.match(r"require\s+([^\s]+)\s+([^\s]+)", line)
                    if single_require_match:
                        module_path = single_require_match.group(1)
                        version = single_require_match.group(2)
                        deps.append(
                            {
                                "name": module_path,
                                "version": version.lstrip("v"),  # Strip leading 'v'
                                "type": "runtime",
                            }
                        )

                # Dependency inside a require block
                elif in_require_block and line and not line.startswith("//"):
                    # Extract module and version
                    block_require_match = re.match(r"([^\s]+)\s+([^\s]+)", line)
                    if block_require_match:
                        module_path = block_require_match.group(1)
                        version = block_require_match.group(2)
                        deps.append(
                            {
                                "name": module_path,
                                "version": version.lstrip("v"),  # Strip leading 'v'
                                "type": "runtime",
                            }
                        )
<<<<<<< HEAD
            logger.debug(f"Parsed {len(deps)} dependencies from go.mod")
=======
>>>>>>> d5db550d9587a3942a812e51e400250bb668badc

        # PHP (composer.json)
        elif lang == "php":
            try:
                composer_json = json.loads(content)
                # Process require and require-dev
                for key in ["require", "require-dev"]:
                    if key in composer_json and isinstance(composer_json[key], dict):
                        dep_type = "runtime" if key == "require" else "development"
                        for name, version in composer_json[key].items():
                            # Skip PHP itself
                            if name != "php":
                                deps.append(
                                    {"name": name, "version": str(version), "type": dep_type}
                                )
<<<<<<< HEAD
                logger.debug(f"Parsed {len(deps)} dependencies from composer.json")
=======
>>>>>>> d5db550d9587a3942a812e51e400250bb668badc
            except json.JSONDecodeError as e:
                logger.debug(f"JSONDecodeError parsing composer.json: {e}")

        # .NET (packages.config)
        elif lang == "dotnet_packages":
            package_pattern = re.compile(
                r'<package\s+id=["\']([^"\']+)["\'](?:\s+version=["\']([^"\']+)["\'])?',
                re.IGNORECASE,
            )
            matches = package_pattern.findall(content)
            for match in matches:
                name = match[0]
                version = match[1] if len(match) > 1 and match[1] else "latest"
                deps.append({"name": name, "version": version, "type": "runtime"})
<<<<<<< HEAD
            logger.debug(f"Parsed {len(deps)} dependencies from packages.config")
=======
>>>>>>> d5db550d9587a3942a812e51e400250bb668badc

        # .NET (csproj or vbproj)
        elif lang == "dotnet_project":
            # Modern PackageReference format
            package_ref_pattern = re.compile(
                r'<PackageReference\s+Include=["\']([^"\']+)["\'](?:\s+Version=["\']([^"\']+)["\'])?',
                re.IGNORECASE,
            )
            package_refs = package_ref_pattern.findall(content)

            for package_ref in package_refs:
                name = package_ref[0]
                version = package_ref[1] if len(package_ref) > 1 and package_ref[1] else "latest"
                deps.append({"name": name, "version": version, "type": "runtime"})
<<<<<<< HEAD
            logger.debug(f"Parsed {len(deps)} dependencies from .NET project file")
=======
>>>>>>> d5db550d9587a3942a812e51e400250bb668badc

        # Rust (Cargo.toml)
        elif lang == "rust":
            in_deps_section = False
            in_dev_deps_section = False
            section_type = "runtime"

            for line in content.splitlines():
                line = line.strip()
                if "[dependencies]" in line:
                    in_deps_section = True
                    in_dev_deps_section = False
                    section_type = "runtime"
                    continue
                elif "[dev-dependencies]" in line:
                    in_deps_section = False
                    in_dev_deps_section = True
                    section_type = "development"
                    continue
                elif line.startswith("[") and (in_deps_section or in_dev_deps_section):
                    in_deps_section = False
                    in_dev_deps_section = False
                    continue

                if (
                    (in_deps_section or in_dev_deps_section)
                    and not line.startswith("#")
                    and "=" in line
                ):
                    # Extract package name and version
                    name_match = re.match(r'^(["\']?)([a-zA-Z0-9_\-\.]+)\1\s*=\s*(.+)$', line)
                    if name_match:
                        name = name_match.group(2)
                        version_spec = name_match.group(3).strip()

                        # Handle different version formats
                        if version_spec.startswith('"') or version_spec.startswith("'"):
                            # Simple string version
                            version = version_spec.strip("'\"")
                        elif version_spec.startswith("{"):
                            # Dict format with version
                            version_match = re.search(
                                r'version\s*=\s*["\']([^"\']+)["\']', version_spec
                            )
                            version = version_match.group(1) if version_match else "latest"
                        else:
                            version = version_spec

                        deps.append({"name": name, "version": version, "type": section_type})
<<<<<<< HEAD
            logger.debug(f"Parsed {len(deps)} dependencies from Cargo.toml")
=======
>>>>>>> d5db550d9587a3942a812e51e400250bb668badc

        return deps

    def detect_package_manager(self, owner: str, repo: str) -> List[str]:
        """
        Detect which package managers are used in a repository.

        Args:
            owner: Repository owner
            repo: Repository name

        Returns:
            List[str]: List of detected package managers
        """
        detected = []

        for lang, config in PACKAGE_MANAGERS.items():
            try:
                # Check if the package manager file exists
                content = self.get_file_content(owner, repo, config["file"])
<<<<<<< HEAD
                if content and len(content.strip()) > 0:
                    detected.append(lang)
                    logger.debug(f"Detected {lang} package manager via {config['file']}")
            except Exception as e:
                logger.debug(f"Error checking for {lang} package manager: {e}")

        logger.info(f"Detected package managers for {owner}/{repo}: {detected}")
=======
                if content:
                    detected.append(lang)
            except Exception:
                pass

>>>>>>> d5db550d9587a3942a812e51e400250bb668badc
        return detected

    def check_for_security_issues(self, owner: str, repo: str) -> Dict:
        """
        Check for known security issues in dependencies.

        This is a placeholder for integration with vulnerability databases.

        Args:
            owner: Repository owner
            repo: Repository name

        Returns:
            Dict: Security analysis results
        """
        # This will integrate with a vulnerability database
        # For now, we return a placeholder
        return {
            "has_vulnerabilities": False,
            "vulnerable_dependencies": [],
            "message": "Security scanning not implemented in this version",
<<<<<<< HEAD
        }
=======
        }
>>>>>>> d5db550d9587a3942a812e51e400250bb668badc
