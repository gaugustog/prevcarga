"""Version management for model serialization and deployment.

This module provides the VersionManager class for parsing, comparing, and managing
semantic versions of models. It supports version history tracking, version
comparison, and automatic version incrementing for model releases.

Example:
    ```python
    from src.models.serialization.version_manager import VersionManager
    from pathlib import Path

    vm = VersionManager()

    # Parse and compare versions
    v1 = vm.parse_version("1.2.3")
    v2 = vm.parse_version("1.3.0")
    comparison = vm.compare_versions(v1, v2)  # Returns -1 (v1 < v2)

    # Increment versions
    new_version = vm.increment_version("1.2.3", bump="minor")  # "1.3.0"
    new_version = vm.increment_version("1.2.3", bump="patch")  # "1.2.4"

    # Create version tags for filenames
    filename = vm.create_version_tag("lgbm", "1.2.3")  # "lgbm_v1.2.3"

    # Get version history from models directory
    models_dir = Path("models/")
    history = vm.get_version_history(models_dir)
    # Returns DataFrame with columns: version, model_type, file_path, created_at, etc.

    # Get latest version
    latest = vm.get_latest_version(models_dir, model_type="lgbm")  # "1.2.3"
    ```
"""

import json
from pathlib import Path
from typing import Literal

import pandas as pd

from src.models.base.versioning import SemanticVersion
from src.utils.logger import get_logger

logger = get_logger(__name__)


class VersionManager:
    """Manages semantic versioning for model serialization and deployment.

    This class provides utilities for:
    - Parsing version strings into structured tuples
    - Comparing versions to determine ordering
    - Incrementing versions (major, minor, patch bumps)
    - Creating version tags for filenames
    - Tracking version history across saved models
    - Finding latest versions of models

    The class uses the SemanticVersion class from src.models.base.versioning
    for version parsing and comparison.
    """

    def parse_version(self, version_str: str) -> tuple[int, int, int]:
        """Parse semantic version string into tuple of (major, minor, patch).

        Args:
            version_str: Version string in format "MAJOR.MINOR.PATCH"
                        (e.g., "1.2.3", "2.0.0-beta.1").

        Returns:
            Tuple of (major, minor, patch) as integers.
            Pre-release suffixes are ignored in the tuple representation.

        Raises:
            ValueError: If version string format is invalid.

        Example:
            ```python
            vm = VersionManager()
            version = vm.parse_version("1.2.3")  # (1, 2, 3)
            version = vm.parse_version("2.0.0-beta.1")  # (2, 0, 0)
            ```
        """
        try:
            sem_ver = SemanticVersion(version_str)
        except ValueError as e:
            msg = f"Failed to parse version '{version_str}': {e}"
            raise ValueError(msg) from e

        return (sem_ver.major, sem_ver.minor, sem_ver.patch)

    def compare_versions(
        self,
        v1: str | tuple[int, int, int],
        v2: str | tuple[int, int, int],
    ) -> int:
        """Compare two versions and return their ordering.

        Args:
            v1: First version (string or tuple).
            v2: Second version (string or tuple).

        Returns:
            -1 if v1 < v2
             0 if v1 == v2
             1 if v1 > v2

        Raises:
            ValueError: If version format is invalid.

        Example:
            ```python
            vm = VersionManager()
            result = vm.compare_versions("1.2.3", "1.3.0")  # -1
            result = vm.compare_versions("2.0.0", "1.9.9")  # 1
            result = vm.compare_versions("1.0.0", "1.0.0")  # 0
            ```
        """
        # Parse versions if they are strings
        if isinstance(v1, str):
            v1 = self.parse_version(v1)
        if isinstance(v2, str):
            v2 = self.parse_version(v2)

        # Compare tuples
        if v1 < v2:
            return -1
        if v1 > v2:
            return 1
        return 0

    def increment_version(
        self,
        current_version: str,
        bump: Literal["major", "minor", "patch"] = "patch",
    ) -> str:
        """Increment version number based on bump type.

        Follows semantic versioning rules:
        - Major bump: Increment major, reset minor and patch to 0
        - Minor bump: Increment minor, reset patch to 0
        - Patch bump: Increment patch only

        Args:
            current_version: Current version string (e.g., "1.2.3").
            bump: Version component to increment.
                 Options: "major", "minor", "patch".
                 Default is "patch".

        Returns:
            New version string with incremented version.

        Raises:
            ValueError: If current_version format is invalid or bump type is invalid.

        Example:
            ```python
            vm = VersionManager()
            new_ver = vm.increment_version("1.2.3", bump="major")  # "2.0.0"
            new_ver = vm.increment_version("1.2.3", bump="minor")  # "1.3.0"
            new_ver = vm.increment_version("1.2.3", bump="patch")  # "1.2.4"
            ```
        """
        # Parse current version
        major, minor, patch = self.parse_version(current_version)

        # Increment based on bump type
        if bump == "major":
            major += 1
            minor = 0
            patch = 0
        elif bump == "minor":
            minor += 1
            patch = 0
        elif bump == "patch":
            patch += 1
        else:
            msg = f"Invalid bump type: {bump}. Must be 'major', 'minor', or 'patch'"
            raise ValueError(msg)

        new_version = f"{major}.{minor}.{patch}"
        logger.debug(
            "Incremented version %s -> %s (bump=%s)",
            current_version,
            new_version,
            bump,
        )

        return new_version

    def create_version_tag(self, model_type: str, version: str) -> str:
        """Create a version tag for use in filenames.

        Args:
            model_type: Model type identifier (e.g., "lgbm", "random_forest").
            version: Version string (e.g., "1.2.3").

        Returns:
            Version tag string in format "{model_type}_v{version}".
            Example: "lgbm_v1.2.3"

        Raises:
            ValueError: If version format is invalid.

        Example:
            ```python
            vm = VersionManager()
            tag = vm.create_version_tag("lgbm", "1.2.3")  # "lgbm_v1.2.3"
            tag = vm.create_version_tag("random_forest", "2.0.0")  # "random_forest_v2.0.0"
            ```
        """
        # Validate version format
        self.parse_version(version)

        # Create tag
        tag = f"{model_type}_v{version}"
        logger.debug("Created version tag: %s", tag)

        return tag

    def get_version_history(
        self,
        models_dir: str | Path,
        model_type: str | None = None,
    ) -> pd.DataFrame:
        """Get version history of models in a directory.

        This method scans a directory for model files with metadata,
        extracts version information, and returns a DataFrame with
        version history sorted by version (newest first).

        Args:
            models_dir: Directory containing saved models.
            model_type: Optional filter for specific model type
                       (e.g., "lgbm", "random_forest").
                       If None, returns all models.

        Returns:
            DataFrame with columns:
            - version: Semantic version string
            - model_type: Model type identifier
            - model_name: Human-readable model name
            - file_path: Path to model file
            - created_at: When model was created/trained
            - serialized_at: When model was serialized
            - file_size_bytes: Model file size in bytes

            Sorted by version (newest first).
            Empty DataFrame if no models found.

        Example:
            ```python
            vm = VersionManager()
            history = vm.get_version_history("models/", model_type="lgbm")
            print(history)
            #   version model_type    model_name              file_path  ...
            # 0   1.2.3       lgbm  lgbm_model  models/lgbm_v1.2.3.pkl  ...
            # 1   1.2.2       lgbm  lgbm_model  models/lgbm_v1.2.2.pkl  ...
            ```
        """
        models_dir = Path(models_dir)

        if not models_dir.exists():
            logger.warning("Models directory not found: %s", models_dir)
            return pd.DataFrame()

        logger.info("Scanning models directory: %s (filter=%s)", models_dir, model_type)

        # Find all metadata files
        metadata_files = list(models_dir.rglob("*.meta.json"))
        logger.debug("Found %d metadata files", len(metadata_files))

        if not metadata_files:
            return pd.DataFrame()

        # Extract version information from metadata
        records = []
        for meta_file in metadata_files:
            try:
                with meta_file.open("r") as f:
                    metadata = json.load(f)

                # Filter by model type if specified
                if model_type is not None and metadata.get("model_type") != model_type:
                    continue

                # Extract fields
                record = {
                    "version": metadata.get("version"),
                    "model_type": metadata.get("model_type"),
                    "model_name": metadata.get("model_name"),
                    "file_path": str(
                        meta_file.with_suffix("").with_suffix("")
                    ),  # Remove .meta.json
                    "created_at": metadata.get("created_at"),
                    "serialized_at": metadata.get("serialized_at"),
                    "file_size_bytes": metadata.get("file_size_bytes"),
                    "checksum": metadata.get("checksum"),
                }

                records.append(record)

            except Exception as e:
                logger.warning("Failed to read metadata from %s: %s", meta_file, e)
                continue

        if not records:
            logger.info("No models found matching criteria")
            return pd.DataFrame()

        # Create DataFrame
        df = pd.DataFrame(records)

        # Sort by version (newest first)
        # Parse versions into tuples for proper sorting
        try:
            df["_version_tuple"] = df["version"].apply(self.parse_version)
            df = df.sort_values("_version_tuple", ascending=False)
            df = df.drop(columns=["_version_tuple"])
        except Exception as e:
            logger.warning("Failed to sort by version: %s", e)
            # Fallback: sort by serialized_at
            if "serialized_at" in df.columns:
                df = df.sort_values("serialized_at", ascending=False)

        logger.info("Found %d model versions", len(df))
        return df.reset_index(drop=True)

    def get_latest_version(
        self,
        models_dir: str | Path,
        model_type: str | None = None,
    ) -> str | None:
        """Get the latest version string for models in a directory.

        Args:
            models_dir: Directory containing saved models.
            model_type: Optional filter for specific model type.
                       If None, returns latest across all models.

        Returns:
            Latest version string (e.g., "1.2.3"), or None if no models found.

        Example:
            ```python
            vm = VersionManager()
            latest = vm.get_latest_version("models/", model_type="lgbm")
            print(f"Latest LGBM version: {latest}")  # "1.2.3"
            ```
        """
        history = self.get_version_history(models_dir, model_type=model_type)

        if history.empty:
            logger.info("No models found in %s", models_dir)
            return None

        latest_version = history.iloc[0]["version"]
        logger.info("Latest version: %s (model_type=%s)", latest_version, model_type)

        return latest_version
