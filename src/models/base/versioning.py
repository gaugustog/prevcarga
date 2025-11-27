"""Semantic versioning implementation for model version management.

This module provides a SemanticVersion class that parses, compares, and validates
version strings following the Semantic Versioning 2.0.0 specification with support
for pre-release versions (alpha, beta, rc).

Example:
    ```python
    from src.models.base.versioning import SemanticVersion

    # Parse version strings
    v1 = SemanticVersion("1.2.3")
    v2 = SemanticVersion("1.2.4-beta.1")

    # Compare versions
    assert v1 < v2
    assert v2.is_prerelease()

    # Check version constraints
    v1.satisfies(">=1.0.0")  # True
    v1.satisfies("~1.2.0")   # True (compatible with 1.2.x)
    v1.satisfies("^1.0.0")   # True (compatible with 1.x.x)
    ```
"""

import re
from typing import Any

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Regex pattern for semantic version parsing
# Matches: MAJOR.MINOR.PATCH[-PRERELEASE]
VERSION_PATTERN = re.compile(
    r"^(?P<major>0|[1-9]\d*)\."
    r"(?P<minor>0|[1-9]\d*)\."
    r"(?P<patch>0|[1-9]\d*)"
    r"(?:-(?P<prerelease>(?:alpha|beta|rc)(?:\.\d+)?))?"
    r"$"
)


class SemanticVersion:
    """Semantic versioning implementation with comparison and constraint checking.

    This class parses version strings following semantic versioning format
    (MAJOR.MINOR.PATCH[-PRERELEASE]) and provides comparison operators.

    Attributes:
        major: Major version number (breaking changes).
        minor: Minor version number (backward-compatible features).
        patch: Patch version number (backward-compatible fixes).
        prerelease: Pre-release version string (alpha, beta, rc) or None.
    """

    def __init__(self, version_string: str) -> None:
        """Initialize semantic version from version string.

        Args:
            version_string: Version string in format "MAJOR.MINOR.PATCH[-PRERELEASE]".
                            Pre-release format: alpha, beta, rc (optionally with .N).
                            Examples: "1.0.0", "2.1.3-beta.1", "1.5.0-rc".

        Raises:
            ValueError: If version string format is invalid.
        """
        self._version_string = version_string
        self._parse_version(version_string)
        logger.debug("Parsed version: %s", self)

    def _parse_version(self, version_string: str) -> None:
        """Parse version string and extract components.

        Args:
            version_string: Version string to parse.

        Raises:
            ValueError: If version string format is invalid.
        """
        match = VERSION_PATTERN.match(version_string)
        if not match:
            msg = (
                f"Invalid version string: '{version_string}'. "
                "Expected format: MAJOR.MINOR.PATCH[-PRERELEASE] "
                "(e.g., '1.0.0', '2.1.3-beta.1')"
            )
            raise ValueError(msg)

        self.major = int(match.group("major"))
        self.minor = int(match.group("minor"))
        self.patch = int(match.group("patch"))
        self.prerelease = match.group("prerelease")

    def is_prerelease(self) -> bool:
        """Check if this is a pre-release version.

        Returns:
            True if version has a pre-release identifier, False otherwise.
        """
        return self.prerelease is not None

    def _prerelease_priority(self) -> tuple[int, int]:
        """Get pre-release priority for comparison.

        Returns:
            Tuple of (priority, number) where priority is:
            - 0: alpha
            - 1: beta
            - 2: rc
            - 3: stable (no pre-release)
            Number is the pre-release version number (0 if not specified).
        """
        if not self.prerelease:
            return (3, 0)

        # Parse pre-release string (e.g., "alpha", "beta.1", "rc.2")
        parts = self.prerelease.split(".")
        prerelease_type = parts[0]
        prerelease_num = int(parts[1]) if len(parts) > 1 else 0

        priority_map = {"alpha": 0, "beta": 1, "rc": 2}
        priority = priority_map.get(prerelease_type, 0)

        return (priority, prerelease_num)

    def __eq__(self, other: object) -> bool:
        """Check if two versions are equal.

        Args:
            other: Another SemanticVersion instance.

        Returns:
            True if versions are equal, False otherwise.

        Raises:
            TypeError: If other is not a SemanticVersion instance.
        """
        if not isinstance(other, SemanticVersion):
            msg = f"Cannot compare SemanticVersion with {type(other)}"
            raise TypeError(msg)

        return (
            self.major == other.major
            and self.minor == other.minor
            and self.patch == other.patch
            and self.prerelease == other.prerelease
        )

    def __lt__(self, other: object) -> bool:
        """Check if this version is less than another.

        Args:
            other: Another SemanticVersion instance.

        Returns:
            True if this version is less than other, False otherwise.

        Raises:
            TypeError: If other is not a SemanticVersion instance.
        """
        if not isinstance(other, SemanticVersion):
            msg = f"Cannot compare SemanticVersion with {type(other)}"
            raise TypeError(msg)

        # Compare major.minor.patch first
        if (self.major, self.minor, self.patch) != (other.major, other.minor, other.patch):
            return (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)

        # If base versions are equal, compare pre-release
        # Pre-release versions are less than release versions
        self_prerelease = self._prerelease_priority()
        other_prerelease = other._prerelease_priority()

        return self_prerelease < other_prerelease

    def __le__(self, other: object) -> bool:
        """Check if this version is less than or equal to another.

        Args:
            other: Another SemanticVersion instance.

        Returns:
            True if this version is less than or equal to other.

        Raises:
            TypeError: If other is not a SemanticVersion instance.
        """
        return self == other or self < other

    def __gt__(self, other: object) -> bool:
        """Check if this version is greater than another.

        Args:
            other: Another SemanticVersion instance.

        Returns:
            True if this version is greater than other, False otherwise.

        Raises:
            TypeError: If other is not a SemanticVersion instance.
        """
        if not isinstance(other, SemanticVersion):
            msg = f"Cannot compare SemanticVersion with {type(other)}"
            raise TypeError(msg)

        return not self <= other

    def __ge__(self, other: object) -> bool:
        """Check if this version is greater than or equal to another.

        Args:
            other: Another SemanticVersion instance.

        Returns:
            True if this version is greater than or equal to other.

        Raises:
            TypeError: If other is not a SemanticVersion instance.
        """
        return self == other or self > other

    def __str__(self) -> str:
        """Return string representation of version.

        Returns:
            Version string in format "MAJOR.MINOR.PATCH[-PRERELEASE]".
        """
        return self._version_string

    def __repr__(self) -> str:
        """Return detailed string representation.

        Returns:
            String representation for debugging.
        """
        return f"SemanticVersion('{self._version_string}')"

    def __hash__(self) -> int:
        """Return hash of version for use in sets and dicts.

        Returns:
            Hash value based on version components.
        """
        return hash((self.major, self.minor, self.patch, self.prerelease))

    def satisfies(self, constraint: str) -> bool:
        """Check if version satisfies a version constraint.

        Supported constraint formats:
        - Exact: "1.2.3"
        - Greater than or equal: ">=1.0.0"
        - Less than: "<2.0.0"
        - Greater than: ">1.0.0"
        - Less than or equal: "<=1.5.0"
        - Tilde (patch-level changes): "~1.2.0" (matches 1.2.x)
        - Caret (minor-level changes): "^1.0.0" (matches 1.x.x)

        Args:
            constraint: Version constraint string.

        Returns:
            True if this version satisfies the constraint, False otherwise.

        Raises:
            ValueError: If constraint format is invalid.
        """
        constraint = constraint.strip()

        # Exact match
        if not any(c in constraint for c in [">=", "<=", ">", "<", "~", "^"]):
            try:
                return self == SemanticVersion(constraint)
            except ValueError as e:
                msg = f"Invalid constraint version: {constraint}"
                raise ValueError(msg) from e

        # Tilde constraint (~): allows patch-level changes
        # ~1.2.3 means >=1.2.3 <1.3.0
        if constraint.startswith("~"):
            version_str = constraint[1:].strip()
            try:
                target = SemanticVersion(version_str)
            except ValueError as e:
                msg = f"Invalid tilde constraint version: {version_str}"
                raise ValueError(msg) from e

            return self.major == target.major and self.minor == target.minor and self >= target

        # Caret constraint (^): allows minor-level changes
        # ^1.2.3 means >=1.2.3 <2.0.0
        if constraint.startswith("^"):
            version_str = constraint[1:].strip()
            try:
                target = SemanticVersion(version_str)
            except ValueError as e:
                msg = f"Invalid caret constraint version: {version_str}"
                raise ValueError(msg) from e

            return self.major == target.major and self >= target

        # Comparison operators
        if constraint.startswith(">="):
            version_str = constraint[2:].strip()
            try:
                return self >= SemanticVersion(version_str)
            except ValueError as e:
                msg = f"Invalid >= constraint version: {version_str}"
                raise ValueError(msg) from e

        if constraint.startswith("<="):
            version_str = constraint[2:].strip()
            try:
                return self <= SemanticVersion(version_str)
            except ValueError as e:
                msg = f"Invalid <= constraint version: {version_str}"
                raise ValueError(msg) from e

        if constraint.startswith(">"):
            version_str = constraint[1:].strip()
            try:
                return self > SemanticVersion(version_str)
            except ValueError as e:
                msg = f"Invalid > constraint version: {version_str}"
                raise ValueError(msg) from e

        if constraint.startswith("<"):
            version_str = constraint[1:].strip()
            try:
                return self < SemanticVersion(version_str)
            except ValueError as e:
                msg = f"Invalid < constraint version: {version_str}"
                raise ValueError(msg) from e

        msg = f"Unsupported constraint format: {constraint}"
        raise ValueError(msg)

    def to_dict(self) -> dict[str, Any]:
        """Convert version to dictionary representation.

        Returns:
            Dictionary with version components.
        """
        return {
            "version": self._version_string,
            "major": self.major,
            "minor": self.minor,
            "patch": self.patch,
            "prerelease": self.prerelease,
            "is_prerelease": self.is_prerelease(),
        }
