"""Tests for SemanticVersion class."""

import pytest

from src.models.base.versioning import SemanticVersion


class TestSemanticVersionParsing:
    """Test version string parsing."""

    def test_parse_valid_version(self):
        """Test parsing valid version strings."""
        v = SemanticVersion("1.2.3")
        assert v.major == 1
        assert v.minor == 2
        assert v.patch == 3
        assert v.prerelease is None
        assert not v.is_prerelease()

    def test_parse_version_with_alpha(self):
        """Test parsing alpha pre-release version."""
        v = SemanticVersion("1.2.3-alpha")
        assert v.major == 1
        assert v.minor == 2
        assert v.patch == 3
        assert v.prerelease == "alpha"
        assert v.is_prerelease()

    def test_parse_version_with_beta_number(self):
        """Test parsing beta pre-release with version number."""
        v = SemanticVersion("2.0.0-beta.5")
        assert v.major == 2
        assert v.minor == 0
        assert v.patch == 0
        assert v.prerelease == "beta.5"
        assert v.is_prerelease()

    def test_parse_version_with_rc(self):
        """Test parsing release candidate version."""
        v = SemanticVersion("3.1.0-rc.2")
        assert v.major == 3
        assert v.minor == 1
        assert v.patch == 0
        assert v.prerelease == "rc.2"
        assert v.is_prerelease()

    def test_parse_zero_version(self):
        """Test parsing 0.0.0 version."""
        v = SemanticVersion("0.0.0")
        assert v.major == 0
        assert v.minor == 0
        assert v.patch == 0

    def test_parse_invalid_version_format(self):
        """Test that invalid version strings raise ValueError."""
        invalid_versions = [
            "1.2",  # Missing patch
            "1",  # Missing minor and patch
            "a.b.c",  # Non-numeric
            "1.2.3.4",  # Too many parts
            "1.2.3-gamma",  # Invalid pre-release type
            "1.2.3-",  # Empty pre-release
            "01.2.3",  # Leading zero in major
            "1.02.3",  # Leading zero in minor
            "1.2.03",  # Leading zero in patch
            "",  # Empty string
        ]
        for invalid in invalid_versions:
            with pytest.raises(ValueError, match="Invalid version string"):
                SemanticVersion(invalid)


class TestSemanticVersionComparison:
    """Test version comparison operators."""

    def test_equality(self):
        """Test version equality."""
        v1 = SemanticVersion("1.2.3")
        v2 = SemanticVersion("1.2.3")
        assert v1 == v2
        assert v1 == v2

    def test_inequality(self):
        """Test version inequality."""
        v1 = SemanticVersion("1.2.3")
        v2 = SemanticVersion("1.2.4")
        assert v1 != v2
        assert v1 != v2

    def test_less_than_major(self):
        """Test less than comparison on major version."""
        v1 = SemanticVersion("1.0.0")
        v2 = SemanticVersion("2.0.0")
        assert v1 < v2
        assert not v2 < v1

    def test_less_than_minor(self):
        """Test less than comparison on minor version."""
        v1 = SemanticVersion("1.1.0")
        v2 = SemanticVersion("1.2.0")
        assert v1 < v2
        assert not v2 < v1

    def test_less_than_patch(self):
        """Test less than comparison on patch version."""
        v1 = SemanticVersion("1.0.1")
        v2 = SemanticVersion("1.0.2")
        assert v1 < v2
        assert not v2 < v1

    def test_prerelease_less_than_release(self):
        """Test that pre-release versions are less than release versions."""
        v1 = SemanticVersion("1.0.0-beta.1")
        v2 = SemanticVersion("1.0.0")
        assert v1 < v2
        assert not v2 < v1

    def test_prerelease_ordering(self):
        """Test ordering among pre-release versions."""
        alpha = SemanticVersion("1.0.0-alpha")
        beta = SemanticVersion("1.0.0-beta")
        rc = SemanticVersion("1.0.0-rc")

        assert alpha < beta < rc

    def test_prerelease_number_ordering(self):
        """Test ordering among numbered pre-releases."""
        v1 = SemanticVersion("1.0.0-beta.1")
        v2 = SemanticVersion("1.0.0-beta.2")
        assert v1 < v2

    def test_greater_than(self):
        """Test greater than comparison."""
        v1 = SemanticVersion("2.0.0")
        v2 = SemanticVersion("1.0.0")
        assert v1 > v2
        assert not v2 > v1

    def test_less_than_or_equal(self):
        """Test less than or equal comparison."""
        v1 = SemanticVersion("1.0.0")
        v2 = SemanticVersion("1.0.0")
        v3 = SemanticVersion("2.0.0")
        assert v1 <= v2
        assert v1 <= v3
        assert not v3 <= v1

    def test_greater_than_or_equal(self):
        """Test greater than or equal comparison."""
        v1 = SemanticVersion("2.0.0")
        v2 = SemanticVersion("2.0.0")
        v3 = SemanticVersion("1.0.0")
        assert v1 >= v2
        assert v1 >= v3
        assert not v3 >= v1

    def test_comparison_type_error(self):
        """Test that comparing with non-SemanticVersion raises TypeError."""
        v = SemanticVersion("1.0.0")
        with pytest.raises(TypeError):
            v < "1.0.0"
        with pytest.raises(TypeError):
            v > 1
        with pytest.raises(TypeError):
            v == 1.0


class TestSemanticVersionConstraints:
    """Test version constraint satisfaction."""

    def test_satisfies_exact_match(self):
        """Test exact version matching."""
        v = SemanticVersion("1.2.3")
        assert v.satisfies("1.2.3")
        assert not v.satisfies("1.2.4")

    def test_satisfies_greater_than_or_equal(self):
        """Test >= constraint."""
        v = SemanticVersion("1.5.0")
        assert v.satisfies(">=1.0.0")
        assert v.satisfies(">=1.5.0")
        assert not v.satisfies(">=2.0.0")

    def test_satisfies_less_than(self):
        """Test < constraint."""
        v = SemanticVersion("1.5.0")
        assert v.satisfies("<2.0.0")
        assert not v.satisfies("<1.0.0")
        assert not v.satisfies("<1.5.0")

    def test_satisfies_greater_than(self):
        """Test > constraint."""
        v = SemanticVersion("1.5.0")
        assert v.satisfies(">1.0.0")
        assert not v.satisfies(">2.0.0")
        assert not v.satisfies(">1.5.0")

    def test_satisfies_less_than_or_equal(self):
        """Test <= constraint."""
        v = SemanticVersion("1.5.0")
        assert v.satisfies("<=2.0.0")
        assert v.satisfies("<=1.5.0")
        assert not v.satisfies("<=1.0.0")

    def test_satisfies_tilde_patch_level(self):
        """Test ~ (tilde) constraint for patch-level changes."""
        v1 = SemanticVersion("1.2.3")
        v2 = SemanticVersion("1.2.5")
        v3 = SemanticVersion("1.3.0")

        assert v1.satisfies("~1.2.0")
        assert v2.satisfies("~1.2.0")
        assert not v3.satisfies("~1.2.0")

    def test_satisfies_caret_minor_level(self):
        """Test ^ (caret) constraint for minor-level changes."""
        v1 = SemanticVersion("1.2.3")
        v2 = SemanticVersion("1.5.0")
        v3 = SemanticVersion("2.0.0")

        assert v1.satisfies("^1.0.0")
        assert v2.satisfies("^1.0.0")
        assert not v3.satisfies("^1.0.0")

    def test_satisfies_prerelease_in_constraints(self):
        """Test constraints with pre-release versions."""
        v = SemanticVersion("1.5.0-beta.1")
        assert v.satisfies(">=1.0.0")
        assert v.satisfies("<2.0.0")
        assert v.satisfies("^1.0.0")

    def test_satisfies_invalid_constraint(self):
        """Test that invalid constraints raise ValueError."""
        v = SemanticVersion("1.0.0")

        with pytest.raises(ValueError, match="Invalid.*constraint"):
            v.satisfies(">=invalid")

        with pytest.raises(ValueError, match="Invalid.*constraint"):
            v.satisfies("!1.0.0")


class TestSemanticVersionUtilities:
    """Test utility methods and representations."""

    def test_str_representation(self):
        """Test string representation."""
        v = SemanticVersion("1.2.3-beta.1")
        assert str(v) == "1.2.3-beta.1"

    def test_repr_representation(self):
        """Test repr representation."""
        v = SemanticVersion("1.2.3")
        assert repr(v) == "SemanticVersion('1.2.3')"

    def test_hash(self):
        """Test that versions are hashable."""
        v1 = SemanticVersion("1.2.3")
        v2 = SemanticVersion("1.2.3")
        v3 = SemanticVersion("1.2.4")

        # Same versions should have same hash
        assert hash(v1) == hash(v2)

        # Can be used in sets
        version_set = {v1, v2, v3}
        assert len(version_set) == 2

    def test_to_dict(self):
        """Test conversion to dictionary."""
        v = SemanticVersion("1.2.3-beta.1")
        d = v.to_dict()

        assert d["version"] == "1.2.3-beta.1"
        assert d["major"] == 1
        assert d["minor"] == 2
        assert d["patch"] == 3
        assert d["prerelease"] == "beta.1"
        assert d["is_prerelease"] is True

    def test_to_dict_stable(self):
        """Test to_dict for stable version."""
        v = SemanticVersion("2.0.0")
        d = v.to_dict()

        assert d["version"] == "2.0.0"
        assert d["prerelease"] is None
        assert d["is_prerelease"] is False


class TestSemanticVersionEdgeCases:
    """Test edge cases and special scenarios."""

    def test_multiple_digit_versions(self):
        """Test versions with multiple digits."""
        v = SemanticVersion("10.20.30")
        assert v.major == 10
        assert v.minor == 20
        assert v.patch == 30

    def test_comparison_chain(self):
        """Test chained comparisons."""
        v1 = SemanticVersion("1.0.0")
        v2 = SemanticVersion("1.5.0")
        v3 = SemanticVersion("2.0.0")

        assert v1 < v2 < v3
        assert v3 > v2 > v1

    def test_version_sorting(self):
        """Test that versions can be sorted."""
        versions = [
            SemanticVersion("2.0.0"),
            SemanticVersion("1.0.0"),
            SemanticVersion("1.5.0-beta"),
            SemanticVersion("1.5.0"),
            SemanticVersion("1.5.0-alpha"),
        ]

        sorted_versions = sorted(versions)

        assert str(sorted_versions[0]) == "1.0.0"
        assert str(sorted_versions[1]) == "1.5.0-alpha"
        assert str(sorted_versions[2]) == "1.5.0-beta"
        assert str(sorted_versions[3]) == "1.5.0"
        assert str(sorted_versions[4]) == "2.0.0"
