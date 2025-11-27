"""Tests for VersionManager class."""

import json
from datetime import UTC, datetime

import pandas as pd
import pytest

from src.models.serialization.version_manager import VersionManager


@pytest.fixture
def version_manager():
    """Create VersionManager instance."""
    return VersionManager()


@pytest.fixture
def temp_models_dir(tmp_path):
    """Create temporary directory with sample model metadata files."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()

    # Create sample metadata files for different versions
    versions = [
        ("1.0.0", "lgbm"),
        ("1.0.1", "lgbm"),
        ("1.1.0", "lgbm"),
        ("2.0.0", "lgbm"),
        ("1.0.0", "random_forest"),
        ("1.2.0", "random_forest"),
    ]

    for version, model_type in versions:
        # Create model file
        model_file = models_dir / f"{model_type}_v{version}.pkl"
        model_file.write_text("dummy model data")

        # Create metadata file
        metadata = {
            "version": version,
            "model_type": model_type,
            "model_name": f"{model_type}_model",
            "created_at": datetime.now(UTC).isoformat(),
            "serialized_at": datetime.now(UTC).isoformat(),
            "file_size_bytes": len("dummy model data"),
            "checksum": "a" * 64,
        }
        meta_file = models_dir / f"{model_type}_v{version}.pkl.meta.json"
        with meta_file.open("w") as f:
            json.dump(metadata, f)

    return models_dir


class TestVersionManagerParseVersion:
    """Test parse_version() method."""

    def test_parse_version_basic(self, version_manager):
        """Test parsing basic semantic version."""
        result = version_manager.parse_version("1.2.3")
        assert result == (1, 2, 3)

    def test_parse_version_zeros(self, version_manager):
        """Test parsing version with zeros."""
        result = version_manager.parse_version("0.0.1")
        assert result == (0, 0, 1)

        result = version_manager.parse_version("1.0.0")
        assert result == (1, 0, 0)

    def test_parse_version_large_numbers(self, version_manager):
        """Test parsing version with large numbers."""
        result = version_manager.parse_version("10.20.30")
        assert result == (10, 20, 30)

    def test_parse_version_prerelease_ignored(self, version_manager):
        """Test that pre-release suffix is ignored in tuple."""
        result = version_manager.parse_version("1.2.3-beta.1")
        assert result == (1, 2, 3)

        result = version_manager.parse_version("2.0.0-alpha")
        assert result == (2, 0, 0)

    def test_parse_version_invalid_format(self, version_manager):
        """Test parsing invalid version formats raises ValueError."""
        invalid_versions = [
            "1.2",
            "1.2.3.4",
            "v1.2.3",
            "1.2.x",
            "invalid",
            "",
        ]

        for version in invalid_versions:
            with pytest.raises(ValueError, match="Failed to parse version"):
                version_manager.parse_version(version)


class TestVersionManagerCompareVersions:
    """Test compare_versions() method."""

    def test_compare_versions_less_than(self, version_manager):
        """Test comparing versions where v1 < v2."""
        assert version_manager.compare_versions("1.0.0", "1.0.1") == -1
        assert version_manager.compare_versions("1.0.0", "1.1.0") == -1
        assert version_manager.compare_versions("1.0.0", "2.0.0") == -1
        assert version_manager.compare_versions("1.2.3", "1.2.4") == -1

    def test_compare_versions_greater_than(self, version_manager):
        """Test comparing versions where v1 > v2."""
        assert version_manager.compare_versions("1.0.1", "1.0.0") == 1
        assert version_manager.compare_versions("1.1.0", "1.0.0") == 1
        assert version_manager.compare_versions("2.0.0", "1.0.0") == 1
        assert version_manager.compare_versions("1.2.4", "1.2.3") == 1

    def test_compare_versions_equal(self, version_manager):
        """Test comparing equal versions."""
        assert version_manager.compare_versions("1.0.0", "1.0.0") == 0
        assert version_manager.compare_versions("1.2.3", "1.2.3") == 0
        assert version_manager.compare_versions("10.20.30", "10.20.30") == 0

    def test_compare_versions_with_tuples(self, version_manager):
        """Test comparing version tuples directly."""
        assert version_manager.compare_versions((1, 0, 0), (1, 0, 1)) == -1
        assert version_manager.compare_versions((1, 0, 1), (1, 0, 0)) == 1
        assert version_manager.compare_versions((1, 2, 3), (1, 2, 3)) == 0

    def test_compare_versions_mixed_types(self, version_manager):
        """Test comparing mixed string and tuple versions."""
        assert version_manager.compare_versions("1.0.0", (1, 0, 1)) == -1
        assert version_manager.compare_versions((1, 0, 1), "1.0.0") == 1
        assert version_manager.compare_versions("1.2.3", (1, 2, 3)) == 0

    def test_compare_versions_invalid_string(self, version_manager):
        """Test comparing invalid version strings raises ValueError."""
        with pytest.raises(ValueError):
            version_manager.compare_versions("invalid", "1.0.0")


class TestVersionManagerIncrementVersion:
    """Test increment_version() method."""

    def test_increment_version_patch(self, version_manager):
        """Test incrementing patch version."""
        result = version_manager.increment_version("1.2.3", bump="patch")
        assert result == "1.2.4"

        result = version_manager.increment_version("1.0.0", bump="patch")
        assert result == "1.0.1"

    def test_increment_version_minor(self, version_manager):
        """Test incrementing minor version."""
        result = version_manager.increment_version("1.2.3", bump="minor")
        assert result == "1.3.0"

        result = version_manager.increment_version("1.0.0", bump="minor")
        assert result == "1.1.0"

    def test_increment_version_major(self, version_manager):
        """Test incrementing major version."""
        result = version_manager.increment_version("1.2.3", bump="major")
        assert result == "2.0.0"

        result = version_manager.increment_version("1.0.0", bump="major")
        assert result == "2.0.0"

    def test_increment_version_default_is_patch(self, version_manager):
        """Test that default bump type is patch."""
        result = version_manager.increment_version("1.2.3")
        assert result == "1.2.4"

    def test_increment_version_invalid_bump_type(self, version_manager):
        """Test that invalid bump type raises ValueError."""
        with pytest.raises(ValueError, match="Invalid bump type"):
            version_manager.increment_version("1.2.3", bump="invalid")

    def test_increment_version_invalid_current_version(self, version_manager):
        """Test that invalid current version raises ValueError."""
        with pytest.raises(ValueError):
            version_manager.increment_version("invalid", bump="patch")


class TestVersionManagerCreateVersionTag:
    """Test create_version_tag() method."""

    def test_create_version_tag_basic(self, version_manager):
        """Test creating basic version tag."""
        result = version_manager.create_version_tag("lgbm", "1.2.3")
        assert result == "lgbm_v1.2.3"

    def test_create_version_tag_different_models(self, version_manager):
        """Test creating tags for different model types."""
        result = version_manager.create_version_tag("random_forest", "2.0.0")
        assert result == "random_forest_v2.0.0"

        result = version_manager.create_version_tag("my_model", "0.1.0")
        assert result == "my_model_v0.1.0"

    def test_create_version_tag_validates_version(self, version_manager):
        """Test that create_version_tag validates version format."""
        with pytest.raises(ValueError):
            version_manager.create_version_tag("lgbm", "invalid")


class TestVersionManagerGetVersionHistory:
    """Test get_version_history() method."""

    def test_get_version_history_all_models(self, version_manager, temp_models_dir):
        """Test getting version history for all models."""
        history = version_manager.get_version_history(temp_models_dir)

        # Should have 6 models total
        assert len(history) == 6
        assert isinstance(history, pd.DataFrame)

        # Check columns
        expected_columns = [
            "version",
            "model_type",
            "model_name",
            "file_path",
            "created_at",
            "serialized_at",
            "file_size_bytes",
            "checksum",
        ]
        for col in expected_columns:
            assert col in history.columns

    def test_get_version_history_filtered_by_model_type(self, version_manager, temp_models_dir):
        """Test getting version history filtered by model type."""
        # Get LGBM models only
        history = version_manager.get_version_history(
            temp_models_dir,
            model_type="lgbm",
        )

        assert len(history) == 4
        assert all(history["model_type"] == "lgbm")

        # Get random_forest models only
        history = version_manager.get_version_history(
            temp_models_dir,
            model_type="random_forest",
        )

        assert len(history) == 2
        assert all(history["model_type"] == "random_forest")

    def test_get_version_history_sorted_by_version(self, version_manager, temp_models_dir):
        """Test that version history is sorted by version (newest first)."""
        history = version_manager.get_version_history(
            temp_models_dir,
            model_type="lgbm",
        )

        # First row should be newest version (2.0.0)
        assert history.iloc[0]["version"] == "2.0.0"

        # Last row should be oldest version (1.0.0)
        assert history.iloc[-1]["version"] == "1.0.0"

        # Check order is descending
        versions = history["version"].tolist()
        assert versions == ["2.0.0", "1.1.0", "1.0.1", "1.0.0"]

    def test_get_version_history_nonexistent_directory(self, version_manager, tmp_path):
        """Test getting version history from non-existent directory."""
        nonexistent_dir = tmp_path / "nonexistent"

        history = version_manager.get_version_history(nonexistent_dir)

        # Should return empty DataFrame
        assert isinstance(history, pd.DataFrame)
        assert len(history) == 0

    def test_get_version_history_empty_directory(self, version_manager, tmp_path):
        """Test getting version history from empty directory."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        history = version_manager.get_version_history(empty_dir)

        # Should return empty DataFrame
        assert isinstance(history, pd.DataFrame)
        assert len(history) == 0

    def test_get_version_history_no_matching_models(self, version_manager, temp_models_dir):
        """Test getting version history with no matching model type."""
        history = version_manager.get_version_history(
            temp_models_dir,
            model_type="nonexistent_model",
        )

        # Should return empty DataFrame
        assert isinstance(history, pd.DataFrame)
        assert len(history) == 0


class TestVersionManagerGetLatestVersion:
    """Test get_latest_version() method."""

    def test_get_latest_version_all_models(self, version_manager, temp_models_dir):
        """Test getting latest version across all models."""
        latest = version_manager.get_latest_version(temp_models_dir)

        # Latest across all models should be 2.0.0 (lgbm)
        assert latest == "2.0.0"

    def test_get_latest_version_filtered_by_model_type(self, version_manager, temp_models_dir):
        """Test getting latest version for specific model type."""
        # Latest LGBM version
        latest = version_manager.get_latest_version(
            temp_models_dir,
            model_type="lgbm",
        )
        assert latest == "2.0.0"

        # Latest random_forest version
        latest = version_manager.get_latest_version(
            temp_models_dir,
            model_type="random_forest",
        )
        assert latest == "1.2.0"

    def test_get_latest_version_no_models(self, version_manager, tmp_path):
        """Test getting latest version when no models exist."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        latest = version_manager.get_latest_version(empty_dir)

        # Should return None
        assert latest is None

    def test_get_latest_version_no_matching_model_type(self, version_manager, temp_models_dir):
        """Test getting latest version with no matching model type."""
        latest = version_manager.get_latest_version(
            temp_models_dir,
            model_type="nonexistent_model",
        )

        # Should return None
        assert latest is None


class TestVersionManagerIntegration:
    """Integration tests for VersionManager."""

    def test_version_workflow(self, version_manager):
        """Test typical version management workflow."""
        # Start with version 1.0.0
        current_version = "1.0.0"

        # Parse version
        parsed = version_manager.parse_version(current_version)
        assert parsed == (1, 0, 0)

        # Increment patch for bug fix
        new_version = version_manager.increment_version(current_version, bump="patch")
        assert new_version == "1.0.1"

        # Compare versions
        comparison = version_manager.compare_versions(current_version, new_version)
        assert comparison == -1  # current < new

        # Increment minor for new feature
        new_version = version_manager.increment_version(current_version, bump="minor")
        assert new_version == "1.1.0"

        # Create version tag
        tag = version_manager.create_version_tag("lgbm", new_version)
        assert tag == "lgbm_v1.1.0"

    def test_version_history_workflow(self, version_manager, temp_models_dir):
        """Test typical version history workflow."""
        # Get latest version for model type
        latest = version_manager.get_latest_version(
            temp_models_dir,
            model_type="lgbm",
        )
        assert latest == "2.0.0"

        # Increment for new release
        new_version = version_manager.increment_version(latest, bump="minor")
        assert new_version == "2.1.0"

        # Get full version history
        history = version_manager.get_version_history(
            temp_models_dir,
            model_type="lgbm",
        )
        assert len(history) == 4
        assert history.iloc[0]["version"] == "2.0.0"

    def test_comparing_versions_from_history(self, version_manager, temp_models_dir):
        """Test comparing versions extracted from history."""
        history = version_manager.get_version_history(
            temp_models_dir,
            model_type="lgbm",
        )

        # Get first two versions
        v1 = history.iloc[0]["version"]
        v2 = history.iloc[1]["version"]

        # v1 should be newer than v2 (history is sorted newest first)
        comparison = version_manager.compare_versions(v1, v2)
        assert comparison == 1  # v1 > v2
