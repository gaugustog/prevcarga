"""Tests for ModelMetadata Pydantic model."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from src.models.base.metadata import ModelMetadata


class TestModelMetadataCreation:
    """Test ModelMetadata instance creation and validation."""

    def test_create_minimal_metadata(self):
        """Test creating metadata with only required fields."""
        metadata = ModelMetadata(
            model_id="test_model_001",
            model_name="test_model",
            model_version="1.0.0",
            model_class="TestModel",
            trained_at=datetime.now(UTC),
        )

        assert metadata.model_id == "test_model_001"
        assert metadata.model_name == "test_model"
        assert metadata.model_version == "1.0.0"
        assert metadata.model_class == "TestModel"
        assert metadata.trained_at.tzinfo is not None
        assert metadata.registered_at.tzinfo is not None
        assert metadata.supported_horizons == []
        assert metadata.feature_dependencies == []
        assert metadata.performance_metrics == {}
        assert metadata.training_config == {}
        assert metadata.custom_metadata == {}

    def test_create_full_metadata(self):
        """Test creating metadata with all fields."""
        trained_at = datetime.now(UTC)
        registered_at = datetime.now(UTC)

        metadata = ModelMetadata(
            model_id="lgbm_20241124_001",
            model_name="lgbm",
            model_version="1.2.3",
            model_class="LGBMModel",
            trained_at=trained_at,
            registered_at=registered_at,
            supported_horizons=[0, 1, 2, 3],
            feature_dependencies=["hour", "day_of_week", "temperature"],
            performance_metrics={"mape": 2.5, "rmse": 150.0},
            training_config={"max_depth": 7, "learning_rate": 0.1},
            custom_metadata={"data_source": "ONS", "preprocessing": "standard"},
        )

        assert metadata.model_id == "lgbm_20241124_001"
        assert metadata.supported_horizons == [0, 1, 2, 3]
        assert len(metadata.feature_dependencies) == 3
        assert metadata.performance_metrics["mape"] == 2.5
        assert metadata.training_config["max_depth"] == 7
        assert metadata.custom_metadata["data_source"] == "ONS"

    def test_auto_registered_at(self):
        """Test that registered_at is auto-set if not provided."""
        before = datetime.now(UTC)
        metadata = ModelMetadata(
            model_id="test_001",
            model_name="test",
            model_version="1.0.0",
            model_class="Test",
            trained_at=datetime.now(UTC),
        )
        after = datetime.now(UTC)

        assert before <= metadata.registered_at <= after


class TestModelMetadataValidation:
    """Test validation of ModelMetadata fields."""

    def test_validate_model_name_pattern(self):
        """Test that model_name must match pattern."""
        # Valid names
        valid_names = ["lgbm", "random_forest", "model_v2", "my_model123"]
        for name in valid_names:
            metadata = ModelMetadata(
                model_id="test_001",
                model_name=name,
                model_version="1.0.0",
                model_class="Test",
                trained_at=datetime.now(UTC),
            )
            assert metadata.model_name == name

        # Invalid names
        invalid_names = ["LGBM", "Model-1", "model with spaces", "1model", ""]
        for name in invalid_names:
            with pytest.raises(ValidationError):
                ModelMetadata(
                    model_id="test_001",
                    model_name=name,
                    model_version="1.0.0",
                    model_class="Test",
                    trained_at=datetime.now(UTC),
                )

    def test_validate_version_format(self):
        """Test that model_version must be valid semantic version."""
        # Valid versions
        valid_versions = ["1.0.0", "2.1.3", "0.0.1", "1.0.0-alpha", "2.1.0-beta.5"]
        for version in valid_versions:
            metadata = ModelMetadata(
                model_id="test_001",
                model_name="test",
                model_version=version,
                model_class="Test",
                trained_at=datetime.now(UTC),
            )
            assert metadata.model_version == version

        # Invalid versions
        invalid_versions = ["1.0", "v1.0.0", "1.0.0.0", "invalid"]
        for version in invalid_versions:
            with pytest.raises(ValidationError):
                ModelMetadata(
                    model_id="test_001",
                    model_name="test",
                    model_version=version,
                    model_class="Test",
                    trained_at=datetime.now(UTC),
                )

    def test_validate_horizons_range(self):
        """Test that horizons must be in range 0-8."""
        # Valid horizons
        metadata = ModelMetadata(
            model_id="test_001",
            model_name="test",
            model_version="1.0.0",
            model_class="Test",
            trained_at=datetime.now(UTC),
            supported_horizons=[0, 1, 2, 3, 4, 5, 6, 7, 8],
        )
        assert metadata.supported_horizons == [0, 1, 2, 3, 4, 5, 6, 7, 8]

        # Invalid horizons (negative)
        with pytest.raises(ValidationError, match="Invalid horizons"):
            ModelMetadata(
                model_id="test_001",
                model_name="test",
                model_version="1.0.0",
                model_class="Test",
                trained_at=datetime.now(UTC),
                supported_horizons=[-1, 0, 1],
            )

        # Invalid horizons (> 8)
        with pytest.raises(ValidationError, match="Invalid horizons"):
            ModelMetadata(
                model_id="test_001",
                model_name="test",
                model_version="1.0.0",
                model_class="Test",
                trained_at=datetime.now(UTC),
                supported_horizons=[0, 1, 9],
            )

    def test_validate_horizons_duplicates_removed(self):
        """Test that duplicate horizons are removed and sorted."""
        metadata = ModelMetadata(
            model_id="test_001",
            model_name="test",
            model_version="1.0.0",
            model_class="Test",
            trained_at=datetime.now(UTC),
            supported_horizons=[3, 1, 2, 1, 3, 0],
        )
        # Should be deduplicated and sorted
        assert metadata.supported_horizons == [0, 1, 2, 3]

    def test_validate_metrics_finite(self):
        """Test that metric values must be finite numbers."""
        import math

        # Valid metrics
        metadata = ModelMetadata(
            model_id="test_001",
            model_name="test",
            model_version="1.0.0",
            model_class="Test",
            trained_at=datetime.now(UTC),
            performance_metrics={"mape": 2.5, "rmse": 150.0, "mae": 100},
        )
        assert metadata.performance_metrics["mape"] == 2.5

        # Invalid metrics (inf)
        with pytest.raises(ValidationError, match="non-finite"):
            ModelMetadata(
                model_id="test_001",
                model_name="test",
                model_version="1.0.0",
                model_class="Test",
                trained_at=datetime.now(UTC),
                performance_metrics={"mape": math.inf},
            )

        # Invalid metrics (nan)
        with pytest.raises(ValidationError, match="non-finite"):
            ModelMetadata(
                model_id="test_001",
                model_name="test",
                model_version="1.0.0",
                model_class="Test",
                trained_at=datetime.now(UTC),
                performance_metrics={"mape": math.nan},
            )

    def test_validate_timezone_aware(self):
        """Test that datetimes are timezone-aware."""
        # Naive datetime should be converted to UTC
        naive_dt = datetime(2024, 11, 24, 12, 0, 0)
        metadata = ModelMetadata(
            model_id="test_001",
            model_name="test",
            model_version="1.0.0",
            model_class="Test",
            trained_at=naive_dt,
        )
        assert metadata.trained_at.tzinfo is not None

    def test_required_fields(self):
        """Test that required fields are enforced."""
        # Missing model_id
        with pytest.raises(ValidationError):
            ModelMetadata(
                model_name="test",
                model_version="1.0.0",
                model_class="Test",
                trained_at=datetime.now(UTC),
            )

        # Missing model_name
        with pytest.raises(ValidationError):
            ModelMetadata(
                model_id="test_001",
                model_version="1.0.0",
                model_class="Test",
                trained_at=datetime.now(UTC),
            )


class TestModelMetadataMethods:
    """Test ModelMetadata methods."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        trained_at = datetime(2024, 11, 24, 12, 0, 0, tzinfo=UTC)
        metadata = ModelMetadata(
            model_id="test_001",
            model_name="test",
            model_version="1.0.0",
            model_class="Test",
            trained_at=trained_at,
            supported_horizons=[0, 1, 2],
            performance_metrics={"mape": 2.5},
        )

        d = metadata.to_dict()

        assert d["model_id"] == "test_001"
        assert d["model_name"] == "test"
        assert d["supported_horizons"] == [0, 1, 2]
        assert isinstance(d["trained_at"], str)  # Serialized to ISO format
        assert isinstance(d["registered_at"], str)

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "model_id": "test_001",
            "model_name": "test",
            "model_version": "1.0.0",
            "model_class": "Test",
            "trained_at": "2024-11-24T12:00:00+00:00",
            "registered_at": "2024-11-24T12:00:00+00:00",
            "supported_horizons": [0, 1, 2],
            "feature_dependencies": ["a", "b"],
            "performance_metrics": {"mape": 2.5},
            "training_config": {"lr": 0.1},
            "custom_metadata": {"key": "value"},
        }

        metadata = ModelMetadata.from_dict(data)

        assert metadata.model_id == "test_001"
        assert metadata.model_name == "test"
        assert isinstance(metadata.trained_at, datetime)
        assert metadata.supported_horizons == [0, 1, 2]

    def test_get_version_info(self):
        """Test getting version information."""
        metadata = ModelMetadata(
            model_id="test_001",
            model_name="test",
            model_version="1.2.3-beta.1",
            model_class="Test",
            trained_at=datetime.now(UTC),
        )

        version_info = metadata.get_version_info()

        assert version_info["major"] == 1
        assert version_info["minor"] == 2
        assert version_info["patch"] == 3
        assert version_info["prerelease"] == "beta.1"
        assert version_info["is_prerelease"] is True

    def test_get_best_metric(self):
        """Test getting specific metric value."""
        metadata = ModelMetadata(
            model_id="test_001",
            model_name="test",
            model_version="1.0.0",
            model_class="Test",
            trained_at=datetime.now(UTC),
            performance_metrics={"mape": 2.5, "rmse": 150.0},
        )

        assert metadata.get_best_metric("mape") == 2.5
        assert metadata.get_best_metric("rmse", lower_is_better=False) == 150.0
        assert metadata.get_best_metric("nonexistent") is None

    def test_update_metrics(self):
        """Test updating performance metrics."""
        metadata = ModelMetadata(
            model_id="test_001",
            model_name="test",
            model_version="1.0.0",
            model_class="Test",
            trained_at=datetime.now(UTC),
            performance_metrics={"mape": 2.5},
        )

        metadata.update_metrics({"rmse": 150.0, "mae": 100.0})

        assert metadata.performance_metrics["mape"] == 2.5
        assert metadata.performance_metrics["rmse"] == 150.0
        assert metadata.performance_metrics["mae"] == 100.0

        # Update existing metric
        metadata.update_metrics({"mape": 2.0})
        assert metadata.performance_metrics["mape"] == 2.0

    def test_update_metrics_validation(self):
        """Test that update_metrics validates new metrics."""
        import math

        metadata = ModelMetadata(
            model_id="test_001",
            model_name="test",
            model_version="1.0.0",
            model_class="Test",
            trained_at=datetime.now(UTC),
        )

        # Invalid metric value
        with pytest.raises(ValueError, match="non-finite"):
            metadata.update_metrics({"mape": math.inf})


class TestModelMetadataRepresentation:
    """Test string representations."""

    def test_str_representation(self):
        """Test __str__ method."""
        metadata = ModelMetadata(
            model_id="lgbm_20241124_001",
            model_name="lgbm",
            model_version="1.2.3",
            model_class="LGBMModel",
            trained_at=datetime.now(UTC),
        )

        s = str(metadata)
        assert "lgbm" in s
        assert "1.2.3" in s
        assert "lgbm_20241124_001" in s

    def test_repr_representation(self):
        """Test __repr__ method."""
        metadata = ModelMetadata(
            model_id="test_001",
            model_name="test",
            model_version="1.0.0",
            model_class="Test",
            trained_at=datetime.now(UTC),
            supported_horizons=[0, 1, 2],
        )

        r = repr(metadata)
        assert "ModelMetadata" in r
        assert "test_001" in r
        assert "test" in r
        assert "1.0.0" in r
        assert "[0, 1, 2]" in r


class TestModelMetadataSerialization:
    """Test serialization and deserialization."""

    def test_roundtrip_serialization(self):
        """Test that to_dict/from_dict roundtrip preserves data."""
        original = ModelMetadata(
            model_id="test_001",
            model_name="test",
            model_version="1.2.3-beta.1",
            model_class="TestModel",
            trained_at=datetime(2024, 11, 24, 12, 0, 0, tzinfo=UTC),
            supported_horizons=[0, 1, 2, 3],
            feature_dependencies=["a", "b", "c"],
            performance_metrics={"mape": 2.5, "rmse": 150.0},
            training_config={"lr": 0.1, "epochs": 100},
            custom_metadata={"source": "test"},
        )

        # Serialize to dict
        data = original.to_dict()

        # Deserialize from dict
        restored = ModelMetadata.from_dict(data)

        # Verify all fields match
        assert restored.model_id == original.model_id
        assert restored.model_name == original.model_name
        assert restored.model_version == original.model_version
        assert restored.model_class == original.model_class
        assert restored.supported_horizons == original.supported_horizons
        assert restored.feature_dependencies == original.feature_dependencies
        assert restored.performance_metrics == original.performance_metrics
        assert restored.training_config == original.training_config
        assert restored.custom_metadata == original.custom_metadata

    def test_pydantic_json_serialization(self):
        """Test Pydantic's built-in JSON serialization."""
        metadata = ModelMetadata(
            model_id="test_001",
            model_name="test",
            model_version="1.0.0",
            model_class="Test",
            trained_at=datetime.now(UTC),
            supported_horizons=[0, 1, 2],
        )

        # Serialize to JSON string
        json_str = metadata.model_dump_json()
        assert isinstance(json_str, str)
        assert "test_001" in json_str

        # Deserialize from dict (Pydantic's model_dump)
        data = metadata.model_dump()
        restored = ModelMetadata(**data)
        assert restored.model_id == metadata.model_id
