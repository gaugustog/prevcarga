"""Tests for serialization metadata schemas."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from src.models.serialization.schemas import (
    SerializedModelMetadata,
    TrainingMetadata,
)


class TestTrainingMetadata:
    """Test TrainingMetadata Pydantic model."""

    def test_create_minimal_training_metadata(self):
        """Test creating TrainingMetadata with only required fields."""
        metadata = TrainingMetadata(
            n_samples_train=1000,
            n_samples_val=200,
            n_features=10,
            training_duration_seconds=60.5,
        )

        assert metadata.n_samples_train == 1000
        assert metadata.n_samples_val == 200
        assert metadata.n_features == 10
        assert metadata.training_duration_seconds == 60.5
        assert metadata.feature_names == []
        assert metadata.training_config == {}
        assert metadata.preprocessing_steps == []
        assert metadata.performance_metrics == {}
        assert metadata.cv_scores == {}
        assert metadata.hyperparameter_optimization is False
        assert metadata.best_hyperparams == {}

    def test_create_full_training_metadata(self):
        """Test creating TrainingMetadata with all fields."""
        metadata = TrainingMetadata(
            n_samples_train=10000,
            n_samples_val=2000,
            n_features=50,
            feature_names=["hour", "day_of_week", "temperature"],
            training_config={"learning_rate": 0.1, "max_depth": 7},
            preprocessing_steps=["standardization", "outlier_removal"],
            training_duration_seconds=120.5,
            performance_metrics={"mape": 2.5, "rmse": 150.0, "mae": 100.0},
            cv_scores={"mean_cv_mape": 2.8, "std_cv_mape": 0.3},
            hyperparameter_optimization=True,
            best_hyperparams={"learning_rate": 0.08, "max_depth": 8},
        )

        assert metadata.n_samples_train == 10000
        assert metadata.n_samples_val == 2000
        assert metadata.n_features == 50
        assert len(metadata.feature_names) == 3
        assert "hour" in metadata.feature_names
        assert metadata.training_config["learning_rate"] == 0.1
        assert len(metadata.preprocessing_steps) == 2
        assert metadata.performance_metrics["mape"] == 2.5
        assert metadata.cv_scores["mean_cv_mape"] == 2.8
        assert metadata.hyperparameter_optimization is True
        assert metadata.best_hyperparams["learning_rate"] == 0.08

    def test_validate_n_samples_positive(self):
        """Test that sample counts must be positive."""
        # Valid
        metadata = TrainingMetadata(
            n_samples_train=1,
            n_samples_val=0,
            n_features=1,
            training_duration_seconds=1.0,
        )
        assert metadata.n_samples_train == 1

        # Invalid: n_samples_train = 0
        with pytest.raises(ValidationError):
            TrainingMetadata(
                n_samples_train=0,
                n_samples_val=200,
                n_features=10,
                training_duration_seconds=60.0,
            )

        # Invalid: negative n_samples_train
        with pytest.raises(ValidationError):
            TrainingMetadata(
                n_samples_train=-100,
                n_samples_val=200,
                n_features=10,
                training_duration_seconds=60.0,
            )

    def test_validate_n_features_positive(self):
        """Test that n_features must be positive."""
        # Invalid: n_features = 0
        with pytest.raises(ValidationError):
            TrainingMetadata(
                n_samples_train=1000,
                n_samples_val=200,
                n_features=0,
                training_duration_seconds=60.0,
            )

    def test_validate_feature_names_no_duplicates(self):
        """Test that feature names cannot have duplicates."""
        with pytest.raises(ValidationError, match="Duplicate feature names"):
            TrainingMetadata(
                n_samples_train=1000,
                n_samples_val=200,
                n_features=3,
                feature_names=["hour", "day_of_week", "hour"],  # Duplicate
                training_duration_seconds=60.0,
            )

    def test_validate_feature_names_no_empty_strings(self):
        """Test that feature names cannot be empty strings."""
        with pytest.raises(ValidationError, match="cannot be empty"):
            TrainingMetadata(
                n_samples_train=1000,
                n_samples_val=200,
                n_features=3,
                feature_names=["hour", "", "day_of_week"],  # Empty string
                training_duration_seconds=60.0,
            )

    def test_validate_metrics_finite(self):
        """Test that performance metrics must be finite numbers."""
        # Valid
        metadata = TrainingMetadata(
            n_samples_train=1000,
            n_samples_val=200,
            n_features=10,
            training_duration_seconds=60.0,
            performance_metrics={"mape": 2.5, "rmse": 150.0},
        )
        assert metadata.performance_metrics["mape"] == 2.5

        # Invalid: inf value
        with pytest.raises(ValidationError, match="non-finite"):
            TrainingMetadata(
                n_samples_train=1000,
                n_samples_val=200,
                n_features=10,
                training_duration_seconds=60.0,
                performance_metrics={"mape": float("inf")},
            )

        # Invalid: nan value
        with pytest.raises(ValidationError, match="non-finite"):
            TrainingMetadata(
                n_samples_train=1000,
                n_samples_val=200,
                n_features=10,
                training_duration_seconds=60.0,
                performance_metrics={"mape": float("nan")},
            )

    def test_validate_training_duration_positive(self):
        """Test that training duration must be non-negative."""
        # Valid: 0 seconds
        metadata = TrainingMetadata(
            n_samples_train=1000,
            n_samples_val=200,
            n_features=10,
            training_duration_seconds=0.0,
        )
        assert metadata.training_duration_seconds == 0.0

        # Invalid: negative duration
        with pytest.raises(ValidationError):
            TrainingMetadata(
                n_samples_train=1000,
                n_samples_val=200,
                n_features=10,
                training_duration_seconds=-10.0,
            )


class TestSerializedModelMetadata:
    """Test SerializedModelMetadata Pydantic model."""

    def test_create_minimal_serialized_metadata(self):
        """Test creating SerializedModelMetadata with minimal fields."""
        created_at = datetime.now(UTC)

        metadata = SerializedModelMetadata(
            model_type="lgbm",
            model_name="lgbm_test",
            version="1.0.0",
            created_at=created_at,
            checksum="a" * 64,  # Valid SHA256
            file_size_bytes=1024,
        )

        assert metadata.model_type == "lgbm"
        assert metadata.model_name == "lgbm_test"
        assert metadata.version == "1.0.0"
        assert metadata.created_at == created_at
        assert metadata.serialization_format == "joblib"
        assert metadata.compression == "gzip"
        assert metadata.compression_level == 6
        assert metadata.checksum == "a" * 64
        assert metadata.file_size_bytes == 1024
        assert metadata.dependencies == {}
        assert metadata.model_id is not None  # Auto-generated UUID

    def test_create_full_serialized_metadata(self):
        """Test creating SerializedModelMetadata with all fields."""
        created_at = datetime.now(UTC)
        serialized_at = datetime.now(UTC)

        metadata = SerializedModelMetadata(
            model_id="test-uuid-123",
            model_type="lgbm",
            model_name="lgbm_production",
            version="1.2.3",
            created_at=created_at,
            serialized_at=serialized_at,
            serialization_format="joblib",
            compression="gzip",
            compression_level=9,
            checksum="abc123" + "0" * 58,  # 64 chars
            file_size_bytes=1024000,
            python_version="3.12.0",
            dependencies={"lightgbm": "4.1.0", "pandas": "2.0.0"},
        )

        assert metadata.model_id == "test-uuid-123"
        assert metadata.compression_level == 9
        assert metadata.python_version == "3.12.0"
        assert metadata.dependencies["lightgbm"] == "4.1.0"

    def test_validate_model_type_pattern(self):
        """Test that model_type must match pattern."""
        created_at = datetime.now(UTC)
        checksum = "a" * 64

        # Valid model types
        valid_types = ["lgbm", "random_forest", "model_v2", "my_model123"]
        for model_type in valid_types:
            metadata = SerializedModelMetadata(
                model_type=model_type,
                model_name="test",
                version="1.0.0",
                created_at=created_at,
                checksum=checksum,
                file_size_bytes=1024,
            )
            assert metadata.model_type == model_type

        # Invalid model types
        invalid_types = ["LGBM", "model-1", "model with spaces", "1model"]
        for model_type in invalid_types:
            with pytest.raises(ValidationError):
                SerializedModelMetadata(
                    model_type=model_type,
                    model_name="test",
                    version="1.0.0",
                    created_at=created_at,
                    checksum=checksum,
                    file_size_bytes=1024,
                )

    def test_validate_version_semantic(self):
        """Test that version must be valid semantic version."""
        created_at = datetime.now(UTC)
        checksum = "a" * 64

        # Valid versions
        valid_versions = ["1.0.0", "2.1.3", "0.0.1", "1.0.0-beta.1"]
        for version in valid_versions:
            metadata = SerializedModelMetadata(
                model_type="lgbm",
                model_name="test",
                version=version,
                created_at=created_at,
                checksum=checksum,
                file_size_bytes=1024,
            )
            assert metadata.version == version

        # Invalid versions
        invalid_versions = ["1.0", "v1.0.0", "1.0.0.0", "invalid"]
        for version in invalid_versions:
            with pytest.raises(ValidationError):
                SerializedModelMetadata(
                    model_type="lgbm",
                    model_name="test",
                    version=version,
                    created_at=created_at,
                    checksum=checksum,
                    file_size_bytes=1024,
                )

    def test_validate_compression_supported(self):
        """Test that compression must be supported algorithm."""
        created_at = datetime.now(UTC)
        checksum = "a" * 64

        # Valid compression
        valid_compression = ["gzip", "lz4", "zstd", "none", None]
        for compression in valid_compression:
            metadata = SerializedModelMetadata(
                model_type="lgbm",
                model_name="test",
                version="1.0.0",
                created_at=created_at,
                checksum=checksum,
                file_size_bytes=1024,
                compression=compression,
            )
            assert metadata.compression == compression

        # Invalid compression
        with pytest.raises(ValidationError, match="Unsupported compression"):
            SerializedModelMetadata(
                model_type="lgbm",
                model_name="test",
                version="1.0.0",
                created_at=created_at,
                checksum=checksum,
                file_size_bytes=1024,
                compression="invalid_compression",
            )

    def test_validate_compression_level_range(self):
        """Test that compression level must be in range 0-9."""
        created_at = datetime.now(UTC)
        checksum = "a" * 64

        # Valid levels
        for level in range(10):
            metadata = SerializedModelMetadata(
                model_type="lgbm",
                model_name="test",
                version="1.0.0",
                created_at=created_at,
                checksum=checksum,
                file_size_bytes=1024,
                compression_level=level,
            )
            assert metadata.compression_level == level

        # Invalid levels
        with pytest.raises(ValidationError):
            SerializedModelMetadata(
                model_type="lgbm",
                model_name="test",
                version="1.0.0",
                created_at=created_at,
                checksum=checksum,
                file_size_bytes=1024,
                compression_level=-1,
            )

        with pytest.raises(ValidationError):
            SerializedModelMetadata(
                model_type="lgbm",
                model_name="test",
                version="1.0.0",
                created_at=created_at,
                checksum=checksum,
                file_size_bytes=1024,
                compression_level=10,
            )

    def test_validate_checksum_format(self):
        """Test that checksum must be valid SHA256."""
        created_at = datetime.now(UTC)

        # Valid SHA256 (64 hex chars)
        valid_checksum = "a1b2c3d4" * 8  # 64 chars
        metadata = SerializedModelMetadata(
            model_type="lgbm",
            model_name="test",
            version="1.0.0",
            created_at=created_at,
            checksum=valid_checksum,
            file_size_bytes=1024,
        )
        assert metadata.checksum == valid_checksum.lower()

        # Invalid: wrong length
        with pytest.raises(ValidationError, match="Invalid SHA256 checksum length"):
            SerializedModelMetadata(
                model_type="lgbm",
                model_name="test",
                version="1.0.0",
                created_at=created_at,
                checksum="abc123",  # Too short
                file_size_bytes=1024,
            )

        # Invalid: non-hex characters
        with pytest.raises(ValidationError, match="not hexadecimal"):
            SerializedModelMetadata(
                model_type="lgbm",
                model_name="test",
                version="1.0.0",
                created_at=created_at,
                checksum="z" * 64,  # Invalid hex
                file_size_bytes=1024,
            )

    def test_validate_file_size_non_negative(self):
        """Test that file size must be non-negative."""
        created_at = datetime.now(UTC)
        checksum = "a" * 64

        # Valid: 0 bytes
        metadata = SerializedModelMetadata(
            model_type="lgbm",
            model_name="test",
            version="1.0.0",
            created_at=created_at,
            checksum=checksum,
            file_size_bytes=0,
        )
        assert metadata.file_size_bytes == 0

        # Invalid: negative size
        with pytest.raises(ValidationError):
            SerializedModelMetadata(
                model_type="lgbm",
                model_name="test",
                version="1.0.0",
                created_at=created_at,
                checksum=checksum,
                file_size_bytes=-1024,
            )

    def test_validate_timezone_added_if_missing(self):
        """Test that timezone is added to datetime if missing."""
        # Naive datetime (no timezone)
        naive_dt = datetime(2024, 11, 24, 10, 30, 0)

        metadata = SerializedModelMetadata(
            model_type="lgbm",
            model_name="test",
            version="1.0.0",
            created_at=naive_dt,
            checksum="a" * 64,
            file_size_bytes=1024,
        )

        # Should have timezone added (UTC)
        assert metadata.created_at.tzinfo is not None
        assert metadata.serialized_at.tzinfo is not None

    def test_to_dict_serialization(self):
        """Test to_dict() converts datetimes to ISO format."""
        created_at = datetime.now(UTC)

        metadata = SerializedModelMetadata(
            model_type="lgbm",
            model_name="test",
            version="1.0.0",
            created_at=created_at,
            checksum="a" * 64,
            file_size_bytes=1024,
        )

        data = metadata.to_dict()

        # Check datetimes are strings
        assert isinstance(data["created_at"], str)
        assert isinstance(data["serialized_at"], str)
        # Check ISO format
        assert "T" in data["created_at"]

    def test_from_dict_deserialization(self):
        """Test from_dict() parses ISO format datetimes."""
        data = {
            "model_type": "lgbm",
            "model_name": "test",
            "version": "1.0.0",
            "created_at": "2024-11-24T10:30:00+00:00",
            "serialized_at": "2024-11-24T11:00:00+00:00",
            "checksum": "a" * 64,
            "file_size_bytes": 1024,
        }

        metadata = SerializedModelMetadata.from_dict(data)

        assert isinstance(metadata.created_at, datetime)
        assert isinstance(metadata.serialized_at, datetime)
        assert metadata.created_at.year == 2024
        assert metadata.created_at.month == 11

    def test_round_trip_serialization(self):
        """Test round-trip: to_dict() -> from_dict()."""
        original = SerializedModelMetadata(
            model_type="lgbm",
            model_name="test_model",
            version="1.2.3",
            created_at=datetime.now(UTC),
            checksum="abc123" + "0" * 58,
            file_size_bytes=2048000,
            compression="gzip",
            compression_level=9,
        )

        # Convert to dict and back
        data = original.to_dict()
        reconstructed = SerializedModelMetadata.from_dict(data)

        # Compare fields
        assert reconstructed.model_type == original.model_type
        assert reconstructed.model_name == original.model_name
        assert reconstructed.version == original.version
        assert reconstructed.checksum == original.checksum
        assert reconstructed.file_size_bytes == original.file_size_bytes
        assert reconstructed.compression == original.compression
        assert reconstructed.compression_level == original.compression_level
