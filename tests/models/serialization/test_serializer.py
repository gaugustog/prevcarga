"""Tests for ModelSerializer class."""

import json
from datetime import UTC, datetime

import pytest

from src.models.serialization.model_serializer import ModelSerializer
from src.models.serialization.schemas import TrainingMetadata


class DummyModel:
    """Dummy model for testing serialization."""

    def __init__(self, name="test_model", data=None):
        """Initialize dummy model."""
        self.name = name
        self.data = data or {"param1": 1.0, "param2": 2.0}
        self._training_metadata = {
            "trained_at": datetime.now(UTC),
        }


@pytest.fixture
def serializer():
    """Create ModelSerializer instance."""
    return ModelSerializer()


@pytest.fixture
def dummy_model():
    """Create dummy model for testing."""
    return DummyModel(name="test_model")


@pytest.fixture
def training_metadata():
    """Create sample TrainingMetadata."""
    return TrainingMetadata(
        n_samples_train=1000,
        n_samples_val=200,
        n_features=10,
        feature_names=["feat1", "feat2", "feat3"],
        training_config={"learning_rate": 0.1},
        preprocessing_steps=["standardization"],
        training_duration_seconds=60.5,
        performance_metrics={"mape": 2.5, "rmse": 150.0},
        cv_scores={"mean_cv_mape": 2.8},
        hyperparameter_optimization=True,
        best_hyperparams={"learning_rate": 0.08},
    )


@pytest.fixture
def temp_model_dir(tmp_path):
    """Create temporary directory for model files."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    return model_dir


class TestModelSerializerSave:
    """Test ModelSerializer.save_model() method."""

    def test_save_model_basic(self, serializer, dummy_model, training_metadata, temp_model_dir):
        """Test basic model saving."""
        model_path = temp_model_dir / "test_model.pkl"

        saved_path, meta_path = serializer.save_model(
            model=dummy_model,
            path=model_path,
            training_metadata=training_metadata,
            version="1.0.0",
            compression="gzip",
            compression_level=6,
        )

        # Check files were created
        assert saved_path.exists()
        assert meta_path.exists()
        assert saved_path == model_path
        assert meta_path.name == "test_model.pkl.meta.json"

    def test_save_model_creates_parent_directory(
        self, serializer, dummy_model, training_metadata, temp_model_dir
    ):
        """Test that save_model creates parent directory if it doesn't exist."""
        model_path = temp_model_dir / "subdir" / "nested" / "test_model.pkl"

        saved_path, meta_path = serializer.save_model(
            model=dummy_model,
            path=model_path,
            training_metadata=training_metadata,
            version="1.0.0",
        )

        assert saved_path.exists()
        assert meta_path.exists()
        assert model_path.parent.exists()

    def test_save_model_with_different_compression(
        self, serializer, dummy_model, training_metadata, temp_model_dir
    ):
        """Test saving with different compression algorithms."""
        # Only test gzip and none (always available)
        # lz4 and zstd require optional dependencies
        compression_types = ["gzip", None, "none"]

        for compression in compression_types:
            model_path = temp_model_dir / f"test_{compression or 'none'}.pkl"

            saved_path, meta_path = serializer.save_model(
                model=dummy_model,
                path=model_path,
                training_metadata=training_metadata,
                version="1.0.0",
                compression=compression,
            )

            assert saved_path.exists()
            assert meta_path.exists()

            # Check metadata has correct compression
            with meta_path.open("r") as f:
                metadata = json.load(f)
            if compression == "none":
                assert metadata["compression"] is None
            else:
                assert metadata["compression"] == compression

    def test_save_model_with_different_compression_levels(
        self, serializer, dummy_model, training_metadata, temp_model_dir
    ):
        """Test saving with different compression levels."""
        for level in [0, 3, 6, 9]:
            model_path = temp_model_dir / f"test_level_{level}.pkl"

            _, meta_path = serializer.save_model(
                model=dummy_model,
                path=model_path,
                training_metadata=training_metadata,
                version="1.0.0",
                compression="gzip",
                compression_level=level,
            )

            # Check metadata has correct compression level
            with meta_path.open("r") as f:
                metadata = json.load(f)
            assert metadata["compression_level"] == level

    def test_save_model_metadata_contents(
        self, serializer, dummy_model, training_metadata, temp_model_dir
    ):
        """Test that saved metadata contains all required fields."""
        model_path = temp_model_dir / "test_model.pkl"

        _saved_path, meta_path = serializer.save_model(
            model=dummy_model,
            path=model_path,
            training_metadata=training_metadata,
            version="1.0.0",
        )

        # Load and check metadata
        with meta_path.open("r") as f:
            metadata = json.load(f)

        # Check SerializedModelMetadata fields
        assert metadata["model_type"] == "test_model"
        assert metadata["version"] == "1.0.0"
        assert "checksum" in metadata
        assert "file_size_bytes" in metadata
        assert "python_version" in metadata
        assert "dependencies" in metadata
        assert "created_at" in metadata
        assert "serialized_at" in metadata

        # Check TrainingMetadata is included
        assert "training_metadata" in metadata
        assert metadata["training_metadata"]["n_samples_train"] == 1000
        assert metadata["training_metadata"]["n_features"] == 10

    def test_save_model_checksum_calculated(
        self, serializer, dummy_model, training_metadata, temp_model_dir
    ):
        """Test that checksum is calculated correctly."""
        model_path = temp_model_dir / "test_model.pkl"

        _saved_path, meta_path = serializer.save_model(
            model=dummy_model,
            path=model_path,
            training_metadata=training_metadata,
            version="1.0.0",
        )

        # Load metadata
        with meta_path.open("r") as f:
            metadata = json.load(f)

        # Check checksum is valid SHA256 (64 hex chars)
        checksum = metadata["checksum"]
        assert len(checksum) == 64
        assert all(c in "0123456789abcdef" for c in checksum.lower())

    def test_save_model_file_size_recorded(
        self, serializer, dummy_model, training_metadata, temp_model_dir
    ):
        """Test that file size is recorded correctly."""
        model_path = temp_model_dir / "test_model.pkl"

        saved_path, meta_path = serializer.save_model(
            model=dummy_model,
            path=model_path,
            training_metadata=training_metadata,
            version="1.0.0",
        )

        # Load metadata
        with meta_path.open("r") as f:
            metadata = json.load(f)

        # Check file size matches actual file
        actual_size = saved_path.stat().st_size
        assert metadata["file_size_bytes"] == actual_size
        assert actual_size > 0


class TestModelSerializerLoad:
    """Test ModelSerializer.load_model() method."""

    def test_load_model_basic(self, serializer, dummy_model, training_metadata, temp_model_dir):
        """Test basic model loading."""
        model_path = temp_model_dir / "test_model.pkl"

        # Save model first
        serializer.save_model(
            model=dummy_model,
            path=model_path,
            training_metadata=training_metadata,
            version="1.0.0",
        )

        # Load model
        loaded_model, metadata = serializer.load_model(model_path)

        # Check model was loaded correctly
        assert loaded_model.name == dummy_model.name
        assert loaded_model.data == dummy_model.data

        # Check metadata
        assert metadata.model_type == "test_model"
        assert metadata.version == "1.0.0"

    def test_load_model_with_integrity_validation(
        self, serializer, dummy_model, training_metadata, temp_model_dir
    ):
        """Test loading with integrity validation enabled."""
        model_path = temp_model_dir / "test_model.pkl"

        # Save model
        serializer.save_model(
            model=dummy_model,
            path=model_path,
            training_metadata=training_metadata,
            version="1.0.0",
        )

        # Load with validation
        loaded_model, _ = serializer.load_model(
            model_path,
            validate_integrity=True,
        )

        assert loaded_model.name == dummy_model.name

    def test_load_model_without_integrity_validation(
        self, serializer, dummy_model, training_metadata, temp_model_dir
    ):
        """Test loading without integrity validation."""
        model_path = temp_model_dir / "test_model.pkl"

        # Save model
        serializer.save_model(
            model=dummy_model,
            path=model_path,
            training_metadata=training_metadata,
            version="1.0.0",
        )

        # Load without validation
        loaded_model, _ = serializer.load_model(
            model_path,
            validate_integrity=False,
        )

        assert loaded_model.name == dummy_model.name

    def test_load_model_file_not_found(self, serializer, temp_model_dir):
        """Test loading raises error when model file doesn't exist."""
        model_path = temp_model_dir / "nonexistent.pkl"

        with pytest.raises(FileNotFoundError, match="Model file not found"):
            serializer.load_model(model_path)

    def test_load_model_metadata_not_found(
        self, serializer, dummy_model, training_metadata, temp_model_dir
    ):
        """Test loading raises error when metadata file doesn't exist."""
        model_path = temp_model_dir / "test_model.pkl"

        # Save model
        serializer.save_model(
            model=dummy_model,
            path=model_path,
            training_metadata=training_metadata,
            version="1.0.0",
        )

        # Delete metadata file
        meta_path = model_path.with_suffix(".pkl.meta.json")
        meta_path.unlink()

        with pytest.raises(FileNotFoundError, match="Metadata file not found"):
            serializer.load_model(model_path)


class TestModelSerializerIntegrity:
    """Test integrity validation methods."""

    def test_validate_integrity_success(
        self, serializer, dummy_model, training_metadata, temp_model_dir
    ):
        """Test integrity validation passes for valid file."""
        model_path = temp_model_dir / "test_model.pkl"

        # Save model
        serializer.save_model(
            model=dummy_model,
            path=model_path,
            training_metadata=training_metadata,
            version="1.0.0",
        )

        # Validate integrity
        result = serializer.validate_integrity(model_path)
        assert result is True

    def test_validate_integrity_with_explicit_checksum(
        self, serializer, dummy_model, training_metadata, temp_model_dir
    ):
        """Test integrity validation with explicit checksum."""
        model_path = temp_model_dir / "test_model.pkl"

        # Save model
        _saved_path, meta_path = serializer.save_model(
            model=dummy_model,
            path=model_path,
            training_metadata=training_metadata,
            version="1.0.0",
        )

        # Load metadata to get checksum
        with meta_path.open("r") as f:
            metadata = json.load(f)
        expected_checksum = metadata["checksum"]

        # Validate with explicit checksum
        result = serializer.validate_integrity(model_path, expected_checksum=expected_checksum)
        assert result is True

    def test_validate_integrity_checksum_mismatch(
        self, serializer, dummy_model, training_metadata, temp_model_dir
    ):
        """Test integrity validation fails for corrupted file."""
        model_path = temp_model_dir / "test_model.pkl"

        # Save model
        serializer.save_model(
            model=dummy_model,
            path=model_path,
            training_metadata=training_metadata,
            version="1.0.0",
        )

        # Corrupt the model file
        with model_path.open("ab") as f:
            f.write(b"CORRUPTED DATA")

        # Validation should fail
        with pytest.raises(ValueError, match="Checksum mismatch"):
            serializer.validate_integrity(model_path)

    def test_validate_integrity_metadata_not_found(self, serializer, temp_model_dir):
        """Test validate_integrity raises error when metadata missing."""
        model_path = temp_model_dir / "test_model.pkl"

        # Create dummy file without metadata
        model_path.write_text("dummy content")

        with pytest.raises(FileNotFoundError, match="Metadata file not found"):
            serializer.validate_integrity(model_path)

    def test_load_model_with_corrupted_file_fails(
        self, serializer, dummy_model, training_metadata, temp_model_dir
    ):
        """Test that loading corrupted file with validation enabled fails."""
        model_path = temp_model_dir / "test_model.pkl"

        # Save model
        serializer.save_model(
            model=dummy_model,
            path=model_path,
            training_metadata=training_metadata,
            version="1.0.0",
        )

        # Corrupt the model file
        with model_path.open("ab") as f:
            f.write(b"CORRUPTED DATA")

        # Loading with validation should fail
        with pytest.raises(ValueError, match="Integrity validation failed"):
            serializer.load_model(model_path, validate_integrity=True)


class TestModelSerializerHelperMethods:
    """Test helper methods of ModelSerializer."""

    def test_calculate_checksum(self, serializer, temp_model_dir):
        """Test _calculate_checksum() method."""
        test_file = temp_model_dir / "test.txt"
        test_file.write_text("Hello, World!")

        checksum = serializer._calculate_checksum(test_file)

        # Check it's a valid SHA256 (64 hex chars)
        assert len(checksum) == 64
        assert all(c in "0123456789abcdef" for c in checksum.lower())

        # Check it's deterministic
        checksum2 = serializer._calculate_checksum(test_file)
        assert checksum == checksum2

    def test_calculate_checksum_different_files(self, serializer, temp_model_dir):
        """Test that different files produce different checksums."""
        file1 = temp_model_dir / "file1.txt"
        file2 = temp_model_dir / "file2.txt"

        file1.write_text("Content A")
        file2.write_text("Content B")

        checksum1 = serializer._calculate_checksum(file1)
        checksum2 = serializer._calculate_checksum(file2)

        assert checksum1 != checksum2

    def test_capture_dependencies(self, serializer):
        """Test _capture_dependencies() method."""
        dependencies = serializer._capture_dependencies()

        # Check it returns a dict
        assert isinstance(dependencies, dict)

        # Check it includes common packages (at least some should be installed)
        # We know pandas and pydantic are installed for tests
        assert "pandas" in dependencies or "pydantic" in dependencies

        # Check versions are strings
        for package, version in dependencies.items():
            assert isinstance(package, str)
            assert isinstance(version, str)
            assert len(version) > 0


class TestModelSerializerRoundTrip:
    """Test complete save/load round-trip scenarios."""

    def test_round_trip_basic(self, serializer, dummy_model, training_metadata, temp_model_dir):
        """Test basic save and load round-trip."""
        model_path = temp_model_dir / "test_model.pkl"

        # Save
        serializer.save_model(
            model=dummy_model,
            path=model_path,
            training_metadata=training_metadata,
            version="1.0.0",
        )

        # Load
        loaded_model, metadata = serializer.load_model(model_path)

        # Verify model data
        assert loaded_model.name == dummy_model.name
        assert loaded_model.data == dummy_model.data

        # Verify metadata
        assert metadata.version == "1.0.0"
        assert metadata.model_type == "test_model"

    def test_round_trip_multiple_models(self, serializer, training_metadata, temp_model_dir):
        """Test saving and loading multiple different models."""
        models = [
            DummyModel(name="model1", data={"a": 1}),
            DummyModel(name="model2", data={"b": 2}),
            DummyModel(name="model3", data={"c": 3}),
        ]

        # Save all models
        for i, model in enumerate(models):
            model_path = temp_model_dir / f"model_{i}.pkl"
            serializer.save_model(
                model=model,
                path=model_path,
                training_metadata=training_metadata,
                version=f"1.{i}.0",
            )

        # Load and verify all models
        for i, original_model in enumerate(models):
            model_path = temp_model_dir / f"model_{i}.pkl"
            loaded_model, metadata = serializer.load_model(model_path)

            assert loaded_model.name == original_model.name
            assert loaded_model.data == original_model.data
            assert metadata.version == f"1.{i}.0"

    def test_round_trip_with_path_as_string(
        self, serializer, dummy_model, training_metadata, temp_model_dir
    ):
        """Test round-trip with path as string instead of Path object."""
        model_path = str(temp_model_dir / "test_model.pkl")

        # Save with string path
        serializer.save_model(
            model=dummy_model,
            path=model_path,
            training_metadata=training_metadata,
            version="1.0.0",
        )

        # Load with string path
        loaded_model, _ = serializer.load_model(model_path)

        assert loaded_model.name == dummy_model.name
