"""Model serialization with metadata capture and integrity validation.

This module provides the ModelSerializer class for saving and loading models
with comprehensive metadata, compression support, and integrity validation
via SHA256 checksums.

Example:
    ```python
    from src.models.serialization.model_serializer import ModelSerializer
    from src.models.serialization.schemas import TrainingMetadata
    from src.models.end_to_end.lgbm_model import LGBMModel
    from pathlib import Path

    # Train a model
    model = LGBMModel()
    model.fit(X_train, y_train, config)

    # Create training metadata
    training_meta = TrainingMetadata(
        n_samples_train=len(X_train),
        n_samples_val=len(X_val),
        n_features=X_train.shape[1],
        feature_names=list(X_train.columns),
        training_config=config,
        preprocessing_steps=["standardization"],
        training_duration_seconds=120.5,
        performance_metrics={"mape": 2.5, "rmse": 150.0},
        cv_scores={"mean_cv_mape": 2.8},
        hyperparameter_optimization=True,
        best_hyperparams={"learning_rate": 0.1},
    )

    # Save model with metadata
    serializer = ModelSerializer()
    path = Path("models/lgbm_v1.pkl")
    serializer.save_model(
        model=model,
        path=path,
        training_metadata=training_meta,
        version="1.0.0",
        compression="gzip",
        compression_level=6,
    )

    # Load model with integrity validation
    loaded_model, metadata = serializer.load_model(path, validate_integrity=True)
    print(f"Loaded {metadata.model_name} v{metadata.version}")
    ```
"""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib

from src.models.serialization.schemas import (
    SerializedModelMetadata,
    TrainingMetadata,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelSerializer:
    """Serializes and deserializes models with comprehensive metadata.

    This class provides methods for saving models to disk with full metadata
    capture, compression, and integrity validation via SHA256 checksums.
    Models are saved using joblib with optional compression, and metadata
    is stored in a separate JSON file alongside the model file.

    The serializer captures:
    - Model binary (via joblib with compression)
    - Serialization metadata (format, compression, checksum, file size)
    - Training metadata (samples, features, metrics, hyperparameters)
    - Python environment (version, dependencies)

    Attributes:
        None (stateless class)
    """

    def save_model(
        self,
        model: Any,
        path: str | Path,
        training_metadata: TrainingMetadata,
        version: str,
        compression: str | None = "gzip",
        compression_level: int = 6,
    ) -> tuple[Path, Path]:
        """Save model to disk with comprehensive metadata.

        This method:
        1. Serializes the model using joblib with compression
        2. Calculates SHA256 checksum for integrity validation
        3. Captures Python environment dependencies
        4. Creates SerializedModelMetadata with all serialization info
        5. Saves metadata to separate JSON file

        Args:
            model: Model instance to serialize (typically a BaseModel subclass).
            path: File path to save the model (e.g., "models/lgbm_v1.pkl").
                 Parent directory will be created if it doesn't exist.
            training_metadata: TrainingMetadata instance with training information.
            version: Semantic version string (e.g., "1.0.0").
            compression: Compression algorithm to use. Options:
                        - "gzip" (default): Good balance of speed and compression
                        - "lz4": Fastest, lower compression
                        - "zstd": Best compression, slower
                        - None or "none": No compression
            compression_level: Compression level (0-9, where 0=none, 9=max).
                              Default is 6 for balanced performance.

        Returns:
            Tuple of (model_path, metadata_path) pointing to saved files.

        Raises:
            ValueError: If inputs are invalid (e.g., invalid version format).
            OSError: If files cannot be written to disk.

        Example:
            ```python
            serializer = ModelSerializer()
            model_path, meta_path = serializer.save_model(
                model=my_model,
                path="models/lgbm_v1.pkl",
                training_metadata=training_meta,
                version="1.0.0",
                compression="gzip",
                compression_level=6,
            )
            ```
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Saving model to %s (compression=%s, level=%d)",
            path,
            compression,
            compression_level,
        )

        # Get model type and name from model if available
        model_type = getattr(model, "name", "unknown")
        model_name = f"{model_type}_model"

        # Normalize compression
        if compression == "none":
            compression = None

        # Prepare joblib compression parameter
        joblib_compress = None
        if compression and compression_level > 0:
            joblib_compress = (compression, compression_level)

        # Serialize model to disk
        try:
            start_time = datetime.now(UTC)
            joblib.dump(model, path, compress=joblib_compress)
            end_time = datetime.now(UTC)
            serialization_time = (end_time - start_time).total_seconds()
            logger.debug("Model serialization took %.2f seconds", serialization_time)
        except Exception as e:
            msg = f"Failed to serialize model to {path}: {e}"
            raise OSError(msg) from e

        # Calculate checksum
        try:
            checksum = self._calculate_checksum(path)
            logger.debug("Calculated SHA256 checksum: %s", checksum)
        except Exception as e:
            msg = f"Failed to calculate checksum for {path}: {e}"
            raise OSError(msg) from e

        # Get file size
        file_size_bytes = path.stat().st_size
        logger.debug(
            "Model file size: %d bytes (%.2f MB)", file_size_bytes, file_size_bytes / 1024 / 1024
        )

        # Capture Python dependencies
        dependencies = self._capture_dependencies()

        # Create serialization metadata
        created_at = getattr(model, "_training_metadata", {}).get("trained_at")
        if created_at is None or not isinstance(created_at, datetime):
            created_at = datetime.now(UTC)
        elif created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=UTC)

        metadata = SerializedModelMetadata(
            model_type=model_type,
            model_name=model_name,
            version=version,
            created_at=created_at,
            serialized_at=datetime.now(UTC),
            serialization_format="joblib",
            compression=compression,
            compression_level=compression_level,
            checksum=checksum,
            file_size_bytes=file_size_bytes,
            dependencies=dependencies,
        )

        # Save metadata to JSON file
        metadata_path = path.with_suffix(path.suffix + ".meta.json")
        try:
            metadata_dict = metadata.to_dict()
            # Add training metadata to the saved metadata
            metadata_dict["training_metadata"] = training_metadata.model_dump()

            with metadata_path.open("w") as f:
                json.dump(metadata_dict, f, indent=2)
            logger.debug("Saved metadata to %s", metadata_path)
        except Exception as e:
            msg = f"Failed to save metadata to {metadata_path}: {e}"
            raise OSError(msg) from e

        logger.info(
            "Model saved successfully: %s (%.2f MB, checksum: %s...)",
            path,
            file_size_bytes / 1024 / 1024,
            checksum[:8],
        )

        return path, metadata_path

    def load_model(
        self,
        path: str | Path,
        validate_integrity: bool = True,
    ) -> tuple[Any, SerializedModelMetadata]:
        """Load model from disk with optional integrity validation.

        This method:
        1. Loads metadata from JSON file
        2. Optionally validates file integrity via checksum
        3. Loads model using joblib
        4. Returns both model and metadata

        Args:
            path: File path to load the model from.
            validate_integrity: Whether to validate file checksum matches
                              metadata. Default is True for safety.

        Returns:
            Tuple of (model, metadata) where model is the deserialized model
            instance and metadata is SerializedModelMetadata.

        Raises:
            FileNotFoundError: If model or metadata file doesn't exist.
            ValueError: If integrity validation fails (checksum mismatch).
            OSError: If files cannot be read.

        Example:
            ```python
            serializer = ModelSerializer()
            model, metadata = serializer.load_model(
                "models/lgbm_v1.pkl",
                validate_integrity=True
            )
            print(f"Loaded {metadata.model_name} v{metadata.version}")
            ```
        """
        path = Path(path)

        if not path.exists():
            msg = f"Model file not found: {path}"
            raise FileNotFoundError(msg)

        logger.info("Loading model from %s (validate_integrity=%s)", path, validate_integrity)

        # Load metadata
        metadata_path = path.with_suffix(path.suffix + ".meta.json")
        if not metadata_path.exists():
            msg = f"Metadata file not found: {metadata_path}"
            raise FileNotFoundError(msg)

        try:
            with metadata_path.open("r") as f:
                metadata_dict = json.load(f)
            logger.debug("Loaded metadata from %s", metadata_path)
        except Exception as e:
            msg = f"Failed to load metadata from {metadata_path}: {e}"
            raise OSError(msg) from e

        # Parse metadata (remove training_metadata from dict before parsing)
        _ = metadata_dict.pop("training_metadata", None)
        try:
            metadata = SerializedModelMetadata.from_dict(metadata_dict)
        except Exception as e:
            msg = f"Failed to parse metadata: {e}"
            raise ValueError(msg) from e

        # Validate integrity if requested
        if validate_integrity:
            try:
                self.validate_integrity(path, expected_checksum=metadata.checksum)
            except ValueError as e:
                msg = f"Integrity validation failed for {path}: {e}"
                raise ValueError(msg) from e

        # Load model
        try:
            start_time = datetime.now(UTC)
            model = joblib.load(path)
            end_time = datetime.now(UTC)
            load_time = (end_time - start_time).total_seconds()
            logger.debug("Model loading took %.2f seconds", load_time)
        except Exception as e:
            msg = f"Failed to load model from {path}: {e}"
            raise OSError(msg) from e

        logger.info(
            "Model loaded successfully: %s v%s (%.2f MB)",
            metadata.model_name,
            metadata.version,
            metadata.file_size_bytes / 1024 / 1024,
        )

        return model, metadata

    def _calculate_checksum(self, path: Path) -> str:
        """Calculate SHA256 checksum of a file.

        Args:
            path: Path to file to checksum.

        Returns:
            SHA256 checksum as hexadecimal string.

        Raises:
            OSError: If file cannot be read.
        """
        sha256_hash = hashlib.sha256()

        try:
            with path.open("rb") as f:
                # Read file in chunks to handle large files
                for byte_block in iter(lambda: f.read(8192), b""):
                    sha256_hash.update(byte_block)
        except Exception as e:
            msg = f"Failed to read file for checksum: {path}: {e}"
            raise OSError(msg) from e

        return sha256_hash.hexdigest()

    def _capture_dependencies(self) -> dict[str, str]:
        """Capture current Python package versions.

        Returns:
            Dictionary mapping package names to version strings.
            Includes key packages used in model training (pandas, numpy,
            scikit-learn, lightgbm, etc.).

        Note:
            Only captures versions of packages that are currently installed.
            Missing packages are silently skipped.
        """
        import importlib.metadata

        # Key packages to capture
        packages_to_capture = [
            "pandas",
            "numpy",
            "scikit-learn",
            "lightgbm",
            "xgboost",
            "joblib",
            "pydantic",
        ]

        dependencies = {}
        for package_name in packages_to_capture:
            try:
                version = importlib.metadata.version(package_name)
                dependencies[package_name] = version
            except importlib.metadata.PackageNotFoundError:
                # Package not installed, skip
                logger.debug("Package not found: %s", package_name)
                continue

        logger.debug("Captured %d package dependencies", len(dependencies))
        return dependencies

    def validate_integrity(
        self,
        path: Path | str,
        expected_checksum: str | None = None,
    ) -> bool:
        """Validate model file integrity via checksum.

        Args:
            path: Path to model file to validate.
            expected_checksum: Expected SHA256 checksum. If None, loads from
                             metadata file.

        Returns:
            True if checksum matches, raises ValueError otherwise.

        Raises:
            FileNotFoundError: If model or metadata file doesn't exist.
            ValueError: If checksum doesn't match expected value.
            OSError: If file cannot be read.

        Example:
            ```python
            serializer = ModelSerializer()
            # Validate using metadata file
            serializer.validate_integrity("models/lgbm_v1.pkl")

            # Validate with explicit checksum
            serializer.validate_integrity(
                "models/lgbm_v1.pkl",
                expected_checksum="abc123..."
            )
            ```
        """
        path = Path(path)

        if not path.exists():
            msg = f"Model file not found: {path}"
            raise FileNotFoundError(msg)

        # Load expected checksum from metadata if not provided
        if expected_checksum is None:
            metadata_path = path.with_suffix(path.suffix + ".meta.json")
            if not metadata_path.exists():
                msg = f"Metadata file not found: {metadata_path}"
                raise FileNotFoundError(msg)

            try:
                with metadata_path.open("r") as f:
                    metadata_dict = json.load(f)
                expected_checksum = metadata_dict["checksum"]
            except Exception as e:
                msg = f"Failed to load checksum from metadata: {e}"
                raise OSError(msg) from e

        # Calculate actual checksum
        actual_checksum = self._calculate_checksum(path)

        # Compare checksums (case-insensitive)
        if actual_checksum.lower() != expected_checksum.lower():
            msg = (
                f"Checksum mismatch for {path}. "
                f"Expected: {expected_checksum}, Got: {actual_checksum}. "
                "File may be corrupted or tampered with."
            )
            raise ValueError(msg)

        logger.debug("Integrity validation passed for %s", path)
        return True
