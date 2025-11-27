"""Abstract base class for all forecasting models.

This module defines the BaseModel abstract class that establishes the interface
for all forecasting models in the PrevCarga system. It provides abstract methods
for training, prediction, and feature importance, along with concrete methods
for model management and serialization.

Example:
    ```python
    from src.models.base.model import BaseModel
    import pandas as pd
    import joblib

    class MyModel(BaseModel):
        @property
        def name(self) -> str:
            return "my_model"

        @property
        def version(self) -> str:
            return "1.0.0"

        @property
        def supported_horizons(self) -> list[int]:
            return [0, 1, 2, 3]

        def fit(
            self,
            X: pd.DataFrame,
            y: pd.Series | pd.DataFrame,
            config: dict[str, Any]
        ) -> None:
            # Store feature names
            self._feature_names = list(X.columns)
            # Train model logic here
            self._model = ...  # Your model implementation
            self._is_fitted = True

        def predict(
            self,
            X: pd.DataFrame,
            horizons: list[int] | None = None
        ) -> pd.DataFrame:
            self._check_is_fitted()
            # Prediction logic here
            return predictions

        def get_feature_importance(self) -> dict[str, float]:
            self._check_is_fitted()
            # Feature importance logic
            return importance_dict
    ```
"""

from abc import ABC, abstractmethod
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.models.base.metadata import ModelMetadata
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BaseModel(ABC):
    """Abstract base class for forecasting models.

    This class defines the interface that all forecasting models must implement.
    It provides abstract methods for model training, prediction, and feature
    importance, along with concrete methods for state management, serialization,
    and metadata handling.

    Subclasses must implement:
        - name: Model identifier (property)
        - version: Model version (property)
        - supported_horizons: List of forecast horizons (property)
        - fit: Training method
        - predict: Prediction method
        - get_feature_importance: Feature importance extraction

    Attributes:
        _is_fitted: Boolean indicating whether model has been trained.
        _feature_names: List of feature column names used during training.
        _training_metadata: Dictionary of metadata collected during training.
    """

    def __init__(self) -> None:
        """Initialize base model with empty state."""
        self._is_fitted: bool = False
        self._feature_names: list[str] = []
        self._training_metadata: dict[str, Any] = {}
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the model name identifier.

        The name must be lowercase with underscores only, matching the
        pattern ^[a-z][a-z0-9_]*$.

        Returns:
            Model name string (e.g., "lgbm", "random_forest").
        """
        ...

    @property
    @abstractmethod
    def version(self) -> str:
        """Return the model version string.

        The version must follow semantic versioning format: MAJOR.MINOR.PATCH.
        Pre-release versions are supported (e.g., "1.0.0-beta.1").

        Returns:
            Version string in semantic versioning format (e.g., "1.0.0").
        """
        ...

    @property
    @abstractmethod
    def supported_horizons(self) -> list[int]:
        """Return list of forecast horizons this model supports.

        Returns:
            List of integers from 0 to 8 representing D+0 to D+8 horizons.
            Example: [0, 1, 2, 3] for D+0 through D+3.
        """
        ...

    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | pd.DataFrame,
        config: dict[str, Any],
    ) -> None:
        """Train the model on the provided data.

        This method must:
        1. Store feature names in self._feature_names
        2. Set self._is_fitted = True after successful training
        3. Optionally populate self._training_metadata with training info

        Args:
            X: Training features DataFrame with shape (n_samples, n_features).
            y: Training target as Series (single horizon) or DataFrame (multi-horizon).
            config: Model-specific training configuration dictionary.

        Raises:
            ValueError: If input data is invalid or incompatible with model.
            RuntimeError: If training fails.
        """
        ...

    @abstractmethod
    def predict(
        self,
        X: pd.DataFrame,
        horizons: list[int] | None = None,
    ) -> pd.DataFrame:
        """Generate predictions for the given features.

        Args:
            X: Features DataFrame with shape (n_samples, n_features).
            horizons: List of horizons to predict. If None, predict all
                     supported_horizons. Must be subset of supported_horizons.

        Returns:
            DataFrame with predictions. Shape (n_samples, n_horizons).
            Columns are named "h0", "h1", etc. for each horizon.

        Raises:
            ValueError: If model is not fitted or horizons are invalid.
            RuntimeError: If prediction fails.
        """
        ...

    @abstractmethod
    def get_feature_importance(self) -> dict[str, float]:
        """Return feature importance scores.

        Returns:
            Dictionary mapping feature names to importance scores.
            Scores should be normalized (sum to 1.0 or represent permutation
            importance, SHAP values, etc.).

        Raises:
            ValueError: If model is not fitted.
            NotImplementedError: If model doesn't support feature importance.
        """
        ...

    def is_fitted(self) -> bool:
        """Check if model has been trained.

        Returns:
            True if model is fitted and ready for predictions, False otherwise.
        """
        return self._is_fitted

    def _check_is_fitted(self) -> None:
        """Check if model is fitted and raise error if not.

        Raises:
            ValueError: If model has not been fitted.
        """
        if not self._is_fitted:
            msg = f"Model {self.name} has not been fitted. Call fit() first."
            raise ValueError(msg)

    def get_feature_names(self) -> list[str]:
        """Return list of feature names used during training.

        Returns:
            List of feature column names. Empty list if not fitted.
        """
        return self._feature_names.copy()

    def validate_features(self, X: pd.DataFrame) -> None:
        """Validate that input features match training features.

        Args:
            X: Features DataFrame to validate.

        Raises:
            ValueError: If model is not fitted or features don't match.
        """
        self._check_is_fitted()

        if not self._feature_names:
            logger.warning("No feature names stored during training")
            return

        missing = set(self._feature_names) - set(X.columns)
        if missing:
            msg = f"Missing required features: {sorted(missing)}"
            raise ValueError(msg)

        extra = set(X.columns) - set(self._feature_names)
        if extra:
            logger.debug("Extra features in input (will be ignored): %s", sorted(extra))

    def get_metadata(self, model_id: str | None = None) -> ModelMetadata:
        """Generate ModelMetadata for this model instance.

        Args:
            model_id: Unique model identifier. If None, auto-generated from
                     name and timestamp.

        Returns:
            ModelMetadata instance with model information.
        """
        if model_id is None:
            timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
            model_id = f"{self.name}_{timestamp}"

        # Get training timestamp from metadata or use current time
        trained_at = self._training_metadata.get("trained_at")
        if trained_at is None:
            trained_at = datetime.now(UTC)
        elif not isinstance(trained_at, datetime):
            # If it's a string, parse it
            trained_at = datetime.fromisoformat(str(trained_at))

        # Extract performance metrics if available
        performance_metrics = self._training_metadata.get("performance_metrics", {})

        # Extract training config if available
        training_config = self._training_metadata.get("training_config", {})

        metadata = ModelMetadata(
            model_id=model_id,
            model_name=self.name,
            model_version=self.version,
            model_class=self.__class__.__name__,
            trained_at=trained_at,
            supported_horizons=self.supported_horizons,
            feature_dependencies=self._feature_names,
            performance_metrics=performance_metrics,
            training_config=training_config,
            custom_metadata=self._training_metadata.get("custom_metadata", {}),
        )

        logger.debug("Generated metadata for model: %s", metadata)
        return metadata

    def save(self, path: str | Path) -> None:
        """Save model to disk using joblib.

        This method saves the entire model instance including fitted state,
        feature names, and training metadata.

        Args:
            path: File path to save the model. Parent directory will be created
                 if it doesn't exist.

        Raises:
            ValueError: If model is not fitted.
            OSError: If file cannot be written.
        """
        self._check_is_fitted()

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            joblib.dump(self, path)
            logger.info("Saved model %s to %s", self.name, path)
        except Exception as e:
            msg = f"Failed to save model to {path}: {e}"
            raise OSError(msg) from e

    @classmethod
    def load(cls, path: str | Path) -> "BaseModel":
        """Load model from disk using joblib.

        Args:
            path: File path to load the model from.

        Returns:
            Loaded model instance.

        Raises:
            FileNotFoundError: If model file doesn't exist.
            ValueError: If loaded object is not a BaseModel instance.
            OSError: If file cannot be read.
        """
        path = Path(path)

        if not path.exists():
            msg = f"Model file not found: {path}"
            raise FileNotFoundError(msg)

        try:
            model = joblib.load(path)
        except Exception as e:
            msg = f"Failed to load model from {path}: {e}"
            raise OSError(msg) from e

        if not isinstance(model, BaseModel):
            msg = f"Loaded object is not a BaseModel instance: {type(model)}"
            raise ValueError(msg)

        logger.info("Loaded model %s from %s", model.name, path)
        return model

    def __repr__(self) -> str:
        """Return string representation of the model.

        Returns:
            String in format: ModelClass(name='model_name', version='1.0.0', fitted=True)
        """
        return (
            f"{self.__class__.__name__}("
            f"name='{self.name}', "
            f"version='{self.version}', "
            f"fitted={self._is_fitted})"
        )

    def __str__(self) -> str:
        """Return human-readable string representation.

        Returns:
            String with model name and version.
        """
        return f"{self.name} v{self.version}"
