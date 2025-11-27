"""Random Forest model configuration schema.

This module provides Pydantic models for Random Forest hyperparameter configuration,
including tree parameters, feature selection, and multi-horizon forecasting settings.

Example:
    ```python
    from src.models.config.rf_config import RandomForestConfig

    # Use default configuration
    config = RandomForestConfig()

    # Customize parameters
    config = RandomForestConfig(
        n_estimators=200,
        max_depth=15,
        max_features_per_horizon=30,
        n_jobs=-1
    )

    # Convert to scikit-learn params
    sklearn_params = config.to_sklearn_params()
    ```
"""

from typing import Any

from pydantic import BaseModel, Field, field_validator

from src.utils.logger import get_logger

logger = get_logger(__name__)


class RandomForestConfig(BaseModel):
    """Configuration for Random Forest model training and prediction.

    This class defines all hyperparameters for Random Forest models, including:
    - Tree structure parameters (n_estimators, max_depth, min_samples)
    - Bootstrap and out-of-bag scoring
    - Horizon-specific feature selection
    - Time series parameters for multi-horizon forecasting

    Attributes:
        n_estimators: Number of trees in the forest. Higher values improve accuracy
            but increase training time and memory usage.
        max_depth: Maximum tree depth. None means nodes expand until all leaves are pure
            or contain min_samples_split samples.
        min_samples_split: Minimum samples required to split an internal node.
        min_samples_leaf: Minimum samples required to be at a leaf node.
        max_features: Number of features to consider for best split.
            - "sqrt": sqrt(n_features)
            - "log2": log2(n_features)
            - int: exact number of features
            - float: fraction of features
        bootstrap: Whether to use bootstrap samples when building trees.
        oob_score: Whether to use out-of-bag samples to estimate generalization error.
            Only available if bootstrap=True.
        max_features_per_horizon: Maximum number of features to select per horizon
            after horizon-safe filtering. Uses feature importance for selection.
        feature_selection_method: Method for selecting top features per horizon.
            - "importance": Use RandomForest feature importance
            - "mutual_info": Use mutual information (future enhancement)
        importance_threshold: Minimum feature importance threshold (0.0 to 1.0).
            Features with importance below this are filtered out.
        periods_per_day: Number of time periods per day in the data (e.g., 48 for 30-min).
            Used to calculate minimum lags for horizon-safe features.
        n_jobs: Number of parallel jobs. -1 uses all processors.
        random_state: Random seed for reproducibility.
    """

    # Tree parameters
    n_estimators: int = Field(default=100, ge=1, le=1000)
    max_depth: int | None = Field(default=None, ge=1, le=100)
    min_samples_split: int = Field(default=2, ge=2, le=100)
    min_samples_leaf: int = Field(default=1, ge=1, le=100)
    max_features: str | int | float = Field(default="sqrt")

    # Bootstrap and OOB
    bootstrap: bool = Field(default=True)
    oob_score: bool = Field(default=True)

    # Feature selection
    max_features_per_horizon: int = Field(default=50, ge=1, le=500)
    feature_selection_method: str = Field(default="importance")
    importance_threshold: float = Field(default=0.01, ge=0.0, le=1.0)

    # Time series parameters
    periods_per_day: int = Field(default=48, ge=1, le=288)

    # Execution parameters
    n_jobs: int = Field(default=-1, ge=-1)
    random_state: int = Field(default=42, ge=0)

    @field_validator("max_features")
    @classmethod
    def validate_max_features(cls, v: str | int | float) -> str | int | float:
        """Validate max_features parameter.

        Args:
            v: max_features value.

        Returns:
            Validated max_features value.

        Raises:
            ValueError: If max_features is invalid.
        """
        if isinstance(v, str):
            valid_strings = {"sqrt", "log2"}
            if v not in valid_strings:
                msg = f"max_features string must be one of {valid_strings}, got '{v}'"
                raise ValueError(msg)
        elif isinstance(v, float):
            if not 0.0 < v <= 1.0:
                msg = f"max_features as float must be in (0.0, 1.0], got {v}"
                raise ValueError(msg)
        elif isinstance(v, int):
            if v < 1:
                msg = f"max_features as int must be >= 1, got {v}"
                raise ValueError(msg)
        return v

    @field_validator("feature_selection_method")
    @classmethod
    def validate_feature_selection_method(cls, v: str) -> str:
        """Validate feature selection method.

        Args:
            v: Feature selection method name.

        Returns:
            Validated method name.

        Raises:
            ValueError: If method is not supported.
        """
        valid_methods = {"importance", "mutual_info"}
        if v not in valid_methods:
            msg = f"feature_selection_method must be one of {valid_methods}, got '{v}'"
            raise ValueError(msg)
        return v

    @field_validator("oob_score")
    @classmethod
    def validate_oob_score(cls, v: bool, info: Any) -> bool:
        """Validate oob_score is consistent with bootstrap.

        Args:
            v: oob_score value.
            info: Validation context with other field values.

        Returns:
            Validated oob_score value.
        """
        data = info.data if hasattr(info, "data") else {}
        bootstrap = data.get("bootstrap", True)

        if v and not bootstrap:
            logger.warning(
                "oob_score is True but bootstrap is False. "
                "Out-of-bag score requires bootstrap=True. Setting oob_score to False."
            )
            return False
        return v

    def to_sklearn_params(self) -> dict[str, Any]:
        """Convert configuration to scikit-learn RandomForestRegressor parameters.

        Returns:
            Dictionary of parameters ready for sklearn.ensemble.RandomForestRegressor.

        Example:
            ```python
            config = RandomForestConfig(n_estimators=200, max_depth=15)
            params = config.to_sklearn_params()
            # Use with RandomForestRegressor
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(**params)
            ```
        """
        params = {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "max_features": self.max_features,
            "bootstrap": self.bootstrap,
            "oob_score": self.oob_score,
            "n_jobs": self.n_jobs,
            "random_state": self.random_state,
            "verbose": 0,
        }

        logger.debug("Generated scikit-learn RandomForest parameters: %s", params)
        return params

    model_config = {
        "frozen": False,  # Allow modification after creation
        "validate_assignment": True,  # Validate on attribute assignment
        "extra": "forbid",  # Raise error on extra fields
    }
