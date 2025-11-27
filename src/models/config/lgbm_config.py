"""LightGBM model configuration schema.

This module provides Pydantic models for LightGBM hyperparameter configuration,
including boosting parameters, regularization, early stopping, and Optuna
optimization settings.

Example:
    ```python
    from src.models.config.lgbm_config import LGBMConfig

    # Use default configuration
    config = LGBMConfig()

    # Customize parameters
    config = LGBMConfig(
        n_estimators=500,
        learning_rate=0.1,
        optimize_hyperparams=True,
        n_trials=50
    )

    # Convert to dict for LightGBM
    lgbm_params = config.to_lgbm_params()
    ```
"""

from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from src.utils.logger import get_logger

logger = get_logger(__name__)


class LGBMConfig(BaseModel):
    """Configuration for LightGBM model training and hyperparameter optimization.

    This class defines all hyperparameters for LightGBM models, including:
    - Boosting parameters (n_estimators, learning_rate, tree structure)
    - Subsampling parameters (feature_fraction, bagging)
    - Regularization (L1/L2 lambda)
    - Early stopping configuration
    - Optuna hyperparameter optimization settings

    Attributes:
        n_estimators: Number of boosting rounds. Higher values improve accuracy
            but increase training time and overfitting risk.
        learning_rate: Step size shrinkage to prevent overfitting. Lower values
            require more boosting rounds.
        num_leaves: Maximum number of leaves in one tree. Controls model complexity.
        max_depth: Maximum tree depth. -1 means no limit. Use with num_leaves.
        min_child_samples: Minimum number of samples per leaf. Prevents overfitting.
        feature_fraction: Fraction of features to use per tree (0.0 to 1.0).
        bagging_fraction: Fraction of data to use per iteration (0.0 to 1.0).
        bagging_freq: Frequency for bagging. 0 means disable bagging.
        lambda_l1: L1 regularization term. Helps with feature sparsity.
        lambda_l2: L2 regularization term. Reduces overfitting.
        early_stopping_rounds: Stop if no improvement for N rounds. None to disable.
        categorical_features: List of categorical feature column names.
        optimize_hyperparams: Whether to use Optuna for hyperparameter optimization.
        n_trials: Number of Optuna trials for hyperparameter search.
        cv_folds: Number of cross-validation folds for Optuna.
        objective: LightGBM objective function. "regression" for load forecasting.
        metric: Evaluation metric. "mae" for mean absolute error.
        verbosity: LightGBM verbosity level. -1 for silent, 0 for warning, 1+ for info.
        n_jobs: Number of parallel threads. -1 uses all cores.
        random_state: Random seed for reproducibility.
    """

    # Boosting parameters
    n_estimators: int = Field(default=1000, ge=1, le=10000)
    learning_rate: float = Field(default=0.05, gt=0.0, le=1.0)
    num_leaves: int = Field(default=31, ge=2, le=1024)
    max_depth: int = Field(default=-1, ge=-1, le=100)
    min_child_samples: int = Field(default=20, ge=1, le=1000)

    # Subsampling parameters
    feature_fraction: float = Field(default=0.8, gt=0.0, le=1.0)
    bagging_fraction: float = Field(default=0.8, gt=0.0, le=1.0)
    bagging_freq: int = Field(default=5, ge=0, le=100)

    # Regularization
    lambda_l1: float = Field(default=0.0, ge=0.0, le=100.0)
    lambda_l2: float = Field(default=0.0, ge=0.0, le=100.0)

    # Early stopping
    early_stopping_rounds: int | None = Field(default=50, ge=1, le=500)

    # Feature configuration
    categorical_features: list[str] = Field(default_factory=list)

    # Optuna optimization
    optimize_hyperparams: bool = Field(default=True)
    n_trials: int = Field(default=100, ge=1, le=1000)
    cv_folds: int = Field(default=5, ge=2, le=10)

    # Model behavior
    objective: str = Field(default="regression")
    metric: str = Field(default="mae")
    verbosity: int = Field(default=-1, ge=-1, le=2)
    n_jobs: int = Field(default=-1, ge=-1)
    random_state: int = Field(default=42, ge=0)

    @model_validator(mode="after")
    def validate_num_leaves_max_depth(self) -> "LGBMConfig":
        """Validate num_leaves is reasonable for the max_depth.

        Returns:
            Validated configuration.

        Raises:
            ValueError: If num_leaves exceeds 2^max_depth when max_depth is set.
        """
        if self.max_depth > 0:
            max_possible_leaves = 2**self.max_depth
            if self.num_leaves > max_possible_leaves:
                msg = (
                    f"num_leaves ({self.num_leaves}) should be <= 2^max_depth ({max_possible_leaves}) "
                    f"when max_depth is set to {self.max_depth}"
                )
                raise ValueError(msg)
        return self

    @field_validator("bagging_freq")
    @classmethod
    def validate_bagging_freq(cls, v: int, info: Any) -> int:
        """Validate bagging_freq is consistent with bagging_fraction.

        Args:
            v: Bagging frequency value.
            info: Validation context with other field values.

        Returns:
            Validated bagging_freq value.
        """
        data = info.data if hasattr(info, "data") else {}
        bagging_fraction = data.get("bagging_fraction", 1.0)

        if v > 0 and bagging_fraction == 1.0:
            logger.warning(
                "bagging_freq is set to %d but bagging_fraction is 1.0. "
                "Bagging will be disabled. Set bagging_fraction < 1.0 to enable.",
                v,
            )
        return v

    @field_validator("objective")
    @classmethod
    def validate_objective(cls, v: str) -> str:
        """Validate objective is supported.

        Args:
            v: Objective function name.

        Returns:
            Validated objective name.

        Raises:
            ValueError: If objective is not supported for load forecasting.
        """
        valid_objectives = {
            "regression",
            "regression_l1",
            "regression_l2",
            "huber",
            "fair",
            "poisson",
            "quantile",
            "mape",
        }
        if v not in valid_objectives:
            msg = f"Objective '{v}' not supported. Valid options: {valid_objectives}"
            raise ValueError(msg)
        return v

    @field_validator("metric")
    @classmethod
    def validate_metric(cls, v: str) -> str:
        """Validate metric is supported.

        Args:
            v: Metric name.

        Returns:
            Validated metric name.

        Raises:
            ValueError: If metric is not supported.
        """
        valid_metrics = {
            "mae",
            "mse",
            "rmse",
            "mape",
            "huber",
            "fair",
            "poisson",
            "quantile",
        }
        if v not in valid_metrics:
            msg = f"Metric '{v}' not supported. Valid options: {valid_metrics}"
            raise ValueError(msg)
        return v

    def to_lgbm_params(self) -> dict[str, Any]:
        """Convert configuration to LightGBM parameter dictionary.

        Returns:
            Dictionary of parameters ready for lightgbm.train() or lgb.LGBMRegressor.

        Example:
            ```python
            config = LGBMConfig(n_estimators=500, learning_rate=0.1)
            params = config.to_lgbm_params()
            # Use with lightgbm.train()
            model = lgb.train(params, train_data, num_boost_round=params['n_estimators'])
            ```
        """
        params = {
            "objective": self.objective,
            "metric": self.metric,
            "num_leaves": self.num_leaves,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "n_estimators": self.n_estimators,
            "min_child_samples": self.min_child_samples,
            "subsample": self.bagging_fraction,
            "subsample_freq": self.bagging_freq,
            "colsample_bytree": self.feature_fraction,
            "reg_alpha": self.lambda_l1,
            "reg_lambda": self.lambda_l2,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
            "verbosity": self.verbosity,
            "force_col_wise": True,  # More efficient for wide datasets
        }

        # Add categorical features if specified
        if self.categorical_features:
            params["categorical_feature"] = self.categorical_features

        logger.debug("Generated LightGBM parameters: %s", params)
        return params

    def get_optuna_search_space(self) -> dict[str, tuple[Any, Any]]:
        """Get hyperparameter search space for Optuna optimization.

        Returns:
            Dictionary mapping parameter names to (min, max) tuples or lists of values.
            Used to define Optuna trial suggestions.

        Example:
            ```python
            config = LGBMConfig()
            search_space = config.get_optuna_search_space()
            # Returns:
            # {
            #     "learning_rate": (0.01, 0.3),
            #     "num_leaves": (20, 100),
            #     ...
            # }
            ```
        """
        search_space = {
            "learning_rate": (0.01, 0.3),
            "num_leaves": (20, 100),
            "max_depth": (3, 12),
            "min_child_samples": (10, 100),
            "feature_fraction": (0.6, 1.0),
            "bagging_fraction": (0.6, 1.0),
            "bagging_freq": (1, 10),
            "lambda_l1": (0.0, 10.0),
            "lambda_l2": (0.0, 10.0),
        }
        logger.debug("Generated Optuna search space with %d parameters", len(search_space))
        return search_space

    model_config = {
        "frozen": False,  # Allow modification after creation
        "validate_assignment": True,  # Validate on attribute assignment
        "extra": "forbid",  # Raise error on extra fields
    }
