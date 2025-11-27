"""SVM Profile model configuration schema.

This module provides Pydantic models for SVM profile forecasting configuration,
including hyperparameter optimization settings, profile bounds, and training
constraints for semi-hourly profile ratio prediction.

Example:
    ```python
    from src.models.config.svm_config import SVMProfileConfig

    # Use default configuration
    config = SVMProfileConfig()

    # Customize parameters
    config = SVMProfileConfig(
        kernel="rbf",
        C=10.0,
        gamma="scale",
        optimize_hyperparams=True,
        cv_folds=5,
        min_profile_ratio=0.1,
        max_profile_ratio=5.0
    )

    # Get GridSearchCV parameter grid
    param_grid = config.get_param_grid()
    ```
"""

from typing import Any

from pydantic import BaseModel, Field, field_validator

from src.utils.logger import get_logger

logger = get_logger(__name__)


class SVMProfileConfig(BaseModel):
    """Configuration for SVM profile model training and hyperparameter optimization.

    This class defines all hyperparameters for SVM profile models, including:
    - Kernel configuration (RBF is default for profile forecasting)
    - Regularization parameters (C, epsilon)
    - Gamma parameter for RBF kernel
    - Hyperparameter optimization via GridSearchCV
    - Profile ratio bounds for clipping predictions
    - Parallel training settings

    Attributes:
        kernel: SVM kernel type. "rbf" (radial basis function) is recommended
            for profile forecasting as it captures non-linear patterns.
        C: Regularization parameter. Higher values = less regularization.
            Controls the trade-off between smooth decision boundary and
            classifying training points correctly.
        gamma: Kernel coefficient for RBF. "scale" uses 1/(n_features * X.var()),
            "auto" uses 1/n_features. Can also be a float.
        epsilon: Epsilon in epsilon-SVR model. Specifies the epsilon-tube
            within which no penalty is associated with training loss.
        optimize_hyperparams: Whether to use GridSearchCV for hyperparameter
            optimization. Recommended for production use.
        param_grid: Custom parameter grid for GridSearchCV. If None, uses
            default grid from get_param_grid().
        cv_folds: Number of cross-validation folds for GridSearchCV.
        min_profile_ratio: Minimum allowable profile ratio. Values below this
            will be clipped. Default 0.1 (10% of daily mean).
        max_profile_ratio: Maximum allowable profile ratio. Values above this
            will be clipped. Default 5.0 (500% of daily mean).
        min_samples_per_period: Minimum number of training samples required per
            semi-hourly period (0-47). Periods with fewer samples will be skipped.
        n_jobs: Number of parallel jobs for training and optimization.
            -1 uses all available cores.
        random_state: Random seed for reproducibility.
    """

    # SVM parameters
    kernel: str = Field(default="rbf", description="SVM kernel type")
    C: float = Field(default=1.0, gt=0.0, le=1000.0, description="Regularization parameter")  # noqa: N815
    gamma: str | float = Field(
        default="scale",
        description="Kernel coefficient (scale/auto or float)",
    )
    epsilon: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Epsilon for epsilon-SVR",
    )

    # Hyperparameter optimization
    optimize_hyperparams: bool = Field(
        default=True,
        description="Use GridSearchCV for hyperparameter tuning",
    )
    param_grid: dict[str, list[Any]] | None = Field(
        default=None,
        description="Custom parameter grid for GridSearchCV",
    )
    cv_folds: int = Field(default=5, ge=2, le=10, description="CV folds for GridSearchCV")

    # Profile ratio bounds
    min_profile_ratio: float = Field(
        default=0.1,
        gt=0.0,
        le=1.0,
        description="Minimum allowable profile ratio",
    )
    max_profile_ratio: float = Field(
        default=5.0,
        ge=1.0,
        le=10.0,
        description="Maximum allowable profile ratio",
    )

    # Training constraints
    min_samples_per_period: int = Field(
        default=50,
        ge=10,
        le=500,
        description="Minimum samples required per semi-hourly period",
    )

    # Parallelization
    n_jobs: int = Field(default=-1, ge=-1, description="Number of parallel jobs")
    random_state: int = Field(default=42, ge=0, description="Random seed")

    @field_validator("kernel")
    @classmethod
    def validate_kernel(cls, v: str) -> str:
        """Validate kernel is supported.

        Args:
            v: Kernel name.

        Returns:
            Validated kernel name.

        Raises:
            ValueError: If kernel is not supported.
        """
        valid_kernels = {"rbf", "linear", "poly", "sigmoid"}
        if v not in valid_kernels:
            msg = f"Kernel '{v}' not supported. Valid options: {valid_kernels}"
            raise ValueError(msg)

        if v != "rbf":
            logger.warning(
                "Using kernel '%s'. RBF kernel is recommended for profile forecasting.",
                v,
            )

        return v

    @field_validator("gamma")
    @classmethod
    def validate_gamma(cls, v: str | float) -> str | float:
        """Validate gamma parameter.

        Args:
            v: Gamma value.

        Returns:
            Validated gamma value.

        Raises:
            ValueError: If gamma is invalid.
        """
        if isinstance(v, str):
            valid_gamma_str = {"scale", "auto"}
            if v not in valid_gamma_str:
                msg = f"Gamma string '{v}' not supported. Valid options: {valid_gamma_str}"
                raise ValueError(msg)
        elif isinstance(v, (int, float)):
            if v <= 0.0:
                msg = f"Gamma must be positive, got {v}"
                raise ValueError(msg)
        else:
            msg = f"Gamma must be str or float, got {type(v)}"
            raise ValueError(msg)

        return v

    def get_param_grid(self) -> dict[str, list[Any]]:
        """Get hyperparameter search space for GridSearchCV.

        Returns:
            Dictionary mapping parameter names to lists of values to search.
            If custom param_grid is provided, returns that. Otherwise, returns
            default grid suitable for profile forecasting.

        Example:
            ```python
            config = SVMProfileConfig()
            param_grid = config.get_param_grid()
            # Returns:
            # {
            #     "C": [0.1, 1, 10, 100],
            #     "gamma": ["scale", "auto", 0.001, 0.01, 0.1],
            #     "epsilon": [0.01, 0.1, 0.2]
            # }
            ```
        """
        if self.param_grid is not None:
            logger.debug("Using custom parameter grid with %d parameters", len(self.param_grid))
            return self.param_grid

        # Default grid for profile forecasting
        default_grid = {
            "C": [0.1, 1, 10, 100],
            "gamma": ["scale", "auto", 0.001, 0.01, 0.1],
            "epsilon": [0.01, 0.1, 0.2],
        }

        logger.debug("Using default parameter grid with %d parameters", len(default_grid))
        return default_grid

    def to_svr_params(self) -> dict[str, Any]:
        """Convert configuration to sklearn SVR parameter dictionary.

        Returns:
            Dictionary of parameters ready for sklearn.svm.SVR.

        Example:
            ```python
            config = SVMProfileConfig(C=10.0, kernel="rbf", gamma="scale")
            params = config.to_svr_params()
            # Use with SVR
            from sklearn.svm import SVR
            model = SVR(**params)
            ```
        """
        params = {
            "kernel": self.kernel,
            "C": self.C,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
        }

        logger.debug("Generated SVR parameters: %s", params)
        return params

    model_config = {
        "frozen": False,  # Allow modification after creation
        "validate_assignment": True,  # Validate on attribute assignment
        "extra": "forbid",  # Raise error on extra fields
    }
