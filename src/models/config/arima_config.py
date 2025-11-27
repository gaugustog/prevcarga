"""ARIMA/SARIMA model configuration schema.

This module provides Pydantic models for ARIMA and SARIMA hyperparameter
configuration, including seasonal parameters, information criteria, and
training constraints for demand mean forecasting.

Example:
    ```python
    from src.models.config.arima_config import ARIMAConfig

    # Use default configuration
    config = ARIMAConfig()

    # Customize parameters for non-seasonal ARIMA
    config = ARIMAConfig(
        seasonal=False,
        max_p=3,
        max_d=1,
        max_q=3,
        information_criterion="bic"
    )

    # Customize for SARIMA with weekly seasonality
    config = ARIMAConfig(
        seasonal=True,
        season_length=7,
        max_P=2,
        max_D=1,
        max_Q=2,
        stepwise=True
    )
    ```
"""

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ARIMAConfig(BaseModel):
    """Configuration for ARIMA/SARIMA demand mean forecasting.

    This class defines all hyperparameters for ARIMA and SARIMA models,
    including:
    - Non-seasonal parameters (p, d, q)
    - Seasonal parameters (P, D, Q, season_length)
    - Model selection criteria (AIC, BIC, AICc)
    - Search strategy (stepwise vs exhaustive)
    - Training constraints (minimum data requirements)

    Attributes:
        seasonal: Whether to use seasonal ARIMA (SARIMA). Default True for
            weekly patterns in daily demand data.
        season_length: Number of periods in seasonal cycle. Default 7 for
            weekly seasonality in daily data.
        max_p: Maximum order of non-seasonal autoregressive term (AR).
            Controls how many lagged values to use.
        max_d: Maximum degree of non-seasonal differencing. Used to make
            series stationary. Usually 0, 1, or 2.
        max_q: Maximum order of non-seasonal moving average term (MA).
            Controls how many lagged forecast errors to use.
        max_P: Maximum order of seasonal autoregressive term (SAR).
            Used only if seasonal=True.
        max_D: Maximum degree of seasonal differencing. Usually 0 or 1.
            Used only if seasonal=True.
        max_Q: Maximum order of seasonal moving average term (SMA).
            Used only if seasonal=True.
        stepwise: Whether to use stepwise search for faster parameter selection.
            If True, uses heuristic search. If False, exhaustive grid search.
        approximation: Whether to use approximation for likelihood computation.
            Faster but less accurate. Generally set to False for production.
        information_criterion: Model selection criterion. "aic" (Akaike),
            "bic" (Bayesian), or "aicc" (corrected Akaike). Lower is better.
        min_training_days: Minimum number of daily observations required for
            training. Model will raise error if data is insufficient.
        prediction_intervals: Confidence levels for prediction intervals.
            Default [80, 95] for 80% and 95% intervals.
    """

    # Seasonal configuration
    seasonal: bool = Field(default=True, description="Enable seasonal ARIMA (SARIMA)")
    season_length: int = Field(
        default=7,
        ge=2,
        le=365,
        description="Seasonal period length (7 for weekly pattern)",
    )

    # Non-seasonal ARIMA parameters (p, d, q)
    max_p: int = Field(default=5, ge=0, le=10, description="Max AR order")
    max_d: int = Field(default=2, ge=0, le=2, description="Max differencing order")
    max_q: int = Field(default=5, ge=0, le=10, description="Max MA order")

    # Seasonal ARIMA parameters (P, D, Q)
    max_P: int = Field(default=2, ge=0, le=5, description="Max seasonal AR order")  # noqa: N815
    max_D: int = Field(default=1, ge=0, le=2, description="Max seasonal differencing")  # noqa: N815
    max_Q: int = Field(default=2, ge=0, le=5, description="Max seasonal MA order")  # noqa: N815

    # Model selection and search
    stepwise: bool = Field(
        default=True,
        description="Use stepwise search for faster parameter selection",
    )
    approximation: bool = Field(
        default=False,
        description="Use approximation for likelihood computation",
    )
    information_criterion: Literal["aic", "bic", "aicc"] = Field(
        default="aic",
        description="Information criterion for model selection",
    )

    # Training constraints
    min_training_days: int = Field(
        default=14,
        ge=7,
        le=365,
        description="Minimum daily observations required for training",
    )

    # Prediction configuration
    prediction_intervals: list[int] = Field(
        default=[80, 95],
        description="Confidence levels for prediction intervals",
    )

    @field_validator("prediction_intervals")
    @classmethod
    def validate_prediction_intervals(cls, v: list[int]) -> list[int]:
        """Validate prediction intervals are valid percentages.

        Args:
            v: List of prediction interval confidence levels.

        Returns:
            Validated list of intervals.

        Raises:
            ValueError: If any interval is not between 1 and 99.
        """
        min_interval = 1
        max_interval = 99
        for interval in v:
            if not min_interval <= interval <= max_interval:
                msg = f"Prediction interval {interval} must be between 1 and 99"
                raise ValueError(msg)

        # Sort intervals in ascending order
        return sorted(v)

    @model_validator(mode="after")
    def validate_seasonal_params(self) -> "ARIMAConfig":
        """Validate seasonal parameters are reasonable.

        Returns:
            Validated configuration.

        Raises:
            ValueError: If seasonal parameters are invalid.
        """
        min_season_length = 2
        weekly_season_length = 7

        if self.seasonal:
            if self.season_length < min_season_length:
                msg = f"season_length must be >= 2 when seasonal=True, got {self.season_length}"
                raise ValueError(msg)

            # Warn if season_length doesn't match weekly pattern for daily data
            if self.season_length != weekly_season_length:
                logger.warning(
                    "season_length is %d (expected 7 for weekly pattern in daily data)",
                    self.season_length,
                )

        return self

    @model_validator(mode="after")
    def validate_min_training_days(self) -> "ARIMAConfig":
        """Validate minimum training days is sufficient for model complexity.

        Returns:
            Validated configuration.

        Raises:
            ValueError: If min_training_days is too small for the model.
        """
        # Calculate minimum observations needed based on parameters
        min_required = (self.max_p + self.max_d + self.max_q) * 2

        if self.seasonal:
            min_required += (self.max_P + self.max_D + self.max_Q) * self.season_length * 2

        if self.min_training_days < min_required:
            logger.warning(
                "min_training_days (%d) may be too small for model complexity. "
                "Recommended minimum: %d days based on max parameters.",
                self.min_training_days,
                min_required,
            )

        return self

    def to_statsforecast_params(self) -> dict[str, Any]:
        """Convert configuration to statsforecast AutoARIMA parameters.

        Returns:
            Dictionary of parameters ready for statsforecast.models.AutoARIMA.

        Example:
            ```python
            config = ARIMAConfig(seasonal=True, season_length=7)
            params = config.to_statsforecast_params()
            # Use with AutoARIMA
            from statsforecast.models import AutoARIMA
            model = AutoARIMA(**params)
            ```
        """
        params = {
            "season_length": self.season_length if self.seasonal else 1,
            "seasonal": self.seasonal,
            "max_p": self.max_p,
            "max_d": self.max_d,
            "max_q": self.max_q,
            "max_P": self.max_P if self.seasonal else 0,
            "max_D": self.max_D if self.seasonal else 0,
            "max_Q": self.max_Q if self.seasonal else 0,
            "stepwise": self.stepwise,
            "approximation": self.approximation,
        }

        logger.debug(
            "Generated statsforecast AutoARIMA parameters: seasonal=%s, season_length=%d",
            self.seasonal,
            self.season_length,
        )
        return params

    model_config = {
        "frozen": False,  # Allow modification after creation
        "validate_assignment": True,  # Validate on attribute assignment
        "extra": "forbid",  # Raise error on extra fields
    }
