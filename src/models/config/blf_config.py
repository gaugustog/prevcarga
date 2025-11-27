"""BLF (Baseline Load Forecast) predictor configuration schema.

This module provides Pydantic models for BLF predictor configuration,
including correction parameters, error modeling, confidence intervals,
and operational settings for intraday forecast corrections.

Example:
    ```python
    from src.models.config.blf_config import BLFConfig

    # Use default configuration
    config = BLFConfig()

    # Customize parameters for Brazilian SIN
    config = BLFConfig(
        correction_window_hours=8,
        min_observations=5,
        use_error_model=True,
        error_model_type="ridge",
        periods_per_day=48,  # Semi-hourly Brazilian grid
        subsystem="SECO",
        correction_mode="multiplicative"
    )

    # Convert to dict for use
    params = config.model_dump()
    ```
"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Validation thresholds
LARGE_CORRECTION_WINDOW_THRESHOLD = 12
LOW_DECAY_FACTOR_THRESHOLD = 0.8
MIN_ERROR_MODEL_OBSERVATIONS = 5
LARGE_MAX_CORRECTION_THRESHOLD = 50.0
LOW_CONFIDENCE_LEVEL_THRESHOLD = 0.8

# Brazilian SIN constants
VALID_SUBSYSTEMS = {"SECO", "S", "NE", "N"}
BRAZIL_TIMEZONE = "America/Sao_Paulo"


class BLFConfig(BaseModel):
    """Configuration for BLF (Baseline Load Forecast) predictor.

    This class defines all parameters for the BLF predictor, including:
    - Correction parameters (window size, minimum observations, decay factor)
    - Error modeling configuration (Ridge/Lasso regression)
    - Confidence interval settings
    - Operational constraints (history retention)
    - Brazilian SIN domain settings (subsystem, timezone, periods)

    The BLF predictor updates base model predictions using real-time observations
    to improve intraday forecast accuracy.

    Attributes:
        correction_window_hours: Number of recent hours to use for correction factor
            calculation. Larger windows provide more stable corrections but may be
            less responsive to recent changes.
        min_observations: Minimum number of observations required before applying
            corrections. Prevents corrections based on insufficient data.
        error_decay_factor: Exponential decay factor for weighting recent errors more
            heavily (0.0 to 1.0). Higher values give more weight to recent errors.
        max_correction_pct: Maximum correction percentage (positive or negative) to
            apply to base predictions. Prevents extreme corrections from outliers.
        use_error_model: Whether to train a statistical error model (Ridge/Lasso) on
            correction history to predict future errors.
        error_model_type: Type of error model to use ("ridge" or "lasso").
        error_model_alpha: Regularization strength for error model (L1/L2 penalty).
        confidence_level: Confidence level for prediction intervals (0.0 to 1.0).
            Default 0.95 produces 95% confidence bounds.
        max_history_days: Maximum number of days of correction history to retain.
            Prevents unbounded memory growth.
        periods_per_day: Number of periods per day (48 for semi-hourly Brazilian SIN,
            24 for hourly). Affects correction factor granularity.
        timezone: Timezone for timestamp normalization. Brazilian SIN uses
            America/Sao_Paulo.
        subsystem: Brazilian SIN subsystem identifier (SECO, S, NE, N). Affects
            correction behavior and can be used for subsystem-specific parameters.
        correction_mode: Mode for applying corrections. "additive" adds raw error
            correction, "multiplicative" applies ratio-based correction (preferred
            for load forecasting as errors scale with load magnitude).
        horizon_decay_factors: Optional dictionary mapping horizon (0-8) to decay
            factor. Allows more aggressive corrections for D+0 vs conservative for D+2+.
    """

    # Correction parameters
    correction_window_hours: int = Field(default=6, ge=1, le=24)
    min_observations: int = Field(default=3, ge=1, le=100)
    error_decay_factor: float = Field(default=0.95, ge=0.0, le=1.0)
    max_correction_pct: float = Field(default=20.0, ge=0.0, le=100.0)

    # Error model parameters
    use_error_model: bool = Field(default=True)
    error_model_type: Literal["ridge", "lasso"] = Field(default="ridge")
    error_model_alpha: float = Field(default=1.0, ge=0.0, le=1000.0)

    # Confidence interval parameters
    confidence_level: float = Field(default=0.95, ge=0.5, le=0.999)

    # Operational parameters
    max_history_days: int = Field(default=7, ge=1, le=30)

    # Brazilian SIN domain parameters
    periods_per_day: int = Field(default=48, ge=24, le=96)
    timezone: str = Field(default=BRAZIL_TIMEZONE)
    subsystem: str | None = Field(default=None)
    correction_mode: Literal["additive", "multiplicative"] = Field(default="additive")
    horizon_decay_factors: dict[int, float] | None = Field(default=None)

    @field_validator("correction_window_hours")
    @classmethod
    def validate_correction_window(cls, v: int) -> int:
        """Validate correction window is reasonable.

        Args:
            v: Correction window hours value.

        Returns:
            Validated correction window value.
        """
        if v > LARGE_CORRECTION_WINDOW_THRESHOLD:
            logger.warning(
                "Large correction window: %d hours. "
                "This may reduce responsiveness to recent changes.",
                v,
            )
        return v

    @field_validator("error_decay_factor")
    @classmethod
    def validate_decay_factor(cls, v: float) -> float:
        """Validate decay factor is reasonable.

        Args:
            v: Decay factor value.

        Returns:
            Validated decay factor value.
        """
        if v < LOW_DECAY_FACTOR_THRESHOLD:
            logger.warning(
                "Low decay factor: %.2f. This gives very little weight to older observations.",
                v,
            )
        return v

    @model_validator(mode="after")
    def validate_error_model_config(self) -> "BLFConfig":
        """Validate error model configuration is consistent.

        Returns:
            Validated configuration.
        """
        if self.use_error_model and self.min_observations < MIN_ERROR_MODEL_OBSERVATIONS:
            logger.warning(
                "Error model enabled with min_observations=%d. "
                "Consider min_observations >= %d for stable error model training.",
                self.min_observations,
                MIN_ERROR_MODEL_OBSERVATIONS,
            )
        return self

    @field_validator("max_correction_pct")
    @classmethod
    def validate_max_correction(cls, v: float) -> float:
        """Validate maximum correction percentage.

        Args:
            v: Maximum correction percentage.

        Returns:
            Validated max correction value.
        """
        if v > LARGE_MAX_CORRECTION_THRESHOLD:
            logger.warning(
                "Large max correction percentage: %.1f%%. "
                "This may allow unrealistic forecast adjustments.",
                v,
            )
        return v

    @field_validator("confidence_level")
    @classmethod
    def validate_confidence_level(cls, v: float) -> float:
        """Validate confidence level is in reasonable range.

        Args:
            v: Confidence level value.

        Returns:
            Validated confidence level.
        """
        if v < LOW_CONFIDENCE_LEVEL_THRESHOLD:
            logger.warning(
                "Low confidence level: %.2f. "
                "This produces narrow confidence intervals that may underestimate uncertainty.",
                v,
            )
        return v

    @field_validator("subsystem")
    @classmethod
    def validate_subsystem(cls, v: str | None) -> str | None:
        """Validate subsystem is a valid Brazilian SIN subsystem.

        Args:
            v: Subsystem identifier.

        Returns:
            Validated subsystem identifier.

        Raises:
            ValueError: If subsystem is not valid.
        """
        if v is not None and v not in VALID_SUBSYSTEMS:
            msg = f"Invalid subsystem '{v}'. Must be one of: {VALID_SUBSYSTEMS}"
            raise ValueError(msg)
        return v

    @field_validator("horizon_decay_factors")
    @classmethod
    def validate_horizon_decay_factors(
        cls, v: dict[int, float] | None
    ) -> dict[int, float] | None:
        """Validate horizon decay factors are in valid range.

        Args:
            v: Dictionary mapping horizon to decay factor.

        Returns:
            Validated horizon decay factors.

        Raises:
            ValueError: If any decay factor is out of range.
        """
        if v is None:
            return v
        for horizon, decay in v.items():
            if not 0 <= horizon <= 8:
                msg = f"Horizon {horizon} out of range [0, 8]"
                raise ValueError(msg)
            if not 0.0 <= decay <= 1.0:
                msg = f"Decay factor {decay} for horizon {horizon} out of range [0.0, 1.0]"
                raise ValueError(msg)
        return v

    @model_validator(mode="after")
    def validate_periods_timezone(self) -> "BLFConfig":
        """Validate periods_per_day is appropriate for timezone.

        Returns:
            Validated configuration.
        """
        if self.timezone == BRAZIL_TIMEZONE and self.periods_per_day != 48:
            logger.warning(
                "Brazilian SIN uses semi-hourly (48 periods/day) resolution. "
                "periods_per_day=%d may not match operational requirements.",
                self.periods_per_day,
            )
        return self

    model_config = {
        "frozen": False,  # Allow modification after creation
        "validate_assignment": True,  # Validate on attribute assignment
        "extra": "forbid",  # Raise error on extra fields
    }
