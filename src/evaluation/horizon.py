"""Horizon analyzer for evaluating forecast performance by forecast horizon.

This module provides the HorizonAnalyzer class that breaks down forecast
performance metrics by forecast horizon (D+0, D+1, ..., D+8) to identify
degradation patterns and horizon-specific performance characteristics.

Key Features:
- Per-horizon metrics calculation (h0 through h8)
- Degradation pattern modeling (linear, exponential, power law)
- Skill score computation vs persistence/climatology
- Statistical testing across horizons (ANOVA, Kruskal-Wallis)
- Degradation rate quantification
- Best/worst horizon identification

Example:
    ```python
    from src.evaluation.horizon import HorizonAnalyzer, HorizonConfig
    import pandas as pd

    # Multi-horizon forecast data
    forecasts = pd.DataFrame({
        "h0": [...],  # D+0 (today)
        "h1": [...],  # D+1 (tomorrow)
        "h2": [...],  # D+2
        ...
    })
    actuals = pd.DataFrame({...})  # Same structure

    # Configure analyzer
    config = HorizonConfig(
        horizons=["h0", "h1", "h2", "h3", "h4", "h5", "h6", "h7", "h8"],
        metrics=["mape", "rmse"],
        baseline_horizon="h0",
        include_skill_scores=True,
        degradation_model="exponential"
    )

    # Analyze by horizon
    analyzer = HorizonAnalyzer(config=config)
    result = analyzer.analyze_by_horizon(actuals, forecasts)

    print(f"h0 MAPE: {result.metrics_by_horizon['h0'].mape:.2f}%")
    print(f"h8 MAPE: {result.metrics_by_horizon['h8'].mape:.2f}%")
    print(f"Degradation rate: {result.degradation_rate:.3f} per day")
    ```
"""

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator
from scipy import optimize, stats

from src.evaluation.metrics import MetricsCalculator, MetricsConfig, MetricsResult
from src.utils.logger import get_logger

logger = get_logger(__name__)


class HorizonConfig(BaseModel):
    """Configuration for horizon analysis.

    Attributes:
        horizons: List of horizon names to analyze (e.g., ["h0", "h1", "h2"]).
        metrics: List of metrics to compute per horizon.
            Options: "mape", "mae", "rmse", "mse", "r2"
        baseline_horizon: Reference horizon for relative comparisons (default "h0").
        include_skill_scores: Whether to compute skill scores vs baselines.
        degradation_model: Model type for degradation curve fitting.
            Options: "linear", "exponential", "power_law"
        min_samples: Minimum samples required per horizon for valid analysis.
        alpha: Significance level for statistical tests.

    Example:
        >>> config = HorizonConfig(
        ...     horizons=["h0", "h1", "h2", "h3"],
        ...     metrics=["mape", "rmse"],
        ...     baseline_horizon="h0",
        ...     degradation_model="exponential"
        ... )
    """

    horizons: list[str] = Field(
        default=["h0", "h1", "h2", "h3", "h4", "h5", "h6", "h7", "h8"],
        description="Horizon names to analyze",
    )
    metrics: list[str] = Field(
        default=["mape", "mae", "rmse"],
        description="Metrics to compute per horizon",
    )
    baseline_horizon: str = Field(
        default="h0",
        description="Reference horizon for comparisons",
    )
    include_skill_scores: bool = Field(
        default=True,
        description="Compute skill scores vs persistence/climatology",
    )
    degradation_model: Literal["linear", "exponential", "power_law"] = Field(
        default="exponential",
        description="Degradation curve model type",
    )
    min_samples: int = Field(
        default=30,
        ge=1,
        description="Minimum samples per horizon",
    )
    alpha: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Significance level for tests",
    )

    @field_validator("horizons")
    @classmethod
    def validate_horizons(cls, v: list[str]) -> list[str]:
        """Validate horizon names.

        Args:
            v: List of horizon names.

        Returns:
            Validated list.

        Raises:
            ValueError: If invalid horizon format.
        """
        if not v:
            msg = "horizons cannot be empty"
            raise ValueError(msg)

        # Check format (should be "h" followed by digit(s))
        for horizon in v:
            if not horizon.startswith("h"):
                msg = f"Invalid horizon format '{horizon}'. Must start with 'h' (e.g., 'h0', 'h1')"
                raise ValueError(msg)
            try:
                int(horizon[1:])
            except ValueError as exc:
                msg = f"Invalid horizon format '{horizon}'. Must be 'h' followed by number"
                raise ValueError(msg) from exc

        return v

    @field_validator("metrics")
    @classmethod
    def validate_metrics(cls, v: list[str]) -> list[str]:
        """Validate metrics.

        Args:
            v: List of metrics.

        Returns:
            Validated list.

        Raises:
            ValueError: If invalid metric.
        """
        valid_metrics = {"mape", "mae", "rmse", "mse", "r2"}
        for metric in v:
            if metric not in valid_metrics:
                msg = f"Invalid metric '{metric}'. Must be one of {valid_metrics}"
                raise ValueError(msg)
        return v

    @field_validator("baseline_horizon")
    @classmethod
    def validate_baseline_horizon(cls, v: str, info) -> str:
        """Validate baseline horizon.

        Args:
            v: Baseline horizon name.
            info: Validation info with other fields.

        Returns:
            Validated baseline horizon.

        Raises:
            ValueError: If baseline not in horizons.
        """
        # Note: horizons may not be set yet during validation
        # We'll validate this in __init__ or during runtime
        if not v.startswith("h"):
            msg = f"Invalid baseline_horizon format '{v}'. Must start with 'h'"
            raise ValueError(msg)
        return v


@dataclass
class SkillScores:
    """Skill scores vs baseline forecasts.

    Attributes:
        persistence: Skill score vs persistence forecast (1 - MSE_forecast/MSE_persistence).
        climatology: Skill score vs climatology forecast (1 - MSE_forecast/MSE_climatology).
        random_walk: Skill score vs random walk forecast (1 - MSE_forecast/MSE_rw).
        metadata: Additional metadata about skill score calculation.

    Example:
        >>> scores = SkillScores(
        ...     persistence=0.85,
        ...     climatology=0.92,
        ...     random_walk=0.80
        ... )
    """

    persistence: float
    climatology: float
    random_walk: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "persistence": self.persistence,
            "climatology": self.climatology,
            "random_walk": self.random_walk,
            "metadata": self.metadata.copy(),
        }


@dataclass
class DegradationModel:
    """Fitted degradation model.

    Attributes:
        model_type: Type of model ("linear", "exponential", "power_law").
        parameters: Model parameters (e.g., [a, b] for y = a + b*x).
        degradation_rate: Rate parameter (interpretation depends on model type).
        r_squared: Goodness of fit (R²).
        predictions: Predicted error values for each horizon.

    Example:
        >>> model = DegradationModel(
        ...     model_type="exponential",
        ...     parameters=[2.5, 0.12],
        ...     degradation_rate=0.12,
        ...     r_squared=0.95,
        ...     predictions=[2.5, 2.8, 3.2, 3.6]
        ... )
    """

    model_type: str
    parameters: list[float]
    degradation_rate: float
    r_squared: float
    predictions: np.ndarray = field(default_factory=lambda: np.array([]))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "model_type": self.model_type,
            "parameters": self.parameters,
            "degradation_rate": self.degradation_rate,
            "r_squared": self.r_squared,
            "predictions": self.predictions.tolist(),
        }


@dataclass
class HorizonResult:
    """Results from horizon analysis.

    Attributes:
        metrics_by_horizon: Dictionary mapping horizon names to MetricsResult.
        degradation_model: Fitted degradation model.
        degradation_rate: Rate of error increase per horizon step.
        best_horizons: List of (horizon_name, mape) tuples, best first.
        worst_horizons: List of (horizon_name, mape) tuples, worst first.
        skill_scores_by_horizon: Dictionary mapping horizons to SkillScores (if enabled).
        statistical_tests: Results from statistical tests (ANOVA, etc.).
        degradation_pattern: Identified pattern type ("linear", "exponential", "plateau").
        metadata: Additional metadata about the analysis.

    Example:
        >>> result = HorizonResult(
        ...     metrics_by_horizon={"h0": metrics_h0, "h1": metrics_h1},
        ...     degradation_model=model,
        ...     degradation_rate=0.12,
        ...     best_horizons=[("h0", 2.5), ("h1", 2.8)],
        ...     worst_horizons=[("h8", 8.5), ("h7", 7.2)],
        ...     degradation_pattern="exponential"
        ... )
    """

    metrics_by_horizon: dict[str, MetricsResult]
    degradation_model: DegradationModel
    degradation_rate: float
    best_horizons: list[tuple[str, float]]
    worst_horizons: list[tuple[str, float]]
    skill_scores_by_horizon: dict[str, SkillScores] = field(default_factory=dict)
    statistical_tests: dict[str, Any] = field(default_factory=dict)
    degradation_pattern: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_horizons(self) -> int:
        """Get number of horizons analyzed.

        Returns:
            Number of horizons.
        """
        return len(self.metrics_by_horizon)

    @property
    def best_horizon(self) -> tuple[str, float] | None:
        """Get best performing horizon.

        Returns:
            Tuple of (horizon_name, mape) or None.
        """
        if self.best_horizons:
            return self.best_horizons[0]
        return None

    @property
    def worst_horizon(self) -> tuple[str, float] | None:
        """Get worst performing horizon.

        Returns:
            Tuple of (horizon_name, mape) or None.
        """
        if self.worst_horizons:
            return self.worst_horizons[0]
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "metrics_by_horizon": {
                name: result.to_dict() for name, result in self.metrics_by_horizon.items()
            },
            "degradation_model": self.degradation_model.to_dict(),
            "degradation_rate": self.degradation_rate,
            "best_horizons": self.best_horizons.copy(),
            "worst_horizons": self.worst_horizons.copy(),
            "skill_scores_by_horizon": {
                name: scores.to_dict() for name, scores in self.skill_scores_by_horizon.items()
            },
            "statistical_tests": self.statistical_tests.copy(),
            "degradation_pattern": self.degradation_pattern,
            "metadata": self.metadata.copy(),
        }

    def __repr__(self) -> str:
        """Return string representation.

        Returns:
            String representation.
        """
        return (
            f"HorizonResult("
            f"n_horizons={self.n_horizons}, "
            f"pattern={self.degradation_pattern!r}, "
            f"rate={self.degradation_rate:.4f}, "
            f"best={self.best_horizon}, "
            f"worst={self.worst_horizon})"
        )


class HorizonAnalyzer:
    """Horizon analyzer for forecast performance by forecast horizon.

    Analyzes forecast performance across different forecast horizons (D+0 to D+8)
    to identify degradation patterns, quantify error growth, and compute skill scores.

    Supported Features:
        - Per-horizon metrics (MAPE, MAE, RMSE, R²)
        - Degradation modeling (linear, exponential, power law)
        - Skill scores (vs persistence, climatology, random walk)
        - Statistical testing (ANOVA, Kruskal-Wallis)
        - Degradation pattern identification
        - Best/worst horizon identification

    Degradation Models:
        - Linear: error(h) = a + b*h
        - Exponential: error(h) = a * exp(b*h)
        - Power Law: error(h) = a * h^b

    Skill Scores:
        - Persistence: Improvement over naive "tomorrow = today"
        - Climatology: Improvement over historical mean
        - Random Walk: Improvement over random walk baseline
        - Formula: skill = 1 - (MSE_forecast / MSE_baseline)

    Example:
        >>> config = HorizonConfig(
        ...     horizons=["h0", "h1", "h2", "h3"],
        ...     metrics=["mape", "rmse"],
        ...     degradation_model="exponential"
        ... )
        >>> analyzer = HorizonAnalyzer(config=config)
        >>> result = analyzer.analyze_by_horizon(actuals, forecasts)
        >>> print(result.degradation_rate)
    """

    def __init__(
        self,
        config: HorizonConfig | None = None,
        metrics_config: MetricsConfig | None = None,
    ) -> None:
        """Initialize analyzer.

        Args:
            config: Configuration for horizon analysis.
            metrics_config: Configuration for metrics calculation.
        """
        self.config = config or HorizonConfig()
        self.metrics_calculator = MetricsCalculator(metrics_config or MetricsConfig())

        # Validate baseline_horizon is in horizons
        if self.config.baseline_horizon not in self.config.horizons:
            msg = (
                f"baseline_horizon '{self.config.baseline_horizon}' "
                f"not in horizons {self.config.horizons}"
            )
            raise ValueError(msg)

        logger.debug(
            "Initialized HorizonAnalyzer with %d horizons, degradation_model=%s",
            len(self.config.horizons),
            self.config.degradation_model,
        )

    def analyze_by_horizon(
        self,
        actual: pd.DataFrame | dict[str, np.ndarray],
        forecast: pd.DataFrame | dict[str, np.ndarray],
    ) -> HorizonResult:
        """Analyze forecast performance by horizon.

        Args:
            actual: Actual values. Either DataFrame with horizon columns (h0, h1, ...)
                or dict mapping horizon names to arrays.
            forecast: Forecast values. Same structure as actual.

        Returns:
            HorizonResult with comprehensive analysis.

        Raises:
            ValueError: If input structures don't match or missing horizons.

        Example:
            >>> actuals = pd.DataFrame({"h0": [100, 200], "h1": [110, 210]})
            >>> forecasts = pd.DataFrame({"h0": [98, 202], "h1": [105, 215]})
            >>> result = analyzer.analyze_by_horizon(actuals, forecasts)
        """
        # Convert to dict format if DataFrame
        if isinstance(actual, pd.DataFrame):
            actual_dict = {col: actual[col].to_numpy() for col in actual.columns}
        else:
            actual_dict = actual

        if isinstance(forecast, pd.DataFrame):
            forecast_dict = {col: forecast[col].to_numpy() for col in forecast.columns}
        else:
            forecast_dict = forecast

        # Validate all required horizons are present
        missing_actual = set(self.config.horizons) - set(actual_dict.keys())
        missing_forecast = set(self.config.horizons) - set(forecast_dict.keys())

        if missing_actual:
            msg = f"Missing horizons in actual data: {missing_actual}"
            raise ValueError(msg)

        if missing_forecast:
            msg = f"Missing horizons in forecast data: {missing_forecast}"
            raise ValueError(msg)

        # Step 1: Compute metrics for each horizon
        logger.debug("Computing metrics for %d horizons", len(self.config.horizons))
        metrics_by_horizon = self._compute_metrics_by_horizon(actual_dict, forecast_dict)

        # Step 2: Fit degradation model
        logger.debug("Fitting degradation model: %s", self.config.degradation_model)
        degradation_model = self._fit_degradation_model(metrics_by_horizon)

        # Step 3: Rank horizons
        best_horizons = self._rank_horizons(metrics_by_horizon, ascending=True)
        worst_horizons = self._rank_horizons(metrics_by_horizon, ascending=False)

        # Step 4: Run statistical tests
        statistical_tests = {}
        if len(self.config.horizons) >= 2:
            logger.debug("Running statistical tests")
            statistical_tests = self._run_statistical_tests(actual_dict, forecast_dict)

        # Step 5: Compute skill scores if requested
        skill_scores_by_horizon = {}
        if self.config.include_skill_scores:
            logger.debug("Computing skill scores")
            skill_scores_by_horizon = self._compute_skill_scores(actual_dict, forecast_dict)

        # Step 6: Identify degradation pattern
        degradation_pattern = self.identify_degradation_pattern(degradation_model)

        # Build metadata
        sample_sizes = {h: len(actual_dict[h]) for h in self.config.horizons}
        metadata = {
            "n_horizons": len(self.config.horizons),
            "sample_sizes": sample_sizes,
            "degradation_model_type": self.config.degradation_model,
            "baseline_horizon": self.config.baseline_horizon,
        }

        return HorizonResult(
            metrics_by_horizon=metrics_by_horizon,
            degradation_model=degradation_model,
            degradation_rate=degradation_model.degradation_rate,
            best_horizons=best_horizons,
            worst_horizons=worst_horizons,
            skill_scores_by_horizon=skill_scores_by_horizon,
            statistical_tests=statistical_tests,
            degradation_pattern=degradation_pattern,
            metadata=metadata,
        )

    def compute_degradation(
        self,
        metrics_by_horizon: dict[str, MetricsResult],
        model: str = "exponential",
    ) -> tuple[float, list[float]]:
        """Model error increase with horizon.

        Fits degradation curve to observed errors.

        Args:
            metrics_by_horizon: Dictionary mapping horizon names to MetricsResult.
            model: Model type ("linear", "exponential", "power_law").

        Returns:
            Tuple of (degradation_rate, parameters).

        Raises:
            ValueError: If model fitting fails or invalid model type.

        Example:
            >>> rate, params = analyzer.compute_degradation(metrics, "exponential")
            >>> print(f"Exponential rate: {rate:.4f}")
        """
        # Extract horizon indices and errors
        horizon_indices = []
        errors = []

        for horizon in self.config.horizons:
            if horizon in metrics_by_horizon:
                # Extract numeric index from "h0", "h1", etc.
                h_idx = int(horizon[1:])
                horizon_indices.append(h_idx)
                errors.append(metrics_by_horizon[horizon].mape)

        horizon_indices = np.array(horizon_indices, dtype=np.float64)
        errors = np.array(errors, dtype=np.float64)

        if len(horizon_indices) < 2:
            msg = "Need at least 2 horizons for degradation modeling"
            raise ValueError(msg)

        # Define model functions
        if model == "linear":

            def func(h, a, b):
                return a + b * h

        elif model == "exponential":

            def func(h, a, b):
                return a * np.exp(b * h)

        elif model == "power_law":

            def func(h, a, b):
                # Avoid h=0 issues
                return a * np.power(np.maximum(h, 0.1), b)

        else:
            msg = f"Invalid model type: {model}"
            raise ValueError(msg)

        # Fit model
        try:
            # Provide initial guesses
            if model == "exponential":
                p0 = [errors[0], 0.1]
            elif model == "power_law":
                p0 = [errors[0], 0.5]
            else:  # linear
                p0 = [errors[0], (errors[-1] - errors[0]) / (horizon_indices[-1] - horizon_indices[0])]

            params, _ = optimize.curve_fit(func, horizon_indices, errors, p0=p0, maxfev=10000)
            params_list = params.tolist()

            # Extract degradation rate (second parameter)
            degradation_rate = float(params[1])

        except Exception as e:
            logger.warning("Degradation model fitting failed: %s. Using linear fallback.", e)
            # Fallback to simple linear fit
            params_list = [float(errors[0]), 0.0]
            degradation_rate = 0.0

        return degradation_rate, params_list

    def compute_skill_scores(
        self,
        actuals: dict[str, np.ndarray],
        forecasts: dict[str, np.ndarray],
        baseline: str = "persistence",
    ) -> dict[str, float]:
        """Compute skill scores vs baseline for all horizons.

        Skill score = 1 - (MSE_forecast / MSE_baseline)

        Args:
            actuals: Dictionary mapping horizon names to actual arrays.
            forecasts: Dictionary mapping horizon names to forecast arrays.
            baseline: Baseline type ("persistence", "climatology", "random_walk").

        Returns:
            Dictionary mapping horizon names to skill scores.

        Example:
            >>> skills = analyzer.compute_skill_scores(actuals, forecasts, "persistence")
            >>> print(f"h0 skill: {skills['h0']:.3f}")
        """
        skill_scores = {}

        for horizon in self.config.horizons:
            if horizon not in actuals or horizon not in forecasts:
                continue

            actual = actuals[horizon]
            forecast = forecasts[horizon]

            # Generate baseline forecast
            if baseline == "persistence":
                # Persistence: forecast[t] = actual[t-1]
                baseline_forecast = np.roll(actual, 1)
                # First value has no previous, use actual
                baseline_forecast[0] = actual[0]

            elif baseline == "climatology":
                # Climatology: forecast = mean of actuals
                baseline_forecast = np.full_like(actual, np.mean(actual))

            elif baseline == "random_walk":
                # Random walk: cumsum of differences
                # For simplicity, use persistence as random walk approximation
                baseline_forecast = np.roll(actual, 1)
                baseline_forecast[0] = actual[0]

            else:
                msg = f"Invalid baseline type: {baseline}"
                raise ValueError(msg)

            # Compute MSE
            mse_forecast = float(np.mean((forecast - actual) ** 2))
            mse_baseline = float(np.mean((baseline_forecast - actual) ** 2))

            # Compute skill score
            if mse_baseline > 0:
                skill = 1.0 - (mse_forecast / mse_baseline)
            else:
                skill = 1.0 if mse_forecast == 0 else 0.0

            skill_scores[horizon] = skill

        return skill_scores

    def identify_degradation_pattern(
        self,
        degradation_model: DegradationModel,
    ) -> str:
        """Identify the degradation pattern type.

        Classifies the degradation curve into:
        - "linear": Constant rate of increase
        - "exponential": Accelerating increase
        - "power_law": Sublinear or superlinear increase
        - "plateau": Initial increase then stabilization
        - "unknown": Cannot determine

        Args:
            degradation_model: Fitted degradation model.

        Returns:
            Pattern type string.

        Example:
            >>> pattern = analyzer.identify_degradation_pattern(model)
            >>> print(f"Pattern: {pattern}")
        """
        model_type = degradation_model.model_type
        rate = degradation_model.degradation_rate

        if model_type == "linear":
            return "linear"

        elif model_type == "exponential":
            # If rate is near zero, it's effectively constant (plateau)
            if abs(rate) < 0.01:
                return "plateau"
            # If rate is positive, it's exponential growth
            if rate > 0:
                return "exponential"
            # Negative rate is decay
            return "exponential_decay"

        elif model_type == "power_law":
            # b > 1: superlinear (accelerating)
            # b ~ 1: linear
            # 0 < b < 1: sublinear (decelerating)
            # b ~ 0: plateau
            if abs(rate) < 0.1:
                return "plateau"
            if rate > 1.1:
                return "superlinear"
            if 0.9 <= rate <= 1.1:
                return "linear"
            if 0 < rate < 0.9:
                return "sublinear"
            return "power_law"

        return "unknown"

    def compare_horizons(
        self,
        result: HorizonResult,
        horizon_a: str,
        horizon_b: str,
    ) -> dict[str, Any]:
        """Statistical comparison between two horizons.

        Args:
            result: HorizonResult from analysis.
            horizon_a: First horizon name.
            horizon_b: Second horizon name.

        Returns:
            Dictionary with comparison metrics.

        Raises:
            ValueError: If horizons not found in result.

        Example:
            >>> comparison = analyzer.compare_horizons(result, "h0", "h8")
            >>> print(f"MAPE difference: {comparison['mape_diff']:.2f}%")
        """
        if horizon_a not in result.metrics_by_horizon:
            msg = f"Horizon {horizon_a} not found in result"
            raise ValueError(msg)

        if horizon_b not in result.metrics_by_horizon:
            msg = f"Horizon {horizon_b} not found in result"
            raise ValueError(msg)

        metrics_a = result.metrics_by_horizon[horizon_a]
        metrics_b = result.metrics_by_horizon[horizon_b]

        return {
            "horizon_a": horizon_a,
            "horizon_b": horizon_b,
            "mape_diff": metrics_b.mape - metrics_a.mape,
            "mae_diff": metrics_b.mae - metrics_a.mae,
            "rmse_diff": metrics_b.rmse - metrics_a.rmse,
            "relative_mape_change": (
                (metrics_b.mape - metrics_a.mape) / metrics_a.mape * 100 if metrics_a.mape > 0 else 0.0
            ),
            "relative_rmse_change": (
                (metrics_b.rmse - metrics_a.rmse) / metrics_a.rmse * 100 if metrics_a.rmse > 0 else 0.0
            ),
        }

    def generate_horizon_report(
        self,
        result: HorizonResult,
    ) -> str:
        """Generate text summary report of horizon analysis.

        Args:
            result: HorizonResult from analysis.

        Returns:
            Formatted text report.

        Example:
            >>> report = analyzer.generate_horizon_report(result)
            >>> print(report)
        """
        lines = []
        lines.append("Horizon Analysis Report")
        lines.append("=" * 70)
        lines.append(f"Number of horizons analyzed: {result.n_horizons}")
        lines.append(f"Degradation pattern: {result.degradation_pattern}")
        lines.append(f"Degradation rate: {result.degradation_rate:.4f} per horizon")
        lines.append(f"Model type: {result.degradation_model.model_type}")
        lines.append(f"Model R²: {result.degradation_model.r_squared:.3f}")
        lines.append("")

        # Best and worst horizons
        if result.best_horizon:
            best_name, best_value = result.best_horizon
            lines.append(f"Best horizon: {best_name} (MAPE: {best_value:.2f}%)")

        if result.worst_horizon:
            worst_name, worst_value = result.worst_horizon
            lines.append(f"Worst horizon: {worst_name} (MAPE: {worst_value:.2f}%)")

        lines.append("")

        # Horizon-by-horizon metrics
        lines.append("Horizon-by-Horizon Metrics:")
        lines.append("-" * 70)
        for horizon in self.config.horizons:
            if horizon in result.metrics_by_horizon:
                metrics = result.metrics_by_horizon[horizon]
                skill_str = ""
                if horizon in result.skill_scores_by_horizon:
                    skill = result.skill_scores_by_horizon[horizon]
                    skill_str = f" | Skill(P): {skill.persistence:6.3f}"

                lines.append(
                    f"  {horizon:4s} | "
                    f"MAPE: {metrics.mape:6.2f}% | "
                    f"MAE: {metrics.mae:8.2f} | "
                    f"RMSE: {metrics.rmse:8.2f} | "
                    f"R²: {metrics.r2:6.3f}"
                    f"{skill_str}"
                )

        # Statistical tests
        if result.statistical_tests:
            lines.append("")
            lines.append("Statistical Tests:")
            lines.append("-" * 70)

            if "anova" in result.statistical_tests:
                anova = result.statistical_tests["anova"]
                lines.append(
                    f"  ANOVA: F={anova['f_statistic']:.4f}, p-value={anova['p_value']:.4f}"
                )
                if anova.get("significant", False):
                    lines.append("  → Significant difference between horizons detected!")

            if "kruskal_wallis" in result.statistical_tests:
                kw = result.statistical_tests["kruskal_wallis"]
                lines.append(
                    f"  Kruskal-Wallis: H={kw['h_statistic']:.4f}, "
                    f"p-value={kw['p_value']:.4f}"
                )
                if kw.get("significant", False):
                    lines.append("  → Significant difference between horizons detected!")

        lines.append("")
        lines.append("=" * 70)

        return "\n".join(lines)

    # Private helper methods

    def _compute_metrics_by_horizon(
        self,
        actuals: dict[str, np.ndarray],
        forecasts: dict[str, np.ndarray],
    ) -> dict[str, MetricsResult]:
        """Compute metrics for each horizon.

        Args:
            actuals: Dictionary mapping horizon names to actual arrays.
            forecasts: Dictionary mapping horizon names to forecast arrays.

        Returns:
            Dictionary mapping horizon names to MetricsResult.
        """
        metrics_by_horizon = {}

        for horizon in self.config.horizons:
            if horizon not in actuals or horizon not in forecasts:
                logger.warning("Skipping horizon %s: missing data", horizon)
                continue

            actual = actuals[horizon]
            forecast = forecasts[horizon]

            # Validate sufficient samples
            if len(actual) < self.config.min_samples:
                logger.warning(
                    "Skipping horizon %s: only %d samples (min: %d)",
                    horizon,
                    len(actual),
                    self.config.min_samples,
                )
                continue

            metrics = self.metrics_calculator.calculate_single(actual, forecast)
            metrics_by_horizon[horizon] = metrics

        return metrics_by_horizon

    def _fit_degradation_model(
        self,
        metrics_by_horizon: dict[str, MetricsResult],
    ) -> DegradationModel:
        """Fit degradation model to metrics.

        Args:
            metrics_by_horizon: Dictionary mapping horizon names to MetricsResult.

        Returns:
            DegradationModel with fitted parameters.
        """
        # Extract data for fitting
        horizon_indices = []
        errors = []

        for horizon in self.config.horizons:
            if horizon in metrics_by_horizon:
                h_idx = int(horizon[1:])
                horizon_indices.append(h_idx)
                errors.append(metrics_by_horizon[horizon].mape)

        horizon_indices = np.array(horizon_indices, dtype=np.float64)
        errors = np.array(errors, dtype=np.float64)

        if len(horizon_indices) < 2:
            # Not enough data for fitting
            return DegradationModel(
                model_type=self.config.degradation_model,
                parameters=[0.0, 0.0],
                degradation_rate=0.0,
                r_squared=0.0,
                predictions=np.array([]),
            )

        # Fit model
        degradation_rate, parameters = self.compute_degradation(
            metrics_by_horizon, self.config.degradation_model
        )

        # Compute predictions for R²
        if self.config.degradation_model == "linear":

            def func(h, a, b):
                return a + b * h

        elif self.config.degradation_model == "exponential":

            def func(h, a, b):
                return a * np.exp(b * h)

        elif self.config.degradation_model == "power_law":

            def func(h, a, b):
                return a * np.power(np.maximum(h, 0.1), b)

        else:
            func = None

        predictions = np.array([])
        r_squared = 0.0

        if func is not None:
            try:
                predictions = func(horizon_indices, *parameters)

                # Compute R²
                ss_res = np.sum((errors - predictions) ** 2)
                ss_tot = np.sum((errors - np.mean(errors)) ** 2)

                if ss_tot > 1e-10:
                    r_squared = float(1.0 - (ss_res / ss_tot))
                else:
                    r_squared = 1.0

            except Exception as e:
                logger.warning("Failed to compute predictions/R²: %s", e)

        return DegradationModel(
            model_type=self.config.degradation_model,
            parameters=parameters,
            degradation_rate=degradation_rate,
            r_squared=r_squared,
            predictions=predictions,
        )

    def _rank_horizons(
        self,
        metrics_by_horizon: dict[str, MetricsResult],
        ascending: bool = True,
    ) -> list[tuple[str, float]]:
        """Rank horizons by MAPE.

        Args:
            metrics_by_horizon: Dictionary mapping horizon names to MetricsResult.
            ascending: If True, rank from best (lowest) to worst.

        Returns:
            List of (horizon_name, mape) tuples, sorted.
        """
        if not metrics_by_horizon:
            return []

        horizons_with_mape = [
            (name, result.mape) for name, result in metrics_by_horizon.items()
        ]

        # Sort by MAPE
        horizons_with_mape.sort(key=lambda x: x[1], reverse=not ascending)

        return horizons_with_mape

    def _run_statistical_tests(
        self,
        actuals: dict[str, np.ndarray],
        forecasts: dict[str, np.ndarray],
    ) -> dict[str, Any]:
        """Run statistical tests to compare horizons.

        Args:
            actuals: Dictionary mapping horizon names to actual arrays.
            forecasts: Dictionary mapping horizon names to forecast arrays.

        Returns:
            Dictionary with test results.
        """
        tests = {}

        # Prepare groups for testing
        groups = []

        for horizon in self.config.horizons:
            if horizon in actuals and horizon in forecasts:
                actual = actuals[horizon]
                forecast = forecasts[horizon]

                if len(actual) >= self.config.min_samples:
                    errors = np.abs(actual - forecast)
                    groups.append(errors)

        if len(groups) < 2:
            return {"note": "Insufficient horizons for statistical testing"}

        # ANOVA test (parametric)
        try:
            f_stat, p_value = stats.f_oneway(*groups)
            tests["anova"] = {
                "f_statistic": float(f_stat),
                "p_value": float(p_value),
                "significant": p_value < self.config.alpha,
                "n_groups": len(groups),
                "alpha": self.config.alpha,
            }
        except Exception as e:
            logger.warning("ANOVA test failed: %s", e)
            tests["anova"] = {"error": str(e)}

        # Kruskal-Wallis test (non-parametric)
        try:
            h_stat, p_value = stats.kruskal(*groups)
            tests["kruskal_wallis"] = {
                "h_statistic": float(h_stat),
                "p_value": float(p_value),
                "significant": p_value < self.config.alpha,
                "n_groups": len(groups),
                "alpha": self.config.alpha,
            }
        except Exception as e:
            logger.warning("Kruskal-Wallis test failed: %s", e)
            tests["kruskal_wallis"] = {"error": str(e)}

        return tests

    def _compute_skill_scores(
        self,
        actuals: dict[str, np.ndarray],
        forecasts: dict[str, np.ndarray],
    ) -> dict[str, SkillScores]:
        """Compute skill scores for all horizons.

        Args:
            actuals: Dictionary mapping horizon names to actual arrays.
            forecasts: Dictionary mapping horizon names to forecast arrays.

        Returns:
            Dictionary mapping horizon names to SkillScores.
        """
        skill_scores_by_horizon = {}

        for horizon in self.config.horizons:
            if horizon not in actuals or horizon not in forecasts:
                continue

            actual = actuals[horizon]
            forecast = forecasts[horizon]

            # Compute skill scores for different baselines
            skill_persistence = self._compute_single_skill_score(
                actual, forecast, baseline="persistence"
            )
            skill_climatology = self._compute_single_skill_score(
                actual, forecast, baseline="climatology"
            )
            skill_random_walk = self._compute_single_skill_score(
                actual, forecast, baseline="random_walk"
            )

            skill_scores_by_horizon[horizon] = SkillScores(
                persistence=skill_persistence,
                climatology=skill_climatology,
                random_walk=skill_random_walk,
            )

        return skill_scores_by_horizon

    def _compute_single_skill_score(
        self,
        actual: np.ndarray,
        forecast: np.ndarray,
        baseline: str,
    ) -> float:
        """Compute skill score vs baseline.

        Args:
            actual: Actual values.
            forecast: Forecast values.
            baseline: Baseline type.

        Returns:
            Skill score.
        """
        # Generate baseline forecast
        if baseline == "persistence":
            baseline_forecast = np.roll(actual, 1)
            baseline_forecast[0] = actual[0]
        elif baseline == "climatology":
            baseline_forecast = np.full_like(actual, np.mean(actual))
        elif baseline == "random_walk":
            baseline_forecast = np.roll(actual, 1)
            baseline_forecast[0] = actual[0]
        else:
            return 0.0

        # Compute MSE
        mse_forecast = float(np.mean((forecast - actual) ** 2))
        mse_baseline = float(np.mean((baseline_forecast - actual) ** 2))

        # Compute skill score
        if mse_baseline > 0:
            skill = 1.0 - (mse_forecast / mse_baseline)
        else:
            skill = 1.0 if mse_forecast == 0 else 0.0

        return float(skill)

    def __repr__(self) -> str:
        """Return string representation.

        Returns:
            String representation.
        """
        return (
            f"HorizonAnalyzer("
            f"n_horizons={len(self.config.horizons)}, "
            f"degradation_model={self.config.degradation_model!r})"
        )
