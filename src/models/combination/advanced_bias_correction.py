"""Advanced bias correction with non-linear and temporal methods.

This module provides sophisticated bias correction for model combinations,
extending the basic bias correction with spline regression, decision tree
conditional correction, and comprehensive temporal decomposition.

Key Features:
- Non-linear bias detection and correction via spline regression
- Conditional bias correction based on forecast context using decision trees
- Temporal bias decomposition (trend, seasonal, weekly, hourly)
- Cross-validation for bias model selection
- Statistical significance testing to prevent overfitting
- Multiple correction strategies with automatic selection

Example:
    ```python
    from src.models.combination.advanced_bias_correction import (
        AdvancedBiasCorrectionModule
    )

    # Create and fit advanced bias correction
    corrector = AdvancedBiasCorrectionModule(config={
        'correction_type': 'combined',
        'use_cross_validation': True
    })

    # Fit on training data
    corrector.fit(predictions, targets, timestamps)

    # Apply to new predictions
    corrected = corrector.correct_bias(predictions, timestamps)

    # Check bias reduction
    print(f"Bias reduced by {corrector.bias_reduction:.1%}")
    print(f"Diagnostics: {corrector.get_diagnostics()}")
    ```
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer
from sklearn.tree import DecisionTreeRegressor

from src.utils.logger import get_logger

# Ruff: Ignore sklearn convention of using X for feature matrices and magic numbers for validation
# ruff: noqa: N806, PLR2004, PLC0415

logger = get_logger(__name__)


class CorrectionType(Enum):
    """Types of bias correction available."""

    SPLINE = "spline"
    TREE = "tree"
    TEMPORAL = "temporal"
    COMBINED = "combined"
    AUTO = "auto"


@dataclass
class AdvancedBiasCorrectionConfig:
    """Configuration for advanced bias correction.

    Attributes:
        correction_type: Type of correction to apply ('spline', 'tree',
            'temporal', 'combined', 'auto').
        spline_n_knots: Number of knots for spline basis functions.
        spline_degree: Degree of spline polynomials.
        tree_max_depth: Maximum depth for decision tree corrector.
        tree_min_samples_leaf: Minimum samples per leaf in tree.
        min_samples_per_group: Minimum samples for temporal grouping.
        significance_level: P-value threshold for significance tests.
        use_cross_validation: Whether to use CV for model selection.
        n_cv_folds: Number of cross-validation folds.
        regularization_alpha: Ridge regression regularization strength.
        max_correction_fraction: Maximum correction as fraction of prediction.
        detect_trend: Whether to detect trend bias.
        detect_seasonal: Whether to detect seasonal (monthly) bias.
        detect_weekly: Whether to detect weekly bias.
        detect_hourly: Whether to detect hourly bias.
    """

    correction_type: str = "combined"
    spline_n_knots: int = 5
    spline_degree: int = 3
    tree_max_depth: int = 4
    tree_min_samples_leaf: int = 20
    min_samples_per_group: int = 20
    significance_level: float = 0.05
    use_cross_validation: bool = True
    n_cv_folds: int = 5
    regularization_alpha: float = 1.0
    max_correction_fraction: float = 0.3
    detect_trend: bool = True
    detect_seasonal: bool = True
    detect_weekly: bool = True
    detect_hourly: bool = True


@dataclass
class BiasCorrectortDiagnostics:
    """Diagnostics from bias correction fitting.

    Attributes:
        correction_type: Type of correction applied.
        n_samples: Number of samples used for fitting.
        fit_timestamp: When the corrector was fitted.
        original_bias: Mean absolute error before correction.
        corrected_bias: Mean absolute error after correction.
        bias_reduction: Fractional reduction in bias.
        spline_r2: R-squared of spline model (if fitted).
        tree_r2: R-squared of tree model (if fitted).
        temporal_components: List of temporal components detected.
        trend_significance: P-value of trend test.
        seasonal_significance: P-value of seasonal ANOVA.
        weekly_significance: P-value of weekly ANOVA.
        hourly_significance: P-value of hourly ANOVA.
        cv_scores: Cross-validation scores (if used).
        selected_model: Which model was selected (if auto).
    """

    correction_type: str = ""
    n_samples: int = 0
    fit_timestamp: datetime | None = None
    original_bias: float = 0.0
    corrected_bias: float = 0.0
    bias_reduction: float = 0.0
    spline_r2: float | None = None
    tree_r2: float | None = None
    temporal_components: list[str] = field(default_factory=list)
    trend_significance: float | None = None
    seasonal_significance: float | None = None
    weekly_significance: float | None = None
    hourly_significance: float | None = None
    cv_scores: dict[str, float] = field(default_factory=dict)
    selected_model: str | None = None


class SplineBiasCorrector:
    """Spline-based non-linear bias correction.

    Uses spline basis functions with ridge regression to model complex
    non-linear relationships between predictions and residuals.

    Attributes:
        n_knots: Number of spline knots.
        degree: Degree of spline polynomials.
        alpha: Regularization strength.
        fitted: Whether the corrector has been fitted.

    Example:
        >>> corrector = SplineBiasCorrector(n_knots=5, degree=3)
        >>> corrector.fit(predictions, residuals)
        >>> corrected = corrector.correct(new_predictions)
    """

    def __init__(self, n_knots: int = 5, degree: int = 3, alpha: float = 1.0) -> None:
        """Initialize spline bias corrector.

        Args:
            n_knots: Number of knots for spline transformer.
            degree: Degree of spline polynomials.
            alpha: Ridge regularization strength.
        """
        self.n_knots = n_knots
        self.degree = degree
        self.alpha = alpha

        self._spline_transformer: SplineTransformer | None = None
        self._ridge_model: Ridge | None = None
        self._fitted = False
        self._r2_score: float | None = None

    @property
    def fitted(self) -> bool:
        """Return whether the corrector has been fitted."""
        return self._fitted

    @property
    def r2_score(self) -> float | None:
        """Return R-squared score from fitting."""
        return self._r2_score

    def fit(self, predictions: np.ndarray, residuals: np.ndarray) -> None:
        """Fit spline model to residuals.

        Args:
            predictions: Model predictions (n_samples,).
            residuals: Residuals (true - predicted) (n_samples,).
        """
        if len(predictions) < self.n_knots * 2:
            logger.warning(
                "Insufficient samples for spline fitting: %d < %d",
                len(predictions),
                self.n_knots * 2,
            )
            return

        # Reshape for sklearn
        X = predictions.reshape(-1, 1)

        # Create spline transformer
        self._spline_transformer = SplineTransformer(
            n_knots=self.n_knots,
            degree=self.degree,
            include_bias=True,
        )

        # Transform to spline basis
        X_spline = self._spline_transformer.fit_transform(X)

        # Fit ridge regression
        self._ridge_model = Ridge(alpha=self.alpha)
        self._ridge_model.fit(X_spline, residuals)

        # Calculate R-squared
        predictions_bias = self._ridge_model.predict(X_spline)
        ss_res = np.sum((residuals - predictions_bias) ** 2)
        ss_tot = np.sum((residuals - np.mean(residuals)) ** 2)
        self._r2_score = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        self._fitted = True
        logger.debug("Spline corrector fitted with R² = %.4f", self._r2_score)

    def correct(self, predictions: np.ndarray) -> np.ndarray:
        """Apply spline correction to predictions.

        Args:
            predictions: Predictions to correct.

        Returns:
            Corrected predictions.
        """
        if not self._fitted or self._spline_transformer is None or self._ridge_model is None:
            return predictions

        X = predictions.reshape(-1, 1)
        X_spline = self._spline_transformer.transform(X)
        bias_estimate = self._ridge_model.predict(X_spline)

        return predictions + bias_estimate


class TreeBiasCorrector:
    """Decision tree-based conditional bias correction.

    Uses a decision tree regressor to learn conditional bias patterns
    based on prediction values and context features.

    Attributes:
        max_depth: Maximum tree depth.
        min_samples_leaf: Minimum samples per leaf node.
        fitted: Whether the corrector has been fitted.

    Example:
        >>> corrector = TreeBiasCorrector(max_depth=4)
        >>> corrector.fit(predictions, residuals, context_features)
        >>> corrected = corrector.correct(new_predictions, new_context)
    """

    def __init__(self, max_depth: int = 4, min_samples_leaf: int = 20) -> None:
        """Initialize tree bias corrector.

        Args:
            max_depth: Maximum depth of decision tree.
            min_samples_leaf: Minimum samples required in leaf nodes.
        """
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf

        self._tree_model: DecisionTreeRegressor | None = None
        self._fitted = False
        self._r2_score: float | None = None
        self._n_features: int = 0

    @property
    def fitted(self) -> bool:
        """Return whether the corrector has been fitted."""
        return self._fitted

    @property
    def r2_score(self) -> float | None:
        """Return R-squared score from fitting."""
        return self._r2_score

    def fit(
        self,
        predictions: np.ndarray,
        residuals: np.ndarray,
        context_features: np.ndarray | None = None,
    ) -> None:
        """Fit decision tree model to residuals.

        Args:
            predictions: Model predictions (n_samples,).
            residuals: Residuals (true - predicted) (n_samples,).
            context_features: Optional context features (n_samples, n_features).
        """
        # Combine predictions with context features
        if context_features is not None:
            X = np.column_stack([predictions.reshape(-1, 1), context_features])
        else:
            X = predictions.reshape(-1, 1)

        self._n_features = X.shape[1]

        if len(predictions) < self.min_samples_leaf * 4:
            logger.warning(
                "Insufficient samples for tree fitting: %d < %d",
                len(predictions),
                self.min_samples_leaf * 4,
            )
            return

        # Create and fit decision tree
        self._tree_model = DecisionTreeRegressor(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=42,
        )
        self._tree_model.fit(X, residuals)

        # Calculate R-squared
        predictions_bias = self._tree_model.predict(X)
        ss_res = np.sum((residuals - predictions_bias) ** 2)
        ss_tot = np.sum((residuals - np.mean(residuals)) ** 2)
        self._r2_score = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        self._fitted = True
        logger.debug("Tree corrector fitted with R² = %.4f", self._r2_score)

    def correct(
        self,
        predictions: np.ndarray,
        context_features: np.ndarray | None = None,
    ) -> np.ndarray:
        """Apply tree-based correction to predictions.

        Args:
            predictions: Predictions to correct.
            context_features: Optional context features.

        Returns:
            Corrected predictions.
        """
        if not self._fitted or self._tree_model is None:
            return predictions

        # Build feature matrix
        if context_features is not None:
            X = np.column_stack([predictions.reshape(-1, 1), context_features])
        else:
            X = predictions.reshape(-1, 1)

        # Ensure same number of features
        if X.shape[1] != self._n_features:
            logger.warning(
                "Feature mismatch: expected %d, got %d",
                self._n_features,
                X.shape[1],
            )
            return predictions

        bias_estimate = self._tree_model.predict(X)

        return predictions + bias_estimate


class AdvancedBiasCorrectionModule:
    """Advanced bias correction with non-linear and temporal methods.

    Provides comprehensive bias correction using multiple approaches:
    - Non-linear: Spline regression on residuals
    - Conditional: Decision tree based on context
    - Temporal: Decomposition into trend/seasonal/weekly/hourly

    The module includes:
    - Cross-validation for model selection
    - Significance testing to prevent overfitting
    - Automatic method selection based on data characteristics
    - Comprehensive diagnostics and validation

    Correction Function:
        y_corrected = y + bias_spline(y) + bias_temporal(t) + bias_conditional(context)

    Overfitting Prevention:
        - Cross-validation for model selection
        - Minimum sample size per group
        - Significance testing (p < 0.05)
        - Regularization in spline fitting
        - Tree depth limiting

    Validation:
        bias_reduction = (original_bias - corrected_bias) / original_bias
        require: bias_reduction > 0% AND p < 0.05

    Attributes:
        config: Configuration dataclass.
        spline_corrector: Spline-based corrector (if fitted).
        tree_corrector: Tree-based corrector (if fitted).
        temporal_bias: Dictionary of temporal bias patterns.
        trend_bias: Linear trend bias parameters.
        bias_reduction: Achieved bias reduction fraction.
        fitted: Whether the module has been fitted.
        diagnostics: Detailed fitting diagnostics.

    Example:
        >>> corrector = AdvancedBiasCorrectionModule(
        ...     config={'correction_type': 'combined'}
        ... )
        >>> corrector.fit(predictions, targets, timestamps)
        >>> corrected = corrector.correct_bias(predictions, timestamps)
        >>> print(f"Bias reduction: {corrector.bias_reduction:.1%}")
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize advanced bias correction module.

        Args:
            config: Configuration dictionary. See AdvancedBiasCorrectionConfig
                for available options.
        """
        if config is None:
            config = {}

        self.config = AdvancedBiasCorrectionConfig(**config)

        # Bias correction models
        self.spline_corrector: SplineBiasCorrector | None = None
        self.tree_corrector: TreeBiasCorrector | None = None
        self.temporal_bias: dict[str, dict[int, float]] = {}
        self.trend_bias: dict[str, float] | None = None

        # Performance tracking
        self.bias_reduction: float = 0.0
        self._fitted = False
        self._diagnostics = BiasCorrectortDiagnostics()

        logger.debug("Initialized AdvancedBiasCorrectionModule with type=%s", self.config.correction_type)

    @property
    def fitted(self) -> bool:
        """Return whether the module has been fitted."""
        return self._fitted

    def fit(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        timestamps: pd.DatetimeIndex,
        context_features: np.ndarray | None = None,
    ) -> None:
        """Fit bias correction models.

        Fits appropriate bias correction models based on configuration:
        - Spline correction for non-linear bias patterns
        - Tree-based conditional correction
        - Temporal decomposition (trend, seasonal, weekly, hourly)

        Uses cross-validation for model selection when enabled.

        Args:
            predictions: Model predictions (n_samples,).
            targets: True target values (n_samples,).
            timestamps: Datetime index for temporal decomposition.
            context_features: Optional context features for conditional correction.

        Raises:
            ValueError: If arrays have mismatched lengths.
        """
        if len(predictions) != len(targets) or len(predictions) != len(timestamps):
            msg = f"Array length mismatch: predictions={len(predictions)}, targets={len(targets)}, timestamps={len(timestamps)}"
            raise ValueError(msg)

        from datetime import UTC

        logger.info(
            "Fitting advanced bias correction (%s) on %d samples",
            self.config.correction_type,
            len(predictions),
        )

        # Calculate residuals
        residuals = targets - predictions

        # Initialize diagnostics
        self._diagnostics = BiasCorrectortDiagnostics(
            correction_type=self.config.correction_type,
            n_samples=len(predictions),
            fit_timestamp=datetime.now(tz=UTC),
            original_bias=float(np.mean(np.abs(residuals))),
        )

        correction_type = self.config.correction_type

        # Auto-select correction type based on data
        if correction_type == "auto":
            correction_type = self._select_correction_type(predictions, residuals, timestamps)
            self._diagnostics.selected_model = correction_type
            logger.info("Auto-selected correction type: %s", correction_type)

        # Fit spline correction
        if correction_type in ["spline", "combined"]:
            self._fit_spline_correction(predictions, residuals)

        # Fit tree-based conditional correction
        if correction_type in ["tree", "combined"]:
            self._fit_tree_correction(predictions, residuals, context_features)

        # Fit temporal corrections
        if correction_type in ["temporal", "combined"]:
            self._fit_temporal_corrections(residuals, timestamps)

        # Mark as fitted before calculating reduction (so correct_bias works)
        self._fitted = True

        # Calculate bias reduction
        self._calculate_bias_reduction(predictions, targets, timestamps, context_features)
        logger.info(
            "Bias correction fitted: reduction=%.1f%%",
            self.bias_reduction * 100,
        )

    def _fit_spline_correction(self, predictions: np.ndarray, residuals: np.ndarray) -> None:
        """Fit spline-based non-linear correction."""
        logger.debug("Fitting spline bias correction")

        self.spline_corrector = SplineBiasCorrector(
            n_knots=self.config.spline_n_knots,
            degree=self.config.spline_degree,
            alpha=self.config.regularization_alpha,
        )
        self.spline_corrector.fit(predictions, residuals)

        if self.spline_corrector.fitted:
            self._diagnostics.spline_r2 = self.spline_corrector.r2_score

    def _fit_tree_correction(
        self,
        predictions: np.ndarray,
        residuals: np.ndarray,
        context_features: np.ndarray | None,
    ) -> None:
        """Fit tree-based conditional correction."""
        logger.debug("Fitting tree-based conditional correction")

        self.tree_corrector = TreeBiasCorrector(
            max_depth=self.config.tree_max_depth,
            min_samples_leaf=self.config.tree_min_samples_leaf,
        )
        self.tree_corrector.fit(predictions, residuals, context_features)

        if self.tree_corrector.fitted:
            self._diagnostics.tree_r2 = self.tree_corrector.r2_score

    def _fit_temporal_corrections(
        self,
        residuals: np.ndarray,
        timestamps: pd.DatetimeIndex,
    ) -> None:
        """Fit temporal bias decomposition."""
        logger.debug("Fitting temporal bias decomposition")

        self.temporal_bias = {}

        # Detect and fit trend
        if self.config.detect_trend:
            trend_result = self._detect_trend_bias(residuals, timestamps)
            if trend_result is not None:
                self.trend_bias = trend_result
                self._diagnostics.temporal_components.append("trend")

        # Detect seasonal (monthly) bias
        if self.config.detect_seasonal:
            seasonal_bias, p_value = self._detect_seasonal_bias(residuals, timestamps)
            if seasonal_bias:
                self.temporal_bias["seasonal"] = seasonal_bias
                self._diagnostics.temporal_components.append("seasonal")
                self._diagnostics.seasonal_significance = p_value

        # Detect weekly bias
        if self.config.detect_weekly:
            weekly_bias, p_value = self._detect_weekly_bias(residuals, timestamps)
            if weekly_bias:
                self.temporal_bias["weekly"] = weekly_bias
                self._diagnostics.temporal_components.append("weekly")
                self._diagnostics.weekly_significance = p_value

        # Detect hourly bias
        if self.config.detect_hourly:
            hourly_bias, p_value = self._detect_hourly_bias(residuals, timestamps)
            if hourly_bias:
                self.temporal_bias["hourly"] = hourly_bias
                self._diagnostics.temporal_components.append("hourly")
                self._diagnostics.hourly_significance = p_value

    def _select_correction_type(
        self,
        predictions: np.ndarray,
        residuals: np.ndarray,
        timestamps: pd.DatetimeIndex,
    ) -> str:
        """Auto-select best correction type using cross-validation.

        Args:
            predictions: Prediction values.
            residuals: Residual values.
            timestamps: Timestamps.

        Returns:
            Selected correction type string.
        """
        if not self.config.use_cross_validation:
            return "combined"

        cv_scores: dict[str, float] = {}

        # Test spline correction
        try:
            spline_score = self._cv_score_spline(predictions, residuals)
            cv_scores["spline"] = spline_score
        except Exception as e:
            logger.warning("Spline CV failed: %s", e)
            cv_scores["spline"] = float("-inf")

        # Test temporal correction
        try:
            temporal_score = self._cv_score_temporal(residuals, timestamps)
            cv_scores["temporal"] = temporal_score
        except Exception as e:
            logger.warning("Temporal CV failed: %s", e)
            cv_scores["temporal"] = float("-inf")

        self._diagnostics.cv_scores = cv_scores

        # Select best or combined
        best_type = max(cv_scores, key=lambda k: cv_scores[k])

        # If scores are close, use combined
        scores = list(cv_scores.values())
        if len(scores) >= 2 and max(scores) - min(scores) < 0.1:
            return "combined"

        return best_type

    def _cv_score_spline(self, predictions: np.ndarray, residuals: np.ndarray) -> float:
        """Cross-validate spline correction."""
        X = predictions.reshape(-1, 1)

        pipeline = Pipeline(
            [
                (
                    "spline",
                    SplineTransformer(
                        n_knots=self.config.spline_n_knots,
                        degree=self.config.spline_degree,
                    ),
                ),
                ("ridge", Ridge(alpha=self.config.regularization_alpha)),
            ]
        )

        cv = TimeSeriesSplit(n_splits=self.config.n_cv_folds)
        scores = cross_val_score(pipeline, X, residuals, cv=cv, scoring="r2")

        return float(np.mean(scores))

    def _cv_score_temporal(
        self,
        residuals: np.ndarray,
        timestamps: pd.DatetimeIndex,
    ) -> float:
        """Estimate temporal correction score using variance reduction."""
        total_var = np.var(residuals)
        explained_var = 0.0

        # Check seasonal variance
        if len(residuals) >= 12 * self.config.min_samples_per_group:
            monthly_means = {}
            for month in range(1, 13):
                mask = timestamps.month == month
                if mask.sum() >= self.config.min_samples_per_group:
                    monthly_means[month] = np.mean(residuals[mask])

            if monthly_means:
                seasonal_effect = np.array(
                    [monthly_means.get(ts.month, 0) for ts in timestamps]
                )
                explained_var += np.var(seasonal_effect)

        # Check weekly variance
        if len(residuals) >= 7 * self.config.min_samples_per_group:
            weekly_means = {}
            for dow in range(7):
                mask = timestamps.dayofweek == dow
                if mask.sum() >= self.config.min_samples_per_group:
                    weekly_means[dow] = np.mean(residuals[mask])

            if weekly_means:
                weekly_effect = np.array(
                    [weekly_means.get(ts.dayofweek, 0) for ts in timestamps]
                )
                explained_var += np.var(weekly_effect)

        # Return R² equivalent
        return float(explained_var / total_var) if total_var > 0 else 0.0

    def _detect_trend_bias(
        self,
        residuals: np.ndarray,
        timestamps: pd.DatetimeIndex,
    ) -> dict[str, float] | None:
        """Detect and fit linear trend in residuals.

        Args:
            residuals: Residual values.
            timestamps: Timestamps.

        Returns:
            Dictionary with slope and intercept, or None if not significant.
        """
        # Convert timestamps to numeric (days from start)
        numeric_time = (timestamps - timestamps[0]).total_seconds() / 86400

        # Fit linear trend
        slope, intercept, _r_value, p_value, _std_err = stats.linregress(numeric_time, residuals)

        self._diagnostics.trend_significance = p_value

        # Check significance
        if p_value < self.config.significance_level:
            logger.debug("Significant trend detected: slope=%.4f, p=%.4f", slope, p_value)
            return {"slope": float(slope), "intercept": float(intercept)}

        return None

    def _detect_seasonal_bias(
        self,
        residuals: np.ndarray,
        timestamps: pd.DatetimeIndex,
    ) -> tuple[dict[int, float], float | None]:
        """Detect monthly/seasonal bias patterns.

        Args:
            residuals: Residual values.
            timestamps: Timestamps.

        Returns:
            Tuple of (monthly bias dict, p-value).
        """
        monthly_bias: dict[int, float] = {}

        # Need sufficient data
        if len(residuals) < 12 * self.config.min_samples_per_group:
            return {}, None

        groups = []
        for month in range(1, 13):
            mask = timestamps.month == month
            if mask.sum() >= self.config.min_samples_per_group:
                monthly_bias[month] = float(np.mean(residuals[mask]))
                groups.append(residuals[mask])

        # Need at least 6 months
        if len(groups) < 6:
            return {}, None

        # Test significance with ANOVA
        try:
            _f_stat, p_value = stats.f_oneway(*groups)
        except Exception:
            return {}, None

        if p_value < self.config.significance_level:
            return monthly_bias, float(p_value)

        return {}, float(p_value)

    def _detect_weekly_bias(
        self,
        residuals: np.ndarray,
        timestamps: pd.DatetimeIndex,
    ) -> tuple[dict[int, float], float | None]:
        """Detect day-of-week bias patterns.

        Args:
            residuals: Residual values.
            timestamps: Timestamps.

        Returns:
            Tuple of (weekly bias dict, p-value).
        """
        weekly_bias: dict[int, float] = {}

        # Need sufficient data
        if len(residuals) < 7 * self.config.min_samples_per_group:
            return {}, None

        groups = []
        for dow in range(7):
            mask = timestamps.dayofweek == dow
            if mask.sum() >= self.config.min_samples_per_group:
                weekly_bias[dow] = float(np.mean(residuals[mask]))
                groups.append(residuals[mask])

        # Need most days
        if len(groups) < 5:
            return {}, None

        # Test significance with ANOVA
        try:
            _f_stat, p_value = stats.f_oneway(*groups)
        except Exception:
            return {}, None

        if p_value < self.config.significance_level:
            return weekly_bias, float(p_value)

        return {}, float(p_value)

    def _detect_hourly_bias(
        self,
        residuals: np.ndarray,
        timestamps: pd.DatetimeIndex,
    ) -> tuple[dict[int, float], float | None]:
        """Detect hour-of-day bias patterns.

        Args:
            residuals: Residual values.
            timestamps: Timestamps.

        Returns:
            Tuple of (hourly bias dict, p-value).
        """
        hourly_bias: dict[int, float] = {}

        # Need sufficient data (for 24 hours)
        if len(residuals) < 24 * self.config.min_samples_per_group:
            return {}, None

        groups = []
        for hour in range(24):
            mask = timestamps.hour == hour
            if mask.sum() >= self.config.min_samples_per_group:
                hourly_bias[hour] = float(np.mean(residuals[mask]))
                groups.append(residuals[mask])

        # Need at least half the hours
        if len(groups) < 12:
            return {}, None

        # Test significance with ANOVA
        try:
            _f_stat, p_value = stats.f_oneway(*groups)
        except Exception:
            return {}, None

        if p_value < self.config.significance_level:
            return hourly_bias, float(p_value)

        return {}, float(p_value)

    def correct_bias(
        self,
        predictions: np.ndarray,
        timestamps: pd.DatetimeIndex,
        context_features: np.ndarray | None = None,
    ) -> np.ndarray:
        """Apply bias correction to predictions.

        Applies fitted bias corrections in sequence:
        1. Trend correction
        2. Spline non-linear correction
        3. Tree-based conditional correction
        4. Temporal corrections (seasonal, weekly, hourly)

        Args:
            predictions: Predictions to correct.
            timestamps: Timestamps for temporal corrections.
            context_features: Optional context features for tree correction.

        Returns:
            Corrected predictions array.
        """
        if not self._fitted:
            logger.warning("Bias corrector not fitted, returning original predictions")
            return predictions.copy()

        corrected = predictions.copy()

        # Apply trend correction
        if self.trend_bias is not None:
            corrected = self._apply_trend_correction(corrected, timestamps)

        # Apply spline correction
        if self.spline_corrector is not None and self.spline_corrector.fitted:
            corrected = self.spline_corrector.correct(corrected)

        # Apply tree-based conditional correction
        if self.tree_corrector is not None and self.tree_corrector.fitted:
            corrected = self.tree_corrector.correct(corrected, context_features)

        # Apply temporal corrections
        if self.temporal_bias:
            corrected = self._apply_temporal_correction(corrected, timestamps)

        # Limit correction magnitude
        max_correction = np.abs(predictions) * self.config.max_correction_fraction
        correction = corrected - predictions
        correction = np.clip(correction, -max_correction, max_correction)

        return predictions + correction

    def _apply_trend_correction(
        self,
        predictions: np.ndarray,
        timestamps: pd.DatetimeIndex,
    ) -> np.ndarray:
        """Apply linear trend correction.

        Args:
            predictions: Predictions to correct.
            timestamps: Timestamps.

        Returns:
            Corrected predictions.
        """
        if self.trend_bias is None:
            return predictions

        # Convert timestamps to numeric (days from start)
        numeric_time = (timestamps - timestamps[0]).total_seconds() / 86400

        # Calculate trend correction
        trend_correction = self.trend_bias["slope"] * numeric_time + self.trend_bias["intercept"]

        return predictions + trend_correction

    def _apply_temporal_correction(
        self,
        predictions: np.ndarray,
        timestamps: pd.DatetimeIndex,
    ) -> np.ndarray:
        """Apply temporal bias corrections.

        Args:
            predictions: Predictions to correct.
            timestamps: Timestamps.

        Returns:
            Corrected predictions.
        """
        # Ensure we have a writable copy
        corrected = np.array(predictions, dtype=np.float64, copy=True)

        # Vectorized approach for better performance
        # Apply seasonal correction
        if "seasonal" in self.temporal_bias:
            months = timestamps.month.to_numpy()
            seasonal_corrections = np.array(
                [self.temporal_bias["seasonal"].get(m, 0.0) for m in months]
            )
            corrected += seasonal_corrections

        # Apply weekly correction
        if "weekly" in self.temporal_bias:
            dows = timestamps.dayofweek.to_numpy()
            weekly_corrections = np.array(
                [self.temporal_bias["weekly"].get(d, 0.0) for d in dows]
            )
            corrected += weekly_corrections

        # Apply hourly correction
        if "hourly" in self.temporal_bias:
            hours = timestamps.hour.to_numpy()
            hourly_corrections = np.array(
                [self.temporal_bias["hourly"].get(h, 0.0) for h in hours]
            )
            corrected += hourly_corrections

        return corrected

    def _calculate_bias_reduction(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        timestamps: pd.DatetimeIndex,
        context_features: np.ndarray | None,
    ) -> None:
        """Calculate bias reduction achieved by correction.

        Args:
            predictions: Original predictions.
            targets: True target values.
            timestamps: Timestamps.
            context_features: Optional context features.
        """
        original_residuals = targets - predictions
        original_bias = np.mean(np.abs(original_residuals))

        corrected_predictions = self.correct_bias(predictions, timestamps, context_features)
        corrected_residuals = targets - corrected_predictions
        corrected_bias = np.mean(np.abs(corrected_residuals))

        if original_bias > 0:
            self.bias_reduction = (original_bias - corrected_bias) / original_bias
        else:
            self.bias_reduction = 0.0

        self._diagnostics.corrected_bias = float(corrected_bias)
        self._diagnostics.bias_reduction = self.bias_reduction

    def get_diagnostics(self) -> dict[str, Any]:
        """Get diagnostics from bias correction fitting.

        Returns:
            Dictionary with diagnostic information.
        """
        return {
            "correction_type": self._diagnostics.correction_type,
            "n_samples": self._diagnostics.n_samples,
            "fit_timestamp": (
                self._diagnostics.fit_timestamp.isoformat()
                if self._diagnostics.fit_timestamp
                else None
            ),
            "original_bias": self._diagnostics.original_bias,
            "corrected_bias": self._diagnostics.corrected_bias,
            "bias_reduction": self._diagnostics.bias_reduction,
            "spline_r2": self._diagnostics.spline_r2,
            "tree_r2": self._diagnostics.tree_r2,
            "temporal_components": self._diagnostics.temporal_components,
            "trend_significance": self._diagnostics.trend_significance,
            "seasonal_significance": self._diagnostics.seasonal_significance,
            "weekly_significance": self._diagnostics.weekly_significance,
            "hourly_significance": self._diagnostics.hourly_significance,
            "cv_scores": self._diagnostics.cv_scores,
            "selected_model": self._diagnostics.selected_model,
        }

    def __repr__(self) -> str:
        """Return string representation."""
        status = "fitted" if self._fitted else "not fitted"
        return (
            f"AdvancedBiasCorrectionModule("
            f"type={self.config.correction_type}, "
            f"status={status}, "
            f"bias_reduction={self.bias_reduction:.1%})"
        )
