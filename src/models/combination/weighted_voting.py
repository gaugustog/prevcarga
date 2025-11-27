"""Weighted voting combiner with learned optimal weights.

This module provides the WeightedVotingCombiner class that learns optimal
combination weights from historical forecast performance. Unlike simple averaging
with fixed weights, this combiner optimizes weights to minimize forecast error.

The combiner supports multiple optimization methods:
- inverse_error: Simple inverse error weighting (w_i ∝ 1/error_i)
- minimize_mse: Minimize MSE using scipy.optimize
- minimize_mae: Minimize MAE using scipy.optimize
- cvxpy: Convex optimization with advanced constraints (if cvxpy available)
- differential_evolution: Global optimization for non-convex problems

All methods enforce:
- Non-negativity: w_i >= 0 for all i
- Sum to one: Σw_i = 1
- Optional bounds and sparsity constraints

Example:
    ```python
    from src.models.combination import WeightedVotingCombiner, CombinerConfig
    import numpy as np
    import pandas as pd

    # Historical data for training
    train_predictions = {
        "lgbm": pd.DataFrame({"pred_h0": [100, 200, 300]}),
        "rf": pd.DataFrame({"pred_h0": [110, 190, 310]}),
        "arima": pd.DataFrame({"pred_h0": [105, 195, 305]})
    }
    train_targets = pd.DataFrame({"h0": [102, 198, 302]})

    # Configure combiner
    config = CombinerConfig(require_fit=True)
    combiner = WeightedVotingCombiner(
        optimization_method="minimize_mse",
        config=config
    )

    # Learn optimal weights
    combiner.fit(train_predictions, train_targets)
    print(f"Learned weights: {combiner.get_weights()}")

    # Apply to test data
    test_predictions = {
        "lgbm": pd.DataFrame({"pred_h0": [100, 200, 300]}),
        "rf": pd.DataFrame({"pred_h0": [110, 190, 310]}),
        "arima": pd.DataFrame({"pred_h0": [105, 195, 305]})
    }
    result = combiner.combine(test_predictions)
    ```
"""

import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.models.combination.base_combiner import BaseCombiner
from src.models.combination.data_structures import (
    CombinationResult,
    CombinerConfig,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Type alias for optimization methods
OptimizationMethod = Literal[
    "inverse_error",
    "minimize_mse",
    "minimize_mae",
    "cvxpy",
    "differential_evolution",
]


class WeightedVotingCombiner(BaseCombiner):
    """Combiner that learns optimal weights from historical performance.

    This combiner uses historical forecast performance to learn optimal
    combination weights that minimize prediction error. Multiple optimization
    methods are supported, from simple inverse error weighting to advanced
    convex optimization.

    Key Features:
        - Multiple optimization methods (inverse_error, minimize_mse, minimize_mae, etc.)
        - Enforces non-negativity and sum-to-one constraints
        - Optional weight bounds and sparsity constraints
        - Supports per-horizon and per-time-series weight learning
        - Regularization to prevent overfitting
        - Ensemble size selection by dropping low-weight models

    Attributes:
        config: CombinerConfig instance with configuration options.
        optimization_method: Method used for weight optimization.
        weight_bounds: Optional min/max bounds for weights.
        sparsity_threshold: Threshold for dropping low-weight models.
        regularization_strength: L2 regularization strength.

    Example:
        >>> # Learn weights with MSE minimization
        >>> combiner = WeightedVotingCombiner(
        ...     optimization_method="minimize_mse",
        ...     weight_bounds=(0.0, 0.8),
        ...     sparsity_threshold=0.05
        ... )
        >>> combiner.fit(train_predictions, train_targets)
        >>> result = combiner.combine(test_predictions)
        >>> print(f"Weights: {result.weights}")
    """

    def __init__(
        self,
        config: CombinerConfig | dict[str, Any] | None = None,
        optimization_method: OptimizationMethod = "minimize_mse",
        weight_bounds: tuple[float, float] | None = None,
        sparsity_threshold: float = 0.0,
        regularization_strength: float = 0.0,
    ) -> None:
        """Initialize the weighted voting combiner.

        Args:
            config: Configuration as CombinerConfig instance or dictionary.
                If None, default configuration is used. Sets require_fit=True
                by default since weights must be learned.
            optimization_method: Method for optimizing weights.
                Options: "inverse_error", "minimize_mse", "minimize_mae",
                "cvxpy", "differential_evolution". Default is "minimize_mse".
            weight_bounds: Optional (min, max) bounds for individual weights.
                If None, weights are only constrained to [0, 1] by sum constraint.
                Example: (0.1, 0.5) ensures each weight is between 10% and 50%.
            sparsity_threshold: Minimum weight threshold. Models with learned
                weights below this threshold are dropped from the ensemble.
                Default is 0.0 (no sparsity).
            regularization_strength: L2 regularization strength for weight
                optimization. Helps prevent overfitting. Default is 0.0.

        Raises:
            ValueError: If optimization_method is invalid.
            ValueError: If weight_bounds are invalid.
            ValueError: If sparsity_threshold or regularization_strength are negative.

        Example:
            >>> # Minimize MSE with bounded weights
            >>> config = CombinerConfig(require_fit=True)
            >>> combiner = WeightedVotingCombiner(
            ...     config=config,
            ...     optimization_method="minimize_mse",
            ...     weight_bounds=(0.1, 0.6),
            ...     sparsity_threshold=0.05
            ... )
        """
        # Set default config with require_fit=True
        if config is None:
            config = CombinerConfig(require_fit=True)
        elif isinstance(config, dict):
            config_dict = config.copy()
            config_dict.setdefault("require_fit", True)
            config = CombinerConfig.from_dict(config_dict)

        super().__init__(config=config)

        # Validate optimization method
        valid_methods: set[OptimizationMethod] = {
            "inverse_error",
            "minimize_mse",
            "minimize_mae",
            "cvxpy",
            "differential_evolution",
        }
        if optimization_method not in valid_methods:
            msg = (
                f"Invalid optimization_method: {optimization_method!r}. "
                f"Must be one of {valid_methods}"
            )
            raise ValueError(msg)

        self.optimization_method = optimization_method

        # Validate weight bounds
        if weight_bounds is not None:
            min_bound, max_bound = weight_bounds
            if min_bound < 0.0 or max_bound > 1.0:
                msg = f"Weight bounds must be in [0, 1], got {weight_bounds}"
                raise ValueError(msg)
            if min_bound >= max_bound:
                msg = f"Min bound must be less than max bound, got {weight_bounds}"
                raise ValueError(msg)

        self.weight_bounds = weight_bounds

        # Validate sparsity threshold
        if sparsity_threshold < 0.0 or sparsity_threshold > 1.0:
            msg = f"sparsity_threshold must be in [0, 1], got {sparsity_threshold}"
            raise ValueError(msg)

        self.sparsity_threshold = sparsity_threshold

        # Validate regularization strength
        if regularization_strength < 0.0:
            msg = f"regularization_strength must be non-negative, got {regularization_strength}"
            raise ValueError(msg)

        self.regularization_strength = regularization_strength

        # Store in config for persistence
        self.config.custom_config.update(
            {
                "optimization_method": optimization_method,
                "weight_bounds": weight_bounds,
                "sparsity_threshold": sparsity_threshold,
                "regularization_strength": regularization_strength,
            }
        )

        logger.debug(
            "Initialized WeightedVotingCombiner with method=%s, bounds=%s, "
            "sparsity=%s, regularization=%s",
            optimization_method,
            weight_bounds,
            sparsity_threshold,
            regularization_strength,
        )

    @property
    def name(self) -> str:
        """Return combiner name.

        Returns:
            Name string identifying this combiner type.
        """
        return "weighted_voting"

    @property
    def version(self) -> str:
        """Return combiner version.

        Returns:
            Version string following semantic versioning.
        """
        return "1.0.0"

    def fit(  # noqa: PLR0912, PLR0915
        self,
        train_predictions: dict[str, pd.DataFrame],
        train_targets: pd.DataFrame,
        validation_predictions: dict[str, pd.DataFrame] | None = None,
        validation_targets: pd.DataFrame | None = None,
    ) -> None:
        """Learn optimal weights from historical performance.

        Optimizes combination weights to minimize prediction error on the
        training set (or validation set if provided). The optimization method
        is specified at initialization.

        Process:
            1. Validate input predictions and targets
            2. Extract prediction columns and align with targets
            3. Select optimization method
            4. Optimize weights to minimize error metric
            5. Validate constraints (non-negative, sum to 1)
            6. Apply sparsity threshold if configured
            7. Store fitted weights

        Args:
            train_predictions: Dictionary mapping model names to predictions
                on the training set. Each DataFrame should have columns like
                'pred_h0', 'pred_h1', etc.
            train_targets: True target values for training period.
                Should have columns like 'h0', 'h1', etc.
            validation_predictions: Optional validation predictions for
                weight optimization. If provided, weights are optimized on
                validation set instead of training set.
            validation_targets: Optional validation targets.

        Raises:
            ValueError: If input format is invalid.
            RuntimeError: If optimization fails.

        Example:
            >>> combiner = WeightedVotingCombiner(optimization_method="minimize_mse")
            >>> combiner.fit(train_predictions, train_targets)
            >>> print(combiner.get_weights())
            {'lgbm': 0.45, 'rf': 0.35, 'arima': 0.20}
        """
        # Validate predictions
        self._validate_predictions(train_predictions)

        # Use validation set if provided, otherwise use training set
        if validation_predictions is not None and validation_targets is not None:
            self._validate_predictions(validation_predictions)
            fit_predictions = validation_predictions
            fit_targets = validation_targets
            logger.info(
                "Using validation set for weight optimization (%d samples)",
                len(validation_targets),
            )
        else:
            fit_predictions = train_predictions
            fit_targets = train_targets
            logger.info(
                "Using training set for weight optimization (%d samples)",
                len(train_targets),
            )

        self._model_names = list(train_predictions.keys())

        # Get prediction columns
        first_pred = next(iter(fit_predictions.values()))
        pred_cols = sorted([col for col in first_pred.columns if col.startswith("pred_h")])

        # Prepare data for optimization
        # Stack all predictions and targets into matrices
        # Shape: (n_samples * n_horizons, n_models)
        all_preds = []
        all_targets = []

        for pred_col in pred_cols:
            # Extract horizon from column name (e.g., "pred_h0" -> "h0")
            horizon_col = pred_col.replace("pred_", "")

            if horizon_col not in fit_targets.columns:
                logger.warning(
                    "Target column %s not found for prediction column %s, skipping",
                    horizon_col,
                    pred_col,
                )
                continue

            # Stack predictions from all models for this horizon
            # Shape: (n_samples, n_models)
            horizon_preds = np.column_stack(
                [fit_predictions[name][pred_col].to_numpy() for name in self._model_names]
            )

            # Get targets for this horizon
            horizon_targets = fit_targets[horizon_col].to_numpy()

            # Remove samples with NaN in predictions or targets
            valid_mask = ~np.isnan(horizon_preds).any(axis=1) & ~np.isnan(horizon_targets)

            if valid_mask.sum() > 0:
                all_preds.append(horizon_preds[valid_mask])
                all_targets.append(horizon_targets[valid_mask])

        if not all_preds:
            msg = "No valid data for weight optimization (all predictions or targets contain NaN)"
            raise ValueError(msg)

        # Concatenate all horizons
        # Shape: (n_samples * n_valid_horizons, n_models)
        pred_matrix = np.vstack(all_preds)
        target_vector = np.concatenate(all_targets)

        logger.info(
            "Optimizing weights on %d samples across %d horizons using %s",
            pred_matrix.shape[0],
            len(all_preds),
            self.optimization_method,
        )

        # Optimize weights based on selected method
        if self.optimization_method == "inverse_error":
            weights = self._optimize_inverse_error(pred_matrix, target_vector)
        elif self.optimization_method == "minimize_mse":
            weights = self._optimize_scipy(pred_matrix, target_vector, metric="mse")
        elif self.optimization_method == "minimize_mae":
            weights = self._optimize_scipy(pred_matrix, target_vector, metric="mae")
        elif self.optimization_method == "cvxpy":
            weights = self._optimize_cvxpy(pred_matrix, target_vector)
        elif self.optimization_method == "differential_evolution":
            weights = self._optimize_differential_evolution(pred_matrix, target_vector)
        else:
            msg = f"Unsupported optimization method: {self.optimization_method}"
            raise ValueError(msg)

        # Convert to dictionary
        weights_dict = dict(zip(self._model_names, weights, strict=False))

        # Apply sparsity threshold
        if self.sparsity_threshold > 0.0:
            original_count = len(weights_dict)
            weights_dict = {
                name: weight
                for name, weight in weights_dict.items()
                if weight >= self.sparsity_threshold
            }

            if len(weights_dict) != original_count:
                logger.info(
                    "Applied sparsity threshold %.3f: dropped %d models with low weights",
                    self.sparsity_threshold,
                    original_count - len(weights_dict),
                )

                # Renormalize after dropping models
                weights_dict = self._normalize_weights(weights_dict)

        # Store fitted weights
        self._global_weights = weights_dict

        # Validate final weights
        if self.config.validate_weights:
            self._validate_weights(self._global_weights)

        self._fitted = True
        self._fit_timestamp = datetime.now()  # noqa: DTZ005

        logger.info(
            "Fitted weighted voting combiner with %d models (method=%s)",
            len(self._global_weights),
            self.optimization_method,
        )
        logger.debug("Learned weights: %s", self._global_weights)

    def combine(
        self,
        predictions: dict[str, pd.DataFrame],
        metadata: dict[str, Any] | None = None,  # noqa: ARG002
        weights: dict[str, float] | None = None,
    ) -> CombinationResult:
        """Combine predictions using fitted weights.

        Applies the learned optimal weights to combine predictions from
        multiple models. Custom weights can optionally override fitted weights.

        Args:
            predictions: Dictionary mapping model names to prediction DataFrames.
                Each DataFrame should have columns like 'pred_h0', 'pred_h1', etc.
            metadata: Optional metadata about predictions (currently unused).
            weights: Optional custom weights to override fitted weights.
                Must sum to 1.0 and match prediction model names.

        Returns:
            CombinationResult with:
                - combined_predictions: DataFrame with weighted combined predictions
                - weights: Dictionary of weights used for each model
                - metadata: Information about the combination process
                - model_contributions: Individual contributions (if config.store_contributions=True)
                - confidence_intervals: Combined intervals (if config.store_intervals=True)

        Raises:
            ValueError: If predictions format is invalid or weights are invalid.
            RuntimeError: If combiner not fitted (when config.require_fit=True).

        Example:
            >>> # Use fitted weights
            >>> result = combiner.combine(test_predictions)
            >>> print(result.weights)
            {'lgbm': 0.45, 'rf': 0.35, 'arima': 0.20}

            >>> # Override with custom weights
            >>> custom_weights = {"lgbm": 0.5, "rf": 0.3, "arima": 0.2}
            >>> result = combiner.combine(test_predictions, weights=custom_weights)
        """
        start_time = time.time()

        # Check if fitting is required
        self._check_is_fitted()

        # Validate predictions format
        self._validate_predictions(predictions)

        # Handle models with excessive missing data
        cleaned_predictions, excluded_models = self._handle_missing_predictions(predictions)

        # Determine weights to use
        if weights is not None:
            # Use custom weights provided at combination time
            # Filter to only include models present in cleaned_predictions
            final_weights = {name: weights[name] for name in cleaned_predictions if name in weights}

            # Check if any models are missing from weights
            missing_models = set(cleaned_predictions) - set(final_weights)
            if missing_models:
                msg = (
                    f"Custom weights missing for models: {missing_models}. "
                    f"Weights must be provided for all models."
                )
                raise ValueError(msg)

            # Normalize weights to ensure they sum to 1
            final_weights = self._normalize_weights(final_weights)

            # Validate weights if configured
            if self.config.validate_weights:
                self._validate_weights(final_weights)

            logger.debug("Using custom weights for %d models", len(final_weights))
        else:
            # Use fitted weights
            final_weights = {
                name: self._global_weights.get(name, 0.0) for name in cleaned_predictions
            }

            # Normalize in case some models were excluded
            final_weights = self._normalize_weights(final_weights)

            logger.debug("Using fitted weights for %d models", len(final_weights))

        # Get prediction columns
        first_pred = next(iter(cleaned_predictions.values()))
        pred_cols = sorted([col for col in first_pred.columns if col.startswith("pred_h")])

        # Perform weighted averaging with NaN handling
        combined_df = pd.DataFrame(index=first_pred.index)
        model_contributions = {} if self.config.store_contributions else None

        for col in pred_cols:
            # Stack predictions from all models
            # Shape: (n_samples, n_models)
            pred_stack = np.stack(
                [cleaned_predictions[name][col].to_numpy() for name in final_weights],
                axis=1,
            )

            # Create mask for valid (non-NaN) predictions
            valid_mask = ~np.isnan(pred_stack)

            # Compute weighted average handling NaN
            # For each sample, renormalize weights to only include valid models
            weighted_sum = np.zeros(len(first_pred))
            weight_sum = np.zeros(len(first_pred))

            for i, (_name, weight) in enumerate(final_weights.items()):
                pred_values = pred_stack[:, i]
                valid = valid_mask[:, i]

                # Add weighted contribution where valid
                weighted_sum += np.where(valid, pred_values * weight, 0.0)
                weight_sum += np.where(valid, weight, 0.0)

            # Compute average (avoid division by zero)
            with np.errstate(divide="ignore", invalid="ignore"):
                combined_values = np.where(
                    weight_sum > 0,
                    weighted_sum / weight_sum,
                    np.nan,
                )

            combined_df[col] = combined_values

            # Store individual contributions if requested
            if self.config.store_contributions:
                for name, weight in final_weights.items():
                    if name not in model_contributions:
                        model_contributions[name] = pd.DataFrame(index=first_pred.index)
                    # Store weighted contribution
                    model_contributions[name][col] = (
                        cleaned_predictions[name][col].to_numpy() * weight
                    )

        # Combine prediction intervals if requested
        confidence_intervals = None
        if self.config.store_intervals:
            confidence_intervals = self._combine_intervals(cleaned_predictions, final_weights)

        # Calculate processing time
        processing_time = time.time() - start_time

        # Create metadata
        metadata_obj = self._create_metadata(
            models_used=list(cleaned_predictions.keys()),
            models_excluded=excluded_models,
            processing_time=processing_time,
            combination_method=self.name,
            optimization_method=self.optimization_method,
            weights_used=final_weights.copy(),
        )

        # Add custom metadata fields
        metadata_obj.configuration["optimization_method"] = self.optimization_method
        metadata_obj.configuration["weight_bounds"] = self.weight_bounds
        metadata_obj.configuration["sparsity_threshold"] = self.sparsity_threshold
        metadata_obj.performance_metrics["weights_entropy"] = float(
            -sum(w * np.log(w + 1e-10) for w in final_weights.values())
        )

        # Create result
        result = CombinationResult(
            combined_predictions=combined_df,
            weights=final_weights,
            metadata=metadata_obj,
            model_contributions=model_contributions,
            confidence_intervals=confidence_intervals,
        )

        logger.info(
            "Combined %d models using weighted voting (processing time: %.3fs)",
            len(final_weights),
            processing_time,
        )

        return result

    def _optimize_inverse_error(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
    ) -> np.ndarray:
        """Optimize weights using inverse error weighting.

        Simple and fast method: w_i ∝ 1/MAE_i. Models with lower error
        receive higher weights.

        Args:
            predictions: Prediction matrix of shape (n_samples, n_models).
            targets: Target vector of shape (n_samples,).

        Returns:
            Optimal weights as numpy array of shape (n_models,).

        Example:
            >>> predictions = np.array([[100, 110], [200, 190]])
            >>> targets = np.array([105, 195])
            >>> weights = self._optimize_inverse_error(predictions, targets)
            >>> print(weights)
            [0.5, 0.5]  # Both models have similar error
        """
        # Compute MAE for each model
        errors = np.mean(np.abs(predictions - targets[:, np.newaxis]), axis=0)

        # Handle zero errors (perfect predictions)
        min_error = 1e-10
        errors = np.maximum(errors, min_error)

        # Inverse weighting
        inverse_errors = 1.0 / errors

        # Normalize to sum to 1
        weights = inverse_errors / inverse_errors.sum()

        logger.debug("Inverse error weights: MAE=%s, weights=%s", errors, weights)

        return weights

    def _optimize_scipy(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        metric: Literal["mse", "mae"] = "mse",
    ) -> np.ndarray:
        """Optimize weights using scipy.optimize.

        Uses SLSQP (Sequential Least Squares Programming) to minimize
        MSE or MAE subject to constraints.

        Constraints:
            - Sum to 1: Σw_i = 1
            - Non-negative: w_i >= 0
            - Optional bounds: w_min <= w_i <= w_max

        Args:
            predictions: Prediction matrix of shape (n_samples, n_models).
            targets: Target vector of shape (n_samples,).
            metric: Objective metric, either "mse" or "mae".

        Returns:
            Optimal weights as numpy array of shape (n_models,).

        Raises:
            RuntimeError: If optimization fails.

        Example:
            >>> predictions = np.array([[100, 110], [200, 190]])
            >>> targets = np.array([105, 195])
            >>> weights = self._optimize_scipy(predictions, targets, metric="mse")
        """
        n_models = predictions.shape[1]

        # Define objective function
        def objective(w: np.ndarray) -> float:
            combined = predictions @ w
            if metric == "mse":
                error = np.mean((combined - targets) ** 2)
            elif metric == "mae":
                error = np.mean(np.abs(combined - targets))
            else:
                msg = f"Unknown metric: {metric}"
                raise ValueError(msg)

            # Add L2 regularization if configured
            if self.regularization_strength > 0.0:
                error += self.regularization_strength * np.sum(w**2)

            return error

        # Constraints: sum to 1
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        # Bounds: non-negative, optional custom bounds
        if self.weight_bounds is not None:
            bounds = [self.weight_bounds for _ in range(n_models)]
        else:
            bounds = [(0.0, 1.0) for _ in range(n_models)]

        # Initial guess: equal weights
        x0 = np.ones(n_models) / n_models

        # Optimize
        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-9},
        )

        if not result.success:
            logger.warning(
                "Scipy optimization did not converge: %s. Using result anyway.",
                result.message,
            )

        weights = result.x

        # Ensure exact normalization (numerical precision)
        weights = weights / weights.sum()

        logger.debug(
            "Scipy optimization (metric=%s): final_error=%.6f, weights=%s",
            metric,
            result.fun,
            weights,
        )

        return weights

    def _optimize_cvxpy(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
    ) -> np.ndarray:
        """Optimize weights using cvxpy convex optimization.

        Provides more advanced optimization with additional constraints
        and better numerical stability. Requires cvxpy package.

        Args:
            predictions: Prediction matrix of shape (n_samples, n_models).
            targets: Target vector of shape (n_samples,).

        Returns:
            Optimal weights as numpy array of shape (n_models,).

        Raises:
            ImportError: If cvxpy is not installed.
            RuntimeError: If optimization fails.

        Example:
            >>> predictions = np.array([[100, 110], [200, 190]])
            >>> targets = np.array([105, 195])
            >>> weights = self._optimize_cvxpy(predictions, targets)
        """
        try:
            import cvxpy as cp  # noqa: PLC0415
        except ImportError as e:
            msg = (
                "cvxpy optimization requires the cvxpy package. "
                "Install it with: pip install cvxpy"
            )
            raise ImportError(msg) from e

        n_models = predictions.shape[1]

        # Define optimization variable
        w = cp.Variable(n_models)

        # Objective: minimize MSE
        combined = predictions @ w
        mse = cp.sum_squares(combined - targets) / len(targets)

        # Add regularization if configured
        if self.regularization_strength > 0.0:
            objective = mse + self.regularization_strength * cp.sum_squares(w)
        else:
            objective = mse

        # Constraints
        constraints = [
            cp.sum(w) == 1.0,  # Sum to 1
            w >= 0.0,  # Non-negative
        ]

        # Optional bounds
        if self.weight_bounds is not None:
            min_bound, max_bound = self.weight_bounds
            constraints.extend([w >= min_bound, w <= max_bound])

        # Solve
        problem = cp.Problem(cp.Minimize(objective), constraints)

        try:
            problem.solve(solver=cp.ECOS)
        except Exception:
            # Try alternative solver
            logger.warning("ECOS solver failed, trying SCS")
            problem.solve(solver=cp.SCS)

        if problem.status not in ["optimal", "optimal_inaccurate"]:
            msg = f"cvxpy optimization failed with status: {problem.status}"
            raise RuntimeError(msg)

        weights = w.value

        # Ensure exact normalization
        weights = weights / weights.sum()

        logger.debug(
            "cvxpy optimization: final_mse=%.6f, weights=%s",
            problem.value,
            weights,
        )

        return weights

    def _optimize_differential_evolution(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
    ) -> np.ndarray:
        """Optimize weights using differential evolution global optimizer.

        Global optimization method useful for non-convex problems.
        Slower than gradient-based methods but more robust.

        Args:
            predictions: Prediction matrix of shape (n_samples, n_models).
            targets: Target vector of shape (n_samples,).

        Returns:
            Optimal weights as numpy array of shape (n_models,).

        Example:
            >>> predictions = np.array([[100, 110], [200, 190]])
            >>> targets = np.array([105, 195])
            >>> weights = self._optimize_differential_evolution(predictions, targets)
        """
        from scipy.optimize import differential_evolution  # noqa: PLC0415

        n_models = predictions.shape[1]

        # Define objective function with sum-to-one constraint via normalization
        def objective(w_raw: np.ndarray) -> float:
            # Normalize to enforce sum-to-one
            w = w_raw / w_raw.sum()

            combined = predictions @ w
            mse = np.mean((combined - targets) ** 2)

            # Add regularization if configured
            if self.regularization_strength > 0.0:
                mse += self.regularization_strength * np.sum(w**2)

            return mse

        # Bounds
        if self.weight_bounds is not None:
            bounds = [self.weight_bounds for _ in range(n_models)]
        else:
            bounds = [(0.0, 1.0) for _ in range(n_models)]

        # Optimize
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = differential_evolution(
                objective,
                bounds=bounds,
                strategy="best1bin",
                maxiter=1000,
                popsize=15,
                tol=1e-7,
                seed=42,
            )

        weights = result.x

        # Normalize to ensure exact sum to 1
        weights = weights / weights.sum()

        logger.debug(
            "Differential evolution optimization: final_error=%.6f, weights=%s",
            result.fun,
            weights,
        )

        return weights

    @classmethod
    def load(cls, filepath: str | Path) -> "WeightedVotingCombiner":
        """Load combiner from file.

        Deserializes a previously saved combiner state, including optimization
        parameters and learned weights.

        Args:
            filepath: Path to load file.

        Returns:
            Loaded WeightedVotingCombiner instance.

        Example:
            >>> combiner = WeightedVotingCombiner.load("combiners/weighted_voting.pkl")
            >>> print(combiner.optimization_method)
            'minimize_mse'
        """
        filepath = Path(filepath)
        state = joblib.load(filepath)

        # Extract custom parameters from config
        custom_config = state["config"].get("custom_config", {})
        optimization_method = custom_config.get("optimization_method", "minimize_mse")
        weight_bounds = custom_config.get("weight_bounds", None)
        sparsity_threshold = custom_config.get("sparsity_threshold", 0.0)
        regularization_strength = custom_config.get("regularization_strength", 0.0)

        # Create instance with saved config and parameters
        instance = cls(
            config=state["config"],
            optimization_method=optimization_method,
            weight_bounds=weight_bounds,
            sparsity_threshold=sparsity_threshold,
            regularization_strength=regularization_strength,
        )

        # Restore state
        instance._weights = state["weights"]
        instance._global_weights = state["global_weights"]
        instance._fitted = state["fitted"]
        instance._fit_timestamp = (
            datetime.fromisoformat(state["fit_timestamp"]) if state["fit_timestamp"] else None
        )
        instance._performance_history = state["performance_history"]
        instance._model_names = state["model_names"]

        logger.info("Loaded %s combiner from %s", instance.name, filepath)

        return instance
