"""Random Forest feature selector plugin with horizon awareness.

This module provides the RFFeatureSelectorPlugin for performing horizon-specific
feature selection in multi-horizon forecasting. It prevents temporal data leakage
by ensuring that selected features are available at forecast time for each horizon.

The plugin supports multiple selection methods:
    - RFECV: Recursive Feature Elimination with Cross-Validation
    - Permutation: Permutation importance-based selection
    - SHAP: SHapley Additive exPlanations-based selection
    - Combined: Ensemble of multiple methods for robust selection

Example:
    ```python
    from src.features.plugins.rf_selector import RFFeatureSelectorPlugin
    import pandas as pd

    # Create plugin
    plugin = RFFeatureSelectorPlugin()

    # Configure for multiple horizons
    config = {
        "target_column": "carga",
        "horizons": [0, 1, 2, 3],
        "selection_method": "rfecv",
        "max_features_per_horizon": 30,
        "n_estimators": 50,
    }

    # Perform feature selection
    # Note: This plugin doesn't return features directly,
    # it stores selection results in metadata
    result = plugin.generate_features(df, config)

    # Retrieve selected features for specific horizon
    selected_features_d1 = plugin.get_selected_features(horizon=1)
    importance_scores = plugin.get_importance_scores(horizon=1)

    # Generate report
    report = plugin.generate_selection_report()
    ```
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFECV
from sklearn.inspection import permutation_importance
from sklearn.model_selection import TimeSeriesSplit

from src.features.base.advanced_plugin import AdvancedFeaturePlugin
from src.features.evaluation.leakage_detector import DataLeakageDetector
from src.utils.logger import get_logger

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = get_logger(__name__)


class RFFeatureSelectorConfig(BaseModel):
    """Configuration model for RFFeatureSelectorPlugin.

    This model defines configuration options for Random Forest-based
    feature selection, including target column, forecast horizons,
    selection method, and Random Forest hyperparameters.

    Attributes:
        target_column: Name of target column for feature selection.
        horizons: List of forecast horizons in days (e.g., [0, 1, 2] for D+0 to D+2).
        max_features_per_horizon: Maximum number of features to select per horizon.
        selection_method: Method for feature selection.
        cv_folds: Number of cross-validation folds for RFECV.
        importance_threshold: Minimum importance threshold for feature selection.
        n_estimators: Number of trees in Random Forest.
        random_state: Random state for reproducibility.
        n_jobs: Number of parallel jobs (-1 for all cores).
        periods_per_day: Number of periods per day (48 for semi-hourly).

    Example:
        ```python
        config = RFFeatureSelectorConfig(
            target_column="carga",
            horizons=[0, 1, 2, 3, 4],
            selection_method="rfecv",
            max_features_per_horizon=30,
            n_estimators=50,
        )
        ```
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    target_column: str = Field(
        default="carga",
        description="Target column for feature selection",
    )
    horizons: list[int] = Field(
        default_factory=lambda: list(range(0, 9)),  # D+0 to D+8
        description="Forecast horizons (in days) for selection",
    )
    max_features_per_horizon: int = Field(
        default=50,
        ge=1,
        description="Maximum features to select per horizon",
    )
    selection_method: Literal["rfecv", "permutation", "shap", "combined"] = Field(
        default="rfecv",
        description="Feature selection method",
    )
    cv_folds: int = Field(
        default=5,
        ge=2,
        le=10,
        description="Cross-validation folds for RFECV",
    )
    importance_threshold: float = Field(
        default=0.01,
        ge=0.0,
        le=1.0,
        description="Minimum importance threshold for selection",
    )
    n_estimators: int = Field(
        default=50,
        ge=10,
        le=500,
        description="Number of trees in Random Forest",
    )
    random_state: int = Field(
        default=42,
        description="Random state for reproducibility",
    )
    n_jobs: int = Field(
        default=-1,
        description="Parallel jobs (-1 for all cores)",
    )
    periods_per_day: int = Field(
        default=48,
        ge=1,
        description="Periods per day (48 for semi-hourly)",
    )

    @field_validator("horizons")
    @classmethod
    def validate_horizons(cls, v: list[int]) -> list[int]:
        """Validate horizons are non-negative and sorted."""
        if not all(h >= 0 for h in v):
            msg = "All horizons must be non-negative"
            raise ValueError(msg)
        return sorted(v)


class RFFeatureSelectorPlugin(AdvancedFeaturePlugin):
    """Random Forest-based horizon-aware feature selector.

    Performs feature selection independently for each forecast horizon,
    ensuring that selected features are available at forecast time and
    do not introduce temporal data leakage.

    Selection Methods:
        1. RFECV: Recursive Feature Elimination with Cross-Validation
           - Removes least important features iteratively
           - Selects optimal subset via CV performance

        2. Permutation Importance: Post-hoc importance via shuffling
           - Measures impact of each feature on model performance
           - More reliable than tree-based importance

        3. SHAP: SHapley Additive exPlanations
           - Game-theoretic feature importance
           - Consistent and accurate attribution

        4. Combined: Ensemble of multiple methods
           - More robust feature selection
           - Consensus across methods

    Note:
        This plugin doesn't generate new features in the traditional sense.
        Instead, it performs feature selection and stores the results in
        self.selection_metadata. Use get_selected_features() to retrieve results.

    Attributes:
        selection_metadata: Dictionary storing selected features per horizon.
        leakage_detector: DataLeakageDetector instance for checking feature safety.

    Example:
        >>> plugin = RFFeatureSelectorPlugin()
        >>> config = {
        ...     "target_column": "carga",
        ...     "horizons": [0, 1, 2, 3],
        ...     "selection_method": "rfecv",
        ...     "max_features_per_horizon": 30
        ... }
        >>> result = plugin.generate_features(df, config)
        >>> selected = plugin.get_selected_features(horizon=1)
    """

    def __init__(self) -> None:
        """Initialize plugin with storage for selection results."""
        self.selection_metadata: dict[str, dict[str, Any]] = {}
        self.leakage_detector: DataLeakageDetector | None = None

    @property
    def name(self) -> str:
        """Return the plugin name."""
        return "rf_feature_selector"

    @property
    def version(self) -> str:
        """Return the plugin version."""
        return "1.0.0"

    @property
    def computational_complexity(self) -> Literal["low", "medium", "high"]:
        """Return computational complexity level."""
        return "high"

    def estimate_compute_time(self, data_size: int) -> float:
        """Estimate selection time.

        RFECV is expensive: multiple CV folds × feature iterations × RF training.

        Args:
            data_size: Number of samples in the dataset.

        Returns:
            Estimated computation time in seconds.
        """
        # Empirical: ~60 seconds per horizon for 1 year of semi-hourly data (8760 samples)
        return 60.0 * (data_size / 8760)

    def validate_config(self, config: dict[str, Any]) -> bool:
        """Validate configuration.

        Args:
            config: Plugin configuration dictionary.

        Returns:
            True if configuration is valid.

        Raises:
            ValueError: If configuration is invalid.
        """
        RFFeatureSelectorConfig(**config)
        logger.debug("RF feature selector config validated")
        return True

    def generate_features(
        self,
        df: pd.DataFrame,
        config: dict[str, Any],
    ) -> pd.DataFrame:
        """Perform horizon-aware feature selection.

        Note: This plugin doesn't generate new features in the traditional sense.
        Instead, it performs feature selection and stores the results in
        self.selection_metadata. Use get_selected_features() to retrieve results.

        Args:
            df: Input DataFrame with all features.
            config: Plugin configuration.

        Returns:
            Empty DataFrame (selection results stored in metadata).

        Raises:
            ValueError: If target column is missing or configuration is invalid.
        """
        # Validate config
        validated_config = RFFeatureSelectorConfig(**config)

        # Initialize leakage detector
        self.leakage_detector = DataLeakageDetector(
            periods_per_day=validated_config.periods_per_day
        )

        # Validate target column
        if validated_config.target_column not in df.columns:
            msg = f"Target column '{validated_config.target_column}' not found"
            raise ValueError(msg)

        logger.info(
            "Starting feature selection for %d horizons using %s method",
            len(validated_config.horizons),
            validated_config.selection_method,
        )

        # Get all feature columns (excluding target)
        feature_cols = [
            col for col in df.columns if col != validated_config.target_column
        ]

        # Perform selection for each horizon
        for horizon in validated_config.horizons:
            logger.info("Selecting features for horizon D+%d", horizon)

            try:
                selected_features = self._select_for_horizon(
                    df,
                    feature_cols,
                    validated_config.target_column,
                    horizon,
                    validated_config,
                )

                self.selection_metadata[f"horizon_{horizon}"] = selected_features

                logger.info(
                    "Horizon D+%d: selected %d features",
                    horizon,
                    len(selected_features["features"]),
                )

            except Exception as e:
                logger.error("Feature selection failed for horizon %d: %s", horizon, e)
                self.selection_metadata[f"horizon_{horizon}"] = {
                    "features": [],
                    "importance": {},
                    "error": str(e),
                }

        # Return empty DataFrame (results in metadata)
        logger.info(
            "Feature selection complete. Use get_selected_features() to retrieve results."
        )
        return pd.DataFrame(index=df.index)

    def _select_for_horizon(
        self,
        df: pd.DataFrame,
        all_features: list[str],
        target_col: str,
        horizon: int,
        config: RFFeatureSelectorConfig,
    ) -> dict[str, Any]:
        """Select features for specific horizon.

        Args:
            df: Input DataFrame.
            all_features: All available feature names.
            target_col: Target column name.
            horizon: Forecast horizon (days).
            config: Configuration object.

        Returns:
            Dictionary with selected features and metadata.
        """
        if self.leakage_detector is None:
            msg = "Leakage detector not initialized"
            raise RuntimeError(msg)

        # Get horizon-safe features
        safe_features = self.leakage_detector.get_horizon_safe_features(
            all_features,
            horizon,
            target_col,
        )

        if len(safe_features) == 0:
            logger.warning("No safe features for horizon %d", horizon)
            return {"features": [], "importance": {}}

        # Create horizon-specific target
        target = self._create_horizon_target(
            df[target_col],
            horizon,
            config.periods_per_day,
        )

        # Prepare feature matrix
        X = df[safe_features].copy()
        y = target.copy()

        # Align and drop NaN
        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]

        # Drop rows with any NaN
        valid_idx = X.notna().all(axis=1) & y.notna()
        X_clean = X.loc[valid_idx]
        y_clean = y.loc[valid_idx]

        if len(X_clean) < 100:
            logger.warning(
                "Insufficient samples for horizon %d: %d",
                horizon,
                len(X_clean),
            )
            return {
                "features": safe_features[: config.max_features_per_horizon],
                "importance": {},
            }

        logger.info(
            "Horizon %d: %d samples, %d features",
            horizon,
            len(X_clean),
            len(safe_features),
        )

        # Perform selection based on method
        if config.selection_method == "rfecv":
            selected = self._select_features_rfecv(X_clean, y_clean, config)
        elif config.selection_method == "permutation":
            selected = self._select_features_permutation(X_clean, y_clean, config)
        elif config.selection_method == "shap":
            selected = self._select_features_shap(X_clean, y_clean, config)
        elif config.selection_method == "combined":
            selected = self._select_features_combined(X_clean, y_clean, config)
        else:
            msg = f"Unknown selection method: {config.selection_method}"
            raise ValueError(msg)

        # Limit to max features
        if len(selected["features"]) > config.max_features_per_horizon:
            # Sort by importance and take top-k
            sorted_features = sorted(
                selected["features"],
                key=lambda f: selected["importance"].get(f, 0),
                reverse=True,
            )
            selected["features"] = sorted_features[: config.max_features_per_horizon]

        return selected

    def _create_horizon_target(
        self,
        target: pd.Series,
        horizon: int,
        periods_per_day: int,
    ) -> pd.Series:
        """Create horizon-shifted target variable.

        Args:
            target: Original target series.
            horizon: Forecast horizon in days.
            periods_per_day: Number of periods per day.

        Returns:
            Shifted target series.
        """
        shift_periods = horizon * periods_per_day
        return target.shift(-shift_periods)

    def _select_features_rfecv(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        config: RFFeatureSelectorConfig,
    ) -> dict[str, Any]:
        """Select features using RFECV.

        Args:
            X: Feature matrix.
            y: Target series.
            config: Configuration object.

        Returns:
            Dictionary with selected features and importance scores.
        """
        logger.info("Running RFECV feature selection")

        # Create Random Forest
        rf = RandomForestRegressor(
            n_estimators=config.n_estimators,
            random_state=config.random_state,
            n_jobs=config.n_jobs,
        )

        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=config.cv_folds)

        # RFECV
        selector = RFECV(
            estimator=rf,
            cv=tscv,
            scoring="neg_mean_absolute_error",
            n_jobs=config.n_jobs,
            min_features_to_select=min(5, len(X.columns)),
        )

        selector.fit(X, y)

        # Get selected features
        selected_features = X.columns[selector.support_].tolist()

        # Get feature importances
        rf.fit(X[selected_features], y)
        importance_dict = dict(zip(selected_features, rf.feature_importances_))

        return {
            "features": selected_features,
            "importance": importance_dict,
            "cv_scores": selector.cv_results_,
            "optimal_n_features": selector.n_features_,
        }

    def _select_features_permutation(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        config: RFFeatureSelectorConfig,
    ) -> dict[str, Any]:
        """Select features using permutation importance.

        Args:
            X: Feature matrix.
            y: Target series.
            config: Configuration object.

        Returns:
            Dictionary with selected features and importance scores.
        """
        logger.info("Running permutation importance feature selection")

        # Train Random Forest
        rf = RandomForestRegressor(
            n_estimators=config.n_estimators,
            random_state=config.random_state,
            n_jobs=config.n_jobs,
        )

        rf.fit(X, y)

        # Calculate permutation importance
        perm_importance = permutation_importance(
            rf,
            X,
            y,
            n_repeats=10,
            random_state=config.random_state,
            n_jobs=config.n_jobs,
        )

        # Create importance dictionary
        importance_dict = dict(zip(X.columns, perm_importance.importances_mean))

        # Select features above threshold
        selected_features = [
            feature
            for feature, importance in importance_dict.items()
            if importance >= config.importance_threshold
        ]

        return {
            "features": selected_features,
            "importance": importance_dict,
            "importance_std": dict(zip(X.columns, perm_importance.importances_std)),
        }

    def _select_features_shap(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        config: RFFeatureSelectorConfig,
    ) -> dict[str, Any]:
        """Select features using SHAP values.

        Args:
            X: Feature matrix.
            y: Target series.
            config: Configuration object.

        Returns:
            Dictionary with selected features and importance scores.
        """
        try:
            import shap
        except ImportError:
            logger.warning("SHAP not available, falling back to permutation importance")
            return self._select_features_permutation(X, y, config)

        logger.info("Running SHAP-based feature selection")

        # Train Random Forest
        rf = RandomForestRegressor(
            n_estimators=config.n_estimators,
            random_state=config.random_state,
            n_jobs=config.n_jobs,
        )

        rf.fit(X, y)

        # Calculate SHAP values (use subset for speed)
        sample_size = min(500, len(X))
        X_sample = X.sample(n=sample_size, random_state=config.random_state)

        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(X_sample)

        # Mean absolute SHAP value per feature
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        importance_dict = dict(zip(X.columns, mean_abs_shap))

        # Select features above threshold
        threshold = np.percentile(
            mean_abs_shap, 100 * (1 - config.max_features_per_horizon / len(X.columns))
        )
        selected_features = [
            feature
            for feature, shap_val in importance_dict.items()
            if shap_val >= threshold
        ]

        return {
            "features": selected_features,
            "importance": importance_dict,
            "method": "shap",
        }

    def _select_features_combined(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        config: RFFeatureSelectorConfig,
    ) -> dict[str, Any]:
        """Combine multiple selection methods.

        Args:
            X: Feature matrix.
            y: Target series.
            config: Configuration object.

        Returns:
            Dictionary with selected features and importance scores.
        """
        logger.info("Running combined feature selection")

        # Run RFECV
        rfecv_results = self._select_features_rfecv(X, y, config)

        # Run permutation
        perm_results = self._select_features_permutation(X, y, config)

        # Consensus: features selected by both methods
        consensus_features = list(
            set(rfecv_results["features"]) & set(perm_results["features"])
        )

        # If too few, add high-ranking features from either method
        if len(consensus_features) < config.max_features_per_horizon // 2:
            all_features = set(rfecv_results["features"]) | set(
                perm_results["features"]
            )
            consensus_features = list(all_features)[: config.max_features_per_horizon]

        # Combine importance scores (average)
        combined_importance = {}
        for feature in consensus_features:
            rfecv_imp = rfecv_results["importance"].get(feature, 0)
            perm_imp = perm_results["importance"].get(feature, 0)
            combined_importance[feature] = (rfecv_imp + perm_imp) / 2

        return {
            "features": consensus_features,
            "importance": combined_importance,
            "rfecv_selected": rfecv_results["features"],
            "perm_selected": perm_results["features"],
        }

    def get_selected_features(self, horizon: int) -> list[str]:
        """Get selected features for specific horizon.

        Args:
            horizon: Forecast horizon (days).

        Returns:
            List of selected feature names.
        """
        key = f"horizon_{horizon}"
        if key not in self.selection_metadata:
            logger.warning("No selection results for horizon %d", horizon)
            return []

        return self.selection_metadata[key].get("features", [])

    def get_importance_scores(self, horizon: int) -> dict[str, float]:
        """Get feature importance scores for horizon.

        Args:
            horizon: Forecast horizon (days).

        Returns:
            Dictionary mapping feature names to importance scores.
        """
        key = f"horizon_{horizon}"
        if key not in self.selection_metadata:
            return {}

        return self.selection_metadata[key].get("importance", {})

    def generate_selection_report(
        self, output_path: str | None = None
    ) -> pd.DataFrame:
        """Generate feature selection report.

        Args:
            output_path: Optional path to save report as CSV.

        Returns:
            DataFrame with selection summary.
        """
        report_data = []

        for horizon_key, metadata in self.selection_metadata.items():
            horizon = int(horizon_key.split("_")[1])

            importance_vals = list(metadata.get("importance", {}).values())

            report_data.append(
                {
                    "horizon": horizon,
                    "n_features": len(metadata.get("features", [])),
                    "top_feature": (
                        max(
                            metadata.get("importance", {}),
                            key=metadata.get("importance", {}).get,
                            default="N/A",
                        )
                        if metadata.get("importance")
                        else "N/A"
                    ),
                    "mean_importance": (
                        np.mean(importance_vals) if importance_vals else 0
                    ),
                }
            )

        report_df = pd.DataFrame(report_data).sort_values("horizon")

        if output_path:
            report_df.to_csv(output_path, index=False)
            logger.info("Selection report saved to %s", output_path)

        return report_df

    def get_feature_names(self, config: dict[str, Any]) -> list[str]:
        """This plugin doesn't generate features directly.

        Args:
            config: Plugin configuration (unused).

        Returns:
            Empty list as this plugin performs selection, not generation.
        """
        return []
