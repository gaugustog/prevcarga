"""Abstract base class for hierarchical forecasting models.

This module provides the BaseHierarchicalModel abstract class that defines the
interface for two-stage hierarchical forecasting models. These models first
forecast demand mean (daily average load), then forecast profile ratios for
each semi-hourly period, and finally combine them to produce semi-hourly load
predictions.

Example:
    ```python
    from src.models.hierarchical.base_hierarchical import BaseHierarchicalModel
    from src.models.base.model import BaseModel
    import pandas as pd

    class MyHierarchicalModel(BaseHierarchicalModel):
        @property
        def name(self) -> str:
            return "my_hierarchical"

        @property
        def version(self) -> str:
            return "1.0.0"

        @property
        def supported_horizons(self) -> list[int]:
            return [0, 1, 2, 3]

        @property
        def demand_mean_model_class(self) -> type[BaseModel]:
            return MyDemandMeanModel

        @property
        def profile_model_class(self) -> type[BaseModel]:
            return MyProfileModel

        def _prepare_demand_mean_data(
            self, df: pd.DataFrame
        ) -> tuple[pd.DataFrame, pd.Series]:
            # Aggregate to daily and compute demand mean
            ...

        def _prepare_profile_data(
            self, df: pd.DataFrame, demand_mean: pd.Series, period: int
        ) -> tuple[pd.DataFrame, pd.Series]:
            # Filter by period and compute profile ratios
            ...

    # Use the model
    model = MyHierarchicalModel()
    model.fit(X_train, y_train, config)
    predictions = model.predict(X_test)
    ```
"""

from abc import abstractmethod
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.models.base.model import BaseModel
from src.models.hierarchical.profile_combiner import ProfileCombiner
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BaseHierarchicalModel(BaseModel):
    """Abstract base class for two-stage hierarchical forecasting models.

    This class provides the orchestration framework for hierarchical models that
    decompose load forecasting into two stages:
    1. Demand mean forecasting: Predict daily average load
    2. Profile forecasting: Predict semi-hourly profile ratios for each period

    The final semi-hourly load is computed as: load = demand_mean * profile_ratio

    Subclasses must implement:
        - demand_mean_model_class: Property returning the class for demand mean model
        - profile_model_class: Property returning the class for profile models
        - _prepare_demand_mean_data: Method to prepare daily aggregated data
        - _prepare_profile_data: Method to prepare profile ratio data for each period

    Attributes:
        demand_mean_model: Fitted model for demand mean forecasting.
        profile_models: Dictionary mapping period (0-47) to fitted profile models.
        combiner: ProfileCombiner instance for combining predictions.
    """

    def __init__(self) -> None:
        """Initialize hierarchical model with empty state."""
        super().__init__()
        self.demand_mean_model: BaseModel | None = None
        self.profile_models: dict[int, BaseModel] = {}
        self.combiner = ProfileCombiner()
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    @abstractmethod
    def demand_mean_model_class(self) -> type[BaseModel]:
        """Return the class for demand mean forecasting.

        Returns:
            BaseModel subclass for forecasting daily demand mean.
        """
        ...

    @property
    @abstractmethod
    def profile_model_class(self) -> type[BaseModel]:
        """Return the class for profile forecasting.

        Returns:
            BaseModel subclass for forecasting semi-hourly profile ratios.
        """
        ...

    @abstractmethod
    def _prepare_demand_mean_data(
        self,
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Prepare data for demand mean model training.

        This method should aggregate semi-hourly data to daily resolution and
        compute the demand mean (daily average load).

        Args:
            df: Input DataFrame with semi-hourly data. Must contain a timestamp
               column and a load column (target).

        Returns:
            Tuple of (X_daily, y_daily) where:
                - X_daily: Features aggregated to daily resolution
                - y_daily: Daily demand mean (average load)

        Raises:
            ValueError: If required columns are missing or data is invalid.
        """
        ...

    @abstractmethod
    def _prepare_profile_data(
        self,
        df: pd.DataFrame,
        demand_mean: pd.Series,
        period: int,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Prepare data for profile model training for a specific period.

        This method should filter data for the given semi-hourly period (0-47)
        and compute profile ratios as: profile_ratio = actual_load / demand_mean

        Args:
            df: Input DataFrame with semi-hourly data. Must contain a timestamp
               column, load column, and a period indicator.
            demand_mean: Series of daily demand mean values, indexed by date.
            period: Semi-hourly period (0-47) to prepare data for.

        Returns:
            Tuple of (X_period, y_period) where:
                - X_period: Features for the specified period
                - y_period: Profile ratios (load / demand_mean)

        Raises:
            ValueError: If period is invalid or data is missing for this period.
        """
        ...

    def fit(
        self,
        X: pd.DataFrame,  # noqa: N803
        y: pd.Series | pd.DataFrame,
        config: dict[str, Any],
    ) -> None:
        """Train the hierarchical model using two-stage approach.

        This method orchestrates the two-stage training process:
        1. Prepare and train demand mean model on daily aggregated data
        2. Prepare and train profile models for each semi-hourly period (0-47)

        The demand mean model is trained first because its predictions are
        needed to compute profile ratios for the second stage.

        Args:
            X: Training features DataFrame with semi-hourly resolution.
               Must contain timestamp column for aggregation.
            y: Training target (load) as Series or single-column DataFrame.
            config: Training configuration dictionary. Should contain:
                - "demand_mean_config": Config for demand mean model
                - "profile_config": Config for profile models
                - Other model-specific parameters

        Raises:
            ValueError: If input data is invalid or training fails.
            KeyError: If required config keys are missing.
        """
        logger.info("Starting two-stage hierarchical model training")

        # Convert y to Series if DataFrame
        if isinstance(y, pd.DataFrame):
            if y.shape[1] != 1:
                msg = f"y must be single column, got {y.shape[1]} columns"
                raise ValueError(msg)
            y = y.iloc[:, 0]

        # Combine X and y for data preparation
        df = X.copy()
        df["load"] = y

        # Stage 1: Train demand mean model
        logger.info("Stage 1: Training demand mean model")
        X_daily, y_daily = self._prepare_demand_mean_data(df)  # noqa: N806

        demand_mean_config = config.get("demand_mean_config", {})
        self.demand_mean_model = self.demand_mean_model_class()
        self.demand_mean_model.fit(X_daily, y_daily, demand_mean_config)

        logger.info(
            "Demand mean model trained on %d daily samples",
            len(y_daily),
        )

        # Get demand mean predictions for profile ratio calculation
        demand_mean_pred = self.demand_mean_model.predict(X_daily)
        # Extract first horizon if multiple horizons
        if isinstance(demand_mean_pred, pd.DataFrame):
            demand_mean_series = demand_mean_pred.iloc[:, 0]
        else:
            demand_mean_series = demand_mean_pred

        # Stage 2: Train profile models for each period
        logger.info("Stage 2: Training profile models for 48 periods")
        profile_config = config.get("profile_config", {})

        n_periods_trained = 0
        for period in range(48):
            try:
                X_period, y_period = self._prepare_profile_data(  # noqa: N806
                    df,
                    demand_mean_series,
                    period,
                )

                if len(y_period) == 0:
                    logger.warning(
                        "Period %d: No data available, skipping",
                        period,
                    )
                    continue

                # Train profile model
                profile_model = self.profile_model_class()
                profile_model.fit(X_period, y_period, profile_config)
                self.profile_models[period] = profile_model

                n_periods_trained += 1

                if period % 12 == 0:  # Log progress every 12 periods (6 hours)
                    logger.debug("Trained profile models for periods 0-%d", period)

            except Exception as e:
                logger.error(
                    "Failed to train profile model for period %d: %s",
                    period,
                    e,
                    exc_info=True,
                )
                # Continue training other periods
                continue

        logger.info(
            "Profile models trained for %d/%d periods",
            n_periods_trained,
            48,
        )

        # Store metadata
        self._feature_names = list(X.columns)
        self._training_metadata = {
            "trained_at": pd.Timestamp.now("UTC"),
            "training_config": config,
            "n_samples": len(y),
            "n_daily_samples": len(y_daily),
            "n_profile_periods": n_periods_trained,
        }
        self._is_fitted = True

        logger.info("Hierarchical model training completed successfully")

    def predict(
        self,
        X: pd.DataFrame,  # noqa: N803
        horizons: list[int] | None = None,
    ) -> pd.DataFrame:
        """Generate semi-hourly load predictions using two-stage approach.

        This method orchestrates the two-stage prediction process:
        1. Predict demand mean using demand mean model
        2. Predict profile ratios for each period using profile models
        3. Combine demand mean and profiles to get semi-hourly load

        Args:
            X: Features DataFrame with semi-hourly resolution.
            horizons: List of forecast horizons. If None, uses all supported_horizons.
                     Note: Currently only the first horizon is used for both stages.

        Returns:
            DataFrame with semi-hourly load predictions. Shape (n_samples, 48).
            Columns are named "period_0", "period_1", ..., "period_47".

        Raises:
            ValueError: If model is not fitted or input is invalid.
        """
        self._check_is_fitted()

        if horizons is None:
            horizons = self.supported_horizons

        logger.info(
            "Generating semi-hourly predictions for %d samples",
            len(X),
        )

        # Combine X for data preparation (no target needed for prediction)
        df = X.copy()

        # Stage 1: Predict demand mean
        logger.debug("Stage 1: Predicting demand mean")
        X_daily, _ = self._prepare_demand_mean_data(df)  # noqa: N806

        demand_mean_pred = self.demand_mean_model.predict(X_daily, horizons=[horizons[0]])

        # Extract first horizon
        if isinstance(demand_mean_pred, pd.DataFrame):
            demand_mean = demand_mean_pred.iloc[:, 0]
        else:
            demand_mean = demand_mean_pred

        logger.debug("Predicted demand mean for %d daily samples", len(demand_mean))

        # Stage 2: Predict profiles for each period
        logger.debug("Stage 2: Predicting profiles for %d periods", len(self.profile_models))
        profiles = {}

        for period in range(48):
            if period not in self.profile_models:
                # Profile model not available, will use default ratio of 1.0
                logger.debug("Period %d: No profile model, will use default ratio", period)
                continue

            try:
                # Prepare data for this period
                X_period, _ = self._prepare_profile_data(  # noqa: N806
                    df,
                    demand_mean,  # Use predicted demand mean
                    period,
                )

                # Predict profile ratio
                profile_pred = self.profile_models[period].predict(
                    X_period,
                    horizons=[horizons[0]],
                )

                # Extract first horizon
                if isinstance(profile_pred, pd.DataFrame):
                    profiles[period] = profile_pred.iloc[:, 0]
                else:
                    profiles[period] = profile_pred

            except Exception as e:
                logger.warning(
                    "Failed to predict profile for period %d: %s. Using default.",
                    period,
                    e,
                )
                # Profile will be missing, combiner will use default ratio
                continue

        # Stage 3: Combine demand mean and profiles
        logger.debug("Stage 3: Combining demand mean and profiles")
        combined_load = self.combiner.combine(demand_mean, profiles)

        # Validate energy conservation
        validation_result = self.combiner.validate_energy_conservation(
            demand_mean,
            combined_load,
            tolerance=0.1,  # 10% tolerance
        )

        if not validation_result["is_valid"]:
            logger.warning(
                "Energy conservation validation failed: mean error %.2f%%, "
                "max error %.2f%%, %d/%d samples exceed tolerance",
                validation_result["mean_error"] * 100,
                validation_result["max_error"] * 100,
                validation_result["n_violations"],
                len(demand_mean),
            )

        logger.info(
            "Generated semi-hourly predictions with %d profiles (mean error: %.2f%%)",
            len(profiles),
            validation_result["mean_error"] * 100,
        )

        return combined_load

    def get_feature_importance(self) -> dict[str, float]:
        """Return combined feature importance from demand mean and profile models.

        This method aggregates feature importance from the demand mean model
        and all profile models. The importance scores are averaged across models.

        Returns:
            Dictionary mapping feature names to aggregated importance scores.
            Scores are normalized to sum to 1.0.

        Raises:
            ValueError: If model is not fitted.
        """
        self._check_is_fitted()

        # Collect importance from all models
        all_importances: dict[str, list[float]] = {}

        # Add demand mean model importance
        try:
            dm_importance = self.demand_mean_model.get_feature_importance()
            for feature, score in dm_importance.items():
                all_importances.setdefault(feature, []).append(score)
        except (NotImplementedError, ValueError) as e:
            logger.warning("Could not get demand mean model importance: %s", e)

        # Add profile models importance
        for period, profile_model in self.profile_models.items():
            try:
                prof_importance = profile_model.get_feature_importance()
                for feature, score in prof_importance.items():
                    all_importances.setdefault(feature, []).append(score)
            except (NotImplementedError, ValueError) as e:
                logger.debug("Could not get profile model %d importance: %s", period, e)
                continue

        if not all_importances:
            logger.warning("No feature importance available from any model")
            return {}

        # Average importance across all models
        aggregated_importance = {
            feature: sum(scores) / len(scores) for feature, scores in all_importances.items()
        }

        # Normalize to sum to 1.0
        total = sum(aggregated_importance.values())
        if total > 0:
            aggregated_importance = {
                feature: score / total for feature, score in aggregated_importance.items()
            }

        logger.debug(
            "Aggregated feature importance from %d models",
            1 + len(self.profile_models),
        )

        return aggregated_importance

    def get_decomposition(
        self,
        X: pd.DataFrame,  # noqa: N803
    ) -> dict[str, Any]:
        """Get decomposition of predictions into demand mean and profiles.

        This helper method returns the intermediate components of the
        hierarchical prediction, useful for debugging and analysis.

        Args:
            X: Features DataFrame with semi-hourly resolution.

        Returns:
            Dictionary containing:
                - "demand_mean": Series of demand mean predictions
                - "profiles": Dictionary of profile predictions by period
                - "combined_load": DataFrame of final semi-hourly predictions

        Raises:
            ValueError: If model is not fitted.
        """
        self._check_is_fitted()

        logger.debug("Computing decomposition for %d samples", len(X))

        # Prepare data
        df = X.copy()
        X_daily, _ = self._prepare_demand_mean_data(df)  # noqa: N806

        # Predict demand mean
        demand_mean_pred = self.demand_mean_model.predict(X_daily)
        if isinstance(demand_mean_pred, pd.DataFrame):
            demand_mean = demand_mean_pred.iloc[:, 0]
        else:
            demand_mean = demand_mean_pred

        # Predict profiles
        profiles = {}
        for period in range(48):
            if period not in self.profile_models:
                continue

            try:
                X_period, _ = self._prepare_profile_data(df, demand_mean, period)  # noqa: N806
                profile_pred = self.profile_models[period].predict(X_period)

                if isinstance(profile_pred, pd.DataFrame):
                    profiles[period] = profile_pred.iloc[:, 0]
                else:
                    profiles[period] = profile_pred

            except Exception as e:
                logger.warning(
                    "Failed to predict profile for period %d: %s",
                    period,
                    e,
                )
                continue

        # Combine
        combined_load = self.combiner.combine(demand_mean, profiles)

        decomposition = {
            "demand_mean": demand_mean,
            "profiles": profiles,
            "combined_load": combined_load,
        }

        logger.info(
            "Decomposition computed: demand_mean (%d samples), "
            "profiles (%d periods), combined_load (%dx48)",
            len(demand_mean),
            len(profiles),
            len(combined_load),
        )

        return decomposition

    def save(self, path: str | Path) -> None:
        """Save hierarchical model to disk.

        This method saves the entire hierarchical model including the demand
        mean model, all profile models, combiner, and metadata.

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
            # Save entire model instance
            joblib.dump(self, path)
            logger.info(
                "Saved hierarchical model %s with %d profile models to %s",
                self.name,
                len(self.profile_models),
                path,
            )
        except Exception as e:
            msg = f"Failed to save hierarchical model to {path}: {e}"
            raise OSError(msg) from e

    @classmethod
    def load(cls, path: str | Path) -> "BaseHierarchicalModel":
        """Load hierarchical model from disk.

        Args:
            path: File path to load the model from.

        Returns:
            Loaded hierarchical model instance.

        Raises:
            FileNotFoundError: If model file doesn't exist.
            ValueError: If loaded object is not a BaseHierarchicalModel instance.
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

        if not isinstance(model, BaseHierarchicalModel):
            msg = f"Loaded object is not a BaseHierarchicalModel instance: {type(model)}"
            raise ValueError(msg)

        logger.info(
            "Loaded hierarchical model %s with %d profile models from %s",
            model.name,
            len(model.profile_models),
            path,
        )
        return model

    def __repr__(self) -> str:
        """Return string representation of the hierarchical model.

        Returns:
            String representation with model info and number of profile models.
        """
        return (
            f"{self.__class__.__name__}("
            f"name='{self.name}', "
            f"version='{self.version}', "
            f"fitted={self._is_fitted}, "
            f"n_profile_models={len(self.profile_models)})"
        )
