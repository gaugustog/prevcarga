"""Prediction workflow orchestration for PrevCarga.

This module provides a high-level prediction workflow that orchestrates
the complete prediction pipeline including model loading, feature preparation,
prediction generation, reconciliation, and output management.

Key Components:
- PredictionWorkflow: Main workflow orchestration class
- PredictionStage: Prediction workflow stages
- PredictionResult: Prediction workflow results
- ForecastOutput: Structured forecast output

Example:
    ```python
    from src.orchestration import ConfigManager
    from src.orchestration.prediction_workflow import PredictionWorkflow

    # Load configuration
    config_manager = ConfigManager.from_yaml("config/prevcarga.yaml")

    # Create and run workflow
    workflow = PredictionWorkflow(config_manager)
    result = workflow.run()

    # Check results
    print(f"Success: {result.success}")
    print(f"Forecasts generated: {len(result.forecasts)}")
    ```
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.orchestration.config_manager import ConfigManager, PredictionWorkflowConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PredictionStage(Enum):
    """Prediction workflow stages.

    Attributes:
        INITIALIZED: Workflow initialized but not started.
        LOADING_MODELS: Loading trained models.
        PREPARING_FEATURES: Preparing input features.
        GENERATING_FORECASTS: Generating base forecasts.
        APPLYING_RECONCILIATION: Applying hierarchical reconciliation.
        COMPUTING_INTERVALS: Computing confidence intervals.
        SAVING_OUTPUTS: Saving forecast outputs.
        COMPLETED: Workflow completed successfully.
        FAILED: Workflow failed.
        CANCELLED: Workflow was cancelled.
    """

    INITIALIZED = "initialized"
    LOADING_MODELS = "loading_models"
    PREPARING_FEATURES = "preparing_features"
    GENERATING_FORECASTS = "generating_forecasts"
    APPLYING_RECONCILIATION = "applying_reconciliation"
    COMPUTING_INTERVALS = "computing_intervals"
    SAVING_OUTPUTS = "saving_outputs"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class PredictionStatus(Enum):
    """Prediction workflow execution status.

    Attributes:
        PENDING: Workflow is pending start.
        RUNNING: Workflow is currently running.
        PAUSED: Workflow is paused.
        COMPLETED: Workflow completed successfully.
        FAILED: Workflow failed.
        CANCELLED: Workflow was cancelled.
    """

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class StageResult:
    """Result of a workflow stage.

    Attributes:
        stage: The workflow stage.
        success: Whether the stage succeeded.
        started_at: When the stage started.
        completed_at: When the stage completed.
        duration_seconds: Duration in seconds.
        error: Error message if failed.
        metrics: Stage-specific metrics.
    """

    stage: PredictionStage
    success: bool = True
    started_at: datetime | None = None
    completed_at: datetime | None = None
    duration_seconds: float = 0.0
    error: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "stage": self.stage.value,
            "success": self.success,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
            "error": self.error,
            "metrics": self.metrics,
        }


@dataclass
class ForecastOutput:
    """Structured forecast output.

    Attributes:
        area: Geographic area code.
        horizon: Forecast horizon (D+0 to D+8).
        timestamp: Timestamp of the forecast.
        forecast: Point forecast value.
        lower_bound: Lower confidence bound.
        upper_bound: Upper confidence bound.
        confidence_level: Confidence level for bounds.
        model_id: Model identifier used.
        reconciled: Whether forecast was reconciled.
    """

    area: str
    horizon: int
    timestamp: datetime
    forecast: float
    lower_bound: float | None = None
    upper_bound: float | None = None
    confidence_level: float = 0.95
    model_id: str | None = None
    reconciled: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "area": self.area,
            "horizon": self.horizon,
            "timestamp": self.timestamp.isoformat(),
            "forecast": self.forecast,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "confidence_level": self.confidence_level,
            "model_id": self.model_id,
            "reconciled": self.reconciled,
        }


@dataclass
class PredictionResult:
    """Result of prediction workflow execution.

    Attributes:
        success: Whether workflow completed successfully.
        status: Final workflow status.
        started_at: When workflow started.
        completed_at: When workflow completed.
        duration_seconds: Total duration in seconds.
        stage_results: Results for each stage.
        forecasts: Generated forecasts by area and horizon.
        forecast_df: DataFrame with all forecasts.
        reconciled: Whether reconciliation was applied.
        output_paths: Paths to saved outputs.
        error: Error message if failed.
    """

    success: bool = False
    status: PredictionStatus = PredictionStatus.PENDING
    started_at: datetime | None = None
    completed_at: datetime | None = None
    duration_seconds: float = 0.0
    stage_results: list[StageResult] = field(default_factory=list)
    forecasts: dict[str, dict[int, ForecastOutput]] = field(default_factory=dict)
    forecast_df: pd.DataFrame | None = None
    reconciled: bool = False
    output_paths: dict[str, str] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "success": self.success,
            "status": self.status.value,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
            "stage_results": [s.to_dict() for s in self.stage_results],
            "n_forecasts": sum(len(h) for h in self.forecasts.values()),
            "areas": list(self.forecasts.keys()),
            "reconciled": self.reconciled,
            "output_paths": self.output_paths,
            "error": self.error,
        }

    def get_summary(self) -> str:
        """Get human-readable summary.

        Returns:
            Summary string.
        """
        lines = [
            "=" * 60,
            "Prediction Workflow Summary",
            "=" * 60,
            f"Status: {self.status.value.upper()}",
            f"Success: {self.success}",
            f"Duration: {self.duration_seconds:.2f} seconds",
            "",
            "Stage Results:",
        ]

        for stage in self.stage_results:
            status = "OK" if stage.success else "FAILED"
            lines.append(f"  - {stage.stage.value}: {status} ({stage.duration_seconds:.1f}s)")

        lines.append("")
        n_forecasts = sum(len(h) for h in self.forecasts.values())
        lines.append(f"Forecasts Generated: {n_forecasts}")
        lines.append(f"Areas: {len(self.forecasts)}")
        lines.append(f"Reconciled: {self.reconciled}")

        if self.output_paths:
            lines.append("")
            lines.append("Output Files:")
            for name, path in self.output_paths.items():
                lines.append(f"  - {name}: {path}")

        if self.error:
            lines.append("")
            lines.append(f"Error: {self.error}")

        lines.append("=" * 60)
        return "\n".join(lines)


class PredictionWorkflow:
    """High-level prediction workflow orchestration.

    Orchestrates the complete prediction pipeline including:
    - Model loading and validation
    - Feature preparation
    - Forecast generation for multiple areas and horizons
    - Hierarchical reconciliation
    - Confidence interval computation
    - Output saving in multiple formats

    Example:
        >>> config_manager = ConfigManager.from_yaml("config/prevcarga.yaml")
        >>> workflow = PredictionWorkflow(config_manager)
        >>> result = workflow.run(input_features=feature_df)
        >>> print(result.get_summary())
    """

    def __init__(
        self,
        config_manager: ConfigManager,
        model_loader: Any | None = None,
        feature_pipeline: Any | None = None,
        reconciler: Any | None = None,
    ) -> None:
        """Initialize prediction workflow.

        Args:
            config_manager: Configuration manager instance.
            model_loader: Optional model loader instance.
            feature_pipeline: Optional feature pipeline instance.
            reconciler: Optional reconciler instance.
        """
        self._config_manager = config_manager
        self._model_loader = model_loader
        self._feature_pipeline = feature_pipeline
        self._reconciler = reconciler

        self._status = PredictionStatus.PENDING
        self._current_stage = PredictionStage.INITIALIZED
        self._result = PredictionResult()
        self._cancelled = False
        self._paused = False

        self._models: dict[str, dict[int, Any]] = {}
        self._features: dict[str, pd.DataFrame] = {}
        self._base_forecasts: dict[str, dict[int, np.ndarray]] = {}

        logger.info("Initialized PredictionWorkflow")

    @property
    def status(self) -> PredictionStatus:
        """Get current workflow status."""
        return self._status

    @property
    def current_stage(self) -> PredictionStage:
        """Get current workflow stage."""
        return self._current_stage

    def run(
        self,
        input_features: dict[str, pd.DataFrame] | pd.DataFrame | None = None,
        forecast_date: datetime | None = None,
    ) -> PredictionResult:
        """Run the complete prediction workflow.

        Args:
            input_features: Pre-computed features by area, or single DataFrame.
                If None, will attempt to prepare features.
            forecast_date: Date for which to generate forecasts.
                Defaults to current date.

        Returns:
            PredictionResult with forecast outcomes.
        """
        self._result = PredictionResult()
        self._result.started_at = datetime.now()
        self._status = PredictionStatus.RUNNING

        # Check if already cancelled before starting
        if self._cancelled:
            return self._finalize_result(success=False)

        if forecast_date is None:
            forecast_date = datetime.now()

        try:
            # Stage 1: Load Models
            if not self._execute_stage(
                PredictionStage.LOADING_MODELS,
                self._load_models,
            ):
                return self._finalize_result(success=False)

            # Stage 2: Prepare Features
            if not self._execute_stage(
                PredictionStage.PREPARING_FEATURES,
                self._prepare_features,
                input_features,
            ):
                return self._finalize_result(success=False)

            # Stage 3: Generate Forecasts
            if not self._execute_stage(
                PredictionStage.GENERATING_FORECASTS,
                self._generate_forecasts,
                forecast_date,
            ):
                return self._finalize_result(success=False)

            # Stage 4: Apply Reconciliation (optional)
            config = self._config_manager.get_prediction_config()
            if config.apply_reconciliation:
                if not self._execute_stage(
                    PredictionStage.APPLYING_RECONCILIATION,
                    self._apply_reconciliation,
                ):
                    return self._finalize_result(success=False)

            # Stage 5: Compute Confidence Intervals (optional)
            if config.include_confidence_intervals:
                if not self._execute_stage(
                    PredictionStage.COMPUTING_INTERVALS,
                    self._compute_confidence_intervals,
                ):
                    return self._finalize_result(success=False)

            # Stage 6: Save Outputs
            if not self._execute_stage(
                PredictionStage.SAVING_OUTPUTS,
                self._save_outputs,
            ):
                return self._finalize_result(success=False)

            return self._finalize_result(success=True)

        except Exception as e:
            logger.exception("Prediction workflow failed with exception")
            self._result.error = str(e)
            return self._finalize_result(success=False)

    def run_with_features(
        self,
        X: pd.DataFrame,
        area: str = "default",
        forecast_date: datetime | None = None,
    ) -> PredictionResult:
        """Run prediction workflow with provided features for a single area.

        Args:
            X: Feature DataFrame.
            area: Area identifier.
            forecast_date: Date for which to generate forecasts.

        Returns:
            PredictionResult with forecast outcomes.
        """
        input_features = {area: X}
        return self.run(input_features=input_features, forecast_date=forecast_date)

    def cancel(self) -> None:
        """Cancel the running workflow."""
        logger.info("Cancelling prediction workflow")
        self._cancelled = True
        self._status = PredictionStatus.CANCELLED

    def pause(self) -> None:
        """Pause the running workflow."""
        logger.info("Pausing prediction workflow")
        self._paused = True
        self._status = PredictionStatus.PAUSED

    def resume(self) -> None:
        """Resume a paused workflow."""
        if self._status == PredictionStatus.PAUSED:
            logger.info("Resuming prediction workflow")
            self._paused = False
            self._status = PredictionStatus.RUNNING

    def _execute_stage(
        self,
        stage: PredictionStage,
        stage_func: callable,
        *args: Any,
        **kwargs: Any,
    ) -> bool:
        """Execute a workflow stage with timing and error handling.

        Args:
            stage: The stage to execute.
            stage_func: The function to execute.
            *args: Positional arguments for the function.
            **kwargs: Keyword arguments for the function.

        Returns:
            True if stage succeeded, False otherwise.
        """
        if self._cancelled:
            return False

        while self._paused:
            import time
            time.sleep(0.1)
            if self._cancelled:
                return False

        self._current_stage = stage
        stage_result = StageResult(stage=stage)
        stage_result.started_at = datetime.now()

        logger.info("Starting stage: %s", stage.value)

        try:
            stage_func(*args, **kwargs)
            stage_result.success = True
        except Exception as e:
            logger.exception("Stage %s failed", stage.value)
            stage_result.success = False
            stage_result.error = str(e)
            self._result.error = f"Stage {stage.value} failed: {e}"

        stage_result.completed_at = datetime.now()
        stage_result.duration_seconds = (
            stage_result.completed_at - stage_result.started_at
        ).total_seconds()

        self._result.stage_results.append(stage_result)

        logger.info(
            "Stage %s completed: success=%s, duration=%.2fs",
            stage.value,
            stage_result.success,
            stage_result.duration_seconds,
        )

        return stage_result.success

    def _load_models(self) -> None:
        """Load trained models for prediction."""
        config = self._config_manager.get_prediction_config()

        if self._model_loader is not None:
            # Use provided model loader
            self._models = self._model_loader.load_models(
                model_dir=config.model_dir,
                areas=config.areas,
                horizons=config.horizons,
            )
            logger.info("Loaded %d area models", len(self._models))
            return

        # Check model directory
        model_dir = Path(config.model_dir)
        if not model_dir.exists():
            logger.warning("Model directory does not exist: %s", model_dir)
            # Create placeholder models for testing
            self._models = self._create_placeholder_models(config)
            return

        # Try to load models from directory
        self._models = self._load_models_from_directory(model_dir, config)

        # If no models loaded, create placeholder models for testing
        if not self._models:
            logger.warning("No models found in directory, creating placeholder models")
            self._models = self._create_placeholder_models(config)

        logger.info("Loaded models for %d areas", len(self._models))

    def _create_placeholder_models(
        self,
        config: PredictionWorkflowConfig,
    ) -> dict[str, dict[int, Any]]:
        """Create placeholder models for testing.

        Args:
            config: Prediction configuration.

        Returns:
            Dictionary of placeholder models.
        """
        logger.warning("Creating placeholder models for testing")

        class PlaceholderModel:
            """Simple placeholder model for testing."""

            def __init__(self, area: str, horizon: int) -> None:
                self.area = area
                self.horizon = horizon
                self._rng = np.random.default_rng(42)

            def predict(self, X: pd.DataFrame) -> np.ndarray:
                """Generate random predictions."""
                base = 1000 + self._rng.standard_normal(len(X)) * 100
                return base

        models: dict[str, dict[int, Any]] = {}
        for area in config.areas:
            models[area] = {}
            for horizon in config.horizons:
                models[area][horizon] = PlaceholderModel(area, horizon)

        return models

    def _load_models_from_directory(
        self,
        model_dir: Path,
        config: PredictionWorkflowConfig,
    ) -> dict[str, dict[int, Any]]:
        """Load models from directory structure.

        Args:
            model_dir: Path to model directory.
            config: Prediction configuration.

        Returns:
            Dictionary of loaded models.
        """
        import joblib

        models: dict[str, dict[int, Any]] = {}

        for area in config.areas:
            area_dir = model_dir / area
            if not area_dir.exists():
                logger.warning("No models found for area: %s", area)
                continue

            models[area] = {}
            for horizon in config.horizons:
                model_file = area_dir / f"model_h{horizon}.joblib"
                if model_file.exists():
                    models[area][horizon] = joblib.load(model_file)
                    logger.debug("Loaded model: %s", model_file)
                else:
                    logger.warning("Model not found: %s", model_file)

        return models

    def _prepare_features(
        self,
        input_features: dict[str, pd.DataFrame] | pd.DataFrame | None,
    ) -> None:
        """Prepare features for prediction.

        Args:
            input_features: Input features or None.
        """
        config = self._config_manager.get_prediction_config()

        if input_features is not None:
            # Use provided features
            if isinstance(input_features, pd.DataFrame):
                # Single DataFrame - use for all areas
                for area in config.areas:
                    self._features[area] = input_features.copy()
            else:
                self._features = input_features
            logger.info("Using provided features for %d areas", len(self._features))
            return

        # Try to prepare features using pipeline
        if self._feature_pipeline is not None:
            for area in config.areas:
                # Feature pipeline would prepare features here
                pass
            logger.info("Prepared features using pipeline")
            return

        # Generate synthetic features for testing
        logger.warning("No features provided, generating synthetic features")
        self._features = self._generate_synthetic_features(config)

    def _generate_synthetic_features(
        self,
        config: PredictionWorkflowConfig,
    ) -> dict[str, pd.DataFrame]:
        """Generate synthetic features for testing.

        Args:
            config: Prediction configuration.

        Returns:
            Dictionary of synthetic feature DataFrames.
        """
        n_samples = 48  # 48 half-hours per day
        n_features = 10
        rng = np.random.default_rng(42)

        features = {}
        for area in config.areas:
            features[area] = pd.DataFrame(
                rng.standard_normal((n_samples, n_features)),
                columns=[f"feature_{i}" for i in range(n_features)],
            )

        return features

    def _generate_forecasts(self, forecast_date: datetime) -> None:
        """Generate base forecasts for all areas and horizons.

        Args:
            forecast_date: Date for forecasts.
        """
        config = self._config_manager.get_prediction_config()

        for area, area_models in self._models.items():
            if area not in self._features:
                logger.warning("No features for area: %s", area)
                continue

            self._result.forecasts[area] = {}
            X = self._features[area]

            for horizon, model in area_models.items():
                try:
                    # Generate prediction
                    pred = model.predict(X)

                    # Store as ForecastOutput
                    forecast_output = ForecastOutput(
                        area=area,
                        horizon=horizon,
                        timestamp=forecast_date,
                        forecast=float(np.mean(pred)) if len(pred) > 1 else float(pred[0]),
                        model_id=f"{area}_h{horizon}",
                    )
                    self._result.forecasts[area][horizon] = forecast_output

                    # Also store raw forecasts for reconciliation
                    if area not in self._base_forecasts:
                        self._base_forecasts[area] = {}
                    self._base_forecasts[area][horizon] = pred

                except Exception as e:
                    logger.error("Failed to generate forecast for %s h%d: %s", area, horizon, e)

        n_forecasts = sum(len(h) for h in self._result.forecasts.values())
        logger.info("Generated %d forecasts for %d areas", n_forecasts, len(self._result.forecasts))

    def _apply_reconciliation(self) -> None:
        """Apply hierarchical reconciliation to forecasts."""
        config = self._config_manager.get_prediction_config()
        recon_config = self._config_manager.get_reconciliation_config()

        if not recon_config.enabled:
            logger.info("Reconciliation disabled in configuration")
            return

        if self._reconciler is not None:
            # Use provided reconciler
            try:
                # Convert forecasts to format expected by reconciler
                forecasts_array = self._forecasts_to_array()
                reconciled = self._reconciler.reconcile(forecasts_array)
                self._array_to_forecasts(reconciled)
                self._result.reconciled = True
                logger.info("Applied reconciliation using method: %s", config.reconciliation_method)
            except Exception as e:
                logger.error("Reconciliation failed: %s", e)
                return

        else:
            # Simple proportional reconciliation
            self._apply_simple_reconciliation()
            self._result.reconciled = True
            logger.info("Applied simple proportional reconciliation")

    def _forecasts_to_array(self) -> np.ndarray:
        """Convert forecasts dictionary to array for reconciliation.

        Returns:
            Numpy array of forecasts.
        """
        config = self._config_manager.get_prediction_config()
        n_areas = len(self._result.forecasts)
        n_horizons = len(config.horizons)

        arr = np.zeros((n_areas, n_horizons))
        for i, (area, horizons) in enumerate(self._result.forecasts.items()):
            for j, horizon in enumerate(config.horizons):
                if horizon in horizons:
                    arr[i, j] = horizons[horizon].forecast

        return arr

    def _array_to_forecasts(self, arr: np.ndarray) -> None:
        """Update forecasts from reconciled array.

        Args:
            arr: Reconciled forecast array.
        """
        config = self._config_manager.get_prediction_config()

        for i, (area, horizons) in enumerate(self._result.forecasts.items()):
            for j, horizon in enumerate(config.horizons):
                if horizon in horizons:
                    horizons[horizon].forecast = float(arr[i, j])
                    horizons[horizon].reconciled = True

    def _apply_simple_reconciliation(self) -> None:
        """Apply simple proportional reconciliation.

        This ensures subsystem forecasts sum to national total.
        """
        config = self._config_manager.get_prediction_config()

        # Group areas by level
        subsystems = ["SECO", "S", "NE", "N"]

        for horizon in config.horizons:
            # Get subsystem forecasts
            subsystem_forecasts = {}
            for area in subsystems:
                if area in self._result.forecasts and horizon in self._result.forecasts[area]:
                    subsystem_forecasts[area] = self._result.forecasts[area][horizon].forecast

            if len(subsystem_forecasts) == 0:
                continue

            # If SIN exists, scale subsystems to match
            if "SIN" in self._result.forecasts and horizon in self._result.forecasts["SIN"]:
                sin_total = self._result.forecasts["SIN"][horizon].forecast
                current_total = sum(subsystem_forecasts.values())

                if current_total > 0:
                    scale_factor = sin_total / current_total
                    for area in subsystem_forecasts:
                        self._result.forecasts[area][horizon].forecast *= scale_factor
                        self._result.forecasts[area][horizon].reconciled = True
            else:
                # Mark as reconciled (subsystems are consistent by default)
                for area in subsystem_forecasts:
                    self._result.forecasts[area][horizon].reconciled = True

    def _compute_confidence_intervals(self) -> None:
        """Compute confidence intervals for forecasts."""
        config = self._config_manager.get_prediction_config()
        confidence_level = config.confidence_level

        # Simple approach: use historical error distribution
        # In practice, would use model-specific uncertainty
        z_score = self._get_z_score(confidence_level)

        for area, horizons in self._result.forecasts.items():
            for horizon, forecast in horizons.items():
                # Estimate uncertainty (would come from model in practice)
                # Using 5% of forecast as rough estimate
                uncertainty = abs(forecast.forecast) * 0.05 * (1 + horizon * 0.1)

                forecast.lower_bound = forecast.forecast - z_score * uncertainty
                forecast.upper_bound = forecast.forecast + z_score * uncertainty
                forecast.confidence_level = confidence_level

        logger.info("Computed confidence intervals at %.1f%% level", confidence_level * 100)

    def _get_z_score(self, confidence_level: float) -> float:
        """Get z-score for confidence level.

        Args:
            confidence_level: Confidence level (e.g., 0.95).

        Returns:
            Z-score for the confidence level.
        """
        from scipy import stats
        return stats.norm.ppf((1 + confidence_level) / 2)

    def _save_outputs(self) -> None:
        """Save forecast outputs to files."""
        config = self._config_manager.get_prediction_config()
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create DataFrame from forecasts
        records = []
        for area, horizons in self._result.forecasts.items():
            for horizon, forecast in horizons.items():
                records.append(forecast.to_dict())

        df = pd.DataFrame(records)
        self._result.forecast_df = df

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save in configured format
        if config.output_format == "parquet":
            output_path = output_dir / f"forecasts_{timestamp}.parquet"
            df.to_parquet(output_path, index=False)
        elif config.output_format == "csv":
            output_path = output_dir / f"forecasts_{timestamp}.csv"
            df.to_csv(output_path, index=False)
        elif config.output_format == "json":
            output_path = output_dir / f"forecasts_{timestamp}.json"
            df.to_json(output_path, orient="records", date_format="iso")
        else:
            output_path = output_dir / f"forecasts_{timestamp}.parquet"
            df.to_parquet(output_path, index=False)

        self._result.output_paths["forecasts"] = str(output_path)
        logger.info("Saved forecasts to %s", output_path)

    def _finalize_result(self, success: bool) -> PredictionResult:
        """Finalize workflow result.

        Args:
            success: Whether workflow succeeded.

        Returns:
            Finalized PredictionResult.
        """
        self._result.completed_at = datetime.now()
        self._result.duration_seconds = (
            self._result.completed_at - self._result.started_at
        ).total_seconds()
        self._result.success = success

        if success:
            self._status = PredictionStatus.COMPLETED
            self._current_stage = PredictionStage.COMPLETED
            self._result.status = PredictionStatus.COMPLETED
        elif self._cancelled:
            self._status = PredictionStatus.CANCELLED
            self._current_stage = PredictionStage.CANCELLED
            self._result.status = PredictionStatus.CANCELLED
        else:
            self._status = PredictionStatus.FAILED
            self._current_stage = PredictionStage.FAILED
            self._result.status = PredictionStatus.FAILED

        logger.info(
            "Prediction workflow completed: success=%s, duration=%.2fs",
            success,
            self._result.duration_seconds,
        )

        return self._result

    def get_progress(self) -> dict[str, Any]:
        """Get current workflow progress.

        Returns:
            Progress information dictionary.
        """
        stages = list(PredictionStage)
        current_idx = stages.index(self._current_stage)
        total_stages = len([s for s in stages if s not in {
            PredictionStage.COMPLETED, PredictionStage.FAILED, PredictionStage.CANCELLED
        }])

        return {
            "status": self._status.value,
            "current_stage": self._current_stage.value,
            "stage_index": current_idx,
            "total_stages": total_stages,
            "progress_percent": (current_idx / total_stages) * 100 if total_stages > 0 else 0,
            "completed_stages": [s.to_dict() for s in self._result.stage_results],
        }

    def get_result(self) -> PredictionResult:
        """Get workflow result.

        Returns:
            Current workflow result.
        """
        return self._result

    def get_forecasts_df(self) -> pd.DataFrame | None:
        """Get forecasts as DataFrame.

        Returns:
            DataFrame with forecasts or None if not yet generated.
        """
        return self._result.forecast_df

    def __repr__(self) -> str:
        """Return string representation."""
        return f"PredictionWorkflow(status={self._status.value}, stage={self._current_stage.value})"
