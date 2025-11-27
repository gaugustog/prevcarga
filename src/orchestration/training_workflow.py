"""Training workflow orchestration for PrevCarga.

This module provides a high-level training workflow that orchestrates
the complete training pipeline including data loading, feature engineering,
model training, validation, and artifact management.

Key Components:
- TrainingWorkflow: Main workflow orchestration class
- WorkflowStatus: Workflow execution status tracking
- WorkflowStage: Training workflow stages
- WorkflowResult: Training workflow results

Example:
    ```python
    from src.orchestration import ConfigManager
    from src.orchestration.training_workflow import TrainingWorkflow

    # Load configuration
    config_manager = ConfigManager.from_yaml("config/prevcarga.yaml")

    # Create and run workflow
    workflow = TrainingWorkflow(config_manager)
    result = workflow.run()

    # Check results
    print(f"Success: {result.success}")
    print(f"Trained models: {len(result.trained_models)}")
    ```
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.orchestration.config_manager import ConfigManager, TrainingWorkflowConfig
from src.training import CheckpointManager, ProgressTracker, TrainingConfig, UniversalTrainer
from src.utils.logger import get_logger

logger = get_logger(__name__)


class WorkflowStage(Enum):
    """Training workflow stages.

    Attributes:
        INITIALIZED: Workflow initialized but not started.
        LOADING_DATA: Loading training data.
        FEATURE_ENGINEERING: Generating features.
        TRAINING: Training models.
        VALIDATING: Validating trained models.
        SAVING: Saving models and artifacts.
        COMPLETED: Workflow completed successfully.
        FAILED: Workflow failed.
        CANCELLED: Workflow was cancelled.
    """

    INITIALIZED = "initialized"
    LOADING_DATA = "loading_data"
    FEATURE_ENGINEERING = "feature_engineering"
    TRAINING = "training"
    VALIDATING = "validating"
    SAVING = "saving"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WorkflowStatus(Enum):
    """Workflow execution status.

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

    stage: WorkflowStage
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
class WorkflowResult:
    """Result of training workflow execution.

    Attributes:
        success: Whether workflow completed successfully.
        status: Final workflow status.
        started_at: When workflow started.
        completed_at: When workflow completed.
        duration_seconds: Total duration in seconds.
        stage_results: Results for each stage.
        trained_models: Dictionary of trained models by area.
        training_metrics: Training metrics by area.
        validation_results: Validation results by area.
        model_paths: Paths to saved models.
        error: Error message if failed.
    """

    success: bool = False
    status: WorkflowStatus = WorkflowStatus.PENDING
    started_at: datetime | None = None
    completed_at: datetime | None = None
    duration_seconds: float = 0.0
    stage_results: list[StageResult] = field(default_factory=list)
    trained_models: dict[str, Any] = field(default_factory=dict)
    training_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    validation_results: dict[str, bool] = field(default_factory=dict)
    model_paths: dict[str, str] = field(default_factory=dict)
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
            "trained_models": list(self.trained_models.keys()),
            "training_metrics": self.training_metrics,
            "validation_results": self.validation_results,
            "model_paths": self.model_paths,
            "error": self.error,
        }

    def get_summary(self) -> str:
        """Get human-readable summary.

        Returns:
            Summary string.
        """
        lines = [
            "=" * 60,
            "Training Workflow Summary",
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
        lines.append(f"Models Trained: {len(self.trained_models)}")
        lines.append(f"Models Validated: {sum(self.validation_results.values())}/{len(self.validation_results)}")

        if self.error:
            lines.append("")
            lines.append(f"Error: {self.error}")

        lines.append("=" * 60)
        return "\n".join(lines)


class TrainingWorkflow:
    """High-level training workflow orchestration.

    Orchestrates the complete training pipeline including:
    - Data loading and preparation
    - Feature engineering
    - Model training with parallel execution
    - Model validation
    - Artifact saving and management

    Example:
        >>> config_manager = ConfigManager.from_yaml("config/prevcarga.yaml")
        >>> workflow = TrainingWorkflow(config_manager)
        >>> result = workflow.run()
        >>> print(result.get_summary())
    """

    def __init__(
        self,
        config_manager: ConfigManager,
        data_loader: Any | None = None,
        feature_pipeline: Any | None = None,
    ) -> None:
        """Initialize training workflow.

        Args:
            config_manager: Configuration manager instance.
            data_loader: Optional data loader instance.
            feature_pipeline: Optional feature pipeline instance.
        """
        self._config_manager = config_manager
        self._data_loader = data_loader
        self._feature_pipeline = feature_pipeline

        self._status = WorkflowStatus.PENDING
        self._current_stage = WorkflowStage.INITIALIZED
        self._result = WorkflowResult()
        self._cancelled = False
        self._paused = False

        self._trainer: UniversalTrainer | None = None
        self._training_data: dict[str, tuple[pd.DataFrame, pd.Series]] = {}

        logger.info("Initialized TrainingWorkflow")

    @property
    def status(self) -> WorkflowStatus:
        """Get current workflow status."""
        return self._status

    @property
    def current_stage(self) -> WorkflowStage:
        """Get current workflow stage."""
        return self._current_stage

    def run(
        self,
        training_data: dict[str, tuple[pd.DataFrame, pd.Series]] | None = None,
    ) -> WorkflowResult:
        """Run the complete training workflow.

        Args:
            training_data: Pre-loaded training data by area.
                If None, will attempt to load data.

        Returns:
            WorkflowResult with training outcomes.
        """
        self._result = WorkflowResult()
        self._result.started_at = datetime.now()
        self._status = WorkflowStatus.RUNNING
        self._cancelled = False

        try:
            # Stage 1: Load Data
            if not self._execute_stage(WorkflowStage.LOADING_DATA, self._load_data, training_data):
                return self._finalize_result(success=False)

            # Stage 2: Feature Engineering
            if not self._execute_stage(WorkflowStage.FEATURE_ENGINEERING, self._generate_features):
                return self._finalize_result(success=False)

            # Stage 3: Training
            if not self._execute_stage(WorkflowStage.TRAINING, self._train_models):
                return self._finalize_result(success=False)

            # Stage 4: Validation
            if not self._execute_stage(WorkflowStage.VALIDATING, self._validate_models):
                return self._finalize_result(success=False)

            # Stage 5: Save Models
            if not self._execute_stage(WorkflowStage.SAVING, self._save_models):
                return self._finalize_result(success=False)

            return self._finalize_result(success=True)

        except Exception as e:
            logger.exception("Training workflow failed with exception")
            self._result.error = str(e)
            return self._finalize_result(success=False)

    def run_with_data(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series | pd.DataFrame,
        area: str = "default",
    ) -> WorkflowResult:
        """Run training workflow with provided data for a single area.

        Args:
            X_train: Training features DataFrame.
            y_train: Training target.
            area: Area identifier.

        Returns:
            WorkflowResult with training outcomes.
        """
        training_data = {area: (X_train, y_train)}
        return self.run(training_data=training_data)

    def cancel(self) -> None:
        """Cancel the running workflow."""
        logger.info("Cancelling training workflow")
        self._cancelled = True
        self._status = WorkflowStatus.CANCELLED

    def pause(self) -> None:
        """Pause the running workflow."""
        logger.info("Pausing training workflow")
        self._paused = True
        self._status = WorkflowStatus.PAUSED

    def resume(self) -> None:
        """Resume a paused workflow."""
        if self._status == WorkflowStatus.PAUSED:
            logger.info("Resuming training workflow")
            self._paused = False
            self._status = WorkflowStatus.RUNNING

    def _execute_stage(
        self,
        stage: WorkflowStage,
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

    def _load_data(
        self,
        training_data: dict[str, tuple[pd.DataFrame, pd.Series]] | None = None,
    ) -> None:
        """Load training data.

        Args:
            training_data: Pre-loaded training data or None.

        Raises:
            ValueError: If no data available.
        """
        if training_data is not None:
            self._training_data = training_data
            logger.info("Using provided training data for %d areas", len(training_data))
            return

        # Try to load data using data loader
        if self._data_loader is not None:
            config = self._config_manager.get_training_config()
            self._training_data = self._data_loader.load_training_data(
                areas=config.areas,
            )
            logger.info("Loaded training data for %d areas", len(self._training_data))
            return

        # Generate synthetic data for testing
        logger.warning("No data loader provided, generating synthetic data for testing")
        config = self._config_manager.get_training_config()
        self._training_data = self._generate_synthetic_data(config.areas)

    def _generate_synthetic_data(
        self,
        areas: list[str],
    ) -> dict[str, tuple[pd.DataFrame, pd.Series]]:
        """Generate synthetic training data for testing.

        Args:
            areas: List of area codes.

        Returns:
            Dictionary mapping area to (X, y) tuples.
        """
        n_samples = 1000
        n_features = 10
        rng = np.random.default_rng(42)

        data = {}
        for area in areas:
            X = pd.DataFrame(
                rng.standard_normal((n_samples, n_features)),
                columns=[f"feature_{i}" for i in range(n_features)],
            )
            y = pd.Series(
                X.sum(axis=1) + rng.standard_normal(n_samples) * 0.1,
                name="target",
            )
            data[area] = (X, y)

        return data

    def _generate_features(self) -> None:
        """Generate features using feature pipeline."""
        if self._feature_pipeline is None:
            logger.info("No feature pipeline configured, using raw data")
            return

        config = self._config_manager.get_feature_config()
        logger.info("Generating features with pipeline")

        for area, (X, y) in self._training_data.items():
            # Apply feature pipeline
            X_features = self._feature_pipeline.transform(X)
            self._training_data[area] = (X_features, y)
            logger.debug("Generated features for area %s: %d features", area, X_features.shape[1])

    def _train_models(self) -> None:
        """Train models using UniversalTrainer."""
        config = self._config_manager.get_training_config()

        # Create TrainingConfig for UniversalTrainer
        training_config = TrainingConfig(
            model_type=config.model_types[0] if config.model_types else "lgbm",
            areas=list(self._training_data.keys()),
            horizons=config.horizons if config.horizons else None,
            parallel_workers=config.parallel_workers,
            enable_checkpointing=config.checkpoint_enabled,
            checkpoint_dir=config.checkpoint_dir,
            optimize_hyperparameters=config.optimize_hyperparameters,
            optimization_trials=config.optimization_trials,
            cv_folds=config.cv_folds,
        )

        self._trainer = UniversalTrainer(training_config)

        # Train models
        self._result.trained_models = self._trainer.train_multiple_areas(self._training_data)

        # Collect training metrics
        for area, model in self._result.trained_models.items():
            if hasattr(model, "get_metrics"):
                self._result.training_metrics[area] = model.get_metrics()
            else:
                self._result.training_metrics[area] = {}

        logger.info("Trained %d models", len(self._result.trained_models))

    def _validate_models(self) -> None:
        """Validate trained models."""
        if self._trainer is None:
            logger.warning("No trainer available for validation")
            return

        # Use trainer's validation method
        self._result.validation_results = self._trainer.validate_trained_models()

        valid_count = sum(self._result.validation_results.values())
        total_count = len(self._result.validation_results)
        logger.info("Validated %d/%d models successfully", valid_count, total_count)

    def _save_models(self) -> None:
        """Save trained models to disk."""
        config = self._config_manager.get_training_config()

        if not config.save_models:
            logger.info("Model saving disabled in configuration")
            return

        if self._trainer is None:
            logger.warning("No trainer available for saving models")
            return

        output_dir = Path(config.model_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self._result.model_paths = self._trainer.save_all_models(str(output_dir))
        logger.info("Saved %d models to %s", len(self._result.model_paths), output_dir)

    def _finalize_result(self, success: bool) -> WorkflowResult:
        """Finalize workflow result.

        Args:
            success: Whether workflow succeeded.

        Returns:
            Finalized WorkflowResult.
        """
        self._result.completed_at = datetime.now()
        self._result.duration_seconds = (
            self._result.completed_at - self._result.started_at
        ).total_seconds()
        self._result.success = success

        if success:
            self._status = WorkflowStatus.COMPLETED
            self._current_stage = WorkflowStage.COMPLETED
            self._result.status = WorkflowStatus.COMPLETED
        elif self._cancelled:
            self._status = WorkflowStatus.CANCELLED
            self._current_stage = WorkflowStage.CANCELLED
            self._result.status = WorkflowStatus.CANCELLED
        else:
            self._status = WorkflowStatus.FAILED
            self._current_stage = WorkflowStage.FAILED
            self._result.status = WorkflowStatus.FAILED

        logger.info(
            "Training workflow completed: success=%s, duration=%.2fs",
            success,
            self._result.duration_seconds,
        )

        return self._result

    def get_progress(self) -> dict[str, Any]:
        """Get current workflow progress.

        Returns:
            Progress information dictionary.
        """
        stages = list(WorkflowStage)
        current_idx = stages.index(self._current_stage)
        total_stages = len([s for s in stages if s not in {
            WorkflowStage.COMPLETED, WorkflowStage.FAILED, WorkflowStage.CANCELLED
        }])

        return {
            "status": self._status.value,
            "current_stage": self._current_stage.value,
            "stage_index": current_idx,
            "total_stages": total_stages,
            "progress_percent": (current_idx / total_stages) * 100 if total_stages > 0 else 0,
            "completed_stages": [s.to_dict() for s in self._result.stage_results],
        }

    def get_result(self) -> WorkflowResult:
        """Get workflow result.

        Returns:
            Current workflow result.
        """
        return self._result

    def generate_report(self) -> str:
        """Generate training workflow report.

        Returns:
            HTML or text report.
        """
        if self._trainer is not None:
            return self._trainer.generate_training_report()
        return self._result.get_summary()

    def __repr__(self) -> str:
        """Return string representation."""
        return f"TrainingWorkflow(status={self._status.value}, stage={self._current_stage.value})"
