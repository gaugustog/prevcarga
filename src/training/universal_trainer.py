"""Universal trainer for parallel multi-area model training.

This module provides the UniversalTrainer class for training multiple models
across different areas with parallel execution, checkpointing, progress tracking,
and automated hyperparameter optimization.

Example:
    ```python
    from src.training import UniversalTrainer, TrainingConfig
    import pandas as pd

    # Prepare data
    data_dict = {
        "SP": (X_train_sp, y_train_sp),
        "RJ": (X_train_rj, y_train_rj),
        "MG": (X_train_mg, y_train_mg),
    }

    # Configure training
    config = TrainingConfig(
        model_type="lgbm",
        areas=["SP", "RJ", "MG"],
        parallel_workers=4,
        enable_checkpointing=True,
        optimize_hyperparameters=True
    )

    # Train models
    trainer = UniversalTrainer(config)
    trained_models = trainer.train_multiple_areas(data_dict)

    # Generate report
    report = trainer.generate_training_report()
    print(report)
    ```
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from src.models.base import BaseModel, get_model_class
from src.training.checkpoint_manager import CheckpointManager
from src.training.progress_tracker import ProgressTracker
from src.training.training_config import TrainingConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


class UniversalTrainer:
    """Universal trainer for multi-area parallel model training.

    This class provides a complete training interface that:
    - Trains multiple models in parallel across areas
    - Supports checkpointing and resume
    - Tracks progress with time estimates
    - Performs automated hyperparameter optimization
    - Validates trained models
    - Generates training reports

    Attributes:
        config: Training configuration.
        checkpoint_manager: Checkpoint manager instance.
        progress_tracker: Progress tracker instance.
        trained_models: Dictionary mapping area -> trained model.
        training_results: Dictionary with training metrics per area.
    """

    def __init__(self, config: TrainingConfig) -> None:
        """Initialize universal trainer.

        Args:
            config: Training configuration instance.
        """
        self.config = config
        self.checkpoint_manager = CheckpointManager(config.checkpoint_dir)
        self.progress_tracker: ProgressTracker | None = None
        self.trained_models: dict[str, BaseModel] = {}
        self.training_results: dict[str, dict[str, Any]] = {}

        logger.info(
            "Initialized UniversalTrainer: model=%s, areas=%s, workers=%d",
            config.model_type,
            config.areas,
            config.parallel_workers,
        )

    def train_multiple_areas(
        self, data_dict: dict[str, tuple[pd.DataFrame, pd.Series | pd.DataFrame]]
    ) -> dict[str, BaseModel]:
        """Train models for multiple areas in parallel.

        This is the main entry point for training. It:
        1. Checks for existing checkpoints
        2. Initializes progress tracking
        3. Trains models in parallel using ProcessPoolExecutor
        4. Saves checkpoints periodically
        5. Returns all trained models

        Args:
            data_dict: Dictionary mapping area code -> (X_train, y_train).
                      X_train is features DataFrame, y_train is target Series/DataFrame.

        Returns:
            Dictionary mapping area code -> trained BaseModel instance.

        Raises:
            ValueError: If data_dict doesn't match configured areas.
            RuntimeError: If training fails.

        Example:
            ```python
            data_dict = {
                "SP": (X_sp, y_sp),
                "RJ": (X_rj, y_rj),
            }
            trained_models = trainer.train_multiple_areas(data_dict)
            ```
        """
        # Validate data_dict matches configured areas
        missing_areas = set(self.config.areas) - set(data_dict.keys())
        if missing_areas:
            msg = f"Missing data for areas: {sorted(missing_areas)}"
            raise ValueError(msg)

        extra_areas = set(data_dict.keys()) - set(self.config.areas)
        if extra_areas:
            logger.warning("Extra areas in data_dict (will be ignored): %s", sorted(extra_areas))

        # Check for checkpoint to resume from
        checkpoint = None
        if self.config.enable_checkpointing:
            checkpoint = self.checkpoint_manager.load_checkpoint()

        if checkpoint:
            logger.info("Resuming training from checkpoint")
            return self.resume_training(data_dict, checkpoint)

        # Initialize progress tracker
        self.progress_tracker = ProgressTracker(
            total_tasks=len(self.config.areas), task_description="areas"
        )

        # Add logging callback
        def log_progress(tracker: ProgressTracker) -> None:
            summary = tracker.get_summary()
            logger.info(
                "Training progress: %d/%d areas (%.1f%%) | Elapsed: %s | ETA: %s",
                summary["completed_tasks"],
                summary["total_tasks"],
                summary["progress_percentage"],
                summary["elapsed_time"],
                summary["estimated_time_remaining"] or "calculating...",
            )

        self.progress_tracker.add_callback(log_progress)

        # Determine parallelization strategy
        if self.config.parallel_workers == 1:
            # Sequential training
            logger.info("Training %d areas sequentially", len(self.config.areas))
            return self._train_sequential(data_dict)
        # Parallel training
        logger.info(
            "Training %d areas in parallel with %d workers",
            len(self.config.areas),
            self.config.parallel_workers,
        )
        return self._train_parallel(data_dict)

    def _train_sequential(
        self, data_dict: dict[str, tuple[pd.DataFrame, pd.Series | pd.DataFrame]]
    ) -> dict[str, BaseModel]:
        """Train models sequentially (single-threaded).

        Args:
            data_dict: Dictionary mapping area -> (X, y) data.

        Returns:
            Dictionary mapping area -> trained model.
        """
        areas_to_train = self.config.areas.copy()

        for i, area in enumerate(areas_to_train):
            X, y = data_dict[area]

            logger.info("Training model for area %s (%d/%d)", area, i + 1, len(areas_to_train))

            try:
                model, result = self._train_single_area(area=area, X=X, y=y, config=self.config)

                self.trained_models[area] = model
                self.training_results[area] = result

                # Update progress
                if self.progress_tracker:
                    self.progress_tracker.mark_completed(area, metadata=result)

                # Save checkpoint if enabled
                if (
                    self.config.enable_checkpointing
                    and (i + 1) % self.config.checkpoint_frequency == 0
                ):
                    self._save_checkpoint(
                        completed_areas=list(self.trained_models.keys()),
                        remaining_areas=areas_to_train[i + 1 :],
                    )

            except Exception as e:
                logger.error("Training failed for area %s: %s", area, e, exc_info=True)
                # Save checkpoint before failing
                if self.config.enable_checkpointing:
                    self._save_checkpoint(
                        completed_areas=list(self.trained_models.keys()),
                        remaining_areas=areas_to_train[i:],
                    )
                raise RuntimeError(f"Training failed for area {area}") from e

        # Clear checkpoint on success
        if self.config.enable_checkpointing:
            self.checkpoint_manager.clear_checkpoint()

        logger.info(
            "Sequential training completed successfully for %d areas", len(self.trained_models)
        )
        return self.trained_models

    def _train_parallel(
        self, data_dict: dict[str, tuple[pd.DataFrame, pd.Series | pd.DataFrame]]
    ) -> dict[str, BaseModel]:
        """Train models in parallel using ProcessPoolExecutor.

        Args:
            data_dict: Dictionary mapping area -> (X, y) data.

        Returns:
            Dictionary mapping area -> trained model.
        """
        areas_to_train = self.config.areas.copy()
        completed_count = 0

        with ProcessPoolExecutor(max_workers=self.config.parallel_workers) as executor:
            # Submit all training jobs
            future_to_area = {}
            for area in areas_to_train:
                X, y = data_dict[area]
                future = executor.submit(
                    self._train_single_area, area=area, X=X, y=y, config=self.config
                )
                future_to_area[future] = area

            # Collect results as they complete
            for future in as_completed(future_to_area):
                area = future_to_area[future]

                try:
                    model, result = future.result()

                    self.trained_models[area] = model
                    self.training_results[area] = result
                    completed_count += 1

                    # Update progress
                    if self.progress_tracker:
                        self.progress_tracker.mark_completed(area, metadata=result)

                    logger.info("Successfully trained model for area %s", area)

                    # Save checkpoint if enabled
                    if (
                        self.config.enable_checkpointing
                        and completed_count % self.config.checkpoint_frequency == 0
                    ):
                        remaining = [a for a in areas_to_train if a not in self.trained_models]
                        self._save_checkpoint(
                            completed_areas=list(self.trained_models.keys()),
                            remaining_areas=remaining,
                        )

                except Exception as e:
                    logger.error("Training failed for area %s: %s", area, e, exc_info=True)
                    # Save checkpoint before failing
                    if self.config.enable_checkpointing:
                        remaining = [a for a in areas_to_train if a not in self.trained_models]
                        self._save_checkpoint(
                            completed_areas=list(self.trained_models.keys()),
                            remaining_areas=remaining,
                        )
                    raise RuntimeError(f"Training failed for area {area}") from e

        # Clear checkpoint on success
        if self.config.enable_checkpointing:
            self.checkpoint_manager.clear_checkpoint()

        logger.info(
            "Parallel training completed successfully for %d areas", len(self.trained_models)
        )
        return self.trained_models

    @staticmethod
    def _train_single_area(
        area: str,
        X: pd.DataFrame,  # noqa: N803
        y: pd.Series | pd.DataFrame,
        config: TrainingConfig,
    ) -> tuple[BaseModel, dict[str, Any]]:
        """Train a single model for one area.

        This is a static method to enable parallel execution with ProcessPoolExecutor.

        Args:
            area: Area code.
            X: Training features DataFrame.
            y: Training target Series or DataFrame.
            config: Training configuration.

        Returns:
            Tuple of (trained_model, training_results_dict).

        Raises:
            RuntimeError: If training fails.
        """
        train_start = datetime.now(UTC)

        logger.info("Training %s model for area %s", config.model_type, area)

        try:
            # Get model class
            model_class = get_model_class(config.model_type)
            model = model_class()

            # Prepare model configuration
            model_config = config.training_params.copy()

            # Override optimize_hyperparams if trainer-level optimization is enabled
            if config.optimize_hyperparameters:
                model_config["optimize_hyperparams"] = True
                model_config["n_trials"] = config.optimization_trials
                model_config["cv_folds"] = config.cv_folds

            # Train model
            model.fit(X, y, model_config)

            train_end = datetime.now(UTC)
            training_time = (train_end - train_start).total_seconds()

            # Collect training results
            result = {
                "area": area,
                "model_type": config.model_type,
                "training_time_seconds": training_time,
                "training_samples": len(X),
                "training_features": len(X.columns),
                "trained_at": train_end,
            }

            # Add model-specific metadata if available
            if hasattr(model, "_training_metadata"):
                result.update(model._training_metadata)

            logger.info(
                "Successfully trained %s for area %s in %.1f seconds",
                config.model_type,
                area,
                training_time,
            )

            return model, result

        except Exception as e:
            logger.error("Training failed for area %s: %s", area, e, exc_info=True)
            raise RuntimeError(f"Training failed for area {area}") from e

    def _save_checkpoint(self, completed_areas: list[str], remaining_areas: list[str]) -> None:
        """Save training checkpoint.

        Args:
            completed_areas: List of areas with completed training.
            remaining_areas: List of areas yet to be trained.
        """
        checkpoint = {
            "completed_areas": completed_areas,
            "remaining_areas": remaining_areas,
            "trained_models": self.trained_models,
            "training_results": self.training_results,
            "config": self.config.model_dump(),
            "iteration": len(completed_areas),
        }

        self.checkpoint_manager.save_checkpoint(checkpoint)

    def resume_training(
        self,
        data_dict: dict[str, tuple[pd.DataFrame, pd.Series | pd.DataFrame]],
        checkpoint: dict[str, Any],
    ) -> dict[str, BaseModel]:
        """Resume training from a checkpoint.

        Args:
            data_dict: Dictionary mapping area -> (X, y) data.
            checkpoint: Loaded checkpoint dictionary.

        Returns:
            Dictionary mapping area -> trained model.
        """
        # Restore state from checkpoint
        self.trained_models = checkpoint["trained_models"]
        self.training_results = checkpoint.get("training_results", {})
        completed_areas = checkpoint["completed_areas"]
        remaining_areas = checkpoint["remaining_areas"]

        logger.info(
            "Resuming training: %d areas completed, %d remaining",
            len(completed_areas),
            len(remaining_areas),
        )

        # Initialize progress tracker with remaining areas
        self.progress_tracker = ProgressTracker(
            total_tasks=len(self.config.areas), task_description="areas"
        )

        # Mark completed areas in progress tracker
        for area in completed_areas:
            result = self.training_results.get(area, {})
            self.progress_tracker.mark_completed(area, metadata=result)

        # Create new data_dict with only remaining areas
        remaining_data_dict = {area: data_dict[area] for area in remaining_areas}

        # Train remaining areas
        if self.config.parallel_workers == 1:
            self._train_sequential_remaining(remaining_data_dict, remaining_areas)
        else:
            self._train_parallel_remaining(remaining_data_dict, remaining_areas)

        # Clear checkpoint on success
        if self.config.enable_checkpointing:
            self.checkpoint_manager.clear_checkpoint()

        return self.trained_models

    def _train_sequential_remaining(
        self,
        data_dict: dict[str, tuple[pd.DataFrame, pd.Series | pd.DataFrame]],
        remaining_areas: list[str],
    ) -> None:
        """Train remaining areas sequentially after resume.

        Args:
            data_dict: Data for remaining areas.
            remaining_areas: List of areas to train.
        """
        for i, area in enumerate(remaining_areas):
            X, y = data_dict[area]

            logger.info("Training model for area %s (%d/%d)", area, i + 1, len(remaining_areas))

            try:
                model, result = self._train_single_area(area, X, y, self.config)

                self.trained_models[area] = model
                self.training_results[area] = result

                if self.progress_tracker:
                    self.progress_tracker.mark_completed(area, metadata=result)

                # Save checkpoint
                if (
                    self.config.enable_checkpointing
                    and (i + 1) % self.config.checkpoint_frequency == 0
                ):
                    self._save_checkpoint(
                        completed_areas=list(self.trained_models.keys()),
                        remaining_areas=remaining_areas[i + 1 :],
                    )

            except Exception as e:
                logger.error("Training failed for area %s: %s", area, e, exc_info=True)
                if self.config.enable_checkpointing:
                    self._save_checkpoint(
                        completed_areas=list(self.trained_models.keys()),
                        remaining_areas=remaining_areas[i:],
                    )
                raise

    def _train_parallel_remaining(
        self,
        data_dict: dict[str, tuple[pd.DataFrame, pd.Series | pd.DataFrame]],
        remaining_areas: list[str],
    ) -> None:
        """Train remaining areas in parallel after resume.

        Args:
            data_dict: Data for remaining areas.
            remaining_areas: List of areas to train.
        """
        completed_count = 0

        with ProcessPoolExecutor(max_workers=self.config.parallel_workers) as executor:
            future_to_area = {}
            for area in remaining_areas:
                X, y = data_dict[area]
                future = executor.submit(self._train_single_area, area, X, y, self.config)
                future_to_area[future] = area

            for future in as_completed(future_to_area):
                area = future_to_area[future]

                try:
                    model, result = future.result()

                    self.trained_models[area] = model
                    self.training_results[area] = result
                    completed_count += 1

                    if self.progress_tracker:
                        self.progress_tracker.mark_completed(area, metadata=result)

                    if (
                        self.config.enable_checkpointing
                        and completed_count % self.config.checkpoint_frequency == 0
                    ):
                        still_remaining = [
                            a for a in remaining_areas if a not in self.trained_models
                        ]
                        self._save_checkpoint(
                            completed_areas=list(self.trained_models.keys()),
                            remaining_areas=still_remaining,
                        )

                except Exception as e:
                    logger.error("Training failed for area %s: %s", area, e, exc_info=True)
                    if self.config.enable_checkpointing:
                        still_remaining = [
                            a for a in remaining_areas if a not in self.trained_models
                        ]
                        self._save_checkpoint(
                            completed_areas=list(self.trained_models.keys()),
                            remaining_areas=still_remaining,
                        )
                    raise

    def _cross_validate_model(
        self,
        model: BaseModel,
        X: pd.DataFrame,  # noqa: N803
        y: pd.Series | pd.DataFrame,
        cv_folds: int = 5,
    ) -> dict[str, float]:
        """Perform time series cross-validation on a model.

        Args:
            model: Trained model instance.
            X: Features DataFrame.
            y: Target Series or DataFrame.
            cv_folds: Number of CV folds.

        Returns:
            Dictionary with cross-validation metrics.
        """
        logger.info("Performing %d-fold time series cross-validation", cv_folds)

        # Handle DataFrame y
        if isinstance(y, pd.DataFrame):
            y = y.iloc[:, 0]

        tscv = TimeSeriesSplit(n_splits=cv_folds)
        cv_scores = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Get model class and create new instance for CV
            model_class = type(model)
            cv_model = model_class()

            # Train on fold
            config = self.config.training_params.copy()
            config["optimize_hyperparams"] = False  # Skip hyperopt in CV
            cv_model.fit(X_train, y_train, config)

            # Predict on validation
            preds = cv_model.predict(X_val)

            # Calculate MAE (use first horizon column if multi-horizon)
            pred_col = preds.columns[0]
            mae = np.mean(np.abs(preds[pred_col] - y_val))
            cv_scores.append(mae)

            logger.debug("Fold %d MAE: %.2f", fold + 1, mae)

        return {
            "cv_mae_mean": float(np.mean(cv_scores)),
            "cv_mae_std": float(np.std(cv_scores)),
            "cv_folds": cv_folds,
        }

    def train_with_hyperopt(
        self,
        area: str,
        X: pd.DataFrame,  # noqa: N803
        y: pd.Series | pd.DataFrame,
        n_trials: int = 100,
    ) -> BaseModel:
        """Train model with Optuna hyperparameter optimization.

        This is an alternative training method that uses Optuna directly
        at the trainer level rather than delegating to the model's internal
        optimization.

        Args:
            area: Area code.
            X: Training features.
            y: Training target.
            n_trials: Number of Optuna trials.

        Returns:
            Trained model with optimized hyperparameters.

        Note:
            Most users should use train_multiple_areas with
            optimize_hyperparameters=True in config instead.
        """
        logger.info("Starting Optuna hyperparameter optimization for area %s", area)

        # This is a simplified version - actual implementation would define
        # an objective function and use Optuna's study API
        # For now, delegate to model's internal optimization

        model_class = get_model_class(self.config.model_type)
        model = model_class()

        config = self.config.training_params.copy()
        config["optimize_hyperparams"] = True
        config["n_trials"] = n_trials

        model.fit(X, y, config)

        return model

    def validate_trained_models(self) -> dict[str, bool]:
        """Validate all trained models are properly fitted.

        Returns:
            Dictionary mapping area -> is_valid (True if model is fitted).

        Example:
            ```python
            validation = trainer.validate_trained_models()
            if all(validation.values()):
                print("All models valid!")
            else:
                invalid = [k for k, v in validation.items() if not v]
                print(f"Invalid models: {invalid}")
            ```
        """
        validation_results = {}

        for area, model in self.trained_models.items():
            is_valid = model.is_fitted()
            validation_results[area] = is_valid

            if not is_valid:
                logger.error("Model for area %s is not properly fitted", area)
            else:
                logger.debug("Model for area %s is valid", area)

        logger.info(
            "Model validation: %d/%d models valid",
            sum(validation_results.values()),
            len(validation_results),
        )

        return validation_results

    def generate_training_report(self) -> str:
        """Generate a summary report of training results.

        Returns:
            Formatted string report with training statistics.

        Example:
            ```python
            report = trainer.generate_training_report()
            print(report)
            # Outputs formatted report with timing, areas, etc.
            ```
        """
        if not self.trained_models:
            return "No models trained yet."

        lines = []
        lines.append("=" * 80)
        lines.append("UNIVERSAL TRAINER - TRAINING REPORT")
        lines.append("=" * 80)

        # Configuration
        lines.append(f"\nModel Type: {self.config.model_type}")
        lines.append(f"Areas Trained: {len(self.trained_models)}")
        lines.append(f"Parallel Workers: {self.config.parallel_workers}")
        lines.append(f"Hyperparameter Optimization: {self.config.optimize_hyperparameters}")

        # Progress summary
        if self.progress_tracker:
            summary = self.progress_tracker.get_summary()
            lines.append(f"\nTotal Training Time: {summary['elapsed_time']}")
            avg_time = summary.get("average_task_time")
            if avg_time:
                lines.append(f"Average Time per Area: {avg_time}")

        # Per-area results
        lines.append("\nPer-Area Results:")
        lines.append("-" * 80)
        lines.append(
            f"{'Area':<10} {'Samples':<10} {'Features':<10} {'Time (s)':<12} {'Status':<10}"
        )
        lines.append("-" * 80)

        for area in sorted(self.trained_models.keys()):
            result = self.training_results.get(area, {})
            samples = result.get("training_samples", "N/A")
            features = result.get("training_features", "N/A")
            time_sec = result.get("training_time_seconds", "N/A")
            if isinstance(time_sec, (int, float)):
                time_str = f"{time_sec:.1f}"
            else:
                time_str = str(time_sec)

            is_valid = self.trained_models[area].is_fitted()
            status = "OK" if is_valid else "INVALID"

            lines.append(f"{area:<10} {samples:<10} {features:<10} {time_str:<12} {status:<10}")

        lines.append("=" * 80)

        return "\n".join(lines)

    def save_all_models(self, output_dir: str | Path) -> dict[str, Path]:
        """Save all trained models to disk.

        Args:
            output_dir: Directory to save models. Created if doesn't exist.

        Returns:
            Dictionary mapping area -> saved model path.

        Example:
            ```python
            saved_paths = trainer.save_all_models("models/trained")
            for area, path in saved_paths.items():
                print(f"{area}: {path}")
            ```
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_paths = {}

        for area, model in self.trained_models.items():
            model_filename = f"{self.config.model_type}_{area}.joblib"
            model_path = output_dir / model_filename

            try:
                model.save(model_path)
                saved_paths[area] = model_path
                logger.info("Saved model for area %s to %s", area, model_path)
            except Exception as e:
                logger.error("Failed to save model for area %s: %s", area, e)

        return saved_paths

    def __repr__(self) -> str:
        """Return string representation of trainer.

        Returns:
            String with configuration and training status.
        """
        return (
            f"UniversalTrainer(model_type='{self.config.model_type}', "
            f"areas={len(self.config.areas)}, "
            f"trained={len(self.trained_models)})"
        )
