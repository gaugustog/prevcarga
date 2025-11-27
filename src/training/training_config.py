"""Training configuration for universal trainer.

This module provides the TrainingConfig Pydantic model for configuring
multi-area model training with parallelization, checkpointing, and
hyperparameter optimization.

Example:
    ```python
    from src.training.training_config import TrainingConfig

    # Default configuration
    config = TrainingConfig(
        model_type="lgbm",
        areas=["SP", "RJ", "MG"]
    )

    # With hyperparameter optimization
    config = TrainingConfig(
        model_type="random_forest",
        areas=["SP", "RJ"],
        optimize_hyperparameters=True,
        optimization_trials=50,
        parallel_workers=8
    )

    # With checkpointing
    config = TrainingConfig(
        model_type="lgbm",
        areas=["SP", "RJ", "MG", "ES"],
        enable_checkpointing=True,
        checkpoint_dir="checkpoints/training",
        checkpoint_frequency=2
    )
    ```
"""

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Valid area codes for PrevCarga
VALID_AREAS = {
    "SP",
    "RJ",
    "MG",
    "ES",  # SECO
    "PR",
    "SC",
    "RS",  # S
    "BA",
    "PE",
    "CE",
    "ALPE",
    "PBRN",
    "BAOE",  # NE
    "PA",
    "AM",
    "AP",
    "AC",
    "RO",
    "TON",  # N
    "SECO",
    "S",
    "NE",
    "N",  # Subsystems
    "SIN",  # National total
}


class TrainingConfig(BaseModel):
    """Configuration for universal training interface.

    This class defines all parameters for parallel training of multiple models
    across different areas and horizons, including checkpointing, progress tracking,
    and automated hyperparameter optimization.

    Attributes:
        model_type: Type of model to train ("lgbm" or "random_forest").
        training_params: Model-specific configuration dictionary passed to fit().
        areas: List of area codes to train models for.
        horizons: List of forecast horizons (0-8). If None, trains all supported
            horizons for the model type.
        parallel_workers: Number of parallel workers for training. Defaults to 4.
            Set to 1 for sequential training.
        backend: Parallelization backend ("multiprocessing" or "threading").
            "multiprocessing" recommended for CPU-bound training.
        enable_checkpointing: Whether to save checkpoints during training.
        checkpoint_dir: Directory to save checkpoints. Created if doesn't exist.
        checkpoint_frequency: Save checkpoint every N areas completed.
        optimize_hyperparameters: Whether to run hyperparameter optimization
            before training. Uses Optuna for optimization.
        optimization_trials: Number of Optuna trials for hyperparameter search.
        cv_folds: Number of cross-validation folds for evaluation.
        memory_limit_gb: Maximum memory usage in GB. If set, limits parallel
            workers based on available memory. None means no limit.
        gpu_enabled: Whether to enable GPU acceleration (for compatible models).
            Currently not implemented for LGBM/RF in CPU-only mode.
    """

    # Model configuration
    model_type: Literal["lgbm", "random_forest"] = Field(description="Type of model to train")
    training_params: dict[str, Any] = Field(
        default_factory=dict, description="Model-specific configuration passed to fit()"
    )

    # Areas and horizons
    areas: list[str] = Field(description="List of area codes to train models for")
    horizons: list[int] | None = Field(
        default=None,
        description="List of forecast horizons (0-8). None means all supported horizons",
    )

    # Parallelization
    parallel_workers: int = Field(
        default=4, ge=1, le=64, description="Number of parallel workers for training"
    )
    backend: Literal["multiprocessing", "threading"] = Field(
        default="multiprocessing", description="Parallelization backend"
    )

    # Checkpointing
    enable_checkpointing: bool = Field(
        default=False, description="Whether to save checkpoints during training"
    )
    checkpoint_dir: str = Field(
        default="checkpoints/training", description="Directory to save checkpoints"
    )
    checkpoint_frequency: int = Field(
        default=1, ge=1, description="Save checkpoint every N areas completed"
    )

    # Hyperparameter optimization
    optimize_hyperparameters: bool = Field(
        default=False, description="Whether to run hyperparameter optimization"
    )
    optimization_trials: int = Field(
        default=100, ge=1, le=1000, description="Number of Optuna trials for hyperparameter search"
    )
    cv_folds: int = Field(default=5, ge=2, le=10, description="Number of cross-validation folds")

    # Resource management
    memory_limit_gb: float | None = Field(
        default=None, ge=1.0, description="Maximum memory usage in GB"
    )
    gpu_enabled: bool = Field(default=False, description="Whether to enable GPU acceleration")

    @field_validator("areas")
    @classmethod
    def validate_areas(cls, v: list[str]) -> list[str]:
        """Validate that all areas are recognized.

        Args:
            v: List of area codes.

        Returns:
            Validated list of area codes.

        Raises:
            ValueError: If any area is not recognized.
        """
        if not v:
            msg = "areas list cannot be empty"
            raise ValueError(msg)

        invalid_areas = set(v) - VALID_AREAS
        if invalid_areas:
            msg = (
                f"Invalid area codes: {sorted(invalid_areas)}. "
                f"Valid areas: {sorted(VALID_AREAS)}"
            )
            raise ValueError(msg)

        # Check for duplicates
        if len(v) != len(set(v)):
            duplicates = [area for area in set(v) if v.count(area) > 1]
            msg = f"Duplicate area codes found: {sorted(duplicates)}"
            raise ValueError(msg)

        return v

    @field_validator("horizons")
    @classmethod
    def validate_horizons(cls, v: list[int] | None) -> list[int] | None:
        """Validate that all horizons are in valid range.

        Args:
            v: List of horizon values or None.

        Returns:
            Validated list of horizons or None.

        Raises:
            ValueError: If any horizon is out of range [0, 8].
        """
        if v is None:
            return v

        if not v:
            msg = "horizons list cannot be empty if specified"
            raise ValueError(msg)

        invalid_horizons = [h for h in v if h < 0 or h > 8]
        if invalid_horizons:
            msg = f"Invalid horizons {invalid_horizons}. Must be in range [0, 8]"
            raise ValueError(msg)

        # Check for duplicates
        if len(v) != len(set(v)):
            duplicates = [h for h in set(v) if v.count(h) > 1]
            msg = f"Duplicate horizons found: {sorted(duplicates)}"
            raise ValueError(msg)

        return sorted(v)

    @field_validator("parallel_workers")
    @classmethod
    def validate_parallel_workers(cls, v: int, info: Any) -> int:
        """Validate parallel_workers is reasonable.

        Args:
            v: Number of parallel workers.
            info: Validation context.

        Returns:
            Validated number of workers.
        """
        if v > 32:
            logger.warning(
                "Large number of parallel workers (%d) may cause resource contention. "
                "Consider reducing to 8-16 workers.",
                v,
            )
        return v

    @field_validator("checkpoint_frequency")
    @classmethod
    def validate_checkpoint_frequency(cls, v: int, info: Any) -> int:
        """Validate checkpoint_frequency is reasonable.

        Args:
            v: Checkpoint frequency value.
            info: Validation context.

        Returns:
            Validated checkpoint frequency.
        """
        data = info.data if hasattr(info, "data") else {}
        enable_checkpointing = data.get("enable_checkpointing", False)

        if enable_checkpointing and v > 10:
            logger.warning(
                "Large checkpoint_frequency (%d) means infrequent checkpoints. "
                "Consider reducing to 1-5 for better recovery.",
                v,
            )
        return v

    @model_validator(mode="after")
    def validate_memory_and_workers(self) -> "TrainingConfig":
        """Validate memory limit is compatible with parallel workers.

        Returns:
            Validated configuration.
        """
        if self.memory_limit_gb is not None:
            # Rough estimate: each worker needs ~2GB for model training
            estimated_memory = self.parallel_workers * 2.0
            if estimated_memory > self.memory_limit_gb:
                logger.warning(
                    "Memory limit (%.1f GB) may be insufficient for %d workers. "
                    "Estimated memory needed: %.1f GB. Consider reducing parallel_workers.",
                    self.memory_limit_gb,
                    self.parallel_workers,
                    estimated_memory,
                )

        return self

    @model_validator(mode="after")
    def validate_gpu_enabled(self) -> "TrainingConfig":
        """Validate GPU configuration.

        Returns:
            Validated configuration.
        """
        if self.gpu_enabled:
            logger.warning(
                "gpu_enabled is True but GPU training not yet implemented for %s. "
                "Training will proceed on CPU.",
                self.model_type,
            )

        return self

    @model_validator(mode="after")
    def validate_hyperparameter_optimization(self) -> "TrainingConfig":
        """Validate hyperparameter optimization configuration.

        Returns:
            Validated configuration.
        """
        if self.optimize_hyperparameters:
            # If optimizing hyperparameters, ensure training_params doesn't
            # conflict with optimization
            if self.training_params.get("optimize_hyperparams"):
                logger.warning(
                    "Both TrainingConfig.optimize_hyperparameters and "
                    "training_params.optimize_hyperparams are True. Using TrainingConfig setting."
                )

        return self

    model_config = {
        "frozen": False,
        "validate_assignment": True,
        "extra": "forbid",
    }
