"""Universal training infrastructure for PrevCarga.

This package provides a complete training system for multi-area model training
with parallelization, checkpointing, progress tracking, and automated
hyperparameter optimization.

Example:
    ```python
    from src.training import UniversalTrainer, TrainingConfig
    import pandas as pd

    # Prepare training data
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
        checkpoint_frequency=1,
        optimize_hyperparameters=True,
        optimization_trials=50
    )

    # Train models
    trainer = UniversalTrainer(config)
    trained_models = trainer.train_multiple_areas(data_dict)

    # Validate and save
    validation = trainer.validate_trained_models()
    print(f"Valid models: {sum(validation.values())}/{len(validation)}")

    saved_paths = trainer.save_all_models("models/trained")

    # Generate report
    report = trainer.generate_training_report()
    print(report)
    ```
"""

from src.training.checkpoint_manager import CheckpointManager
from src.training.progress_tracker import ProgressTracker
from src.training.training_config import TrainingConfig
from src.training.universal_trainer import UniversalTrainer

__all__ = [
    "CheckpointManager",
    "ProgressTracker",
    "TrainingConfig",
    "UniversalTrainer",
]
