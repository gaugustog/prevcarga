"""Checkpoint management for training state persistence.

This module provides the CheckpointManager class for saving and loading
training checkpoints, enabling resume functionality for long-running
multi-area training jobs.

Example:
    ```python
    from src.training.checkpoint_manager import CheckpointManager

    # Create checkpoint manager
    manager = CheckpointManager(checkpoint_dir="checkpoints/training")

    # Save checkpoint
    checkpoint = {
        "completed_areas": ["SP", "RJ"],
        "remaining_areas": ["MG", "ES"],
        "trained_models": {"SP": model_sp, "RJ": model_rj},
        "config": training_config.model_dump(),
        "iteration": 2
    }
    manager.save_checkpoint(checkpoint)

    # Load checkpoint
    loaded = manager.load_checkpoint()
    if loaded:
        print(f"Resuming from area {loaded['iteration']}")

    # Clear checkpoint after successful training
    manager.clear_checkpoint()
    ```
"""

import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib

from src.utils.logger import get_logger

logger = get_logger(__name__)


class CheckpointManager:
    """Manages training checkpoints for resume functionality.

    This class provides methods to save, load, and clear training checkpoints.
    Checkpoints include completed areas, remaining areas, trained models,
    configuration, and iteration information.

    The checkpoint file format uses joblib for efficient serialization of
    both metadata and model objects.

    Attributes:
        checkpoint_dir: Directory where checkpoints are stored.
        checkpoint_file: Path to the main checkpoint file.
    """

    def __init__(self, checkpoint_dir: str | Path = "checkpoints/training") -> None:
        """Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoints. Created if doesn't exist.
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_file = self.checkpoint_dir / "latest_checkpoint.pkl"

        logger.debug("Initialized CheckpointManager at %s", self.checkpoint_dir)

    def save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Save training checkpoint to disk.

        This method saves the checkpoint atomically using a temporary file
        to prevent corruption if the process is interrupted during save.

        The checkpoint dictionary should contain:
        - completed_areas: List of areas already trained
        - remaining_areas: List of areas yet to train
        - trained_models: Dictionary mapping area -> model instance
        - config: Training configuration dictionary
        - iteration: Current iteration/progress counter
        - timestamp: When checkpoint was created (added automatically)

        Args:
            checkpoint: Dictionary containing checkpoint data.

        Raises:
            ValueError: If checkpoint is missing required fields.
            OSError: If checkpoint cannot be saved to disk.

        Example:
            ```python
            checkpoint = {
                "completed_areas": ["SP", "RJ"],
                "remaining_areas": ["MG"],
                "trained_models": {"SP": model1, "RJ": model2},
                "config": {"model_type": "lgbm"},
                "iteration": 2
            }
            manager.save_checkpoint(checkpoint)
            ```
        """
        # Validate required fields
        required_fields = ["completed_areas", "remaining_areas", "config"]
        missing_fields = [f for f in required_fields if f not in checkpoint]
        if missing_fields:
            msg = f"Checkpoint missing required fields: {missing_fields}"
            raise ValueError(msg)

        # Add metadata
        checkpoint_with_meta = checkpoint.copy()
        checkpoint_with_meta["timestamp"] = datetime.now(UTC)
        checkpoint_with_meta["checkpoint_version"] = "1.0.0"

        # Save atomically using temporary file
        try:
            # Create temporary file in the same directory for atomic rename
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=self.checkpoint_dir, delete=False, suffix=".tmp"
            ) as tmp_file:
                tmp_path = Path(tmp_file.name)
                joblib.dump(checkpoint_with_meta, tmp_file)

            # Atomic rename (on most filesystems)
            tmp_path.replace(self.checkpoint_file)

            logger.info(
                "Saved checkpoint: %d areas completed, %d remaining",
                len(checkpoint["completed_areas"]),
                len(checkpoint["remaining_areas"]),
            )

        except Exception as e:
            # Clean up temporary file if it exists
            if tmp_path.exists():
                tmp_path.unlink()
            msg = f"Failed to save checkpoint: {e}"
            raise OSError(msg) from e

    def load_checkpoint(self) -> dict[str, Any] | None:
        """Load the most recent checkpoint from disk.

        Returns:
            Checkpoint dictionary if found, None otherwise.
            The dictionary contains all saved fields including:
            - completed_areas: List of areas already trained
            - remaining_areas: List of areas yet to train
            - trained_models: Dictionary mapping area -> model instance
            - config: Training configuration dictionary
            - iteration: Current iteration/progress counter
            - timestamp: When checkpoint was created
            - checkpoint_version: Checkpoint format version

        Raises:
            OSError: If checkpoint file exists but cannot be loaded.

        Example:
            ```python
            checkpoint = manager.load_checkpoint()
            if checkpoint:
                print(f"Resuming from iteration {checkpoint['iteration']}")
                print(f"Completed: {checkpoint['completed_areas']}")
            else:
                print("No checkpoint found, starting fresh")
            ```
        """
        if not self.checkpoint_file.exists():
            logger.debug("No checkpoint file found at %s", self.checkpoint_file)
            return None

        try:
            checkpoint = joblib.load(self.checkpoint_file)

            # Validate checkpoint structure
            if not isinstance(checkpoint, dict):
                logger.error("Invalid checkpoint format: expected dict, got %s", type(checkpoint))
                return None

            logger.info(
                "Loaded checkpoint from %s: %d areas completed, %d remaining",
                checkpoint.get("timestamp", "unknown time"),
                len(checkpoint.get("completed_areas", [])),
                len(checkpoint.get("remaining_areas", [])),
            )

            return checkpoint

        except Exception as e:
            msg = f"Failed to load checkpoint from {self.checkpoint_file}: {e}"
            raise OSError(msg) from e

    def clear_checkpoint(self) -> None:
        """Remove checkpoint file from disk.

        This should be called after successful completion of training
        to clean up checkpoint files.

        Example:
            ```python
            # After all training completed successfully
            manager.clear_checkpoint()
            print("Checkpoint cleared")
            ```
        """
        if self.checkpoint_file.exists():
            try:
                self.checkpoint_file.unlink()
                logger.info("Cleared checkpoint file: %s", self.checkpoint_file)
            except Exception as e:
                logger.error("Failed to clear checkpoint file: %s", e)
        else:
            logger.debug("No checkpoint file to clear")

    def has_checkpoint(self) -> bool:
        """Check if a checkpoint file exists.

        Returns:
            True if checkpoint file exists, False otherwise.

        Example:
            ```python
            if manager.has_checkpoint():
                print("Checkpoint available for resume")
            else:
                print("No checkpoint, starting fresh")
            ```
        """
        return self.checkpoint_file.exists()

    def get_checkpoint_info(self) -> dict[str, Any] | None:
        """Get metadata about the checkpoint without loading full data.

        This is useful for displaying checkpoint information without
        loading potentially large model objects.

        Returns:
            Dictionary with checkpoint metadata if found, None otherwise.
            Contains: timestamp, completed_areas count, remaining_areas count,
            checkpoint_version, file_size.

        Example:
            ```python
            info = manager.get_checkpoint_info()
            if info:
                print(f"Checkpoint from {info['timestamp']}")
                print(f"File size: {info['file_size_mb']:.1f} MB")
                print(f"Progress: {info['completed_count']}/{info['total_count']}")
            ```
        """
        if not self.has_checkpoint():
            return None

        try:
            # Load checkpoint to get metadata
            checkpoint = self.load_checkpoint()
            if checkpoint is None:
                return None

            # Get file size
            file_size_bytes = self.checkpoint_file.stat().st_size
            file_size_mb = file_size_bytes / (1024 * 1024)

            info = {
                "timestamp": checkpoint.get("timestamp"),
                "checkpoint_version": checkpoint.get("checkpoint_version"),
                "completed_count": len(checkpoint.get("completed_areas", [])),
                "remaining_count": len(checkpoint.get("remaining_areas", [])),
                "total_count": (
                    len(checkpoint.get("completed_areas", []))
                    + len(checkpoint.get("remaining_areas", []))
                ),
                "file_size_mb": file_size_mb,
                "iteration": checkpoint.get("iteration", 0),
            }

            return info

        except Exception as e:
            logger.error("Failed to get checkpoint info: %s", e)
            return None

    def create_backup(self, backup_name: str | None = None) -> Path:
        """Create a backup copy of the current checkpoint.

        Args:
            backup_name: Optional name for backup file. If None, uses timestamp.

        Returns:
            Path to the backup file.

        Raises:
            FileNotFoundError: If no checkpoint exists to backup.
            OSError: If backup cannot be created.

        Example:
            ```python
            # Create timestamped backup
            backup_path = manager.create_backup()
            print(f"Backup created at {backup_path}")

            # Create named backup
            backup_path = manager.create_backup("before_hyperopt")
            ```
        """
        if not self.has_checkpoint():
            msg = "No checkpoint file exists to backup"
            raise FileNotFoundError(msg)

        try:
            # Generate backup filename
            if backup_name is None:
                timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
                backup_name = f"checkpoint_backup_{timestamp}.pkl"
            elif not backup_name.endswith(".pkl"):
                backup_name = f"{backup_name}.pkl"

            backup_path = self.checkpoint_dir / backup_name

            # Copy checkpoint file
            import shutil

            shutil.copy2(self.checkpoint_file, backup_path)

            logger.info("Created checkpoint backup at %s", backup_path)
            return backup_path

        except Exception as e:
            msg = f"Failed to create checkpoint backup: {e}"
            raise OSError(msg) from e

    def __repr__(self) -> str:
        """Return string representation of checkpoint manager.

        Returns:
            String with checkpoint directory and existence status.
        """
        has_ckpt = "with checkpoint" if self.has_checkpoint() else "no checkpoint"
        return f"CheckpointManager(dir='{self.checkpoint_dir}', {has_ckpt})"
