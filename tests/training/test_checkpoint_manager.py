"""Tests for CheckpointManager."""

import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

from src.training.checkpoint_manager import CheckpointManager


class TestCheckpointManager:
    """Test suite for CheckpointManager."""

    @pytest.fixture
    def temp_checkpoint_dir(self):
        """Create temporary directory for checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def manager(self, temp_checkpoint_dir):
        """Create CheckpointManager instance."""
        return CheckpointManager(checkpoint_dir=temp_checkpoint_dir)

    def test_initialization(self, temp_checkpoint_dir):
        """Test checkpoint manager initialization."""
        manager = CheckpointManager(checkpoint_dir=temp_checkpoint_dir)

        assert manager.checkpoint_dir == Path(temp_checkpoint_dir)
        assert manager.checkpoint_file.name == "latest_checkpoint.pkl"
        assert manager.checkpoint_dir.exists()

    def test_has_checkpoint_false(self, manager):
        """Test has_checkpoint returns False when no checkpoint exists."""
        assert not manager.has_checkpoint()

    def test_save_checkpoint(self, manager):
        """Test saving a checkpoint."""
        checkpoint = {
            "completed_areas": ["SP", "RJ"],
            "remaining_areas": ["MG", "ES"],
            "config": {"model_type": "lgbm"},
            "iteration": 2,
        }

        manager.save_checkpoint(checkpoint)

        assert manager.has_checkpoint()
        assert manager.checkpoint_file.exists()

    def test_save_checkpoint_missing_fields(self, manager):
        """Test that saving checkpoint with missing fields raises error."""
        incomplete_checkpoint = {
            "completed_areas": ["SP"]
            # Missing required fields
        }

        with pytest.raises(ValueError) as exc_info:
            manager.save_checkpoint(incomplete_checkpoint)

        assert "missing required fields" in str(exc_info.value).lower()

    def test_load_checkpoint(self, manager):
        """Test loading a checkpoint."""
        original_checkpoint = {
            "completed_areas": ["SP", "RJ"],
            "remaining_areas": ["MG"],
            "config": {"model_type": "lgbm", "areas": ["SP", "RJ", "MG"]},
            "iteration": 2,
            "trained_models": {},
        }

        manager.save_checkpoint(original_checkpoint)
        loaded_checkpoint = manager.load_checkpoint()

        assert loaded_checkpoint is not None
        assert loaded_checkpoint["completed_areas"] == ["SP", "RJ"]
        assert loaded_checkpoint["remaining_areas"] == ["MG"]
        assert loaded_checkpoint["config"] == original_checkpoint["config"]
        assert loaded_checkpoint["iteration"] == 2
        assert "timestamp" in loaded_checkpoint
        assert "checkpoint_version" in loaded_checkpoint

    def test_load_checkpoint_not_exists(self, manager):
        """Test loading checkpoint when none exists."""
        loaded = manager.load_checkpoint()
        assert loaded is None

    def test_clear_checkpoint(self, manager):
        """Test clearing a checkpoint."""
        checkpoint = {
            "completed_areas": ["SP"],
            "remaining_areas": ["RJ"],
            "config": {"model_type": "lgbm"},
        }

        manager.save_checkpoint(checkpoint)
        assert manager.has_checkpoint()

        manager.clear_checkpoint()
        assert not manager.has_checkpoint()

    def test_clear_checkpoint_not_exists(self, manager):
        """Test clearing checkpoint when none exists (should not error)."""
        manager.clear_checkpoint()  # Should not raise error
        assert not manager.has_checkpoint()

    def test_get_checkpoint_info(self, manager):
        """Test getting checkpoint metadata."""
        checkpoint = {
            "completed_areas": ["SP", "RJ"],
            "remaining_areas": ["MG", "ES"],
            "config": {"model_type": "lgbm"},
            "iteration": 2,
        }

        manager.save_checkpoint(checkpoint)
        info = manager.get_checkpoint_info()

        assert info is not None
        assert info["completed_count"] == 2
        assert info["remaining_count"] == 2
        assert info["total_count"] == 4
        assert info["iteration"] == 2
        assert "timestamp" in info
        assert "file_size_mb" in info
        assert info["file_size_mb"] > 0

    def test_get_checkpoint_info_not_exists(self, manager):
        """Test getting checkpoint info when none exists."""
        info = manager.get_checkpoint_info()
        assert info is None

    def test_create_backup(self, manager):
        """Test creating checkpoint backup."""
        checkpoint = {
            "completed_areas": ["SP"],
            "remaining_areas": ["RJ"],
            "config": {"model_type": "lgbm"},
        }

        manager.save_checkpoint(checkpoint)
        backup_path = manager.create_backup()

        assert backup_path.exists()
        assert backup_path.suffix == ".pkl"
        assert "checkpoint_backup_" in backup_path.name

        # Original checkpoint should still exist
        assert manager.has_checkpoint()

    def test_create_backup_with_name(self, manager):
        """Test creating named backup."""
        checkpoint = {
            "completed_areas": ["SP"],
            "remaining_areas": ["RJ"],
            "config": {"model_type": "lgbm"},
        }

        manager.save_checkpoint(checkpoint)
        backup_path = manager.create_backup("my_backup")

        assert backup_path.exists()
        assert backup_path.name == "my_backup.pkl"

    def test_create_backup_not_exists(self, manager):
        """Test creating backup when no checkpoint exists."""
        with pytest.raises(FileNotFoundError):
            manager.create_backup()

    def test_checkpoint_with_models(self, manager):
        """Test saving and loading checkpoint with model objects."""
        # Use a picklable simple object instead of a class defined in test
        # In production, this would be BaseModel instances which are picklable
        models = {
            "SP": {"name": "model_sp", "data": [1, 2, 3]},
            "RJ": {"name": "model_rj", "data": [4, 5, 6]},
        }

        checkpoint = {
            "completed_areas": ["SP", "RJ"],
            "remaining_areas": [],
            "trained_models": models,
            "config": {"model_type": "lgbm"},
        }

        manager.save_checkpoint(checkpoint)
        loaded = manager.load_checkpoint()

        assert loaded is not None
        assert "trained_models" in loaded
        assert "SP" in loaded["trained_models"]
        assert "RJ" in loaded["trained_models"]
        assert loaded["trained_models"]["SP"]["name"] == "model_sp"
        assert loaded["trained_models"]["SP"]["data"] == [1, 2, 3]

    def test_atomic_save(self, manager):
        """Test that checkpoint save is atomic (uses temp file)."""
        checkpoint = {
            "completed_areas": ["SP"],
            "remaining_areas": ["RJ", "MG"],
            "config": {"model_type": "lgbm"},
        }

        # Save checkpoint
        manager.save_checkpoint(checkpoint)

        # Check that temp files are cleaned up
        temp_files = list(manager.checkpoint_dir.glob("*.tmp"))
        assert len(temp_files) == 0

    def test_repr(self, manager):
        """Test string representation."""
        repr_str = repr(manager)
        assert "CheckpointManager" in repr_str
        assert "no checkpoint" in repr_str

        # Save checkpoint and check repr updates
        checkpoint = {"completed_areas": ["SP"], "remaining_areas": ["RJ"], "config": {}}
        manager.save_checkpoint(checkpoint)

        repr_str = repr(manager)
        assert "with checkpoint" in repr_str

    def test_multiple_save_load_cycles(self, manager):
        """Test multiple save/load cycles."""
        for i in range(3):
            checkpoint = {
                "completed_areas": [f"area_{j}" for j in range(i + 1)],
                "remaining_areas": [f"area_{j}" for j in range(i + 1, 5)],
                "config": {"iteration": i},
                "iteration": i,
            }

            manager.save_checkpoint(checkpoint)
            loaded = manager.load_checkpoint()

            assert loaded is not None
            assert loaded["iteration"] == i
            assert len(loaded["completed_areas"]) == i + 1

    def test_checkpoint_metadata(self, manager):
        """Test that checkpoint includes metadata fields."""
        checkpoint = {
            "completed_areas": ["SP"],
            "remaining_areas": ["RJ"],
            "config": {"model_type": "lgbm"},
        }

        manager.save_checkpoint(checkpoint)
        loaded = manager.load_checkpoint()

        # Check metadata fields added by manager
        assert "timestamp" in loaded
        assert isinstance(loaded["timestamp"], datetime)
        assert "checkpoint_version" in loaded
        assert loaded["checkpoint_version"] == "1.0.0"
