"""Tests for TrainingConfig validation."""

import pytest
from pydantic import ValidationError

from src.training.training_config import TrainingConfig


class TestTrainingConfig:
    """Test suite for TrainingConfig."""

    def test_minimal_config(self):
        """Test minimal valid configuration."""
        config = TrainingConfig(model_type="lgbm", areas=["SP", "RJ"])

        assert config.model_type == "lgbm"
        assert config.areas == ["SP", "RJ"]
        assert config.parallel_workers == 4  # Default
        assert config.enable_checkpointing is False  # Default

    def test_full_config(self):
        """Test configuration with all fields specified."""
        config = TrainingConfig(
            model_type="random_forest",
            training_params={"n_estimators": 200},
            areas=["SP", "RJ", "MG", "ES"],
            horizons=[0, 1, 2],
            parallel_workers=8,
            backend="multiprocessing",
            enable_checkpointing=True,
            checkpoint_dir="checkpoints/test",
            checkpoint_frequency=2,
            optimize_hyperparameters=True,
            optimization_trials=50,
            cv_folds=3,
            memory_limit_gb=16.0,
            gpu_enabled=False,
        )

        assert config.model_type == "random_forest"
        assert config.areas == ["SP", "RJ", "MG", "ES"]
        assert config.horizons == [0, 1, 2]
        assert config.parallel_workers == 8
        assert config.enable_checkpointing is True

    def test_invalid_model_type(self):
        """Test that invalid model_type raises error."""
        with pytest.raises(ValidationError) as exc_info:
            TrainingConfig(model_type="invalid_model", areas=["SP"])  # type: ignore[arg-type]

        assert "model_type" in str(exc_info.value)

    def test_empty_areas_list(self):
        """Test that empty areas list raises error."""
        with pytest.raises(ValidationError) as exc_info:
            TrainingConfig(model_type="lgbm", areas=[])

        assert "areas list cannot be empty" in str(exc_info.value)

    def test_invalid_area_codes(self):
        """Test that invalid area codes raise error."""
        with pytest.raises(ValidationError) as exc_info:
            TrainingConfig(model_type="lgbm", areas=["SP", "INVALID", "RJ"])

        assert "Invalid area codes" in str(exc_info.value)
        assert "INVALID" in str(exc_info.value)

    def test_duplicate_areas(self):
        """Test that duplicate areas raise error."""
        with pytest.raises(ValidationError) as exc_info:
            TrainingConfig(model_type="lgbm", areas=["SP", "RJ", "SP"])

        assert "Duplicate area codes" in str(exc_info.value)

    def test_valid_horizons(self):
        """Test valid horizon specification."""
        config = TrainingConfig(model_type="lgbm", areas=["SP"], horizons=[0, 1, 2, 3])

        assert config.horizons == [0, 1, 2, 3]

    def test_invalid_horizons(self):
        """Test that invalid horizons raise error."""
        with pytest.raises(ValidationError) as exc_info:
            TrainingConfig(
                model_type="lgbm", areas=["SP"], horizons=[0, 1, 9, 10]  # 9 and 10 are invalid
            )

        assert "Invalid horizons" in str(exc_info.value)

    def test_duplicate_horizons(self):
        """Test that duplicate horizons raise error."""
        with pytest.raises(ValidationError) as exc_info:
            TrainingConfig(model_type="lgbm", areas=["SP"], horizons=[0, 1, 1, 2])

        assert "Duplicate horizons" in str(exc_info.value)

    def test_empty_horizons_list(self):
        """Test that empty horizons list (when specified) raises error."""
        with pytest.raises(ValidationError) as exc_info:
            TrainingConfig(model_type="lgbm", areas=["SP"], horizons=[])

        assert "horizons list cannot be empty" in str(exc_info.value)

    def test_horizons_none(self):
        """Test that horizons=None is valid (means all supported)."""
        config = TrainingConfig(model_type="lgbm", areas=["SP"], horizons=None)

        assert config.horizons is None

    def test_parallel_workers_bounds(self):
        """Test parallel_workers bounds validation."""
        # Valid: 1 worker
        config = TrainingConfig(model_type="lgbm", areas=["SP"], parallel_workers=1)
        assert config.parallel_workers == 1

        # Valid: max workers
        config = TrainingConfig(model_type="lgbm", areas=["SP"], parallel_workers=64)
        assert config.parallel_workers == 64

        # Invalid: 0 workers
        with pytest.raises(ValidationError):
            TrainingConfig(model_type="lgbm", areas=["SP"], parallel_workers=0)

        # Invalid: too many workers
        with pytest.raises(ValidationError):
            TrainingConfig(model_type="lgbm", areas=["SP"], parallel_workers=65)

    def test_checkpoint_frequency_bounds(self):
        """Test checkpoint_frequency validation."""
        config = TrainingConfig(
            model_type="lgbm", areas=["SP"], enable_checkpointing=True, checkpoint_frequency=5
        )
        assert config.checkpoint_frequency == 5

        # Invalid: 0 frequency
        with pytest.raises(ValidationError):
            TrainingConfig(model_type="lgbm", areas=["SP"], checkpoint_frequency=0)

    def test_optimization_trials_bounds(self):
        """Test optimization_trials bounds."""
        config = TrainingConfig(
            model_type="lgbm", areas=["SP"], optimize_hyperparameters=True, optimization_trials=100
        )
        assert config.optimization_trials == 100

        # Invalid: 0 trials
        with pytest.raises(ValidationError):
            TrainingConfig(model_type="lgbm", areas=["SP"], optimization_trials=0)

    def test_cv_folds_bounds(self):
        """Test cv_folds validation."""
        config = TrainingConfig(model_type="lgbm", areas=["SP"], cv_folds=5)
        assert config.cv_folds == 5

        # Invalid: 1 fold
        with pytest.raises(ValidationError):
            TrainingConfig(model_type="lgbm", areas=["SP"], cv_folds=1)

        # Invalid: too many folds
        with pytest.raises(ValidationError):
            TrainingConfig(model_type="lgbm", areas=["SP"], cv_folds=11)

    def test_memory_limit_validation(self):
        """Test memory_limit_gb validation."""
        config = TrainingConfig(model_type="lgbm", areas=["SP"], memory_limit_gb=8.0)
        assert config.memory_limit_gb == 8.0

        config_none = TrainingConfig(model_type="lgbm", areas=["SP"], memory_limit_gb=None)
        assert config_none.memory_limit_gb is None

        # Invalid: negative memory
        with pytest.raises(ValidationError):
            TrainingConfig(model_type="lgbm", areas=["SP"], memory_limit_gb=-1.0)

    def test_backend_validation(self):
        """Test backend parameter validation."""
        config_mp = TrainingConfig(model_type="lgbm", areas=["SP"], backend="multiprocessing")
        assert config_mp.backend == "multiprocessing"

        config_thread = TrainingConfig(model_type="lgbm", areas=["SP"], backend="threading")
        assert config_thread.backend == "threading"

        # Invalid backend
        with pytest.raises(ValidationError):
            TrainingConfig(
                model_type="lgbm", areas=["SP"], backend="invalid"  # type: ignore[arg-type]
            )

    def test_subsystem_areas(self):
        """Test that subsystem area codes are valid."""
        config = TrainingConfig(model_type="lgbm", areas=["SECO", "S", "NE", "N"])
        assert config.areas == ["SECO", "S", "NE", "N"]

    def test_national_area(self):
        """Test that SIN (national) area code is valid."""
        config = TrainingConfig(model_type="lgbm", areas=["SIN"])
        assert config.areas == ["SIN"]

    def test_training_params_default(self):
        """Test that training_params defaults to empty dict."""
        config = TrainingConfig(model_type="lgbm", areas=["SP"])
        assert config.training_params == {}

    def test_training_params_custom(self):
        """Test custom training_params."""
        custom_params = {"n_estimators": 500, "learning_rate": 0.1, "max_depth": 10}

        config = TrainingConfig(model_type="lgbm", areas=["SP"], training_params=custom_params)

        assert config.training_params == custom_params

    def test_config_immutability(self):
        """Test that config can be modified (frozen=False)."""
        config = TrainingConfig(model_type="lgbm", areas=["SP"])

        # Should be able to modify
        config.parallel_workers = 8
        assert config.parallel_workers == 8

    def test_config_extra_fields_forbidden(self):
        """Test that extra fields raise error."""
        with pytest.raises(ValidationError) as exc_info:
            TrainingConfig(
                model_type="lgbm", areas=["SP"], extra_field="not_allowed"  # type: ignore[call-arg]
            )

        assert "Extra inputs are not permitted" in str(exc_info.value)
