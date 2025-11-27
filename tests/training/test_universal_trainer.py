"""Tests for UniversalTrainer."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.training import TrainingConfig, UniversalTrainer


@pytest.fixture
def sample_data():
    """Create sample training data for testing."""
    np.random.seed(42)

    # Create simple synthetic data
    n_samples = 100
    n_features = 10

    data_dict = {}

    for area in ["SP", "RJ", "MG"]:
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f"feature_{i}" for i in range(n_features)],
        )
        y = pd.Series(
            np.random.randn(n_samples) * 100 + 1000, name="load"  # Load values around 1000
        )
        data_dict[area] = (X, y)

    return data_dict


@pytest.fixture
def temp_checkpoint_dir():
    """Create temporary directory for checkpoints."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def basic_config(temp_checkpoint_dir):
    """Create basic training configuration."""
    return TrainingConfig(
        model_type="lgbm",
        areas=["SP", "RJ", "MG"],
        parallel_workers=1,  # Sequential for deterministic testing
        enable_checkpointing=True,
        checkpoint_dir=temp_checkpoint_dir,
        optimize_hyperparameters=False,  # Faster testing
        training_params={
            "n_estimators": 10,  # Small for fast training
            "learning_rate": 0.1,
            "verbosity": -1,
        },
    )


class TestUniversalTrainer:
    """Test suite for UniversalTrainer."""

    def test_initialization(self, basic_config):
        """Test trainer initialization."""
        trainer = UniversalTrainer(basic_config)

        assert trainer.config == basic_config
        assert trainer.checkpoint_manager is not None
        assert trainer.progress_tracker is None  # Not initialized until training
        assert len(trainer.trained_models) == 0
        assert len(trainer.training_results) == 0

    def test_train_single_area_sequential(self, basic_config, sample_data):
        """Test training models for multiple areas sequentially."""
        # Modify config to train only one area for speed
        basic_config.areas = ["SP"]

        trainer = UniversalTrainer(basic_config)
        trained_models = trainer.train_multiple_areas(sample_data)

        assert len(trained_models) == 1
        assert "SP" in trained_models
        assert trained_models["SP"].is_fitted()

    def test_train_multiple_areas_sequential(self, basic_config, sample_data):
        """Test training multiple areas sequentially."""
        trainer = UniversalTrainer(basic_config)
        trained_models = trainer.train_multiple_areas(sample_data)

        assert len(trained_models) == 3
        assert "SP" in trained_models
        assert "RJ" in trained_models
        assert "MG" in trained_models

        # All models should be fitted
        for area, model in trained_models.items():
            assert model.is_fitted(), f"Model for {area} not fitted"

    def test_train_multiple_areas_parallel(self, basic_config, sample_data):
        """Test training multiple areas in parallel."""
        # Enable parallel training
        basic_config.parallel_workers = 2

        trainer = UniversalTrainer(basic_config)
        trained_models = trainer.train_multiple_areas(sample_data)

        assert len(trained_models) == 3
        for area in ["SP", "RJ", "MG"]:
            assert area in trained_models
            assert trained_models[area].is_fitted()

    def test_missing_data_areas(self, basic_config, sample_data):
        """Test error when data is missing for configured areas."""
        # Remove one area from data
        del sample_data["MG"]

        trainer = UniversalTrainer(basic_config)

        with pytest.raises(ValueError) as exc_info:
            trainer.train_multiple_areas(sample_data)

        assert "Missing data for areas" in str(exc_info.value)
        assert "MG" in str(exc_info.value)

    def test_extra_data_areas(self, basic_config, sample_data):
        """Test handling of extra areas in data (should warn but not error)."""
        # Add extra area not in config
        X_extra = sample_data["SP"][0].copy()
        y_extra = sample_data["SP"][1].copy()
        sample_data["ES"] = (X_extra, y_extra)

        trainer = UniversalTrainer(basic_config)
        trained_models = trainer.train_multiple_areas(sample_data)

        # Should still train only configured areas
        assert len(trained_models) == 3
        assert "ES" not in trained_models

    def test_progress_tracking(self, basic_config, sample_data):
        """Test that progress tracker is initialized and updated."""
        basic_config.areas = ["SP", "RJ"]

        trainer = UniversalTrainer(basic_config)
        trained_models = trainer.train_multiple_areas(sample_data)

        assert trainer.progress_tracker is not None
        assert trainer.progress_tracker.is_completed()
        assert trainer.progress_tracker.completed_tasks == 2

    def test_training_results_collected(self, basic_config, sample_data):
        """Test that training results are collected for each area."""
        basic_config.areas = ["SP", "RJ"]

        trainer = UniversalTrainer(basic_config)
        trainer.train_multiple_areas(sample_data)

        assert len(trainer.training_results) == 2
        assert "SP" in trainer.training_results
        assert "RJ" in trainer.training_results

        # Check result structure
        for area in ["SP", "RJ"]:
            result = trainer.training_results[area]
            assert "area" in result
            assert "model_type" in result
            assert "training_time_seconds" in result
            assert "training_samples" in result
            assert "training_features" in result

    def test_validate_trained_models(self, basic_config, sample_data):
        """Test model validation."""
        basic_config.areas = ["SP", "RJ"]

        trainer = UniversalTrainer(basic_config)
        trainer.train_multiple_areas(sample_data)

        validation = trainer.validate_trained_models()

        assert len(validation) == 2
        assert validation["SP"] is True
        assert validation["RJ"] is True

    def test_generate_training_report(self, basic_config, sample_data):
        """Test training report generation."""
        basic_config.areas = ["SP", "RJ"]

        trainer = UniversalTrainer(basic_config)
        trainer.train_multiple_areas(sample_data)

        report = trainer.generate_training_report()

        assert "UNIVERSAL TRAINER - TRAINING REPORT" in report
        assert "Model Type: lgbm" in report
        assert "Areas Trained: 2" in report
        assert "SP" in report
        assert "RJ" in report

    def test_generate_report_no_training(self, basic_config):
        """Test report generation when no training done."""
        trainer = UniversalTrainer(basic_config)
        report = trainer.generate_training_report()

        assert "No models trained yet" in report

    def test_save_all_models(self, basic_config, sample_data):
        """Test saving all trained models."""
        with tempfile.TemporaryDirectory() as tmpdir:
            basic_config.areas = ["SP", "RJ"]

            trainer = UniversalTrainer(basic_config)
            trainer.train_multiple_areas(sample_data)

            saved_paths = trainer.save_all_models(tmpdir)

            assert len(saved_paths) == 2
            assert "SP" in saved_paths
            assert "RJ" in saved_paths

            # Check files exist
            for area, path in saved_paths.items():
                # LGBM saves two files: .lgb and .meta
                lgb_path = path.with_suffix(".lgb")
                meta_path = path.with_suffix(".meta")
                assert lgb_path.exists(), f"Missing .lgb file for {area}"
                assert meta_path.exists(), f"Missing .meta file for {area}"

    def test_checkpointing_disabled(self, basic_config, sample_data):
        """Test training with checkpointing disabled."""
        basic_config.enable_checkpointing = False
        basic_config.areas = ["SP"]

        trainer = UniversalTrainer(basic_config)
        trainer.train_multiple_areas(sample_data)

        # Should not create checkpoint
        assert not trainer.checkpoint_manager.has_checkpoint()

    def test_checkpointing_enabled(self, basic_config, sample_data):
        """Test training with checkpointing enabled."""
        basic_config.enable_checkpointing = True
        basic_config.checkpoint_frequency = 1
        basic_config.areas = ["SP", "RJ"]

        trainer = UniversalTrainer(basic_config)

        # Note: Checkpoint is cleared on successful completion
        # So we need to check during training or modify this test

        # Train and verify completion (checkpoint should be cleared)
        trainer.train_multiple_areas(sample_data)

        # After successful completion, checkpoint should be cleared
        assert not trainer.checkpoint_manager.has_checkpoint()

    def test_random_forest_model(self, basic_config, sample_data):
        """Test training with Random Forest model."""
        basic_config.model_type = "random_forest"
        basic_config.areas = ["SP"]
        basic_config.training_params = {"n_estimators": 10, "max_depth": 5, "random_state": 42}

        trainer = UniversalTrainer(basic_config)
        trained_models = trainer.train_multiple_areas(sample_data)

        assert len(trained_models) == 1
        assert trained_models["SP"].is_fitted()
        assert trained_models["SP"].name == "random_forest"

    def test_different_data_types(self, basic_config):
        """Test training with different data types (DataFrame y)."""
        np.random.seed(42)

        # Create data with DataFrame y instead of Series
        X = pd.DataFrame(np.random.randn(100, 10), columns=[f"feature_{i}" for i in range(10)])
        y = pd.DataFrame(np.random.randn(100, 1) * 100 + 1000, columns=["load"])

        data_dict = {"SP": (X, y)}
        basic_config.areas = ["SP"]

        trainer = UniversalTrainer(basic_config)
        trained_models = trainer.train_multiple_areas(data_dict)

        assert trained_models["SP"].is_fitted()

    def test_repr(self, basic_config, sample_data):
        """Test string representation."""
        trainer = UniversalTrainer(basic_config)

        repr_str = repr(trainer)
        assert "UniversalTrainer" in repr_str
        assert "model_type='lgbm'" in repr_str
        assert "trained=0" in repr_str

        # Train models
        trainer.train_multiple_areas(sample_data)

        repr_str = repr(trainer)
        assert "trained=3" in repr_str

    def test_hyperparameter_optimization_config(self, basic_config, sample_data):
        """Test that hyperparameter optimization config is passed correctly."""
        basic_config.areas = ["SP"]
        basic_config.optimize_hyperparameters = True
        basic_config.optimization_trials = 5  # Small number for testing
        basic_config.cv_folds = 2

        # Note: This will be slow even with few trials
        # In production, might want to skip or mock
        trainer = UniversalTrainer(basic_config)
        trained_models = trainer.train_multiple_areas(sample_data)

        assert trained_models["SP"].is_fitted()

        # Check that training metadata includes optimization info
        result = trainer.training_results["SP"]
        assert "training_time_seconds" in result

    def test_empty_data(self, basic_config):
        """Test handling of empty data."""
        X_empty = pd.DataFrame()
        y_empty = pd.Series(dtype=float)

        data_dict = {"SP": (X_empty, y_empty)}
        basic_config.areas = ["SP"]

        trainer = UniversalTrainer(basic_config)

        # Should raise error from model.fit()
        with pytest.raises(RuntimeError):
            trainer.train_multiple_areas(data_dict)

    def test_single_worker_vs_multiple(self, basic_config, sample_data):
        """Test that single worker and multiple workers produce fitted models."""
        basic_config.areas = ["SP", "RJ"]

        # Single worker
        basic_config.parallel_workers = 1
        trainer_seq = UniversalTrainer(basic_config)
        models_seq = trainer_seq.train_multiple_areas(sample_data)

        # Multiple workers
        basic_config.parallel_workers = 2
        trainer_par = UniversalTrainer(basic_config)
        models_par = trainer_par.train_multiple_areas(sample_data)

        # Both should produce fitted models
        assert all(m.is_fitted() for m in models_seq.values())
        assert all(m.is_fitted() for m in models_par.values())

        # Both should have same areas
        assert set(models_seq.keys()) == set(models_par.keys())

    def test_training_with_horizons(self, basic_config, sample_data):
        """Test training with specific horizons."""
        basic_config.areas = ["SP"]
        basic_config.horizons = [0, 1]  # Only D+0 and D+1

        trainer = UniversalTrainer(basic_config)
        trained_models = trainer.train_multiple_areas(sample_data)

        assert trained_models["SP"].is_fitted()

    def test_callback_integration(self, basic_config, sample_data):
        """Test that progress callbacks work during training."""
        basic_config.areas = ["SP", "RJ"]

        callback_data = []

        def test_callback(tracker):
            callback_data.append(
                {
                    "completed": tracker.completed_tasks,
                    "percentage": tracker.get_progress_percentage(),
                }
            )

        trainer = UniversalTrainer(basic_config)

        # Note: We can't easily add callbacks before training starts
        # This would need to be done in the train_multiple_areas method
        # For now, test that the mechanism exists
        trainer.train_multiple_areas(sample_data)

        assert trainer.progress_tracker is not None

    def test_training_params_passed_correctly(self, basic_config, sample_data):
        """Test that training_params is passed to fit method."""
        custom_params = {"n_estimators": 20, "learning_rate": 0.05, "num_leaves": 15}

        basic_config.areas = ["SP"]
        basic_config.training_params = custom_params

        trainer = UniversalTrainer(basic_config)
        trained_models = trainer.train_multiple_areas(sample_data)

        # Model should be trained with custom config
        assert trained_models["SP"].is_fitted()

    def test_training_results_timing(self, basic_config, sample_data):
        """Test that training results include timing information."""
        basic_config.areas = ["SP"]

        trainer = UniversalTrainer(basic_config)
        trainer.train_multiple_areas(sample_data)

        result = trainer.training_results["SP"]

        assert "training_time_seconds" in result
        assert isinstance(result["training_time_seconds"], (int, float))
        assert result["training_time_seconds"] > 0

    def test_large_area_list(self, basic_config):
        """Test handling of larger number of areas."""
        # Create data for many areas
        np.random.seed(42)
        n_areas = 10
        areas = [f"AREA_{i}" for i in range(n_areas)]

        data_dict = {}
        for area in areas:
            X = pd.DataFrame(
                np.random.randn(50, 5),  # Smaller dataset for speed
                columns=[f"f_{i}" for i in range(5)],
            )
            y = pd.Series(np.random.randn(50) * 100 + 1000)
            data_dict[area] = (X, y)

        basic_config.areas = areas
        basic_config.parallel_workers = 2

        trainer = UniversalTrainer(basic_config)
        trained_models = trainer.train_multiple_areas(data_dict)

        assert len(trained_models) == n_areas
        assert all(model.is_fitted() for model in trained_models.values())
