"""Tests for ConfigManager and workflow configurations.

Tests cover:
- Configuration loading from YAML
- Environment variable overrides
- Configuration validation
- Hot-reloading
- Thread-safe access
- Workflow-specific configurations
"""

import os
import tempfile
import threading
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from src.orchestration.config_manager import (
    BacktestingWorkflowConfig,
    ConfigManager,
    ConfigSnapshot,
    Environment,
    FeatureConfig,
    LogLevel,
    ParallelBackend,
    PredictionWorkflowConfig,
    PrevCargaConfig,
    ReconciliationConfig,
    StorageBackendType,
    StorageConfig,
    SystemConfig,
    TrainingWorkflowConfig,
    WorkflowConfig,
)


class TestEnums:
    """Tests for configuration enums."""

    def test_environment_values(self):
        """Test Environment enum values."""
        assert Environment.DEVELOPMENT.value == "development"
        assert Environment.STAGING.value == "staging"
        assert Environment.PRODUCTION.value == "production"

    def test_log_level_values(self):
        """Test LogLevel enum values."""
        assert LogLevel.DEBUG.value == "DEBUG"
        assert LogLevel.INFO.value == "INFO"
        assert LogLevel.WARNING.value == "WARNING"
        assert LogLevel.ERROR.value == "ERROR"
        assert LogLevel.CRITICAL.value == "CRITICAL"

    def test_storage_backend_type_values(self):
        """Test StorageBackendType enum values."""
        assert StorageBackendType.LOCAL.value == "local"
        assert StorageBackendType.S3.value == "s3"

    def test_parallel_backend_values(self):
        """Test ParallelBackend enum values."""
        assert ParallelBackend.MULTIPROCESSING.value == "multiprocessing"
        assert ParallelBackend.THREADING.value == "threading"
        assert ParallelBackend.SEQUENTIAL.value == "sequential"


class TestStorageConfig:
    """Tests for StorageConfig."""

    def test_default_storage_config(self):
        """Test default storage configuration."""
        config = StorageConfig()

        assert config.backend == StorageBackendType.LOCAL
        assert config.base_path == "data"
        assert config.s3_bucket is None
        assert config.s3_region == "sa-east-1"

    def test_local_storage_config(self):
        """Test local storage configuration."""
        config = StorageConfig(
            backend=StorageBackendType.LOCAL,
            base_path="/data/prevcarga",
        )

        assert config.backend == StorageBackendType.LOCAL
        assert config.base_path == "/data/prevcarga"

    def test_s3_storage_config(self):
        """Test S3 storage configuration."""
        config = StorageConfig(
            backend=StorageBackendType.S3,
            s3_bucket="prevcarga-bucket",
            s3_region="us-east-1",
        )

        assert config.backend == StorageBackendType.S3
        assert config.s3_bucket == "prevcarga-bucket"
        assert config.s3_region == "us-east-1"

    def test_s3_config_requires_bucket(self):
        """Test S3 configuration requires bucket name."""
        with pytest.raises(ValueError, match="s3_bucket is required"):
            StorageConfig(backend=StorageBackendType.S3)


class TestSystemConfig:
    """Tests for SystemConfig."""

    def test_default_system_config(self):
        """Test default system configuration."""
        config = SystemConfig()

        assert config.environment == Environment.DEVELOPMENT
        assert config.log_level == LogLevel.INFO
        assert config.log_format == "json"
        assert config.timezone == "America/Sao_Paulo"
        assert config.cache_enabled is True
        assert config.cache_ttl_seconds == 3600

    def test_custom_system_config(self):
        """Test custom system configuration."""
        config = SystemConfig(
            environment=Environment.PRODUCTION,
            log_level=LogLevel.WARNING,
            log_format="text",
            cache_ttl_seconds=7200,
        )

        assert config.environment == Environment.PRODUCTION
        assert config.log_level == LogLevel.WARNING
        assert config.log_format == "text"
        assert config.cache_ttl_seconds == 7200


class TestWorkflowConfig:
    """Tests for WorkflowConfig base class."""

    def test_default_workflow_config(self):
        """Test default workflow configuration."""
        config = WorkflowConfig()

        assert config.enabled is True
        assert config.parallel_workers == 4
        assert config.parallel_backend == ParallelBackend.MULTIPROCESSING
        assert config.timeout_seconds == 3600
        assert config.retry_attempts == 3
        assert config.checkpoint_enabled is True

    def test_custom_workflow_config(self):
        """Test custom workflow configuration."""
        config = WorkflowConfig(
            enabled=False,
            parallel_workers=8,
            parallel_backend=ParallelBackend.THREADING,
            timeout_seconds=7200,
        )

        assert config.enabled is False
        assert config.parallel_workers == 8
        assert config.parallel_backend == ParallelBackend.THREADING
        assert config.timeout_seconds == 7200


class TestTrainingWorkflowConfig:
    """Tests for TrainingWorkflowConfig."""

    def test_default_training_config(self):
        """Test default training configuration."""
        config = TrainingWorkflowConfig()

        assert config.model_types == ["lgbm", "random_forest"]
        assert config.areas == ["SECO", "S", "NE", "N"]
        assert config.horizons == [0, 1, 2, 3, 4, 5, 6, 7, 8]
        assert config.optimize_hyperparameters is False
        assert config.cv_folds == 5
        assert config.save_models is True

    def test_horizon_validation(self):
        """Test horizon validation."""
        config = TrainingWorkflowConfig(horizons=[0, 3, 6])
        assert config.horizons == [0, 3, 6]

        with pytest.raises(ValueError, match="Invalid horizons"):
            TrainingWorkflowConfig(horizons=[0, 10])

    def test_area_validation(self):
        """Test area validation."""
        config = TrainingWorkflowConfig(areas=["SP", "RJ", "MG"])
        assert "SP" in config.areas
        assert "RJ" in config.areas
        assert "MG" in config.areas

        with pytest.raises(ValueError, match="Invalid areas"):
            TrainingWorkflowConfig(areas=["INVALID_AREA"])

    def test_horizon_deduplication(self):
        """Test horizon list is deduplicated and sorted."""
        config = TrainingWorkflowConfig(horizons=[5, 3, 5, 1, 3])
        assert config.horizons == [1, 3, 5]


class TestPredictionWorkflowConfig:
    """Tests for PredictionWorkflowConfig."""

    def test_default_prediction_config(self):
        """Test default prediction configuration."""
        config = PredictionWorkflowConfig()

        assert config.model_dir == "models/trained"
        assert config.output_format == "parquet"
        assert config.include_confidence_intervals is True
        assert config.confidence_level == 0.95
        assert config.apply_reconciliation is True
        assert config.reconciliation_method == "mint"

    def test_custom_prediction_config(self):
        """Test custom prediction configuration."""
        config = PredictionWorkflowConfig(
            model_dir="custom/models",
            output_format="csv",
            confidence_level=0.9,
        )

        assert config.model_dir == "custom/models"
        assert config.output_format == "csv"
        assert config.confidence_level == 0.9

    def test_horizon_validation(self):
        """Test horizon validation in prediction config."""
        with pytest.raises(ValueError, match="Invalid horizons"):
            PredictionWorkflowConfig(horizons=[-1, 0, 1])


class TestBacktestingWorkflowConfig:
    """Tests for BacktestingWorkflowConfig."""

    def test_default_backtesting_config(self):
        """Test default backtesting configuration."""
        config = BacktestingWorkflowConfig()

        assert config.start_date is None
        assert config.end_date is None
        assert config.step_days == 7
        assert config.metrics == ["mape", "mae", "rmse", "r2"]
        assert config.generate_report is True
        assert config.save_forecasts is False

    def test_metric_validation(self):
        """Test metric validation."""
        config = BacktestingWorkflowConfig(metrics=["mape", "mae"])
        assert "mape" in config.metrics

        with pytest.raises(ValueError, match="Invalid metrics"):
            BacktestingWorkflowConfig(metrics=["invalid_metric"])


class TestFeatureConfig:
    """Tests for FeatureConfig."""

    def test_default_feature_config(self):
        """Test default feature configuration."""
        config = FeatureConfig()

        assert "temporal" in config.enabled_plugins
        assert "calendar" in config.enabled_plugins
        assert config.temporal_features is True
        assert config.lag_values == [24, 48, 168, 336]
        assert config.target_column == "load_mw"


class TestReconciliationConfig:
    """Tests for ReconciliationConfig."""

    def test_default_reconciliation_config(self):
        """Test default reconciliation configuration."""
        config = ReconciliationConfig()

        assert config.enabled is True
        assert config.method == "mint"
        assert config.enforce_non_negativity is True


class TestPrevCargaConfig:
    """Tests for complete PrevCargaConfig."""

    def test_default_config(self):
        """Test default complete configuration."""
        config = PrevCargaConfig()

        assert isinstance(config.system, SystemConfig)
        assert isinstance(config.training, TrainingWorkflowConfig)
        assert isinstance(config.prediction, PredictionWorkflowConfig)
        assert isinstance(config.backtesting, BacktestingWorkflowConfig)
        assert isinstance(config.features, FeatureConfig)
        assert isinstance(config.reconciliation, ReconciliationConfig)

    def test_nested_config(self):
        """Test nested configuration."""
        config = PrevCargaConfig(
            system=SystemConfig(environment=Environment.PRODUCTION),
            training=TrainingWorkflowConfig(parallel_workers=16),
        )

        assert config.system.environment == Environment.PRODUCTION
        assert config.training.parallel_workers == 16


class TestConfigManager:
    """Tests for ConfigManager."""

    def test_default_manager(self):
        """Test default configuration manager."""
        manager = ConfigManager.default()

        assert manager is not None
        config = manager.get_config()
        assert isinstance(config, PrevCargaConfig)

    def test_from_dict(self):
        """Test creating manager from dictionary."""
        config_dict = {
            "system": {"environment": "production"},
            "training": {"parallel_workers": 8},
        }

        manager = ConfigManager.from_dict(config_dict)
        assert manager.get_system_config().environment == Environment.PRODUCTION
        assert manager.get_training_config().parallel_workers == 8

    def test_from_yaml(self):
        """Test loading configuration from YAML."""
        config_dict = {
            "system": {
                "environment": "staging",
                "log_level": "DEBUG",
            },
            "training": {
                "parallel_workers": 4,
                "model_types": ["lgbm"],
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_dict, f)
            filepath = f.name

        try:
            manager = ConfigManager.from_yaml(filepath)
            assert manager.get_system_config().environment == Environment.STAGING
            assert manager.get_system_config().log_level == LogLevel.DEBUG
            assert manager.get_training_config().parallel_workers == 4
        finally:
            os.unlink(filepath)

    def test_from_yaml_file_not_found(self):
        """Test loading from non-existent file."""
        with pytest.raises(FileNotFoundError):
            ConfigManager.from_yaml("/nonexistent/config.yaml")

    def test_get_configs(self):
        """Test getting individual configurations."""
        manager = ConfigManager.default()

        system = manager.get_system_config()
        training = manager.get_training_config()
        prediction = manager.get_prediction_config()
        backtesting = manager.get_backtesting_config()
        features = manager.get_feature_config()
        reconciliation = manager.get_reconciliation_config()

        assert isinstance(system, SystemConfig)
        assert isinstance(training, TrainingWorkflowConfig)
        assert isinstance(prediction, PredictionWorkflowConfig)
        assert isinstance(backtesting, BacktestingWorkflowConfig)
        assert isinstance(features, FeatureConfig)
        assert isinstance(reconciliation, ReconciliationConfig)

    def test_update_training_config(self):
        """Test updating training configuration."""
        manager = ConfigManager.default()

        manager.update_training_config(parallel_workers=16, cv_folds=10)

        config = manager.get_training_config()
        assert config.parallel_workers == 16
        assert config.cv_folds == 10

    def test_update_training_config_invalid_key(self):
        """Test updating with invalid key."""
        manager = ConfigManager.default()

        with pytest.raises(ValueError, match="Invalid training config key"):
            manager.update_training_config(invalid_key="value")

    def test_update_prediction_config(self):
        """Test updating prediction configuration."""
        manager = ConfigManager.default()

        manager.update_prediction_config(output_format="csv")

        config = manager.get_prediction_config()
        assert config.output_format == "csv"

    def test_update_backtesting_config(self):
        """Test updating backtesting configuration."""
        manager = ConfigManager.default()

        manager.update_backtesting_config(step_days=14)

        config = manager.get_backtesting_config()
        assert config.step_days == 14

    def test_save_to_yaml(self):
        """Test saving configuration to YAML."""
        manager = ConfigManager.default()
        manager.update_training_config(parallel_workers=8)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            filepath = f.name

        try:
            manager.save_to_yaml(filepath)

            with open(filepath) as f:
                saved = yaml.safe_load(f)

            assert saved["training"]["parallel_workers"] == 8
        finally:
            os.unlink(filepath)

    def test_reload(self):
        """Test configuration reload."""
        config_dict = {"training": {"parallel_workers": 4}}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_dict, f)
            filepath = f.name

        try:
            manager = ConfigManager.from_yaml(filepath)
            assert manager.get_training_config().parallel_workers == 4

            # Modify the file
            config_dict["training"]["parallel_workers"] = 16
            with open(filepath, "w") as f:
                yaml.dump(config_dict, f)

            # Reload
            changed = manager.reload()
            assert changed is True
            assert manager.get_training_config().parallel_workers == 16

            # Reload again without changes
            changed = manager.reload()
            assert changed is False
        finally:
            os.unlink(filepath)

    def test_reload_without_source_file(self):
        """Test reload fails without source file."""
        manager = ConfigManager.default()

        with pytest.raises(RuntimeError, match="Cannot reload"):
            manager.reload()

    def test_get_snapshot(self):
        """Test getting configuration snapshot."""
        manager = ConfigManager.default()
        snapshot = manager.get_snapshot()

        assert isinstance(snapshot, ConfigSnapshot)
        assert isinstance(snapshot.config, PrevCargaConfig)
        assert isinstance(snapshot.loaded_at, datetime)
        assert snapshot.checksum != ""

    def test_snapshot_to_dict(self):
        """Test snapshot to dictionary conversion."""
        manager = ConfigManager.default()
        snapshot = manager.get_snapshot()
        result = snapshot.to_dict()

        assert "config" in result
        assert "loaded_at" in result
        assert "checksum" in result

    def test_validate(self):
        """Test configuration validation."""
        manager = ConfigManager.default()
        warnings = manager.validate()

        # Default config should be valid with minimal warnings
        assert isinstance(warnings, list)

    def test_validate_high_parallel_workers(self):
        """Test validation warns on high parallel workers."""
        manager = ConfigManager.default()
        manager.update_training_config(parallel_workers=32)

        warnings = manager.validate()
        assert any("parallel_workers" in w for w in warnings)

    def test_to_dict(self):
        """Test conversion to dictionary."""
        manager = ConfigManager.default()
        result = manager.to_dict()

        assert "system" in result
        assert "training" in result
        assert "prediction" in result
        assert "backtesting" in result

    def test_repr(self):
        """Test string representation."""
        manager = ConfigManager.default()
        repr_str = repr(manager)

        assert "ConfigManager" in repr_str
        assert "development" in repr_str

    def test_register_change_callback(self):
        """Test registering change callback."""
        config_dict = {"training": {"parallel_workers": 4}}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_dict, f)
            filepath = f.name

        try:
            manager = ConfigManager.from_yaml(filepath)

            callback_called = []

            def callback(config):
                callback_called.append(config)

            manager.register_change_callback(callback)

            # Modify and reload
            config_dict["training"]["parallel_workers"] = 8
            with open(filepath, "w") as f:
                yaml.dump(config_dict, f)

            manager.reload()
            assert len(callback_called) == 1
        finally:
            os.unlink(filepath)


class TestEnvironmentOverrides:
    """Tests for environment variable overrides."""

    def test_env_override_training_workers(self):
        """Test environment override for training workers."""
        config_dict = {"training": {"parallel_workers": 4}}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_dict, f)
            filepath = f.name

        try:
            with patch.dict(os.environ, {"PREVCARGA_TRAINING_PARALLEL_WORKERS": "16"}):
                manager = ConfigManager.from_yaml_with_env_overrides(filepath)
                assert manager.get_training_config().parallel_workers == 16
        finally:
            os.unlink(filepath)

    def test_env_override_system_log_level(self):
        """Test environment override for log level."""
        config_dict = {"system": {"log_level": "INFO"}}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_dict, f)
            filepath = f.name

        try:
            with patch.dict(os.environ, {"PREVCARGA_SYSTEM_LOG_LEVEL": "DEBUG"}):
                manager = ConfigManager.from_yaml_with_env_overrides(filepath)
                assert manager.get_system_config().log_level == LogLevel.DEBUG
        finally:
            os.unlink(filepath)

    def test_env_override_boolean(self):
        """Test environment override for boolean values."""
        config_dict = {"training": {"optimize_hyperparameters": False}}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_dict, f)
            filepath = f.name

        try:
            with patch.dict(os.environ, {"PREVCARGA_TRAINING_OPTIMIZE_HYPERPARAMETERS": "true"}):
                manager = ConfigManager.from_yaml_with_env_overrides(filepath)
                assert manager.get_training_config().optimize_hyperparameters is True
        finally:
            os.unlink(filepath)

    def test_env_override_prediction_format(self):
        """Test environment override for prediction output format."""
        config_dict = {"prediction": {"output_format": "parquet"}}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_dict, f)
            filepath = f.name

        try:
            with patch.dict(os.environ, {"PREVCARGA_PREDICTION_OUTPUT_FORMAT": "csv"}):
                manager = ConfigManager.from_yaml_with_env_overrides(filepath)
                assert manager.get_prediction_config().output_format == "csv"
        finally:
            os.unlink(filepath)


class TestThreadSafety:
    """Tests for thread-safe access."""

    def test_concurrent_reads(self):
        """Test concurrent configuration reads."""
        manager = ConfigManager.default()
        results = []
        errors = []

        def read_config():
            try:
                for _ in range(100):
                    config = manager.get_training_config()
                    results.append(config.parallel_workers)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=read_config) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 1000

    def test_concurrent_updates(self):
        """Test concurrent configuration updates."""
        manager = ConfigManager.default()
        errors = []

        def update_config(value):
            try:
                for _ in range(50):
                    manager.update_training_config(parallel_workers=value)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=update_config, args=(i + 1,))
            for i in range(10)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        # Final value should be one of 1-10
        assert 1 <= manager.get_training_config().parallel_workers <= 10


class TestIntegration:
    """Integration tests."""

    def test_complete_workflow_configuration(self):
        """Test complete workflow configuration scenario."""
        config_dict = {
            "system": {
                "environment": "production",
                "log_level": "WARNING",
                "storage": {
                    "backend": "s3",
                    "s3_bucket": "prevcarga-prod",
                },
            },
            "training": {
                "parallel_workers": 8,
                "model_types": ["lgbm"],
                "areas": ["SECO", "S", "NE", "N"],
                "horizons": [0, 1, 2, 3],
                "optimize_hyperparameters": True,
                "optimization_trials": 100,
            },
            "prediction": {
                "apply_reconciliation": True,
                "reconciliation_method": "mint",
            },
            "reconciliation": {
                "enabled": True,
                "method": "mint",
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_dict, f)
            filepath = f.name

        try:
            manager = ConfigManager.from_yaml(filepath)

            # Verify system config
            system = manager.get_system_config()
            assert system.environment == Environment.PRODUCTION
            assert system.storage.backend == StorageBackendType.S3
            assert system.storage.s3_bucket == "prevcarga-prod"

            # Verify training config
            training = manager.get_training_config()
            assert training.parallel_workers == 8
            assert training.optimize_hyperparameters is True

            # Verify prediction config
            prediction = manager.get_prediction_config()
            assert prediction.apply_reconciliation is True

            # Verify reconciliation config
            reconciliation = manager.get_reconciliation_config()
            assert reconciliation.method == "mint"

            # Test validation
            warnings = manager.validate()
            assert isinstance(warnings, list)

            # Test snapshot
            snapshot = manager.get_snapshot()
            assert snapshot.source_file == filepath

        finally:
            os.unlink(filepath)

    def test_brazilian_load_forecasting_scenario(self):
        """Test configuration for Brazilian load forecasting."""
        config = PrevCargaConfig(
            system=SystemConfig(
                timezone="America/Sao_Paulo",
            ),
            training=TrainingWorkflowConfig(
                areas=["SECO", "S", "NE", "N", "SIN"],
                horizons=[0, 1, 2, 3, 4, 5, 6, 7, 8],
                model_types=["lgbm", "random_forest"],
            ),
            reconciliation=ReconciliationConfig(
                enabled=True,
                method="mint",
                hierarchy_file="config/hierarchy_brazil.yaml",
            ),
        )

        manager = ConfigManager(config)

        # Verify Brazilian-specific configuration
        training = manager.get_training_config()
        assert "SIN" in training.areas
        assert len(training.horizons) == 9  # D+0 to D+8

        reconciliation = manager.get_reconciliation_config()
        assert reconciliation.enabled is True
        assert "brazil" in reconciliation.hierarchy_file
