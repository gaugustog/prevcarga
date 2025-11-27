"""Configuration manager for PrevCarga workflow orchestration.

This module provides centralized configuration management for the PrevCarga
system, supporting YAML files, environment variable overrides, validation,
and hot-reloading capabilities.

Key Components:
- ConfigManager: Central configuration management class
- WorkflowConfig: Base configuration for all workflows
- TrainingWorkflowConfig: Configuration for training workflows
- PredictionWorkflowConfig: Configuration for prediction workflows
- BacktestingWorkflowConfig: Configuration for backtesting workflows
- SystemConfig: System-level configuration

Example:
    ```python
    from src.orchestration.config_manager import ConfigManager

    # Initialize from YAML file
    manager = ConfigManager.from_yaml("config/prevcarga.yaml")

    # Get workflow configurations
    training_config = manager.get_training_config()
    prediction_config = manager.get_prediction_config()

    # Environment variable overrides
    # PREVCARGA_TRAINING_PARALLEL_WORKERS=8 python ...
    manager = ConfigManager.from_yaml_with_env_overrides("config/prevcarga.yaml")

    # Hot reload configuration
    manager.reload()
    ```
"""

import os
import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from src.utils.logger import get_logger

logger = get_logger(__name__)


class Environment(Enum):
    """Deployment environment.

    Attributes:
        DEVELOPMENT: Local development environment.
        STAGING: Staging/testing environment.
        PRODUCTION: Production environment.
    """

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(Enum):
    """Log level configuration.

    Attributes:
        DEBUG: Debug level logging.
        INFO: Info level logging.
        WARNING: Warning level logging.
        ERROR: Error level logging.
        CRITICAL: Critical level logging.
    """

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class StorageBackendType(Enum):
    """Storage backend type.

    Attributes:
        LOCAL: Local filesystem storage.
        S3: AWS S3 storage.
    """

    LOCAL = "local"
    S3 = "s3"


class ParallelBackend(Enum):
    """Parallelization backend.

    Attributes:
        MULTIPROCESSING: Process-based parallelization.
        THREADING: Thread-based parallelization.
        SEQUENTIAL: No parallelization.
    """

    MULTIPROCESSING = "multiprocessing"
    THREADING = "threading"
    SEQUENTIAL = "sequential"


class StorageConfig(BaseModel):
    """Storage configuration.

    Attributes:
        backend: Storage backend type (local or s3).
        base_path: Base path for local storage or S3 prefix.
        s3_bucket: S3 bucket name (required if backend is s3).
        s3_region: AWS region for S3.
        s3_endpoint_url: Custom S3 endpoint URL.
    """

    backend: StorageBackendType = StorageBackendType.LOCAL
    base_path: str = Field(default="data", description="Base path for storage")
    s3_bucket: str | None = Field(default=None, description="S3 bucket name")
    s3_region: str = Field(default="sa-east-1", description="AWS region")
    s3_endpoint_url: str | None = Field(default=None, description="Custom S3 endpoint")

    @model_validator(mode="after")
    def validate_s3_config(self) -> "StorageConfig":
        """Validate S3 configuration when S3 backend is selected."""
        if self.backend == StorageBackendType.S3 and not self.s3_bucket:
            msg = "s3_bucket is required when backend is 's3'"
            raise ValueError(msg)
        return self


class SystemConfig(BaseModel):
    """System-level configuration.

    Attributes:
        environment: Deployment environment.
        log_level: Logging level.
        log_format: Log format (json or text).
        timezone: Default timezone.
        storage: Storage configuration.
        temp_dir: Temporary directory for intermediate files.
        cache_enabled: Whether to enable caching.
        cache_ttl_seconds: Cache TTL in seconds.
    """

    environment: Environment = Environment.DEVELOPMENT
    log_level: LogLevel = LogLevel.INFO
    log_format: str = Field(default="json", pattern="^(json|text)$")
    timezone: str = Field(default="America/Sao_Paulo")
    storage: StorageConfig = Field(default_factory=StorageConfig)
    temp_dir: str = Field(default="/tmp/prevcarga")
    cache_enabled: bool = Field(default=True)
    cache_ttl_seconds: int = Field(default=3600, ge=0)


class WorkflowConfig(BaseModel):
    """Base configuration for workflows.

    Attributes:
        enabled: Whether the workflow is enabled.
        parallel_workers: Number of parallel workers.
        parallel_backend: Parallelization backend.
        timeout_seconds: Workflow timeout in seconds.
        retry_attempts: Number of retry attempts.
        retry_delay_seconds: Delay between retries.
        checkpoint_enabled: Whether to enable checkpointing.
        checkpoint_dir: Directory for checkpoints.
    """

    enabled: bool = Field(default=True)
    parallel_workers: int = Field(default=4, ge=1, le=64)
    parallel_backend: ParallelBackend = ParallelBackend.MULTIPROCESSING
    timeout_seconds: int = Field(default=3600, ge=60)
    retry_attempts: int = Field(default=3, ge=0, le=10)
    retry_delay_seconds: int = Field(default=60, ge=0)
    checkpoint_enabled: bool = Field(default=True)
    checkpoint_dir: str = Field(default="checkpoints")


class TrainingWorkflowConfig(WorkflowConfig):
    """Configuration for training workflows.

    Attributes:
        model_types: List of model types to train.
        areas: List of areas to train models for.
        horizons: List of forecast horizons.
        optimize_hyperparameters: Whether to optimize hyperparameters.
        optimization_trials: Number of Optuna trials.
        cv_folds: Number of cross-validation folds.
        early_stopping_rounds: Early stopping rounds for gradient boosting.
        validation_split: Validation split ratio.
        save_models: Whether to save trained models.
        model_output_dir: Directory for saved models.
    """

    model_types: list[str] = Field(
        default=["lgbm", "random_forest"],
        description="Model types to train",
    )
    areas: list[str] = Field(
        default=["SECO", "S", "NE", "N"],
        description="Areas to train models for",
    )
    horizons: list[int] = Field(
        default=[0, 1, 2, 3, 4, 5, 6, 7, 8],
        description="Forecast horizons (D+0 to D+8)",
    )
    optimize_hyperparameters: bool = Field(default=False)
    optimization_trials: int = Field(default=50, ge=1, le=500)
    cv_folds: int = Field(default=5, ge=2, le=10)
    early_stopping_rounds: int = Field(default=50, ge=10)
    validation_split: float = Field(default=0.2, ge=0.05, le=0.4)
    save_models: bool = Field(default=True)
    model_output_dir: str = Field(default="models/trained")

    @field_validator("horizons")
    @classmethod
    def validate_horizons(cls, v: list[int]) -> list[int]:
        """Validate horizon values."""
        invalid = [h for h in v if h < 0 or h > 8]
        if invalid:
            msg = f"Invalid horizons {invalid}. Must be in range [0, 8]"
            raise ValueError(msg)
        return sorted(set(v))

    @field_validator("areas")
    @classmethod
    def validate_areas(cls, v: list[str]) -> list[str]:
        """Validate area codes."""
        valid_areas = {
            "SP", "RJ", "MG", "ES",  # SECO
            "PR", "SC", "RS",  # S
            "BA", "PE", "CE", "ALPE", "PBRN", "BAOE",  # NE
            "PA", "AM", "AP", "AC", "RO", "TON",  # N
            "SECO", "S", "NE", "N",  # Subsystems
            "SIN",  # National
        }
        invalid = set(v) - valid_areas
        if invalid:
            msg = f"Invalid areas: {sorted(invalid)}. Valid: {sorted(valid_areas)}"
            raise ValueError(msg)
        return list(set(v))


class PredictionWorkflowConfig(WorkflowConfig):
    """Configuration for prediction workflows.

    Attributes:
        model_dir: Directory containing trained models.
        horizons: Forecast horizons to predict.
        areas: Areas to generate predictions for.
        output_format: Output format (csv, parquet, json).
        output_dir: Directory for prediction outputs.
        include_confidence_intervals: Include confidence intervals.
        confidence_level: Confidence level for intervals.
        apply_reconciliation: Apply hierarchical reconciliation.
        reconciliation_method: Reconciliation method.
    """

    model_dir: str = Field(default="models/trained")
    horizons: list[int] = Field(default=[0, 1, 2, 3, 4, 5, 6, 7, 8])
    areas: list[str] = Field(default=["SECO", "S", "NE", "N"])
    output_format: str = Field(default="parquet", pattern="^(csv|parquet|json)$")
    output_dir: str = Field(default="predictions")
    include_confidence_intervals: bool = Field(default=True)
    confidence_level: float = Field(default=0.95, ge=0.8, le=0.99)
    apply_reconciliation: bool = Field(default=True)
    reconciliation_method: str = Field(default="mint")

    @field_validator("horizons")
    @classmethod
    def validate_horizons(cls, v: list[int]) -> list[int]:
        """Validate horizon values."""
        invalid = [h for h in v if h < 0 or h > 8]
        if invalid:
            msg = f"Invalid horizons {invalid}. Must be in range [0, 8]"
            raise ValueError(msg)
        return sorted(set(v))


class BacktestingWorkflowConfig(WorkflowConfig):
    """Configuration for backtesting workflows.

    Attributes:
        start_date: Start date for backtesting.
        end_date: End date for backtesting.
        step_days: Number of days between backtests.
        horizons: Forecast horizons to backtest.
        areas: Areas to backtest.
        metrics: Metrics to calculate.
        generate_report: Whether to generate HTML report.
        report_dir: Directory for reports.
        save_forecasts: Whether to save individual forecasts.
        forecasts_dir: Directory for forecast outputs.
    """

    start_date: str | None = Field(default=None, description="Start date (YYYY-MM-DD)")
    end_date: str | None = Field(default=None, description="End date (YYYY-MM-DD)")
    step_days: int = Field(default=7, ge=1, le=30)
    horizons: list[int] = Field(default=[0, 1, 2, 3, 4, 5, 6, 7, 8])
    areas: list[str] = Field(default=["SECO", "S", "NE", "N"])
    metrics: list[str] = Field(
        default=["mape", "mae", "rmse", "r2"],
        description="Metrics to calculate",
    )
    generate_report: bool = Field(default=True)
    report_dir: str = Field(default="reports/backtests")
    save_forecasts: bool = Field(default=False)
    forecasts_dir: str = Field(default="backtests/forecasts")

    @field_validator("metrics")
    @classmethod
    def validate_metrics(cls, v: list[str]) -> list[str]:
        """Validate metric names."""
        valid_metrics = {"mape", "smape", "mae", "rmse", "r2", "mse", "bias"}
        invalid = set(v) - valid_metrics
        if invalid:
            msg = f"Invalid metrics: {sorted(invalid)}. Valid: {sorted(valid_metrics)}"
            raise ValueError(msg)
        return list(set(v))


class FeatureConfig(BaseModel):
    """Feature engineering configuration.

    Attributes:
        enabled_plugins: List of enabled feature plugins.
        temporal_features: Enable temporal features.
        calendar_features: Enable calendar/holiday features.
        lag_features: Enable lag features.
        rolling_features: Enable rolling statistics.
        lag_values: List of lag values.
        rolling_windows: List of rolling window sizes.
        target_column: Target column name.
    """

    enabled_plugins: list[str] = Field(
        default=["temporal", "calendar", "lag", "rolling"],
        description="Enabled feature plugins",
    )
    temporal_features: bool = Field(default=True)
    calendar_features: bool = Field(default=True)
    lag_features: bool = Field(default=True)
    rolling_features: bool = Field(default=True)
    lag_values: list[int] = Field(default=[24, 48, 168, 336])
    rolling_windows: list[int] = Field(default=[24, 48, 168])
    target_column: str = Field(default="load_mw")


class ReconciliationConfig(BaseModel):
    """Hierarchical reconciliation configuration.

    Attributes:
        enabled: Whether reconciliation is enabled.
        method: Reconciliation method.
        hierarchy_file: Path to hierarchy definition file.
        enforce_non_negativity: Enforce non-negative forecasts.
        weight_by_variance: Weight by forecast variance.
    """

    enabled: bool = Field(default=True)
    method: str = Field(default="mint", pattern="^(ols|wls|mint|shrinkage)$")
    hierarchy_file: str = Field(default="config/hierarchy_brazil.yaml")
    enforce_non_negativity: bool = Field(default=True)
    weight_by_variance: bool = Field(default=True)


class PrevCargaConfig(BaseModel):
    """Complete PrevCarga system configuration.

    Attributes:
        system: System-level configuration.
        training: Training workflow configuration.
        prediction: Prediction workflow configuration.
        backtesting: Backtesting workflow configuration.
        features: Feature engineering configuration.
        reconciliation: Reconciliation configuration.
    """

    system: SystemConfig = Field(default_factory=SystemConfig)
    training: TrainingWorkflowConfig = Field(default_factory=TrainingWorkflowConfig)
    prediction: PredictionWorkflowConfig = Field(default_factory=PredictionWorkflowConfig)
    backtesting: BacktestingWorkflowConfig = Field(default_factory=BacktestingWorkflowConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    reconciliation: ReconciliationConfig = Field(default_factory=ReconciliationConfig)

    model_config = {"frozen": False, "extra": "allow"}


@dataclass
class ConfigSnapshot:
    """Immutable snapshot of configuration at a point in time.

    Attributes:
        config: The configuration at snapshot time.
        loaded_at: When the configuration was loaded.
        source_file: Source file path if loaded from file.
        checksum: Configuration checksum for change detection.
    """

    config: PrevCargaConfig
    loaded_at: datetime
    source_file: str | None = None
    checksum: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary representation.
        """
        return {
            "config": self.config.model_dump(),
            "loaded_at": self.loaded_at.isoformat(),
            "source_file": self.source_file,
            "checksum": self.checksum,
        }


class ConfigManager:
    """Central configuration manager for PrevCarga.

    Provides centralized configuration management with support for:
    - Loading from YAML files
    - Environment variable overrides
    - Configuration validation
    - Hot-reloading
    - Thread-safe access

    Example:
        >>> manager = ConfigManager.from_yaml("config/prevcarga.yaml")
        >>> training_config = manager.get_training_config()
        >>> manager.update_training_config(parallel_workers=8)
        >>> manager.reload()
    """

    # Environment variable prefix
    ENV_PREFIX = "PREVCARGA_"

    def __init__(self, config: PrevCargaConfig | None = None) -> None:
        """Initialize configuration manager.

        Args:
            config: Initial configuration. Uses defaults if None.
        """
        self._config = config or PrevCargaConfig()
        self._lock = threading.RLock()
        self._source_file: str | None = None
        self._loaded_at = datetime.now()
        self._checksum = self._compute_checksum()
        self._change_callbacks: list[callable] = []

        logger.debug("Initialized ConfigManager")

    @classmethod
    def from_yaml(cls, filepath: str | Path) -> "ConfigManager":
        """Load configuration from YAML file.

        Args:
            filepath: Path to YAML configuration file.

        Returns:
            ConfigManager instance.

        Raises:
            FileNotFoundError: If file doesn't exist.
            ValueError: If YAML is invalid or config validation fails.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            msg = f"Configuration file not found: {filepath}"
            raise FileNotFoundError(msg)

        with open(filepath, encoding="utf-8") as f:
            raw_config = yaml.safe_load(f) or {}

        config = PrevCargaConfig(**raw_config)
        manager = cls(config)
        manager._source_file = str(filepath)

        logger.info("Loaded configuration from %s", filepath)
        return manager

    @classmethod
    def from_yaml_with_env_overrides(cls, filepath: str | Path) -> "ConfigManager":
        """Load configuration from YAML with environment variable overrides.

        Environment variables override YAML values. Use format:
        PREVCARGA_<SECTION>_<KEY>=value

        Examples:
            PREVCARGA_TRAINING_PARALLEL_WORKERS=8
            PREVCARGA_SYSTEM_LOG_LEVEL=DEBUG
            PREVCARGA_PREDICTION_OUTPUT_FORMAT=csv

        Args:
            filepath: Path to YAML configuration file.

        Returns:
            ConfigManager instance with env overrides applied.
        """
        manager = cls.from_yaml(filepath)
        manager._apply_env_overrides()
        return manager

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "ConfigManager":
        """Create ConfigManager from dictionary.

        Args:
            config_dict: Configuration dictionary.

        Returns:
            ConfigManager instance.
        """
        config = PrevCargaConfig(**config_dict)
        return cls(config)

    @classmethod
    def default(cls) -> "ConfigManager":
        """Create ConfigManager with default configuration.

        Returns:
            ConfigManager with defaults.
        """
        return cls(PrevCargaConfig())

    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides to configuration."""
        env_mappings = {
            # System config
            f"{self.ENV_PREFIX}SYSTEM_ENVIRONMENT": ("system", "environment"),
            f"{self.ENV_PREFIX}SYSTEM_LOG_LEVEL": ("system", "log_level"),
            f"{self.ENV_PREFIX}SYSTEM_LOG_FORMAT": ("system", "log_format"),
            f"{self.ENV_PREFIX}SYSTEM_TIMEZONE": ("system", "timezone"),
            f"{self.ENV_PREFIX}SYSTEM_TEMP_DIR": ("system", "temp_dir"),
            f"{self.ENV_PREFIX}SYSTEM_CACHE_ENABLED": ("system", "cache_enabled"),
            # Training config
            f"{self.ENV_PREFIX}TRAINING_PARALLEL_WORKERS": ("training", "parallel_workers"),
            f"{self.ENV_PREFIX}TRAINING_TIMEOUT_SECONDS": ("training", "timeout_seconds"),
            f"{self.ENV_PREFIX}TRAINING_OPTIMIZE_HYPERPARAMETERS": ("training", "optimize_hyperparameters"),
            f"{self.ENV_PREFIX}TRAINING_OPTIMIZATION_TRIALS": ("training", "optimization_trials"),
            f"{self.ENV_PREFIX}TRAINING_CV_FOLDS": ("training", "cv_folds"),
            f"{self.ENV_PREFIX}TRAINING_MODEL_OUTPUT_DIR": ("training", "model_output_dir"),
            # Prediction config
            f"{self.ENV_PREFIX}PREDICTION_MODEL_DIR": ("prediction", "model_dir"),
            f"{self.ENV_PREFIX}PREDICTION_OUTPUT_FORMAT": ("prediction", "output_format"),
            f"{self.ENV_PREFIX}PREDICTION_OUTPUT_DIR": ("prediction", "output_dir"),
            f"{self.ENV_PREFIX}PREDICTION_APPLY_RECONCILIATION": ("prediction", "apply_reconciliation"),
            # Backtesting config
            f"{self.ENV_PREFIX}BACKTESTING_START_DATE": ("backtesting", "start_date"),
            f"{self.ENV_PREFIX}BACKTESTING_END_DATE": ("backtesting", "end_date"),
            f"{self.ENV_PREFIX}BACKTESTING_STEP_DAYS": ("backtesting", "step_days"),
            f"{self.ENV_PREFIX}BACKTESTING_GENERATE_REPORT": ("backtesting", "generate_report"),
            # Reconciliation config
            f"{self.ENV_PREFIX}RECONCILIATION_ENABLED": ("reconciliation", "enabled"),
            f"{self.ENV_PREFIX}RECONCILIATION_METHOD": ("reconciliation", "method"),
        }

        with self._lock:
            config_dict = self._config.model_dump()

            for env_var, (section, key) in env_mappings.items():
                value = os.environ.get(env_var)
                if value is not None:
                    # Convert string to appropriate type
                    converted = self._convert_env_value(value, section, key)
                    if section in config_dict:
                        config_dict[section][key] = converted
                        logger.debug("Applied env override: %s=%s", env_var, converted)

            self._config = PrevCargaConfig(**config_dict)
            self._checksum = self._compute_checksum()

    def _convert_env_value(self, value: str, section: str, key: str) -> Any:
        """Convert environment variable string to appropriate type.

        Args:
            value: String value from environment.
            section: Configuration section name.
            key: Configuration key name.

        Returns:
            Converted value.
        """
        # Boolean conversions
        if value.lower() in ("true", "1", "yes"):
            return True
        if value.lower() in ("false", "0", "no"):
            return False

        # Integer conversions
        try:
            return int(value)
        except ValueError:
            pass

        # Float conversions
        try:
            return float(value)
        except ValueError:
            pass

        # List conversions (comma-separated)
        if "," in value:
            items = [item.strip() for item in value.split(",")]
            # Try to convert list items to integers
            try:
                return [int(item) for item in items]
            except ValueError:
                return items

        return value

    def _compute_checksum(self) -> str:
        """Compute checksum of current configuration.

        Returns:
            MD5 checksum string.
        """
        import hashlib
        import json

        def serialize(obj: Any) -> Any:
            """Convert enums and other non-serializable types."""
            if isinstance(obj, Enum):
                return obj.value
            if isinstance(obj, dict):
                return {k: serialize(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [serialize(item) for item in obj]
            return obj

        config_dict = serialize(self._config.model_dump())
        config_str = json.dumps(config_dict, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()

    def reload(self) -> bool:
        """Reload configuration from source file.

        Returns:
            True if configuration changed, False otherwise.

        Raises:
            RuntimeError: If no source file is set.
        """
        if not self._source_file:
            msg = "Cannot reload: no source file set"
            raise RuntimeError(msg)

        with self._lock:
            old_checksum = self._checksum

            with open(self._source_file, encoding="utf-8") as f:
                raw_config = yaml.safe_load(f) or {}

            self._config = PrevCargaConfig(**raw_config)
            self._apply_env_overrides()
            self._loaded_at = datetime.now()
            self._checksum = self._compute_checksum()

            changed = self._checksum != old_checksum

            if changed:
                logger.info("Configuration reloaded with changes from %s", self._source_file)
                self._notify_change_callbacks()
            else:
                logger.debug("Configuration reloaded (no changes)")

            return changed

    def register_change_callback(self, callback: callable) -> None:
        """Register callback for configuration changes.

        Args:
            callback: Callable to invoke when configuration changes.
        """
        with self._lock:
            self._change_callbacks.append(callback)

    def _notify_change_callbacks(self) -> None:
        """Notify all registered change callbacks."""
        for callback in self._change_callbacks:
            try:
                callback(self._config)
            except Exception as e:
                logger.error("Error in config change callback: %s", e)

    def get_config(self) -> PrevCargaConfig:
        """Get the complete configuration.

        Returns:
            Complete configuration.
        """
        with self._lock:
            return self._config.model_copy(deep=True)

    def get_system_config(self) -> SystemConfig:
        """Get system configuration.

        Returns:
            System configuration.
        """
        with self._lock:
            return self._config.system.model_copy(deep=True)

    def get_training_config(self) -> TrainingWorkflowConfig:
        """Get training workflow configuration.

        Returns:
            Training configuration.
        """
        with self._lock:
            return self._config.training.model_copy(deep=True)

    def get_prediction_config(self) -> PredictionWorkflowConfig:
        """Get prediction workflow configuration.

        Returns:
            Prediction configuration.
        """
        with self._lock:
            return self._config.prediction.model_copy(deep=True)

    def get_backtesting_config(self) -> BacktestingWorkflowConfig:
        """Get backtesting workflow configuration.

        Returns:
            Backtesting configuration.
        """
        with self._lock:
            return self._config.backtesting.model_copy(deep=True)

    def get_feature_config(self) -> FeatureConfig:
        """Get feature engineering configuration.

        Returns:
            Feature configuration.
        """
        with self._lock:
            return self._config.features.model_copy(deep=True)

    def get_reconciliation_config(self) -> ReconciliationConfig:
        """Get reconciliation configuration.

        Returns:
            Reconciliation configuration.
        """
        with self._lock:
            return self._config.reconciliation.model_copy(deep=True)

    def update_training_config(self, **kwargs: Any) -> None:
        """Update training configuration.

        Args:
            **kwargs: Configuration key-value pairs to update.

        Raises:
            ValueError: If invalid configuration key.
        """
        with self._lock:
            config_dict = self._config.model_dump()
            for key, value in kwargs.items():
                if key not in config_dict["training"]:
                    msg = f"Invalid training config key: {key}"
                    raise ValueError(msg)
                config_dict["training"][key] = value

            self._config = PrevCargaConfig(**config_dict)
            self._checksum = self._compute_checksum()
            logger.debug("Updated training config: %s", kwargs)

    def update_prediction_config(self, **kwargs: Any) -> None:
        """Update prediction configuration.

        Args:
            **kwargs: Configuration key-value pairs to update.
        """
        with self._lock:
            config_dict = self._config.model_dump()
            for key, value in kwargs.items():
                if key not in config_dict["prediction"]:
                    msg = f"Invalid prediction config key: {key}"
                    raise ValueError(msg)
                config_dict["prediction"][key] = value

            self._config = PrevCargaConfig(**config_dict)
            self._checksum = self._compute_checksum()
            logger.debug("Updated prediction config: %s", kwargs)

    def update_backtesting_config(self, **kwargs: Any) -> None:
        """Update backtesting configuration.

        Args:
            **kwargs: Configuration key-value pairs to update.
        """
        with self._lock:
            config_dict = self._config.model_dump()
            for key, value in kwargs.items():
                if key not in config_dict["backtesting"]:
                    msg = f"Invalid backtesting config key: {key}"
                    raise ValueError(msg)
                config_dict["backtesting"][key] = value

            self._config = PrevCargaConfig(**config_dict)
            self._checksum = self._compute_checksum()
            logger.debug("Updated backtesting config: %s", kwargs)

    def save_to_yaml(self, filepath: str | Path) -> None:
        """Save current configuration to YAML file.

        Args:
            filepath: Path to save configuration.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with self._lock:
            config_dict = self._config.model_dump()

        # Convert enums to their values for YAML
        def convert_enums(obj: Any) -> Any:
            if isinstance(obj, Enum):
                return obj.value
            if isinstance(obj, dict):
                return {k: convert_enums(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert_enums(item) for item in obj]
            return obj

        config_dict = convert_enums(config_dict)

        with open(filepath, "w", encoding="utf-8") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

        logger.info("Saved configuration to %s", filepath)

    def get_snapshot(self) -> ConfigSnapshot:
        """Get immutable snapshot of current configuration.

        Returns:
            ConfigSnapshot with current state.
        """
        with self._lock:
            return ConfigSnapshot(
                config=self._config.model_copy(deep=True),
                loaded_at=self._loaded_at,
                source_file=self._source_file,
                checksum=self._checksum,
            )

    def validate(self) -> list[str]:
        """Validate current configuration and return any warnings.

        Returns:
            List of warning messages.
        """
        warnings = []
        config = self._config

        # Check training configuration
        if config.training.enabled:
            if config.training.parallel_workers > 16:
                warnings.append(
                    f"High parallel_workers ({config.training.parallel_workers}) "
                    "may cause resource contention"
                )
            if config.training.optimize_hyperparameters and config.training.optimization_trials < 20:
                warnings.append(
                    f"Low optimization_trials ({config.training.optimization_trials}) "
                    "may result in suboptimal hyperparameters"
                )

        # Check prediction configuration
        if config.prediction.enabled:
            if not Path(config.prediction.model_dir).exists():
                warnings.append(
                    f"Model directory does not exist: {config.prediction.model_dir}"
                )

        # Check backtesting configuration
        if config.backtesting.enabled:
            if config.backtesting.start_date and config.backtesting.end_date:
                if config.backtesting.start_date >= config.backtesting.end_date:
                    warnings.append(
                        "Backtesting start_date must be before end_date"
                    )

        # Check reconciliation configuration
        if config.reconciliation.enabled:
            hierarchy_path = Path(config.reconciliation.hierarchy_file)
            if not hierarchy_path.exists():
                warnings.append(
                    f"Hierarchy file does not exist: {hierarchy_path}"
                )

        return warnings

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation.
        """
        with self._lock:
            return self._config.model_dump()

    def __repr__(self) -> str:
        """Return string representation.

        Returns:
            String representation.
        """
        return (
            f"ConfigManager(env={self._config.system.environment.value}, "
            f"source={self._source_file or 'default'})"
        )
