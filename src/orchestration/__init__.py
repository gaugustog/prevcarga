"""Workflow orchestration and execution coordination.

This package provides workflow orchestration for the PrevCarga electric
load forecasting system, including configuration management, training,
prediction, and backtesting workflows.

Key Components:
- ConfigManager: Central configuration management
- WorkflowConfig: Base workflow configuration
- TrainingWorkflowConfig: Training workflow configuration
- PredictionWorkflowConfig: Prediction workflow configuration
- BacktestingWorkflowConfig: Backtesting workflow configuration
- SystemConfig: System-level configuration
- PrevCargaConfig: Complete system configuration

Example:
    ```python
    from src.orchestration import ConfigManager

    # Load configuration
    manager = ConfigManager.from_yaml("config/prevcarga.yaml")

    # Get workflow configurations
    training_config = manager.get_training_config()
    prediction_config = manager.get_prediction_config()

    # Apply environment variable overrides
    manager = ConfigManager.from_yaml_with_env_overrides("config/prevcarga.yaml")
    ```
"""

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
from src.orchestration.training_workflow import (
    StageResult,
    TrainingWorkflow,
    WorkflowResult,
    WorkflowStage,
    WorkflowStatus,
)
from src.orchestration.prediction_workflow import (
    ForecastOutput,
    PredictionResult,
    PredictionStage,
    PredictionStatus,
    PredictionWorkflow,
    StageResult as PredictionStageResult,
)
from src.orchestration.backtesting_workflow import (
    BacktestingResult,
    BacktestingStage,
    BacktestingStatus,
    BacktestingWorkflow,
    BacktestWindow,
    StageResult as BacktestingStageResult,
)
from src.orchestration.parallel_executor import (
    BatchResult,
    ExecutionStatus,
    ParallelExecutor,
    ProgressTrackingExecutor,
    TaskResult,
    create_executor,
)
from src.orchestration.structured_logger import (
    AggregatedStats,
    LogAggregator,
    LogEntry,
    LogFormat,
    LogLevel as StructuredLogLevel,
    PerformanceLogger,
    StructuredLogger,
    WorkflowLogContext,
    create_workflow_logger,
)

__all__ = [
    "ConfigManager",
    "ConfigSnapshot",
    "Environment",
    "LogLevel",
    "StorageBackendType",
    "ParallelBackend",
    "StorageConfig",
    "SystemConfig",
    "WorkflowConfig",
    "TrainingWorkflowConfig",
    "PredictionWorkflowConfig",
    "BacktestingWorkflowConfig",
    "FeatureConfig",
    "ReconciliationConfig",
    "PrevCargaConfig",
    "TrainingWorkflow",
    "WorkflowResult",
    "WorkflowStage",
    "WorkflowStatus",
    "StageResult",
    "PredictionWorkflow",
    "PredictionResult",
    "PredictionStage",
    "PredictionStatus",
    "PredictionStageResult",
    "ForecastOutput",
    "BacktestingWorkflow",
    "BacktestingResult",
    "BacktestingStage",
    "BacktestingStatus",
    "BacktestingStageResult",
    "BacktestWindow",
    "ParallelExecutor",
    "ProgressTrackingExecutor",
    "TaskResult",
    "BatchResult",
    "ExecutionStatus",
    "create_executor",
    "StructuredLogger",
    "StructuredLogLevel",
    "LogFormat",
    "LogEntry",
    "LogAggregator",
    "AggregatedStats",
    "PerformanceLogger",
    "WorkflowLogContext",
    "create_workflow_logger",
]
