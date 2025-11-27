"""CLI commands package for PrevCarga.

This package contains the command implementations for the CLI.

Key Modules:
- train: Training commands (train model, train batch)
- training_display: Training result display utilities
- predict: Prediction commands (predict batch, predict intraday)
- prediction_display: Prediction result display utilities
- features: Feature engineering commands (generate, evaluate, validate)
- feature_display: Feature result display utilities
- evaluation: Evaluation commands (metrics, drift, compare, report)
- evaluation_display: Evaluation result display utilities
- config: Configuration management commands (show, validate, init, diff, export)
"""

from src.cli.commands.config import (
    ConfigComparator,
    ConfigDiffResult,
    ConfigFormat,
    ConfigTemplate,
    ConfigTemplateGenerator,
    ConfigValidator,
    ValidationIssue,
    ValidationResult,
    ValidationSeverity,
    config,
    config_to_env_vars,
    config_to_json,
    config_to_yaml,
    load_config_file,
)
from src.cli.commands.evaluation import evaluate
from src.cli.commands.evaluation_display import (
    ComparisonDisplay,
    ComparisonDisplayResult,
    DisplayTheme,
    DriftDisplay,
    DriftDisplayResult,
    EvaluationDisplayConfig,
    MetricsDisplay,
    MetricsDisplayResult,
    display_evaluation_summary,
    display_metrics_plan,
    save_evaluation_report as save_eval_report,
)
from src.cli.commands.feature_display import (
    FeatureEvaluationResult,
    FeatureGenerationResult,
    FeatureProgressBar,
    FeatureResultDisplay,
    FeatureValidationResult,
    display_evaluation_results,
    display_generation_plan,
    display_generation_summary,
    display_validation_results,
    save_evaluation_report,
)
from src.cli.commands.features import features
from src.cli.commands.predict import predict
from src.cli.commands.prediction_display import (
    BatchPredictionResult,
    IntradayPredictionResult,
    PredictionResultDisplay,
    display_intraday_summary,
    display_prediction_plan,
    display_prediction_summary,
    save_predictions,
)
from src.cli.commands.train import train
from src.cli.commands.training_display import (
    TrainingResultDisplay,
    display_batch_results,
    display_batch_training_plan,
    display_training_plan,
    display_training_results,
)

__all__ = [
    # Training
    "TrainingResultDisplay",
    "display_batch_results",
    "display_batch_training_plan",
    "display_training_plan",
    "display_training_results",
    "train",
    # Prediction
    "BatchPredictionResult",
    "IntradayPredictionResult",
    "PredictionResultDisplay",
    "display_intraday_summary",
    "display_prediction_plan",
    "display_prediction_summary",
    "predict",
    "save_predictions",
    # Features
    "FeatureEvaluationResult",
    "FeatureGenerationResult",
    "FeatureProgressBar",
    "FeatureResultDisplay",
    "FeatureValidationResult",
    "display_evaluation_results",
    "display_generation_plan",
    "display_generation_summary",
    "display_validation_results",
    "features",
    "save_evaluation_report",
    # Evaluation
    "ComparisonDisplay",
    "ComparisonDisplayResult",
    "DisplayTheme",
    "DriftDisplay",
    "DriftDisplayResult",
    "EvaluationDisplayConfig",
    "MetricsDisplay",
    "MetricsDisplayResult",
    "display_evaluation_summary",
    "display_metrics_plan",
    "evaluate",
    "save_eval_report",
    # Config
    "ConfigComparator",
    "ConfigDiffResult",
    "ConfigFormat",
    "ConfigTemplate",
    "ConfigTemplateGenerator",
    "ConfigValidator",
    "ValidationIssue",
    "ValidationResult",
    "ValidationSeverity",
    "config",
    "config_to_env_vars",
    "config_to_json",
    "config_to_yaml",
    "load_config_file",
]
