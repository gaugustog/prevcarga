"""Validation module for PrevCarga.

This package provides validation frameworks for comparing system predictions
against PrevCargaDESSEM baseline and establishing reference standards for
system accuracy validation.

Key Components:
- BaselineValidator: Validates system accuracy against PrevCargaDESSEM baseline
- BaselineDataLoader: Loads and processes historical baseline data
- StatisticalTestSuite: Statistical tests for validation
- ComprehensiveBacktester: Full year backtesting with walk-forward validation

Example:
    ```python
    from src.validation import BaselineValidator, ValidationConfig

    # Create validator
    config = ValidationConfig(
        baseline_path="/data/baseline",
        areas=["SECO", "S", "NE", "N"],
        models=["lgbm", "rf", "regdin_svm", "holt_winters"]
    )
    validator = BaselineValidator(config)

    # Reproduce and validate baseline
    result = validator.reproduce_baseline(
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 1, 31)
    )

    # Comprehensive backtesting
    from src.validation import ComprehensiveBacktester, BacktestConfig

    backtest_config = BacktestConfig(backtest_year=2024)
    backtester = ComprehensiveBacktester(backtest_config)
    result = await backtester.execute_yearly_backtest()
    ```
"""

from src.validation.baseline_data_loader import (
    BaselineDataLoader,
    BaselineDataset,
    PredictionDataset,
)
from src.validation.baseline_validator import (
    BaselineReproductionResult,
    BaselineValidator,
    ComparisonResult,
    ValidationConfig,
    ValidationResult,
)
from src.validation.comprehensive_backtester import (
    BacktestConfig,
    BacktestExecutionError,
    BacktestPeriod,
    BacktestPeriodResult,
    CheckpointManager,
    ComprehensiveBacktester,
    PerformanceMonitor,
    PerformanceTargetViolation,
    YearlyBacktestResult,
)
from src.validation.model_performance_analyzer import (
    CombinationValidationResult,
    DriftDetectionResult,
    IndividualModelAnalysis,
    ModelPerformanceAnalyzer,
    PerformanceAnalysisConfig,
    ReconciliationImpactAnalysis,
)
from src.validation.statistical_tests import (
    StatisticalTestResult,
    StatisticalTestSuite,
)
from src.validation.system_performance_validator import (
    BenchmarkValidation,
    LoadTestResult,
    PerformanceBenchmarkResult,
    PerformanceMonitor as SystemPerformanceMonitor,
    PerformanceValidationConfig,
    ScenarioResult,
    SystemLoadTester,
    SystemPerformanceValidator,
)
from src.validation.robustness_validator import (
    ErrorInjector,
    FailureScenario,
    ResourceImpact,
    RobustnessConfig,
    RobustnessTestResult,
    RobustnessValidationResult,
    SystemRobustnessValidator,
)
from src.validation.epic_acceptance_validator import (
    AcceptanceCriterion,
    AcceptanceValidationConfig,
    CriterionStatus,
    CriterionValidationResult,
    EpicAcceptanceCriteriaValidator,
    EpicAcceptanceValidationResult,
    EpicValidationResult,
    GapAnalysisResult,
    ValidationType,
)
from src.validation.final_report_generator import (
    ApprovalStatus,
    ExecutiveSummary,
    FinalValidationReport,
    FinalValidationReportGenerator,
    ReportConfig,
    ReportMetadata,
    RiskLevel,
    SimpleEpicValidation,
    SimplePerformanceValidation,
    SimpleQualityValidation,
    SimpleRobustnessValidation,
    SimpleSecurityValidation,
    ValidationComponentSummary,
)

__all__ = [
    # Validator
    "BaselineValidator",
    "ValidationConfig",
    "ValidationResult",
    "BaselineReproductionResult",
    "ComparisonResult",
    # Data Loader
    "BaselineDataLoader",
    "BaselineDataset",
    "PredictionDataset",
    # Statistical Tests
    "StatisticalTestSuite",
    "StatisticalTestResult",
    # Comprehensive Backtester
    "ComprehensiveBacktester",
    "BacktestConfig",
    "BacktestPeriod",
    "BacktestPeriodResult",
    "YearlyBacktestResult",
    "CheckpointManager",
    "PerformanceMonitor",
    "BacktestExecutionError",
    "PerformanceTargetViolation",
    # Model Performance Analyzer
    "ModelPerformanceAnalyzer",
    "PerformanceAnalysisConfig",
    "IndividualModelAnalysis",
    "CombinationValidationResult",
    "ReconciliationImpactAnalysis",
    "DriftDetectionResult",
    # System Performance Validator
    "SystemPerformanceValidator",
    "PerformanceValidationConfig",
    "PerformanceBenchmarkResult",
    "BenchmarkValidation",
    "SystemPerformanceMonitor",
    "SystemLoadTester",
    "LoadTestResult",
    "ScenarioResult",
    # Robustness Validator
    "SystemRobustnessValidator",
    "RobustnessConfig",
    "RobustnessTestResult",
    "RobustnessValidationResult",
    "ErrorInjector",
    "FailureScenario",
    "ResourceImpact",
    # Epic Acceptance Validator
    "EpicAcceptanceCriteriaValidator",
    "AcceptanceValidationConfig",
    "AcceptanceCriterion",
    "CriterionValidationResult",
    "EpicValidationResult",
    "EpicAcceptanceValidationResult",
    "GapAnalysisResult",
    "ValidationType",
    "CriterionStatus",
    # Final Report Generator
    "FinalValidationReportGenerator",
    "ReportConfig",
    "ReportMetadata",
    "ExecutiveSummary",
    "FinalValidationReport",
    "ValidationComponentSummary",
    "ApprovalStatus",
    "RiskLevel",
    "SimpleEpicValidation",
    "SimplePerformanceValidation",
    "SimpleQualityValidation",
    "SimpleSecurityValidation",
    "SimpleRobustnessValidation",
]
