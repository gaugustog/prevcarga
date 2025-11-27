"""Metrics calculation and model evaluation module.

This module provides comprehensive metrics calculation and evaluation tools
for the PrevCarga electric load forecasting system.

Key Components:
- MetricsCalculator: Core metrics calculation engine (MAPE, MAE, RMSE, R²)
- MetricsConfig: Configuration for metrics calculation
- MetricsResult: Result container with calculated metrics
- StatisticalTests: Paired t-test, Wilcoxon signed-rank test
- DriftDetector: Drift detection for model monitoring
- DriftConfig: Configuration for drift detection
- DriftResult: Result container for drift detection

Example:
    ```python
    from src.evaluation import MetricsCalculator, MetricsConfig

    # Configure calculator
    config = MetricsConfig(
        use_symmetric_mape=True,
        confidence_level=0.95,
        bootstrap_iterations=1000
    )

    # Calculate metrics
    calculator = MetricsCalculator(config)
    result = calculator.calculate_metrics(
        predictions={"model_a": pred_array},
        actuals={"model_a": actual_array}
    )

    print(f"MAPE: {result['model_a'].mape:.2f}%")
    print(f"RMSE: {result['model_a'].rmse:.2f}")

    # Drift detection
    from src.evaluation import DriftDetector, DriftConfig

    detector = DriftDetector(DriftConfig())
    detector.fit(reference_data)
    drift_result = detector.detect(test_data)
    if drift_result.drift_detected:
        print(f"Drift detected! Score: {drift_result.drift_score:.4f}")
    ```
"""

from src.evaluation.drift import (
    DriftAlert,
    DriftConfig,
    DriftDetector,
    DriftMethod,
    DriftResult,
    DriftType,
)
from src.evaluation.model_comparator import (
    ComparisonConfig,
    ComparisonMetric,
    ComparisonResult,
    ModelComparator,
    ModelRanking,
    PairwiseComparison,
    RankingMethod,
)
from src.evaluation.horizon import (
    DegradationModel,
    HorizonAnalyzer,
    HorizonConfig,
    HorizonResult,
    SkillScores,
)
from src.evaluation.metrics import (
    MetricsCalculator,
    MetricsConfig,
    MetricsResult,
    StatisticalTests,
)
from src.evaluation.percentile import (
    DistributionAnalysis,
    OutlierAnalysis,
    PercentileAnalyzer,
)
from src.evaluation.time_period import (
    TimePeriodAnalyzer,
    TimePeriodConfig,
    TimePeriodResult,
)
from src.evaluation.report_generator import (
    EvaluationReport,
    ReportConfig,
    ReportFormat,
    ReportGenerator,
    ReportSection,
    SectionType,
)

__all__ = [
    "MetricsCalculator",
    "MetricsConfig",
    "MetricsResult",
    "StatisticalTests",
    "PercentileAnalyzer",
    "DistributionAnalysis",
    "OutlierAnalysis",
    "TimePeriodAnalyzer",
    "TimePeriodConfig",
    "TimePeriodResult",
    "HorizonAnalyzer",
    "HorizonConfig",
    "HorizonResult",
    "DegradationModel",
    "SkillScores",
    "DriftDetector",
    "DriftConfig",
    "DriftResult",
    "DriftType",
    "DriftMethod",
    "DriftAlert",
    "ModelComparator",
    "ComparisonConfig",
    "ComparisonResult",
    "ComparisonMetric",
    "RankingMethod",
    "ModelRanking",
    "PairwiseComparison",
    "ReportGenerator",
    "ReportConfig",
    "ReportSection",
    "ReportFormat",
    "SectionType",
    "EvaluationReport",
]
