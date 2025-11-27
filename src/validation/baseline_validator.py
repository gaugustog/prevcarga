"""Baseline validator for PrevCargaDESSEM comparison.

This module provides the main validation framework for comparing system
predictions against the PrevCargaDESSEM baseline to establish reference
standards for system accuracy.

Key Features:
- Baseline reproduction with tolerance validation
- System vs baseline comparison across all metrics
- Support for 26 time series (4 subsystems × models)
- Configurable tolerance thresholds

Example:
    ```python
    from src.validation.baseline_validator import BaselineValidator, ValidationConfig
    from datetime import datetime

    config = ValidationConfig(
        baseline_path="/data/baseline",
        areas=["SECO", "S", "NE", "N"],
        models=["lgbm", "rf", "regdin_svm", "holt_winters"]
    )
    validator = BaselineValidator(config)

    # Reproduce baseline
    result = validator.reproduce_baseline(
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 1, 31)
    )
    print(f"Valid: {result.validation_result.is_valid}")
    ```
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from src.utils.logger import get_logger
from src.validation.baseline_data_loader import (
    BaselineDataLoader,
    BaselineDataset,
    PredictionDataset,
)
from src.validation.statistical_tests import StatisticalTestSuite, ValidationResult

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


# Default areas for Brazilian grid
DEFAULT_AREAS = ["SECO", "S", "NE", "N", "SIN"]

# Default models in PrevCarga system
DEFAULT_MODELS = ["lgbm", "rf", "arima", "holt_winters", "regdin_svm"]


@dataclass
class ValidationConfig:
    """Configuration for baseline validation.

    Attributes:
        baseline_path: Path to baseline data directory.
        areas: List of areas to validate.
        models: List of models to validate.
        mape_tolerance: MAPE tolerance threshold (default ±5%).
        mae_tolerance: MAE tolerance threshold in MW (default 100).
        rmse_tolerance: RMSE tolerance threshold in MW (default 150).
        confidence_level: Confidence level for statistical tests (default 0.95).
    """

    baseline_path: str | Path
    areas: list[str] = field(default_factory=lambda: DEFAULT_AREAS.copy())
    models: list[str] = field(default_factory=lambda: DEFAULT_MODELS.copy())
    mape_tolerance: float = 0.05  # ±5%
    mae_tolerance: float = 100.0  # MW
    rmse_tolerance: float = 150.0  # MW
    confidence_level: float = 0.95

    def __post_init__(self) -> None:
        """Validate configuration."""
        self.baseline_path = Path(self.baseline_path)

        if self.mape_tolerance < 0:
            raise ValueError(f"mape_tolerance must be non-negative, got {self.mape_tolerance}")
        if self.mae_tolerance < 0:
            raise ValueError(f"mae_tolerance must be non-negative, got {self.mae_tolerance}")
        if self.rmse_tolerance < 0:
            raise ValueError(f"rmse_tolerance must be non-negative, got {self.rmse_tolerance}")
        if not 0 < self.confidence_level < 1:
            raise ValueError(f"confidence_level must be between 0 and 1, got {self.confidence_level}")

    @property
    def tolerance_thresholds(self) -> dict[str, float]:
        """Get tolerance thresholds as dictionary.

        Returns:
            Dictionary of tolerance thresholds.
        """
        return {
            "mape": self.mape_tolerance,
            "mae": self.mae_tolerance,
            "rmse": self.rmse_tolerance,
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "baseline_path": str(self.baseline_path),
            "areas": self.areas,
            "models": self.models,
            "mape_tolerance": self.mape_tolerance,
            "mae_tolerance": self.mae_tolerance,
            "rmse_tolerance": self.rmse_tolerance,
            "confidence_level": self.confidence_level,
        }


@dataclass
class BaselineReproductionResult:
    """Results from baseline reproduction.

    Attributes:
        baseline_metrics: Calculated baseline metrics by area_model key.
        validation_result: Validation result with pass/fail status.
        reference_dataset: Reference data for archival.
        reproduction_timestamp: When reproduction was performed.
        config: Configuration used.
    """

    baseline_metrics: dict[str, dict[str, float]]
    validation_result: ValidationResult
    reference_dataset: dict[str, Any]
    reproduction_timestamp: datetime
    config: ValidationConfig | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation.
        """
        return {
            "baseline_metrics": self.baseline_metrics,
            "validation_result": self.validation_result.to_dict(),
            "reference_dataset": self.reference_dataset,
            "timestamp": self.reproduction_timestamp.isoformat(),
            "config": self.config.to_dict() if self.config else None,
        }


@dataclass
class ComparisonResult:
    """Results from baseline comparison.

    Attributes:
        metrics: Detailed comparison metrics by area_model.
        summary: Summary statistics.
        violations: List of tolerance violations.
        statistical_tests: Results of statistical tests.
    """

    metrics: dict[str, dict[str, float]]
    summary: dict[str, float]
    violations: list[str]
    statistical_tests: dict[str, Any] = field(default_factory=dict)

    def passes_validation(self) -> bool:
        """Check if all comparisons pass tolerance thresholds.

        Returns:
            True if no violations.
        """
        return len(self.violations) == 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "metrics": self.metrics,
            "summary": self.summary,
            "violations": self.violations,
            "passes_validation": self.passes_validation(),
            "statistical_tests": self.statistical_tests,
        }


class BaselineValidator:
    """Validates system accuracy against PrevCargaDESSEM baseline.

    This class provides the main validation framework for comparing
    PrevCarga system predictions against the reference PrevCargaDESSEM
    baseline.

    Attributes:
        config: Validation configuration.
        baseline_data: Baseline data loader.
        statistical_tests: Statistical test suite.
    """

    def __init__(self, config: ValidationConfig) -> None:
        """Initialize validator.

        Args:
            config: Validation configuration.
        """
        self.config = config
        self.baseline_data = BaselineDataLoader(config.baseline_path)
        self.statistical_tests = StatisticalTestSuite(config.confidence_level)

    def reproduce_baseline(
        self,
        start_date: datetime,
        end_date: datetime,
        models: list[str] | None = None,
    ) -> BaselineReproductionResult:
        """Reproduce PrevCargaDESSEM baseline for validation period.

        Args:
            start_date: Start of validation period.
            end_date: End of validation period.
            models: Models to validate (default: all configured models).

        Returns:
            BaselineReproductionResult with metrics and validation status.
        """
        logger.info(f"Reproducing baseline from {start_date} to {end_date}")

        models_to_use = models or self.config.models

        # Calculate baseline metrics
        baseline_metrics = self.baseline_data.calculate_metrics(
            start_date, end_date, models_to_use
        )

        # Validate baseline reproduction
        validation_result = self.statistical_tests.validate_baseline(
            baseline_metrics, self.config.tolerance_thresholds
        )

        # Create reference dataset
        reference_dataset = self.baseline_data.to_reference_format(baseline_metrics)

        result = BaselineReproductionResult(
            baseline_metrics=baseline_metrics,
            validation_result=validation_result,
            reference_dataset=reference_dataset,
            reproduction_timestamp=datetime.now(),
            config=self.config,
        )

        logger.info(f"Baseline reproduction complete. Valid: {validation_result.is_valid}")
        return result

    def compare_with_baseline(
        self,
        system_predictions: PredictionDataset,
        baseline_predictions: BaselineDataset,
    ) -> ComparisonResult:
        """Compare system predictions with baseline across all metrics.

        Args:
            system_predictions: Predictions from PrevCarga system.
            baseline_predictions: Predictions from PrevCargaDESSEM.

        Returns:
            ComparisonResult with detailed comparison metrics.
        """
        logger.info("Comparing system predictions with baseline")

        comparison_metrics: dict[str, dict[str, float]] = {}
        violations: list[str] = []
        statistical_tests: dict[str, Any] = {}

        # Get areas and models to compare
        areas = list(set(self.config.areas) & set(system_predictions.areas) & set(baseline_predictions.areas))
        models = list(set(self.config.models) & set(system_predictions.models) & set(baseline_predictions.models))

        if not areas or not models:
            logger.warning("No common areas or models to compare")
            return ComparisonResult(
                metrics={},
                summary={},
                violations=["No common areas or models to compare"],
            )

        for area in areas:
            for model in models:
                key = f"{area}_{model}"

                # Get system predictions and actuals
                system_preds = system_predictions.get_predictions(area, model)
                system_actuals = system_predictions.get_actuals(area)

                if len(system_preds) == 0 or len(system_actuals) == 0:
                    logger.warning(f"No system data for {key}")
                    continue

                # Calculate system MAPE
                system_mape = self._calculate_mape(system_preds, system_actuals)

                # Get baseline MAPE
                baseline_mape = baseline_predictions.get_mape(area, model)

                if np.isnan(system_mape) or np.isnan(baseline_mape):
                    logger.warning(f"NaN metrics for {key}")
                    continue

                # Calculate difference
                difference = system_mape - baseline_mape
                within_tolerance = abs(difference) <= (self.config.mape_tolerance * 100)  # Convert to percentage points

                comparison_metrics[key] = {
                    "system_mape": system_mape,
                    "baseline_mape": baseline_mape,
                    "difference": difference,
                    "difference_percentage": (difference / baseline_mape) * 100 if baseline_mape != 0 else 0,
                    "within_tolerance": float(within_tolerance),
                }

                # Track violations
                if not within_tolerance:
                    violations.append(
                        f"{key}: difference {difference:.3f}% exceeds tolerance "
                        f"{self.config.mape_tolerance * 100:.1f}%"
                    )

                # Perform statistical test if we have enough data
                if len(system_preds) >= 10:
                    baseline_preds = baseline_predictions.get_predictions(area, model)
                    if len(baseline_preds) >= 10:
                        test_result = self.statistical_tests.test_significance(
                            system_preds[:min(len(system_preds), len(baseline_preds))],
                            baseline_preds[:min(len(system_preds), len(baseline_preds))],
                        )
                        statistical_tests[key] = test_result.to_dict()

        # Calculate summary statistics
        summary = self._calculate_summary(comparison_metrics)

        result = ComparisonResult(
            metrics=comparison_metrics,
            summary=summary,
            violations=violations,
            statistical_tests=statistical_tests,
        )

        logger.info(f"Comparison complete. Violations: {len(violations)}")
        return result

    def validate_data_alignment(
        self,
        system_data: PredictionDataset,
        baseline_data: BaselineDataset,
    ) -> dict[str, Any]:
        """Validate data alignment between systems.

        Args:
            system_data: System prediction dataset.
            baseline_data: Baseline prediction dataset.

        Returns:
            Alignment validation results.
        """
        logger.info("Validating data alignment")

        alignment_results: dict[str, Any] = {
            "date_range_match": False,
            "area_match": False,
            "model_match": False,
            "record_count_match": False,
            "details": {},
        }

        # Check date ranges
        if system_data.start_date == baseline_data.start_date and system_data.end_date == baseline_data.end_date:
            alignment_results["date_range_match"] = True
        alignment_results["details"]["date_range"] = {
            "system": f"{system_data.start_date} to {system_data.end_date}",
            "baseline": f"{baseline_data.start_date} to {baseline_data.end_date}",
        }

        # Check areas
        system_areas = set(system_data.areas)
        baseline_areas = set(baseline_data.areas)
        common_areas = system_areas & baseline_areas
        alignment_results["area_match"] = len(common_areas) >= min(len(system_areas), len(baseline_areas)) * 0.8
        alignment_results["details"]["areas"] = {
            "system": list(system_areas),
            "baseline": list(baseline_areas),
            "common": list(common_areas),
        }

        # Check models
        system_models = set(system_data.models)
        baseline_models = set(baseline_data.models)
        common_models = system_models & baseline_models
        alignment_results["model_match"] = len(common_models) >= min(len(system_models), len(baseline_models)) * 0.8
        alignment_results["details"]["models"] = {
            "system": list(system_models),
            "baseline": list(baseline_models),
            "common": list(common_models),
        }

        # Check record counts
        system_count = len(system_data.data)
        baseline_count = len(baseline_data.data)
        count_ratio = min(system_count, baseline_count) / max(system_count, baseline_count) if max(system_count, baseline_count) > 0 else 0
        alignment_results["record_count_match"] = count_ratio >= 0.9
        alignment_results["details"]["record_counts"] = {
            "system": system_count,
            "baseline": baseline_count,
            "ratio": count_ratio,
        }

        alignment_results["overall_aligned"] = all([
            alignment_results["date_range_match"],
            alignment_results["area_match"],
            alignment_results["model_match"],
            alignment_results["record_count_match"],
        ])

        return alignment_results

    def generate_reference_archive(
        self,
        reproduction_result: BaselineReproductionResult,
    ) -> dict[str, Any]:
        """Generate reference archive for all time series.

        Args:
            reproduction_result: Result from baseline reproduction.

        Returns:
            Reference archive dictionary.
        """
        logger.info("Generating reference archive")

        archive: dict[str, Any] = {
            "generated_at": datetime.now().isoformat(),
            "config": self.config.to_dict(),
            "metrics_summary": {},
            "time_series": {},
            "validation_status": reproduction_result.validation_result.to_dict(),
        }

        # Organize by time series
        for key, metrics in reproduction_result.baseline_metrics.items():
            area, model = key.rsplit("_", 1)

            if area not in archive["time_series"]:
                archive["time_series"][area] = {}

            archive["time_series"][area][model] = {
                "mape": metrics.get("mape"),
                "mae": metrics.get("mae"),
                "rmse": metrics.get("rmse"),
                "within_tolerance": {
                    "mape": metrics.get("mape", float("inf")) <= self.config.mape_tolerance,
                    "mae": metrics.get("mae", float("inf")) <= self.config.mae_tolerance,
                    "rmse": metrics.get("rmse", float("inf")) <= self.config.rmse_tolerance,
                },
            }

        # Add overall summary
        archive["metrics_summary"] = reproduction_result.validation_result.summary

        return archive

    def _calculate_mape(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
    ) -> float:
        """Calculate Mean Absolute Percentage Error.

        Args:
            predictions: Predicted values.
            actuals: Actual values.

        Returns:
            MAPE as percentage.
        """
        if len(predictions) == 0 or len(actuals) == 0:
            return np.nan

        # Align lengths
        min_len = min(len(predictions), len(actuals))
        predictions = predictions[:min_len]
        actuals = actuals[:min_len]

        # Avoid division by zero
        mask = actuals != 0
        if not mask.any():
            return np.nan

        return float(np.mean(np.abs((actuals[mask] - predictions[mask]) / actuals[mask])) * 100)

    def _calculate_summary(
        self,
        comparison_metrics: dict[str, dict[str, float]],
    ) -> dict[str, float]:
        """Calculate summary statistics from comparison metrics.

        Args:
            comparison_metrics: Dictionary of comparison metrics.

        Returns:
            Summary statistics.
        """
        if not comparison_metrics:
            return {
                "mean_difference": 0.0,
                "std_difference": 0.0,
                "max_difference": 0.0,
                "min_difference": 0.0,
                "pass_rate": 0.0,
                "total_comparisons": 0,
            }

        differences = [m["difference"] for m in comparison_metrics.values()]
        within_tolerance = [m["within_tolerance"] for m in comparison_metrics.values()]

        return {
            "mean_difference": float(np.mean(differences)),
            "std_difference": float(np.std(differences)),
            "max_difference": float(np.max(differences)),
            "min_difference": float(np.min(differences)),
            "pass_rate": float(sum(within_tolerance)) / len(within_tolerance),
            "total_comparisons": len(comparison_metrics),
        }
