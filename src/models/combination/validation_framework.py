"""Validation framework for model combinations.

This module provides comprehensive validation tools for ensemble combination
weights and performance, including statistical significance testing, stability
analysis, drift detection, and automated alerting.

Example:
    >>> validator = WeightValidationFramework()
    >>>
    >>> # Validate weights
    >>> stability_result = validator.test_weight_stability(
    ...     combiner, predictions, targets
    ... )
    >>>
    >>> # Run full validation suite
    >>> results = validator.run_validation_suite(
    ...     combiner, predictions, targets
    ... )
    >>>
    >>> # Generate report
    >>> validator.generate_validation_report(results, 'report.html')
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ValidationConfig:
    """Configuration for validation framework.

    Attributes:
        weight_stability_threshold: Maximum acceptable standard deviation for weights
        extreme_weight_threshold: Flag weights above this value
        min_weight_threshold: Flag weights below this value
        significance_level: Alpha level for statistical tests
        bootstrap_samples: Number of bootstrap iterations
        drift_window_size: Minimum samples for drift detection
        drift_sigma: Control limit multiplier (standard deviations)
    """

    weight_stability_threshold: float = 0.1
    extreme_weight_threshold: float = 0.8
    min_weight_threshold: float = 0.05
    significance_level: float = 0.05
    bootstrap_samples: int = 100
    drift_window_size: int = 20
    drift_sigma: float = 3.0


@dataclass
class ValidationResult:
    """Results from validation tests.

    Attributes:
        passed: Whether the validation test passed
        test_name: Name of the validation test
        metrics: Numeric metrics from the test
        warnings: List of warning messages
        details: Additional detailed information
    """

    passed: bool
    test_name: str
    metrics: Dict[str, float]
    warnings: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


class AlertSystem:
    """Alert system for validation failures.

    Manages alerts generated during validation, supporting different
    severity levels and structured storage.
    """

    def __init__(self) -> None:
        """Initialize alert system."""
        self.alerts: List[Dict[str, Any]] = []

    def add_alert(
        self,
        severity: str,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add alert to system.

        Args:
            severity: Alert severity ('info', 'warning', 'critical')
            message: Alert message
            details: Optional additional details
        """
        alert = {
            'timestamp': pd.Timestamp.now(),
            'severity': severity,
            'message': message,
            'details': details or {}
        }
        self.alerts.append(alert)

        log_level_map = {'info': 20, 'warning': 30, 'critical': 40}  # INFO, WARNING, ERROR
        logger.log(
            log_level_map.get(severity, 30),
            f"ALERT [{severity.upper()}]: {message}"
        )

    def get_alerts(self, severity: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get alerts, optionally filtered by severity.

        Args:
            severity: Optional severity filter

        Returns:
            List of alert dictionaries
        """
        if severity is None:
            return self.alerts
        return [a for a in self.alerts if a['severity'] == severity]

    def clear_alerts(self) -> None:
        """Clear all alerts."""
        self.alerts = []


class WeightValidationFramework:
    """Comprehensive validation framework for model combinations.

    Validates combination weights and performance through multiple tests:
    - Weight stability (via bootstrapping)
    - Weight reasonableness (distribution checks)
    - Performance improvement (statistical significance)
    - Weight drift over time (statistical process control)

    Features:
    - Automatic alert generation for issues
    - Comprehensive HTML reporting
    - Historical tracking of validation results
    - Bootstrap-based confidence intervals

    Example:
        >>> config = {'bootstrap_samples': 50}
        >>> validator = WeightValidationFramework(config=config)
        >>>
        >>> # Test weight stability
        >>> result = validator.test_weight_stability(
        ...     combiner, predictions, targets
        ... )
        >>> print(f"Stability: {'PASS' if result.passed else 'FAIL'}")
        >>>
        >>> # Run full validation
        >>> results = validator.run_validation_suite(
        ...     combiner, predictions, targets
        ... )
        >>>
        >>> # Generate report
        >>> validator.generate_validation_report(results, 'val_report.html')
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize validation framework.

        Args:
            config: Configuration dictionary for validation parameters.
                If None, uses default configuration.
        """
        if config is None:
            config = {}

        self.config = ValidationConfig(**config)
        self.alert_system = AlertSystem()
        self.validation_history: List[Dict[str, Any]] = []

    def test_weight_stability(
        self,
        combiner: Any,
        predictions: Dict[str, pd.DataFrame],
        targets: pd.DataFrame
    ) -> ValidationResult:
        """Test weight stability using bootstrapping.

        Resamples data and refits combiner to assess weight variance.
        Stable weights indicate robust combination strategy.

        Args:
            combiner: Fitted combiner instance with get_weights() method
            predictions: Dictionary mapping model names to prediction DataFrames
            targets: True target values DataFrame

        Returns:
            ValidationResult with stability metrics and warnings
        """
        logger.info("Testing weight stability via bootstrapping")

        bootstrap_weights = self._bootstrap_weights(
            combiner, predictions, targets
        )

        # Calculate statistics
        weight_means = {
            model: np.mean([w[model] for w in bootstrap_weights])
            for model in bootstrap_weights[0].keys()
        }

        weight_stds = {
            model: np.std([w[model] for w in bootstrap_weights])
            for model in bootstrap_weights[0].keys()
        }

        # Check if any weights are unstable
        max_std = max(weight_stds.values())
        passed = bool(max_std < self.config.weight_stability_threshold)

        warnings = []
        for model, std in weight_stds.items():
            if std > self.config.weight_stability_threshold:
                warnings.append(
                    f"Model {model} has unstable weight (std={std:.3f})"
                )
                self.alert_system.add_alert(
                    'warning',
                    f"Unstable weight for {model}",
                    {'std': float(std), 'mean': float(weight_means[model])}
                )

        return ValidationResult(
            passed=passed,
            test_name='weight_stability',
            metrics={
                'max_std': float(max_std),
                'mean_std': float(np.mean(list(weight_stds.values())))
            },
            warnings=warnings,
            details={'weight_means': weight_means, 'weight_stds': weight_stds}
        )

    def test_weight_reasonableness(
        self,
        weights: Dict[str, float]
    ) -> ValidationResult:
        """Test if weights are reasonable.

        Checks for extreme weights, near-zero weights, and near-uniform
        distributions that might indicate problems.

        Args:
            weights: Dictionary mapping model names to weight values

        Returns:
            ValidationResult with reasonableness checks
        """
        logger.info("Testing weight reasonableness")

        warnings = []

        # Check for extreme weights
        for model, weight in weights.items():
            if weight > self.config.extreme_weight_threshold:
                warnings.append(f"Extreme weight for {model}: {weight:.3f}")
                self.alert_system.add_alert(
                    'warning',
                    f"Extreme weight for {model}",
                    {'weight': float(weight)}
                )

            if 0 < weight < self.config.min_weight_threshold:
                warnings.append(f"Very low weight for {model}: {weight:.3f}")

        # Check for near-uniform weights
        weight_values = list(weights.values())
        weight_variance = np.var(weight_values)

        if weight_variance < 0.01:
            warnings.append("Weights are nearly uniform, simple averaging may be sufficient")

        passed = len(warnings) == 0

        return ValidationResult(
            passed=passed,
            test_name='weight_reasonableness',
            metrics={'weight_variance': float(weight_variance)},
            warnings=warnings,
            details={'weights': weights}
        )

    def test_performance_significance(
        self,
        combination_errors: np.ndarray,
        individual_errors: Dict[str, np.ndarray]
    ) -> ValidationResult:
        """Test if combination significantly outperforms individual models.

        Uses paired t-test and Wilcoxon signed-rank test to assess
        statistical significance of performance improvements.

        Args:
            combination_errors: Absolute errors from combination
            individual_errors: Dict mapping model names to error arrays

        Returns:
            ValidationResult with significance test results
        """
        logger.info("Testing performance significance")

        test_results = {}
        warnings = []

        for model_name, model_errors in individual_errors.items():
            # Ensure same length
            min_len = min(len(model_errors), len(combination_errors))
            model_errors_trim = model_errors[:min_len]
            combo_errors_trim = combination_errors[:min_len]

            # Paired t-test
            t_stat, t_pvalue = stats.ttest_rel(model_errors_trim, combo_errors_trim)

            # Wilcoxon signed-rank test
            try:
                w_stat, w_pvalue = stats.wilcoxon(model_errors_trim, combo_errors_trim)
            except ValueError:
                # Handle case where all differences are zero
                w_stat, w_pvalue = 0.0, 1.0

            test_results[model_name] = {
                't_statistic': float(t_stat),
                't_pvalue': float(t_pvalue),
                'wilcoxon_statistic': float(w_stat),
                'wilcoxon_pvalue': float(w_pvalue),
                'significant': t_pvalue < self.config.significance_level
            }

            if t_pvalue >= self.config.significance_level:
                warnings.append(
                    f"Not significantly better than {model_name} (p={t_pvalue:.3f})"
                )

        # Check if combination beats at least some models significantly
        significant_improvements = sum(
            1 for r in test_results.values() if r['significant']
        )

        passed = significant_improvements > 0

        if not passed:
            self.alert_system.add_alert(
                'critical',
                "Combination not significantly better than any individual model",
                {'test_results': test_results}
            )

        return ValidationResult(
            passed=passed,
            test_name='performance_significance',
            metrics={'significant_improvements': float(significant_improvements)},
            warnings=warnings,
            details={'test_results': test_results}
        )

    def detect_weight_drift(
        self,
        weight_history: List[Dict[str, float]]
    ) -> ValidationResult:
        """Detect weight drift using statistical process control.

        Monitors weight evolution over time and detects out-of-control
        points using ±3σ control limits.

        Args:
            weight_history: List of weight dictionaries over time

        Returns:
            ValidationResult with drift detection results
        """
        logger.info("Detecting weight drift")

        if len(weight_history) < self.config.drift_window_size:
            return ValidationResult(
                passed=True,
                test_name='weight_drift',
                metrics={},
                warnings=['Insufficient history for drift detection'],
                details={}
            )

        # Convert to DataFrame for easier analysis
        weight_df = pd.DataFrame(weight_history)

        # Calculate control limits for each model
        drift_detected = {}
        warnings = []

        for model in weight_df.columns:
            weights = weight_df[model].values

            # Calculate moving statistics
            mean = np.mean(weights)
            std = np.std(weights)

            # Control limits (±3σ)
            ucl = mean + self.config.drift_sigma * std
            lcl = mean - self.config.drift_sigma * std

            # Check recent weights
            recent_weights = weights[-5:]
            out_of_control = np.any((recent_weights > ucl) | (recent_weights < lcl))

            drift_detected[model] = bool(out_of_control)

            if out_of_control:
                warnings.append(f"Weight drift detected for {model}")
                self.alert_system.add_alert(
                    'warning',
                    f"Weight drift detected for {model}",
                    {
                        'mean': float(mean),
                        'std': float(std),
                        'recent_weights': recent_weights.tolist()
                    }
                )

        passed = not any(drift_detected.values())

        return ValidationResult(
            passed=passed,
            test_name='weight_drift',
            metrics={'models_with_drift': float(sum(drift_detected.values()))},
            warnings=warnings,
            details={'drift_detected': drift_detected}
        )

    def run_validation_suite(
        self,
        combiner: Any,
        predictions: Dict[str, pd.DataFrame],
        targets: pd.DataFrame,
        weight_history: Optional[List[Dict[str, float]]] = None
    ) -> Dict[str, ValidationResult]:
        """Run complete validation suite.

        Executes all enabled validation tests and aggregates results.

        Args:
            combiner: Fitted combiner instance
            predictions: Dictionary of model predictions
            targets: True target values
            weight_history: Optional weight history for drift detection

        Returns:
            Dictionary mapping test names to ValidationResult objects
        """
        logger.info("Running validation suite")

        results = {}

        # Test 1: Weight stability
        results['stability'] = self.test_weight_stability(
            combiner, predictions, targets
        )

        # Test 2: Weight reasonableness
        weights = combiner.get_weights()
        results['reasonableness'] = self.test_weight_reasonableness(weights)

        # Test 3: Weight drift (if history available)
        if weight_history:
            results['drift'] = self.detect_weight_drift(weight_history)

        # Overall pass/fail
        all_passed = all(r.passed for r in results.values())

        logger.info(f"Validation suite: {'PASSED' if all_passed else 'FAILED'}")

        # Store in history
        self.validation_history.append({
            'timestamp': pd.Timestamp.now(),
            'results': results,
            'passed': all_passed
        })

        return results

    def generate_validation_report(
        self,
        results: Dict[str, ValidationResult],
        output_path: str
    ) -> None:
        """Generate comprehensive validation report.

        Creates an HTML report with validation results, metrics,
        warnings, and alerts.

        Args:
            output_path: Path to save HTML report
            results: Validation results dictionary
        """
        logger.info(f"Generating validation report: {output_path}")

        # Create HTML report
        html = ["<html><head><title>Validation Report</title>"]
        html.append("<style>")
        html.append("body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }")
        html.append(".container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; }")
        html.append(".passed { color: #28a745; font-weight: bold; }")
        html.append(".failed { color: #dc3545; font-weight: bold; }")
        html.append("table { border-collapse: collapse; width: 100%; margin: 20px 0; }")
        html.append("th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }")
        html.append("th { background-color: #007bff; color: white; }")
        html.append("tr:nth-child(even) { background-color: #f2f2f2; }")
        html.append("h1 { color: #333; border-bottom: 3px solid #007bff; padding-bottom: 10px; }")
        html.append("h2 { color: #555; margin-top: 30px; }")
        html.append("h3 { color: #666; }")
        html.append("</style></head><body><div class='container'>")

        html.append("<h1>Model Combination Validation Report</h1>")
        html.append(f"<p><strong>Generated:</strong> {pd.Timestamp.now()}</p>")

        # Summary table
        html.append("<h2>Validation Summary</h2>")
        html.append("<table>")
        html.append("<tr><th>Test</th><th>Status</th><th>Warnings</th></tr>")

        for test_name, result in results.items():
            status_class = 'passed' if result.passed else 'failed'
            status_text = 'PASSED' if result.passed else 'FAILED'

            html.append("<tr>")
            html.append(f"<td>{result.test_name}</td>")
            html.append(f"<td class='{status_class}'>{status_text}</td>")
            html.append(f"<td>{len(result.warnings)}</td>")
            html.append("</tr>")

        html.append("</table>")

        # Detailed results
        html.append("<h2>Detailed Results</h2>")

        for test_name, result in results.items():
            html.append(f"<h3>{result.test_name}</h3>")
            status_class = 'passed' if result.passed else 'failed'
            status_text = 'PASSED' if result.passed else 'FAILED'
            html.append(f"<p><strong>Status:</strong> <span class='{status_class}'>{status_text}</span></p>")

            if result.metrics:
                html.append("<p><strong>Metrics:</strong></p><ul>")
                for metric, value in result.metrics.items():
                    html.append(f"<li>{metric}: {value:.4f}</li>")
                html.append("</ul>")

            if result.warnings:
                html.append("<p><strong>Warnings:</strong></p><ul>")
                for warning in result.warnings:
                    html.append(f"<li>{warning}</li>")
                html.append("</ul>")

        # Alerts
        alerts = self.alert_system.get_alerts()
        if alerts:
            html.append("<h2>Alerts</h2>")
            html.append("<table>")
            html.append("<tr><th>Timestamp</th><th>Severity</th><th>Message</th></tr>")

            for alert in alerts:
                html.append("<tr>")
                html.append(f"<td>{alert['timestamp']}</td>")
                html.append(f"<td>{alert['severity'].upper()}</td>")
                html.append(f"<td>{alert['message']}</td>")
                html.append("</tr>")

            html.append("</table>")

        html.append("</div></body></html>")

        # Write to file
        with open(output_path, 'w') as f:
            f.write('\n'.join(html))

        logger.info(f"Report saved to {output_path}")

    def _bootstrap_weights(
        self,
        combiner: Any,
        predictions: Dict[str, pd.DataFrame],
        targets: pd.DataFrame
    ) -> List[Dict[str, float]]:
        """Bootstrap weights by resampling data.

        Performs bootstrap resampling to estimate weight variability
        and confidence intervals.

        Args:
            combiner: Combiner instance to refit
            predictions: Model predictions dictionary
            targets: True target values

        Returns:
            List of weight dictionaries from bootstrap samples
        """
        n_samples = len(next(iter(predictions.values())))
        bootstrap_weights = []

        for i in range(self.config.bootstrap_samples):
            if i % 20 == 0:
                logger.debug(f"Bootstrap iteration {i}/{self.config.bootstrap_samples}")

            # Resample indices with replacement
            indices = np.random.choice(n_samples, size=n_samples, replace=True)

            # Resample data
            boot_predictions = {
                model: pred.iloc[indices].copy()
                for model, pred in predictions.items()
            }
            boot_targets = targets.iloc[indices].copy()

            # Fit combiner on bootstrap sample
            try:
                boot_combiner = combiner.__class__(config=combiner.config)
                boot_combiner.fit(boot_predictions, boot_targets)

                # Get weights
                weights = boot_combiner.get_weights()
                bootstrap_weights.append(weights)
            except Exception as e:
                logger.warning(f"Bootstrap iteration {i} failed: {e}")
                continue

        if len(bootstrap_weights) < self.config.bootstrap_samples // 2:
            logger.warning(
                f"Only {len(bootstrap_weights)} bootstrap samples succeeded"
            )

        return bootstrap_weights

    def __repr__(self) -> str:
        """String representation of validation framework."""
        return (
            f"WeightValidationFramework("
            f"bootstrap_samples={self.config.bootstrap_samples}, "
            f"validation_history={len(self.validation_history)})"
        )
