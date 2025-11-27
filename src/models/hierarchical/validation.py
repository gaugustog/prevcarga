"""Validation framework for hierarchical models.

This module provides comprehensive validation tools for hierarchical forecasting
models, including energy conservation tests, profile bounds checks, seasonal
consistency validation, and detailed component analysis.

The validation framework ensures:
1. Energy conservation: Daily sums match demand mean × 48
2. Profile bounds: Ratios stay within reasonable range [0.1, 5.0]
3. Seasonal consistency: Stable patterns across weeks
4. Component validity: All model components function correctly

Example:
    ```python
    from src.models.hierarchical.validation import HierarchicalValidator
    from src.models.hierarchical.regdin_svm import RegDinSVMModel
    import pandas as pd

    # Create validator
    validator = HierarchicalValidator()

    # Train model
    model = RegDinSVMModel()
    model.fit(X_train, y_train, config)

    # Run validation
    results = validator.validate_model(model, X_test, y_test)

    # Generate report
    report = validator.generate_validation_report(results)
    print(report)

    # Check if validation passed
    if results["passed"]:
        print("Model passed all validation tests")
    else:
        print("Model failed validation")
    ```
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.models.hierarchical.base_hierarchical import BaseHierarchicalModel
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Validation constants
DEFAULT_ENERGY_TOLERANCE = 0.05  # 5% tolerance for energy conservation
DEFAULT_PROFILE_MIN = 0.1  # Minimum acceptable profile ratio
DEFAULT_PROFILE_MAX = 5.0  # Maximum acceptable profile ratio
DEFAULT_PROFILE_VIOLATION_THRESHOLD = 0.01  # 1% violation rate threshold
DEFAULT_SEASONAL_STD_THRESHOLD = 0.5  # Seasonal consistency threshold
PERIODS_PER_DAY = 48  # Semi-hourly periods in a day


class HierarchicalValidator:
    """Comprehensive validation framework for hierarchical models.

    This class provides a complete validation suite for hierarchical forecasting
    models, including multiple validation tests, component analysis tools, and
    detailed reporting capabilities.

    Validation Tests:
        - Energy Conservation: Verifies daily energy sums match expected values
        - Profile Bounds: Checks profile ratios stay within [0.1, 5.0]
        - Seasonal Consistency: Validates stable patterns across weeks
        - Component Analysis: Analyzes individual model components

    Attributes:
        validation_results: Dictionary storing the most recent validation results.

    Example:
        >>> validator = HierarchicalValidator()
        >>> results = validator.validate_model(model, X_test, y_test)
        >>> print(f"Validation {'PASSED' if results['passed'] else 'FAILED'}")
        >>> report = validator.generate_validation_report(results)
    """

    def __init__(self) -> None:
        """Initialize the hierarchical validator."""
        self.validation_results: dict[str, Any] = {}
        logger.debug("Initialized HierarchicalValidator")

    def validate_model(
        self,
        model: BaseHierarchicalModel,
        X: pd.DataFrame,  # noqa: N803
        y: pd.Series | pd.DataFrame | None = None,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run comprehensive validation suite on a hierarchical model.

        This method executes all validation tests and aggregates results into
        a comprehensive validation report.

        Args:
            model: Trained hierarchical model to validate.
            X: Validation features with semi-hourly resolution.
            y: Validation target (load). Optional, required for some tests.
            config: Validation configuration with optional keys:
                   - "energy_tolerance": Max energy conservation error (default: 0.05)
                   - "profile_min": Minimum acceptable profile ratio (default: 0.1)
                   - "profile_max": Maximum acceptable profile ratio (default: 5.0)
                   - "seasonal_std_threshold": Max std for consistency (default: 0.5)

        Returns:
            Dictionary with validation results containing:
                - "model_name": Name of the validated model
                - "timestamp": Validation timestamp
                - "energy_conservation": Energy conservation test results
                - "profile_bounds": Profile bounds test results
                - "seasonal_consistency": Seasonal consistency test results
                - "component_analysis": Component analysis results
                - "passed": Overall pass/fail status

        Raises:
            ValueError: If model is not fitted or input data is invalid.

        Example:
            >>> validator = HierarchicalValidator()
            >>> config = {"energy_tolerance": 0.05}
            >>> results = validator.validate_model(model, X_test, y_test, config)
        """
        # Check model is fitted
        if not model._is_fitted:
            msg = "Model must be fitted before validation"
            raise ValueError(msg)

        if config is None:
            config = {}

        logger.info("Starting comprehensive validation for %s", model.name)

        # Convert y to Series if DataFrame
        if isinstance(y, pd.DataFrame):
            if y.shape[1] != 1:
                msg = f"y must be single column, got {y.shape[1]} columns"
                raise ValueError(msg)
            y = y.iloc[:, 0]

        results: dict[str, Any] = {
            "model_name": model.name,
            "timestamp": pd.Timestamp.now(),
            "n_samples": len(X),
        }

        # Test 1: Energy conservation
        logger.info("Running Test 1: Energy conservation")
        try:
            results["energy_conservation"] = self._test_energy_conservation(
                model,
                X,
                threshold=config.get("energy_tolerance", DEFAULT_ENERGY_TOLERANCE),
            )
        except Exception as e:
            logger.error("Energy conservation test failed: %s", e, exc_info=True)
            results["energy_conservation"] = {
                "passed": False,
                "error": str(e),
            }

        # Test 2: Profile bounds
        logger.info("Running Test 2: Profile bounds")
        try:
            results["profile_bounds"] = self._test_profile_bounds(
                model,
                X,
                min_ratio=config.get("profile_min", DEFAULT_PROFILE_MIN),
                max_ratio=config.get("profile_max", DEFAULT_PROFILE_MAX),
            )
        except Exception as e:
            logger.error("Profile bounds test failed: %s", e, exc_info=True)
            results["profile_bounds"] = {
                "passed": False,
                "error": str(e),
            }

        # Test 3: Seasonal consistency
        logger.info("Running Test 3: Seasonal consistency")
        try:
            results["seasonal_consistency"] = self._test_seasonal_consistency(
                model,
                X,
                threshold=config.get("seasonal_std_threshold", DEFAULT_SEASONAL_STD_THRESHOLD),
            )
        except Exception as e:
            logger.error("Seasonal consistency test failed: %s", e, exc_info=True)
            results["seasonal_consistency"] = {
                "passed": False,
                "error": str(e),
            }

        # Test 4: Component analysis
        logger.info("Running Test 4: Component analysis")
        try:
            results["component_analysis"] = self.analyze_components(model, X)
        except Exception as e:
            logger.error("Component analysis failed: %s", e, exc_info=True)
            results["component_analysis"] = {
                "error": str(e),
            }

        # Overall pass/fail
        results["passed"] = all(
            [
                results.get("energy_conservation", {}).get("passed", False),
                results.get("profile_bounds", {}).get("passed", False),
                results.get("seasonal_consistency", {}).get("passed", False),
            ]
        )

        self.validation_results = results

        logger.info(
            "Validation complete: %s",
            "PASSED" if results["passed"] else "FAILED",
        )

        return results

    def _test_energy_conservation(
        self,
        model: BaseHierarchicalModel,
        X: pd.DataFrame,  # noqa: N803
        threshold: float = DEFAULT_ENERGY_TOLERANCE,
    ) -> dict[str, Any]:
        """Test energy conservation in hierarchical model predictions.

        Validates that the sum of 48 semi-hourly predictions equals the demand
        mean × 48 (within tolerance). This ensures energy conservation in the
        hierarchical decomposition.

        Args:
            model: Trained hierarchical model.
            X: Validation features.
            threshold: Maximum allowed relative error (default: 0.05 = 5%).

        Returns:
            Dictionary with test results:
                - "passed": bool, whether test passed
                - "mean_error": float, mean relative error
                - "max_error": float, maximum relative error
                - "threshold": float, threshold used
                - "n_violations": int, number of samples exceeding threshold

        Raises:
            ValueError: If predictions cannot be generated.
        """
        logger.debug("Testing energy conservation with threshold %.1f%%", threshold * 100)

        # Get decomposition
        try:
            decomposition = model.get_decomposition(X)
        except Exception as e:
            msg = f"Failed to get model decomposition: {e}"
            raise ValueError(msg) from e

        demand_mean = decomposition["demand_mean"]
        combined_load = decomposition["combined_load"]

        # Expected daily energy: demand_mean × PERIODS_PER_DAY
        expected_energy = demand_mean * PERIODS_PER_DAY

        # Actual daily energy: sum of all 48 semi-hourly loads
        actual_energy = combined_load.sum(axis=1)

        # Align indices - sometimes profiles have timestamps while demand_mean has dates
        # Convert both to dates for comparison
        if isinstance(demand_mean.index, pd.DatetimeIndex):
            demand_mean_dates = pd.DatetimeIndex(demand_mean.index.date)
        else:
            demand_mean_dates = demand_mean.index

        if isinstance(actual_energy.index, pd.DatetimeIndex):
            actual_energy_dates = pd.DatetimeIndex(actual_energy.index.date)
        else:
            actual_energy_dates = actual_energy.index

        # Reindex to dates
        expected_energy_by_date = pd.Series(expected_energy.values, index=demand_mean_dates)
        actual_energy_by_date = pd.Series(actual_energy.values, index=actual_energy_dates)

        # Find common dates
        common_dates = expected_energy_by_date.index.intersection(actual_energy_by_date.index)
        if len(common_dates) == 0:
            msg = "No common dates between demand_mean and actual_energy"
            raise ValueError(msg)

        expected_energy = expected_energy_by_date.loc[common_dates]
        actual_energy = actual_energy_by_date.loc[common_dates]

        # Calculate relative errors
        relative_error = np.abs(actual_energy - expected_energy) / (expected_energy + 1e-8)

        # Statistics
        mean_error = float(relative_error.mean())
        max_error = float(relative_error.max())
        n_violations = int((relative_error > threshold).sum())
        passed = n_violations == 0

        result = {
            "passed": passed,
            "mean_error": mean_error,
            "max_error": max_error,
            "threshold": threshold,
            "n_violations": n_violations,
            "n_samples": len(common_dates),
        }

        if not passed:
            logger.warning(
                "Energy conservation test FAILED: %d/%d violations "
                "(mean error: %.2f%%, max error: %.2f%%)",
                n_violations,
                len(common_dates),
                mean_error * 100,
                max_error * 100,
            )
        else:
            logger.info(
                "Energy conservation test PASSED: mean error %.2f%%, max error %.2f%%",
                mean_error * 100,
                max_error * 100,
            )

        return result

    def _test_profile_bounds(
        self,
        model: BaseHierarchicalModel,
        X: pd.DataFrame,  # noqa: N803
        min_ratio: float = DEFAULT_PROFILE_MIN,
        max_ratio: float = DEFAULT_PROFILE_MAX,
    ) -> dict[str, Any]:
        """Test that profile ratios stay within reasonable bounds.

        Validates that profile ratios are within [min_ratio, max_ratio] range.
        Excessive violations indicate model instability or data quality issues.

        Args:
            model: Trained hierarchical model.
            X: Validation features.
            min_ratio: Minimum acceptable profile ratio (default: 0.1).
            max_ratio: Maximum acceptable profile ratio (default: 5.0).

        Returns:
            Dictionary with test results:
                - "passed": bool, whether violation rate < 1%
                - "violation_rate": float, percentage of violations
                - "lower_violations": int, count of values below min
                - "upper_violations": int, count of values above max
                - "profile_min": float, minimum profile value observed
                - "profile_max": float, maximum profile value observed

        Raises:
            ValueError: If decomposition fails.
        """
        logger.debug(
            "Testing profile bounds: [%.2f, %.2f]",
            min_ratio,
            max_ratio,
        )

        # Get decomposition
        try:
            decomposition = model.get_decomposition(X)
        except Exception as e:
            msg = f"Failed to get model decomposition: {e}"
            raise ValueError(msg) from e

        profiles = decomposition.get("profiles", {})

        if not profiles:
            logger.warning("No profiles available for bounds testing")
            return {
                "passed": False,
                "error": "No profiles available",
            }

        # Combine all profile ratios into single series
        all_profiles = []
        for period, profile_series in profiles.items():
            all_profiles.append(profile_series)

        if not all_profiles:
            return {
                "passed": False,
                "error": "No profile data available",
            }

        profiles_combined = pd.concat(all_profiles)

        # Check bounds
        lower_violations = int((profiles_combined < min_ratio).sum())
        upper_violations = int((profiles_combined > max_ratio).sum())
        total_violations = lower_violations + upper_violations

        violation_rate = float(total_violations / len(profiles_combined))
        passed = violation_rate < DEFAULT_PROFILE_VIOLATION_THRESHOLD

        result = {
            "passed": passed,
            "violation_rate": violation_rate,
            "lower_violations": lower_violations,
            "upper_violations": upper_violations,
            "total_violations": total_violations,
            "n_samples": len(profiles_combined),
            "profile_min": float(profiles_combined.min()),
            "profile_max": float(profiles_combined.max()),
            "bounds": [min_ratio, max_ratio],
        }

        if not passed:
            logger.warning(
                "Profile bounds test FAILED: %.2f%% violations "
                "(lower: %d, upper: %d)",
                violation_rate * 100,
                lower_violations,
                upper_violations,
            )
        else:
            logger.info(
                "Profile bounds test PASSED: %.2f%% violations within tolerance",
                violation_rate * 100,
            )

        return result

    def _test_seasonal_consistency(
        self,
        model: BaseHierarchicalModel,
        X: pd.DataFrame,  # noqa: N803
        threshold: float = DEFAULT_SEASONAL_STD_THRESHOLD,
    ) -> dict[str, Any]:
        """Test seasonal pattern consistency across weeks.

        Validates that profile patterns are stable across weeks for the same
        day-of-week and hour. High variability may indicate model instability
        or insufficient training data.

        Args:
            model: Trained hierarchical model.
            X: Validation features with DatetimeIndex.
            threshold: Maximum acceptable standard deviation (default: 0.5).

        Returns:
            Dictionary with test results:
                - "passed": bool, whether mean_std < threshold
                - "mean_std": float, mean standard deviation across periods
                - "max_std": float, maximum standard deviation
                - "unstable_periods": int, count of periods with std > 1.0
                - "threshold": float, threshold used

        Raises:
            ValueError: If X doesn't have DatetimeIndex or decomposition fails.
        """
        logger.debug(
            "Testing seasonal consistency with threshold %.2f",
            threshold,
        )

        if not isinstance(X.index, pd.DatetimeIndex):
            msg = "X must have DatetimeIndex for seasonal consistency testing"
            raise ValueError(msg)

        # Get decomposition
        try:
            decomposition = model.get_decomposition(X)
        except Exception as e:
            msg = f"Failed to get model decomposition: {e}"
            raise ValueError(msg) from e

        profiles = decomposition.get("profiles", {})

        if not profiles:
            logger.warning("No profiles available for seasonal consistency testing")
            return {
                "passed": False,
                "error": "No profiles available",
            }

        # Combine all profiles with timestamp information
        profile_records = []
        for period, profile_series in profiles.items():
            for timestamp, value in profile_series.items():
                if pd.notna(value):
                    profile_records.append(
                        {
                            "timestamp": timestamp,
                            "period": period,
                            "profile": value,
                            "dow": timestamp.dayofweek,
                            "hour": period,
                        }
                    )

        if not profile_records:
            return {
                "passed": False,
                "error": "No valid profile data",
            }

        profiles_df = pd.DataFrame(profile_records)

        # Calculate consistency (std across weeks for same dow/hour)
        consistency = profiles_df.groupby(["dow", "hour"])["profile"].std()

        # Handle NaN values (can occur with single observation per group)
        consistency = consistency.fillna(0.0)

        mean_std = float(consistency.mean())
        max_std = float(consistency.max())
        unstable_periods = int((consistency > 1.0).sum())
        passed = mean_std < threshold

        result = {
            "passed": passed,
            "mean_std": mean_std,
            "max_std": max_std,
            "unstable_periods": unstable_periods,
            "threshold": threshold,
            "n_groups": len(consistency),
        }

        if not passed:
            logger.warning(
                "Seasonal consistency test FAILED: mean std %.3f > threshold %.3f",
                mean_std,
                threshold,
            )
        else:
            logger.info(
                "Seasonal consistency test PASSED: mean std %.3f < threshold %.3f",
                mean_std,
                threshold,
            )

        return result

    def analyze_components(
        self,
        model: BaseHierarchicalModel,
        X: pd.DataFrame,  # noqa: N803
    ) -> dict[str, Any]:
        """Analyze individual components of the hierarchical model.

        Provides detailed analysis of model components including demand mean
        and profile contributions, useful for debugging and understanding
        model behavior.

        Args:
            model: Trained hierarchical model.
            X: Features for analysis.

        Returns:
            Dictionary with component analysis:
                - "demand_mean_stats": Statistics on demand mean predictions
                - "profile_stats": Statistics on profile ratios by period
                - "combined_stats": Statistics on final predictions
                - "n_profile_models": Number of trained profile models

        Raises:
            ValueError: If decomposition fails.
        """
        logger.debug("Analyzing model components")

        # Get decomposition
        try:
            decomposition = model.get_decomposition(X)
        except Exception as e:
            msg = f"Failed to get model decomposition: {e}"
            raise ValueError(msg) from e

        demand_mean = decomposition["demand_mean"]
        profiles = decomposition.get("profiles", {})
        combined_load = decomposition["combined_load"]

        # Demand mean statistics
        demand_mean_stats = {
            "mean": float(demand_mean.mean()),
            "std": float(demand_mean.std()),
            "min": float(demand_mean.min()),
            "max": float(demand_mean.max()),
            "n_samples": len(demand_mean),
        }

        # Profile statistics by period
        profile_stats = {}
        for period, profile_series in profiles.items():
            profile_stats[period] = {
                "mean": float(profile_series.mean()),
                "std": float(profile_series.std()),
                "min": float(profile_series.min()),
                "max": float(profile_series.max()),
            }

        # Combined load statistics
        combined_stats = {
            "mean": float(combined_load.values.mean()),
            "std": float(combined_load.values.std()),
            "min": float(combined_load.values.min()),
            "max": float(combined_load.values.max()),
            "shape": combined_load.shape,
        }

        analysis = {
            "demand_mean_stats": demand_mean_stats,
            "profile_stats": profile_stats,
            "combined_stats": combined_stats,
            "n_profile_models": len(model.profile_models),
            "n_profiles_predicted": len(profiles),
        }

        logger.info(
            "Component analysis complete: %d profile models, "
            "demand mean range [%.2f, %.2f]",
            len(model.profile_models),
            demand_mean_stats["min"],
            demand_mean_stats["max"],
        )

        return analysis

    def generate_validation_report(
        self,
        results: dict[str, Any] | None = None,
        output_path: str | Path | None = None,
    ) -> str:
        """Generate formatted validation report.

        Creates a human-readable validation report with all test results,
        statistics, and pass/fail status.

        Args:
            results: Validation results dictionary. If None, uses last validation.
            output_path: Optional path to save the report. If provided, report
                        will be written to this file.

        Returns:
            Formatted validation report as string.

        Raises:
            ValueError: If no validation results are available.

        Example:
            >>> validator = HierarchicalValidator()
            >>> results = validator.validate_model(model, X, y)
            >>> report = validator.generate_validation_report(
            ...     results, "validation_report.txt"
            ... )
            >>> print(report)
        """
        if results is None:
            if not self.validation_results:
                msg = "No validation results available. Run validate_model() first."
                raise ValueError(msg)
            results = self.validation_results

        logger.debug("Generating validation report")

        report = []
        report.append("=" * 70)
        report.append("HIERARCHICAL MODEL VALIDATION REPORT")
        report.append("=" * 70)
        report.append(f"Model: {results['model_name']}")
        report.append(f"Timestamp: {results['timestamp']}")
        report.append(f"Samples: {results.get('n_samples', 'N/A')}")
        report.append(
            f"Overall Status: {'PASSED' if results['passed'] else 'FAILED'}"
        )
        report.append("")

        # Energy conservation
        if "energy_conservation" in results:
            ec = results["energy_conservation"]
            report.append("Test 1: Energy Conservation")
            report.append("-" * 70)
            if "error" in ec:
                report.append(f"  Status: ERROR - {ec['error']}")
            else:
                report.append(f"  Status: {'PASS' if ec['passed'] else 'FAIL'}")
                report.append(f"  Mean Error: {ec['mean_error']:.4f} ({ec['mean_error']*100:.2f}%)")
                report.append(f"  Max Error: {ec['max_error']:.4f} ({ec['max_error']*100:.2f}%)")
                report.append(f"  Threshold: {ec['threshold']:.4f} ({ec['threshold']*100:.2f}%)")
                report.append(f"  Violations: {ec['n_violations']}/{ec.get('n_samples', 'N/A')}")
            report.append("")

        # Profile bounds
        if "profile_bounds" in results:
            pb = results["profile_bounds"]
            report.append("Test 2: Profile Bounds")
            report.append("-" * 70)
            if "error" in pb:
                report.append(f"  Status: ERROR - {pb['error']}")
            else:
                report.append(f"  Status: {'PASS' if pb['passed'] else 'FAIL'}")
                report.append(f"  Violation Rate: {pb['violation_rate']:.4f} ({pb['violation_rate']*100:.2f}%)")
                report.append(f"  Lower Violations: {pb['lower_violations']} (< {pb.get('bounds', [0.1, 5.0])[0]})")
                report.append(f"  Upper Violations: {pb['upper_violations']} (> {pb.get('bounds', [0.1, 5.0])[1]})")
                report.append(f"  Profile Range: [{pb['profile_min']:.3f}, {pb['profile_max']:.3f}]")
            report.append("")

        # Seasonal consistency
        if "seasonal_consistency" in results:
            sc = results["seasonal_consistency"]
            report.append("Test 3: Seasonal Consistency")
            report.append("-" * 70)
            if "error" in sc:
                report.append(f"  Status: ERROR - {sc['error']}")
            else:
                report.append(f"  Status: {'PASS' if sc['passed'] else 'FAIL'}")
                report.append(f"  Mean Std: {sc['mean_std']:.4f}")
                report.append(f"  Max Std: {sc['max_std']:.4f}")
                report.append(f"  Threshold: {sc['threshold']:.4f}")
                report.append(f"  Unstable Periods: {sc['unstable_periods']}")
            report.append("")

        # Component analysis
        if "component_analysis" in results:
            ca = results["component_analysis"]
            report.append("Component Analysis")
            report.append("-" * 70)
            if "error" in ca:
                report.append(f"  ERROR: {ca['error']}")
            else:
                dm = ca.get("demand_mean_stats", {})
                report.append(f"  Demand Mean: mean={dm.get('mean', 0):.2f}, std={dm.get('std', 0):.2f}")
                report.append(f"  Profile Models: {ca.get('n_profile_models', 0)} trained, {ca.get('n_profiles_predicted', 0)} predicted")
                cs = ca.get("combined_stats", {})
                report.append(f"  Combined Load: mean={cs.get('mean', 0):.2f}, std={cs.get('std', 0):.2f}")
            report.append("")

        report.append("=" * 70)

        report_text = "\n".join(report)

        # Save to file if requested
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                f.write(report_text)
            logger.info("Validation report saved to %s", output_path)

        return report_text

    def __repr__(self) -> str:
        """Return string representation of the validator.

        Returns:
            String representation.
        """
        return "HierarchicalValidator()"
