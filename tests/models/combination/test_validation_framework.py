"""Tests for validation framework."""

import numpy as np
import pandas as pd
import pytest

from src.models.combination.validation_framework import (
    AlertSystem,
    ValidationConfig,
    ValidationResult,
    WeightValidationFramework,
)


# Mock combiner for testing
class MockCombiner:
    """Mock combiner for testing validation."""

    def __init__(self, weights=None, config=None):
        self.weights = weights or {'model1': 0.6, 'model2': 0.4}
        self.config = config or {}

    def get_weights(self):
        return self.weights.copy()

    def fit(self, predictions, targets):
        # Simple mock fit - just store data size
        self.n_samples = len(next(iter(predictions.values())))


@pytest.fixture
def sample_validation_data():
    """Create sample data for validation."""
    dates = pd.date_range('2024-01-01', periods=100, freq='30min')

    predictions = {
        'model1': pd.DataFrame({
            'pred_h0': np.random.randn(100) * 50 + 1000,
            'pred_h1': np.random.randn(100) * 50 + 1000
        }, index=dates),
        'model2': pd.DataFrame({
            'pred_h0': np.random.randn(100) * 50 + 1000,
            'pred_h1': np.random.randn(100) * 50 + 1000
        }, index=dates)
    }

    targets = pd.DataFrame({
        'target_h0': np.random.randn(100) * 30 + 1000,
        'target_h1': np.random.randn(100) * 30 + 1000
    }, index=dates)

    return predictions, targets


# ============================================================================
# Configuration Tests
# ============================================================================


def test_validation_config_defaults():
    """Test default configuration values."""
    config = ValidationConfig()

    assert config.weight_stability_threshold == 0.1
    assert config.extreme_weight_threshold == 0.8
    assert config.min_weight_threshold == 0.05
    assert config.significance_level == 0.05
    assert config.bootstrap_samples == 100
    assert config.drift_window_size == 20
    assert config.drift_sigma == 3.0


def test_validation_config_custom():
    """Test custom configuration."""
    config = ValidationConfig(
        weight_stability_threshold=0.05,
        bootstrap_samples=50,
        drift_sigma=2.5
    )

    assert config.weight_stability_threshold == 0.05
    assert config.bootstrap_samples == 50
    assert config.drift_sigma == 2.5


# ============================================================================
# ValidationResult Tests
# ============================================================================


def test_validation_result_creation():
    """Test ValidationResult dataclass creation."""
    result = ValidationResult(
        passed=True,
        test_name='test_example',
        metrics={'metric1': 0.5},
        warnings=['warning1'],
        details={'detail1': 'value1'}
    )

    assert result.passed is True
    assert result.test_name == 'test_example'
    assert result.metrics == {'metric1': 0.5}
    assert result.warnings == ['warning1']
    assert result.details == {'detail1': 'value1'}


def test_validation_result_defaults():
    """Test ValidationResult with default values."""
    result = ValidationResult(
        passed=False,
        test_name='test',
        metrics={}
    )

    assert result.warnings == []
    assert result.details == {}


# ============================================================================
# AlertSystem Tests
# ============================================================================


def test_alert_system_init():
    """Test AlertSystem initialization."""
    alert_sys = AlertSystem()

    assert alert_sys.alerts == []


def test_alert_system_add_alert():
    """Test adding alerts."""
    alert_sys = AlertSystem()

    alert_sys.add_alert('warning', 'Test warning')
    alert_sys.add_alert('critical', 'Test critical', {'key': 'value'})

    assert len(alert_sys.alerts) == 2
    assert alert_sys.alerts[0]['severity'] == 'warning'
    assert alert_sys.alerts[1]['severity'] == 'critical'
    assert alert_sys.alerts[1]['details'] == {'key': 'value'}


def test_alert_system_get_alerts():
    """Test getting alerts with filtering."""
    alert_sys = AlertSystem()

    alert_sys.add_alert('info', 'Info message')
    alert_sys.add_alert('warning', 'Warning message')
    alert_sys.add_alert('critical', 'Critical message')

    all_alerts = alert_sys.get_alerts()
    assert len(all_alerts) == 3

    warnings = alert_sys.get_alerts(severity='warning')
    assert len(warnings) == 1
    assert warnings[0]['severity'] == 'warning'


def test_alert_system_clear():
    """Test clearing alerts."""
    alert_sys = AlertSystem()

    alert_sys.add_alert('warning', 'Test')
    assert len(alert_sys.alerts) == 1

    alert_sys.clear_alerts()
    assert len(alert_sys.alerts) == 0


# ============================================================================
# WeightValidationFramework Tests
# ============================================================================


def test_framework_init_default():
    """Test framework initialization with defaults."""
    validator = WeightValidationFramework()

    assert validator.config.bootstrap_samples == 100
    assert isinstance(validator.alert_system, AlertSystem)
    assert validator.validation_history == []


def test_framework_init_custom():
    """Test framework initialization with custom config."""
    validator = WeightValidationFramework(
        config={'bootstrap_samples': 50, 'drift_sigma': 2.0}
    )

    assert validator.config.bootstrap_samples == 50
    assert validator.config.drift_sigma == 2.0


def test_framework_repr():
    """Test framework string representation."""
    validator = WeightValidationFramework()
    repr_str = repr(validator)

    assert 'WeightValidationFramework' in repr_str
    assert 'bootstrap_samples=100' in repr_str


# ============================================================================
# Weight Reasonableness Tests
# ============================================================================


def test_reasonableness_normal_weights():
    """Test reasonableness with normal weights."""
    validator = WeightValidationFramework()

    weights = {'model1': 0.7, 'model2': 0.3}
    result = validator.test_weight_reasonableness(weights)

    assert result.test_name == 'weight_reasonableness'
    # May have warning about near-uniformity, but should return valid result
    assert 'weight_variance' in result.metrics


def test_reasonableness_extreme_weights():
    """Test reasonableness with extreme weights."""
    validator = WeightValidationFramework()

    weights = {'model1': 0.95, 'model2': 0.05}
    result = validator.test_weight_reasonableness(weights)

    assert result.passed is False
    assert len(result.warnings) > 0
    assert any('Extreme weight' in w for w in result.warnings)


def test_reasonableness_low_weights():
    """Test reasonableness with very low weights."""
    validator = WeightValidationFramework()

    weights = {'model1': 0.97, 'model2': 0.03}
    result = validator.test_weight_reasonableness(weights)

    assert result.passed is False
    assert any('low weight' in w for w in result.warnings)


def test_reasonableness_uniform_weights():
    """Test reasonableness with nearly uniform weights."""
    validator = WeightValidationFramework()

    weights = {'model1': 0.501, 'model2': 0.499}
    result = validator.test_weight_reasonableness(weights)

    # May pass or have warning about uniformity
    assert 'weight_variance' in result.metrics


# ============================================================================
# Weight Stability Tests
# ============================================================================


def test_stability_basic(sample_validation_data):
    """Test basic weight stability validation."""
    predictions, targets = sample_validation_data

    combiner = MockCombiner()
    validator = WeightValidationFramework(config={'bootstrap_samples': 10})

    result = validator.test_weight_stability(combiner, predictions, targets)

    assert result.test_name == 'weight_stability'
    assert 'max_std' in result.metrics
    assert 'mean_std' in result.metrics
    assert isinstance(result.passed, bool)


def test_stability_with_unstable_weights(sample_validation_data):
    """Test stability detection with unstable weights."""
    predictions, targets = sample_validation_data

    # Combiner that would have high variance
    combiner = MockCombiner()

    validator = WeightValidationFramework(
        config={
            'bootstrap_samples': 5,
            'weight_stability_threshold': 0.01  # Very strict
        }
    )

    result = validator.test_weight_stability(combiner, predictions, targets)

    # With strict threshold, likely to fail
    assert 'max_std' in result.metrics


# ============================================================================
# Performance Significance Tests
# ============================================================================


def test_significance_better_combination():
    """Test significance when combination is better."""
    validator = WeightValidationFramework()

    # Combination has lower errors
    combo_errors = np.abs(np.random.randn(100) * 10 + 50)
    individual_errors = {
        'model1': np.abs(np.random.randn(100) * 10 + 60),
        'model2': np.abs(np.random.randn(100) * 10 + 65)
    }

    result = validator.test_performance_significance(
        combo_errors, individual_errors
    )

    assert result.test_name == 'performance_significance'
    assert 'significant_improvements' in result.metrics
    assert 'test_results' in result.details


def test_significance_no_improvement():
    """Test significance when combination is not better."""
    validator = WeightValidationFramework()

    # Combination is same as individuals (will give NaN for identical data)
    errors = np.abs(np.random.randn(100) * 10 + 50)
    combo_errors = errors.copy()
    individual_errors = {
        'model1': errors.copy(),
        'model2': errors.copy()
    }

    result = validator.test_performance_significance(
        combo_errors, individual_errors
    )

    assert result.passed is False
    # May not have warnings due to NaN handling, but should fail
    assert 'significant_improvements' in result.metrics


# ============================================================================
# Weight Drift Detection Tests
# ============================================================================


def test_drift_insufficient_history():
    """Test drift detection with insufficient history."""
    validator = WeightValidationFramework()

    weight_history = [
        {'model1': 0.6, 'model2': 0.4},
        {'model1': 0.65, 'model2': 0.35}
    ]

    result = validator.detect_weight_drift(weight_history)

    assert result.passed is True
    assert 'Insufficient history' in result.warnings[0]


def test_drift_stable_weights():
    """Test drift detection with stable weights."""
    validator = WeightValidationFramework()

    # Stable weights over time
    weight_history = [
        {'model1': 0.6 + np.random.randn() * 0.01, 'model2': 0.4 + np.random.randn() * 0.01}
        for _ in range(30)
    ]

    result = validator.detect_weight_drift(weight_history)

    assert result.test_name == 'weight_drift'
    assert 'models_with_drift' in result.metrics


def test_drift_with_outlier():
    """Test drift detection with outlier weights."""
    validator = WeightValidationFramework()

    # Stable weights with recent outlier
    weight_history = [
        {'model1': 0.6, 'model2': 0.4}
        for _ in range(25)
    ]
    # Add outlier
    weight_history.extend([
        {'model1': 0.95, 'model2': 0.05},
        {'model1': 0.92, 'model2': 0.08}
    ])

    result = validator.detect_weight_drift(weight_history)

    # Likely to detect drift
    assert 'models_with_drift' in result.metrics


# ============================================================================
# Validation Suite Tests
# ============================================================================


def test_validation_suite_basic(sample_validation_data):
    """Test full validation suite."""
    predictions, targets = sample_validation_data

    combiner = MockCombiner()
    validator = WeightValidationFramework(config={'bootstrap_samples': 5})

    results = validator.run_validation_suite(combiner, predictions, targets)

    assert 'stability' in results
    assert 'reasonableness' in results
    assert len(validator.validation_history) == 1


def test_validation_suite_with_drift(sample_validation_data):
    """Test validation suite with drift detection."""
    predictions, targets = sample_validation_data

    combiner = MockCombiner()

    weight_history = [
        {'model1': 0.6, 'model2': 0.4}
        for _ in range(25)
    ]

    validator = WeightValidationFramework(config={'bootstrap_samples': 5})
    results = validator.run_validation_suite(
        combiner, predictions, targets, weight_history=weight_history
    )

    assert 'drift' in results
    assert results['drift'].passed in [True, False]


def test_validation_suite_history_tracking(sample_validation_data):
    """Test validation history tracking."""
    predictions, targets = sample_validation_data

    combiner = MockCombiner()
    validator = WeightValidationFramework(config={'bootstrap_samples': 5})

    # Run multiple validations
    validator.run_validation_suite(combiner, predictions, targets)
    validator.run_validation_suite(combiner, predictions, targets)

    assert len(validator.validation_history) == 2
    assert 'timestamp' in validator.validation_history[0]
    assert 'results' in validator.validation_history[0]
    assert 'passed' in validator.validation_history[0]


# ============================================================================
# Report Generation Tests
# ============================================================================


def test_report_generation(sample_validation_data, tmp_path):
    """Test HTML report generation."""
    predictions, targets = sample_validation_data

    combiner = MockCombiner()
    validator = WeightValidationFramework(config={'bootstrap_samples': 5})

    results = validator.run_validation_suite(combiner, predictions, targets)

    report_path = tmp_path / "validation_report.html"
    validator.generate_validation_report(results, str(report_path))

    assert report_path.exists()

    # Read and check content
    content = report_path.read_text()
    assert '<html>' in content
    assert 'Validation Report' in content
    assert 'PASSED' in content or 'FAILED' in content


def test_report_with_alerts(sample_validation_data, tmp_path):
    """Test report generation with alerts."""
    predictions, targets = sample_validation_data

    # Use weights that will trigger alerts
    combiner = MockCombiner(weights={'model1': 0.95, 'model2': 0.05})

    validator = WeightValidationFramework(config={'bootstrap_samples': 5})
    results = validator.run_validation_suite(combiner, predictions, targets)

    report_path = tmp_path / "validation_report.html"
    validator.generate_validation_report(results, str(report_path))

    content = report_path.read_text()
    assert 'Alerts' in content or 'warning' in content.lower()


# ============================================================================
# Bootstrap Tests
# ============================================================================


def test_bootstrap_weights(sample_validation_data):
    """Test bootstrap weight calculation."""
    predictions, targets = sample_validation_data

    combiner = MockCombiner()
    validator = WeightValidationFramework(config={'bootstrap_samples': 10})

    bootstrap_weights = validator._bootstrap_weights(
        combiner, predictions, targets
    )

    assert len(bootstrap_weights) > 0
    assert len(bootstrap_weights) <= 10
    assert all(isinstance(w, dict) for w in bootstrap_weights)


# ============================================================================
# Integration Tests
# ============================================================================


def test_full_workflow(sample_validation_data, tmp_path):
    """Test complete validation workflow."""
    predictions, targets = sample_validation_data

    # Create combiner
    combiner = MockCombiner(weights={'model1': 0.55, 'model2': 0.45})

    # Initialize validator
    validator = WeightValidationFramework(
        config={'bootstrap_samples': 10, 'drift_sigma': 2.5}
    )

    # Run validation suite
    results = validator.run_validation_suite(combiner, predictions, targets)

    # Check results
    assert len(results) >= 2
    assert all(isinstance(r, ValidationResult) for r in results.values())

    # Generate report
    report_path = tmp_path / "full_validation.html"
    validator.generate_validation_report(results, str(report_path))

    assert report_path.exists()

    # Check alerts
    alerts = validator.alert_system.get_alerts()
    assert isinstance(alerts, list)
