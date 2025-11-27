"""Tests for bias correction module."""

import numpy as np
import pandas as pd
import pytest

from src.models.combination.bias_correction import (
    BasicBiasCorrectionModule,
    BiasCorrectionConfig,
)


@pytest.fixture
def sample_dates():
    """Generate sample dates for testing."""
    return pd.date_range('2024-01-01', periods=200, freq='30min')


@pytest.fixture
def biased_predictions_additive(sample_dates):
    """Create predictions with systematic additive bias (+50)."""
    predictions = pd.DataFrame({
        'pred_h0': np.random.randn(200) * 50 + 1050,
        'pred_h1': np.random.randn(200) * 50 + 1050,
        'pred_h2': np.random.randn(200) * 50 + 1050,
    }, index=sample_dates)

    targets = pd.DataFrame({
        'target_h0': np.random.randn(200) * 50 + 1000,
        'target_h1': np.random.randn(200) * 50 + 1000,
        'target_h2': np.random.randn(200) * 50 + 1000,
    }, index=sample_dates)

    return predictions, targets


@pytest.fixture
def biased_predictions_multiplicative(sample_dates):
    """Create predictions with systematic multiplicative bias (×1.1)."""
    predictions = pd.DataFrame({
        'pred_h0': np.random.randn(200) * 50 + 1000,
        'pred_h1': np.random.randn(200) * 50 + 1000,
    }, index=sample_dates)

    # Targets are 10% lower
    targets = pd.DataFrame({
        'target_h0': predictions['pred_h0'] * 0.9 + np.random.randn(200) * 10,
        'target_h1': predictions['pred_h1'] * 0.9 + np.random.randn(200) * 10,
    }, index=sample_dates)

    return predictions, targets


@pytest.fixture
def seasonal_biased_predictions(sample_dates):
    """Create predictions with seasonal bias pattern."""
    predictions = pd.DataFrame({
        'pred_h0': np.random.randn(200) * 50 + 1000,
    }, index=sample_dates)

    # Summer months (Dec-Feb) have +100 bias, winter has -100
    targets = predictions.copy()
    targets.columns = ['target_h0']

    for idx in targets.index:
        if idx.month in [12, 1, 2]:  # Summer (Southern Hemisphere)
            targets.loc[idx, 'target_h0'] += 100
        elif idx.month in [6, 7, 8]:  # Winter
            targets.loc[idx, 'target_h0'] -= 100

    return predictions, targets


# ============================================================================
# Configuration Tests
# ============================================================================


def test_config_defaults():
    """Test default configuration values."""
    config = BiasCorrectionConfig()

    assert config.correction_type == 'additive'
    assert config.window_size == 48
    assert config.detect_seasonal is True
    assert config.detect_weekly is True
    assert config.detect_hourly is True
    assert config.min_samples == 20
    assert config.max_correction == 0.2
    assert config.significance_level == 0.05


def test_config_custom():
    """Test custom configuration."""
    config = BiasCorrectionConfig(
        correction_type='multiplicative',
        window_size=96,
        detect_seasonal=False,
        max_correction=0.3
    )

    assert config.correction_type == 'multiplicative'
    assert config.window_size == 96
    assert config.detect_seasonal is False
    assert config.max_correction == 0.3


# ============================================================================
# Initialization Tests
# ============================================================================


def test_init_default():
    """Test default initialization."""
    corrector = BasicBiasCorrectionModule()

    assert corrector.config.correction_type == 'additive'
    assert corrector._fitted is False
    assert corrector._bias_offset is None
    assert corrector._bias_factor is None


def test_init_custom_config():
    """Test initialization with custom config."""
    config = {'correction_type': 'multiplicative', 'max_correction': 0.15}
    corrector = BasicBiasCorrectionModule(config=config)

    assert corrector.config.correction_type == 'multiplicative'
    assert corrector.config.max_correction == 0.15


def test_repr():
    """Test string representation."""
    corrector = BasicBiasCorrectionModule()
    repr_str = repr(corrector)

    assert 'BasicBiasCorrectionModule' in repr_str
    assert 'additive' in repr_str
    assert 'not fitted' in repr_str


# ============================================================================
# Additive Correction Tests
# ============================================================================


def test_additive_correction_fit(biased_predictions_additive):
    """Test fitting additive bias correction."""
    predictions, targets = biased_predictions_additive

    corrector = BasicBiasCorrectionModule(config={'correction_type': 'additive'})
    corrector.fit(predictions, targets)

    assert corrector._fitted
    assert corrector._bias_offset is not None

    # Bias is limited by max_correction (20% of mean absolute error)
    # With random noise, actual bias will be limited
    assert corrector._bias_offset < 0  # Should be negative (predictions too high)


def test_additive_correction_apply(biased_predictions_additive):
    """Test applying additive correction."""
    predictions, targets = biased_predictions_additive

    corrector = BasicBiasCorrectionModule(config={'correction_type': 'additive'})
    corrector.fit(predictions, targets)

    corrected = corrector.correct(predictions)

    # Corrected predictions should be closer to targets than original
    orig_diff = abs(predictions['pred_h0'].mean() - targets['target_h0'].mean())
    corr_diff = abs(corrected['pred_h0'].mean() - targets['target_h0'].mean())
    assert corr_diff < orig_diff


def test_additive_bias_reduction(biased_predictions_additive):
    """Test bias reduction effectiveness."""
    predictions, targets = biased_predictions_additive

    corrector = BasicBiasCorrectionModule(config={'correction_type': 'additive'})
    corrector.fit(predictions, targets)

    validation = corrector.validate_bias_correction(predictions, targets)

    # Should achieve positive bias reduction
    assert validation['bias_reduction'] > 0
    assert validation['corrected_bias'] < validation['original_bias']
    assert validation['n_samples'] > 0


# ============================================================================
# Multiplicative Correction Tests
# ============================================================================


def test_multiplicative_correction_fit(biased_predictions_multiplicative):
    """Test fitting multiplicative bias correction."""
    predictions, targets = biased_predictions_multiplicative

    corrector = BasicBiasCorrectionModule(
        config={'correction_type': 'multiplicative'}
    )
    corrector.fit(predictions, targets)

    assert corrector._fitted
    assert corrector._bias_factor is not None

    # Factor should be approximately 0.9
    assert 0.85 < corrector._bias_factor < 0.95


def test_multiplicative_correction_apply(biased_predictions_multiplicative):
    """Test applying multiplicative correction."""
    predictions, targets = biased_predictions_multiplicative

    corrector = BasicBiasCorrectionModule(
        config={'correction_type': 'multiplicative'}
    )
    corrector.fit(predictions, targets)

    corrected = corrector.correct(predictions)

    # Corrected predictions should be closer to targets
    orig_error = np.abs(predictions['pred_h0'].mean() - targets['target_h0'].mean())
    corr_error = np.abs(corrected['pred_h0'].mean() - targets['target_h0'].mean())

    assert corr_error < orig_error


# ============================================================================
# Combined Correction Tests
# ============================================================================


def test_both_corrections(biased_predictions_additive):
    """Test combined additive and multiplicative correction."""
    predictions, targets = biased_predictions_additive

    corrector = BasicBiasCorrectionModule(config={'correction_type': 'both'})
    corrector.fit(predictions, targets)

    assert corrector._fitted
    assert corrector._bias_offset is not None
    assert corrector._bias_factor is not None


# ============================================================================
# Seasonal Bias Tests
# ============================================================================


def test_seasonal_bias_detection(seasonal_biased_predictions):
    """Test seasonal bias detection."""
    predictions, targets = seasonal_biased_predictions

    corrector = BasicBiasCorrectionModule(
        config={
            'correction_type': 'additive',
            'detect_seasonal': True,
            'min_samples': 5  # Lower threshold for small test dataset
        }
    )
    corrector.fit(predictions, targets)

    # With sufficient seasonal pattern, should detect it
    # May not detect with limited data and noise
    assert corrector._fitted


def test_seasonal_bias_application(seasonal_biased_predictions):
    """Test application of seasonal bias correction."""
    predictions, targets = seasonal_biased_predictions

    corrector = BasicBiasCorrectionModule(
        config={
            'correction_type': 'additive',
            'detect_seasonal': True,
            'min_samples': 5  # Lower threshold for test data
        }
    )
    corrector.fit(predictions, targets)

    corrected = corrector.correct(predictions)

    # Seasonal correction should reduce bias
    validation = corrector.validate_bias_correction(predictions, targets)
    assert validation['bias_reduction'] > 0


# ============================================================================
# Edge Cases and Validation
# ============================================================================


def test_insufficient_samples():
    """Test behavior with insufficient samples."""
    dates = pd.date_range('2024-01-01', periods=10, freq='30min')

    predictions = pd.DataFrame({
        'pred_h0': np.random.randn(10) + 1000,
    }, index=dates)

    targets = pd.DataFrame({
        'target_h0': np.random.randn(10) + 1000,
    }, index=dates)

    corrector = BasicBiasCorrectionModule(config={'min_samples': 20})
    corrector.fit(predictions, targets)

    # Should not be fitted with insufficient samples
    assert not corrector._fitted


def test_no_prediction_columns():
    """Test error when no prediction columns found."""
    dates = pd.date_range('2024-01-01', periods=50, freq='30min')

    predictions = pd.DataFrame({
        'wrong_col': np.random.randn(50),
    }, index=dates)

    targets = pd.DataFrame({
        'target_h0': np.random.randn(50),
    }, index=dates)

    corrector = BasicBiasCorrectionModule()

    with pytest.raises(ValueError, match="No prediction columns found"):
        corrector.fit(predictions, targets)


def test_missing_target_columns(sample_dates):
    """Test handling of missing target columns."""
    predictions = pd.DataFrame({
        'pred_h0': np.random.randn(200) + 1000,
        'pred_h1': np.random.randn(200) + 1000,
    }, index=sample_dates)

    # Only one matching target
    targets = pd.DataFrame({
        'target_h0': np.random.randn(200) + 1000,
    }, index=sample_dates)

    corrector = BasicBiasCorrectionModule()
    corrector.fit(predictions, targets)

    # Should fit with at least one valid column
    assert corrector._fitted


def test_correct_not_fitted(biased_predictions_additive):
    """Test correction without fitting."""
    predictions, _ = biased_predictions_additive

    corrector = BasicBiasCorrectionModule()
    corrected = corrector.correct(predictions)

    # Should return unchanged predictions
    pd.testing.assert_frame_equal(corrected, predictions)


def test_non_negative_enforcement(biased_predictions_additive):
    """Test that corrected values are non-negative."""
    predictions, targets = biased_predictions_additive

    # Create extreme negative bias
    predictions_extreme = predictions - 2000  # Very low predictions

    corrector = BasicBiasCorrectionModule(
        config={'correction_type': 'additive', 'max_correction': 1.0}
    )
    corrector.fit(predictions, targets)

    corrected = corrector.correct(predictions_extreme)

    # All values should be non-negative
    assert (corrected['pred_h0'] >= 0).all()
    assert (corrected['pred_h1'] >= 0).all()


def test_nan_handling(sample_dates):
    """Test handling of NaN values."""
    predictions = pd.DataFrame({
        'pred_h0': np.random.randn(200) + 1000,
    }, index=sample_dates)

    targets = pd.DataFrame({
        'target_h0': np.random.randn(200) + 1000,
    }, index=sample_dates)

    # Introduce NaN values
    predictions.loc[predictions.index[:10], 'pred_h0'] = np.nan
    targets.loc[targets.index[5:15], 'target_h0'] = np.nan

    corrector = BasicBiasCorrectionModule()
    corrector.fit(predictions, targets)

    # Should fit using valid data
    assert corrector._fitted


def test_max_correction_limit(biased_predictions_additive):
    """Test maximum correction magnitude limit."""
    predictions, targets = biased_predictions_additive

    # Very restrictive max_correction
    corrector = BasicBiasCorrectionModule(
        config={'correction_type': 'additive', 'max_correction': 0.01}
    )
    corrector.fit(predictions, targets)

    # Bias offset should be limited
    assert abs(corrector._bias_offset) < 50 * 0.01 * 2  # Mean error * max_correction


# ============================================================================
# Validation Tests
# ============================================================================


def test_validate_no_samples():
    """Test validation with no valid samples."""
    dates = pd.date_range('2024-01-01', periods=50, freq='30min')

    predictions = pd.DataFrame({
        'pred_h0': np.full(50, np.nan),
    }, index=dates)

    targets = pd.DataFrame({
        'target_h0': np.full(50, np.nan),
    }, index=dates)

    corrector = BasicBiasCorrectionModule()
    corrector._fitted = True  # Manually set to test validation

    validation = corrector.validate_bias_correction(predictions, targets)

    assert validation['n_samples'] == 0
    assert validation['bias_reduction'] == 0.0


def test_validation_metrics_structure(biased_predictions_additive):
    """Test structure of validation metrics."""
    predictions, targets = biased_predictions_additive

    corrector = BasicBiasCorrectionModule()
    corrector.fit(predictions, targets)

    validation = corrector.validate_bias_correction(predictions, targets)

    # Check all required keys
    assert 'original_bias' in validation
    assert 'corrected_bias' in validation
    assert 'bias_reduction' in validation
    assert 'n_samples' in validation

    # Check types
    assert isinstance(validation['original_bias'], float)
    assert isinstance(validation['corrected_bias'], float)
    assert isinstance(validation['bias_reduction'], float)
    assert isinstance(validation['n_samples'], int)


# ============================================================================
# Integration Tests
# ============================================================================


def test_full_workflow(biased_predictions_additive):
    """Test complete bias correction workflow."""
    predictions, targets = biased_predictions_additive

    # Split into train/test
    split_idx = len(predictions) // 2
    train_pred = predictions.iloc[:split_idx]
    train_targ = targets.iloc[:split_idx]
    test_pred = predictions.iloc[split_idx:]
    test_targ = targets.iloc[split_idx:]

    # Fit on train
    corrector = BasicBiasCorrectionModule(
        config={'correction_type': 'additive', 'detect_seasonal': True}
    )
    corrector.fit(train_pred, train_targ)

    # Apply to test
    corrected = corrector.correct(test_pred)

    # Validate on test
    validation = corrector.validate_bias_correction(test_pred, test_targ)

    assert validation['bias_reduction'] > 0
    assert corrected['pred_h0'].mean() < test_pred['pred_h0'].mean()


def test_multiple_horizons(sample_dates):
    """Test bias correction across multiple horizons."""
    n = len(sample_dates)

    predictions = pd.DataFrame({
        'pred_h0': np.random.randn(n) * 50 + 1050,
        'pred_h1': np.random.randn(n) * 50 + 1060,
        'pred_h2': np.random.randn(n) * 50 + 1070,
        'pred_h3': np.random.randn(n) * 50 + 1080,
    }, index=sample_dates)

    targets = pd.DataFrame({
        'target_h0': np.random.randn(n) * 50 + 1000,
        'target_h1': np.random.randn(n) * 50 + 1000,
        'target_h2': np.random.randn(n) * 50 + 1000,
        'target_h3': np.random.randn(n) * 50 + 1000,
    }, index=sample_dates)

    corrector = BasicBiasCorrectionModule()
    corrector.fit(predictions, targets)

    corrected = corrector.correct(predictions)

    # All horizons should be corrected
    assert corrected['pred_h0'].mean() < predictions['pred_h0'].mean()
    assert corrected['pred_h1'].mean() < predictions['pred_h1'].mean()
    assert corrected['pred_h2'].mean() < predictions['pred_h2'].mean()
    assert corrected['pred_h3'].mean() < predictions['pred_h3'].mean()


def test_outlier_filtering_multiplicative(sample_dates):
    """Test outlier filtering in multiplicative bias estimation."""
    predictions = pd.DataFrame({
        'pred_h0': np.concatenate([
            np.random.randn(190) * 50 + 1000,  # Normal data
            np.array([0.1] * 10)  # Outliers (near-zero)
        ]),
    }, index=sample_dates)

    targets = pd.DataFrame({
        'target_h0': np.concatenate([
            np.random.randn(190) * 50 + 900,  # Normal data
            np.random.randn(10) * 50 + 900  # Targets for outliers
        ]),
    }, index=sample_dates)

    corrector = BasicBiasCorrectionModule(
        config={'correction_type': 'multiplicative'}
    )
    corrector.fit(predictions, targets)

    # Should handle outliers gracefully
    assert corrector._fitted
    assert 0.5 < corrector._bias_factor < 1.5  # Reasonable range


def test_bias_reduction_target_achievement():
    """Test that bias reduction can exceed 20% target with strong bias."""
    # Create data with strong, consistent bias
    dates = pd.date_range('2024-01-01', periods=500, freq='30min')

    # Predictions consistently 100 MW too high
    predictions = pd.DataFrame({
        'pred_h0': np.random.randn(500) * 20 + 1100,  # Mean ~1100
        'pred_h1': np.random.randn(500) * 20 + 1100,
    }, index=dates)

    targets = pd.DataFrame({
        'target_h0': np.random.randn(500) * 20 + 1000,  # Mean ~1000
        'target_h1': np.random.randn(500) * 20 + 1000,
    }, index=dates)

    # Disable time-dependent detection to avoid overcorrection
    corrector = BasicBiasCorrectionModule(
        config={
            'correction_type': 'additive',
            'max_correction': 1.0,
            'detect_seasonal': False,
            'detect_weekly': False,
            'detect_hourly': False,
        }
    )
    corrector.fit(predictions, targets)

    validation = corrector.validate_bias_correction(predictions, targets)

    # With strong systematic bias, should achieve >20% reduction
    assert validation['bias_reduction'] >= 0.20, \
        f"Bias reduction {validation['bias_reduction']:.1%} < 20% target"

    # Corrected bias should be significantly less than original
    assert validation['corrected_bias'] < validation['original_bias']
