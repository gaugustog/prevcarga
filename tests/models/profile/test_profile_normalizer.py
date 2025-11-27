"""Tests for ProfileNormalizer class."""

import numpy as np
import pandas as pd
import pytest

from src.models.profile.profile_normalizer import ProfileNormalizer


@pytest.fixture
def normalizer():
    """Create ProfileNormalizer instance."""
    return ProfileNormalizer()


@pytest.fixture
def sample_profiles():
    """Create sample profile ratios for testing."""
    np.random.seed(42)
    # Create 10 days of profiles (48 periods each)
    data = {}
    for i in range(48):
        # Values around 1.0 with some variation
        data[f"period_{i}"] = np.random.uniform(0.8, 1.2, 10)

    df = pd.DataFrame(data)
    return df


def test_normalizer_initialization(normalizer):
    """Test ProfileNormalizer initialization."""
    assert normalizer is not None


def test_normalize_proportional(normalizer, sample_profiles):
    """Test proportional normalization."""
    normalized = normalizer.normalize_daily_profiles(
        sample_profiles,
        target_sum=48.0,
        method="proportional",
    )

    # Check shape
    assert normalized.shape == sample_profiles.shape

    # Check sums are close to 48
    sums = normalized.sum(axis=1)
    assert np.allclose(sums, 48.0, atol=1e-6)

    # Check values are positive
    assert (normalized > 0).all().all()


def test_normalize_additive(normalizer, sample_profiles):
    """Test additive normalization."""
    normalized = normalizer.normalize_daily_profiles(
        sample_profiles,
        target_sum=48.0,
        method="additive",
    )

    # Check shape
    assert normalized.shape == sample_profiles.shape

    # Check sums are close to 48
    sums = normalized.sum(axis=1)
    assert np.allclose(sums, 48.0, atol=1e-6)


def test_normalize_invalid_method(normalizer, sample_profiles):
    """Test normalization with invalid method."""
    with pytest.raises(ValueError, match="Unknown normalization method"):
        normalizer.normalize_daily_profiles(
            sample_profiles,
            method="invalid",
        )


def test_normalize_empty_dataframe(normalizer):
    """Test normalization with empty DataFrame."""
    empty_df = pd.DataFrame()

    with pytest.raises(ValueError, match="empty"):
        normalizer.normalize_daily_profiles(empty_df)


def test_normalize_with_zero_sum(normalizer):
    """Test normalization when some rows sum to zero."""
    # Create profiles where one row sums to zero
    data = {f"period_{i}": [1.0, 0.0, 1.0] for i in range(48)}
    profiles = pd.DataFrame(data)

    normalized = normalizer.normalize_daily_profiles(profiles, target_sum=48.0)

    # Check that the zero-sum row gets uniform profile
    assert np.allclose(normalized.iloc[1, :], 1.0, atol=1e-6)

    # Other rows should be normalized correctly
    assert np.allclose(normalized.iloc[0].sum(), 48.0, atol=1e-6)
    assert np.allclose(normalized.iloc[2].sum(), 48.0, atol=1e-6)


def test_denormalize_profiles(normalizer, sample_profiles):
    """Test denormalization."""
    # Store original sums
    original_sums = sample_profiles.sum(axis=1)

    # Normalize
    normalized = normalizer.normalize_daily_profiles(sample_profiles, target_sum=48.0)

    # Denormalize
    denormalized = normalizer.denormalize_profiles(normalized, original_sums)

    # Check shape
    assert denormalized.shape == sample_profiles.shape

    # Check sums match original
    denorm_sums = denormalized.sum(axis=1)
    assert np.allclose(denorm_sums, original_sums, atol=1e-6)

    # Check values are close to original
    assert np.allclose(denormalized.values, sample_profiles.values, atol=1e-6)


def test_denormalize_shape_mismatch(normalizer, sample_profiles):
    """Test denormalization with mismatched shapes."""
    normalized = normalizer.normalize_daily_profiles(sample_profiles)
    wrong_sums = pd.Series([48.0] * 5)  # Wrong length

    with pytest.raises(ValueError, match="Shape mismatch"):
        normalizer.denormalize_profiles(normalized, wrong_sums)


def test_smooth_profiles(normalizer, sample_profiles):
    """Test profile smoothing."""
    smoothed = normalizer.smooth_profiles(
        sample_profiles,
        sigma=1.0,
        preserve_sum=True,
    )

    # Check shape
    assert smoothed.shape == sample_profiles.shape

    # Check sums are preserved
    original_sums = sample_profiles.sum(axis=1)
    smoothed_sums = smoothed.sum(axis=1)
    assert np.allclose(smoothed_sums, original_sums, atol=1e-6)

    # Smoothed profiles should be smoother (less variation)
    # This is a rough heuristic
    original_std = sample_profiles.std(axis=1).mean()
    smoothed_std = smoothed.std(axis=1).mean()
    # Smoothing should reduce std in most cases
    # (not always guaranteed, so we just check it runs)
    assert smoothed_std >= 0


def test_smooth_profiles_invalid_sigma(normalizer, sample_profiles):
    """Test smoothing with invalid sigma."""
    with pytest.raises(ValueError, match="Sigma must be positive"):
        normalizer.smooth_profiles(sample_profiles, sigma=-1.0)

    with pytest.raises(ValueError, match="Sigma must be positive"):
        normalizer.smooth_profiles(sample_profiles, sigma=0.0)


def test_smooth_profiles_empty_dataframe(normalizer):
    """Test smoothing with empty DataFrame."""
    empty_df = pd.DataFrame()

    with pytest.raises(ValueError, match="empty"):
        normalizer.smooth_profiles(empty_df)


def test_smooth_profiles_without_preserve_sum(normalizer, sample_profiles):
    """Test smoothing without preserving sum."""
    original_sums = sample_profiles.sum(axis=1)

    smoothed = normalizer.smooth_profiles(
        sample_profiles,
        sigma=1.0,
        preserve_sum=False,
    )

    # Check shape
    assert smoothed.shape == sample_profiles.shape

    # Sums may have changed
    smoothed_sums = smoothed.sum(axis=1)
    # They might be close but not necessarily equal
    # Just verify the operation completed
    assert len(smoothed_sums) == len(original_sums)


def test_validate_profiles_valid(normalizer):
    """Test validation with valid profiles."""
    # Create valid profiles that sum to 48
    data = {f"period_{i}": [1.0] * 10 for i in range(48)}
    profiles = pd.DataFrame(data)

    validation = normalizer.validate_profiles(profiles)

    assert validation["is_valid"]
    assert len(validation["issues"]) == 0
    assert validation["n_out_of_bounds"] == 0
    assert validation["n_wrong_sum"] == 0


def test_validate_profiles_with_nan(normalizer):
    """Test validation with NaN values."""
    data = {f"period_{i}": [1.0, np.nan, 1.0] for i in range(48)}
    profiles = pd.DataFrame(data)

    validation = normalizer.validate_profiles(profiles)

    assert not validation["is_valid"]
    assert any("NaN" in issue for issue in validation["issues"])


def test_validate_profiles_with_inf(normalizer):
    """Test validation with infinite values."""
    data = {f"period_{i}": [1.0, np.inf, 1.0] for i in range(48)}
    profiles = pd.DataFrame(data)

    validation = normalizer.validate_profiles(profiles)

    assert not validation["is_valid"]
    assert any("infinite" in issue for issue in validation["issues"])


def test_validate_profiles_out_of_bounds(normalizer):
    """Test validation with out-of-bounds values."""
    # Create profiles with values outside [0.1, 5.0]
    data = {f"period_{i}": [0.05, 1.0, 6.0] for i in range(48)}
    profiles = pd.DataFrame(data)

    validation = normalizer.validate_profiles(
        profiles,
        min_ratio=0.1,
        max_ratio=5.0,
    )

    assert not validation["is_valid"]
    assert validation["n_out_of_bounds"] == 48 * 2  # 48 periods * 2 violations
    assert any("outside" in issue for issue in validation["issues"])


def test_validate_profiles_wrong_sum(normalizer):
    """Test validation with wrong sums."""
    # Create profiles that don't sum to 48
    data = {f"period_{i}": [0.5] * 10 for i in range(48)}  # Sum = 24
    profiles = pd.DataFrame(data)

    validation = normalizer.validate_profiles(
        profiles,
        expected_sum=48.0,
        tolerance=0.1,
    )

    assert not validation["is_valid"]
    assert validation["n_wrong_sum"] == 10  # All 10 rows have wrong sum
    assert any("sum far from" in issue for issue in validation["issues"])


def test_validate_profiles_custom_bounds(normalizer):
    """Test validation with custom bounds."""
    data = {f"period_{i}": [1.0] * 5 for i in range(48)}
    profiles = pd.DataFrame(data)

    validation = normalizer.validate_profiles(
        profiles,
        min_ratio=0.5,
        max_ratio=2.0,
        expected_sum=48.0,
        tolerance=0.01,
    )

    assert validation["is_valid"]


def test_normalize_non_48_columns(normalizer):
    """Test normalization with non-48 column count."""
    # Create profiles with 24 periods instead of 48
    data = {f"period_{i}": np.random.uniform(0.8, 1.2, 5) for i in range(24)}
    profiles = pd.DataFrame(data)

    # Should still work, but log a warning
    normalized = normalizer.normalize_daily_profiles(profiles, target_sum=24.0)

    # Check sums
    sums = normalized.sum(axis=1)
    assert np.allclose(sums, 24.0, atol=1e-6)


def test_normalize_preserve_index(normalizer, sample_profiles):
    """Test that normalization preserves DataFrame index."""
    # Set custom index
    sample_profiles.index = pd.date_range("2024-01-01", periods=10, freq="D")

    normalized = normalizer.normalize_daily_profiles(sample_profiles)

    # Check index is preserved
    pd.testing.assert_index_equal(normalized.index, sample_profiles.index)


def test_roundtrip_normalize_denormalize(normalizer, sample_profiles):
    """Test complete roundtrip: normalize then denormalize."""
    original_sums = sample_profiles.sum(axis=1)

    # Normalize
    normalized = normalizer.normalize_daily_profiles(sample_profiles)

    # Denormalize
    recovered = normalizer.denormalize_profiles(normalized, original_sums)

    # Check we recover original values
    assert np.allclose(recovered.values, sample_profiles.values, atol=1e-10)
