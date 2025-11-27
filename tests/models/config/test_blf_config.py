"""Tests for BLFConfig Pydantic model.

This module tests the BLFConfig configuration schema including:
- Default values
- Validation of correction parameters
- Validation of error model parameters
- Validation of confidence and operational parameters
- Cross-field validation
- Serialization
"""

import pytest
from pydantic import ValidationError

from src.models.config.blf_config import BLFConfig


class TestBLFConfigDefaults:
    """Test default configuration values."""

    def test_default_initialization(self) -> None:
        """Test that BLFConfig initializes with correct defaults."""
        config = BLFConfig()

        # Correction parameters
        assert config.correction_window_hours == 6
        assert config.min_observations == 3
        assert config.error_decay_factor == 0.95
        assert config.max_correction_pct == 20.0

        # Error model parameters
        assert config.use_error_model is True
        assert config.error_model_type == "ridge"
        assert config.error_model_alpha == 1.0

        # Confidence parameters
        assert config.confidence_level == 0.95

        # Operational parameters
        assert config.max_history_days == 7

    def test_custom_initialization(self) -> None:
        """Test initialization with custom parameters."""
        config = BLFConfig(
            correction_window_hours=8,
            min_observations=5,
            error_decay_factor=0.90,
            max_correction_pct=15.0,
            use_error_model=False,
            error_model_type="lasso",
            confidence_level=0.90,
            max_history_days=14,
        )

        assert config.correction_window_hours == 8
        assert config.min_observations == 5
        assert config.error_decay_factor == 0.90
        assert config.max_correction_pct == 15.0
        assert config.use_error_model is False
        assert config.error_model_type == "lasso"
        assert config.confidence_level == 0.90
        assert config.max_history_days == 14


class TestBLFConfigValidation:
    """Test parameter validation."""

    def test_correction_window_hours_validation(self) -> None:
        """Test correction_window_hours bounds validation."""
        # Valid values
        BLFConfig(correction_window_hours=1)
        BLFConfig(correction_window_hours=12)
        BLFConfig(correction_window_hours=24)

        # Invalid values
        with pytest.raises(ValidationError):
            BLFConfig(correction_window_hours=0)

        with pytest.raises(ValidationError):
            BLFConfig(correction_window_hours=25)

        with pytest.raises(ValidationError):
            BLFConfig(correction_window_hours=-1)

    def test_min_observations_validation(self) -> None:
        """Test min_observations bounds validation."""
        # Valid values
        BLFConfig(min_observations=1)
        BLFConfig(min_observations=10)
        BLFConfig(min_observations=100)

        # Invalid values
        with pytest.raises(ValidationError):
            BLFConfig(min_observations=0)

        with pytest.raises(ValidationError):
            BLFConfig(min_observations=-1)

        with pytest.raises(ValidationError):
            BLFConfig(min_observations=101)

    def test_error_decay_factor_validation(self) -> None:
        """Test error_decay_factor bounds validation."""
        # Valid values
        BLFConfig(error_decay_factor=0.0)
        BLFConfig(error_decay_factor=0.5)
        BLFConfig(error_decay_factor=0.95)
        BLFConfig(error_decay_factor=1.0)

        # Invalid values
        with pytest.raises(ValidationError):
            BLFConfig(error_decay_factor=-0.1)

        with pytest.raises(ValidationError):
            BLFConfig(error_decay_factor=1.1)

    def test_max_correction_pct_validation(self) -> None:
        """Test max_correction_pct bounds validation."""
        # Valid values
        BLFConfig(max_correction_pct=0.0)
        BLFConfig(max_correction_pct=20.0)
        BLFConfig(max_correction_pct=100.0)

        # Invalid values
        with pytest.raises(ValidationError):
            BLFConfig(max_correction_pct=-1.0)

        with pytest.raises(ValidationError):
            BLFConfig(max_correction_pct=101.0)

    def test_error_model_type_validation(self) -> None:
        """Test error_model_type literal validation."""
        # Valid values
        BLFConfig(error_model_type="ridge")
        BLFConfig(error_model_type="lasso")

        # Invalid values
        with pytest.raises(ValidationError):
            BLFConfig(error_model_type="invalid")  # type: ignore

        with pytest.raises(ValidationError):
            BLFConfig(error_model_type="elasticnet")  # type: ignore

    def test_error_model_alpha_validation(self) -> None:
        """Test error_model_alpha bounds validation."""
        # Valid values
        BLFConfig(error_model_alpha=0.0)
        BLFConfig(error_model_alpha=1.0)
        BLFConfig(error_model_alpha=1000.0)

        # Invalid values
        with pytest.raises(ValidationError):
            BLFConfig(error_model_alpha=-0.1)

        with pytest.raises(ValidationError):
            BLFConfig(error_model_alpha=1001.0)

    def test_confidence_level_validation(self) -> None:
        """Test confidence_level bounds validation."""
        # Valid values
        BLFConfig(confidence_level=0.5)
        BLFConfig(confidence_level=0.95)
        BLFConfig(confidence_level=0.999)

        # Invalid values
        with pytest.raises(ValidationError):
            BLFConfig(confidence_level=0.4)

        with pytest.raises(ValidationError):
            BLFConfig(confidence_level=1.0)

        with pytest.raises(ValidationError):
            BLFConfig(confidence_level=1.5)

    def test_max_history_days_validation(self) -> None:
        """Test max_history_days bounds validation."""
        # Valid values
        BLFConfig(max_history_days=1)
        BLFConfig(max_history_days=7)
        BLFConfig(max_history_days=30)

        # Invalid values
        with pytest.raises(ValidationError):
            BLFConfig(max_history_days=0)

        with pytest.raises(ValidationError):
            BLFConfig(max_history_days=31)

        with pytest.raises(ValidationError):
            BLFConfig(max_history_days=-1)


class TestBLFConfigCrossFieldValidation:
    """Test cross-field validation logic."""

    def test_error_model_with_low_min_observations(self) -> None:
        """Test warning when error model enabled with low min_observations."""
        # Should create successfully but may log warning
        config = BLFConfig(use_error_model=True, min_observations=2)
        assert config.use_error_model is True
        assert config.min_observations == 2

    def test_error_model_disabled(self) -> None:
        """Test configuration with error model disabled."""
        config = BLFConfig(use_error_model=False, min_observations=2)
        assert config.use_error_model is False
        # No warning should be logged since error model is disabled


class TestBLFConfigSerialization:
    """Test configuration serialization."""

    def test_model_dump(self) -> None:
        """Test Pydantic model_dump method."""
        config = BLFConfig(
            correction_window_hours=8,
            min_observations=5,
            error_model_type="lasso",
            max_history_days=14,
        )

        config_dict = config.model_dump()

        assert isinstance(config_dict, dict)
        assert config_dict["correction_window_hours"] == 8
        assert config_dict["min_observations"] == 5
        assert config_dict["error_model_type"] == "lasso"
        assert config_dict["max_history_days"] == 14

    def test_model_dump_json(self) -> None:
        """Test Pydantic model_dump_json method."""
        config = BLFConfig(correction_window_hours=8)

        config_json = config.model_dump_json()

        assert isinstance(config_json, str)
        assert "8" in config_json
        assert "correction_window_hours" in config_json

    def test_reconstruction_from_dict(self) -> None:
        """Test reconstruction from dictionary."""
        original = BLFConfig(
            correction_window_hours=8,
            min_observations=5,
            error_decay_factor=0.90,
            use_error_model=False,
        )

        # Dump and reconstruct
        config_dict = original.model_dump()
        reconstructed = BLFConfig(**config_dict)

        assert reconstructed.correction_window_hours == original.correction_window_hours
        assert reconstructed.min_observations == original.min_observations
        assert reconstructed.error_decay_factor == original.error_decay_factor
        assert reconstructed.use_error_model == original.use_error_model


class TestBLFConfigMutability:
    """Test configuration modification behavior."""

    def test_config_is_mutable(self) -> None:
        """Test that config can be modified after creation."""
        config = BLFConfig()

        # Should allow modification (frozen=False)
        config.correction_window_hours = 12
        assert config.correction_window_hours == 12

    def test_validate_on_assignment(self) -> None:
        """Test that validation occurs on attribute assignment."""
        config = BLFConfig()

        # Valid assignment
        config.error_decay_factor = 0.85
        assert config.error_decay_factor == 0.85

        # Invalid assignment should raise validation error
        with pytest.raises(ValidationError):
            config.error_decay_factor = 1.5  # > 1.0

    def test_extra_fields_forbidden(self) -> None:
        """Test that extra fields are not allowed."""
        with pytest.raises(ValidationError):
            BLFConfig(extra_field="value")  # type: ignore


class TestBLFConfigEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_minimum_valid_values(self) -> None:
        """Test configuration with minimum valid values."""
        config = BLFConfig(
            correction_window_hours=1,
            min_observations=1,
            error_decay_factor=0.0,
            max_correction_pct=0.0,
            error_model_alpha=0.0,
            confidence_level=0.5,
            max_history_days=1,
        )

        assert config.correction_window_hours == 1
        assert config.min_observations == 1
        assert config.error_decay_factor == 0.0
        assert config.max_correction_pct == 0.0

    def test_maximum_valid_values(self) -> None:
        """Test configuration with maximum valid values."""
        config = BLFConfig(
            correction_window_hours=24,
            min_observations=100,
            error_decay_factor=1.0,
            max_correction_pct=100.0,
            error_model_alpha=1000.0,
            confidence_level=0.999,
            max_history_days=30,
        )

        assert config.correction_window_hours == 24
        assert config.min_observations == 100
        assert config.error_decay_factor == 1.0
        assert config.max_correction_pct == 100.0

    def test_zero_decay_factor(self) -> None:
        """Test with zero decay factor (all observations equally weighted)."""
        config = BLFConfig(error_decay_factor=0.0)
        assert config.error_decay_factor == 0.0

    def test_unity_decay_factor(self) -> None:
        """Test with unity decay factor (no decay)."""
        config = BLFConfig(error_decay_factor=1.0)
        assert config.error_decay_factor == 1.0
