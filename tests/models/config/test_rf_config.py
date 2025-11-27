"""Tests for RandomForestConfig implementation.

This module tests the Random Forest configuration schema including:
- Default configuration values
- Parameter validation and constraints
- Field validators (max_features, feature_selection_method, oob_score)
- to_sklearn_params() conversion method
- Pydantic validation errors
"""

import pytest
from pydantic import ValidationError

from src.models.config.rf_config import RandomForestConfig


class TestRandomForestConfigDefaults:
    """Test default configuration values."""

    def test_default_values(self) -> None:
        """Test that default values are set correctly."""
        config = RandomForestConfig()

        # Tree parameters
        assert config.n_estimators == 100
        assert config.max_depth is None
        assert config.min_samples_split == 2
        assert config.min_samples_leaf == 1
        assert config.max_features == "sqrt"

        # Bootstrap and OOB
        assert config.bootstrap is True
        assert config.oob_score is True

        # Feature selection
        assert config.max_features_per_horizon == 50
        assert config.feature_selection_method == "importance"
        assert config.importance_threshold == 0.01

        # Time series
        assert config.periods_per_day == 48

        # Execution
        assert config.n_jobs == -1
        assert config.random_state == 42


class TestRandomForestConfigValidation:
    """Test parameter validation."""

    def test_n_estimators_bounds(self) -> None:
        """Test n_estimators must be in valid range."""
        # Valid values
        config = RandomForestConfig(n_estimators=1)
        assert config.n_estimators == 1

        config = RandomForestConfig(n_estimators=1000)
        assert config.n_estimators == 1000

        # Invalid values
        with pytest.raises(ValidationError) as exc_info:
            RandomForestConfig(n_estimators=0)
        assert "n_estimators" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            RandomForestConfig(n_estimators=1001)
        assert "n_estimators" in str(exc_info.value)

    def test_max_depth_bounds(self) -> None:
        """Test max_depth validation."""
        # Valid values
        config = RandomForestConfig(max_depth=None)
        assert config.max_depth is None

        config = RandomForestConfig(max_depth=10)
        assert config.max_depth == 10

        # Invalid values
        with pytest.raises(ValidationError) as exc_info:
            RandomForestConfig(max_depth=0)
        assert "max_depth" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            RandomForestConfig(max_depth=101)
        assert "max_depth" in str(exc_info.value)

    def test_min_samples_split_bounds(self) -> None:
        """Test min_samples_split validation."""
        # Valid values
        config = RandomForestConfig(min_samples_split=2)
        assert config.min_samples_split == 2

        config = RandomForestConfig(min_samples_split=100)
        assert config.min_samples_split == 100

        # Invalid values
        with pytest.raises(ValidationError) as exc_info:
            RandomForestConfig(min_samples_split=1)
        assert "min_samples_split" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            RandomForestConfig(min_samples_split=101)
        assert "min_samples_split" in str(exc_info.value)

    def test_min_samples_leaf_bounds(self) -> None:
        """Test min_samples_leaf validation."""
        # Valid values
        config = RandomForestConfig(min_samples_leaf=1)
        assert config.min_samples_leaf == 1

        config = RandomForestConfig(min_samples_leaf=100)
        assert config.min_samples_leaf == 100

        # Invalid values
        with pytest.raises(ValidationError) as exc_info:
            RandomForestConfig(min_samples_leaf=0)
        assert "min_samples_leaf" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            RandomForestConfig(min_samples_leaf=101)
        assert "min_samples_leaf" in str(exc_info.value)

    def test_max_features_string_validation(self) -> None:
        """Test max_features string options."""
        # Valid strings
        config = RandomForestConfig(max_features="sqrt")
        assert config.max_features == "sqrt"

        config = RandomForestConfig(max_features="log2")
        assert config.max_features == "log2"

        # Invalid string
        with pytest.raises(ValidationError) as exc_info:
            RandomForestConfig(max_features="invalid")
        assert "max_features" in str(exc_info.value)

    def test_max_features_int_validation(self) -> None:
        """Test max_features as integer."""
        # Valid int
        config = RandomForestConfig(max_features=10)
        assert config.max_features == 10

        # Invalid int (< 1)
        with pytest.raises(ValidationError) as exc_info:
            RandomForestConfig(max_features=0)
        assert "max_features" in str(exc_info.value)

    def test_max_features_float_validation(self) -> None:
        """Test max_features as float (fraction)."""
        # Valid floats
        config = RandomForestConfig(max_features=0.5)
        assert config.max_features == 0.5

        config = RandomForestConfig(max_features=1.0)
        assert config.max_features == 1.0

        # Invalid floats
        with pytest.raises(ValidationError) as exc_info:
            RandomForestConfig(max_features=0.0)
        assert "max_features" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            RandomForestConfig(max_features=1.1)
        assert "max_features" in str(exc_info.value)

    def test_importance_threshold_bounds(self) -> None:
        """Test importance_threshold validation."""
        # Valid values
        config = RandomForestConfig(importance_threshold=0.0)
        assert config.importance_threshold == 0.0

        config = RandomForestConfig(importance_threshold=0.5)
        assert config.importance_threshold == 0.5

        config = RandomForestConfig(importance_threshold=1.0)
        assert config.importance_threshold == 1.0

        # Invalid values
        with pytest.raises(ValidationError) as exc_info:
            RandomForestConfig(importance_threshold=-0.1)
        assert "importance_threshold" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            RandomForestConfig(importance_threshold=1.1)
        assert "importance_threshold" in str(exc_info.value)

    def test_periods_per_day_bounds(self) -> None:
        """Test periods_per_day validation."""
        # Valid values
        config = RandomForestConfig(periods_per_day=24)  # Hourly
        assert config.periods_per_day == 24

        config = RandomForestConfig(periods_per_day=48)  # 30-min
        assert config.periods_per_day == 48

        config = RandomForestConfig(periods_per_day=96)  # 15-min
        assert config.periods_per_day == 96

        # Invalid values
        with pytest.raises(ValidationError) as exc_info:
            RandomForestConfig(periods_per_day=0)
        assert "periods_per_day" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            RandomForestConfig(periods_per_day=289)
        assert "periods_per_day" in str(exc_info.value)


class TestRandomForestConfigValidators:
    """Test field validators."""

    def test_feature_selection_method_validation(self) -> None:
        """Test feature_selection_method validator."""
        # Valid methods
        config = RandomForestConfig(feature_selection_method="importance")
        assert config.feature_selection_method == "importance"

        config = RandomForestConfig(feature_selection_method="mutual_info")
        assert config.feature_selection_method == "mutual_info"

        # Invalid method
        with pytest.raises(ValidationError) as exc_info:
            RandomForestConfig(feature_selection_method="invalid")
        assert "feature_selection_method" in str(exc_info.value)

    def test_oob_score_bootstrap_consistency(self) -> None:
        """Test that oob_score is set to False when bootstrap is False."""
        # When bootstrap=False, oob_score should be forced to False
        config = RandomForestConfig(bootstrap=False, oob_score=True)
        assert config.bootstrap is False
        assert config.oob_score is False  # Should be auto-corrected

        # When bootstrap=True, oob_score can be True
        config = RandomForestConfig(bootstrap=True, oob_score=True)
        assert config.bootstrap is True
        assert config.oob_score is True


class TestRandomForestConfigMethods:
    """Test configuration methods."""

    def test_to_sklearn_params_default(self) -> None:
        """Test conversion to scikit-learn parameters with defaults."""
        config = RandomForestConfig()
        params = config.to_sklearn_params()

        # Check all required sklearn parameters are present
        assert params["n_estimators"] == 100
        assert params["max_depth"] is None
        assert params["min_samples_split"] == 2
        assert params["min_samples_leaf"] == 1
        assert params["max_features"] == "sqrt"
        assert params["bootstrap"] is True
        assert params["oob_score"] is True
        assert params["n_jobs"] == -1
        assert params["random_state"] == 42
        assert params["verbose"] == 0

    def test_to_sklearn_params_custom(self) -> None:
        """Test conversion with custom parameters."""
        config = RandomForestConfig(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features="log2",
            bootstrap=False,
            oob_score=False,
            random_state=123,
        )
        params = config.to_sklearn_params()

        assert params["n_estimators"] == 200
        assert params["max_depth"] == 15
        assert params["min_samples_split"] == 5
        assert params["min_samples_leaf"] == 2
        assert params["max_features"] == "log2"
        assert params["bootstrap"] is False
        assert params["oob_score"] is False
        assert params["random_state"] == 123

    def test_to_sklearn_params_max_features_variants(self) -> None:
        """Test to_sklearn_params with different max_features types."""
        # String
        config = RandomForestConfig(max_features="sqrt")
        params = config.to_sklearn_params()
        assert params["max_features"] == "sqrt"

        # Integer
        config = RandomForestConfig(max_features=10)
        params = config.to_sklearn_params()
        assert params["max_features"] == 10

        # Float
        config = RandomForestConfig(max_features=0.5)
        params = config.to_sklearn_params()
        assert params["max_features"] == 0.5


class TestRandomForestConfigSerialization:
    """Test model serialization and deserialization."""

    def test_model_dump(self) -> None:
        """Test model_dump() method."""
        config = RandomForestConfig(n_estimators=200, max_depth=15)
        dump = config.model_dump()

        assert isinstance(dump, dict)
        assert dump["n_estimators"] == 200
        assert dump["max_depth"] == 15
        assert "bootstrap" in dump
        assert "oob_score" in dump

    def test_model_dump_json(self) -> None:
        """Test model_dump_json() method."""
        config = RandomForestConfig()
        json_str = config.model_dump_json()

        assert isinstance(json_str, str)
        assert "n_estimators" in json_str
        assert "max_depth" in json_str

    def test_round_trip(self) -> None:
        """Test creating config from dumped dict."""
        config1 = RandomForestConfig(
            n_estimators=150,
            max_depth=20,
            max_features_per_horizon=30,
        )

        # Dump and recreate
        dump = config1.model_dump()
        config2 = RandomForestConfig(**dump)

        assert config2.n_estimators == 150
        assert config2.max_depth == 20
        assert config2.max_features_per_horizon == 30


class TestRandomForestConfigEdgeCases:
    """Test edge cases and special scenarios."""

    def test_extra_fields_forbidden(self) -> None:
        """Test that extra fields are not allowed."""
        with pytest.raises(ValidationError) as exc_info:
            RandomForestConfig(unknown_param=123)
        assert "extra" in str(exc_info.value).lower() or "unknown_param" in str(exc_info.value)

    def test_validate_assignment(self) -> None:
        """Test that assignment validation works."""
        config = RandomForestConfig()

        # Valid assignment
        config.n_estimators = 200
        assert config.n_estimators == 200

        # Invalid assignment should raise error
        with pytest.raises(ValidationError):
            config.n_estimators = 0

    def test_max_features_per_horizon_bounds(self) -> None:
        """Test max_features_per_horizon validation."""
        # Valid values
        config = RandomForestConfig(max_features_per_horizon=1)
        assert config.max_features_per_horizon == 1

        config = RandomForestConfig(max_features_per_horizon=500)
        assert config.max_features_per_horizon == 500

        # Invalid values
        with pytest.raises(ValidationError) as exc_info:
            RandomForestConfig(max_features_per_horizon=0)
        assert "max_features_per_horizon" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            RandomForestConfig(max_features_per_horizon=501)
        assert "max_features_per_horizon" in str(exc_info.value)
