"""Tests for LGBMConfig Pydantic model.

This module tests the LGBMConfig configuration schema including:
- Default values
- Validation of hyperparameters
- Conversion to LightGBM parameters
- Optuna search space generation
- Cross-field validation
"""

import pytest
from pydantic import ValidationError

from src.models.config.lgbm_config import LGBMConfig


class TestLGBMConfigDefaults:
    """Test default configuration values."""

    def test_default_initialization(self) -> None:
        """Test that LGBMConfig initializes with correct defaults."""
        config = LGBMConfig()

        # Boosting parameters
        assert config.n_estimators == 1000
        assert config.learning_rate == 0.05
        assert config.num_leaves == 31
        assert config.max_depth == -1
        assert config.min_child_samples == 20

        # Subsampling
        assert config.feature_fraction == 0.8
        assert config.bagging_fraction == 0.8
        assert config.bagging_freq == 5

        # Regularization
        assert config.lambda_l1 == 0.0
        assert config.lambda_l2 == 0.0

        # Early stopping
        assert config.early_stopping_rounds == 50

        # Features
        assert config.categorical_features == []

        # Optuna
        assert config.optimize_hyperparams is True
        assert config.n_trials == 100
        assert config.cv_folds == 5

        # Model behavior
        assert config.objective == "regression"
        assert config.metric == "mae"
        assert config.verbosity == -1
        assert config.n_jobs == -1
        assert config.random_state == 42

    def test_custom_initialization(self) -> None:
        """Test initialization with custom parameters."""
        config = LGBMConfig(
            n_estimators=500,
            learning_rate=0.1,
            num_leaves=50,
            max_depth=8,
            optimize_hyperparams=False,
        )

        assert config.n_estimators == 500
        assert config.learning_rate == 0.1
        assert config.num_leaves == 50
        assert config.max_depth == 8
        assert config.optimize_hyperparams is False


class TestLGBMConfigValidation:
    """Test parameter validation."""

    def test_n_estimators_validation(self) -> None:
        """Test n_estimators bounds validation."""
        # Valid values
        LGBMConfig(n_estimators=1)
        LGBMConfig(n_estimators=1000)
        LGBMConfig(n_estimators=10000)

        # Invalid values
        with pytest.raises(ValidationError):
            LGBMConfig(n_estimators=0)

        with pytest.raises(ValidationError):
            LGBMConfig(n_estimators=-1)

        with pytest.raises(ValidationError):
            LGBMConfig(n_estimators=10001)

    def test_learning_rate_validation(self) -> None:
        """Test learning_rate bounds validation."""
        # Valid values
        LGBMConfig(learning_rate=0.001)
        LGBMConfig(learning_rate=0.1)
        LGBMConfig(learning_rate=1.0)

        # Invalid values
        with pytest.raises(ValidationError):
            LGBMConfig(learning_rate=0.0)

        with pytest.raises(ValidationError):
            LGBMConfig(learning_rate=-0.1)

        with pytest.raises(ValidationError):
            LGBMConfig(learning_rate=1.5)

    def test_num_leaves_validation(self) -> None:
        """Test num_leaves bounds validation."""
        # Valid values
        LGBMConfig(num_leaves=2)
        LGBMConfig(num_leaves=31)
        LGBMConfig(num_leaves=1024)

        # Invalid values
        with pytest.raises(ValidationError):
            LGBMConfig(num_leaves=1)

        with pytest.raises(ValidationError):
            LGBMConfig(num_leaves=2000)

    def test_max_depth_validation(self) -> None:
        """Test max_depth bounds validation."""
        # Valid values
        LGBMConfig(max_depth=-1)  # No limit
        LGBMConfig(max_depth=5)
        LGBMConfig(max_depth=100)

        # Invalid values
        with pytest.raises(ValidationError):
            LGBMConfig(max_depth=-2)

        with pytest.raises(ValidationError):
            LGBMConfig(max_depth=101)

    def test_fraction_validation(self) -> None:
        """Test fraction parameters (0, 1] validation."""
        # Valid values
        LGBMConfig(feature_fraction=0.5, bagging_fraction=0.8)
        LGBMConfig(feature_fraction=1.0, bagging_fraction=1.0)

        # Invalid values
        with pytest.raises(ValidationError):
            LGBMConfig(feature_fraction=0.0)

        with pytest.raises(ValidationError):
            LGBMConfig(feature_fraction=1.1)

        with pytest.raises(ValidationError):
            LGBMConfig(bagging_fraction=-0.1)

    def test_lambda_validation(self) -> None:
        """Test regularization lambda validation."""
        # Valid values
        LGBMConfig(lambda_l1=0.0, lambda_l2=0.0)
        LGBMConfig(lambda_l1=5.0, lambda_l2=10.0)
        LGBMConfig(lambda_l1=100.0, lambda_l2=100.0)

        # Invalid values
        with pytest.raises(ValidationError):
            LGBMConfig(lambda_l1=-1.0)

        with pytest.raises(ValidationError):
            LGBMConfig(lambda_l2=101.0)

    def test_objective_validation(self) -> None:
        """Test objective function validation."""
        # Valid objectives
        valid_objectives = [
            "regression",
            "regression_l1",
            "regression_l2",
            "huber",
            "fair",
            "poisson",
            "quantile",
            "mape",
        ]

        for objective in valid_objectives:
            config = LGBMConfig(objective=objective)
            assert config.objective == objective

        # Invalid objective
        with pytest.raises(ValidationError, match="not supported"):
            LGBMConfig(objective="invalid_objective")

    def test_metric_validation(self) -> None:
        """Test metric validation."""
        # Valid metrics
        valid_metrics = [
            "mae",
            "mse",
            "rmse",
            "mape",
            "huber",
            "fair",
            "poisson",
            "quantile",
        ]

        for metric in valid_metrics:
            config = LGBMConfig(metric=metric)
            assert config.metric == metric

        # Invalid metric
        with pytest.raises(ValidationError, match="not supported"):
            LGBMConfig(metric="invalid_metric")


class TestLGBMConfigCrossFieldValidation:
    """Test cross-field validation logic."""

    def test_num_leaves_max_depth_validation(self) -> None:
        """Test num_leaves vs max_depth validation."""
        # Valid: num_leaves <= 2^max_depth
        LGBMConfig(num_leaves=16, max_depth=4)  # 16 <= 2^4
        LGBMConfig(num_leaves=32, max_depth=5)  # 32 <= 2^5
        LGBMConfig(num_leaves=100, max_depth=-1)  # No limit when max_depth=-1

        # Invalid: num_leaves > 2^max_depth
        with pytest.raises(ValidationError, match="num_leaves.*should be"):
            LGBMConfig(num_leaves=100, max_depth=4)  # 100 > 2^4=16

    def test_bagging_warning_logic(self) -> None:
        """Test bagging configuration consistency (logs warning, doesn't fail)."""
        # This should log a warning but not fail
        # bagging_freq > 0 but bagging_fraction == 1.0 means bagging is disabled
        config = LGBMConfig(bagging_freq=5, bagging_fraction=1.0)
        assert config.bagging_freq == 5
        assert config.bagging_fraction == 1.0

        # Proper bagging configuration (no warning)
        config = LGBMConfig(bagging_freq=5, bagging_fraction=0.8)
        assert config.bagging_freq == 5
        assert config.bagging_fraction == 0.8


class TestLGBMConfigMethods:
    """Test configuration methods."""

    def test_to_lgbm_params(self) -> None:
        """Test conversion to LightGBM parameter dictionary."""
        config = LGBMConfig(
            n_estimators=500,
            learning_rate=0.1,
            num_leaves=50,
            max_depth=8,
            objective="regression",
            metric="mae",
        )

        params = config.to_lgbm_params()

        # Check key parameters are present
        assert params["objective"] == "regression"
        assert params["metric"] == "mae"
        assert params["num_leaves"] == 50
        assert params["max_depth"] == 8
        assert params["learning_rate"] == 0.1
        assert params["n_estimators"] == 500
        assert params["random_state"] == 42
        assert params["verbosity"] == -1
        assert params["force_col_wise"] is True

        # Check subsample/colsample mapping
        assert params["subsample"] == config.bagging_fraction
        assert params["subsample_freq"] == config.bagging_freq
        assert params["colsample_bytree"] == config.feature_fraction

        # Check regularization mapping
        assert params["reg_alpha"] == config.lambda_l1
        assert params["reg_lambda"] == config.lambda_l2

    def test_to_lgbm_params_with_categorical(self) -> None:
        """Test to_lgbm_params includes categorical features."""
        config = LGBMConfig(categorical_features=["hour", "day_of_week", "month"])

        params = config.to_lgbm_params()

        assert "categorical_feature" in params
        assert params["categorical_feature"] == ["hour", "day_of_week", "month"]

    def test_to_lgbm_params_without_categorical(self) -> None:
        """Test to_lgbm_params without categorical features."""
        config = LGBMConfig(categorical_features=[])

        params = config.to_lgbm_params()

        # Should not include categorical_feature key when empty
        assert "categorical_feature" not in params

    def test_get_optuna_search_space(self) -> None:
        """Test Optuna search space generation."""
        config = LGBMConfig()
        search_space = config.get_optuna_search_space()

        # Check expected parameters are in search space
        expected_params = [
            "learning_rate",
            "num_leaves",
            "max_depth",
            "min_child_samples",
            "feature_fraction",
            "bagging_fraction",
            "bagging_freq",
            "lambda_l1",
            "lambda_l2",
        ]

        for param in expected_params:
            assert param in search_space
            assert isinstance(search_space[param], tuple)
            assert len(search_space[param]) == 2
            assert search_space[param][0] <= search_space[param][1]

        # Check specific ranges
        assert search_space["learning_rate"] == (0.01, 0.3)
        assert search_space["num_leaves"] == (20, 100)
        assert search_space["max_depth"] == (3, 12)


class TestLGBMConfigSerialization:
    """Test configuration serialization."""

    def test_model_dump(self) -> None:
        """Test Pydantic model_dump method."""
        config = LGBMConfig(
            n_estimators=500,
            learning_rate=0.1,
            categorical_features=["hour", "day"],
        )

        config_dict = config.model_dump()

        assert isinstance(config_dict, dict)
        assert config_dict["n_estimators"] == 500
        assert config_dict["learning_rate"] == 0.1
        assert config_dict["categorical_features"] == ["hour", "day"]

    def test_model_dump_json(self) -> None:
        """Test Pydantic model_dump_json method."""
        config = LGBMConfig(n_estimators=500)

        config_json = config.model_dump_json()

        assert isinstance(config_json, str)
        assert "500" in config_json
        assert "n_estimators" in config_json

    def test_reconstruction_from_dict(self) -> None:
        """Test reconstruction from dictionary."""
        original = LGBMConfig(
            n_estimators=500,
            learning_rate=0.1,
            num_leaves=50,
            optimize_hyperparams=False,
        )

        # Dump and reconstruct
        config_dict = original.model_dump()
        reconstructed = LGBMConfig(**config_dict)

        assert reconstructed.n_estimators == original.n_estimators
        assert reconstructed.learning_rate == original.learning_rate
        assert reconstructed.num_leaves == original.num_leaves
        assert reconstructed.optimize_hyperparams == original.optimize_hyperparams


class TestLGBMConfigImmutability:
    """Test configuration modification behavior."""

    def test_config_is_mutable(self) -> None:
        """Test that config can be modified after creation."""
        config = LGBMConfig()

        # Should allow modification (frozen=False)
        config.n_estimators = 500
        assert config.n_estimators == 500

    def test_validate_on_assignment(self) -> None:
        """Test that validation occurs on attribute assignment."""
        config = LGBMConfig()

        # Valid assignment
        config.learning_rate = 0.1
        assert config.learning_rate == 0.1

        # Invalid assignment should raise validation error
        with pytest.raises(ValidationError):
            config.learning_rate = 2.0  # > 1.0

    def test_extra_fields_forbidden(self) -> None:
        """Test that extra fields are not allowed."""
        with pytest.raises(ValidationError):
            LGBMConfig(extra_field="value")  # type: ignore
