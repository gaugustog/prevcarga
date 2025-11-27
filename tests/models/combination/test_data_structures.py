"""Tests for combination data structures."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from src.models.combination.data_structures import (
    CombinationMetadata,
    CombinationResult,
    CombinerConfig,
)


# Tests for CombinerConfig


class TestCombinerConfig:
    """Tests for CombinerConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = CombinerConfig()

        assert config.missing_threshold == 0.5
        assert config.require_fit is True
        assert config.validate_weights is True
        assert config.store_contributions is False
        assert config.store_intervals is False
        assert config.log_level == "INFO"
        assert config.custom_config == {}

    def test_custom_config(self):
        """Test custom configuration values."""
        config = CombinerConfig(
            missing_threshold=0.3,
            require_fit=False,
            store_contributions=True,
            log_level="DEBUG",
            custom_config={"param1": "value1"},
        )

        assert config.missing_threshold == 0.3
        assert config.require_fit is False
        assert config.store_contributions is True
        assert config.log_level == "DEBUG"
        assert config.custom_config == {"param1": "value1"}

    def test_invalid_missing_threshold(self):
        """Test validation of missing_threshold."""
        with pytest.raises(ValueError, match="missing_threshold"):
            CombinerConfig(missing_threshold=1.5)

        with pytest.raises(ValueError, match="missing_threshold"):
            CombinerConfig(missing_threshold=-0.1)

    def test_invalid_log_level(self):
        """Test validation of log_level."""
        with pytest.raises(ValueError, match="log_level"):
            CombinerConfig(log_level="INVALID")

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = CombinerConfig(
            missing_threshold=0.4,
            require_fit=True,
        )

        config_dict = config.to_dict()

        assert config_dict["missing_threshold"] == 0.4
        assert config_dict["require_fit"] is True
        assert "log_level" in config_dict

    def test_from_dict(self):
        """Test creation from dictionary."""
        config_dict = {
            "missing_threshold": 0.3,
            "require_fit": False,
            "log_level": "WARNING",
        }

        config = CombinerConfig.from_dict(config_dict)

        assert config.missing_threshold == 0.3
        assert config.require_fit is False
        assert config.log_level == "WARNING"

    def test_from_dict_with_defaults(self):
        """Test creation from dictionary with missing keys."""
        config_dict = {"missing_threshold": 0.2}

        config = CombinerConfig.from_dict(config_dict)

        assert config.missing_threshold == 0.2
        assert config.require_fit is True  # Default
        assert config.log_level == "INFO"  # Default


# Tests for CombinationMetadata


class TestCombinationMetadata:
    """Tests for CombinationMetadata dataclass."""

    def test_basic_creation(self):
        """Test basic metadata creation."""
        metadata = CombinationMetadata(
            timestamp=datetime.now(),
            combination_method="weighted_average",
            models_used=["lgbm", "rf", "xgb"],
        )

        assert metadata.combination_method == "weighted_average"
        assert len(metadata.models_used) == 3
        assert metadata.models_excluded == []
        assert metadata.processing_time_seconds == 0.0

    def test_full_creation(self):
        """Test metadata creation with all fields."""
        metadata = CombinationMetadata(
            timestamp=datetime.now(),
            combination_method="stacking",
            models_used=["lgbm", "rf"],
            models_excluded=["xgb"],
            processing_time_seconds=1.5,
            performance_metrics={"rmse": 10.5},
            configuration={"param": "value"},
            warnings=["Model xgb excluded due to high missing rate"],
        )

        assert metadata.models_excluded == ["xgb"]
        assert metadata.processing_time_seconds == 1.5
        assert metadata.performance_metrics["rmse"] == 10.5
        assert len(metadata.warnings) == 1

    def test_add_warning(self):
        """Test adding warnings."""
        metadata = CombinationMetadata(
            timestamp=datetime.now(),
            combination_method="test",
            models_used=["m1"],
        )

        metadata.add_warning("Warning 1")
        metadata.add_warning("Warning 2")

        assert len(metadata.warnings) == 2
        assert "Warning 1" in metadata.warnings

    def test_add_metric(self):
        """Test adding metrics."""
        metadata = CombinationMetadata(
            timestamp=datetime.now(),
            combination_method="test",
            models_used=["m1"],
        )

        metadata.add_metric("rmse", 10.5)
        metadata.add_metric("mae", 8.2)

        assert metadata.performance_metrics["rmse"] == 10.5
        assert metadata.performance_metrics["mae"] == 8.2

    def test_to_dict(self):
        """Test conversion to dictionary."""
        now = datetime.now()
        metadata = CombinationMetadata(
            timestamp=now,
            combination_method="test",
            models_used=["m1", "m2"],
            models_excluded=["m3"],
            processing_time_seconds=0.5,
        )

        d = metadata.to_dict()

        assert d["timestamp"] == now.isoformat()
        assert d["combination_method"] == "test"
        assert d["models_used"] == ["m1", "m2"]
        assert d["models_excluded"] == ["m3"]
        assert d["processing_time_seconds"] == 0.5

    def test_from_dict(self):
        """Test creation from dictionary."""
        d = {
            "timestamp": "2024-01-01T10:00:00",
            "combination_method": "test",
            "models_used": ["m1"],
            "models_excluded": ["m2"],
            "processing_time_seconds": 1.0,
        }

        metadata = CombinationMetadata.from_dict(d)

        assert metadata.combination_method == "test"
        assert metadata.models_used == ["m1"]
        assert metadata.models_excluded == ["m2"]
        assert metadata.processing_time_seconds == 1.0

    def test_from_dict_with_datetime_object(self):
        """Test creation from dictionary with datetime object."""
        now = datetime.now()
        d = {
            "timestamp": now,
            "combination_method": "test",
            "models_used": ["m1"],
        }

        metadata = CombinationMetadata.from_dict(d)

        assert metadata.timestamp == now

    def test_repr(self):
        """Test string representation."""
        metadata = CombinationMetadata(
            timestamp=datetime.now(),
            combination_method="weighted",
            models_used=["m1", "m2"],
            models_excluded=["m3"],
            processing_time_seconds=0.5,
        )

        repr_str = repr(metadata)

        assert "CombinationMetadata" in repr_str
        assert "weighted" in repr_str
        assert "2" in repr_str  # models_used count
        assert "1" in repr_str  # models_excluded count


# Tests for CombinationResult


class TestCombinationResult:
    """Tests for CombinationResult dataclass."""

    @pytest.fixture
    def sample_combined_predictions(self):
        """Create sample combined predictions."""
        dates = pd.date_range("2024-01-01", periods=48, freq="30min")
        return pd.DataFrame(
            {
                "pred_h0": np.random.randn(48) * 100 + 1000,
                "pred_h1": np.random.randn(48) * 100 + 1000,
            },
            index=dates,
        )

    @pytest.fixture
    def sample_weights(self):
        """Create sample weights."""
        return {"lgbm": 0.4, "rf": 0.35, "xgb": 0.25}

    @pytest.fixture
    def sample_metadata(self):
        """Create sample metadata."""
        return CombinationMetadata(
            timestamp=datetime.now(),
            combination_method="test",
            models_used=["lgbm", "rf", "xgb"],
        )

    def test_basic_creation(
        self, sample_combined_predictions, sample_weights, sample_metadata
    ):
        """Test basic result creation."""
        result = CombinationResult(
            combined_predictions=sample_combined_predictions,
            weights=sample_weights,
            metadata=sample_metadata,
        )

        assert result.n_models == 3
        assert result.n_samples == 48

    def test_empty_predictions_raises(self, sample_weights, sample_metadata):
        """Test that empty predictions raises error."""
        with pytest.raises(ValueError, match="empty"):
            CombinationResult(
                combined_predictions=pd.DataFrame(),
                weights=sample_weights,
                metadata=sample_metadata,
            )

    def test_empty_weights_raises(
        self, sample_combined_predictions, sample_metadata
    ):
        """Test that empty weights raises error."""
        with pytest.raises(ValueError, match="empty"):
            CombinationResult(
                combined_predictions=sample_combined_predictions,
                weights={},
                metadata=sample_metadata,
            )

    def test_weights_not_summing_to_one_raises(
        self, sample_combined_predictions, sample_metadata
    ):
        """Test that weights not summing to 1 raises error."""
        bad_weights = {"m1": 0.5, "m2": 0.3}  # Sum = 0.8

        with pytest.raises(ValueError, match="sum to 1"):
            CombinationResult(
                combined_predictions=sample_combined_predictions,
                weights=bad_weights,
                metadata=sample_metadata,
            )

    def test_negative_weights_raises(
        self, sample_combined_predictions, sample_metadata
    ):
        """Test that negative weights raises error."""
        bad_weights = {"m1": 1.2, "m2": -0.2}  # Sum = 1.0 but negative weight

        with pytest.raises(ValueError, match="non-negative"):
            CombinationResult(
                combined_predictions=sample_combined_predictions,
                weights=bad_weights,
                metadata=sample_metadata,
            )

    def test_horizons_property(
        self, sample_combined_predictions, sample_weights, sample_metadata
    ):
        """Test horizons property."""
        result = CombinationResult(
            combined_predictions=sample_combined_predictions,
            weights=sample_weights,
            metadata=sample_metadata,
        )

        assert result.horizons == [0, 1]

    def test_get_dominant_model(
        self, sample_combined_predictions, sample_metadata
    ):
        """Test get_dominant_model method."""
        weights = {"lgbm": 0.5, "rf": 0.3, "xgb": 0.2}
        result = CombinationResult(
            combined_predictions=sample_combined_predictions,
            weights=weights,
            metadata=sample_metadata,
        )

        assert result.get_dominant_model() == "lgbm"

    def test_model_contributions(
        self, sample_combined_predictions, sample_weights, sample_metadata
    ):
        """Test model contributions."""
        dates = sample_combined_predictions.index
        contributions = {
            "lgbm": pd.DataFrame({"pred_h0": np.random.randn(48)}, index=dates),
            "rf": pd.DataFrame({"pred_h0": np.random.randn(48)}, index=dates),
            "xgb": pd.DataFrame({"pred_h0": np.random.randn(48)}, index=dates),
        }

        result = CombinationResult(
            combined_predictions=sample_combined_predictions,
            weights=sample_weights,
            metadata=sample_metadata,
            model_contributions=contributions,
        )

        lgbm_contrib = result.get_model_contribution("lgbm")
        assert "pred_h0" in lgbm_contrib.columns

    def test_model_contributions_not_available(
        self, sample_combined_predictions, sample_weights, sample_metadata
    ):
        """Test error when contributions not available."""
        result = CombinationResult(
            combined_predictions=sample_combined_predictions,
            weights=sample_weights,
            metadata=sample_metadata,
        )

        with pytest.raises(ValueError, match="not available"):
            result.get_model_contribution("lgbm")

    def test_model_contribution_not_found(
        self, sample_combined_predictions, sample_weights, sample_metadata
    ):
        """Test error when model not in contributions."""
        dates = sample_combined_predictions.index
        contributions = {
            "lgbm": pd.DataFrame({"pred_h0": np.random.randn(48)}, index=dates),
        }

        result = CombinationResult(
            combined_predictions=sample_combined_predictions,
            weights=sample_weights,
            metadata=sample_metadata,
            model_contributions=contributions,
        )

        with pytest.raises(KeyError, match="unknown"):
            result.get_model_contribution("unknown")

    def test_to_dict(
        self, sample_combined_predictions, sample_weights, sample_metadata
    ):
        """Test conversion to dictionary."""
        result = CombinationResult(
            combined_predictions=sample_combined_predictions,
            weights=sample_weights,
            metadata=sample_metadata,
        )

        d = result.to_dict()

        assert "combined_predictions" in d
        assert "weights" in d
        assert "metadata" in d
        assert d["has_contributions"] is False
        assert d["has_intervals"] is False

    def test_from_dict(
        self, sample_combined_predictions, sample_weights, sample_metadata
    ):
        """Test creation from dictionary."""
        result = CombinationResult(
            combined_predictions=sample_combined_predictions,
            weights=sample_weights,
            metadata=sample_metadata,
        )

        d = result.to_dict()
        loaded = CombinationResult.from_dict(d)

        assert loaded.n_models == result.n_models
        assert loaded.weights == result.weights

    def test_repr(
        self, sample_combined_predictions, sample_weights, sample_metadata
    ):
        """Test string representation."""
        result = CombinationResult(
            combined_predictions=sample_combined_predictions,
            weights=sample_weights,
            metadata=sample_metadata,
        )

        repr_str = repr(result)

        assert "CombinationResult" in repr_str
        assert "n_models=3" in repr_str
        assert "n_samples=48" in repr_str
