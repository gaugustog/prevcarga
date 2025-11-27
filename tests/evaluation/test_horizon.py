"""Tests for horizon analyzer (HorizonAnalyzer).

This module tests the HorizonAnalyzer which analyzes forecast performance
by horizon (D+0 to D+8) and models degradation patterns.
"""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.horizon import (
    DegradationModel,
    HorizonAnalyzer,
    HorizonConfig,
    HorizonResult,
    SkillScores,
)
from src.evaluation.metrics import MetricsConfig, MetricsResult


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def default_config():
    """Default horizon configuration."""
    return HorizonConfig(
        horizons=["h0", "h1", "h2", "h3", "h4"],
        metrics=["mape", "mae", "rmse"],
        baseline_horizon="h0",
        degradation_model="linear",
        min_samples=10,
        alpha=0.05,
    )


@pytest.fixture
def analyzer(default_config):
    """HorizonAnalyzer with default config."""
    return HorizonAnalyzer(config=default_config)


@pytest.fixture
def linear_degradation_data():
    """Generate data with linear degradation pattern."""
    np.random.seed(42)
    n_samples = 100
    base_value = 100.0

    actuals = {
        f"h{i}": np.random.normal(base_value, 5, n_samples) for i in range(5)
    }

    # Forecasts degrade linearly: h0 is good, h4 is worse
    forecasts = {}
    for i in range(5):
        error_std = 2 + i * 1.5  # Linear degradation
        noise = np.random.normal(0, error_std, n_samples)
        forecasts[f"h{i}"] = actuals[f"h{i}"] + noise

    return actuals, forecasts


@pytest.fixture
def exponential_degradation_data():
    """Generate data with exponential degradation pattern."""
    np.random.seed(43)
    n_samples = 100
    base_value = 200.0

    actuals = {
        f"h{i}": np.random.normal(base_value, 10, n_samples) for i in range(5)
    }

    # Forecasts degrade exponentially
    forecasts = {}
    for i in range(5):
        error_std = 2 * np.exp(0.3 * i)  # Exponential degradation
        noise = np.random.normal(0, error_std, n_samples)
        forecasts[f"h{i}"] = actuals[f"h{i}"] + noise

    return actuals, forecasts


# ============================================================================
# Test Configuration
# ============================================================================


class TestHorizonConfig:
    """Test HorizonConfig validation and creation."""

    def test_default_config(self):
        """Test default configuration."""
        config = HorizonConfig()

        assert config.horizons == ["h0", "h1", "h2", "h3", "h4", "h5", "h6", "h7", "h8"]
        assert config.metrics == ["mape", "mae", "rmse"]
        assert config.baseline_horizon == "h0"
        assert config.include_skill_scores is True
        assert config.degradation_model == "exponential"
        assert config.min_samples == 30
        assert config.alpha == 0.05

    def test_custom_config(self):
        """Test custom configuration."""
        config = HorizonConfig(
            horizons=["h0", "h1", "h2"],
            metrics=["mape", "rmse"],
            baseline_horizon="h0",
            degradation_model="linear",
            min_samples=20,
            alpha=0.01,
        )

        assert config.horizons == ["h0", "h1", "h2"]
        assert config.metrics == ["mape", "rmse"]
        assert config.degradation_model == "linear"
        assert config.min_samples == 20
        assert config.alpha == 0.01

    def test_invalid_horizon_format(self):
        """Test validation of horizon format."""
        with pytest.raises(ValueError, match="Invalid horizon format"):
            HorizonConfig(horizons=["d0", "d1"])

    def test_invalid_horizon_number(self):
        """Test validation of horizon number."""
        with pytest.raises(ValueError, match="Invalid horizon format"):
            HorizonConfig(horizons=["hx", "hy"])

    def test_empty_horizons(self):
        """Test validation of empty horizons."""
        with pytest.raises(ValueError, match="cannot be empty"):
            HorizonConfig(horizons=[])

    def test_invalid_metric(self):
        """Test validation of invalid metric."""
        with pytest.raises(ValueError, match="Invalid metric"):
            HorizonConfig(metrics=["mape", "invalid_metric"])

    def test_invalid_baseline_format(self):
        """Test validation of baseline horizon format."""
        with pytest.raises(ValueError, match="Invalid baseline_horizon format"):
            HorizonConfig(baseline_horizon="d0")

    def test_power_law_model(self):
        """Test power law degradation model."""
        config = HorizonConfig(degradation_model="power_law")
        assert config.degradation_model == "power_law"


# ============================================================================
# Test Data Structures
# ============================================================================


class TestSkillScores:
    """Test SkillScores dataclass."""

    def test_basic_creation(self):
        """Test basic skill scores creation."""
        scores = SkillScores(
            persistence=0.85,
            climatology=0.92,
            random_walk=0.80,
        )

        assert scores.persistence == 0.85
        assert scores.climatology == 0.92
        assert scores.random_walk == 0.80
        assert scores.metadata == {}

    def test_with_metadata(self):
        """Test skill scores with metadata."""
        scores = SkillScores(
            persistence=0.85,
            climatology=0.92,
            metadata={"horizon": "h0", "n_samples": 100},
        )

        assert scores.metadata["horizon"] == "h0"
        assert scores.metadata["n_samples"] == 100

    def test_to_dict(self):
        """Test conversion to dictionary."""
        scores = SkillScores(
            persistence=0.85,
            climatology=0.92,
            random_walk=0.80,
        )

        result = scores.to_dict()

        assert result["persistence"] == 0.85
        assert result["climatology"] == 0.92
        assert result["random_walk"] == 0.80


class TestDegradationModel:
    """Test DegradationModel dataclass."""

    def test_linear_model(self):
        """Test linear degradation model."""
        model = DegradationModel(
            model_type="linear",
            parameters=[2.5, 0.3],
            degradation_rate=0.3,
            r_squared=0.95,
            predictions=np.array([2.5, 2.8, 3.1, 3.4]),
        )

        assert model.model_type == "linear"
        assert model.parameters == [2.5, 0.3]
        assert model.degradation_rate == 0.3
        assert model.r_squared == 0.95
        assert len(model.predictions) == 4

    def test_exponential_model(self):
        """Test exponential degradation model."""
        model = DegradationModel(
            model_type="exponential",
            parameters=[2.0, 0.15],
            degradation_rate=0.15,
            r_squared=0.98,
        )

        assert model.model_type == "exponential"
        assert model.degradation_rate == 0.15

    def test_to_dict(self):
        """Test conversion to dictionary."""
        model = DegradationModel(
            model_type="linear",
            parameters=[2.5, 0.3],
            degradation_rate=0.3,
            r_squared=0.95,
            predictions=np.array([2.5, 2.8]),
        )

        result = model.to_dict()

        assert result["model_type"] == "linear"
        assert result["parameters"] == [2.5, 0.3]
        assert result["degradation_rate"] == 0.3
        assert result["r_squared"] == 0.95
        assert result["predictions"] == [2.5, 2.8]


class TestHorizonResult:
    """Test HorizonResult dataclass."""

    @pytest.fixture
    def sample_result(self):
        """Create sample horizon result."""
        metrics = {
            "h0": MetricsResult(mape=2.5, mae=10.0, rmse=12.0, mse=144.0, r2=0.95),
            "h1": MetricsResult(mape=3.0, mae=12.0, rmse=15.0, mse=225.0, r2=0.90),
            "h2": MetricsResult(mape=3.8, mae=15.0, rmse=18.0, mse=324.0, r2=0.85),
        }

        model = DegradationModel(
            model_type="linear",
            parameters=[2.5, 0.3],
            degradation_rate=0.3,
            r_squared=0.95,
        )

        return HorizonResult(
            metrics_by_horizon=metrics,
            degradation_model=model,
            degradation_rate=0.3,
            best_horizons=[("h0", 2.5), ("h1", 3.0), ("h2", 3.8)],
            worst_horizons=[("h2", 3.8), ("h1", 3.0), ("h0", 2.5)],
            degradation_pattern="linear",
        )

    def test_n_horizons_property(self, sample_result):
        """Test n_horizons property."""
        assert sample_result.n_horizons == 3

    def test_best_horizon_property(self, sample_result):
        """Test best_horizon property."""
        best = sample_result.best_horizon
        assert best == ("h0", 2.5)

    def test_worst_horizon_property(self, sample_result):
        """Test worst_horizon property."""
        worst = sample_result.worst_horizon
        assert worst == ("h2", 3.8)

    def test_empty_rankings(self):
        """Test with empty rankings."""
        result = HorizonResult(
            metrics_by_horizon={},
            degradation_model=DegradationModel(
                model_type="linear",
                parameters=[0.0, 0.0],
                degradation_rate=0.0,
                r_squared=0.0,
            ),
            degradation_rate=0.0,
            best_horizons=[],
            worst_horizons=[],
        )

        assert result.best_horizon is None
        assert result.worst_horizon is None

    def test_to_dict(self, sample_result):
        """Test conversion to dictionary."""
        result_dict = sample_result.to_dict()

        assert "metrics_by_horizon" in result_dict
        assert "degradation_model" in result_dict
        assert "degradation_rate" in result_dict
        assert "best_horizons" in result_dict
        assert result_dict["degradation_rate"] == 0.3

    def test_repr(self, sample_result):
        """Test string representation."""
        repr_str = repr(sample_result)

        assert "HorizonResult" in repr_str
        assert "n_horizons=3" in repr_str
        assert "pattern='linear'" in repr_str


# ============================================================================
# Test Analyzer Initialization
# ============================================================================


class TestHorizonAnalyzerInit:
    """Test HorizonAnalyzer initialization."""

    def test_default_initialization(self):
        """Test initialization with defaults."""
        analyzer = HorizonAnalyzer()

        assert analyzer.config.horizons == [
            "h0",
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
            "h7",
            "h8",
        ]
        assert analyzer.metrics_calculator is not None

    def test_custom_config_initialization(self, default_config):
        """Test initialization with custom config."""
        analyzer = HorizonAnalyzer(config=default_config)

        assert analyzer.config == default_config
        assert analyzer.config.horizons == ["h0", "h1", "h2", "h3", "h4"]

    def test_custom_metrics_config(self):
        """Test initialization with custom metrics config."""
        metrics_config = MetricsConfig(use_symmetric_mape=False)
        analyzer = HorizonAnalyzer(metrics_config=metrics_config)

        assert analyzer.metrics_calculator.config.use_symmetric_mape is False

    def test_baseline_not_in_horizons_error(self):
        """Test error when baseline not in horizons."""
        config = HorizonConfig(
            horizons=["h0", "h1", "h2"], baseline_horizon="h5"
        )

        with pytest.raises(ValueError, match="not in horizons"):
            HorizonAnalyzer(config=config)

    def test_repr(self, analyzer):
        """Test string representation."""
        repr_str = repr(analyzer)

        assert "HorizonAnalyzer" in repr_str
        assert "n_horizons=5" in repr_str
        assert "degradation_model='linear'" in repr_str


# ============================================================================
# Test Horizon Analysis
# ============================================================================


class TestAnalyzeByHorizon:
    """Test analyze_by_horizon method."""

    def test_dataframe_input(self, analyzer, linear_degradation_data):
        """Test analysis with DataFrame input."""
        actuals_dict, forecasts_dict = linear_degradation_data

        # Convert to DataFrames
        actuals_df = pd.DataFrame(actuals_dict)
        forecasts_df = pd.DataFrame(forecasts_dict)

        result = analyzer.analyze_by_horizon(actuals_df, forecasts_df)

        assert isinstance(result, HorizonResult)
        assert result.n_horizons == 5
        assert "h0" in result.metrics_by_horizon
        assert "h4" in result.metrics_by_horizon

    def test_dict_input(self, analyzer, linear_degradation_data):
        """Test analysis with dict input."""
        actuals, forecasts = linear_degradation_data

        result = analyzer.analyze_by_horizon(actuals, forecasts)

        assert isinstance(result, HorizonResult)
        assert result.n_horizons == 5

    def test_missing_actual_horizon_error(self, analyzer, linear_degradation_data):
        """Test error when actual horizon missing."""
        actuals, forecasts = linear_degradation_data
        del actuals["h2"]

        with pytest.raises(ValueError, match="Missing horizons in actual"):
            analyzer.analyze_by_horizon(actuals, forecasts)

    def test_missing_forecast_horizon_error(self, analyzer, linear_degradation_data):
        """Test error when forecast horizon missing."""
        actuals, forecasts = linear_degradation_data
        del forecasts["h3"]

        with pytest.raises(ValueError, match="Missing horizons in forecast"):
            analyzer.analyze_by_horizon(actuals, forecasts)

    def test_linear_degradation_detected(self, linear_degradation_data):
        """Test linear degradation pattern detection."""
        actuals, forecasts = linear_degradation_data

        config = HorizonConfig(
            horizons=["h0", "h1", "h2", "h3", "h4"],
            degradation_model="linear",
            min_samples=10,
        )
        analyzer = HorizonAnalyzer(config=config)

        result = analyzer.analyze_by_horizon(actuals, forecasts)

        assert result.degradation_model.model_type == "linear"
        assert result.degradation_rate > 0  # Errors should increase

    def test_exponential_degradation_detected(self, exponential_degradation_data):
        """Test exponential degradation pattern detection."""
        actuals, forecasts = exponential_degradation_data

        config = HorizonConfig(
            horizons=["h0", "h1", "h2", "h3", "h4"],
            degradation_model="exponential",
            min_samples=10,
        )
        analyzer = HorizonAnalyzer(config=config)

        result = analyzer.analyze_by_horizon(actuals, forecasts)

        assert result.degradation_model.model_type == "exponential"
        assert result.degradation_rate > 0

    def test_best_worst_horizon_ranking(self, analyzer, linear_degradation_data):
        """Test best and worst horizon identification."""
        actuals, forecasts = linear_degradation_data

        result = analyzer.analyze_by_horizon(actuals, forecasts)

        # h0 should be best (lowest error), h4 should be worst
        assert result.best_horizon[0] == "h0"
        assert result.worst_horizon[0] == "h4"

        # Rankings should be complete
        assert len(result.best_horizons) == 5
        assert len(result.worst_horizons) == 5

    def test_statistical_tests_included(self, analyzer, linear_degradation_data):
        """Test statistical tests are run."""
        actuals, forecasts = linear_degradation_data

        result = analyzer.analyze_by_horizon(actuals, forecasts)

        assert "anova" in result.statistical_tests
        assert "kruskal_wallis" in result.statistical_tests

        # ANOVA should have expected keys
        assert "f_statistic" in result.statistical_tests["anova"]
        assert "p_value" in result.statistical_tests["anova"]
        assert "significant" in result.statistical_tests["anova"]

    def test_skill_scores_computed(self, linear_degradation_data):
        """Test skill scores computation."""
        actuals, forecasts = linear_degradation_data

        config = HorizonConfig(
            horizons=["h0", "h1", "h2", "h3", "h4"],
            include_skill_scores=True,
            min_samples=10,
        )
        analyzer = HorizonAnalyzer(config=config)

        result = analyzer.analyze_by_horizon(actuals, forecasts)

        assert len(result.skill_scores_by_horizon) == 5
        assert "h0" in result.skill_scores_by_horizon

        # Skill scores should be reasonable (typically between -2 and 2)
        # Note: Skill scores can be negative when forecast is worse than baseline
        for horizon, scores in result.skill_scores_by_horizon.items():
            assert -2.0 <= scores.persistence <= 2.0
            assert -2.0 <= scores.climatology <= 2.0

    def test_skill_scores_disabled(self, linear_degradation_data):
        """Test skill scores can be disabled."""
        actuals, forecasts = linear_degradation_data

        config = HorizonConfig(
            horizons=["h0", "h1", "h2", "h3", "h4"],
            include_skill_scores=False,
            min_samples=10,
        )
        analyzer = HorizonAnalyzer(config=config)

        result = analyzer.analyze_by_horizon(actuals, forecasts)

        assert len(result.skill_scores_by_horizon) == 0

    def test_metadata_populated(self, analyzer, linear_degradation_data):
        """Test metadata is populated."""
        actuals, forecasts = linear_degradation_data

        result = analyzer.analyze_by_horizon(actuals, forecasts)

        assert "n_horizons" in result.metadata
        assert "sample_sizes" in result.metadata
        assert "degradation_model_type" in result.metadata
        assert "baseline_horizon" in result.metadata

        assert result.metadata["n_horizons"] == 5
        assert result.metadata["degradation_model_type"] == "linear"
        assert result.metadata["baseline_horizon"] == "h0"


# ============================================================================
# Test Degradation Computation
# ============================================================================


class TestComputeDegradation:
    """Test compute_degradation method."""

    def test_linear_degradation(self, analyzer):
        """Test linear degradation fitting."""
        # Create synthetic metrics with linear degradation
        metrics = {
            "h0": MetricsResult(mape=2.5, mae=10, rmse=12, mse=144, r2=0.95),
            "h1": MetricsResult(mape=3.0, mae=12, rmse=14, mse=196, r2=0.93),
            "h2": MetricsResult(mape=3.5, mae=14, rmse=16, mse=256, r2=0.91),
            "h3": MetricsResult(mape=4.0, mae=16, rmse=18, mse=324, r2=0.89),
            "h4": MetricsResult(mape=4.5, mae=18, rmse=20, mse=400, r2=0.87),
        }

        rate, params = analyzer.compute_degradation(metrics, model="linear")

        # Should detect linear increase (rate ~ 0.5)
        assert 0.4 < rate < 0.6
        assert len(params) == 2

    def test_exponential_degradation(self, analyzer):
        """Test exponential degradation fitting."""
        # Create synthetic metrics with exponential growth
        metrics = {
            "h0": MetricsResult(mape=2.0, mae=10, rmse=12, mse=144, r2=0.95),
            "h1": MetricsResult(mape=2.5, mae=12, rmse=14, mse=196, r2=0.93),
            "h2": MetricsResult(mape=3.3, mae=14, rmse=16, mse=256, r2=0.91),
            "h3": MetricsResult(mape=4.5, mae=16, rmse=18, mse=324, r2=0.89),
            "h4": MetricsResult(mape=6.0, mae=18, rmse=20, mse=400, r2=0.87),
        }

        rate, params = analyzer.compute_degradation(metrics, model="exponential")

        assert rate > 0  # Should show growth
        assert len(params) == 2

    def test_power_law_degradation(self, analyzer):
        """Test power law degradation fitting."""
        metrics = {
            "h0": MetricsResult(mape=2.0, mae=10, rmse=12, mse=144, r2=0.95),
            "h1": MetricsResult(mape=2.8, mae=12, rmse=14, mse=196, r2=0.93),
            "h2": MetricsResult(mape=3.5, mae=14, rmse=16, mse=256, r2=0.91),
            "h3": MetricsResult(mape=4.1, mae=16, rmse=18, mse=324, r2=0.89),
        }

        rate, params = analyzer.compute_degradation(metrics, model="power_law")

        assert len(params) == 2

    def test_insufficient_horizons_error(self, analyzer):
        """Test error with too few horizons."""
        metrics = {
            "h0": MetricsResult(mape=2.5, mae=10, rmse=12, mse=144, r2=0.95),
        }

        with pytest.raises(ValueError, match="at least 2 horizons"):
            analyzer.compute_degradation(metrics, model="linear")

    def test_invalid_model_type_error(self, analyzer):
        """Test error with invalid model type."""
        metrics = {
            "h0": MetricsResult(mape=2.5, mae=10, rmse=12, mse=144, r2=0.95),
            "h1": MetricsResult(mape=3.0, mae=12, rmse=14, mse=196, r2=0.93),
        }

        with pytest.raises(ValueError, match="Invalid model type"):
            analyzer.compute_degradation(metrics, model="invalid")


# ============================================================================
# Test Skill Scores
# ============================================================================


class TestComputeSkillScores:
    """Test compute_skill_scores method."""

    def test_persistence_baseline(self, analyzer):
        """Test skill score vs persistence baseline."""
        np.random.seed(42)
        n = 50

        # Create actuals with autocorrelation
        actuals = {
            "h0": np.cumsum(np.random.normal(0, 1, n)) + 100,
            "h1": np.cumsum(np.random.normal(0, 1, n)) + 100,
        }

        # Good forecasts
        forecasts = {
            "h0": actuals["h0"] + np.random.normal(0, 0.5, n),
            "h1": actuals["h1"] + np.random.normal(0, 0.5, n),
        }

        skills = analyzer.compute_skill_scores(
            actuals, forecasts, baseline="persistence"
        )

        assert "h0" in skills
        assert "h1" in skills

        # Good forecast should have positive skill vs persistence
        assert skills["h0"] > 0
        assert skills["h1"] > 0

    def test_climatology_baseline(self, analyzer):
        """Test skill score vs climatology baseline."""
        np.random.seed(43)
        n = 50

        actuals = {
            "h0": np.random.normal(100, 10, n),
            "h1": np.random.normal(100, 10, n),
        }

        # Good forecasts
        forecasts = {
            "h0": actuals["h0"] + np.random.normal(0, 2, n),
            "h1": actuals["h1"] + np.random.normal(0, 2, n),
        }

        skills = analyzer.compute_skill_scores(
            actuals, forecasts, baseline="climatology"
        )

        # Should have positive skill vs climatology
        assert skills["h0"] > 0
        assert skills["h1"] > 0

    def test_random_walk_baseline(self, analyzer):
        """Test skill score vs random walk baseline."""
        np.random.seed(44)
        n = 50

        actuals = {
            "h0": np.cumsum(np.random.normal(0, 1, n)) + 100,
        }

        forecasts = {
            "h0": actuals["h0"] + np.random.normal(0, 0.5, n),
        }

        skills = analyzer.compute_skill_scores(
            actuals, forecasts, baseline="random_walk"
        )

        assert "h0" in skills

    def test_missing_horizons_handled(self, analyzer):
        """Test graceful handling of missing horizons."""
        actuals = {"h0": np.array([1, 2, 3])}
        forecasts = {"h0": np.array([1.1, 2.1, 3.1]), "h1": np.array([1, 2, 3])}

        skills = analyzer.compute_skill_scores(actuals, forecasts, "persistence")

        # Should only have h0 (missing in actuals but present in both)
        assert "h0" in skills
        assert "h1" not in skills

    def test_invalid_baseline_error(self, analyzer):
        """Test error with invalid baseline type."""
        actuals = {"h0": np.array([1, 2, 3])}
        forecasts = {"h0": np.array([1.1, 2.1, 3.1])}

        with pytest.raises(ValueError, match="Invalid baseline"):
            analyzer.compute_skill_scores(actuals, forecasts, "invalid")


# ============================================================================
# Test Pattern Identification
# ============================================================================


class TestIdentifyDegradationPattern:
    """Test identify_degradation_pattern method."""

    def test_linear_pattern(self, analyzer):
        """Test linear pattern identification."""
        model = DegradationModel(
            model_type="linear",
            parameters=[2.5, 0.3],
            degradation_rate=0.3,
            r_squared=0.95,
        )

        pattern = analyzer.identify_degradation_pattern(model)

        assert pattern == "linear"

    def test_exponential_growth_pattern(self, analyzer):
        """Test exponential growth pattern."""
        model = DegradationModel(
            model_type="exponential",
            parameters=[2.0, 0.15],
            degradation_rate=0.15,
            r_squared=0.98,
        )

        pattern = analyzer.identify_degradation_pattern(model)

        assert pattern == "exponential"

    def test_exponential_decay_pattern(self, analyzer):
        """Test exponential decay pattern."""
        model = DegradationModel(
            model_type="exponential",
            parameters=[5.0, -0.1],
            degradation_rate=-0.1,
            r_squared=0.90,
        )

        pattern = analyzer.identify_degradation_pattern(model)

        assert pattern == "exponential_decay"

    def test_plateau_pattern(self, analyzer):
        """Test plateau pattern (near-zero rate)."""
        model = DegradationModel(
            model_type="exponential",
            parameters=[3.0, 0.005],
            degradation_rate=0.005,
            r_squared=0.85,
        )

        pattern = analyzer.identify_degradation_pattern(model)

        assert pattern == "plateau"

    def test_power_law_superlinear(self, analyzer):
        """Test power law superlinear pattern."""
        model = DegradationModel(
            model_type="power_law",
            parameters=[2.0, 1.5],
            degradation_rate=1.5,
            r_squared=0.92,
        )

        pattern = analyzer.identify_degradation_pattern(model)

        assert pattern == "superlinear"

    def test_power_law_sublinear(self, analyzer):
        """Test power law sublinear pattern."""
        model = DegradationModel(
            model_type="power_law",
            parameters=[2.0, 0.5],
            degradation_rate=0.5,
            r_squared=0.92,
        )

        pattern = analyzer.identify_degradation_pattern(model)

        assert pattern == "sublinear"


# ============================================================================
# Test Horizon Comparison
# ============================================================================


class TestCompareHorizons:
    """Test compare_horizons method."""

    @pytest.fixture
    def sample_result(self):
        """Create sample result for comparison."""
        metrics = {
            "h0": MetricsResult(mape=2.5, mae=10.0, rmse=12.0, mse=144.0, r2=0.95),
            "h1": MetricsResult(mape=3.0, mae=12.0, rmse=15.0, mse=225.0, r2=0.90),
            "h2": MetricsResult(mape=4.0, mae=16.0, rmse=20.0, mse=400.0, r2=0.85),
        }

        model = DegradationModel(
            model_type="linear",
            parameters=[2.5, 0.5],
            degradation_rate=0.5,
            r_squared=0.95,
        )

        return HorizonResult(
            metrics_by_horizon=metrics,
            degradation_model=model,
            degradation_rate=0.5,
            best_horizons=[("h0", 2.5), ("h1", 3.0), ("h2", 4.0)],
            worst_horizons=[("h2", 4.0), ("h1", 3.0), ("h0", 2.5)],
        )

    def test_compare_two_horizons(self, analyzer, sample_result):
        """Test comparison between two horizons."""
        comparison = analyzer.compare_horizons(sample_result, "h0", "h2")

        assert comparison["horizon_a"] == "h0"
        assert comparison["horizon_b"] == "h2"
        assert comparison["mape_diff"] == 1.5  # 4.0 - 2.5
        assert comparison["mae_diff"] == 6.0  # 16 - 10
        assert comparison["rmse_diff"] == 8.0  # 20 - 12

        # Relative changes
        assert comparison["relative_mape_change"] == pytest.approx(60.0, abs=0.1)

    def test_missing_horizon_error(self, analyzer, sample_result):
        """Test error when horizon not found."""
        with pytest.raises(ValueError, match="not found in result"):
            analyzer.compare_horizons(sample_result, "h0", "h5")


# ============================================================================
# Test Report Generation
# ============================================================================


class TestGenerateHorizonReport:
    """Test generate_horizon_report method."""

    def test_basic_report(self, analyzer, linear_degradation_data):
        """Test basic report generation."""
        actuals, forecasts = linear_degradation_data

        result = analyzer.analyze_by_horizon(actuals, forecasts)
        report = analyzer.generate_horizon_report(result)

        assert isinstance(report, str)
        assert "Horizon Analysis Report" in report
        assert "Degradation pattern:" in report
        assert "Best horizon:" in report
        assert "Worst horizon:" in report

    def test_report_with_skill_scores(self, linear_degradation_data):
        """Test report includes skill scores when enabled."""
        actuals, forecasts = linear_degradation_data

        config = HorizonConfig(
            horizons=["h0", "h1", "h2", "h3", "h4"],
            include_skill_scores=True,
            min_samples=10,
        )
        analyzer = HorizonAnalyzer(config=config)

        result = analyzer.analyze_by_horizon(actuals, forecasts)
        report = analyzer.generate_horizon_report(result)

        assert "Skill(P):" in report

    def test_report_with_statistical_tests(self, analyzer, linear_degradation_data):
        """Test report includes statistical tests."""
        actuals, forecasts = linear_degradation_data

        result = analyzer.analyze_by_horizon(actuals, forecasts)
        report = analyzer.generate_horizon_report(result)

        assert "Statistical Tests:" in report
        assert "ANOVA:" in report
        assert "Kruskal-Wallis:" in report


# ============================================================================
# Test Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_horizon(self):
        """Test with single horizon."""
        config = HorizonConfig(horizons=["h0"], min_samples=5)
        analyzer = HorizonAnalyzer(config=config)

        actuals = {"h0": np.array([100, 101, 102, 103, 104])}
        forecasts = {"h0": np.array([99, 100, 101, 104, 105])}

        result = analyzer.analyze_by_horizon(actuals, forecasts)

        # Should complete but degradation model will be trivial
        assert result.n_horizons == 1
        assert "h0" in result.metrics_by_horizon

    def test_two_horizons_minimal(self):
        """Test with two horizons (minimum for degradation)."""
        config = HorizonConfig(horizons=["h0", "h1"], min_samples=5)
        analyzer = HorizonAnalyzer(config=config)

        actuals = {
            "h0": np.array([100, 101, 102, 103, 104]),
            "h1": np.array([110, 111, 112, 113, 114]),
        }
        forecasts = {
            "h0": np.array([99, 100, 101, 104, 105]),
            "h1": np.array([108, 109, 111, 114, 115]),
        }

        result = analyzer.analyze_by_horizon(actuals, forecasts)

        assert result.n_horizons == 2
        assert result.degradation_rate >= 0

    def test_identical_performance_across_horizons(self, analyzer):
        """Test when all horizons have identical performance."""
        np.random.seed(42)
        n = 50

        # All horizons have identical performance
        actuals = {f"h{i}": np.random.normal(100, 10, n) for i in range(5)}
        forecasts = {
            f"h{i}": actuals[f"h{i}"] + np.random.normal(0, 2, n) for i in range(5)
        }

        result = analyzer.analyze_by_horizon(actuals, forecasts)

        # Degradation rate should be near zero
        assert abs(result.degradation_rate) < 0.5

    def test_insufficient_samples_filtered(self):
        """Test filtering of horizons with insufficient samples."""
        config = HorizonConfig(
            horizons=["h0", "h1", "h2"], min_samples=20
        )
        analyzer = HorizonAnalyzer(config=config)

        # h2 has insufficient samples
        actuals = {
            "h0": np.random.normal(100, 10, 50),
            "h1": np.random.normal(100, 10, 50),
            "h2": np.random.normal(100, 10, 10),  # Only 10 samples
        }
        forecasts = {
            "h0": np.random.normal(100, 12, 50),
            "h1": np.random.normal(100, 12, 50),
            "h2": np.random.normal(100, 12, 10),
        }

        result = analyzer.analyze_by_horizon(actuals, forecasts)

        # h2 should be excluded
        assert result.n_horizons == 2
        assert "h0" in result.metrics_by_horizon
        assert "h1" in result.metrics_by_horizon
        assert "h2" not in result.metrics_by_horizon


# ============================================================================
# Test Performance
# ============================================================================


class TestPerformance:
    """Test performance requirements."""

    def test_full_horizon_analysis_performance(self, analyzer):
        """Test full horizon analysis completes in <2 seconds."""
        import time

        np.random.seed(42)
        n = 1000  # Large dataset

        actuals = {f"h{i}": np.random.normal(100, 10, n) for i in range(5)}
        forecasts = {
            f"h{i}": actuals[f"h{i}"] + np.random.normal(0, 5, n) for i in range(5)
        }

        start = time.time()
        result = analyzer.analyze_by_horizon(actuals, forecasts)
        elapsed = time.time() - start

        assert elapsed < 2.0  # Must complete in <2 seconds
        assert result.n_horizons == 5

    def test_degradation_fitting_performance(self, analyzer):
        """Test degradation model fitting is fast."""
        import time

        metrics = {
            f"h{i}": MetricsResult(
                mape=2.0 + i * 0.5, mae=10, rmse=12, mse=144, r2=0.95
            )
            for i in range(9)
        }

        start = time.time()
        rate, params = analyzer.compute_degradation(metrics, "exponential")
        elapsed = time.time() - start

        assert elapsed < 0.2  # Must complete in <200ms
        assert rate is not None
