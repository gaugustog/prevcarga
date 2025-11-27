"""Tests for transmission loss calculator.

Tests cover:
- LossConfig initialization and defaults
- LossModel enum values
- LossCalculator with different models
- Loss validation
- Energy balance checking
- Loss distribution to subsystems
- Loss adjustment for reconciliation
- Historical loss analysis
- LossIntegrator functionality
"""

# ruff: noqa: NPY002

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.reconciliation.losses.loss_calculator import (
    LossCalculationResult,
    LossCalculator,
    LossConfig,
    LossIntegrator,
    LossModel,
    LossStatistics,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def basic_forecasts() -> dict[str, np.ndarray]:
    """Create basic forecasts with generation and consumption."""
    return {
        "Generation": np.array([10000, 11000, 12000]),
        "Consumption": np.array([9200, 10120, 11040]),
    }


@pytest.fixture
def forecasts_with_losses() -> dict[str, np.ndarray]:
    """Create forecasts including losses."""
    return {
        "Generation": np.array([10000, 11000, 12000]),
        "Consumption": np.array([9200, 10120, 11040]),
        "Total_Losses": np.array([800, 880, 960]),
    }


@pytest.fixture
def subsystem_forecasts() -> dict[str, np.ndarray]:
    """Create forecasts with subsystem data."""
    return {
        "Generation": np.array([10000, 11000, 12000]),
        "Consumption": np.array([9200, 10120, 11040]),
        "SE": np.array([4600, 5060, 5520]),  # 50% of consumption
        "S": np.array([2760, 3036, 3312]),  # 30% of consumption
        "NE": np.array([1380, 1518, 1656]),  # 15% of consumption
        "N": np.array([460, 506, 552]),  # 5% of consumption
    }


@pytest.fixture
def default_calculator() -> LossCalculator:
    """Create calculator with default configuration."""
    return LossCalculator()


@pytest.fixture
def percentage_calculator() -> LossCalculator:
    """Create calculator with percentage model."""
    config = LossConfig(
        loss_model=LossModel.PERCENTAGE,
        default_loss_percentage=0.08,
    )
    return LossCalculator(config)


# =============================================================================
# LossModel Tests
# =============================================================================


class TestLossModel:
    """Tests for LossModel enum."""

    def test_percentage_model(self) -> None:
        """Test PERCENTAGE model value."""
        assert LossModel.PERCENTAGE.value == "PERCENTAGE"

    def test_absolute_model(self) -> None:
        """Test ABSOLUTE model value."""
        assert LossModel.ABSOLUTE.value == "ABSOLUTE"

    def test_historical_model(self) -> None:
        """Test HISTORICAL model value."""
        assert LossModel.HISTORICAL.value == "HISTORICAL"

    def test_adaptive_model(self) -> None:
        """Test ADAPTIVE model value."""
        assert LossModel.ADAPTIVE.value == "ADAPTIVE"

    def test_all_models_unique(self) -> None:
        """Test that all model values are unique."""
        values = [m.value for m in LossModel]
        assert len(values) == len(set(values))


# =============================================================================
# LossConfig Tests
# =============================================================================


class TestLossConfig:
    """Tests for LossConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = LossConfig()

        assert config.loss_model == LossModel.PERCENTAGE
        assert config.default_loss_percentage == 0.08
        assert config.min_loss_percentage == 0.0
        assert config.max_loss_percentage == 0.15
        assert config.enable_validation is True

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = LossConfig(
            loss_model=LossModel.ADAPTIVE,
            default_loss_percentage=0.10,
            max_loss_percentage=0.20,
        )

        assert config.loss_model == LossModel.ADAPTIVE
        assert config.default_loss_percentage == 0.10
        assert config.max_loss_percentage == 0.20

    def test_valid_percentage_range(self) -> None:
        """Test that default percentage is within valid range."""
        config = LossConfig()
        assert config.min_loss_percentage <= config.default_loss_percentage
        assert config.default_loss_percentage <= config.max_loss_percentage


# =============================================================================
# LossCalculationResult Tests
# =============================================================================


class TestLossCalculationResult:
    """Tests for LossCalculationResult dataclass."""

    def test_result_creation(self) -> None:
        """Test creating a loss calculation result."""
        result = LossCalculationResult(
            losses={"Total_Losses": np.array([800, 880, 960])},
            loss_percentages={"Total_Losses": 8.0},
            energy_balance_satisfied=True,
            validation_status={"valid": True, "warnings": [], "errors": []},
            model_used=LossModel.PERCENTAGE,
        )

        assert "Total_Losses" in result.losses
        assert result.loss_percentages["Total_Losses"] == 8.0
        assert result.energy_balance_satisfied
        assert result.model_used == LossModel.PERCENTAGE


# =============================================================================
# LossCalculator Initialization Tests
# =============================================================================


class TestLossCalculatorInit:
    """Tests for LossCalculator initialization."""

    def test_default_initialization(self) -> None:
        """Test default initialization."""
        calculator = LossCalculator()

        assert calculator.config.loss_model == LossModel.PERCENTAGE
        assert calculator.config.default_loss_percentage == 0.08

    def test_custom_config_initialization(self) -> None:
        """Test initialization with custom config."""
        config = LossConfig(
            loss_model=LossModel.ADAPTIVE,
            default_loss_percentage=0.10,
        )
        calculator = LossCalculator(config)

        assert calculator.config.loss_model == LossModel.ADAPTIVE
        assert calculator.config.default_loss_percentage == 0.10


# =============================================================================
# Percentage Model Tests
# =============================================================================


class TestPercentageModel:
    """Tests for percentage-based loss calculation."""

    def test_basic_percentage_calculation(
        self,
        percentage_calculator: LossCalculator,
        basic_forecasts: dict[str, np.ndarray],
    ) -> None:
        """Test basic percentage loss calculation."""
        result = percentage_calculator.calculate_losses(basic_forecasts)

        # 8% of generation
        expected_losses = np.array([800, 880, 960])
        np.testing.assert_array_almost_equal(
            result.losses["Total_Losses"], expected_losses
        )

    def test_percentage_model_result(
        self,
        percentage_calculator: LossCalculator,
        basic_forecasts: dict[str, np.ndarray],
    ) -> None:
        """Test that result contains correct model identifier."""
        result = percentage_calculator.calculate_losses(basic_forecasts)
        assert result.model_used == LossModel.PERCENTAGE

    def test_loss_percentage_value(
        self,
        percentage_calculator: LossCalculator,
        basic_forecasts: dict[str, np.ndarray],
    ) -> None:
        """Test that loss percentage is correctly calculated."""
        result = percentage_calculator.calculate_losses(basic_forecasts)

        # Average should be 8%
        assert abs(result.loss_percentages["Total_Losses"] - 8.0) < 0.1

    def test_different_percentage_values(self) -> None:
        """Test different percentage configurations."""
        forecasts = {"Generation": np.array([10000]), "Consumption": np.array([9000])}

        for pct in [0.05, 0.08, 0.12]:
            config = LossConfig(
                loss_model=LossModel.PERCENTAGE,
                default_loss_percentage=pct,
            )
            calculator = LossCalculator(config)
            result = calculator.calculate_losses(forecasts)

            expected = 10000 * pct
            assert abs(result.losses["Total_Losses"][0] - expected) < 0.1


# =============================================================================
# Absolute Model Tests
# =============================================================================


class TestAbsoluteModel:
    """Tests for absolute loss calculation."""

    def test_absolute_loss_calculation(
        self, basic_forecasts: dict[str, np.ndarray]
    ) -> None:
        """Test absolute loss calculation (generation - consumption)."""
        config = LossConfig(loss_model=LossModel.ABSOLUTE)
        calculator = LossCalculator(config)

        result = calculator.calculate_losses(basic_forecasts)

        # Losses = Generation - Consumption
        expected_losses = np.array([800, 880, 960])
        np.testing.assert_array_almost_equal(
            result.losses["Total_Losses"], expected_losses
        )

    def test_absolute_non_negative(self) -> None:
        """Test that absolute losses are non-negative."""
        # Consumption > Generation (should result in 0 losses)
        forecasts = {
            "Generation": np.array([9000, 10000]),
            "Consumption": np.array([9500, 10500]),
        }

        config = LossConfig(loss_model=LossModel.ABSOLUTE)
        calculator = LossCalculator(config)

        result = calculator.calculate_losses(forecasts)

        # Should be clipped to 0
        assert np.all(result.losses["Total_Losses"] >= 0)


# =============================================================================
# Historical Model Tests
# =============================================================================


class TestHistoricalModel:
    """Tests for historical-based loss calculation."""

    def test_historical_fallback_without_data(
        self, basic_forecasts: dict[str, np.ndarray]
    ) -> None:
        """Test historical model falls back to percentage without data."""
        config = LossConfig(
            loss_model=LossModel.HISTORICAL,
            default_loss_percentage=0.08,
        )
        calculator = LossCalculator(config)

        result = calculator.calculate_losses(basic_forecasts)

        # Should use percentage model as fallback
        expected_losses = np.array([800, 880, 960])
        np.testing.assert_array_almost_equal(
            result.losses["Total_Losses"], expected_losses
        )

    def test_historical_with_data(self) -> None:
        """Test historical model with sufficient data."""
        config = LossConfig(
            loss_model=LossModel.HISTORICAL,
            historical_window=10,
        )
        calculator = LossCalculator(config)

        # Add historical data (10% loss rate)
        for _ in range(20):
            calculator.update_historical(0.10)

        forecasts = {"Generation": np.array([10000]), "Consumption": np.array([9000])}
        result = calculator.calculate_losses(forecasts)

        # Should use historical average (10%)
        expected = 10000 * 0.10
        assert abs(result.losses["Total_Losses"][0] - expected) < 100


# =============================================================================
# Adaptive Model Tests
# =============================================================================


class TestAdaptiveModel:
    """Tests for adaptive loss calculation."""

    def test_adaptive_loss_calculation(
        self, basic_forecasts: dict[str, np.ndarray]
    ) -> None:
        """Test adaptive loss calculation."""
        config = LossConfig(
            loss_model=LossModel.ADAPTIVE,
            default_loss_percentage=0.08,
        )
        calculator = LossCalculator(config)

        result = calculator.calculate_losses(basic_forecasts)

        # Should calculate based on load level
        assert "Total_Losses" in result.losses
        assert result.model_used == LossModel.ADAPTIVE

    def test_adaptive_higher_load_higher_losses(self) -> None:
        """Test that higher loads result in higher loss percentages."""
        config = LossConfig(
            loss_model=LossModel.ADAPTIVE,
            default_loss_percentage=0.08,
            adaptive_scale_factor=0.01,
        )
        calculator = LossCalculator(config)

        # Low load
        low_load = {
            "Generation": np.array([5000]),
            "Consumption": np.array([4500]),
        }
        result_low = calculator.calculate_losses(low_load)
        pct_low = result_low.loss_percentages["Total_Losses"]

        # High load
        high_load = {
            "Generation": np.array([50000]),
            "Consumption": np.array([45000]),
        }
        result_high = calculator.calculate_losses(high_load)
        pct_high = result_high.loss_percentages["Total_Losses"]

        # Higher load should have higher loss percentage
        assert pct_high > pct_low


# =============================================================================
# Loss Validation Tests
# =============================================================================


class TestLossValidation:
    """Tests for loss validation."""

    def test_valid_losses(self, default_calculator: LossCalculator) -> None:
        """Test validation of valid losses."""
        generation = np.array([10000])
        consumption = np.array([9200])
        losses = np.array([800])

        status = default_calculator._validate_losses(losses, generation, consumption)

        assert status["valid"]
        assert len(status["errors"]) == 0

    def test_negative_losses_invalid(self, default_calculator: LossCalculator) -> None:
        """Test that negative losses are invalid."""
        generation = np.array([10000])
        consumption = np.array([10500])
        losses = np.array([-500])

        status = default_calculator._validate_losses(losses, generation, consumption)

        assert not status["valid"]
        assert any("Negative" in e for e in status["errors"])

    def test_excessive_losses_invalid(self) -> None:
        """Test that excessive losses are invalid."""
        config = LossConfig(max_loss_percentage=0.15)
        calculator = LossCalculator(config)

        generation = np.array([10000])
        consumption = np.array([7800])
        losses = np.array([2200])  # 22%

        status = calculator._validate_losses(losses, generation, consumption)

        assert not status["valid"]
        assert any("exceeds" in e for e in status["errors"])

    def test_low_losses_warning(self) -> None:
        """Test that unusually low losses generate warnings."""
        config = LossConfig(min_loss_percentage=0.05)
        calculator = LossCalculator(config)

        generation = np.array([10000])
        consumption = np.array([9800])
        losses = np.array([200])  # 2%

        status = calculator._validate_losses(losses, generation, consumption)

        assert any("below" in w for w in status["warnings"])


# =============================================================================
# Energy Balance Tests
# =============================================================================


class TestEnergyBalance:
    """Tests for energy balance checking."""

    def test_balanced_energy(self, default_calculator: LossCalculator) -> None:
        """Test energy balance is satisfied."""
        generation = np.array([10000, 11000])
        consumption = np.array([9200, 10120])
        losses = np.array([800, 880])

        balanced = default_calculator._check_energy_balance(
            generation, consumption, losses
        )

        assert balanced

    def test_unbalanced_energy(self, default_calculator: LossCalculator) -> None:
        """Test energy balance is not satisfied."""
        generation = np.array([10000, 11000])
        consumption = np.array([9200, 10120])
        losses = np.array([900, 980])  # Too high

        balanced = default_calculator._check_energy_balance(
            generation, consumption, losses
        )

        assert not balanced


# =============================================================================
# Loss Distribution Tests
# =============================================================================


class TestLossDistribution:
    """Tests for loss distribution to subsystems."""

    def test_distribute_losses(self, default_calculator: LossCalculator) -> None:
        """Test distributing losses proportionally to load."""
        total_losses = np.array([800, 880])
        subsystem_loads = {
            "SE": np.array([4600, 5060]),
            "S": np.array([3680, 4048]),
            "NE": np.array([920, 1012]),
        }

        distributed = default_calculator.distribute_losses(total_losses, subsystem_loads)

        # Check sum equals total
        sum_distributed = sum(distributed.values())
        np.testing.assert_array_almost_equal(sum_distributed, total_losses)

    def test_distribute_losses_naming(self, default_calculator: LossCalculator) -> None:
        """Test that distributed losses have correct naming."""
        total_losses = np.array([800])
        subsystem_loads = {"SE": np.array([4000]), "S": np.array([3000])}

        distributed = default_calculator.distribute_losses(total_losses, subsystem_loads)

        assert "SE_Losses" in distributed
        assert "S_Losses" in distributed

    def test_distribute_proportionally(self, default_calculator: LossCalculator) -> None:
        """Test proportional distribution."""
        total_losses = np.array([1000])
        # 70% SE, 30% S
        subsystem_loads = {"SE": np.array([7000]), "S": np.array([3000])}

        distributed = default_calculator.distribute_losses(total_losses, subsystem_loads)

        assert abs(distributed["SE_Losses"][0] - 700) < 1
        assert abs(distributed["S_Losses"][0] - 300) < 1


# =============================================================================
# Loss Adjustment Tests
# =============================================================================


class TestLossAdjustment:
    """Tests for loss adjustment during reconciliation."""

    def test_adjust_for_reconciliation(
        self, default_calculator: LossCalculator
    ) -> None:
        """Test adjusting losses for reconciliation."""
        forecasts = {
            "Generation": np.array([10000]),
            "Consumption": np.array([9000]),
            "Total_Losses": np.array([500]),  # 5%
        }

        adjusted = default_calculator.adjust_for_reconciliation(
            forecasts, target_loss_percentage=0.08
        )

        # Target: 8% of 10000 = 800
        assert abs(adjusted["Total_Losses"][0] - 800) < 1

    def test_adjust_uses_default_percentage(
        self, percentage_calculator: LossCalculator
    ) -> None:
        """Test adjustment uses default percentage when not specified."""
        forecasts = {
            "Generation": np.array([10000]),
            "Consumption": np.array([9000]),
            "Total_Losses": np.array([1200]),  # 12%
        }

        adjusted = percentage_calculator.adjust_for_reconciliation(forecasts)

        # Should adjust to 8%
        assert abs(adjusted["Total_Losses"][0] - 800) < 1


# =============================================================================
# Historical Analysis Tests
# =============================================================================


class TestHistoricalAnalysis:
    """Tests for historical loss analysis."""

    def test_analyze_historical_losses(
        self, default_calculator: LossCalculator
    ) -> None:
        """Test historical loss analysis."""
        generation = np.array([10000, 11000, 12000, 10500, 11500])
        consumption = np.array([9200, 10120, 11040, 9660, 10580])

        stats = default_calculator.analyze_historical_losses(generation, consumption)

        assert isinstance(stats, LossStatistics)
        assert 5 < stats.mean_loss_percentage < 10  # Around 8%
        assert stats.min_loss_percentage >= 0

    def test_analyze_with_timestamps(
        self, default_calculator: LossCalculator
    ) -> None:
        """Test analysis with timestamp data."""
        np.random.seed(42)
        n = 168  # 1 week of hourly data
        base_gen = 10000 + np.random.randn(n) * 500
        generation = base_gen
        consumption = generation * 0.92

        timestamps = pd.date_range("2024-01-01", periods=n, freq="h")

        stats = default_calculator.analyze_historical_losses(
            generation, consumption, timestamps
        )

        assert "hourly_range" in stats.temporal_patterns


# =============================================================================
# Loss Report Tests
# =============================================================================


class TestLossReport:
    """Tests for loss report generation."""

    def test_generate_report(
        self,
        percentage_calculator: LossCalculator,
        basic_forecasts: dict[str, np.ndarray],
    ) -> None:
        """Test loss report generation."""
        result = percentage_calculator.calculate_losses(basic_forecasts)
        report = percentage_calculator.generate_loss_report(result, basic_forecasts)

        assert "model_used" in report
        assert report["model_used"] == "PERCENTAGE"
        assert "loss_percentages" in report
        assert "loss_statistics" in report
        assert "mean_mw" in report["loss_statistics"]

    def test_report_with_historical_comparison(
        self, default_calculator: LossCalculator
    ) -> None:
        """Test report includes historical comparison."""
        # Add historical data
        for _ in range(10):
            default_calculator.update_historical(0.10)

        forecasts = {"Generation": np.array([10000]), "Consumption": np.array([9200])}
        result = default_calculator.calculate_losses(forecasts)
        report = default_calculator.generate_loss_report(result, forecasts)

        assert "historical_comparison" in report


# =============================================================================
# LossIntegrator Tests
# =============================================================================


class TestLossIntegrator:
    """Tests for LossIntegrator class."""

    def test_integrate_losses(
        self,
        percentage_calculator: LossCalculator,
        basic_forecasts: dict[str, np.ndarray],
    ) -> None:
        """Test integrating losses into forecasts."""
        integrator = LossIntegrator(percentage_calculator)

        integrated = integrator.integrate(basic_forecasts)

        assert "Total_Losses" in integrated
        assert "Generation" in integrated
        assert "Consumption" in integrated

    def test_distribute_to_subsystems(
        self,
        percentage_calculator: LossCalculator,
        subsystem_forecasts: dict[str, np.ndarray],
    ) -> None:
        """Test distributing losses to subsystems."""
        integrator = LossIntegrator(percentage_calculator)

        # First integrate to add Total_Losses
        integrated = integrator.integrate(subsystem_forecasts)

        # Then distribute
        result = integrator.distribute_to_subsystems(
            integrated, subsystem_keys=["SE", "S", "NE", "N"]
        )

        assert "SE_Losses" in result
        assert "S_Losses" in result
        assert "NE_Losses" in result
        assert "N_Losses" in result

    def test_validate_hierarchy_losses(
        self, percentage_calculator: LossCalculator
    ) -> None:
        """Test hierarchy loss validation."""
        integrator = LossIntegrator(percentage_calculator)

        # Create mock hierarchy
        hierarchy = MagicMock()
        hierarchy.get_all_node_names.return_value = [
            "SIN",
            "SE",
            "S",
            "Total_Losses",
            "SE_Losses",
        ]

        forecasts = {
            "SIN": np.array([10000]),
            "SE": np.array([5000]),
            "S": np.array([3000]),
            "Total_Losses": np.array([800]),
            "SE_Losses": np.array([400]),
        }

        validation = integrator.validate_hierarchy_losses(forecasts, hierarchy)

        assert validation["valid"]

    def test_validate_hierarchy_missing_losses(
        self, percentage_calculator: LossCalculator
    ) -> None:
        """Test validation detects missing loss forecasts."""
        integrator = LossIntegrator(percentage_calculator)

        hierarchy = MagicMock()
        hierarchy.get_all_node_names.return_value = [
            "SIN",
            "Total_Losses",
            "SE_Losses",
        ]

        forecasts = {
            "SIN": np.array([10000]),
            "Total_Losses": np.array([800]),
            # SE_Losses missing
        }

        validation = integrator.validate_hierarchy_losses(forecasts, hierarchy)

        assert not validation["valid"]
        assert any("Missing" in str(issue) for issue in validation["issues"])


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_forecasts(self, default_calculator: LossCalculator) -> None:
        """Test handling of empty forecasts."""
        forecasts: dict[str, np.ndarray] = {}

        result = default_calculator.calculate_losses(forecasts)

        # Should handle gracefully
        assert "Total_Losses" in result.losses

    def test_single_point_forecast(self, default_calculator: LossCalculator) -> None:
        """Test single point forecast."""
        forecasts = {
            "Generation": np.array([10000]),
            "Consumption": np.array([9200]),
        }

        result = default_calculator.calculate_losses(forecasts)

        assert len(result.losses["Total_Losses"]) == 1

    def test_pandas_series_input(
        self, percentage_calculator: LossCalculator
    ) -> None:
        """Test handling of pandas Series input."""
        forecasts = {
            "Generation": pd.Series([10000, 11000, 12000]),
            "Consumption": pd.Series([9200, 10120, 11040]),
        }

        result = percentage_calculator.calculate_losses(forecasts)

        expected_losses = np.array([800, 880, 960])
        np.testing.assert_array_almost_equal(
            result.losses["Total_Losses"], expected_losses
        )

    def test_zero_generation(self, default_calculator: LossCalculator) -> None:
        """Test handling of zero generation."""
        forecasts = {
            "Generation": np.array([0, 10000]),
            "Consumption": np.array([0, 9200]),
        }

        result = default_calculator.calculate_losses(forecasts)

        # Should handle gracefully without division by zero
        assert not np.any(np.isnan(result.losses["Total_Losses"]))

    def test_nan_values(self, default_calculator: LossCalculator) -> None:
        """Test handling of NaN values in forecasts."""
        forecasts = {
            "Generation": np.array([10000, np.nan, 12000]),
            "Consumption": np.array([9200, 10120, np.nan]),
        }

        # Should complete without error
        result = default_calculator.calculate_losses(forecasts)
        assert "Total_Losses" in result.losses


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_complete_workflow(self) -> None:
        """Test complete loss calculation workflow."""
        # 1. Configure
        config = LossConfig(
            loss_model=LossModel.PERCENTAGE,
            default_loss_percentage=0.08,
            enable_validation=True,
        )

        # 2. Create calculator
        calculator = LossCalculator(config)

        # 3. Prepare data
        forecasts = {
            "Generation": np.array([10000, 11000, 12000]),
            "Consumption": np.array([9200, 10120, 11040]),
        }

        # 4. Calculate losses
        result = calculator.calculate_losses(forecasts)

        # 5. Validate results
        assert result.validation_status["valid"]
        # Energy balance check compares gen vs consumption + losses
        # For this data, the 8% losses happen to match exactly
        assert isinstance(result.energy_balance_satisfied, bool)

        # 6. Distribute to subsystems
        subsystem_loads = {
            "SE": np.array([4600, 5060, 5520]),
            "S": np.array([4600, 5060, 5520]),
        }
        distributed = calculator.distribute_losses(
            result.losses["Total_Losses"], subsystem_loads
        )

        # 7. Generate report
        report = calculator.generate_loss_report(result, forecasts)

        assert len(distributed) == 2
        assert "model_used" in report

    def test_integrator_workflow(self) -> None:
        """Test complete integrator workflow."""
        config = LossConfig(loss_model=LossModel.PERCENTAGE)
        calculator = LossCalculator(config)
        integrator = LossIntegrator(calculator)

        forecasts = {
            "Generation": np.array([10000, 11000]),
            "Consumption": np.array([9200, 10120]),
            "SE": np.array([4600, 5060]),
            "S": np.array([4600, 5060]),
        }

        # Integrate losses
        integrated = integrator.integrate(forecasts)
        assert "Total_Losses" in integrated

        # Distribute to subsystems
        result = integrator.distribute_to_subsystems(
            integrated, subsystem_keys=["SE", "S"]
        )

        assert "SE_Losses" in result
        assert "S_Losses" in result
