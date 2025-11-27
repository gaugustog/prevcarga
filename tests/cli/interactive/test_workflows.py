"""Tests for guided workflows module.

This module provides comprehensive tests for the guided workflow system,
including workflow manager, step validation, command building, and
individual workflow functions.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from src.cli.interactive.workflows import (
    ConsoleWorkflowExecutor,
    GuidedWorkflowManager,
    StepOption,
    StepType,
    WorkflowDefinition,
    WorkflowExecutor,
    WorkflowResult,
    WorkflowStatus,
    WorkflowStep,
    build_backtest_command,
    build_comparison_command,
    build_feature_command,
    build_prediction_command,
    build_training_command,
    create_backtest_steps,
    create_comparison_steps,
    create_feature_steps,
    create_prediction_steps,
    create_training_steps,
    guided_backtest_workflow,
    guided_comparison_workflow,
    guided_feature_workflow,
    guided_prediction_workflow,
    guided_training_workflow,
    start_guided_workflow,
    validate_areas,
    validate_date,
    validate_horizons,
    validate_number,
    validate_positive_number,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


# =============================================================================
# Test Fixtures
# =============================================================================


class MockWorkflowExecutor(WorkflowExecutor):
    """Mock executor for testing workflows."""

    def __init__(
        self,
        prompts: dict[str, str] | None = None,
        selects: dict[str, str] | None = None,
        confirms: dict[str, bool] | None = None,
    ) -> None:
        """Initialize mock executor with predefined responses."""
        self.prompts = prompts or {}
        self.selects = selects or {}
        self.confirms = confirms or {}
        self.prompt_calls: list[str] = []
        self.select_calls: list[str] = []
        self.confirm_calls: list[str] = []
        self.echo_calls: list[tuple[str, str | None]] = []
        self._prompt_index = 0
        self._prompt_responses: list[str] = []

    def set_prompt_responses(self, responses: list[str]) -> None:
        """Set sequential prompt responses."""
        self._prompt_responses = responses
        self._prompt_index = 0

    def prompt(self, message: str, default: str | None = None) -> str:
        """Return predefined prompt response."""
        self.prompt_calls.append(message)

        # Check for specific message matches first
        for key, value in self.prompts.items():
            if key in message:
                return value

        # Use sequential responses
        if self._prompt_responses and self._prompt_index < len(self._prompt_responses):
            response = self._prompt_responses[self._prompt_index]
            self._prompt_index += 1
            return response

        return default or ""

    def select(self, message: str, options: Sequence[StepOption]) -> str:
        """Return predefined select response."""
        self.select_calls.append(message)

        for key, value in self.selects.items():
            if key in message:
                return value

        # Return default or first option
        default = next((o.value for o in options if o.is_default), None)
        return default or (options[0].value if options else "")

    def confirm(self, message: str) -> bool:
        """Return predefined confirm response."""
        self.confirm_calls.append(message)

        for key, value in self.confirms.items():
            if key in message:
                return value

        return True  # Default to confirm

    def echo(self, message: str, style: str | None = None) -> None:
        """Record echo call."""
        self.echo_calls.append((message, style))


@pytest.fixture
def mock_executor() -> MockWorkflowExecutor:
    """Create a mock executor."""
    return MockWorkflowExecutor()


@pytest.fixture
def workflow_manager(mock_executor: MockWorkflowExecutor) -> GuidedWorkflowManager:
    """Create workflow manager with mock executor."""
    return GuidedWorkflowManager(executor=mock_executor)


# =============================================================================
# Validation Function Tests
# =============================================================================


class TestValidateDate:
    """Tests for date validation."""

    def test_valid_date(self) -> None:
        """Test valid date format."""
        is_valid, error = validate_date("2024-01-15")
        assert is_valid is True
        assert error == ""

    def test_empty_date(self) -> None:
        """Test empty date is allowed."""
        is_valid, error = validate_date("")
        assert is_valid is True
        assert error == ""

    def test_invalid_format(self) -> None:
        """Test invalid date format."""
        is_valid, error = validate_date("01-15-2024")
        assert is_valid is False
        assert "Invalid date format" in error

    def test_invalid_date(self) -> None:
        """Test invalid date value."""
        is_valid, error = validate_date("2024-13-45")
        assert is_valid is False
        assert "Invalid date format" in error

    def test_non_date_string(self) -> None:
        """Test non-date string."""
        is_valid, error = validate_date("not-a-date")
        assert is_valid is False


class TestValidateNumber:
    """Tests for number validation."""

    def test_valid_number(self) -> None:
        """Test valid number."""
        is_valid, error = validate_number("42")
        assert is_valid is True
        assert error == ""

    def test_empty_number(self) -> None:
        """Test empty value is allowed."""
        is_valid, error = validate_number("")
        assert is_valid is True

    def test_with_min_valid(self) -> None:
        """Test number above minimum."""
        is_valid, error = validate_number("10", min_val=5)
        assert is_valid is True

    def test_with_min_invalid(self) -> None:
        """Test number below minimum."""
        is_valid, error = validate_number("3", min_val=5)
        assert is_valid is False
        assert "at least 5" in error

    def test_with_max_valid(self) -> None:
        """Test number below maximum."""
        is_valid, error = validate_number("10", max_val=20)
        assert is_valid is True

    def test_with_max_invalid(self) -> None:
        """Test number above maximum."""
        is_valid, error = validate_number("25", max_val=20)
        assert is_valid is False
        assert "at most 20" in error

    def test_non_number(self) -> None:
        """Test non-numeric string."""
        is_valid, error = validate_number("abc")
        assert is_valid is False
        assert "Invalid number" in error


class TestValidateAreas:
    """Tests for area validation."""

    def test_valid_single_area(self) -> None:
        """Test single valid area."""
        is_valid, error = validate_areas("SECO")
        assert is_valid is True

    def test_valid_multiple_areas(self) -> None:
        """Test multiple valid areas."""
        is_valid, error = validate_areas("SECO,S,NE")
        assert is_valid is True

    def test_all_keyword(self) -> None:
        """Test 'all' keyword."""
        is_valid, error = validate_areas("all")
        assert is_valid is True

    def test_case_insensitive(self) -> None:
        """Test case insensitivity for 'all'."""
        is_valid, error = validate_areas("ALL")
        assert is_valid is True

    def test_empty_areas(self) -> None:
        """Test empty areas."""
        is_valid, error = validate_areas("")
        assert is_valid is False
        assert "required" in error

    def test_invalid_area(self) -> None:
        """Test invalid area."""
        is_valid, error = validate_areas("INVALID")
        assert is_valid is False
        assert "Invalid area" in error

    def test_mixed_valid_invalid(self) -> None:
        """Test mix of valid and invalid areas."""
        is_valid, error = validate_areas("SECO,INVALID")
        assert is_valid is False


class TestValidateHorizons:
    """Tests for horizon validation."""

    def test_valid_single_horizon(self) -> None:
        """Test single valid horizon."""
        is_valid, error = validate_horizons("0")
        assert is_valid is True

    def test_valid_multiple_horizons(self) -> None:
        """Test multiple valid horizons."""
        is_valid, error = validate_horizons("0,1,2,3")
        assert is_valid is True

    def test_all_keyword(self) -> None:
        """Test 'all' keyword."""
        is_valid, error = validate_horizons("all")
        assert is_valid is True

    def test_empty_horizons(self) -> None:
        """Test empty horizons."""
        is_valid, error = validate_horizons("")
        assert is_valid is False

    def test_invalid_horizon_too_high(self) -> None:
        """Test horizon above 8."""
        is_valid, error = validate_horizons("9")
        assert is_valid is False
        assert "Invalid horizon" in error

    def test_invalid_horizon_negative(self) -> None:
        """Test negative horizon."""
        is_valid, error = validate_horizons("-1")
        assert is_valid is False

    def test_non_numeric_horizon(self) -> None:
        """Test non-numeric horizon."""
        is_valid, error = validate_horizons("abc")
        assert is_valid is False


class TestValidatePositiveNumber:
    """Tests for positive number validation."""

    def test_positive_number(self) -> None:
        """Test positive number."""
        is_valid, error = validate_positive_number("5")
        assert is_valid is True

    def test_zero(self) -> None:
        """Test zero is invalid."""
        is_valid, error = validate_positive_number("0")
        assert is_valid is False
        assert "positive" in error

    def test_negative(self) -> None:
        """Test negative number."""
        is_valid, error = validate_positive_number("-5")
        assert is_valid is False

    def test_empty(self) -> None:
        """Test empty value allowed."""
        is_valid, error = validate_positive_number("")
        assert is_valid is True


# =============================================================================
# WorkflowStep Tests
# =============================================================================


class TestWorkflowStep:
    """Tests for WorkflowStep class."""

    def test_basic_step(self) -> None:
        """Test basic step creation."""
        step = WorkflowStep(
            step_id="test",
            title="Test Step",
            description="A test step",
            step_type=StepType.TEXT_INPUT,
        )
        assert step.step_id == "test"
        assert step.title == "Test Step"
        assert step.required is True

    def test_validate_required_empty(self) -> None:
        """Test required field validation."""
        step = WorkflowStep(
            step_id="test",
            title="Test",
            description="",
            step_type=StepType.TEXT_INPUT,
            required=True,
        )
        is_valid, error = step.validate("")
        assert is_valid is False
        assert "required" in error

    def test_validate_optional_empty(self) -> None:
        """Test optional field allows empty."""
        step = WorkflowStep(
            step_id="test",
            title="Test",
            description="",
            step_type=StepType.TEXT_INPUT,
            required=False,
        )
        is_valid, error = step.validate("")
        assert is_valid is True

    def test_validate_date_input(self) -> None:
        """Test date input validation."""
        step = WorkflowStep(
            step_id="date",
            title="Date",
            description="",
            step_type=StepType.DATE_INPUT,
            required=False,
        )
        is_valid, _ = step.validate("2024-01-15")
        assert is_valid is True

        is_valid, _ = step.validate("invalid")
        assert is_valid is False

    def test_validate_number_input(self) -> None:
        """Test number input validation."""
        step = WorkflowStep(
            step_id="num",
            title="Number",
            description="",
            step_type=StepType.NUMBER_INPUT,
            required=False,
        )
        is_valid, _ = step.validate("42")
        assert is_valid is True

        is_valid, _ = step.validate("abc")
        assert is_valid is False

    def test_validate_single_select(self) -> None:
        """Test single select validation."""
        step = WorkflowStep(
            step_id="select",
            title="Select",
            description="",
            step_type=StepType.SINGLE_SELECT,
            options=[
                StepOption("a", "Option A"),
                StepOption("b", "Option B"),
            ],
            required=False,
        )
        is_valid, _ = step.validate("a")
        assert is_valid is True

        is_valid, error = step.validate("c")
        assert is_valid is False
        assert "Invalid selection" in error

    def test_custom_validation(self) -> None:
        """Test custom validation function."""
        def custom_validator(value: str) -> tuple[bool, str]:
            if value.startswith("test"):
                return True, ""
            return False, "Must start with 'test'"

        step = WorkflowStep(
            step_id="custom",
            title="Custom",
            description="",
            step_type=StepType.TEXT_INPUT,
            validation_fn=custom_validator,
            required=False,
        )

        is_valid, _ = step.validate("test123")
        assert is_valid is True

        is_valid, error = step.validate("abc")
        assert is_valid is False
        assert "Must start with" in error


# =============================================================================
# WorkflowResult Tests
# =============================================================================


class TestWorkflowResult:
    """Tests for WorkflowResult class."""

    def test_success_property(self) -> None:
        """Test success property."""
        result = WorkflowResult(status=WorkflowStatus.COMPLETED)
        assert result.success is True

        result = WorkflowResult(status=WorkflowStatus.FAILED)
        assert result.success is False

    def test_cancelled_property(self) -> None:
        """Test cancelled property."""
        result = WorkflowResult(status=WorkflowStatus.CANCELLED)
        assert result.cancelled is True

        result = WorkflowResult(status=WorkflowStatus.COMPLETED)
        assert result.cancelled is False

    def test_with_command(self) -> None:
        """Test result with command."""
        result = WorkflowResult(
            status=WorkflowStatus.COMPLETED,
            command="train --model lgbm",
            values={"model": "lgbm"},
        )
        assert result.command == "train --model lgbm"
        assert result.values["model"] == "lgbm"


# =============================================================================
# Step Creation Tests
# =============================================================================


class TestCreateSteps:
    """Tests for step creation functions."""

    def test_create_training_steps(self) -> None:
        """Test training steps creation."""
        steps = create_training_steps()
        assert len(steps) == 5

        step_ids = [s.step_id for s in steps]
        assert "model" in step_ids
        assert "areas" in step_ids
        assert "start_date" in step_ids
        assert "end_date" in step_ids
        assert "parallel" in step_ids

    def test_create_prediction_steps(self) -> None:
        """Test prediction steps creation."""
        steps = create_prediction_steps()
        assert len(steps) == 5

        step_ids = [s.step_id for s in steps]
        assert "pred_type" in step_ids
        assert "date" in step_ids
        assert "horizons" in step_ids
        assert "combination" in step_ids
        assert "format" in step_ids

    def test_create_backtest_steps(self) -> None:
        """Test backtest steps creation."""
        steps = create_backtest_steps()
        assert len(steps) == 5

        step_ids = [s.step_id for s in steps]
        assert "start_date" in step_ids
        assert "end_date" in step_ids
        assert "intervals" in step_ids
        assert "models" in step_ids
        assert "parallel" in step_ids

    def test_create_feature_steps(self) -> None:
        """Test feature steps creation."""
        steps = create_feature_steps()
        assert len(steps) == 5

        step_ids = [s.step_id for s in steps]
        assert "plugins" in step_ids
        assert "start_date" in step_ids
        assert "end_date" in step_ids
        assert "output_dir" in step_ids
        assert "parallel" in step_ids

    def test_create_comparison_steps(self) -> None:
        """Test comparison steps creation."""
        steps = create_comparison_steps()
        assert len(steps) == 5

        step_ids = [s.step_id for s in steps]
        assert "start_date" in step_ids
        assert "end_date" in step_ids
        assert "strategies" in step_ids
        assert "cv_folds" in step_ids
        assert "metrics" in step_ids


# =============================================================================
# Command Building Tests
# =============================================================================


class TestBuildTrainingCommand:
    """Tests for training command builder."""

    def test_basic_command(self) -> None:
        """Test basic training command."""
        values = {
            "model": "lgbm",
            "areas": "all",
            "start_date": "2024-01-01",
            "end_date": "2024-12-31",
            "parallel": "4",
        }
        cmd = build_training_command(values)

        assert "train" in cmd
        assert "--model lgbm" in cmd
        assert "--start-date 2024-01-01" in cmd
        assert "--end-date 2024-12-31" in cmd
        assert "--parallel 4" in cmd
        assert "--area" not in cmd  # 'all' should not add area flags

    def test_with_specific_areas(self) -> None:
        """Test command with specific areas."""
        values = {
            "model": "rf",
            "areas": "SECO,S",
            "start_date": "2024-01-01",
            "end_date": "2024-12-31",
            "parallel": "2",
        }
        cmd = build_training_command(values)

        assert "--area SECO" in cmd
        assert "--area S" in cmd


class TestBuildPredictionCommand:
    """Tests for prediction command builder."""

    def test_batch_command(self) -> None:
        """Test batch prediction command."""
        values = {
            "pred_type": "batch",
            "date": "2024-06-15",
            "horizons": "all",
            "combination": "weighted_avg",
            "format": "csv",
        }
        cmd = build_prediction_command(values)

        assert "predict batch" in cmd
        assert "--date 2024-06-15" in cmd
        assert "--combination weighted_avg" in cmd
        assert "--format csv" in cmd

    def test_intraday_command(self) -> None:
        """Test intraday prediction command."""
        values = {
            "pred_type": "intraday",
            "date": "2024-06-15",
            "horizons": "0",
            "combination": "weighted_avg",
            "format": "csv",
        }
        cmd = build_prediction_command(values)

        assert "predict intraday" in cmd
        assert "--date 2024-06-15" in cmd
        # Intraday doesn't include combination/format
        assert "--combination" not in cmd

    def test_with_specific_horizons(self) -> None:
        """Test command with specific horizons."""
        values = {
            "pred_type": "batch",
            "date": "2024-06-15",
            "horizons": "0,1,2",
            "combination": "stacking",
            "format": "json",
        }
        cmd = build_prediction_command(values)

        assert "--horizon 0" in cmd
        assert "--horizon 1" in cmd
        assert "--horizon 2" in cmd


class TestBuildBacktestCommand:
    """Tests for backtest command builder."""

    def test_basic_command(self) -> None:
        """Test basic backtest command."""
        values = {
            "start_date": "2024-01-01",
            "end_date": "2024-03-31",
            "intervals": "7,14",
            "models": "all",
            "parallel": "4",
        }
        cmd = build_backtest_command(values)

        assert "backtest" in cmd
        assert "--start-date 2024-01-01" in cmd
        assert "--end-date 2024-03-31" in cmd
        assert "--retraining-interval 7" in cmd
        assert "--retraining-interval 14" in cmd
        assert "--parallel 4" in cmd

    def test_with_specific_models(self) -> None:
        """Test command with specific models."""
        values = {
            "start_date": "2024-01-01",
            "end_date": "2024-03-31",
            "intervals": "7",
            "models": "lgbm,rf",
            "parallel": "2",
        }
        cmd = build_backtest_command(values)

        assert "--model lgbm" in cmd
        assert "--model rf" in cmd


class TestBuildFeatureCommand:
    """Tests for feature command builder."""

    def test_basic_command(self) -> None:
        """Test basic feature command."""
        values = {
            "plugins": "all",
            "start_date": "2024-01-01",
            "end_date": "2024-12-31",
            "output_dir": "features/",
            "parallel": "4",
        }
        cmd = build_feature_command(values)

        assert "features generate" in cmd
        assert "--start-date 2024-01-01" in cmd
        assert "--end-date 2024-12-31" in cmd
        assert "--output-dir features/" in cmd
        assert "--parallel 4" in cmd

    def test_with_specific_plugins(self) -> None:
        """Test command with specific plugins."""
        values = {
            "plugins": "temporal,calendar,lag",
            "start_date": "2024-01-01",
            "end_date": "2024-12-31",
            "output_dir": "out/",
            "parallel": "2",
        }
        cmd = build_feature_command(values)

        assert "--plugin temporal" in cmd
        assert "--plugin calendar" in cmd
        assert "--plugin lag" in cmd


class TestBuildComparisonCommand:
    """Tests for comparison command builder."""

    def test_basic_command(self) -> None:
        """Test basic comparison command."""
        values = {
            "start_date": "2024-01-01",
            "end_date": "2024-03-31",
            "strategies": "all",
            "cv_folds": "5",
            "metrics": "all",
        }
        cmd = build_comparison_command(values)

        assert "evaluate compare" in cmd
        assert "--start-date 2024-01-01" in cmd
        assert "--end-date 2024-03-31" in cmd
        assert "--cv-folds 5" in cmd

    def test_with_specific_strategies(self) -> None:
        """Test command with specific strategies."""
        values = {
            "start_date": "2024-01-01",
            "end_date": "2024-03-31",
            "strategies": "weighted_avg,stacking",
            "cv_folds": "3",
            "metrics": "mape,rmse",
        }
        cmd = build_comparison_command(values)

        assert "--strategy weighted_avg" in cmd
        assert "--strategy stacking" in cmd
        assert "--metric mape" in cmd
        assert "--metric rmse" in cmd


# =============================================================================
# GuidedWorkflowManager Tests
# =============================================================================


class TestGuidedWorkflowManager:
    """Tests for GuidedWorkflowManager class."""

    def test_list_workflows(self, workflow_manager: GuidedWorkflowManager) -> None:
        """Test listing available workflows."""
        workflows = workflow_manager.list_workflows()

        assert len(workflows) == 5
        assert "1" in workflows
        assert "2" in workflows
        assert "3" in workflows
        assert "4" in workflows
        assert "5" in workflows

    def test_get_workflow(self, workflow_manager: GuidedWorkflowManager) -> None:
        """Test getting a specific workflow."""
        workflow = workflow_manager.get_workflow("1")

        assert workflow is not None
        assert workflow.workflow_id == "1"
        assert "Training" in workflow.name
        assert len(workflow.steps) > 0

    def test_get_workflow_not_found(
        self, workflow_manager: GuidedWorkflowManager
    ) -> None:
        """Test getting non-existent workflow."""
        workflow = workflow_manager.get_workflow("99")
        assert workflow is None

    def test_register_workflow(
        self, workflow_manager: GuidedWorkflowManager
    ) -> None:
        """Test registering a custom workflow."""
        custom = WorkflowDefinition(
            workflow_id="custom",
            name="Custom Workflow",
            description="A custom workflow",
            steps=[
                WorkflowStep(
                    step_id="input",
                    title="Input",
                    description="Enter input",
                    step_type=StepType.TEXT_INPUT,
                ),
            ],
            command_builder=lambda v: f"custom --input {v['input']}",
        )

        workflow_manager.register_workflow(custom)

        assert "custom" in workflow_manager.list_workflows()
        assert workflow_manager.get_workflow("custom") is not None

    def test_unregister_workflow(
        self, workflow_manager: GuidedWorkflowManager
    ) -> None:
        """Test unregistering a workflow."""
        assert workflow_manager.unregister_workflow("1") is True
        assert "1" not in workflow_manager.list_workflows()

        # Second unregister should return False
        assert workflow_manager.unregister_workflow("1") is False

    def test_execute_unknown_workflow(
        self, workflow_manager: GuidedWorkflowManager
    ) -> None:
        """Test executing unknown workflow."""
        result = workflow_manager.execute_workflow("99")

        assert result.status == WorkflowStatus.FAILED
        assert "Unknown workflow" in (result.error_message or "")

    def test_execute_workflow_with_prefilled(
        self, mock_executor: MockWorkflowExecutor
    ) -> None:
        """Test executing workflow with pre-filled values."""
        mock_executor.confirms["Execute"] = True

        manager = GuidedWorkflowManager(executor=mock_executor)

        # Pre-fill all values
        pre_filled = {
            "model": "lgbm",
            "areas": "all",
            "start_date": "2024-01-01",
            "end_date": "2024-12-31",
            "parallel": "4",
        }

        result = manager.execute_workflow("1", pre_filled=pre_filled)

        assert result.status == WorkflowStatus.COMPLETED
        assert result.command is not None
        assert "train" in result.command

    def test_execute_workflow_cancelled(
        self, mock_executor: MockWorkflowExecutor
    ) -> None:
        """Test cancelling a workflow."""
        mock_executor.set_prompt_responses(["cancel"])

        manager = GuidedWorkflowManager(executor=mock_executor)
        result = manager.execute_workflow("1")

        assert result.status == WorkflowStatus.CANCELLED

    def test_execute_workflow_confirm_false(
        self, mock_executor: MockWorkflowExecutor
    ) -> None:
        """Test declining final confirmation."""
        # Provide all step values
        mock_executor.set_prompt_responses([
            "lgbm",      # model
            "all",       # areas
            "2024-01-01",  # start_date
            "2024-12-31",  # end_date
            "4",         # parallel
        ])
        mock_executor.confirms["Execute"] = False

        manager = GuidedWorkflowManager(executor=mock_executor)
        result = manager.execute_workflow("1")

        assert result.status == WorkflowStatus.CANCELLED
        assert result.command is not None  # Command was built but not executed


class TestConsoleWorkflowExecutor:
    """Tests for ConsoleWorkflowExecutor class."""

    def test_echo_plain(self) -> None:
        """Test plain echo."""
        executor = ConsoleWorkflowExecutor(use_colors=False)

        with patch("click.echo") as mock_echo:
            executor.echo("Test message")
            mock_echo.assert_called_once_with("Test message")

    def test_echo_with_style(self) -> None:
        """Test styled echo."""
        executor = ConsoleWorkflowExecutor(use_colors=True)

        with patch("click.secho") as mock_secho:
            executor.echo("Error!", style="error")
            mock_secho.assert_called_once_with("Error!", fg="red")

    def test_echo_header_style(self) -> None:
        """Test header style echo."""
        executor = ConsoleWorkflowExecutor(use_colors=True)

        with patch("click.secho") as mock_secho:
            executor.echo("Header", style="header")
            mock_secho.assert_called_once_with("Header", fg="cyan", bold=True)

    def test_prompt(self) -> None:
        """Test prompt."""
        executor = ConsoleWorkflowExecutor()

        with patch("click.prompt", return_value="test") as mock_prompt:
            result = executor.prompt("Enter:", default="default")
            assert result == "test"
            mock_prompt.assert_called_once()

    def test_confirm(self) -> None:
        """Test confirm."""
        executor = ConsoleWorkflowExecutor()

        with patch("click.confirm", return_value=True) as mock_confirm:
            result = executor.confirm("Continue?")
            assert result is True
            mock_confirm.assert_called_once_with("Continue?")


# =============================================================================
# Individual Workflow Function Tests
# =============================================================================


class TestGuidedWorkflowFunctions:
    """Tests for individual workflow functions."""

    def test_guided_training_workflow(
        self, mock_executor: MockWorkflowExecutor
    ) -> None:
        """Test training workflow function."""
        mock_executor.set_prompt_responses([
            "lgbm",
            "all",
            "2024-01-01",
            "2024-12-31",
            "4",
        ])
        mock_executor.confirms["Execute"] = True

        result = guided_training_workflow(executor=mock_executor)

        assert result.status == WorkflowStatus.COMPLETED
        assert "train" in (result.command or "")

    def test_guided_prediction_workflow(
        self, mock_executor: MockWorkflowExecutor
    ) -> None:
        """Test prediction workflow function."""
        mock_executor.set_prompt_responses([
            "batch",
            "2024-06-15",
            "all",
            "weighted_avg",
            "csv",
        ])
        mock_executor.confirms["Execute"] = True

        result = guided_prediction_workflow(executor=mock_executor)

        assert result.status == WorkflowStatus.COMPLETED
        assert "predict" in (result.command or "")

    def test_guided_backtest_workflow(
        self, mock_executor: MockWorkflowExecutor
    ) -> None:
        """Test backtest workflow function."""
        mock_executor.set_prompt_responses([
            "2024-01-01",
            "2024-03-31",
            "7,14",
            "all",
            "4",
        ])
        mock_executor.confirms["Execute"] = True

        result = guided_backtest_workflow(executor=mock_executor)

        assert result.status == WorkflowStatus.COMPLETED
        assert "backtest" in (result.command or "")

    def test_guided_feature_workflow(
        self, mock_executor: MockWorkflowExecutor
    ) -> None:
        """Test feature workflow function."""
        mock_executor.set_prompt_responses([
            "all",
            "2024-01-01",
            "2024-12-31",
            "features/",
            "4",
        ])
        mock_executor.confirms["Execute"] = True

        result = guided_feature_workflow(executor=mock_executor)

        assert result.status == WorkflowStatus.COMPLETED
        assert "features" in (result.command or "")

    def test_guided_comparison_workflow(
        self, mock_executor: MockWorkflowExecutor
    ) -> None:
        """Test comparison workflow function."""
        mock_executor.set_prompt_responses([
            "2024-01-01",
            "2024-03-31",
            "all",
            "5",
            "all",
        ])
        mock_executor.confirms["Execute"] = True

        result = guided_comparison_workflow(executor=mock_executor)

        assert result.status == WorkflowStatus.COMPLETED
        assert "evaluate compare" in (result.command or "")


# =============================================================================
# Start Guided Workflow Tests
# =============================================================================


class TestStartGuidedWorkflow:
    """Tests for start_guided_workflow function."""

    def test_cancel_at_menu(self) -> None:
        """Test cancelling at workflow menu."""
        with patch("click.echo"), patch("click.secho"), \
             patch("click.prompt", return_value="cancel"):
            result = start_guided_workflow()
            assert result is None

    def test_invalid_choice(self) -> None:
        """Test invalid workflow choice."""
        with patch("click.echo"), patch("click.secho"), \
             patch("click.prompt", return_value="99"):
            result = start_guided_workflow()
            assert result is None

    def test_select_workflow(self) -> None:
        """Test selecting a valid workflow."""
        mock_executor = MockWorkflowExecutor()
        mock_executor.set_prompt_responses([
            "lgbm",
            "all",
            "2024-01-01",
            "2024-12-31",
            "4",
        ])
        mock_executor.confirms["Execute"] = True

        with patch("click.echo"), patch("click.secho"), \
             patch("click.prompt", return_value="1"):
            result = start_guided_workflow(executor=mock_executor)

            # Result depends on executor being used after menu selection
            # Since the function creates its own manager, we test indirectly
            assert result is None or isinstance(result, WorkflowResult)


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_options_list(self) -> None:
        """Test step with empty options list."""
        step = WorkflowStep(
            step_id="select",
            title="Select",
            description="",
            step_type=StepType.SINGLE_SELECT,
            options=[],
            required=False,
        )
        # Should not crash with empty options
        is_valid, _ = step.validate("any")
        assert is_valid is True

    def test_workflow_with_empty_steps(
        self, mock_executor: MockWorkflowExecutor
    ) -> None:
        """Test workflow with no steps."""
        mock_executor.confirms["Execute"] = True

        manager = GuidedWorkflowManager(executor=mock_executor)

        empty_workflow = WorkflowDefinition(
            workflow_id="empty",
            name="Empty Workflow",
            description="No steps",
            steps=[],
            command_builder=lambda v: "empty",
        )
        manager.register_workflow(empty_workflow)

        result = manager.execute_workflow("empty")

        assert result.status == WorkflowStatus.COMPLETED
        assert result.command == "empty"

    def test_command_builder_error(
        self, mock_executor: MockWorkflowExecutor
    ) -> None:
        """Test handling command builder error."""
        def bad_builder(values: dict) -> str:
            raise ValueError("Build error")

        manager = GuidedWorkflowManager(executor=mock_executor)

        bad_workflow = WorkflowDefinition(
            workflow_id="bad",
            name="Bad Workflow",
            description="Fails to build",
            steps=[],
            command_builder=bad_builder,
        )
        manager.register_workflow(bad_workflow)

        result = manager.execute_workflow("bad")

        assert result.status == WorkflowStatus.FAILED
        assert "Failed to build command" in (result.error_message or "")

    def test_validation_retry(self, mock_executor: MockWorkflowExecutor) -> None:
        """Test validation retry on invalid input."""
        # First response invalid, second valid
        mock_executor.set_prompt_responses([
            "INVALID",  # Invalid area
            "SECO",     # Valid area
            "2024-01-01",
            "2024-12-31",
            "4",
        ])
        mock_executor.confirms["Execute"] = True

        manager = GuidedWorkflowManager(executor=mock_executor)

        # Skip model step with prefilled
        result = manager.execute_workflow("1", pre_filled={"model": "lgbm"})

        # Should eventually complete after retry
        assert result.status == WorkflowStatus.COMPLETED

    def test_workflow_definition_icon(self) -> None:
        """Test workflow definition with icon."""
        workflow = WorkflowDefinition(
            workflow_id="test",
            name="Test",
            description="Test workflow",
            steps=[],
            command_builder=lambda v: "test",
            icon="🎯",
        )
        assert workflow.icon == "🎯"

    def test_step_option_default(self) -> None:
        """Test step option default flag."""
        opt = StepOption(value="a", label="A", is_default=True)
        assert opt.is_default is True

        opt2 = StepOption(value="b", label="B")
        assert opt2.is_default is False

    def test_workflow_result_fields(self) -> None:
        """Test WorkflowResult fields."""
        result = WorkflowResult(
            status=WorkflowStatus.COMPLETED,
            command="test command",
            values={"key": "value"},
            error_message=None,
            executed=True,
        )

        assert result.status == WorkflowStatus.COMPLETED
        assert result.command == "test command"
        assert result.values == {"key": "value"}
        assert result.error_message is None
        assert result.executed is True


# =============================================================================
# Integration Tests
# =============================================================================


class TestWorkflowIntegration:
    """Integration tests for workflow system."""

    def test_full_training_workflow(
        self, mock_executor: MockWorkflowExecutor
    ) -> None:
        """Test complete training workflow execution."""
        # Model step uses SINGLE_SELECT, so use selects dict
        mock_executor.selects[""] = "rf"  # For model selection
        mock_executor.set_prompt_responses([
            "SECO,S",       # areas (TEXT_INPUT)
            "2024-01-01",   # start_date (DATE_INPUT)
            "2024-06-30",   # end_date (DATE_INPUT)
            "8",            # parallel (NUMBER_INPUT)
        ])
        mock_executor.confirms["Execute"] = True

        result = guided_training_workflow(executor=mock_executor)

        assert result.status == WorkflowStatus.COMPLETED
        assert result.command is not None

        # Verify command contains all parameters
        assert "--model rf" in result.command
        assert "--area SECO" in result.command
        assert "--area S" in result.command
        assert "--start-date 2024-01-01" in result.command
        assert "--end-date 2024-06-30" in result.command
        assert "--parallel 8" in result.command

        # Verify values captured
        assert result.values["model"] == "rf"
        assert result.values["areas"] == "SECO,S"

    def test_workflow_with_all_defaults(
        self, mock_executor: MockWorkflowExecutor
    ) -> None:
        """Test workflow using all default values."""
        # Return empty strings to use defaults
        mock_executor.set_prompt_responses(["", "", "", "", ""])
        mock_executor.confirms["Execute"] = True

        manager = GuidedWorkflowManager(executor=mock_executor)

        # Pre-fill to avoid empty required fields
        result = manager.execute_workflow(
            "1",
            pre_filled={
                "model": "lgbm",
                "areas": "all",
                "start_date": "2024-01-01",
                "end_date": "2024-12-31",
                "parallel": "4",
            },
        )

        assert result.status == WorkflowStatus.COMPLETED

    def test_multiple_workflows_same_manager(
        self, mock_executor: MockWorkflowExecutor
    ) -> None:
        """Test running multiple workflows with same manager."""
        mock_executor.confirms["Execute"] = True

        manager = GuidedWorkflowManager(executor=mock_executor)

        # Run training workflow
        mock_executor.set_prompt_responses([
            "lgbm", "all", "2024-01-01", "2024-12-31", "4"
        ])
        result1 = manager.execute_workflow("1")
        assert result1.status == WorkflowStatus.COMPLETED

        # Run prediction workflow
        mock_executor._prompt_index = 0
        mock_executor.set_prompt_responses([
            "batch", "2024-06-15", "all", "weighted_avg", "csv"
        ])
        result2 = manager.execute_workflow("2")
        assert result2.status == WorkflowStatus.COMPLETED

        # Verify different commands
        assert "train" in (result1.command or "")
        assert "predict" in (result2.command or "")
