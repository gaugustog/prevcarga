"""Guided workflows for interactive CLI mode.

This module provides step-by-step guided workflows for common operations
in the PrevCarga forecasting system.

Key Components:
- GuidedWorkflowManager: Manages workflow registration and execution
- WorkflowStep: Represents a single step in a workflow
- WorkflowResult: Contains the result of workflow execution
- Individual workflow functions for training, prediction, backtest, features, comparison

Example:
    ```python
    from src.cli.interactive.workflows import GuidedWorkflowManager

    manager = GuidedWorkflowManager()
    workflows = manager.list_workflows()

    # Execute a specific workflow
    result = manager.execute_workflow('1')  # Training workflow
    if result.success:
        print(f"Command: {result.command}")
    ```
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from collections.abc import Sequence


class WorkflowStatus(Enum):
    """Status of a workflow execution."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class StepType(Enum):
    """Type of workflow step."""

    SINGLE_SELECT = "single_select"
    MULTI_SELECT = "multi_select"
    TEXT_INPUT = "text_input"
    DATE_INPUT = "date_input"
    NUMBER_INPUT = "number_input"
    CONFIRM = "confirm"


@dataclass
class StepOption:
    """Option for a selection step."""

    value: str
    label: str
    description: str = ""
    is_default: bool = False


@dataclass
class WorkflowStep:
    """Represents a single step in a guided workflow."""

    step_id: str
    title: str
    description: str
    step_type: StepType
    options: list[StepOption] = field(default_factory=list)
    default_value: str | None = None
    required: bool = True
    validation_fn: Callable[[str], tuple[bool, str]] | None = None

    def validate(self, value: str) -> tuple[bool, str]:
        """Validate the step input.

        Args:
            value: The input value to validate.

        Returns:
            Tuple of (is_valid, error_message).
        """
        if self.required and not value:
            return False, "This field is required"

        if self.validation_fn:
            return self.validation_fn(value)

        if self.step_type == StepType.DATE_INPUT:
            return validate_date(value)

        if self.step_type == StepType.NUMBER_INPUT:
            return validate_number(value)

        if self.step_type == StepType.SINGLE_SELECT and self.options:
            valid_values = [opt.value for opt in self.options]
            if value not in valid_values:
                return False, f"Invalid selection. Choose from: {', '.join(valid_values)}"

        return True, ""


@dataclass
class WorkflowResult:
    """Result of a workflow execution."""

    status: WorkflowStatus
    command: str | None = None
    values: dict[str, Any] = field(default_factory=dict)
    error_message: str | None = None
    executed: bool = False

    @property
    def success(self) -> bool:
        """Check if workflow completed successfully."""
        return self.status == WorkflowStatus.COMPLETED

    @property
    def cancelled(self) -> bool:
        """Check if workflow was cancelled."""
        return self.status == WorkflowStatus.CANCELLED


@dataclass
class WorkflowDefinition:
    """Definition of a guided workflow."""

    workflow_id: str
    name: str
    description: str
    steps: list[WorkflowStep]
    command_builder: Callable[[dict[str, Any]], str]
    icon: str = "📋"


# Validation functions
def validate_date(value: str) -> tuple[bool, str]:
    """Validate date format YYYY-MM-DD."""
    if not value:
        return True, ""  # Empty is ok if not required
    try:
        datetime.strptime(value, "%Y-%m-%d")
        return True, ""
    except ValueError:
        return False, "Invalid date format. Use YYYY-MM-DD"


def validate_number(value: str, min_val: int | None = None, max_val: int | None = None) -> tuple[bool, str]:
    """Validate numeric input."""
    if not value:
        return True, ""
    try:
        num = int(value)
        if min_val is not None and num < min_val:
            return False, f"Value must be at least {min_val}"
        if max_val is not None and num > max_val:
            return False, f"Value must be at most {max_val}"
        return True, ""
    except ValueError:
        return False, "Invalid number"


def validate_areas(value: str) -> tuple[bool, str]:
    """Validate area selection."""
    if not value:
        return False, "At least one area is required"

    if value.lower() == "all":
        return True, ""

    valid_areas = {"SECO", "S", "NE", "N", "SIN"}
    areas = [a.strip().upper() for a in value.split(",")]

    for area in areas:
        if area not in valid_areas:
            return False, f"Invalid area: {area}. Valid areas: {', '.join(sorted(valid_areas))}"

    return True, ""


def validate_horizons(value: str) -> tuple[bool, str]:
    """Validate horizon selection."""
    if not value:
        return False, "At least one horizon is required"

    if value.lower() == "all":
        return True, ""

    try:
        horizons = [int(h.strip()) for h in value.split(",")]
        for h in horizons:
            if h < 0 or h > 8:
                return False, f"Invalid horizon: {h}. Valid range: 0-8"
        return True, ""
    except ValueError:
        return False, "Invalid horizon format. Use comma-separated numbers"


def validate_positive_number(value: str) -> tuple[bool, str]:
    """Validate positive number."""
    if not value:
        return True, ""
    try:
        num = int(value)
        if num < 1:
            return False, "Value must be positive"
        return True, ""
    except ValueError:
        return False, "Invalid number"


# Step definitions for each workflow
def create_training_steps() -> list[WorkflowStep]:
    """Create steps for training workflow."""
    return [
        WorkflowStep(
            step_id="model",
            title="Select Model",
            description="Choose the model to train",
            step_type=StepType.SINGLE_SELECT,
            options=[
                StepOption("lgbm", "LightGBM", "Gradient boosting model", is_default=True),
                StepOption("rf", "Random Forest", "Ensemble tree model"),
                StepOption("arima", "ARIMA", "Time series model"),
                StepOption("holt_winters", "Holt-Winters", "Exponential smoothing"),
                StepOption("all", "All Models", "Train all available models"),
            ],
            default_value="lgbm",
        ),
        WorkflowStep(
            step_id="areas",
            title="Select Areas",
            description="Enter comma-separated areas or 'all' for all areas\nAvailable: SECO, S, NE, N, SIN",
            step_type=StepType.TEXT_INPUT,
            default_value="all",
            validation_fn=validate_areas,
        ),
        WorkflowStep(
            step_id="start_date",
            title="Training Start Date",
            description="Start date for training data (YYYY-MM-DD)",
            step_type=StepType.DATE_INPUT,
            default_value="2023-01-01",
        ),
        WorkflowStep(
            step_id="end_date",
            title="Training End Date",
            description="End date for training data (YYYY-MM-DD)",
            step_type=StepType.DATE_INPUT,
            default_value="2023-12-31",
        ),
        WorkflowStep(
            step_id="parallel",
            title="Parallel Workers",
            description="Number of parallel workers for training",
            step_type=StepType.NUMBER_INPUT,
            default_value="4",
            validation_fn=validate_positive_number,
        ),
    ]


def build_training_command(values: dict[str, Any]) -> str:
    """Build training command from workflow values."""
    cmd = f"train --model {values['model']}"
    cmd += f" --start-date {values['start_date']}"
    cmd += f" --end-date {values['end_date']}"
    cmd += f" --parallel {values['parallel']}"

    if values.get("areas", "").lower() != "all":
        for area in values["areas"].split(","):
            area = area.strip()
            if area:
                cmd += f" --area {area}"

    return cmd


def create_prediction_steps() -> list[WorkflowStep]:
    """Create steps for prediction workflow."""
    return [
        WorkflowStep(
            step_id="pred_type",
            title="Prediction Type",
            description="Select the type of prediction to generate",
            step_type=StepType.SINGLE_SELECT,
            options=[
                StepOption("batch", "Batch Prediction", "D+0 to D+8 full horizon prediction", is_default=True),
                StepOption("intraday", "Intraday Prediction", "D+0 update with latest data"),
            ],
            default_value="batch",
        ),
        WorkflowStep(
            step_id="date",
            title="Prediction Date",
            description="Date for prediction (YYYY-MM-DD)",
            step_type=StepType.DATE_INPUT,
            default_value=datetime.now().strftime("%Y-%m-%d"),
        ),
        WorkflowStep(
            step_id="horizons",
            title="Forecast Horizons",
            description="Enter comma-separated horizons (0-8) or 'all'",
            step_type=StepType.TEXT_INPUT,
            default_value="all",
            validation_fn=validate_horizons,
        ),
        WorkflowStep(
            step_id="combination",
            title="Combination Method",
            description="Method to combine model predictions",
            step_type=StepType.SINGLE_SELECT,
            options=[
                StepOption("weighted_avg", "Weighted Average", "Weighted combination of models", is_default=True),
                StepOption("simple_avg", "Simple Average", "Equal weight combination"),
                StepOption("stacking", "Stacking", "Meta-learner combination"),
                StepOption("best", "Best Model", "Use best performing model only"),
            ],
            default_value="weighted_avg",
        ),
        WorkflowStep(
            step_id="format",
            title="Output Format",
            description="Format for prediction output",
            step_type=StepType.SINGLE_SELECT,
            options=[
                StepOption("csv", "CSV", "Comma-separated values", is_default=True),
                StepOption("json", "JSON", "JavaScript Object Notation"),
                StepOption("parquet", "Parquet", "Apache Parquet columnar format"),
            ],
            default_value="csv",
        ),
    ]


def build_prediction_command(values: dict[str, Any]) -> str:
    """Build prediction command from workflow values."""
    pred_type = values.get("pred_type", "batch")

    if pred_type == "intraday":
        cmd = f"predict intraday --date {values['date']}"
    else:
        cmd = f"predict batch --date {values['date']}"
        cmd += f" --combination {values['combination']}"
        cmd += f" --format {values['format']}"

        if values.get("horizons", "").lower() != "all":
            for h in values["horizons"].split(","):
                h = h.strip()
                if h:
                    cmd += f" --horizon {h}"

    return cmd


def create_backtest_steps() -> list[WorkflowStep]:
    """Create steps for backtest workflow."""
    return [
        WorkflowStep(
            step_id="start_date",
            title="Backtest Start Date",
            description="Start date for backtest period (YYYY-MM-DD)",
            step_type=StepType.DATE_INPUT,
            default_value="2024-01-01",
        ),
        WorkflowStep(
            step_id="end_date",
            title="Backtest End Date",
            description="End date for backtest period (YYYY-MM-DD)",
            step_type=StepType.DATE_INPUT,
            default_value="2024-03-31",
        ),
        WorkflowStep(
            step_id="intervals",
            title="Retraining Intervals",
            description="Comma-separated retraining intervals in days",
            step_type=StepType.TEXT_INPUT,
            default_value="7,14,30",
        ),
        WorkflowStep(
            step_id="models",
            title="Models to Backtest",
            description="Comma-separated model names or 'all'",
            step_type=StepType.TEXT_INPUT,
            default_value="all",
        ),
        WorkflowStep(
            step_id="parallel",
            title="Parallel Workers",
            description="Number of parallel workers",
            step_type=StepType.NUMBER_INPUT,
            default_value="4",
            validation_fn=validate_positive_number,
        ),
    ]


def build_backtest_command(values: dict[str, Any]) -> str:
    """Build backtest command from workflow values."""
    cmd = f"backtest --start-date {values['start_date']}"
    cmd += f" --end-date {values['end_date']}"
    cmd += f" --parallel {values['parallel']}"

    for interval in values.get("intervals", "").split(","):
        interval = interval.strip()
        if interval:
            cmd += f" --retraining-interval {interval}"

    if values.get("models", "").lower() != "all":
        for model in values["models"].split(","):
            model = model.strip()
            if model:
                cmd += f" --model {model}"

    return cmd


def create_feature_steps() -> list[WorkflowStep]:
    """Create steps for feature engineering workflow."""
    return [
        WorkflowStep(
            step_id="plugins",
            title="Feature Plugins",
            description="Comma-separated plugins or 'all'\nAvailable: temporal, calendar, weather, lag, rolling, fourier, wavelet, loess, blf, seasonality",
            step_type=StepType.TEXT_INPUT,
            default_value="all",
        ),
        WorkflowStep(
            step_id="start_date",
            title="Feature Start Date",
            description="Start date for feature generation (YYYY-MM-DD)",
            step_type=StepType.DATE_INPUT,
            default_value="2023-01-01",
        ),
        WorkflowStep(
            step_id="end_date",
            title="Feature End Date",
            description="End date for feature generation (YYYY-MM-DD)",
            step_type=StepType.DATE_INPUT,
            default_value="2023-12-31",
        ),
        WorkflowStep(
            step_id="output_dir",
            title="Output Directory",
            description="Directory for generated features",
            step_type=StepType.TEXT_INPUT,
            default_value="features/",
        ),
        WorkflowStep(
            step_id="parallel",
            title="Parallel Workers",
            description="Number of parallel workers",
            step_type=StepType.NUMBER_INPUT,
            default_value="4",
            validation_fn=validate_positive_number,
        ),
    ]


def build_feature_command(values: dict[str, Any]) -> str:
    """Build feature generation command from workflow values."""
    cmd = f"features generate --start-date {values['start_date']}"
    cmd += f" --end-date {values['end_date']}"
    cmd += f" --parallel {values['parallel']}"
    cmd += f" --output-dir {values['output_dir']}"

    if values.get("plugins", "").lower() != "all":
        for plugin in values["plugins"].split(","):
            plugin = plugin.strip()
            if plugin:
                cmd += f" --plugin {plugin}"

    return cmd


def create_comparison_steps() -> list[WorkflowStep]:
    """Create steps for model comparison workflow."""
    return [
        WorkflowStep(
            step_id="start_date",
            title="Evaluation Start Date",
            description="Start date for evaluation period (YYYY-MM-DD)",
            step_type=StepType.DATE_INPUT,
            default_value="2024-01-01",
        ),
        WorkflowStep(
            step_id="end_date",
            title="Evaluation End Date",
            description="End date for evaluation period (YYYY-MM-DD)",
            step_type=StepType.DATE_INPUT,
            default_value="2024-03-31",
        ),
        WorkflowStep(
            step_id="strategies",
            title="Combination Strategies",
            description="Comma-separated strategies or 'all'\nAvailable: simple_avg, weighted_avg, stacking, best",
            step_type=StepType.TEXT_INPUT,
            default_value="all",
        ),
        WorkflowStep(
            step_id="cv_folds",
            title="Cross-Validation Folds",
            description="Number of CV folds for evaluation",
            step_type=StepType.NUMBER_INPUT,
            default_value="5",
            validation_fn=validate_positive_number,
        ),
        WorkflowStep(
            step_id="metrics",
            title="Evaluation Metrics",
            description="Comma-separated metrics or 'all'\nAvailable: mape, smape, mae, rmse, r2",
            step_type=StepType.TEXT_INPUT,
            default_value="all",
        ),
    ]


def build_comparison_command(values: dict[str, Any]) -> str:
    """Build comparison command from workflow values."""
    cmd = f"evaluate compare --start-date {values['start_date']}"
    cmd += f" --end-date {values['end_date']}"
    cmd += f" --cv-folds {values['cv_folds']}"

    if values.get("strategies", "").lower() != "all":
        for strategy in values["strategies"].split(","):
            strategy = strategy.strip()
            if strategy:
                cmd += f" --strategy {strategy}"

    if values.get("metrics", "").lower() != "all":
        for metric in values["metrics"].split(","):
            metric = metric.strip()
            if metric:
                cmd += f" --metric {metric}"

    return cmd


class WorkflowExecutor(ABC):
    """Abstract base class for workflow executors."""

    @abstractmethod
    def prompt(self, message: str, default: str | None = None) -> str:
        """Prompt the user for input."""

    @abstractmethod
    def select(self, message: str, options: Sequence[StepOption]) -> str:
        """Prompt the user for a selection."""

    @abstractmethod
    def confirm(self, message: str) -> bool:
        """Prompt the user for confirmation."""

    @abstractmethod
    def echo(self, message: str, style: str | None = None) -> None:
        """Display a message to the user."""


class ConsoleWorkflowExecutor(WorkflowExecutor):
    """Console-based workflow executor using click."""

    def __init__(self, use_colors: bool = True) -> None:
        """Initialize the console executor.

        Args:
            use_colors: Whether to use colored output.
        """
        self.use_colors = use_colors

    def prompt(self, message: str, default: str | None = None) -> str:
        """Prompt the user for input."""
        import click

        return click.prompt(message, default=default or "", show_default=bool(default))

    def select(self, message: str, options: Sequence[StepOption]) -> str:
        """Prompt the user for a selection."""
        import click

        self.echo(message)
        for opt in options:
            marker = "*" if opt.is_default else " "
            desc = f" - {opt.description}" if opt.description else ""
            self.echo(f"  {marker} {opt.value}: {opt.label}{desc}")

        default_value = next((o.value for o in options if o.is_default), options[0].value if options else None)
        return click.prompt("Select", default=default_value)

    def confirm(self, message: str) -> bool:
        """Prompt the user for confirmation."""
        import click

        return click.confirm(message)

    def echo(self, message: str, style: str | None = None) -> None:
        """Display a message to the user."""
        import click

        if style == "error" and self.use_colors:
            click.secho(message, fg="red")
        elif style == "success" and self.use_colors:
            click.secho(message, fg="green")
        elif style == "warning" and self.use_colors:
            click.secho(message, fg="yellow")
        elif style == "header" and self.use_colors:
            click.secho(message, fg="cyan", bold=True)
        else:
            click.echo(message)


class GuidedWorkflowManager:
    """Manages guided workflows for complex operations."""

    def __init__(
        self,
        executor: WorkflowExecutor | None = None,
        use_colors: bool = True,
    ) -> None:
        """Initialize the workflow manager.

        Args:
            executor: Custom workflow executor. Defaults to console executor.
            use_colors: Whether to use colored output.
        """
        self.executor = executor or ConsoleWorkflowExecutor(use_colors=use_colors)
        self.use_colors = use_colors
        self._workflows: dict[str, WorkflowDefinition] = {}
        self._register_default_workflows()

    def _register_default_workflows(self) -> None:
        """Register default workflows."""
        self.register_workflow(
            WorkflowDefinition(
                workflow_id="1",
                name="Complete Model Training",
                description="Train models with step-by-step guidance for model selection, areas, and parameters",
                steps=create_training_steps(),
                command_builder=build_training_command,
                icon="🎯",
            )
        )
        self.register_workflow(
            WorkflowDefinition(
                workflow_id="2",
                name="Generate Predictions",
                description="Generate batch or intraday predictions with combination methods",
                steps=create_prediction_steps(),
                command_builder=build_prediction_command,
                icon="📈",
            )
        )
        self.register_workflow(
            WorkflowDefinition(
                workflow_id="3",
                name="Run Backtest Evaluation",
                description="Backtest models with configurable retraining intervals",
                steps=create_backtest_steps(),
                command_builder=build_backtest_command,
                icon="📊",
            )
        )
        self.register_workflow(
            WorkflowDefinition(
                workflow_id="4",
                name="Feature Engineering Pipeline",
                description="Generate features using various plugins and transformations",
                steps=create_feature_steps(),
                command_builder=build_feature_command,
                icon="⚙️",
            )
        )
        self.register_workflow(
            WorkflowDefinition(
                workflow_id="5",
                name="Model Comparison Analysis",
                description="Compare combination strategies with cross-validation",
                steps=create_comparison_steps(),
                command_builder=build_comparison_command,
                icon="🔬",
            )
        )

    def register_workflow(self, workflow: WorkflowDefinition) -> None:
        """Register a workflow.

        Args:
            workflow: The workflow definition to register.
        """
        self._workflows[workflow.workflow_id] = workflow

    def unregister_workflow(self, workflow_id: str) -> bool:
        """Unregister a workflow.

        Args:
            workflow_id: The ID of the workflow to unregister.

        Returns:
            True if workflow was removed, False if not found.
        """
        if workflow_id in self._workflows:
            del self._workflows[workflow_id]
            return True
        return False

    def list_workflows(self) -> dict[str, str]:
        """List available guided workflows.

        Returns:
            Dictionary mapping workflow IDs to names.
        """
        return {wf_id: wf.name for wf_id, wf in self._workflows.items()}

    def get_workflow(self, workflow_id: str) -> WorkflowDefinition | None:
        """Get a workflow definition by ID.

        Args:
            workflow_id: The workflow ID.

        Returns:
            The workflow definition or None if not found.
        """
        return self._workflows.get(workflow_id)

    def execute_workflow(
        self,
        workflow_id: str,
        pre_filled: dict[str, Any] | None = None,
    ) -> WorkflowResult:
        """Execute a guided workflow.

        Args:
            workflow_id: The workflow ID to execute.
            pre_filled: Optional pre-filled values for steps.

        Returns:
            WorkflowResult with status, command, and values.
        """
        if workflow_id not in self._workflows:
            return WorkflowResult(
                status=WorkflowStatus.FAILED,
                error_message=f"Unknown workflow: {workflow_id}",
            )

        workflow = self._workflows[workflow_id]
        values: dict[str, Any] = pre_filled.copy() if pre_filled else {}

        # Display workflow header
        self._display_workflow_header(workflow)

        # Execute each step
        for i, step in enumerate(workflow.steps, 1):
            # Skip if pre-filled
            if step.step_id in values:
                continue

            try:
                result = self._execute_step(step, i, len(workflow.steps))
                if result is None:  # Cancelled
                    return WorkflowResult(
                        status=WorkflowStatus.CANCELLED,
                        values=values,
                    )
                values[step.step_id] = result
            except KeyboardInterrupt:
                self.executor.echo("\n\nWorkflow cancelled", style="warning")
                return WorkflowResult(
                    status=WorkflowStatus.CANCELLED,
                    values=values,
                )

        # Build command
        try:
            command = workflow.command_builder(values)
        except Exception as e:
            return WorkflowResult(
                status=WorkflowStatus.FAILED,
                values=values,
                error_message=f"Failed to build command: {e}",
            )

        # Display summary and confirm
        self._display_summary(workflow, values, command)

        if not self.executor.confirm("Execute this command?"):
            self.executor.echo("Workflow cancelled", style="warning")
            return WorkflowResult(
                status=WorkflowStatus.CANCELLED,
                command=command,
                values=values,
            )

        return WorkflowResult(
            status=WorkflowStatus.COMPLETED,
            command=command,
            values=values,
            executed=False,  # Caller should execute the command
        )

    def _display_workflow_header(self, workflow: WorkflowDefinition) -> None:
        """Display the workflow header."""
        self.executor.echo("")
        self.executor.echo("╔═══════════════════════════════════════════════════════════╗", style="header")
        title = f"  {workflow.icon} {workflow.name}"
        self.executor.echo(f"║{title:<59}║", style="header")
        self.executor.echo("╚═══════════════════════════════════════════════════════════╝", style="header")
        self.executor.echo("")
        self.executor.echo(workflow.description)
        self.executor.echo("")

    def _execute_step(self, step: WorkflowStep, step_num: int, total_steps: int) -> str | None:
        """Execute a single workflow step.

        Args:
            step: The step to execute.
            step_num: Current step number.
            total_steps: Total number of steps.

        Returns:
            The step value or None if cancelled.
        """
        # Display step header
        self.executor.echo(f"Step {step_num}/{total_steps}: {step.title}", style="header")
        if step.description:
            self.executor.echo(f"  {step.description}")

        while True:
            # Get input based on step type
            if step.step_type == StepType.SINGLE_SELECT and step.options:
                value = self.executor.select("", step.options)
            elif step.step_type == StepType.CONFIRM:
                value = "yes" if self.executor.confirm(step.title) else "no"
            else:
                value = self.executor.prompt(step.title, default=step.default_value)

            # Check for cancel
            if value.lower() == "cancel":
                return None

            # Validate
            is_valid, error_msg = step.validate(value)
            if is_valid:
                self.executor.echo("")
                return value
            else:
                self.executor.echo(f"  Error: {error_msg}", style="error")

    def _display_summary(
        self,
        workflow: WorkflowDefinition,
        values: dict[str, Any],
        command: str,
    ) -> None:
        """Display workflow summary before execution."""
        self.executor.echo("")
        self.executor.echo("╔═══════════════════════════════════════════════════════════╗", style="header")
        self.executor.echo("║                    Summary                                ║", style="header")
        self.executor.echo("╚═══════════════════════════════════════════════════════════╝", style="header")
        self.executor.echo("")

        for step in workflow.steps:
            if step.step_id in values:
                value = values[step.step_id]
                self.executor.echo(f"  {step.title}: {value}")

        self.executor.echo("")
        self.executor.echo("Command to execute:", style="header")
        self.executor.echo(f"  {command}")
        self.executor.echo("")


def start_guided_workflow(
    use_colors: bool = True,
    executor: WorkflowExecutor | None = None,
) -> WorkflowResult | None:
    """Start guided workflow selection.

    Args:
        use_colors: Whether to use colored output.
        executor: Optional custom workflow executor.

    Returns:
        WorkflowResult if a workflow was executed, None otherwise.
    """
    import click

    manager = GuidedWorkflowManager(executor=executor, use_colors=use_colors)
    workflows = manager.list_workflows()

    click.echo()
    if use_colors:
        click.secho("╔═══════════════════════════════════════════════════════════╗", fg="cyan", bold=True)
        click.secho("║                  Guided Workflows                         ║", fg="cyan", bold=True)
        click.secho("╚═══════════════════════════════════════════════════════════╝", fg="cyan", bold=True)
    else:
        click.echo("╔═══════════════════════════════════════════════════════════╗")
        click.echo("║                  Guided Workflows                         ║")
        click.echo("╚═══════════════════════════════════════════════════════════╝")
    click.echo()

    for key in sorted(workflows.keys()):
        wf = manager.get_workflow(key)
        if wf:
            click.echo(f"  {key}. {wf.icon} {wf.name}")
            if use_colors:
                click.secho(f"     {wf.description}", fg="bright_black")
            else:
                click.echo(f"     {wf.description}")

    click.echo()
    choice = click.prompt("Select workflow (1-5) or 'cancel'", default="cancel")

    if choice.lower() == "cancel":
        click.echo("Cancelled")
        return None

    if choice not in workflows:
        if use_colors:
            click.secho(f"Invalid choice: {choice}", fg="red")
        else:
            click.echo(f"Invalid choice: {choice}")
        return None

    try:
        return manager.execute_workflow(choice)
    except KeyboardInterrupt:
        click.echo("\n\nWorkflow cancelled")
        return None
    except Exception as e:
        if use_colors:
            click.secho(f"\nWorkflow error: {e}", fg="red")
        else:
            click.echo(f"\nWorkflow error: {e}")
        return None


# Individual workflow functions for direct access
def guided_training_workflow(
    executor: WorkflowExecutor | None = None,
    use_colors: bool = True,
) -> WorkflowResult:
    """Execute guided training workflow directly.

    Args:
        executor: Optional custom workflow executor.
        use_colors: Whether to use colored output.

    Returns:
        WorkflowResult with status and command.
    """
    manager = GuidedWorkflowManager(executor=executor, use_colors=use_colors)
    return manager.execute_workflow("1")


def guided_prediction_workflow(
    executor: WorkflowExecutor | None = None,
    use_colors: bool = True,
) -> WorkflowResult:
    """Execute guided prediction workflow directly.

    Args:
        executor: Optional custom workflow executor.
        use_colors: Whether to use colored output.

    Returns:
        WorkflowResult with status and command.
    """
    manager = GuidedWorkflowManager(executor=executor, use_colors=use_colors)
    return manager.execute_workflow("2")


def guided_backtest_workflow(
    executor: WorkflowExecutor | None = None,
    use_colors: bool = True,
) -> WorkflowResult:
    """Execute guided backtest workflow directly.

    Args:
        executor: Optional custom workflow executor.
        use_colors: Whether to use colored output.

    Returns:
        WorkflowResult with status and command.
    """
    manager = GuidedWorkflowManager(executor=executor, use_colors=use_colors)
    return manager.execute_workflow("3")


def guided_feature_workflow(
    executor: WorkflowExecutor | None = None,
    use_colors: bool = True,
) -> WorkflowResult:
    """Execute guided feature engineering workflow directly.

    Args:
        executor: Optional custom workflow executor.
        use_colors: Whether to use colored output.

    Returns:
        WorkflowResult with status and command.
    """
    manager = GuidedWorkflowManager(executor=executor, use_colors=use_colors)
    return manager.execute_workflow("4")


def guided_comparison_workflow(
    executor: WorkflowExecutor | None = None,
    use_colors: bool = True,
) -> WorkflowResult:
    """Execute guided comparison workflow directly.

    Args:
        executor: Optional custom workflow executor.
        use_colors: Whether to use colored output.

    Returns:
        WorkflowResult with status and command.
    """
    manager = GuidedWorkflowManager(executor=executor, use_colors=use_colors)
    return manager.execute_workflow("5")
