"""Command completion for interactive CLI.

This module provides auto-completion capabilities for the interactive shell,
including command names, arguments, file paths, and context-aware suggestions.

Key Components:
- CompletionItem: Single completion suggestion
- CompletionContext: Context for completion
- CommandCompleter: Main command completion engine
- ArgumentCompleter: Argument-specific completion

Example:
    ```python
    from src.cli.interactive.completer import CommandCompleter

    # Create completer
    completer = CommandCompleter()

    # Get completions
    completions = completer.complete("train --ar")
    for item in completions:
        print(f"{item.text}: {item.description}")
    ```
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable


class CompletionType(Enum):
    """Types of completion items.

    Attributes:
        COMMAND: CLI command.
        SUBCOMMAND: Command subcommand.
        OPTION: Command option/flag.
        ARGUMENT: Command argument.
        VALUE: Option value.
        PATH: File path.
        AREA: Geographic area.
        MODEL: Model name.
        HORIZON: Forecast horizon.
    """

    COMMAND = "command"
    SUBCOMMAND = "subcommand"
    OPTION = "option"
    ARGUMENT = "argument"
    VALUE = "value"
    PATH = "path"
    AREA = "area"
    MODEL = "model"
    HORIZON = "horizon"


@dataclass
class CompletionItem:
    """Single completion suggestion.

    Attributes:
        text: Completion text to insert.
        display: Text to display in completion menu.
        description: Description of the completion.
        completion_type: Type of completion.
        meta: Additional metadata.
    """

    text: str
    display: str | None = None
    description: str = ""
    completion_type: CompletionType = CompletionType.COMMAND
    meta: dict = field(default_factory=dict)

    @property
    def display_text(self) -> str:
        """Get display text.

        Returns:
            Display text.
        """
        return self.display or self.text


@dataclass
class CompletionContext:
    """Context for completion operations.

    Attributes:
        line: Full input line.
        word: Current word being completed.
        words: All words in line.
        word_index: Index of current word.
        cursor_position: Cursor position in line.
        command: Identified command (if any).
        subcommand: Identified subcommand (if any).
        current_option: Current option being completed.
    """

    line: str
    word: str = ""
    words: list[str] = field(default_factory=list)
    word_index: int = 0
    cursor_position: int = 0
    command: str | None = None
    subcommand: str | None = None
    current_option: str | None = None

    @classmethod
    def from_line(cls, line: str, cursor_position: int | None = None) -> "CompletionContext":
        """Create context from input line.

        Args:
            line: Input line.
            cursor_position: Cursor position (defaults to end).

        Returns:
            CompletionContext instance.
        """
        cursor_position = cursor_position or len(line)
        words = line[:cursor_position].split()
        word_index = len(words) - 1 if words else 0
        word = words[-1] if words and not line[:cursor_position].endswith(" ") else ""

        # Identify command and subcommand
        command = None
        subcommand = None
        current_option = None

        if words:
            # First word is command
            if not words[0].startswith("-"):
                command = words[0]

            # Look for subcommand and current option
            for i, w in enumerate(words[1:], 1):
                if not w.startswith("-"):
                    if subcommand is None:
                        subcommand = w
                elif w.startswith("-"):
                    current_option = w

        return cls(
            line=line,
            word=word,
            words=words,
            word_index=word_index,
            cursor_position=cursor_position,
            command=command,
            subcommand=subcommand,
            current_option=current_option,
        )


class ArgumentCompleter:
    """Completer for specific argument types.

    Provides completions for areas, models, horizons, and other
    domain-specific values.
    """

    # Valid areas for Brazilian power system
    AREAS = ["SECO", "S", "NE", "N", "SIN"]

    # Valid model types
    MODELS = [
        "lgbm",
        "random_forest",
        "arima",
        "holt_winters",
        "regdin_svm",
        "blf",
    ]

    # Valid horizons (D+0 to D+8)
    HORIZONS = list(range(9))

    # Valid metrics
    METRICS = ["mape", "smape", "mae", "rmse", "r2"]

    # Valid output formats
    FORMATS = ["text", "json", "table", "csv", "html", "parquet"]

    def complete_areas(self, prefix: str = "") -> list[CompletionItem]:
        """Complete area names.

        Args:
            prefix: Prefix to filter areas.

        Returns:
            List of area completions.
        """
        items = []
        for area in self.AREAS:
            if area.lower().startswith(prefix.lower()):
                items.append(CompletionItem(
                    text=area,
                    description=self._get_area_description(area),
                    completion_type=CompletionType.AREA,
                ))
        return items

    def complete_models(self, prefix: str = "") -> list[CompletionItem]:
        """Complete model names.

        Args:
            prefix: Prefix to filter models.

        Returns:
            List of model completions.
        """
        items = []
        for model in self.MODELS:
            if model.lower().startswith(prefix.lower()):
                items.append(CompletionItem(
                    text=model,
                    description=self._get_model_description(model),
                    completion_type=CompletionType.MODEL,
                ))
        return items

    def complete_horizons(self, prefix: str = "") -> list[CompletionItem]:
        """Complete horizon values.

        Args:
            prefix: Prefix to filter horizons.

        Returns:
            List of horizon completions.
        """
        items = []
        for horizon in self.HORIZONS:
            h_str = str(horizon)
            if h_str.startswith(prefix):
                items.append(CompletionItem(
                    text=h_str,
                    display=f"D+{horizon}",
                    description=f"Forecast horizon day {horizon}",
                    completion_type=CompletionType.HORIZON,
                ))
        return items

    def complete_metrics(self, prefix: str = "") -> list[CompletionItem]:
        """Complete metric names.

        Args:
            prefix: Prefix to filter metrics.

        Returns:
            List of metric completions.
        """
        items = []
        for metric in self.METRICS:
            if metric.lower().startswith(prefix.lower()):
                items.append(CompletionItem(
                    text=metric,
                    description=self._get_metric_description(metric),
                    completion_type=CompletionType.VALUE,
                ))
        return items

    def complete_formats(self, prefix: str = "") -> list[CompletionItem]:
        """Complete output format names.

        Args:
            prefix: Prefix to filter formats.

        Returns:
            List of format completions.
        """
        items = []
        for fmt in self.FORMATS:
            if fmt.lower().startswith(prefix.lower()):
                items.append(CompletionItem(
                    text=fmt,
                    description=f"{fmt.upper()} format",
                    completion_type=CompletionType.VALUE,
                ))
        return items

    def complete_paths(self, prefix: str = "") -> list[CompletionItem]:
        """Complete file paths.

        Args:
            prefix: Path prefix.

        Returns:
            List of path completions.
        """
        items = []
        prefix_path = Path(prefix) if prefix else Path(".")

        # Get parent directory
        if prefix.endswith("/") or not prefix:
            search_dir = prefix_path
            search_prefix = ""
        else:
            search_dir = prefix_path.parent
            search_prefix = prefix_path.name

        try:
            if search_dir.exists() and search_dir.is_dir():
                for path in search_dir.iterdir():
                    if path.name.startswith(search_prefix) or not search_prefix:
                        display = path.name + ("/" if path.is_dir() else "")
                        items.append(CompletionItem(
                            text=str(path),
                            display=display,
                            description="Directory" if path.is_dir() else "File",
                            completion_type=CompletionType.PATH,
                        ))
        except PermissionError:
            pass

        return items[:20]  # Limit to 20 items

    def _get_area_description(self, area: str) -> str:
        """Get area description.

        Args:
            area: Area code.

        Returns:
            Area description.
        """
        descriptions = {
            "SECO": "Southeast/Central-West region",
            "S": "South region",
            "NE": "Northeast region",
            "N": "North region",
            "SIN": "National interconnected system",
        }
        return descriptions.get(area, "")

    def _get_model_description(self, model: str) -> str:
        """Get model description.

        Args:
            model: Model name.

        Returns:
            Model description.
        """
        descriptions = {
            "lgbm": "LightGBM gradient boosting",
            "random_forest": "Random Forest ensemble",
            "arima": "ARIMA time series model",
            "holt_winters": "Holt-Winters exponential smoothing",
            "regdin_svm": "RegDin SVM hybrid model",
            "blf": "Baseline Load Forecast predictor",
        }
        return descriptions.get(model, "")

    def _get_metric_description(self, metric: str) -> str:
        """Get metric description.

        Args:
            metric: Metric name.

        Returns:
            Metric description.
        """
        descriptions = {
            "mape": "Mean Absolute Percentage Error",
            "smape": "Symmetric MAPE",
            "mae": "Mean Absolute Error",
            "rmse": "Root Mean Square Error",
            "r2": "R-squared (coefficient of determination)",
        }
        return descriptions.get(metric, "")


class CommandCompleter:
    """Main command completion engine.

    Provides auto-completion for commands, subcommands, options,
    and their arguments.

    Attributes:
        commands: Dictionary of available commands.
        argument_completer: Argument type completer.
    """

    def __init__(self) -> None:
        """Initialize command completer."""
        self.argument_completer = ArgumentCompleter()
        self._commands = self._build_command_tree()
        self._option_completers: dict[str, Callable[[str], list[CompletionItem]]] = {
            "--areas": self.argument_completer.complete_areas,
            "-a": self.argument_completer.complete_areas,
            "--models": self.argument_completer.complete_models,
            "-m": self.argument_completer.complete_models,
            "--model": self.argument_completer.complete_models,
            "--model-a": self.argument_completer.complete_models,
            "--model-b": self.argument_completer.complete_models,
            "--horizons": self.argument_completer.complete_horizons,
            "-h": self.argument_completer.complete_horizons,
            "--metrics": self.argument_completer.complete_metrics,
            "--metric": self.argument_completer.complete_metrics,
            "--format": self.argument_completer.complete_formats,
            "-o": self.argument_completer.complete_formats,
            "--output-format": self.argument_completer.complete_formats,
            "--config": self.argument_completer.complete_paths,
            "-c": self.argument_completer.complete_paths,
            "--output": self.argument_completer.complete_paths,
        }

    def _build_command_tree(self) -> dict:
        """Build command tree structure.

        Returns:
            Command tree dictionary.
        """
        return {
            "train": {
                "description": "Train forecasting models",
                "options": {
                    "--areas": "Comma-separated areas",
                    "-a": "Comma-separated areas (short)",
                    "--horizons": "Comma-separated horizons",
                    "--model-types": "Model types to train",
                    "--optimize": "Enable hyperparameter optimization",
                    "--no-optimize": "Disable optimization",
                    "--parallel-workers": "Number of workers",
                    "--dry-run": "Show plan without executing",
                    "--config": "Configuration file",
                    "-c": "Configuration file (short)",
                },
                "subcommands": {
                    "model": "Train single model",
                    "batch": "Train batch from config",
                },
            },
            "predict": {
                "description": "Generate load forecasts",
                "options": {
                    "--areas": "Comma-separated areas",
                    "--horizons": "Comma-separated horizons",
                    "--output-dir": "Output directory",
                    "--format": "Output format",
                    "--reconcile": "Apply reconciliation",
                    "--no-reconcile": "Skip reconciliation",
                },
                "subcommands": {
                    "batch": "Batch prediction",
                    "intraday": "Intraday prediction",
                },
            },
            "backtest": {
                "description": "Run backtesting evaluation",
                "options": {
                    "--start": "Start date (YYYY-MM-DD)",
                    "--end": "End date (YYYY-MM-DD)",
                    "--step-days": "Days between backtests",
                    "--areas": "Comma-separated areas",
                    "--report": "Generate HTML report",
                    "--no-report": "Skip report generation",
                },
            },
            "evaluate": {
                "description": "Model evaluation commands",
                "subcommands": {
                    "metrics": "Calculate metrics",
                    "drift": "Detect drift",
                    "compare": "Compare models",
                    "report": "Generate report",
                    "ranking": "Rank models",
                },
            },
            "features": {
                "description": "Feature engineering commands",
                "subcommands": {
                    "generate": "Generate features",
                    "evaluate": "Evaluate importance",
                    "validate": "Validate features",
                    "list": "List available plugins",
                },
            },
            "config": {
                "description": "Configuration management",
                "subcommands": {
                    "show": "Show configuration",
                    "validate": "Validate configuration",
                },
            },
            "status": {
                "description": "Show system status",
                "options": {},
            },
            "help": {
                "description": "Show help information",
                "options": {},
            },
            "exit": {
                "description": "Exit interactive mode",
                "options": {},
            },
            "quit": {
                "description": "Exit interactive mode",
                "options": {},
            },
        }

    def complete(self, line: str, cursor_position: int | None = None) -> list[CompletionItem]:
        """Get completions for input line.

        Args:
            line: Input line.
            cursor_position: Cursor position.

        Returns:
            List of completion items.
        """
        context = CompletionContext.from_line(line, cursor_position)

        # No words yet - complete commands
        if not context.words:
            return self._complete_commands("")

        # Completing first word - command
        if context.word_index == 0:
            return self._complete_commands(context.word)

        # Completing after command
        if context.command:
            return self._complete_command_args(context)

        return []

    def _complete_commands(self, prefix: str) -> list[CompletionItem]:
        """Complete command names.

        Args:
            prefix: Command prefix.

        Returns:
            List of command completions.
        """
        items = []
        for cmd, info in self._commands.items():
            if cmd.startswith(prefix.lower()):
                items.append(CompletionItem(
                    text=cmd,
                    description=info.get("description", ""),
                    completion_type=CompletionType.COMMAND,
                ))
        return items

    def _complete_command_args(self, context: CompletionContext) -> list[CompletionItem]:
        """Complete command arguments.

        Args:
            context: Completion context.

        Returns:
            List of argument completions.
        """
        cmd_info = self._commands.get(context.command, {})
        items = []

        # Check if we're completing an option value
        if context.current_option and context.current_option in self._option_completers:
            completer = self._option_completers[context.current_option]
            return completer(context.word)

        # Check if completing option
        if context.word.startswith("-"):
            # Complete options
            options = cmd_info.get("options", {})
            for opt, desc in options.items():
                if opt.startswith(context.word):
                    items.append(CompletionItem(
                        text=opt,
                        description=desc,
                        completion_type=CompletionType.OPTION,
                    ))
        else:
            # Complete subcommands
            subcommands = cmd_info.get("subcommands", {})
            for subcmd, desc in subcommands.items():
                if subcmd.startswith(context.word.lower()):
                    items.append(CompletionItem(
                        text=subcmd,
                        description=desc,
                        completion_type=CompletionType.SUBCOMMAND,
                    ))

        return items

    def get_command_help(self, command: str) -> str | None:
        """Get help text for command.

        Args:
            command: Command name.

        Returns:
            Help text or None.
        """
        cmd_info = self._commands.get(command)
        if not cmd_info:
            return None

        lines = [
            f"Command: {command}",
            f"Description: {cmd_info.get('description', '')}",
        ]

        subcommands = cmd_info.get("subcommands", {})
        if subcommands:
            lines.append("")
            lines.append("Subcommands:")
            for subcmd, desc in subcommands.items():
                lines.append(f"  {subcmd}: {desc}")

        options = cmd_info.get("options", {})
        if options:
            lines.append("")
            lines.append("Options:")
            for opt, desc in options.items():
                lines.append(f"  {opt}: {desc}")

        return "\n".join(lines)
