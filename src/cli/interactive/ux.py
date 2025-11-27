"""Interactive UX enhancements for PrevCarga CLI.

This module provides user experience enhancements for the interactive shell,
including context-aware help, error recovery, session persistence,
command aliases, and performance optimizations.

Key Components:
- InteractiveSession: Session state management with persistence
- CommandAliases: Command alias system for shortcuts
- ContextHelp: Context-aware help system
- EnhancedHistory: Enhanced command history display
- ErrorHandler: User-friendly error handling with recovery suggestions

Example:
    ```python
    from src.cli.interactive.ux import (
        InteractiveSession,
        display_context_help,
        display_startup_banner,
        handle_command_error,
    )

    # Create session
    session = InteractiveSession()
    session.load_state()

    # Display startup banner
    display_startup_banner(session)

    # Get context-aware help
    display_context_help("training")
    ```
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click

if TYPE_CHECKING:
    from collections.abc import Callable


class HelpContext(Enum):
    """Context for help display."""

    GENERAL = "general"
    TRAINING = "training"
    PREDICTION = "prediction"
    FEATURES = "features"
    EVALUATION = "evaluation"
    CONFIG = "config"
    BACKTEST = "backtest"
    RECONCILIATION = "reconciliation"


@dataclass
class PerformanceTip:
    """A performance tip to show users."""

    message: str
    context: HelpContext | None = None
    icon: str = "💡"


@dataclass
class ErrorRecovery:
    """Error recovery suggestion."""

    error_pattern: str
    suggestion: str
    command_hint: str | None = None


@dataclass
class CommandAlias:
    """Command alias definition."""

    alias: str
    command: str
    description: str


class CommandAliases:
    """Manage command aliases for shortcuts."""

    DEFAULT_ALIASES = [
        CommandAlias("st", "status", "Show system status"),
        CommandAlias("v", "version", "Show version"),
        CommandAlias("h", "help", "Show help"),
        CommandAlias("q", "exit", "Exit interactive mode"),
        CommandAlias("quit", "exit", "Exit interactive mode"),
        CommandAlias("wf", "workflows", "Start guided workflows"),
        CommandAlias("cfg", "config show", "Show configuration"),
        CommandAlias("val", "config validate", "Validate configuration"),
        CommandAlias("lg", "train model --model lgbm", "Train LightGBM model"),
        CommandAlias("rf", "train model --model rf", "Train Random Forest model"),
        CommandAlias("pt", f"predict batch --date {datetime.now().strftime('%Y-%m-%d')}", "Predict today"),
    ]

    def __init__(self, custom_aliases: list[CommandAlias] | None = None) -> None:
        """Initialize aliases.

        Args:
            custom_aliases: Optional custom aliases to add.
        """
        self._aliases: dict[str, CommandAlias] = {}

        # Load default aliases
        for alias in self.DEFAULT_ALIASES:
            self._aliases[alias.alias] = alias

        # Add custom aliases
        if custom_aliases:
            for alias in custom_aliases:
                self._aliases[alias.alias] = alias

    def resolve(self, command: str) -> tuple[str, bool]:
        """Resolve command alias.

        Args:
            command: Command string to resolve.

        Returns:
            Tuple of (resolved_command, was_alias).
        """
        parts = command.strip().split(maxsplit=1)
        if not parts:
            return command, False

        alias_name = parts[0]
        if alias_name in self._aliases:
            resolved = self._aliases[alias_name].command
            if len(parts) > 1:
                resolved += " " + parts[1]
            return resolved, True

        return command, False

    def add_alias(self, alias: str, command: str, description: str = "") -> None:
        """Add a new alias.

        Args:
            alias: Alias name.
            command: Command to expand to.
            description: Description of the alias.
        """
        self._aliases[alias] = CommandAlias(alias, command, description)

    def remove_alias(self, alias: str) -> bool:
        """Remove an alias.

        Args:
            alias: Alias to remove.

        Returns:
            True if removed, False if not found.
        """
        if alias in self._aliases:
            del self._aliases[alias]
            return True
        return False

    def list_aliases(self) -> list[CommandAlias]:
        """Get all aliases.

        Returns:
            List of aliases.
        """
        return list(self._aliases.values())

    def get_alias(self, alias: str) -> CommandAlias | None:
        """Get specific alias.

        Args:
            alias: Alias name.

        Returns:
            CommandAlias or None if not found.
        """
        return self._aliases.get(alias)


@dataclass
class SessionState:
    """Persistent session state."""

    last_used: datetime | None = None
    total_commands: int = 0
    last_command: str | None = None
    last_area: str | None = None
    last_horizon: int | None = None
    last_model: str | None = None
    custom_aliases: list[dict[str, str]] = field(default_factory=list)
    preferences: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "total_commands": self.total_commands,
            "last_command": self.last_command,
            "last_area": self.last_area,
            "last_horizon": self.last_horizon,
            "last_model": self.last_model,
            "custom_aliases": self.custom_aliases,
            "preferences": self.preferences,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SessionState":
        """Create from dictionary."""
        last_used = None
        if data.get("last_used"):
            try:
                last_used = datetime.fromisoformat(data["last_used"])
            except (ValueError, TypeError):
                pass

        return cls(
            last_used=last_used,
            total_commands=data.get("total_commands", 0),
            last_command=data.get("last_command"),
            last_area=data.get("last_area"),
            last_horizon=data.get("last_horizon"),
            last_model=data.get("last_model"),
            custom_aliases=data.get("custom_aliases", []),
            preferences=data.get("preferences", {}),
        )


class InteractiveSession:
    """Manage interactive session state with persistence."""

    DEFAULT_SESSION_FILE = ".prevcarga_session"

    def __init__(
        self,
        session_file: str | Path | None = None,
        auto_save: bool = True,
    ) -> None:
        """Initialize session.

        Args:
            session_file: Path to session file.
            auto_save: Whether to auto-save on command execution.
        """
        self.session_file = Path(session_file or self.DEFAULT_SESSION_FILE)
        self.auto_save = auto_save
        self.start_time = datetime.now()
        self.state = SessionState()
        self.aliases = CommandAliases()
        self.command_history: list[tuple[str, datetime, bool]] = []

    def load_state(self) -> bool:
        """Load session state from file.

        Returns:
            True if state loaded successfully.
        """
        if not self.session_file.exists():
            return False

        try:
            with open(self.session_file, encoding="utf-8") as f:
                data = json.load(f)
            self.state = SessionState.from_dict(data)

            # Load custom aliases
            for alias_data in self.state.custom_aliases:
                self.aliases.add_alias(
                    alias_data.get("alias", ""),
                    alias_data.get("command", ""),
                    alias_data.get("description", ""),
                )

            return True
        except (json.JSONDecodeError, OSError):
            return False

    def save_state(self) -> bool:
        """Save session state to file.

        Returns:
            True if saved successfully.
        """
        self.state.last_used = datetime.now()

        try:
            with open(self.session_file, "w", encoding="utf-8") as f:
                json.dump(self.state.to_dict(), f, indent=2)
            return True
        except OSError:
            return False

    def record_command(
        self,
        command: str,
        success: bool = True,
        area: str | None = None,
        horizon: int | None = None,
        model: str | None = None,
    ) -> None:
        """Record command execution.

        Args:
            command: Command string.
            success: Whether command succeeded.
            area: Area used in command.
            horizon: Horizon used in command.
            model: Model used in command.
        """
        self.state.total_commands += 1
        self.state.last_command = command
        self.command_history.append((command, datetime.now(), success))

        # Update context
        if area:
            self.state.last_area = area
        if horizon is not None:
            self.state.last_horizon = horizon
        if model:
            self.state.last_model = model

        if self.auto_save:
            self.save_state()

    def get_session_duration(self) -> float:
        """Get session duration in seconds."""
        return (datetime.now() - self.start_time).total_seconds()

    def get_recent_commands(self, limit: int = 10) -> list[tuple[str, datetime, bool]]:
        """Get recent commands.

        Args:
            limit: Maximum number of commands.

        Returns:
            List of (command, timestamp, success) tuples.
        """
        return self.command_history[-limit:]

    def clear_history(self) -> None:
        """Clear command history."""
        self.command_history.clear()


class PerformanceTips:
    """Collection of performance tips."""

    TIPS = [
        PerformanceTip(
            "Use '--parallel 8' for faster training with multi-core systems",
            HelpContext.TRAINING,
        ),
        PerformanceTip(
            "Use 'train batch' with a config file for complex training scenarios",
            HelpContext.TRAINING,
        ),
        PerformanceTip(
            "Try 'workflows' for step-by-step guidance",
            HelpContext.GENERAL,
        ),
        PerformanceTip(
            "Use TAB for auto-completion",
            HelpContext.GENERAL,
        ),
        PerformanceTip(
            "Type 'status' to check system health",
            HelpContext.GENERAL,
        ),
        PerformanceTip(
            "Use 'evaluate metrics --output report.html' to save reports",
            HelpContext.EVALUATION,
        ),
        PerformanceTip(
            "Run 'config validate --strict' before production deployments",
            HelpContext.CONFIG,
        ),
        PerformanceTip(
            "Use 'backtest' to evaluate model performance over time",
            HelpContext.BACKTEST,
        ),
        PerformanceTip(
            "UP/DOWN arrows navigate command history",
            HelpContext.GENERAL,
        ),
        PerformanceTip(
            "Type 'help <command>' for detailed command help",
            HelpContext.GENERAL,
        ),
    ]

    @classmethod
    def get_random(cls, context: HelpContext | None = None) -> PerformanceTip:
        """Get a random tip, optionally filtered by context.

        Args:
            context: Optional context to filter tips.

        Returns:
            A random performance tip.
        """
        if context:
            matching = [t for t in cls.TIPS if t.context == context or t.context == HelpContext.GENERAL]
            if matching:
                return random.choice(matching)

        return random.choice(cls.TIPS)


class ErrorHandler:
    """Handle errors with recovery suggestions."""

    RECOVERIES = [
        ErrorRecovery(
            "date",
            "Use date format YYYY-MM-DD (e.g., 2024-01-15)",
            "predict batch --date 2024-01-15",
        ),
        ErrorRecovery(
            "model",
            "Valid models: lgbm, rf, arima, holt_winters, regdin_svm",
            "train model --model lgbm",
        ),
        ErrorRecovery(
            "area",
            "Valid areas: SECO, S, NE, N, SIN",
            "train model --area SECO",
        ),
        ErrorRecovery(
            "config",
            "Check configuration with 'config validate'",
            "config validate --strict",
        ),
        ErrorRecovery(
            "connection",
            "Check network connectivity and credentials",
            "status",
        ),
        ErrorRecovery(
            "s3",
            "Verify S3 bucket access and credentials",
            "status",
        ),
        ErrorRecovery(
            "file",
            "Verify file path exists and is readable",
            None,
        ),
        ErrorRecovery(
            "permission",
            "Check file/directory permissions",
            None,
        ),
        ErrorRecovery(
            "memory",
            "Try reducing parallel workers or data size",
            "--parallel 2",
        ),
        ErrorRecovery(
            "timeout",
            "Increase timeout or reduce operation scope",
            "--timeout 7200",
        ),
    ]

    @classmethod
    def get_suggestion(cls, error: Exception, command: str) -> ErrorRecovery | None:
        """Get recovery suggestion for an error.

        Args:
            error: The exception that occurred.
            command: Command that caused the error.

        Returns:
            ErrorRecovery suggestion or None.
        """
        error_str = str(error).lower()
        command_lower = command.lower()

        for recovery in cls.RECOVERIES:
            if (recovery.error_pattern in error_str or
                recovery.error_pattern in command_lower):
                return recovery

        return None


def display_context_help(context: str | HelpContext | None = None, use_colors: bool = True) -> None:
    """Display context-aware help.

    Args:
        context: Current context for help.
        use_colors: Whether to use colored output.
    """
    if isinstance(context, str):
        try:
            context = HelpContext(context.lower())
        except ValueError:
            context = HelpContext.GENERAL

    # Header
    if use_colors:
        click.secho("=" * 60, fg="cyan")
        click.secho("  Context-Aware Help", fg="cyan", bold=True)
        click.secho("=" * 60, fg="cyan")
    else:
        click.echo("=" * 60)
        click.echo("  Context-Aware Help")
        click.echo("=" * 60)
    click.echo()

    if context == HelpContext.TRAINING:
        click.echo("Training Context:")
        click.echo()
        click.echo("  Commands:")
        click.echo("    train model --model <type> --start-date <date> --end-date <date>")
        click.echo("    train batch --config-file <file>")
        click.echo()
        click.echo("  Models: lgbm, rf, arima, holt_winters, regdin_svm")
        click.echo()
        click.echo("  Examples:")
        click.echo("    train model --model lgbm --start-date 2023-01-01 --end-date 2023-12-31")
        click.echo("    train model --model rf --area SECO --parallel 8")

    elif context == HelpContext.PREDICTION:
        click.echo("Prediction Context:")
        click.echo()
        click.echo("  Commands:")
        click.echo("    predict batch --date <date>")
        click.echo("    predict intraday --datetime <datetime>")
        click.echo()
        click.echo("  Options: --horizon, --combination, --reconcile, --format")
        click.echo()
        click.echo("  Examples:")
        click.echo("    predict batch --date 2024-01-15")
        click.echo("    predict intraday --datetime '2024-01-15 10:00'")

    elif context == HelpContext.FEATURES:
        click.echo("Features Context:")
        click.echo()
        click.echo("  Commands:")
        click.echo("    features generate --plugins <list>")
        click.echo("    features evaluate --importance")
        click.echo("    features validate --check-leakage")
        click.echo("    features list")
        click.echo()
        click.echo("  Plugins: temporal, calendar, lag, rolling, wavelet, blf, seasonality")

    elif context == HelpContext.EVALUATION:
        click.echo("Evaluation Context:")
        click.echo()
        click.echo("  Commands:")
        click.echo("    evaluate metrics --models <list>")
        click.echo("    evaluate drift --method <method>")
        click.echo("    evaluate compare --models <list>")
        click.echo("    evaluate report --output <file>")
        click.echo()
        click.echo("  Metrics: mape, smape, mae, rmse, r2")

    elif context == HelpContext.CONFIG:
        click.echo("Configuration Context:")
        click.echo()
        click.echo("  Commands:")
        click.echo("    config show [--section <name>] [--format yaml|json|table]")
        click.echo("    config validate [--strict]")
        click.echo("    config init --template development|production|testing")
        click.echo("    config diff --config1 <file1> --config2 <file2>")
        click.echo("    config export --format yaml|json|env")

    else:  # General
        click.echo("Quick Commands:")
        click.echo()
        click.echo("  workflows  - Start guided workflows")
        click.echo("  status     - Check system status")
        click.echo("  help       - Show all commands")
        click.echo("  config     - Manage configuration")
        click.echo()
        click.echo("Aliases:")
        click.echo("  st  -> status")
        click.echo("  v   -> version")
        click.echo("  q   -> exit")
        click.echo("  wf  -> workflows")
        click.echo("  cfg -> config show")
        click.echo()
        click.echo("Tips:")
        click.echo("  - Use TAB for auto-completion")
        click.echo("  - UP/DOWN arrows for history")
        click.echo("  - Type 'help <command>' for details")


def display_enhanced_history(
    history: list[tuple[str, datetime, bool]],
    limit: int = 20,
    use_colors: bool = True,
) -> None:
    """Display command history with timestamps and status.

    Args:
        history: List of (command, timestamp, success) tuples.
        limit: Maximum commands to show.
        use_colors: Whether to use colored output.
    """
    # Header
    if use_colors:
        click.secho("=" * 60, fg="cyan")
        click.secho("  Command History", fg="cyan", bold=True)
        click.secho("=" * 60, fg="cyan")
    else:
        click.echo("=" * 60)
        click.echo("  Command History")
        click.echo("=" * 60)
    click.echo()

    if not history:
        click.echo("  No command history yet")
        click.echo()
        return

    recent = history[-limit:] if len(history) > limit else history

    for i, (cmd, timestamp, success) in enumerate(recent, 1):
        status_icon = "+" if success else "x"
        time_str = timestamp.strftime("%H:%M:%S")

        if use_colors:
            status_color = "green" if success else "red"
            click.echo(f"  {i:2d}. ", nl=False)
            click.secho(f"[{status_icon}]", fg=status_color, nl=False)
            click.echo(f" {time_str}  {cmd}")
        else:
            click.echo(f"  {i:2d}. [{status_icon}] {time_str}  {cmd}")

    click.echo()
    click.echo(f"  Showing last {len(recent)} commands")
    click.echo("  Tip: Use UP/DOWN arrows to navigate history")
    click.echo()


def display_startup_banner(
    session: InteractiveSession | None = None,
    show_tip: bool = True,
    use_colors: bool = True,
) -> None:
    """Display enhanced startup banner.

    Args:
        session: Session for showing last used info.
        show_tip: Whether to show a random tip.
        use_colors: Whether to use colored output.
    """
    # Banner
    if use_colors:
        click.secho("=" * 60, fg="cyan", bold=True)
        click.secho("    PrevCarga Interactive Mode", fg="cyan", bold=True)
        click.secho("    Electric Load Forecasting System", fg="cyan")
        click.secho("=" * 60, fg="cyan", bold=True)
    else:
        click.echo("=" * 60)
        click.echo("    PrevCarga Interactive Mode")
        click.echo("    Electric Load Forecasting System")
        click.echo("=" * 60)
    click.echo()

    # Session info
    if session and session.state.last_used:
        last_used = session.state.last_used.strftime("%Y-%m-%d %H:%M")
        click.echo(f"  Last session: {last_used}")

    if session and session.state.total_commands:
        click.echo(f"  Total commands: {session.state.total_commands}")

    click.echo()

    # Quick start
    click.echo("  Quick start:")
    click.echo("    - Type 'workflows' for guided operations")
    click.echo("    - Type 'help' for all commands")
    click.echo("    - Type 'exit' or 'quit' to leave")
    click.echo()

    # Show tip
    if show_tip:
        tip = PerformanceTips.get_random()
        if use_colors:
            click.secho(f"  {tip.icon} Tip: {tip.message}", fg="yellow")
        else:
            click.echo(f"  {tip.icon} Tip: {tip.message}")
        click.echo()


def handle_command_error(
    error: Exception,
    command: str,
    verbose: bool = False,
    use_colors: bool = True,
) -> None:
    """Handle command execution errors with helpful messages.

    Args:
        error: The exception that occurred.
        command: Command that caused the error.
        verbose: Whether to show full traceback.
        use_colors: Whether to use colored output.
    """
    # Error header
    if use_colors:
        click.secho("\nCommand failed", fg="red", bold=True)
    else:
        click.echo("\nCommand failed")

    click.echo(f"  Command: {command}")
    click.echo(f"  Error: {error}")

    # Get suggestion
    recovery = ErrorHandler.get_suggestion(error, command)
    if recovery:
        click.echo()
        if use_colors:
            click.secho(f"  Tip: {recovery.suggestion}", fg="yellow")
        else:
            click.echo(f"  Tip: {recovery.suggestion}")

        if recovery.command_hint:
            click.echo(f"  Example: {recovery.command_hint}")

    # Verbose traceback
    if verbose:
        click.echo("\n  Full traceback:")
        import traceback
        traceback.print_exc()
    else:
        click.echo("\n  Run with '--verbose' for detailed error information")


def clear_screen() -> None:
    """Clear the terminal screen."""
    click.clear()


def show_session_status(session: InteractiveSession, use_colors: bool = True) -> None:
    """Show current session status.

    Args:
        session: Current session.
        use_colors: Whether to use colored output.
    """
    # Header
    if use_colors:
        click.secho("=" * 60, fg="cyan")
        click.secho("  Session Status", fg="cyan", bold=True)
        click.secho("=" * 60, fg="cyan")
    else:
        click.echo("=" * 60)
        click.echo("  Session Status")
        click.echo("=" * 60)
    click.echo()

    # Session info
    duration = session.get_session_duration()
    minutes = int(duration // 60)
    seconds = int(duration % 60)

    click.echo(f"  Session duration: {minutes}m {seconds}s")
    click.echo(f"  Commands this session: {len(session.command_history)}")
    click.echo(f"  Total commands (all sessions): {session.state.total_commands}")
    click.echo()

    # Last context
    if session.state.last_area:
        click.echo(f"  Last area: {session.state.last_area}")
    if session.state.last_model:
        click.echo(f"  Last model: {session.state.last_model}")
    if session.state.last_horizon is not None:
        click.echo(f"  Last horizon: D+{session.state.last_horizon}")

    click.echo()


def display_aliases(aliases: CommandAliases, use_colors: bool = True) -> None:
    """Display available aliases.

    Args:
        aliases: CommandAliases instance.
        use_colors: Whether to use colored output.
    """
    # Header
    if use_colors:
        click.secho("=" * 60, fg="cyan")
        click.secho("  Command Aliases", fg="cyan", bold=True)
        click.secho("=" * 60, fg="cyan")
    else:
        click.echo("=" * 60)
        click.echo("  Command Aliases")
        click.echo("=" * 60)
    click.echo()

    all_aliases = aliases.list_aliases()
    if not all_aliases:
        click.echo("  No aliases defined")
        return

    for alias in sorted(all_aliases, key=lambda a: a.alias):
        click.echo(f"  {alias.alias:8s} -> {alias.command}")
        if alias.description:
            click.echo(f"           {alias.description}")

    click.echo()
    click.echo("  To add an alias: alias <name> <command>")
    click.echo("  To remove: unalias <name>")
    click.echo()
