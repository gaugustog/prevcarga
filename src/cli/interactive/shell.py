"""Interactive shell for PrevCarga CLI.

This module provides the main interactive shell implementation,
allowing users to execute commands in a REPL environment.

Key Components:
- ShellConfig: Configuration for the shell
- ShellCommand: Parsed command representation
- InteractiveShell: Main shell class

Example:
    ```python
    from src.cli.interactive.shell import InteractiveShell, ShellConfig

    # Create shell with custom config
    config = ShellConfig(
        history_size=500,
        enable_completion=True,
    )
    shell = InteractiveShell(config)

    # Run the shell
    shell.run()
    ```
"""

import shlex
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import click

from src.cli.interactive.completer import CommandCompleter, CompletionItem
from src.cli.interactive.prompt import PromptBuilder, PromptStyle, PromptTheme
from src.cli.interactive.session import SessionManager, SessionState


@dataclass
class ShellConfig:
    """Configuration for the interactive shell.

    Attributes:
        history_size: Maximum history entries.
        history_file: Path to persist history.
        enable_completion: Enable auto-completion.
        enable_colors: Enable ANSI colors.
        prompt_theme: Prompt color theme.
        prompt_style: Prompt format style.
        session_file: Path to persist session.
    """

    history_size: int = 1000
    history_file: Path | None = None
    enable_completion: bool = True
    enable_colors: bool = True
    prompt_theme: PromptTheme = PromptTheme.DEFAULT
    prompt_style: PromptStyle = PromptStyle.CONTEXT
    session_file: Path | None = None


@dataclass
class ShellCommand:
    """Parsed shell command.

    Attributes:
        raw: Raw input string.
        name: Command name.
        args: Command arguments.
        options: Command options.
        is_valid: Whether command is valid.
        error: Error message if invalid.
    """

    raw: str
    name: str = ""
    args: list[str] = field(default_factory=list)
    options: dict[str, Any] = field(default_factory=dict)
    is_valid: bool = True
    error: str = ""

    @classmethod
    def parse(cls, line: str) -> "ShellCommand":
        """Parse command line into ShellCommand.

        Args:
            line: Input line.

        Returns:
            Parsed ShellCommand.
        """
        line = line.strip()
        if not line:
            return cls(raw="", is_valid=False, error="Empty command")

        try:
            parts = shlex.split(line)
        except ValueError as e:
            return cls(raw=line, is_valid=False, error=str(e))

        if not parts:
            return cls(raw=line, is_valid=False, error="Empty command")

        name = parts[0].lower()
        args = []
        options: dict[str, Any] = {}

        i = 1
        while i < len(parts):
            part = parts[i]
            if part.startswith("--"):
                # Long option
                if "=" in part:
                    key, value = part[2:].split("=", 1)
                    options[key] = value
                elif i + 1 < len(parts) and not parts[i + 1].startswith("-"):
                    options[part[2:]] = parts[i + 1]
                    i += 1
                else:
                    options[part[2:]] = True
            elif part.startswith("-") and len(part) > 1:
                # Short option
                if i + 1 < len(parts) and not parts[i + 1].startswith("-"):
                    options[part[1:]] = parts[i + 1]
                    i += 1
                else:
                    options[part[1:]] = True
            else:
                args.append(part)
            i += 1

        return cls(raw=line, name=name, args=args, options=options)


class InteractiveShell:
    """Interactive shell for PrevCarga CLI.

    Provides a REPL environment for executing CLI commands with
    auto-completion, history, and session management.

    Attributes:
        config: Shell configuration.
        session: Session manager.
        completer: Command completer.
        prompt_builder: Prompt builder.
    """

    # Built-in shell commands
    BUILTIN_COMMANDS = {
        "exit",
        "quit",
        "help",
        "history",
        "clear",
        "set",
        "unset",
        "context",
        "status",
    }

    def __init__(
        self,
        config: ShellConfig | None = None,
        command_handler: Callable[[str], tuple[bool, str | None]] | None = None,
    ) -> None:
        """Initialize interactive shell.

        Args:
            config: Shell configuration.
            command_handler: Function to handle CLI commands.
        """
        self.config = config or ShellConfig()
        self.session = SessionManager(
            persistence_path=self.config.session_file,
            history_size=self.config.history_size,
        )
        self.completer = CommandCompleter()
        self.prompt_builder = PromptBuilder(
            theme=self.config.prompt_theme,
            style=self.config.prompt_style,
        )
        self._command_handler = command_handler
        self._running = False
        self._last_success = True

    def run(self) -> None:
        """Run the interactive shell REPL."""
        self._running = True

        # Start session
        state = self.session.start()

        # Show welcome message
        welcome = self.prompt_builder.format_welcome_message(
            use_colors=self.config.enable_colors
        )
        click.echo(welcome)

        try:
            while self._running:
                try:
                    # Build prompt
                    prompt = self.prompt_builder.build(
                        session_id=state.session_id,
                        context_area=state.current_area,
                        context_horizon=state.current_horizon,
                        context_model=state.current_model,
                        last_success=self._last_success,
                        command_count=state.command_count,
                        use_colors=self.config.enable_colors,
                    )

                    # Get input
                    line = self._read_input(prompt)

                    # Handle empty input
                    if not line or not line.strip():
                        continue

                    # Execute command
                    start_time = time.time()
                    success, error = self._execute_command(line, state)
                    duration = time.time() - start_time

                    # Record in history
                    self.session.record_command(
                        command=line,
                        success=success,
                        duration_seconds=duration,
                        error_message=error,
                    )

                    self._last_success = success

                except KeyboardInterrupt:
                    click.echo("\nUse 'exit' or 'quit' to leave the shell")
                    continue
                except EOFError:
                    break

        finally:
            # Show goodbye message
            goodbye = self.prompt_builder.format_goodbye_message(
                session_duration=self.session.get_duration(),
                commands_executed=state.command_count,
                use_colors=self.config.enable_colors,
            )
            click.echo(goodbye)

            # Stop session
            self.session.stop()
            self._running = False

    def _read_input(self, prompt: str) -> str:
        """Read input from user.

        Args:
            prompt: Prompt to display.

        Returns:
            User input.
        """
        try:
            # Try using readline for better experience
            return input(prompt)
        except Exception:
            # Fallback to click
            return click.prompt(prompt, default="", show_default=False, prompt_suffix="")

    def _execute_command(
        self,
        line: str,
        state: SessionState,
    ) -> tuple[bool, str | None]:
        """Execute a command.

        Args:
            line: Command line.
            state: Session state.

        Returns:
            Tuple of (success, error_message).
        """
        cmd = ShellCommand.parse(line)

        if not cmd.is_valid:
            self._show_error(cmd.error)
            return False, cmd.error

        # Check for builtin commands
        if cmd.name in self.BUILTIN_COMMANDS:
            return self._execute_builtin(cmd, state)

        # Execute via command handler
        if self._command_handler:
            try:
                return self._command_handler(line)
            except Exception as e:
                error = str(e)
                self._show_error(error)
                return False, error

        # No handler - show error
        error = f"Unknown command: {cmd.name}"
        self._show_error(error)
        return False, error

    def _execute_builtin(
        self,
        cmd: ShellCommand,
        state: SessionState,
    ) -> tuple[bool, str | None]:
        """Execute a builtin command.

        Args:
            cmd: Parsed command.
            state: Session state.

        Returns:
            Tuple of (success, error_message).
        """
        if cmd.name in ("exit", "quit"):
            self._running = False
            return True, None

        elif cmd.name == "help":
            self._show_help(cmd.args[0] if cmd.args else None)
            return True, None

        elif cmd.name == "history":
            self._show_history(int(cmd.args[0]) if cmd.args else 10)
            return True, None

        elif cmd.name == "clear":
            click.clear()
            return True, None

        elif cmd.name == "set":
            if len(cmd.args) >= 2:
                state.set_variable(cmd.args[0], cmd.args[1])
                self._show_success(f"Set {cmd.args[0]} = {cmd.args[1]}")
            else:
                self._show_error("Usage: set <name> <value>")
                return False, "Invalid syntax"
            return True, None

        elif cmd.name == "unset":
            if cmd.args:
                if cmd.args[0] in state.variables:
                    del state.variables[cmd.args[0]]
                    self._show_success(f"Unset {cmd.args[0]}")
                else:
                    self._show_error(f"Variable not found: {cmd.args[0]}")
                    return False, "Variable not found"
            else:
                self._show_error("Usage: unset <name>")
                return False, "Invalid syntax"
            return True, None

        elif cmd.name == "context":
            self._handle_context_command(cmd, state)
            return True, None

        elif cmd.name == "status":
            self._show_session_status(state)
            return True, None

        return False, f"Unknown builtin: {cmd.name}"

    def _handle_context_command(
        self,
        cmd: ShellCommand,
        state: SessionState,
    ) -> None:
        """Handle context command.

        Args:
            cmd: Parsed command.
            state: Session state.
        """
        if not cmd.args:
            # Show current context
            click.echo("Current context:")
            click.echo(f"  Area:    {state.current_area or '(not set)'}")
            click.echo(f"  Horizon: {f'D+{state.current_horizon}' if state.current_horizon is not None else '(not set)'}")
            click.echo(f"  Model:   {state.current_model or '(not set)'}")
            return

        action = cmd.args[0]

        if action == "clear":
            state.clear_context()
            self._show_success("Context cleared")
        elif action == "set":
            area = cmd.options.get("area") or cmd.options.get("a")
            horizon = cmd.options.get("horizon") or cmd.options.get("h")
            model = cmd.options.get("model") or cmd.options.get("m")

            if horizon and isinstance(horizon, str):
                horizon = int(horizon)

            state.set_context(area=area, horizon=horizon, model=model)
            self._show_success("Context updated")
        else:
            self._show_error(f"Unknown context action: {action}")

    def _show_help(self, command: str | None = None) -> None:
        """Show help information.

        Args:
            command: Specific command to show help for.
        """
        if command:
            help_text = self.completer.get_command_help(command)
            if help_text:
                click.echo(help_text)
            else:
                click.echo(f"No help available for: {command}")
        else:
            click.echo("Available commands:")
            click.echo("")
            click.echo("  CLI Commands:")
            click.echo("    train      - Train forecasting models")
            click.echo("    predict    - Generate load forecasts")
            click.echo("    backtest   - Run backtesting evaluation")
            click.echo("    evaluate   - Model evaluation commands")
            click.echo("    features   - Feature engineering commands")
            click.echo("    config     - Configuration management")
            click.echo("")
            click.echo("  Shell Commands:")
            click.echo("    help       - Show this help")
            click.echo("    history    - Show command history")
            click.echo("    clear      - Clear screen")
            click.echo("    context    - Manage context (area, horizon, model)")
            click.echo("    set        - Set session variable")
            click.echo("    unset      - Remove session variable")
            click.echo("    status     - Show session status")
            click.echo("    exit/quit  - Exit the shell")
            click.echo("")
            click.echo("Type 'help <command>' for detailed help")

    def _show_history(self, count: int = 10) -> None:
        """Show command history.

        Args:
            count: Number of entries to show.
        """
        entries = self.session.get_history().get_recent(count)
        if not entries:
            click.echo("No history")
            return

        click.echo("Recent commands:")
        for i, entry in enumerate(entries, 1):
            status = "✓" if entry.success else "✗"
            timestamp = entry.timestamp.strftime("%H:%M:%S")
            click.echo(f"  {i:3}. [{timestamp}] {status} {entry.command}")

    def _show_session_status(self, state: SessionState) -> None:
        """Show session status.

        Args:
            state: Session state.
        """
        summary = self.session.get_summary()
        duration = summary["duration_seconds"]
        minutes = int(duration // 60)
        seconds = int(duration % 60)

        click.echo("Session Status:")
        click.echo(f"  ID:        {state.session_id}")
        click.echo(f"  Duration:  {minutes}m {seconds}s")
        click.echo(f"  Commands:  {state.command_count}")
        click.echo("")
        click.echo("Context:")
        click.echo(f"  Area:      {state.current_area or '(not set)'}")
        click.echo(f"  Horizon:   {f'D+{state.current_horizon}' if state.current_horizon is not None else '(not set)'}")
        click.echo(f"  Model:     {state.current_model or '(not set)'}")

        if state.variables:
            click.echo("")
            click.echo("Variables:")
            for name, value in state.variables.items():
                click.echo(f"  {name} = {value}")

    def _show_error(self, message: str) -> None:
        """Show error message.

        Args:
            message: Error message.
        """
        formatted = self.prompt_builder.format_error_message(
            message, use_colors=self.config.enable_colors
        )
        click.echo(formatted, err=True)

    def _show_success(self, message: str) -> None:
        """Show success message.

        Args:
            message: Success message.
        """
        formatted = self.prompt_builder.format_success_message(
            message, use_colors=self.config.enable_colors
        )
        click.echo(formatted)

    def _show_info(self, message: str) -> None:
        """Show info message.

        Args:
            message: Info message.
        """
        formatted = self.prompt_builder.format_info_message(
            message, use_colors=self.config.enable_colors
        )
        click.echo(formatted)

    def get_completions(self, line: str) -> list[CompletionItem]:
        """Get completions for input line.

        Args:
            line: Input line.

        Returns:
            List of completion items.
        """
        if not self.config.enable_completion:
            return []

        return self.completer.complete(line)

    def is_running(self) -> bool:
        """Check if shell is running.

        Returns:
            True if running.
        """
        return self._running

    def stop(self) -> None:
        """Stop the shell."""
        self._running = False
