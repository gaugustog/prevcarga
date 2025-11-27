"""Tests for interactive CLI module.

This module contains comprehensive tests for the interactive shell components
including session management, completion, prompts, and the shell itself.
"""

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.cli.interactive.completer import (
    ArgumentCompleter,
    CommandCompleter,
    CompletionContext,
    CompletionItem,
    CompletionType,
)
from src.cli.interactive.prompt import (
    PromptBuilder,
    PromptColors,
    PromptStyle,
    PromptTheme,
)
from src.cli.interactive.session import (
    CommandHistory,
    CommandHistoryEntry,
    SessionManager,
    SessionState,
)
from src.cli.interactive.shell import (
    InteractiveShell,
    ShellCommand,
    ShellConfig,
)


# =============================================================================
# CommandHistory Tests
# =============================================================================


class TestCommandHistoryEntry:
    """Tests for CommandHistoryEntry dataclass."""

    def test_basic_entry(self) -> None:
        """Test basic entry creation."""
        entry = CommandHistoryEntry(command="train --areas SECO")
        assert entry.command == "train --areas SECO"
        assert entry.success is True
        assert entry.duration_seconds == 0.0
        assert entry.error_message is None
        assert isinstance(entry.timestamp, datetime)

    def test_failed_entry(self) -> None:
        """Test failed entry creation."""
        entry = CommandHistoryEntry(
            command="invalid",
            success=False,
            error_message="Unknown command",
        )
        assert entry.success is False
        assert entry.error_message == "Unknown command"


class TestCommandHistory:
    """Tests for CommandHistory class."""

    @pytest.fixture
    def history(self) -> CommandHistory:
        """Create command history instance."""
        return CommandHistory(max_size=100)

    def test_add_command(self, history: CommandHistory) -> None:
        """Test adding commands to history."""
        history.add("train --areas SECO")
        assert len(history) == 1
        assert history.entries[0].command == "train --areas SECO"

    def test_no_duplicate_consecutive(self, history: CommandHistory) -> None:
        """Test that consecutive duplicates are not added."""
        history.add("train")
        history.add("train")
        history.add("train")
        assert len(history) == 1

    def test_no_empty_commands(self, history: CommandHistory) -> None:
        """Test that empty commands are not added."""
        history.add("")
        history.add("   ")
        assert len(history) == 0

    def test_max_size(self) -> None:
        """Test history max size trimming."""
        history = CommandHistory(max_size=5)
        for i in range(10):
            history.add(f"command_{i}")
        assert len(history) == 5
        assert history.entries[0].command == "command_5"
        assert history.entries[-1].command == "command_9"

    def test_get_previous(self, history: CommandHistory) -> None:
        """Test getting previous commands."""
        history.add("cmd1")
        history.add("cmd2")
        history.add("cmd3")

        assert history.get_previous() == "cmd3"
        assert history.get_previous() == "cmd2"
        assert history.get_previous() == "cmd1"
        assert history.get_previous() == "cmd1"  # At start

    def test_get_next(self, history: CommandHistory) -> None:
        """Test getting next commands."""
        history.add("cmd1")
        history.add("cmd2")

        history.get_previous()  # cmd2
        history.get_previous()  # cmd1

        assert history.get_next() == "cmd2"
        assert history.get_next() == ""  # At end

    def test_search(self, history: CommandHistory) -> None:
        """Test searching history by prefix."""
        history.add("train --areas SECO")
        history.add("predict --areas S")
        history.add("train --areas NE")

        matches = history.search("train")
        assert len(matches) == 2
        assert "train --areas NE" in matches
        assert "train --areas SECO" in matches

    def test_search_contains(self, history: CommandHistory) -> None:
        """Test searching history by substring."""
        history.add("train --areas SECO")
        history.add("predict --areas SECO")
        history.add("train --areas S")

        matches = history.search_contains("SECO")
        assert len(matches) == 2

    def test_get_all(self, history: CommandHistory) -> None:
        """Test getting all commands."""
        history.add("cmd1")
        history.add("cmd2")

        all_cmds = history.get_all()
        assert all_cmds == ["cmd1", "cmd2"]

    def test_get_recent(self, history: CommandHistory) -> None:
        """Test getting recent entries."""
        for i in range(20):
            history.add(f"cmd_{i}")

        recent = history.get_recent(5)
        assert len(recent) == 5
        assert recent[-1].command == "cmd_19"

    def test_clear(self, history: CommandHistory) -> None:
        """Test clearing history."""
        history.add("cmd1")
        history.add("cmd2")
        history.clear()
        assert len(history) == 0

    def test_reset_position(self, history: CommandHistory) -> None:
        """Test resetting navigation position."""
        history.add("cmd1")
        history.add("cmd2")
        history.get_previous()  # At cmd2
        history.get_previous()  # At cmd1
        history.reset_position()
        # After reset, should be at end
        assert history.get_previous() == "cmd2"


# =============================================================================
# SessionState Tests
# =============================================================================


class TestSessionState:
    """Tests for SessionState dataclass."""

    def test_default_state(self) -> None:
        """Test default state creation."""
        state = SessionState()
        assert state.session_id is not None
        assert state.started_at is not None
        assert state.current_area is None
        assert state.current_horizon is None
        assert state.current_model is None
        assert state.command_count == 0

    def test_set_context(self) -> None:
        """Test setting context."""
        state = SessionState()
        state.set_context(area="SECO", horizon=2, model="lgbm")

        assert state.current_area == "SECO"
        assert state.current_horizon == 2
        assert state.current_model == "lgbm"

    def test_partial_context(self) -> None:
        """Test setting partial context."""
        state = SessionState()
        state.set_context(area="SECO")
        state.set_context(horizon=3)

        assert state.current_area == "SECO"
        assert state.current_horizon == 3
        assert state.current_model is None

    def test_clear_context(self) -> None:
        """Test clearing context."""
        state = SessionState()
        state.set_context(area="SECO", horizon=2, model="lgbm")
        state.clear_context()

        assert state.current_area is None
        assert state.current_horizon is None
        assert state.current_model is None

    def test_variables(self) -> None:
        """Test session variables."""
        state = SessionState()
        state.set_variable("output_dir", "/tmp")
        state.set_variable("verbose", True)

        assert state.get_variable("output_dir") == "/tmp"
        assert state.get_variable("verbose") is True
        assert state.get_variable("missing", "default") == "default"

    def test_to_dict(self) -> None:
        """Test converting state to dictionary."""
        state = SessionState()
        state.set_context(area="SECO")
        state.set_variable("test", "value")

        data = state.to_dict()

        assert "session_id" in data
        assert "started_at" in data
        assert data["current_area"] == "SECO"
        assert data["variables"]["test"] == "value"

    def test_from_dict(self) -> None:
        """Test creating state from dictionary."""
        data = {
            "session_id": "test123",
            "current_area": "NE",
            "current_horizon": 5,
            "variables": {"key": "val"},
            "command_count": 10,
        }

        state = SessionState.from_dict(data)

        assert state.session_id == "test123"
        assert state.current_area == "NE"
        assert state.current_horizon == 5
        assert state.variables["key"] == "val"
        assert state.command_count == 10


# =============================================================================
# SessionManager Tests
# =============================================================================


class TestSessionManager:
    """Tests for SessionManager class."""

    @pytest.fixture
    def manager(self) -> SessionManager:
        """Create session manager instance."""
        return SessionManager()

    def test_start_session(self, manager: SessionManager) -> None:
        """Test starting a session."""
        state = manager.start()

        assert manager.is_active()
        assert state.session_id is not None

    def test_stop_session(self, manager: SessionManager) -> None:
        """Test stopping a session."""
        manager.start()
        manager.stop()

        assert not manager.is_active()

    def test_get_state(self, manager: SessionManager) -> None:
        """Test getting session state."""
        manager.start()
        state = manager.get_state()

        assert state is not None
        assert isinstance(state, SessionState)

    def test_record_command(self, manager: SessionManager) -> None:
        """Test recording commands."""
        manager.start()

        manager.record_command("train", success=True, duration_seconds=1.5)
        manager.record_command("predict", success=False, error_message="Error")

        state = manager.get_state()
        assert state.command_count == 2
        assert state.last_command == "predict"
        assert state.last_command_success is False

        history = manager.get_history()
        assert len(history) == 2

    def test_get_duration(self, manager: SessionManager) -> None:
        """Test getting session duration."""
        manager.start()
        duration = manager.get_duration()

        assert duration >= 0

    def test_get_summary(self, manager: SessionManager) -> None:
        """Test getting session summary."""
        manager.start()
        manager.record_command("cmd1")
        manager.record_command("cmd2")

        summary = manager.get_summary()

        assert summary["commands_executed"] == 2
        assert "session_id" in summary
        assert "duration_seconds" in summary

    def test_persistence(self) -> None:
        """Test session persistence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            session_file = Path(tmpdir) / "session.json"

            # Create and save session
            manager1 = SessionManager(persistence_path=session_file)
            state1 = manager1.start()
            state1.set_context(area="SECO")
            manager1.record_command("test_command")
            manager1.stop(save=True)

            assert session_file.exists()

            # Restore session
            manager2 = SessionManager(persistence_path=session_file)
            state2 = manager2.start(restore=True)

            assert state2.current_area == "SECO"
            assert len(manager2.get_history()) == 1


# =============================================================================
# CompletionItem Tests
# =============================================================================


class TestCompletionItem:
    """Tests for CompletionItem dataclass."""

    def test_basic_item(self) -> None:
        """Test basic item creation."""
        item = CompletionItem(
            text="train",
            description="Train models",
        )
        assert item.text == "train"
        assert item.description == "Train models"
        assert item.display_text == "train"

    def test_item_with_display(self) -> None:
        """Test item with custom display."""
        item = CompletionItem(
            text="0",
            display="D+0",
            description="Day 0 horizon",
        )
        assert item.text == "0"
        assert item.display_text == "D+0"


class TestCompletionContext:
    """Tests for CompletionContext class."""

    def test_empty_line(self) -> None:
        """Test context from empty line."""
        ctx = CompletionContext.from_line("")
        assert ctx.words == []
        assert ctx.command is None

    def test_single_word(self) -> None:
        """Test context from single word."""
        ctx = CompletionContext.from_line("train")
        assert ctx.command == "train"
        assert ctx.word == "train"
        assert ctx.word_index == 0

    def test_command_with_args(self) -> None:
        """Test context from command with arguments."""
        ctx = CompletionContext.from_line("train --areas SECO")
        assert ctx.command == "train"
        assert ctx.words == ["train", "--areas", "SECO"]

    def test_incomplete_option(self) -> None:
        """Test context from incomplete option."""
        ctx = CompletionContext.from_line("train --ar")
        assert ctx.command == "train"
        assert ctx.word == "--ar"


# =============================================================================
# ArgumentCompleter Tests
# =============================================================================


class TestArgumentCompleter:
    """Tests for ArgumentCompleter class."""

    @pytest.fixture
    def completer(self) -> ArgumentCompleter:
        """Create argument completer instance."""
        return ArgumentCompleter()

    def test_complete_areas(self, completer: ArgumentCompleter) -> None:
        """Test area completion."""
        items = completer.complete_areas("S")
        texts = [item.text for item in items]
        assert "SECO" in texts
        assert "S" in texts
        assert "SIN" in texts
        assert "N" not in texts

    def test_complete_models(self, completer: ArgumentCompleter) -> None:
        """Test model completion."""
        items = completer.complete_models("l")
        texts = [item.text for item in items]
        assert "lgbm" in texts

    def test_complete_horizons(self, completer: ArgumentCompleter) -> None:
        """Test horizon completion."""
        items = completer.complete_horizons("")
        assert len(items) == 9  # D+0 to D+8

    def test_complete_metrics(self, completer: ArgumentCompleter) -> None:
        """Test metric completion."""
        items = completer.complete_metrics("m")
        texts = [item.text for item in items]
        assert "mape" in texts
        assert "mae" in texts

    def test_complete_formats(self, completer: ArgumentCompleter) -> None:
        """Test format completion."""
        items = completer.complete_formats("j")
        texts = [item.text for item in items]
        assert "json" in texts


# =============================================================================
# CommandCompleter Tests
# =============================================================================


class TestCommandCompleter:
    """Tests for CommandCompleter class."""

    @pytest.fixture
    def completer(self) -> CommandCompleter:
        """Create command completer instance."""
        return CommandCompleter()

    def test_complete_empty(self, completer: CommandCompleter) -> None:
        """Test completion with empty input."""
        items = completer.complete("")
        texts = [item.text for item in items]
        assert "train" in texts
        assert "predict" in texts
        assert "backtest" in texts

    def test_complete_partial_command(self, completer: CommandCompleter) -> None:
        """Test completion with partial command."""
        items = completer.complete("tr")
        texts = [item.text for item in items]
        assert "train" in texts
        assert len(texts) == 1

    def test_complete_subcommands(self, completer: CommandCompleter) -> None:
        """Test completion of subcommands."""
        items = completer.complete("evaluate m")
        texts = [item.text for item in items]
        # When completing after space, gets subcommands starting with prefix
        assert "metrics" in texts

    def test_complete_options(self, completer: CommandCompleter) -> None:
        """Test completion of options."""
        items = completer.complete("train --")
        texts = [item.text for item in items]
        assert "--areas" in texts
        assert "--horizons" in texts

    def test_get_command_help(self, completer: CommandCompleter) -> None:
        """Test getting command help."""
        help_text = completer.get_command_help("train")
        assert help_text is not None
        assert "train" in help_text.lower()

    def test_get_command_help_missing(self, completer: CommandCompleter) -> None:
        """Test getting help for missing command."""
        help_text = completer.get_command_help("nonexistent")
        assert help_text is None


# =============================================================================
# PromptTheme Tests
# =============================================================================


class TestPromptTheme:
    """Tests for PromptTheme enum."""

    def test_theme_values(self) -> None:
        """Test theme enum values."""
        assert PromptTheme.DEFAULT.value == "default"
        assert PromptTheme.DARK.value == "dark"
        assert PromptTheme.LIGHT.value == "light"
        assert PromptTheme.MINIMAL.value == "minimal"


class TestPromptStyle:
    """Tests for PromptStyle enum."""

    def test_style_values(self) -> None:
        """Test style enum values."""
        assert PromptStyle.SIMPLE.value == "simple"
        assert PromptStyle.CONTEXT.value == "context"
        assert PromptStyle.FULL.value == "full"
        assert PromptStyle.MINIMAL.value == "minimal"


# =============================================================================
# PromptBuilder Tests
# =============================================================================


class TestPromptBuilder:
    """Tests for PromptBuilder class."""

    @pytest.fixture
    def builder(self) -> PromptBuilder:
        """Create prompt builder instance."""
        return PromptBuilder()

    def test_minimal_prompt(self) -> None:
        """Test minimal prompt style."""
        builder = PromptBuilder(style=PromptStyle.MINIMAL)
        prompt = builder.build(use_colors=False)
        assert ">" in prompt

    def test_simple_prompt(self) -> None:
        """Test simple prompt style."""
        builder = PromptBuilder(style=PromptStyle.SIMPLE)
        prompt = builder.build(use_colors=False)
        assert "prevcarga" in prompt
        assert ">" in prompt

    def test_context_prompt(self) -> None:
        """Test context prompt style."""
        builder = PromptBuilder(style=PromptStyle.CONTEXT)
        prompt = builder.build(
            context_area="SECO",
            context_horizon=2,
            use_colors=False,
        )
        assert "prevcarga" in prompt
        assert "SECO" in prompt
        assert "D+2" in prompt

    def test_full_prompt(self) -> None:
        """Test full prompt style."""
        builder = PromptBuilder(style=PromptStyle.FULL)
        prompt = builder.build(
            session_id="abc123",
            context_area="SECO",
            command_count=5,
            use_colors=False,
        )
        assert "prevcarga" in prompt
        assert "abc123" in prompt

    def test_prompt_with_error(self, builder: PromptBuilder) -> None:
        """Test prompt indicator on error."""
        prompt_success = builder.build(last_success=True, use_colors=False)
        prompt_error = builder.build(last_success=False, use_colors=False)
        # Both should contain prompt indicator
        assert ">" in prompt_success
        assert ">" in prompt_error

    def test_set_theme(self, builder: PromptBuilder) -> None:
        """Test setting theme."""
        builder.set_theme(PromptTheme.DARK)
        assert builder.theme == PromptTheme.DARK

    def test_set_style(self, builder: PromptBuilder) -> None:
        """Test setting style."""
        builder.set_style(PromptStyle.FULL)
        assert builder.style == PromptStyle.FULL

    def test_welcome_message(self, builder: PromptBuilder) -> None:
        """Test welcome message formatting."""
        msg = builder.format_welcome_message(use_colors=False)
        assert "PrevCarga" in msg
        assert "help" in msg.lower()

    def test_goodbye_message(self, builder: PromptBuilder) -> None:
        """Test goodbye message formatting."""
        msg = builder.format_goodbye_message(
            session_duration=125.0,
            commands_executed=10,
            use_colors=False,
        )
        assert "Goodbye" in msg
        assert "10" in msg

    def test_error_message(self, builder: PromptBuilder) -> None:
        """Test error message formatting."""
        msg = builder.format_error_message("Test error", use_colors=False)
        assert "Error" in msg
        assert "Test error" in msg

    def test_success_message(self, builder: PromptBuilder) -> None:
        """Test success message formatting."""
        msg = builder.format_success_message("Done!", use_colors=False)
        assert "Done!" in msg

    def test_continuation_prompt(self, builder: PromptBuilder) -> None:
        """Test continuation prompt."""
        prompt = builder.get_continuation_prompt(use_colors=False)
        assert "..." in prompt


# =============================================================================
# ShellCommand Tests
# =============================================================================


class TestShellCommand:
    """Tests for ShellCommand class."""

    def test_parse_simple(self) -> None:
        """Test parsing simple command."""
        cmd = ShellCommand.parse("train")
        assert cmd.name == "train"
        assert cmd.args == []
        assert cmd.options == {}
        assert cmd.is_valid

    def test_parse_with_args(self) -> None:
        """Test parsing command with arguments."""
        cmd = ShellCommand.parse("evaluate metrics")
        assert cmd.name == "evaluate"
        assert cmd.args == ["metrics"]

    def test_parse_with_options(self) -> None:
        """Test parsing command with options."""
        cmd = ShellCommand.parse("train --areas SECO --horizons 0,1,2")
        assert cmd.name == "train"
        assert cmd.options["areas"] == "SECO"
        assert cmd.options["horizons"] == "0,1,2"

    def test_parse_flag_option(self) -> None:
        """Test parsing flag options."""
        cmd = ShellCommand.parse("train --dry-run")
        assert cmd.options["dry-run"] is True

    def test_parse_short_option(self) -> None:
        """Test parsing short options."""
        cmd = ShellCommand.parse("train -a SECO")
        assert cmd.options["a"] == "SECO"

    def test_parse_empty(self) -> None:
        """Test parsing empty input."""
        cmd = ShellCommand.parse("")
        assert not cmd.is_valid

    def test_parse_quoted(self) -> None:
        """Test parsing quoted arguments."""
        cmd = ShellCommand.parse('train --config "path with spaces/config.yaml"')
        assert cmd.options["config"] == "path with spaces/config.yaml"


# =============================================================================
# ShellConfig Tests
# =============================================================================


class TestShellConfig:
    """Tests for ShellConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration."""
        config = ShellConfig()
        assert config.history_size == 1000
        assert config.enable_completion is True
        assert config.enable_colors is True
        assert config.prompt_theme == PromptTheme.DEFAULT
        assert config.prompt_style == PromptStyle.CONTEXT

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = ShellConfig(
            history_size=500,
            enable_colors=False,
            prompt_theme=PromptTheme.MINIMAL,
        )
        assert config.history_size == 500
        assert config.enable_colors is False
        assert config.prompt_theme == PromptTheme.MINIMAL


# =============================================================================
# InteractiveShell Tests
# =============================================================================


class TestInteractiveShell:
    """Tests for InteractiveShell class."""

    @pytest.fixture
    def shell(self) -> InteractiveShell:
        """Create interactive shell instance."""
        return InteractiveShell()

    def test_creation(self, shell: InteractiveShell) -> None:
        """Test shell creation."""
        assert shell.config is not None
        assert shell.session is not None
        assert shell.completer is not None
        assert shell.prompt_builder is not None

    def test_builtin_commands(self, shell: InteractiveShell) -> None:
        """Test builtin commands list."""
        builtins = shell.BUILTIN_COMMANDS
        assert "exit" in builtins
        assert "quit" in builtins
        assert "help" in builtins
        assert "history" in builtins
        assert "clear" in builtins

    def test_is_running(self, shell: InteractiveShell) -> None:
        """Test is_running method."""
        assert not shell.is_running()

    def test_stop(self, shell: InteractiveShell) -> None:
        """Test stop method."""
        shell._running = True
        shell.stop()
        assert not shell.is_running()

    def test_get_completions(self, shell: InteractiveShell) -> None:
        """Test getting completions."""
        items = shell.get_completions("train --")
        assert len(items) > 0

    def test_get_completions_disabled(self) -> None:
        """Test completions when disabled."""
        config = ShellConfig(enable_completion=False)
        shell = InteractiveShell(config)
        items = shell.get_completions("train --")
        assert len(items) == 0

    def test_shell_with_handler(self) -> None:
        """Test shell with command handler."""
        handler_called = []

        def handler(cmd: str) -> tuple[bool, str | None]:
            handler_called.append(cmd)
            return True, None

        shell = InteractiveShell(command_handler=handler)
        assert shell._command_handler == handler

    def test_execute_exit_command(self, shell: InteractiveShell) -> None:
        """Test executing exit command."""
        shell._running = True
        state = shell.session.start()
        cmd = ShellCommand.parse("exit")
        success, error = shell._execute_builtin(cmd, state)

        assert success
        assert not shell._running

    def test_execute_quit_command(self, shell: InteractiveShell) -> None:
        """Test executing quit command."""
        shell._running = True
        state = shell.session.start()
        cmd = ShellCommand.parse("quit")
        success, error = shell._execute_builtin(cmd, state)

        assert success
        assert not shell._running

    def test_execute_set_command(self, shell: InteractiveShell) -> None:
        """Test executing set command."""
        state = shell.session.start()
        cmd = ShellCommand.parse("set myvar myvalue")
        success, error = shell._execute_builtin(cmd, state)

        assert success
        assert state.variables["myvar"] == "myvalue"

    def test_execute_unset_command(self, shell: InteractiveShell) -> None:
        """Test executing unset command."""
        state = shell.session.start()
        state.set_variable("test", "value")

        cmd = ShellCommand.parse("unset test")
        success, error = shell._execute_builtin(cmd, state)

        assert success
        assert "test" not in state.variables

    def test_execute_context_command(self, shell: InteractiveShell) -> None:
        """Test executing context command."""
        state = shell.session.start()

        # Set context
        cmd = ShellCommand.parse("context set --area SECO --horizon 2")
        cmd.args = ["set"]
        cmd.options = {"area": "SECO", "horizon": "2"}
        shell._execute_builtin(cmd, state)

        assert state.current_area == "SECO"
        assert state.current_horizon == 2

    def test_execute_context_clear(self, shell: InteractiveShell) -> None:
        """Test executing context clear command."""
        state = shell.session.start()
        state.set_context(area="SECO", horizon=1)

        cmd = ShellCommand.parse("context clear")
        cmd.args = ["clear"]
        shell._execute_builtin(cmd, state)

        assert state.current_area is None
        assert state.current_horizon is None


class TestInteractiveShellIntegration:
    """Integration tests for InteractiveShell."""

    def test_command_history_integration(self) -> None:
        """Test command history integration."""
        shell = InteractiveShell()
        state = shell.session.start()

        # Simulate command execution
        shell.session.record_command("train", success=True)
        shell.session.record_command("predict", success=True)

        history = shell.session.get_history()
        assert len(history) == 2
        assert state.command_count == 2

    def test_session_persistence_integration(self) -> None:
        """Test session persistence integration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            session_file = Path(tmpdir) / "session.json"
            config = ShellConfig(session_file=session_file)

            # Create shell and record commands
            shell1 = InteractiveShell(config)
            state1 = shell1.session.start()
            state1.set_context(area="SECO")
            shell1.session.record_command("test")
            shell1.session.stop(save=True)

            # Create new shell and restore
            shell2 = InteractiveShell(config)
            state2 = shell2.session.start(restore=True)

            assert state2.current_area == "SECO"
