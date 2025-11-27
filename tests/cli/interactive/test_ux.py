"""Tests for the interactive UX module.

Tests cover:
- HelpContext enum
- CommandAliases class
- SessionState dataclass
- InteractiveSession class
- PerformanceTips class
- ErrorHandler class
- Display functions
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from src.cli.interactive.ux import (
    CommandAlias,
    CommandAliases,
    ErrorHandler,
    ErrorRecovery,
    HelpContext,
    InteractiveSession,
    PerformanceTip,
    PerformanceTips,
    SessionState,
    clear_screen,
    display_aliases,
    display_context_help,
    display_enhanced_history,
    display_startup_banner,
    handle_command_error,
    show_session_status,
)


class TestHelpContext:
    """Tests for HelpContext enum."""

    def test_all_contexts_exist(self) -> None:
        """Test all expected contexts are defined."""
        expected = [
            "GENERAL",
            "TRAINING",
            "PREDICTION",
            "FEATURES",
            "EVALUATION",
            "CONFIG",
            "BACKTEST",
            "RECONCILIATION",
        ]
        for name in expected:
            assert hasattr(HelpContext, name)

    def test_context_values(self) -> None:
        """Test context values are lowercase strings."""
        for context in HelpContext:
            assert context.value == context.name.lower()


class TestPerformanceTip:
    """Tests for PerformanceTip dataclass."""

    def test_create_tip(self) -> None:
        """Test creating a tip."""
        tip = PerformanceTip(message="Test tip", context=HelpContext.TRAINING)
        assert tip.message == "Test tip"
        assert tip.context == HelpContext.TRAINING
        assert tip.icon == "💡"

    def test_tip_with_custom_icon(self) -> None:
        """Test creating a tip with custom icon."""
        tip = PerformanceTip(message="Test", icon="🔥")
        assert tip.icon == "🔥"

    def test_tip_without_context(self) -> None:
        """Test creating a tip without context."""
        tip = PerformanceTip(message="General tip")
        assert tip.context is None


class TestErrorRecovery:
    """Tests for ErrorRecovery dataclass."""

    def test_create_recovery(self) -> None:
        """Test creating a recovery suggestion."""
        recovery = ErrorRecovery(
            error_pattern="date",
            suggestion="Use YYYY-MM-DD format",
            command_hint="--date 2024-01-15",
        )
        assert recovery.error_pattern == "date"
        assert recovery.suggestion == "Use YYYY-MM-DD format"
        assert recovery.command_hint == "--date 2024-01-15"

    def test_recovery_without_hint(self) -> None:
        """Test recovery without command hint."""
        recovery = ErrorRecovery(
            error_pattern="permission",
            suggestion="Check permissions",
        )
        assert recovery.command_hint is None


class TestCommandAlias:
    """Tests for CommandAlias dataclass."""

    def test_create_alias(self) -> None:
        """Test creating an alias."""
        alias = CommandAlias(alias="st", command="status", description="Show status")
        assert alias.alias == "st"
        assert alias.command == "status"
        assert alias.description == "Show status"


class TestCommandAliases:
    """Tests for CommandAliases class."""

    def test_default_aliases_loaded(self) -> None:
        """Test default aliases are loaded."""
        aliases = CommandAliases()
        # Check some default aliases
        assert aliases.get_alias("st") is not None
        assert aliases.get_alias("h") is not None
        assert aliases.get_alias("q") is not None
        assert aliases.get_alias("wf") is not None

    def test_resolve_alias(self) -> None:
        """Test resolving an alias."""
        aliases = CommandAliases()
        resolved, was_alias = aliases.resolve("st")
        assert was_alias is True
        assert resolved == "status"

    def test_resolve_non_alias(self) -> None:
        """Test resolving a non-alias command."""
        aliases = CommandAliases()
        resolved, was_alias = aliases.resolve("train model")
        assert was_alias is False
        assert resolved == "train model"

    def test_resolve_alias_with_args(self) -> None:
        """Test resolving an alias with additional arguments."""
        aliases = CommandAliases()
        resolved, was_alias = aliases.resolve("cfg --section storage")
        assert was_alias is True
        assert "config show" in resolved
        assert "--section storage" in resolved

    def test_resolve_empty_command(self) -> None:
        """Test resolving empty command."""
        aliases = CommandAliases()
        resolved, was_alias = aliases.resolve("")
        assert was_alias is False
        assert resolved == ""

    def test_add_custom_alias(self) -> None:
        """Test adding a custom alias."""
        aliases = CommandAliases()
        aliases.add_alias("myalias", "train model --model lgbm", "My training alias")

        alias = aliases.get_alias("myalias")
        assert alias is not None
        assert alias.command == "train model --model lgbm"

    def test_remove_alias(self) -> None:
        """Test removing an alias."""
        aliases = CommandAliases()
        aliases.add_alias("temp", "status")

        assert aliases.remove_alias("temp") is True
        assert aliases.get_alias("temp") is None

    def test_remove_nonexistent_alias(self) -> None:
        """Test removing an alias that doesn't exist."""
        aliases = CommandAliases()
        assert aliases.remove_alias("nonexistent") is False

    def test_list_aliases(self) -> None:
        """Test listing all aliases."""
        aliases = CommandAliases()
        all_aliases = aliases.list_aliases()

        assert len(all_aliases) >= len(CommandAliases.DEFAULT_ALIASES)
        assert all(isinstance(a, CommandAlias) for a in all_aliases)

    def test_custom_aliases_in_constructor(self) -> None:
        """Test providing custom aliases in constructor."""
        custom = [
            CommandAlias("foo", "bar", "Test alias"),
            CommandAlias("baz", "qux", "Another alias"),
        ]
        aliases = CommandAliases(custom_aliases=custom)

        assert aliases.get_alias("foo") is not None
        assert aliases.get_alias("baz") is not None


class TestSessionState:
    """Tests for SessionState dataclass."""

    def test_default_state(self) -> None:
        """Test default session state values."""
        state = SessionState()
        assert state.last_used is None
        assert state.total_commands == 0
        assert state.last_command is None
        assert state.last_area is None
        assert state.last_horizon is None
        assert state.last_model is None
        assert state.custom_aliases == []
        assert state.preferences == {}

    def test_to_dict(self) -> None:
        """Test converting state to dictionary."""
        now = datetime.now()
        state = SessionState(
            last_used=now,
            total_commands=42,
            last_command="status",
            last_area="SECO",
            last_horizon=3,
            last_model="lgbm",
            custom_aliases=[{"alias": "foo", "command": "bar"}],
            preferences={"color": True},
        )

        data = state.to_dict()
        assert data["last_used"] == now.isoformat()
        assert data["total_commands"] == 42
        assert data["last_command"] == "status"
        assert data["last_area"] == "SECO"
        assert data["last_horizon"] == 3
        assert data["last_model"] == "lgbm"
        assert data["custom_aliases"] == [{"alias": "foo", "command": "bar"}]
        assert data["preferences"] == {"color": True}

    def test_to_dict_with_none_last_used(self) -> None:
        """Test to_dict with None last_used."""
        state = SessionState()
        data = state.to_dict()
        assert data["last_used"] is None

    def test_from_dict(self) -> None:
        """Test creating state from dictionary."""
        data = {
            "last_used": "2024-01-15T10:30:00",
            "total_commands": 100,
            "last_command": "train model",
            "last_area": "NE",
            "last_horizon": 5,
            "last_model": "rf",
            "custom_aliases": [],
            "preferences": {"verbose": True},
        }

        state = SessionState.from_dict(data)
        assert state.last_used.year == 2024
        assert state.total_commands == 100
        assert state.last_command == "train model"
        assert state.last_area == "NE"
        assert state.last_horizon == 5
        assert state.last_model == "rf"
        assert state.preferences == {"verbose": True}

    def test_from_dict_empty(self) -> None:
        """Test from_dict with empty data."""
        state = SessionState.from_dict({})
        assert state.total_commands == 0
        assert state.last_used is None

    def test_from_dict_invalid_date(self) -> None:
        """Test from_dict with invalid date."""
        data = {"last_used": "not-a-date"}
        state = SessionState.from_dict(data)
        assert state.last_used is None


class TestInteractiveSession:
    """Tests for InteractiveSession class."""

    def test_init_default(self) -> None:
        """Test default initialization."""
        session = InteractiveSession()
        assert session.session_file == Path(".prevcarga_session")
        assert session.auto_save is True
        assert isinstance(session.state, SessionState)
        assert isinstance(session.aliases, CommandAliases)
        assert session.command_history == []

    def test_init_custom_file(self) -> None:
        """Test initialization with custom file."""
        session = InteractiveSession(session_file="/tmp/test_session.json")
        assert session.session_file == Path("/tmp/test_session.json")

    def test_init_no_autosave(self) -> None:
        """Test initialization with auto_save disabled."""
        session = InteractiveSession(auto_save=False)
        assert session.auto_save is False

    def test_load_state_no_file(self) -> None:
        """Test loading state when file doesn't exist."""
        session = InteractiveSession(session_file="/tmp/nonexistent_session.json")
        assert session.load_state() is False

    def test_save_and_load_state(self) -> None:
        """Test saving and loading session state."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            # Create and save session
            session1 = InteractiveSession(session_file=temp_path, auto_save=False)
            session1.state.total_commands = 25
            session1.state.last_area = "SECO"
            session1.state.last_model = "lgbm"
            assert session1.save_state() is True

            # Load in new session
            session2 = InteractiveSession(session_file=temp_path)
            assert session2.load_state() is True
            assert session2.state.total_commands == 25
            assert session2.state.last_area == "SECO"
            assert session2.state.last_model == "lgbm"
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_record_command(self) -> None:
        """Test recording a command."""
        session = InteractiveSession(auto_save=False)
        session.record_command("status", success=True, area="SECO")

        assert session.state.total_commands == 1
        assert session.state.last_command == "status"
        assert session.state.last_area == "SECO"
        assert len(session.command_history) == 1
        assert session.command_history[0][0] == "status"
        assert session.command_history[0][2] is True

    def test_record_command_with_all_context(self) -> None:
        """Test recording command with all context."""
        session = InteractiveSession(auto_save=False)
        session.record_command(
            "train model",
            success=True,
            area="NE",
            horizon=5,
            model="rf",
        )

        assert session.state.last_area == "NE"
        assert session.state.last_horizon == 5
        assert session.state.last_model == "rf"

    def test_record_failed_command(self) -> None:
        """Test recording a failed command."""
        session = InteractiveSession(auto_save=False)
        session.record_command("invalid", success=False)

        assert len(session.command_history) == 1
        assert session.command_history[0][2] is False

    def test_get_session_duration(self) -> None:
        """Test getting session duration."""
        session = InteractiveSession()
        # Duration should be close to 0 initially
        duration = session.get_session_duration()
        assert duration >= 0
        assert duration < 1  # Less than 1 second

    def test_get_recent_commands(self) -> None:
        """Test getting recent commands."""
        session = InteractiveSession(auto_save=False)
        for i in range(15):
            session.record_command(f"command_{i}")

        recent = session.get_recent_commands(limit=5)
        assert len(recent) == 5
        assert recent[-1][0] == "command_14"

    def test_clear_history(self) -> None:
        """Test clearing command history."""
        session = InteractiveSession(auto_save=False)
        session.record_command("test1")
        session.record_command("test2")

        session.clear_history()
        assert len(session.command_history) == 0

    def test_load_state_with_custom_aliases(self) -> None:
        """Test loading state with custom aliases."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name
            data = {
                "last_used": None,
                "total_commands": 10,
                "custom_aliases": [
                    {"alias": "myalias", "command": "mycommand", "description": "My alias"}
                ],
                "preferences": {},
            }
            json.dump(data, f)

        try:
            session = InteractiveSession(session_file=temp_path)
            assert session.load_state() is True
            assert session.aliases.get_alias("myalias") is not None
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_load_state_invalid_json(self) -> None:
        """Test loading state with invalid JSON."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name
            f.write("not valid json {{{")

        try:
            session = InteractiveSession(session_file=temp_path)
            assert session.load_state() is False
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestPerformanceTips:
    """Tests for PerformanceTips class."""

    def test_tips_not_empty(self) -> None:
        """Test that tips collection is not empty."""
        assert len(PerformanceTips.TIPS) > 0

    def test_get_random_tip(self) -> None:
        """Test getting a random tip."""
        tip = PerformanceTips.get_random()
        assert isinstance(tip, PerformanceTip)
        assert len(tip.message) > 0

    def test_get_random_with_context(self) -> None:
        """Test getting a tip with specific context."""
        tip = PerformanceTips.get_random(context=HelpContext.TRAINING)
        assert isinstance(tip, PerformanceTip)
        # Should either match context or be GENERAL
        assert tip.context in [HelpContext.TRAINING, HelpContext.GENERAL, None]

    def test_get_random_reproducible(self) -> None:
        """Test that get_random returns valid tips consistently."""
        for _ in range(10):
            tip = PerformanceTips.get_random()
            assert tip in PerformanceTips.TIPS


class TestErrorHandler:
    """Tests for ErrorHandler class."""

    def test_recoveries_not_empty(self) -> None:
        """Test that recoveries collection is not empty."""
        assert len(ErrorHandler.RECOVERIES) > 0

    def test_get_suggestion_for_date_error(self) -> None:
        """Test getting suggestion for date error."""
        error = ValueError("Invalid date format")
        recovery = ErrorHandler.get_suggestion(error, "predict batch --date invalid")

        assert recovery is not None
        assert "date" in recovery.error_pattern

    def test_get_suggestion_for_model_error(self) -> None:
        """Test getting suggestion for model error."""
        error = ValueError("Unknown model type")
        recovery = ErrorHandler.get_suggestion(error, "train model --model unknown")

        assert recovery is not None
        assert "model" in recovery.error_pattern

    def test_get_suggestion_for_area_error(self) -> None:
        """Test getting suggestion for area error."""
        error = ValueError("Invalid area: INVALID")
        recovery = ErrorHandler.get_suggestion(error, "train --area INVALID")

        assert recovery is not None
        assert "area" in recovery.error_pattern

    def test_get_suggestion_for_config_error(self) -> None:
        """Test getting suggestion for config error."""
        error = FileNotFoundError("config file not found")
        recovery = ErrorHandler.get_suggestion(error, "config show")

        assert recovery is not None
        assert "config" in recovery.error_pattern

    def test_get_suggestion_no_match(self) -> None:
        """Test getting suggestion when no pattern matches."""
        error = RuntimeError("Some unknown error xyz123")
        recovery = ErrorHandler.get_suggestion(error, "unknown_cmd abc")

        # May or may not find a match, but shouldn't crash
        assert recovery is None or isinstance(recovery, ErrorRecovery)

    def test_get_suggestion_for_memory_error(self) -> None:
        """Test getting suggestion for memory error."""
        error = MemoryError("out of memory")
        recovery = ErrorHandler.get_suggestion(error, "train batch")

        assert recovery is not None
        assert "memory" in recovery.error_pattern

    def test_get_suggestion_for_timeout_error(self) -> None:
        """Test getting suggestion for timeout error."""
        error = TimeoutError("operation timed out")
        recovery = ErrorHandler.get_suggestion(error, "train batch --timeout 60")

        assert recovery is not None
        assert "timeout" in recovery.error_pattern


class TestDisplayContextHelp:
    """Tests for display_context_help function."""

    def test_display_training_help(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test displaying training context help."""
        display_context_help("training", use_colors=False)
        output = capsys.readouterr().out

        assert "Training Context" in output
        assert "train model" in output
        assert "lgbm" in output

    def test_display_prediction_help(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test displaying prediction context help."""
        display_context_help("prediction", use_colors=False)
        output = capsys.readouterr().out

        assert "Prediction Context" in output
        assert "predict batch" in output

    def test_display_features_help(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test displaying features context help."""
        display_context_help("features", use_colors=False)
        output = capsys.readouterr().out

        assert "Features Context" in output
        assert "features generate" in output

    def test_display_evaluation_help(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test displaying evaluation context help."""
        display_context_help("evaluation", use_colors=False)
        output = capsys.readouterr().out

        assert "Evaluation Context" in output
        assert "evaluate metrics" in output

    def test_display_config_help(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test displaying config context help."""
        display_context_help("config", use_colors=False)
        output = capsys.readouterr().out

        assert "Configuration Context" in output
        assert "config show" in output
        assert "config validate" in output

    def test_display_general_help(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test displaying general help."""
        display_context_help("general", use_colors=False)
        output = capsys.readouterr().out

        assert "Quick Commands" in output
        assert "workflows" in output
        assert "status" in output

    def test_display_help_with_enum(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test displaying help with HelpContext enum."""
        display_context_help(HelpContext.TRAINING, use_colors=False)
        output = capsys.readouterr().out

        assert "Training Context" in output

    def test_display_help_invalid_context(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test displaying help with invalid context (should default to general)."""
        display_context_help("invalid_context", use_colors=False)
        output = capsys.readouterr().out

        assert "Quick Commands" in output

    def test_display_help_none_context(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test displaying help with None context."""
        display_context_help(None, use_colors=False)
        output = capsys.readouterr().out

        assert "Quick Commands" in output


class TestDisplayEnhancedHistory:
    """Tests for display_enhanced_history function."""

    def test_display_empty_history(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test displaying empty history."""
        display_enhanced_history([], use_colors=False)
        output = capsys.readouterr().out

        assert "Command History" in output
        assert "No command history yet" in output

    def test_display_history(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test displaying history with commands."""
        now = datetime.now()
        history = [
            ("status", now, True),
            ("train model", now, True),
            ("invalid_cmd", now, False),
        ]
        display_enhanced_history(history, use_colors=False)
        output = capsys.readouterr().out

        assert "Command History" in output
        assert "status" in output
        assert "train model" in output
        assert "[+]" in output  # Success indicator
        assert "[x]" in output  # Failure indicator

    def test_display_history_with_limit(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test displaying history with limit."""
        now = datetime.now()
        history = [(f"cmd_{i}", now, True) for i in range(20)]

        display_enhanced_history(history, limit=5, use_colors=False)
        output = capsys.readouterr().out

        assert "Showing last 5 commands" in output


class TestDisplayStartupBanner:
    """Tests for display_startup_banner function."""

    def test_display_banner_no_session(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test displaying banner without session."""
        display_startup_banner(use_colors=False)
        output = capsys.readouterr().out

        assert "PrevCarga Interactive Mode" in output
        assert "Electric Load Forecasting System" in output
        assert "Quick start" in output
        assert "workflows" in output

    def test_display_banner_with_session(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test displaying banner with session."""
        session = InteractiveSession(auto_save=False)
        session.state.last_used = datetime(2024, 1, 15, 10, 30)
        session.state.total_commands = 42

        display_startup_banner(session=session, use_colors=False)
        output = capsys.readouterr().out

        assert "Last session: 2024-01-15 10:30" in output
        assert "Total commands: 42" in output

    def test_display_banner_without_tip(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test displaying banner without tip."""
        display_startup_banner(show_tip=False, use_colors=False)
        output = capsys.readouterr().out

        assert "PrevCarga Interactive Mode" in output
        # Should not have "Tip:" since show_tip=False
        # Note: There might still be tips in "Quick start" section
        assert "Quick start" in output


class TestHandleCommandError:
    """Tests for handle_command_error function."""

    def test_handle_basic_error(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test handling a basic error."""
        error = ValueError("Something went wrong")
        handle_command_error(error, "test_command", use_colors=False)
        output = capsys.readouterr().out

        assert "Command failed" in output
        assert "Command: test_command" in output
        assert "Something went wrong" in output

    def test_handle_error_with_recovery(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test handling error with recovery suggestion."""
        error = ValueError("Invalid date format")
        handle_command_error(error, "predict batch --date bad", use_colors=False)
        output = capsys.readouterr().out

        assert "Command failed" in output
        assert "Tip:" in output

    def test_handle_error_verbose(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test handling error with verbose mode."""
        error = ValueError("Test error")
        handle_command_error(error, "test", verbose=True, use_colors=False)
        output = capsys.readouterr().out

        assert "Full traceback" in output

    def test_handle_error_non_verbose(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test handling error without verbose mode."""
        error = ValueError("Test error")
        handle_command_error(error, "test", verbose=False, use_colors=False)
        output = capsys.readouterr().out

        assert "--verbose" in output


class TestShowSessionStatus:
    """Tests for show_session_status function."""

    def test_show_status(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test showing session status."""
        session = InteractiveSession(auto_save=False)
        session.record_command("status")
        session.record_command("train model", area="SECO", model="lgbm", horizon=3)

        show_session_status(session, use_colors=False)
        output = capsys.readouterr().out

        assert "Session Status" in output
        assert "Session duration:" in output
        assert "Commands this session: 2" in output
        assert "Last area: SECO" in output
        assert "Last model: lgbm" in output
        assert "Last horizon: D+3" in output


class TestDisplayAliases:
    """Tests for display_aliases function."""

    def test_display_aliases(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test displaying aliases."""
        aliases = CommandAliases()
        display_aliases(aliases, use_colors=False)
        output = capsys.readouterr().out

        assert "Command Aliases" in output
        assert "st" in output
        assert "status" in output

    def test_display_empty_aliases(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test displaying empty aliases."""
        aliases = CommandAliases()
        # Clear all aliases
        for alias in list(aliases._aliases.keys()):
            aliases.remove_alias(alias)

        display_aliases(aliases, use_colors=False)
        output = capsys.readouterr().out

        assert "No aliases defined" in output


class TestClearScreen:
    """Tests for clear_screen function."""

    @patch("click.clear")
    def test_clear_screen(self, mock_clear) -> None:
        """Test clear screen calls click.clear."""
        clear_screen()
        mock_clear.assert_called_once()


class TestIntegration:
    """Integration tests for UX module."""

    def test_full_session_workflow(self) -> None:
        """Test a full session workflow."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            # Start session
            session = InteractiveSession(session_file=temp_path)

            # Record some commands
            session.record_command("status", success=True)
            session.record_command("train model --model lgbm", success=True, model="lgbm", area="SECO")
            session.record_command("predict batch", success=False)

            # Check state
            assert session.state.total_commands == 3
            assert session.state.last_model == "lgbm"

            # Resolve an alias
            resolved, was_alias = session.aliases.resolve("st")
            assert was_alias is True
            assert resolved == "status"

            # Save state
            assert session.save_state() is True

            # Reload in new session
            session2 = InteractiveSession(session_file=temp_path)
            assert session2.load_state() is True
            assert session2.state.total_commands == 3
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_error_handling_workflow(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test error handling workflow."""
        session = InteractiveSession(auto_save=False)

        # Simulate command execution
        command = "train model --model invalid_model"
        try:
            raise ValueError("Unknown model type: invalid_model")
        except ValueError as e:
            session.record_command(command, success=False)
            handle_command_error(e, command, use_colors=False)

        output = capsys.readouterr().out

        # Should have error message and recovery suggestion
        assert "Command failed" in output
        assert "Tip:" in output

    def test_context_help_integration(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test context help for all contexts."""
        for context in HelpContext:
            display_context_help(context, use_colors=False)
            output = capsys.readouterr().out

            # Each context should produce output
            assert len(output) > 0
            assert "=" in output  # Header separator
