"""Session management for interactive CLI.

This module provides session state management for the interactive shell,
including command history, context preservation, and session persistence.

Key Components:
- SessionState: Container for session state data
- SessionManager: Manages session lifecycle
- CommandHistory: Command history with search capabilities

Example:
    ```python
    from src.cli.interactive.session import SessionManager

    # Create session manager
    manager = SessionManager()
    manager.start()

    # Get session state
    state = manager.get_state()
    print(f"Session ID: {state.session_id}")

    # Save command to history
    manager.add_to_history("train --areas SECO")
    ```
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import uuid


@dataclass
class CommandHistoryEntry:
    """Single command history entry.

    Attributes:
        command: The command string.
        timestamp: When command was executed.
        success: Whether command succeeded.
        duration_seconds: Execution duration.
        error_message: Error message if failed.
    """

    command: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))
    success: bool = True
    duration_seconds: float = 0.0
    error_message: str | None = None


class CommandHistory:
    """Command history manager with search capabilities.

    Provides command history storage, search, and navigation
    for the interactive shell.

    Attributes:
        max_size: Maximum history entries to keep.
        entries: List of history entries.
    """

    def __init__(self, max_size: int = 1000) -> None:
        """Initialize command history.

        Args:
            max_size: Maximum number of history entries.
        """
        self.max_size = max_size
        self.entries: list[CommandHistoryEntry] = []
        self._position: int = -1

    def add(
        self,
        command: str,
        success: bool = True,
        duration_seconds: float = 0.0,
        error_message: str | None = None,
    ) -> None:
        """Add command to history.

        Args:
            command: Command string.
            success: Whether command succeeded.
            duration_seconds: Execution duration.
            error_message: Error message if failed.
        """
        # Don't add duplicate consecutive commands
        if self.entries and self.entries[-1].command == command:
            return

        # Don't add empty commands
        if not command.strip():
            return

        entry = CommandHistoryEntry(
            command=command,
            success=success,
            duration_seconds=duration_seconds,
            error_message=error_message,
        )
        self.entries.append(entry)

        # Trim history if exceeded max size
        if len(self.entries) > self.max_size:
            self.entries = self.entries[-self.max_size:]

        # Reset position to end
        self._position = len(self.entries)

    def get_previous(self) -> str | None:
        """Get previous command in history.

        Returns:
            Previous command or None if at start.
        """
        if not self.entries:
            return None

        if self._position > 0:
            self._position -= 1

        return self.entries[self._position].command

    def get_next(self) -> str | None:
        """Get next command in history.

        Returns:
            Next command or None if at end.
        """
        if not self.entries:
            return None

        if self._position < len(self.entries) - 1:
            self._position += 1
            return self.entries[self._position].command

        # Return empty at end
        self._position = len(self.entries)
        return ""

    def search(self, prefix: str) -> list[str]:
        """Search history for commands starting with prefix.

        Args:
            prefix: Command prefix to search for.

        Returns:
            List of matching commands (most recent first).
        """
        matches: list[str] = []
        seen: set[str] = set()

        for entry in reversed(self.entries):
            if entry.command.startswith(prefix) and entry.command not in seen:
                matches.append(entry.command)
                seen.add(entry.command)

        return matches

    def search_contains(self, substring: str) -> list[str]:
        """Search history for commands containing substring.

        Args:
            substring: Substring to search for.

        Returns:
            List of matching commands (most recent first).
        """
        matches: list[str] = []
        seen: set[str] = set()

        for entry in reversed(self.entries):
            if substring in entry.command and entry.command not in seen:
                matches.append(entry.command)
                seen.add(entry.command)

        return matches

    def get_all(self) -> list[str]:
        """Get all commands in history.

        Returns:
            List of all commands.
        """
        return [entry.command for entry in self.entries]

    def get_recent(self, count: int = 10) -> list[CommandHistoryEntry]:
        """Get recent history entries.

        Args:
            count: Number of entries to get.

        Returns:
            List of recent history entries.
        """
        return self.entries[-count:]

    def clear(self) -> None:
        """Clear all history."""
        self.entries.clear()
        self._position = -1

    def reset_position(self) -> None:
        """Reset navigation position to end."""
        self._position = len(self.entries)

    def __len__(self) -> int:
        """Return history length."""
        return len(self.entries)


@dataclass
class SessionState:
    """Container for interactive session state.

    Attributes:
        session_id: Unique session identifier.
        started_at: Session start time.
        current_area: Currently selected area.
        current_horizon: Currently selected horizon.
        current_model: Currently selected model.
        config_path: Path to active configuration.
        working_dir: Current working directory.
        variables: User-defined session variables.
        command_count: Number of commands executed.
        last_command: Last executed command.
        last_command_success: Whether last command succeeded.
    """

    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    started_at: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))
    current_area: str | None = None
    current_horizon: int | None = None
    current_model: str | None = None
    config_path: Path | None = None
    working_dir: Path = field(default_factory=Path.cwd)
    variables: dict[str, Any] = field(default_factory=dict)
    command_count: int = 0
    last_command: str | None = None
    last_command_success: bool = True

    def set_context(
        self,
        area: str | None = None,
        horizon: int | None = None,
        model: str | None = None,
    ) -> None:
        """Set current context.

        Args:
            area: Area to set.
            horizon: Horizon to set.
            model: Model to set.
        """
        if area is not None:
            self.current_area = area
        if horizon is not None:
            self.current_horizon = horizon
        if model is not None:
            self.current_model = model

    def clear_context(self) -> None:
        """Clear current context."""
        self.current_area = None
        self.current_horizon = None
        self.current_model = None

    def set_variable(self, name: str, value: Any) -> None:
        """Set session variable.

        Args:
            name: Variable name.
            value: Variable value.
        """
        self.variables[name] = value

    def get_variable(self, name: str, default: Any = None) -> Any:
        """Get session variable.

        Args:
            name: Variable name.
            default: Default value if not found.

        Returns:
            Variable value or default.
        """
        return self.variables.get(name, default)

    def to_dict(self) -> dict[str, Any]:
        """Convert state to dictionary.

        Returns:
            State as dictionary.
        """
        return {
            "session_id": self.session_id,
            "started_at": self.started_at.isoformat(),
            "current_area": self.current_area,
            "current_horizon": self.current_horizon,
            "current_model": self.current_model,
            "config_path": str(self.config_path) if self.config_path else None,
            "working_dir": str(self.working_dir),
            "variables": self.variables,
            "command_count": self.command_count,
            "last_command": self.last_command,
            "last_command_success": self.last_command_success,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SessionState":
        """Create state from dictionary.

        Args:
            data: State dictionary.

        Returns:
            SessionState instance.
        """
        state = cls()
        state.session_id = data.get("session_id", state.session_id)
        if data.get("started_at"):
            state.started_at = datetime.fromisoformat(data["started_at"])
        state.current_area = data.get("current_area")
        state.current_horizon = data.get("current_horizon")
        state.current_model = data.get("current_model")
        if data.get("config_path"):
            state.config_path = Path(data["config_path"])
        if data.get("working_dir"):
            state.working_dir = Path(data["working_dir"])
        state.variables = data.get("variables", {})
        state.command_count = data.get("command_count", 0)
        state.last_command = data.get("last_command")
        state.last_command_success = data.get("last_command_success", True)
        return state


class SessionManager:
    """Manages interactive session lifecycle.

    Handles session start, stop, state persistence, and
    command history management.

    Attributes:
        state: Current session state.
        history: Command history.
        persistence_path: Path for session persistence.
    """

    def __init__(
        self,
        persistence_path: Path | None = None,
        history_size: int = 1000,
    ) -> None:
        """Initialize session manager.

        Args:
            persistence_path: Path to persist session data.
            history_size: Maximum history entries.
        """
        self.persistence_path = persistence_path
        self.state = SessionState()
        self.history = CommandHistory(max_size=history_size)
        self._active = False

    def start(self, restore: bool = True) -> SessionState:
        """Start a new session.

        Args:
            restore: Whether to restore previous session.

        Returns:
            Session state.
        """
        if restore and self.persistence_path and self.persistence_path.exists():
            self._restore_session()
        else:
            self.state = SessionState()
            self.history.clear()

        self._active = True
        return self.state

    def stop(self, save: bool = True) -> None:
        """Stop the current session.

        Args:
            save: Whether to save session for later restoration.
        """
        if save and self.persistence_path:
            self._save_session()

        self._active = False

    def is_active(self) -> bool:
        """Check if session is active.

        Returns:
            True if session is active.
        """
        return self._active

    def get_state(self) -> SessionState:
        """Get current session state.

        Returns:
            Current session state.
        """
        return self.state

    def record_command(
        self,
        command: str,
        success: bool = True,
        duration_seconds: float = 0.0,
        error_message: str | None = None,
    ) -> None:
        """Record command execution.

        Args:
            command: Command that was executed.
            success: Whether command succeeded.
            duration_seconds: Execution duration.
            error_message: Error message if failed.
        """
        self.history.add(
            command=command,
            success=success,
            duration_seconds=duration_seconds,
            error_message=error_message,
        )
        self.state.command_count += 1
        self.state.last_command = command
        self.state.last_command_success = success

    def get_history(self) -> CommandHistory:
        """Get command history.

        Returns:
            Command history instance.
        """
        return self.history

    def get_duration(self) -> float:
        """Get session duration in seconds.

        Returns:
            Session duration.
        """
        now = datetime.now(tz=timezone.utc)
        return (now - self.state.started_at).total_seconds()

    def get_summary(self) -> dict[str, Any]:
        """Get session summary.

        Returns:
            Session summary dictionary.
        """
        return {
            "session_id": self.state.session_id,
            "duration_seconds": self.get_duration(),
            "commands_executed": self.state.command_count,
            "current_context": {
                "area": self.state.current_area,
                "horizon": self.state.current_horizon,
                "model": self.state.current_model,
            },
            "variables_set": len(self.state.variables),
        }

    def _save_session(self) -> None:
        """Save session to persistence path."""
        if not self.persistence_path:
            return

        self.persistence_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "state": self.state.to_dict(),
            "history": [
                {
                    "command": entry.command,
                    "timestamp": entry.timestamp.isoformat(),
                    "success": entry.success,
                    "duration_seconds": entry.duration_seconds,
                    "error_message": entry.error_message,
                }
                for entry in self.history.entries
            ],
        }

        with self.persistence_path.open("w") as f:
            json.dump(data, f, indent=2)

    def _restore_session(self) -> None:
        """Restore session from persistence path."""
        if not self.persistence_path or not self.persistence_path.exists():
            return

        try:
            with self.persistence_path.open() as f:
                data = json.load(f)

            self.state = SessionState.from_dict(data.get("state", {}))

            # Restore history
            self.history.clear()
            for entry_data in data.get("history", []):
                entry = CommandHistoryEntry(
                    command=entry_data["command"],
                    timestamp=datetime.fromisoformat(entry_data["timestamp"]),
                    success=entry_data.get("success", True),
                    duration_seconds=entry_data.get("duration_seconds", 0.0),
                    error_message=entry_data.get("error_message"),
                )
                self.history.entries.append(entry)
            self.history.reset_position()

        except (json.JSONDecodeError, KeyError):
            # Start fresh on error
            self.state = SessionState()
            self.history.clear()
