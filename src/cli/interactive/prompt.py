"""Prompt customization for interactive CLI.

This module provides prompt building and theming capabilities
for the interactive shell.

Key Components:
- PromptTheme: Color and style theme
- PromptStyle: Prompt format style
- PromptBuilder: Builds customized prompts

Example:
    ```python
    from src.cli.interactive.prompt import PromptBuilder, PromptTheme

    # Create builder with custom theme
    builder = PromptBuilder(theme=PromptTheme.DARK)

    # Get prompt for current state
    prompt = builder.build(
        session_id="abc123",
        context_area="SECO",
        context_horizon=1,
    )
    print(prompt)  # "prevcarga [SECO:D+1] > "
    ```
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class PromptTheme(Enum):
    """Prompt color themes.

    Attributes:
        DEFAULT: Default blue theme.
        DARK: Dark theme with muted colors.
        LIGHT: Light theme with bright colors.
        MINIMAL: No colors, minimal styling.
    """

    DEFAULT = "default"
    DARK = "dark"
    LIGHT = "light"
    MINIMAL = "minimal"


class PromptStyle(Enum):
    """Prompt format styles.

    Attributes:
        SIMPLE: Simple prompt with name only.
        CONTEXT: Shows current context (area, horizon).
        FULL: Full prompt with all information.
        MINIMAL: Just a simple indicator.
    """

    SIMPLE = "simple"
    CONTEXT = "context"
    FULL = "full"
    MINIMAL = "minimal"


@dataclass
class PromptColors:
    """Color definitions for prompt.

    Attributes:
        primary: Primary color for name.
        secondary: Secondary color for context.
        success: Color for success indicators.
        error: Color for error indicators.
        muted: Color for less important text.
        reset: Reset code.
    """

    primary: str = ""
    secondary: str = ""
    success: str = ""
    error: str = ""
    muted: str = ""
    reset: str = ""


class PromptBuilder:
    """Builds customized prompts for the interactive shell.

    Supports theming, context display, and various format styles.

    Attributes:
        theme: Current prompt theme.
        style: Current prompt style.
        name: Application name.
    """

    # ANSI color codes
    COLORS = {
        "blue": "\033[34m",
        "cyan": "\033[36m",
        "green": "\033[32m",
        "red": "\033[31m",
        "yellow": "\033[33m",
        "magenta": "\033[35m",
        "white": "\033[37m",
        "gray": "\033[90m",
        "bold": "\033[1m",
        "reset": "\033[0m",
    }

    # Theme definitions
    THEMES = {
        PromptTheme.DEFAULT: PromptColors(
            primary=COLORS["blue"] + COLORS["bold"],
            secondary=COLORS["cyan"],
            success=COLORS["green"],
            error=COLORS["red"],
            muted=COLORS["gray"],
            reset=COLORS["reset"],
        ),
        PromptTheme.DARK: PromptColors(
            primary=COLORS["cyan"],
            secondary=COLORS["gray"],
            success=COLORS["green"],
            error=COLORS["red"],
            muted=COLORS["gray"],
            reset=COLORS["reset"],
        ),
        PromptTheme.LIGHT: PromptColors(
            primary=COLORS["magenta"] + COLORS["bold"],
            secondary=COLORS["blue"],
            success=COLORS["green"],
            error=COLORS["red"],
            muted=COLORS["white"],
            reset=COLORS["reset"],
        ),
        PromptTheme.MINIMAL: PromptColors(),
    }

    def __init__(
        self,
        theme: PromptTheme = PromptTheme.DEFAULT,
        style: PromptStyle = PromptStyle.CONTEXT,
        name: str = "prevcarga",
    ) -> None:
        """Initialize prompt builder.

        Args:
            theme: Prompt color theme.
            style: Prompt format style.
            name: Application name.
        """
        self.theme = theme
        self.style = style
        self.name = name
        self._colors = self.THEMES[theme]

    def set_theme(self, theme: PromptTheme) -> None:
        """Set prompt theme.

        Args:
            theme: New theme.
        """
        self.theme = theme
        self._colors = self.THEMES[theme]

    def set_style(self, style: PromptStyle) -> None:
        """Set prompt style.

        Args:
            style: New style.
        """
        self.style = style

    def build(
        self,
        session_id: str | None = None,
        context_area: str | None = None,
        context_horizon: int | None = None,
        context_model: str | None = None,
        last_success: bool = True,
        command_count: int = 0,
        use_colors: bool = True,
    ) -> str:
        """Build prompt string.

        Args:
            session_id: Session identifier.
            context_area: Current area context.
            context_horizon: Current horizon context.
            context_model: Current model context.
            last_success: Whether last command succeeded.
            command_count: Number of commands executed.
            use_colors: Whether to use ANSI colors.

        Returns:
            Prompt string.
        """
        colors = self._colors if use_colors else PromptColors()

        if self.style == PromptStyle.MINIMAL:
            return self._build_minimal(colors, last_success)
        elif self.style == PromptStyle.SIMPLE:
            return self._build_simple(colors, last_success)
        elif self.style == PromptStyle.CONTEXT:
            return self._build_context(
                colors, context_area, context_horizon, last_success
            )
        else:  # FULL
            return self._build_full(
                colors,
                session_id,
                context_area,
                context_horizon,
                context_model,
                last_success,
                command_count,
            )

    def _build_minimal(self, colors: PromptColors, last_success: bool) -> str:
        """Build minimal prompt.

        Args:
            colors: Color definitions.
            last_success: Whether last command succeeded.

        Returns:
            Minimal prompt string.
        """
        indicator_color = colors.success if last_success else colors.error
        return f"{indicator_color}>{colors.reset} "

    def _build_simple(self, colors: PromptColors, last_success: bool) -> str:
        """Build simple prompt.

        Args:
            colors: Color definitions.
            last_success: Whether last command succeeded.

        Returns:
            Simple prompt string.
        """
        indicator_color = colors.success if last_success else colors.error
        return f"{colors.primary}{self.name}{colors.reset} {indicator_color}>{colors.reset} "

    def _build_context(
        self,
        colors: PromptColors,
        area: str | None,
        horizon: int | None,
        last_success: bool,
    ) -> str:
        """Build context-aware prompt.

        Args:
            colors: Color definitions.
            area: Current area.
            horizon: Current horizon.
            last_success: Whether last command succeeded.

        Returns:
            Context prompt string.
        """
        parts = [f"{colors.primary}{self.name}{colors.reset}"]

        # Add context if available
        context_parts = []
        if area:
            context_parts.append(area)
        if horizon is not None:
            context_parts.append(f"D+{horizon}")

        if context_parts:
            context_str = ":".join(context_parts)
            parts.append(f" {colors.secondary}[{context_str}]{colors.reset}")

        indicator_color = colors.success if last_success else colors.error
        parts.append(f" {indicator_color}>{colors.reset} ")

        return "".join(parts)

    def _build_full(
        self,
        colors: PromptColors,
        session_id: str | None,
        area: str | None,
        horizon: int | None,
        model: str | None,
        last_success: bool,
        command_count: int,
    ) -> str:
        """Build full prompt with all information.

        Args:
            colors: Color definitions.
            session_id: Session identifier.
            area: Current area.
            horizon: Current horizon.
            model: Current model.
            last_success: Whether last command succeeded.
            command_count: Number of commands executed.

        Returns:
            Full prompt string.
        """
        lines = []

        # Status line
        status_parts = []
        if session_id:
            status_parts.append(f"session:{session_id}")
        status_parts.append(f"cmds:{command_count}")

        context_parts = []
        if area:
            context_parts.append(f"area:{area}")
        if horizon is not None:
            context_parts.append(f"H:D+{horizon}")
        if model:
            context_parts.append(f"model:{model}")

        if context_parts:
            status_parts.extend(context_parts)

        status_line = f"{colors.muted}[{' | '.join(status_parts)}]{colors.reset}"
        lines.append(status_line)

        # Prompt line
        indicator_color = colors.success if last_success else colors.error
        prompt_line = f"{colors.primary}{self.name}{colors.reset} {indicator_color}>{colors.reset} "
        lines.append(prompt_line)

        return "\n".join(lines)

    def get_continuation_prompt(self, use_colors: bool = True) -> str:
        """Get continuation prompt for multi-line input.

        Args:
            use_colors: Whether to use ANSI colors.

        Returns:
            Continuation prompt string.
        """
        colors = self._colors if use_colors else PromptColors()
        return f"{colors.muted}...{colors.reset} "

    def format_welcome_message(self, use_colors: bool = True) -> str:
        """Format welcome message for shell start.

        Args:
            use_colors: Whether to use ANSI colors.

        Returns:
            Welcome message string.
        """
        colors = self._colors if use_colors else PromptColors()

        lines = [
            "",
            f"{colors.primary}╔══════════════════════════════════════════╗{colors.reset}",
            f"{colors.primary}║{colors.reset}  {colors.secondary}PrevCarga Interactive Shell{colors.reset}            {colors.primary}║{colors.reset}",
            f"{colors.primary}║{colors.reset}  Brazilian Electric Load Forecasting     {colors.primary}║{colors.reset}",
            f"{colors.primary}╚══════════════════════════════════════════╝{colors.reset}",
            "",
            f"{colors.muted}Type 'help' for available commands{colors.reset}",
            f"{colors.muted}Type 'exit' or 'quit' to leave{colors.reset}",
            "",
        ]
        return "\n".join(lines)

    def format_goodbye_message(
        self,
        session_duration: float,
        commands_executed: int,
        use_colors: bool = True,
    ) -> str:
        """Format goodbye message for shell exit.

        Args:
            session_duration: Session duration in seconds.
            commands_executed: Number of commands executed.
            use_colors: Whether to use ANSI colors.

        Returns:
            Goodbye message string.
        """
        colors = self._colors if use_colors else PromptColors()

        # Format duration
        minutes = int(session_duration // 60)
        seconds = int(session_duration % 60)
        duration_str = f"{minutes}m {seconds}s" if minutes else f"{seconds}s"

        lines = [
            "",
            f"{colors.muted}────────────────────────────────────────────{colors.reset}",
            f"{colors.secondary}Session Summary:{colors.reset}",
            f"  Duration: {duration_str}",
            f"  Commands: {commands_executed}",
            f"{colors.muted}────────────────────────────────────────────{colors.reset}",
            f"{colors.success}Goodbye!{colors.reset}",
            "",
        ]
        return "\n".join(lines)

    def format_error_message(
        self,
        error: str,
        use_colors: bool = True,
    ) -> str:
        """Format error message.

        Args:
            error: Error message.
            use_colors: Whether to use ANSI colors.

        Returns:
            Formatted error message.
        """
        colors = self._colors if use_colors else PromptColors()
        return f"{colors.error}Error:{colors.reset} {error}"

    def format_success_message(
        self,
        message: str,
        use_colors: bool = True,
    ) -> str:
        """Format success message.

        Args:
            message: Success message.
            use_colors: Whether to use ANSI colors.

        Returns:
            Formatted success message.
        """
        colors = self._colors if use_colors else PromptColors()
        return f"{colors.success}✓{colors.reset} {message}"

    def format_info_message(
        self,
        message: str,
        use_colors: bool = True,
    ) -> str:
        """Format info message.

        Args:
            message: Info message.
            use_colors: Whether to use ANSI colors.

        Returns:
            Formatted info message.
        """
        colors = self._colors if use_colors else PromptColors()
        return f"{colors.secondary}ℹ{colors.reset} {message}"
