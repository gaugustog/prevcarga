"""Interactive CLI module for PrevCarga.

This package provides interactive shell mode for the PrevCarga CLI,
allowing users to run multiple commands in a session with auto-completion,
history, and guided workflows.

Key Components:
- InteractiveShell: Main interactive shell class
- CommandCompleter: Auto-completion for commands and arguments
- SessionManager: Manages interactive session state
- PromptBuilder: Customizable prompt builder
- GuidedWorkflowManager: Manages step-by-step guided workflows

Example:
    ```python
    from src.cli.interactive import InteractiveShell, GuidedWorkflowManager

    # Start interactive mode
    shell = InteractiveShell()
    shell.run()

    # Or use guided workflows
    manager = GuidedWorkflowManager()
    result = manager.execute_workflow('1')  # Training workflow
    ```
"""

from src.cli.interactive.completer import (
    ArgumentCompleter,
    CommandCompleter,
    CompletionContext,
    CompletionItem,
)
from src.cli.interactive.prompt import (
    PromptBuilder,
    PromptStyle,
    PromptTheme,
)
from src.cli.interactive.session import (
    CommandHistory,
    SessionManager,
    SessionState,
)
from src.cli.interactive.shell import (
    InteractiveShell,
    ShellCommand,
    ShellConfig,
)
from src.cli.interactive.workflows import (
    GuidedWorkflowManager,
    StepOption,
    StepType,
    WorkflowDefinition,
    WorkflowExecutor,
    WorkflowResult,
    WorkflowStatus,
    WorkflowStep,
    guided_backtest_workflow,
    guided_comparison_workflow,
    guided_feature_workflow,
    guided_prediction_workflow,
    guided_training_workflow,
    start_guided_workflow,
)
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

__all__ = [
    # Shell
    "InteractiveShell",
    "ShellCommand",
    "ShellConfig",
    # Session
    "CommandHistory",
    "SessionManager",
    "SessionState",
    # Completion
    "ArgumentCompleter",
    "CommandCompleter",
    "CompletionContext",
    "CompletionItem",
    # Prompt
    "PromptBuilder",
    "PromptStyle",
    "PromptTheme",
    # Workflows
    "GuidedWorkflowManager",
    "WorkflowDefinition",
    "WorkflowExecutor",
    "WorkflowResult",
    "WorkflowStatus",
    "WorkflowStep",
    "StepOption",
    "StepType",
    "start_guided_workflow",
    "guided_training_workflow",
    "guided_prediction_workflow",
    "guided_backtest_workflow",
    "guided_feature_workflow",
    "guided_comparison_workflow",
    # UX
    "CommandAlias",
    "CommandAliases",
    "ErrorHandler",
    "ErrorRecovery",
    "HelpContext",
    "InteractiveSession",
    "PerformanceTip",
    "PerformanceTips",
    "SessionState",
    "clear_screen",
    "display_aliases",
    "display_context_help",
    "display_enhanced_history",
    "display_startup_banner",
    "handle_command_error",
    "show_session_status",
]
