# PC-080-09B: Interactive Mode Core

**Ticket ID:** PC-080-09B  
**Epic:** Epic-09B - Interactive Mode & Advanced Features  
**Parent Epic:** Epic-09 - CLI & User Interface  
**Story Points:** 8  
**Priority:** High  
**Status:** Not Started  

---

## 📋 Description

Implement core interactive CLI mode with prompt_toolkit, providing command auto-completion, persistent history, and interactive command execution.

---

## 🎯 Acceptance Criteria

- [ ] Interactive command with prompt_toolkit integration
- [ ] WordCompleter for command auto-completion
- [ ] FileHistory for persistent command history (.prevcarga_history)
- [ ] Interactive loop with command execution
- [ ] Error handling (KeyboardInterrupt, EOFError)
- [ ] display_interactive_help() function
- [ ] Clean exit handling (exit/quit commands)
- [ ] Command parsing with shlex
- [ ] Click context invocation for commands
- [ ] Startup time <2 seconds
- [ ] Command response time <1 second (non-computational)

---

## 🔧 Technical Implementation

```python
# prevcarga/cli/commands/interactive.py
import click
from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style
import shlex
import sys


@click.command()
@click.pass_context
def interactive(ctx):
    """Start interactive CLI mode with guided workflows.
    
    Provides an interactive shell with command completion,
    history, and guided workflows for common operations.
    """
    click.echo("╔═══════════════════════════════════════════════════════╗")
    click.echo("║   PrevCarga Interactive Mode                          ║")
    click.echo("║   Type 'help' for commands or 'exit' to quit          ║")
    click.echo("╚═══════════════════════════════════════════════════════╝")
    click.echo()
    
    # Setup auto-completion
    completer = WordCompleter([
        'train', 'predict', 'evaluate', 'features', 'status',
        'config', 'help', 'exit', 'quit', 'workflows', 'history', 
        'clear', 'version'
    ], ignore_case=True)
    
    # Setup command history
    history_file = FileHistory('.prevcarga_history')
    
    # Custom style
    style = Style.from_dict({
        'prompt': '#00aa00 bold',
    })
    
    # Interactive loop
    while True:
        try:
            user_input = prompt(
                [('class:prompt', 'prevcarga> ')],
                completer=completer,
                history=history_file,
                complete_while_typing=True,
                style=style
            )
            
            # Skip empty input
            if not user_input.strip():
                continue
            
            # Handle built-in commands
            if user_input.lower() in ['exit', 'quit']:
                click.echo("Goodbye!")
                break
            elif user_input.lower() == 'help':
                display_interactive_help()
            elif user_input.lower() == 'workflows':
                from prevcarga.cli.workflows import start_guided_workflow
                start_guided_workflow()
            elif user_input.lower() == 'status':
                from prevcarga.cli.main import status
                ctx.invoke(status)
            elif user_input.lower() == 'version':
                from prevcarga.cli.main import version
                ctx.invoke(version)
            elif user_input.lower() == 'history':
                display_command_history(history_file)
            elif user_input.lower() == 'clear':
                click.clear()
            else:
                # Execute command through CLI parser
                execute_interactive_command(user_input, ctx)
        
        except KeyboardInterrupt:
            click.echo("\n\nUse 'exit' or 'quit' to exit interactive mode")
            continue
        except EOFError:
            click.echo("\nGoodbye!")
            break
        except Exception as e:
            click.secho(f"Error: {e}", fg='red', err=True)
            if ctx.obj.get('verbose'):
                import traceback
                traceback.print_exc()


def display_interactive_help():
    """Display interactive mode help."""
    click.echo("\n╔═══════════════════════════════════════════════════╗")
    click.echo("║         Interactive Mode Commands                 ║")
    click.echo("╚═══════════════════════════════════════════════════╝")
    click.echo()
    click.echo("Core Commands:")
    click.echo("  train       - Model training commands")
    click.echo("  predict     - Prediction generation")
    click.echo("  evaluate    - Model evaluation")
    click.echo("  features    - Feature engineering")
    click.echo("  config      - Configuration management")
    click.echo()
    click.echo("System Commands:")
    click.echo("  status      - System status check")
    click.echo("  version     - Show version information")
    click.echo()
    click.echo("Interactive Commands:")
    click.echo("  workflows   - Start guided workflows")
    click.echo("  history     - Show command history")
    click.echo("  clear       - Clear screen")
    click.echo("  help        - Show this help")
    click.echo("  exit/quit   - Exit interactive mode")
    click.echo()
    click.echo("Tips:")
    click.echo("  - Use TAB for auto-completion")
    click.echo("  - Use UP/DOWN arrows for command history")
    click.echo("  - Type any command followed by --help for detailed usage")
    click.echo()


def display_command_history(history_file):
    """Display command history."""
    click.echo("\n╔═══════════════════════════════════════════════════╗")
    click.echo("║         Command History                           ║")
    click.echo("╚═══════════════════════════════════════════════════╝")
    click.echo()
    
    try:
        # Read history from file
        history_path = history_file.filename
        if not history_path.exists():
            click.echo("No command history yet")
            return
        
        with open(history_path, 'r') as f:
            lines = f.readlines()
        
        # Display last 20 commands
        recent = lines[-20:] if len(lines) > 20 else lines
        
        if recent:
            for i, cmd in enumerate(recent, 1):
                click.echo(f"  {i:2d}. {cmd.strip()}")
        else:
            click.echo("No command history yet")
    
    except Exception as e:
        click.echo(f"Error reading history: {e}")
    
    click.echo()


def execute_interactive_command(command_str: str, ctx):
    """Execute a command string in interactive mode.
    
    Args:
        command_str: Command string to execute
        ctx: Click context
    """
    import shlex
    from prevcarga.cli.main import cli
    
    try:
        # Parse command string
        args = shlex.split(command_str)
        
        if not args:
            return
        
        # Execute through CLI
        # We need to invoke the CLI with the parsed arguments
        from click.testing import CliRunner
        runner = CliRunner(mix_stderr=False)
        
        # Build full command with global options
        full_args = []
        if ctx.obj.get('config'):
            full_args.extend(['--config', str(ctx.obj['config'])])
        if ctx.obj.get('verbose'):
            full_args.append('--verbose')
        if ctx.obj.get('log_level'):
            full_args.extend(['--log-level', ctx.obj['log_level']])
        
        full_args.extend(args)
        
        result = runner.invoke(cli, full_args, catch_exceptions=False)
        
        if result.output:
            click.echo(result.output, nl=False)
        
        if result.exit_code != 0:
            if result.stderr:
                click.secho(result.stderr, fg='red', err=True)
    
    except SystemExit:
        # Catch SystemExit to prevent exiting interactive mode
        pass
    except Exception as e:
        click.secho(f"Error executing command: {e}", fg='red', err=True)
        if ctx.obj.get('verbose'):
            import traceback
            traceback.print_exc()
```

---

## 🧪 Testing Requirements

```python
# tests/cli/test_interactive.py
import pytest
from click.testing import CliRunner
from prevcarga.cli.main import cli


def test_interactive_mode_help():
    """Test interactive mode help display."""
    runner = CliRunner()
    result = runner.invoke(cli, ['interactive'], input='help\nexit\n')
    assert 'Interactive Mode Commands' in result.output


def test_interactive_mode_exit():
    """Test interactive mode exit."""
    runner = CliRunner()
    result = runner.invoke(cli, ['interactive'], input='exit\n')
    assert result.exit_code == 0
    assert 'Goodbye' in result.output


def test_interactive_mode_command_execution():
    """Test command execution in interactive mode."""
    runner = CliRunner()
    result = runner.invoke(cli, ['interactive'], input='version\nexit\n')
    assert 'PrevCarga CLI' in result.output


def test_interactive_mode_clear():
    """Test clear command."""
    runner = CliRunner()
    result = runner.invoke(cli, ['interactive'], input='clear\nexit\n')
    assert result.exit_code == 0


def test_interactive_mode_history():
    """Test history command."""
    runner = CliRunner()
    result = runner.invoke(cli, ['interactive'], input='history\nexit\n')
    assert 'Command History' in result.output
```

---

## 📦 Dependencies

- `prompt_toolkit>=3.0` - Interactive prompts and completion

---

## ✅ Definition of Done

- [ ] Interactive mode functional
- [ ] Auto-completion working
- [ ] Command history persistent
- [ ] Error handling robust
- [ ] Startup time <2s
- [ ] Unit tests pass (>80% coverage)
- [ ] Integration tests pass
- [ ] Documentation complete

---

**Assignee:** Backend Team  
**Estimated Hours:** 12-16 hours  
**Target Completion:** Week 25, Day 2
