# PC-083-09B: Interactive UX Polish

**Ticket ID:** PC-083-09B  
**Epic:** Epic-09B - Interactive Mode & Advanced Features  
**Parent Epic:** Epic-09 - CLI & User Interface  
**Story Points:** 5  
**Priority:** Medium  
**Status:** Not Started  

---

## 📋 Description

Enhance interactive mode user experience with context-aware help, error recovery, session persistence, and performance optimizations.

---

## 🎯 Acceptance Criteria

- [ ] Context-aware help system showing relevant commands
- [ ] Enhanced command history display with timestamps
- [ ] Screen clear functionality
- [ ] Status display integration in interactive mode
- [ ] execute_interactive_command() with robust error handling
- [ ] User-friendly error messages with recovery suggestions
- [ ] Session state persistence across invocations
- [ ] Command aliases for common operations
- [ ] Command response time <1s for non-computational operations
- [ ] Keyboard interrupt handling without exit
- [ ] Startup banner customization

---

## 🔧 Technical Implementation

```python
# prevcarga/cli/interactive_ux.py
import click
from datetime import datetime
from pathlib import Path
import json


class InteractiveSession:
    """Manage interactive session state."""
    
    def __init__(self):
        self.session_file = Path('.prevcarga_session')
        self.start_time = datetime.now()
        self.command_count = 0
        self.last_command = None
        self.aliases = self._load_aliases()
    
    def load_state(self):
        """Load session state from file."""
        if self.session_file.exists():
            try:
                with open(self.session_file) as f:
                    state = json.load(f)
                return state
            except Exception:
                return {}
        return {}
    
    def save_state(self):
        """Save session state to file."""
        state = {
            'last_used': datetime.now().isoformat(),
            'command_count': self.command_count,
            'last_command': self.last_command
        }
        
        try:
            with open(self.session_file, 'w') as f:
                json.dump(state, f)
        except Exception:
            pass
    
    def _load_aliases(self):
        """Load command aliases."""
        return {
            'ls': 'features generate --help',
            'st': 'status',
            'v': 'version',
            'h': 'help',
            'q': 'exit',
            'train-lgbm': 'train model --model lgbm',
            'predict-today': f'predict batch --date {datetime.now().strftime("%Y-%m-%d")}',
        }
    
    def resolve_alias(self, command: str) -> str:
        """Resolve command alias."""
        parts = command.split(maxsplit=1)
        if parts[0] in self.aliases:
            resolved = self.aliases[parts[0]]
            if len(parts) > 1:
                resolved += ' ' + parts[1]
            return resolved
        return command


def display_context_help(current_context: str = None):
    """Display context-aware help."""
    click.echo("\n╔═══════════════════════════════════════════════════╗")
    click.echo("║         Context-Aware Help                        ║")
    click.echo("╚═══════════════════════════════════════════════════╝")
    
    if current_context == 'training':
        click.echo("\n📚 Training Context:")
        click.echo("  train model --model <type> --start-date <date> --end-date <date>")
        click.echo("  train batch --config-file <file>")
        click.echo("\nQuick examples:")
        click.echo("  train model --model lgbm --start-date 2023-01-01 --end-date 2023-12-31")
        
    elif current_context == 'prediction':
        click.echo("\n🔮 Prediction Context:")
        click.echo("  predict batch --date <date>")
        click.echo("  predict intraday --datetime <datetime>")
        click.echo("\nQuick examples:")
        click.echo("  predict batch --date 2024-01-15")
        
    else:
        click.echo("\n💡 Quick Commands:")
        click.echo("  workflows  - Start guided workflows")
        click.echo("  status     - Check system status")
        click.echo("  help       - Show all commands")
        click.echo("\n🔑 Aliases:")
        click.echo("  st  → status")
        click.echo("  v   → version")
        click.echo("  q   → exit")


def display_enhanced_history(history_file, limit: int = 20):
    """Display command history with timestamps and execution info."""
    click.echo("\n╔═══════════════════════════════════════════════════╗")
    click.echo("║         Command History                           ║")
    click.echo("╚═══════════════════════════════════════════════════╝")
    click.echo()
    
    try:
        history_path = Path(history_file.filename)
        if not history_path.exists():
            click.echo("No command history yet")
            return
        
        with open(history_path, 'r') as f:
            lines = f.readlines()
        
        recent = lines[-limit:] if len(lines) > limit else lines
        
        if recent:
            for i, cmd in enumerate(recent, 1):
                # Format: number, command
                click.echo(f"  {i:2d}. {cmd.strip()}")
            
            click.echo(f"\nShowing last {len(recent)} commands")
            click.echo("Tip: Use UP/DOWN arrows to navigate history")
        else:
            click.echo("No command history yet")
    
    except Exception as e:
        click.echo(f"Error reading history: {e}")
    
    click.echo()


def show_performance_tip():
    """Show performance tip based on usage."""
    tips = [
        "💡 Tip: Use '--parallel 8' for faster training with multi-core systems",
        "💡 Tip: Use 'train batch' with a config file for complex training scenarios",
        "💡 Tip: Try 'workflows' for step-by-step guidance",
        "💡 Tip: Use TAB for auto-completion",
        "💡 Tip: Type 'status' to check system health",
    ]
    
    import random
    click.echo(f"\n{random.choice(tips)}")


def handle_command_error(error: Exception, command: str, verbose: bool = False):
    """Handle command execution errors with helpful messages."""
    click.secho("\n✗ Command failed", fg='red', bold=True)
    click.echo(f"Command: {command}")
    click.echo(f"Error: {str(error)}")
    
    # Provide context-specific help
    if 'date' in command.lower() and 'invalid' in str(error).lower():
        click.echo("\n💡 Tip: Use date format YYYY-MM-DD (e.g., 2024-01-15)")
    
    elif 'model' in command.lower() and 'invalid' in str(error).lower():
        click.echo("\n💡 Tip: Valid models are: lgbm, rf, regdin_svm, holt_winters")
    
    elif 'area' in command.lower() and 'invalid' in str(error).lower():
        click.echo("\n💡 Tip: Use valid area codes: SE, S, NE, N, CO")
    
    elif 'config' in str(error).lower():
        click.echo("\n💡 Tip: Check your configuration with 'config validate'")
    
    elif 'connection' in str(error).lower() or 's3' in str(error).lower():
        click.echo("\n💡 Tip: Check system connectivity with 'status'")
    
    # Show full traceback in verbose mode
    if verbose:
        click.echo("\nFull traceback:")
        import traceback
        traceback.print_exc()
    else:
        click.echo("\nRun with '--verbose' for detailed error information")


def display_startup_banner(session: InteractiveSession):
    """Display enhanced startup banner."""
    state = session.load_state()
    
    click.echo("╔═══════════════════════════════════════════════════════╗")
    click.echo("║   PrevCarga Interactive Mode                          ║")
    click.echo("║   Electric Load Forecasting System                    ║")
    click.echo("╚═══════════════════════════════════════════════════════╝")
    click.echo()
    
    if state.get('last_used'):
        last_used = datetime.fromisoformat(state['last_used'])
        click.echo(f"Last session: {last_used.strftime('%Y-%m-%d %H:%M')}")
    
    if state.get('command_count'):
        click.echo(f"Total commands: {state['command_count']}")
    
    click.echo()
    click.echo("Quick start:")
    click.echo("  • Type 'workflows' for guided operations")
    click.echo("  • Type 'help' for all commands")
    click.echo("  • Type 'exit' or 'quit' to leave")
    click.echo()
    
    # Show a random tip
    show_performance_tip()
    click.echo()


def enhanced_execute_command(command_str: str, ctx, session: InteractiveSession):
    """Execute command with enhanced error handling and tracking."""
    import shlex
    import time
    from prevcarga.cli.main import cli
    
    # Track execution
    session.command_count += 1
    session.last_command = command_str
    start_time = time.time()
    
    # Resolve aliases
    resolved_command = session.resolve_alias(command_str)
    if resolved_command != command_str:
        click.echo(f"→ {resolved_command}")
        command_str = resolved_command
    
    try:
        # Parse command
        args = shlex.split(command_str)
        
        if not args:
            return
        
        # Execute through CLI
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
        
        # Show execution time for long operations
        execution_time = time.time() - start_time
        if execution_time > 2:
            click.echo(f"\n⏱ Completed in {execution_time:.1f}s")
        
        # Save session state
        session.save_state()
        
        if result.exit_code != 0:
            if result.stderr:
                click.secho(result.stderr, fg='red', err=True)
    
    except SystemExit:
        pass
    except KeyboardInterrupt:
        click.echo("\n\n⚠ Operation cancelled")
    except Exception as e:
        handle_command_error(e, command_str, ctx.obj.get('verbose', False))
```

---

## 🧪 Testing Requirements

```python
# tests/cli/test_interactive_ux.py
def test_interactive_session():
    """Test session state management."""
    session = InteractiveSession()
    session.command_count = 5
    session.save_state()
    
    # Load in new session
    new_session = InteractiveSession()
    state = new_session.load_state()
    assert state['command_count'] == 5


def test_alias_resolution():
    """Test command alias resolution."""
    session = InteractiveSession()
    resolved = session.resolve_alias('st')
    assert resolved == 'status'


def test_error_handling():
    """Test error handling with helpful messages."""
    # Test date error
    error = ValueError("Invalid date format")
    # Should provide date format tip
    handle_command_error(error, "train model --start-date invalid", False)
```

---

## ✅ Definition of Done

- [ ] Context-aware help implemented
- [ ] Enhanced history display working
- [ ] Error handling with suggestions
- [ ] Session persistence functional
- [ ] Command aliases working
- [ ] Performance optimized (<1s response)
- [ ] Unit tests pass (>80% coverage)
- [ ] Documentation complete

---

**Assignee:** Backend Team  
**Estimated Hours:** 8-10 hours  
**Target Completion:** Week 25, Day 5
