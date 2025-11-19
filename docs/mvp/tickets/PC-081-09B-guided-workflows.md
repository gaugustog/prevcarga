# PC-081-09B: Guided Workflows

**Ticket ID:** PC-081-09B  
**Epic:** Epic-09B - Interactive Mode & Advanced Features  
**Parent Epic:** Epic-09 - CLI & User Interface  
**Story Points:** 7  
**Priority:** High  
**Status:** Not Started  

---

## 📋 Description

Implement guided workflow system with 5 step-by-step workflows for common operations: training, prediction, backtest, features, and model comparison.

---

## 🎯 Acceptance Criteria

- [ ] GuidedWorkflowManager class for workflow registration and execution
- [ ] guided_training_workflow with model/area/date/parallel selection
- [ ] guided_prediction_workflow with batch/intraday selection
- [ ] guided_backtest_workflow with date range and retraining intervals
- [ ] guided_feature_workflow for feature generation
- [ ] guided_comparison_workflow for model comparison
- [ ] Menu-driven workflow selection
- [ ] Input validation at each step
- [ ] Command construction and execution
- [ ] Error handling and user guidance
- [ ] Workflow cancellation support

---

## 🔧 Technical Implementation

```python
# prevcarga/cli/workflows.py
import click
from prompt_toolkit import prompt
from datetime import datetime


class GuidedWorkflowManager:
    """Manage guided workflows for complex operations."""
    
    def __init__(self):
        self.workflows = {
            '1': ('Complete model training', guided_training_workflow),
            '2': ('Generate predictions', guided_prediction_workflow),
            '3': ('Run backtest evaluation', guided_backtest_workflow),
            '4': ('Feature engineering pipeline', guided_feature_workflow),
            '5': ('Model comparison analysis', guided_comparison_workflow)
        }
    
    def list_workflows(self):
        """List available guided workflows."""
        return {k: v[0] for k, v in self.workflows.items()}
    
    def execute_workflow(self, workflow_id: str):
        """Execute specified guided workflow."""
        if workflow_id not in self.workflows:
            raise ValueError(f"Unknown workflow: {workflow_id}")
        
        _, workflow_func = self.workflows[workflow_id]
        workflow_func()


def start_guided_workflow():
    """Start guided workflow selection."""
    manager = GuidedWorkflowManager()
    workflows = manager.list_workflows()
    
    click.echo("\n╔═════════════════════════════════════════════════╗")
    click.echo("║            Guided Workflows                     ║")
    click.echo("╚═════════════════════════════════════════════════╝")
    click.echo()
    
    for key, description in workflows.items():
        click.echo(f"  {key}. {description}")
    
    click.echo()
    choice = prompt("Select workflow (1-5) or 'cancel': ")
    
    if choice.lower() == 'cancel':
        click.echo("Cancelled")
        return
    
    if choice not in workflows:
        click.echo("Invalid choice")
        return
    
    try:
        manager.execute_workflow(choice)
    except KeyboardInterrupt:
        click.echo("\n\nWorkflow cancelled")
    except Exception as e:
        click.secho(f"\nWorkflow error: {e}", fg='red')


def guided_training_workflow():
    """Guide user through training workflow."""
    click.echo("\n╔═════════════════════════════════════════════════╗")
    click.echo("║      Guided Training Workflow                   ║")
    click.echo("╚═════════════════════════════════════════════════╝")
    click.echo()
    
    # Step 1: Select model
    click.echo("Step 1: Select Model")
    click.echo("  Options: lgbm, rf, regdin_svm, holt_winters, all")
    model = prompt("Model: ", default='lgbm')
    
    if model not in ['lgbm', 'rf', 'regdin_svm', 'holt_winters', 'all']:
        click.echo(f"Invalid model: {model}")
        return
    
    # Step 2: Select areas
    click.echo("\nStep 2: Select Areas")
    click.echo("  Enter comma-separated areas or 'all' for all areas")
    click.echo("  Available: SE, S, NE, N, CO, etc.")
    areas_input = prompt("Areas: ", default='all')
    
    # Step 3: Date range
    click.echo("\nStep 3: Training Date Range")
    start_date = prompt("Start date (YYYY-MM-DD): ", default='2023-01-01')
    end_date = prompt("End date (YYYY-MM-DD): ", default='2023-12-31')
    
    # Validate dates
    try:
        datetime.strptime(start_date, '%Y-%m-%d')
        datetime.strptime(end_date, '%Y-%m-%d')
    except ValueError as e:
        click.echo(f"Invalid date format: {e}")
        return
    
    # Step 4: Parallel workers
    click.echo("\nStep 4: Parallel Configuration")
    parallel = prompt("Parallel workers (default: 4): ", default='4')
    
    try:
        parallel_int = int(parallel)
        if parallel_int < 1:
            click.echo("Parallel workers must be positive")
            return
    except ValueError:
        click.echo("Invalid number")
        return
    
    # Step 5: Confirm and execute
    click.echo("\n╔═════════════════════════════════════════════════╗")
    click.echo("║      Training Summary                           ║")
    click.echo("╚═════════════════════════════════════════════════╝")
    click.echo(f"Model:           {model}")
    click.echo(f"Areas:           {areas_input}")
    click.echo(f"Date range:      {start_date} to {end_date}")
    click.echo(f"Parallel:        {parallel}")
    click.echo()
    
    if not click.confirm("Execute training?"):
        click.echo("Training cancelled")
        return
    
    # Construct command
    cmd = f"train model --model {model} --start-date {start_date} --end-date {end_date} --parallel {parallel}"
    
    if areas_input.lower() != 'all':
        for area in areas_input.split(','):
            area = area.strip()
            if area:
                cmd += f" --area {area}"
    
    click.echo(f"\nExecuting: {cmd}")
    click.echo()
    
    # Execute command
    from prevcarga.cli.commands.interactive import execute_interactive_command
    execute_interactive_command(cmd, click.get_current_context())


def guided_prediction_workflow():
    """Guide user through prediction workflow."""
    click.echo("\n╔═════════════════════════════════════════════════╗")
    click.echo("║      Guided Prediction Workflow                 ║")
    click.echo("╚═════════════════════════════════════════════════╝")
    click.echo()
    
    # Step 1: Prediction type
    click.echo("Step 1: Prediction Type")
    click.echo("  1. Batch prediction (D+0 to D+8)")
    click.echo("  2. Intraday prediction (D+0 update)")
    pred_type = prompt("Select type (1-2): ", default='1')
    
    if pred_type == '1':
        # Batch prediction workflow
        click.echo("\nStep 2: Prediction Date")
        date = prompt("Date (YYYY-MM-DD): ", default=datetime.now().strftime('%Y-%m-%d'))
        
        click.echo("\nStep 3: Horizons")
        click.echo("  Enter comma-separated horizons (0-8) or 'all'")
        horizons = prompt("Horizons: ", default='all')
        
        click.echo("\nStep 4: Combination Method")
        click.echo("  Options: simple_avg, weighted_avg, stacking, markov")
        combination = prompt("Combination: ", default='weighted_avg')
        
        click.echo("\nStep 5: Output Format")
        click.echo("  Options: csv, json, parquet")
        format_type = prompt("Format: ", default='csv')
        
        # Construct command
        cmd = f"predict batch --date {date} --combination {combination} --format {format_type}"
        
        if horizons.lower() != 'all':
            for h in horizons.split(','):
                h = h.strip()
                if h:
                    cmd += f" --horizon {h}"
    
    elif pred_type == '2':
        # Intraday prediction workflow
        click.echo("\nStep 2: Current DateTime")
        datetime_str = prompt(
            "DateTime (YYYY-MM-DD HH:MM): ", 
            default=datetime.now().strftime('%Y-%m-%d %H:%M')
        )
        
        cmd = f"predict intraday --datetime \"{datetime_str}\""
    
    else:
        click.echo("Invalid selection")
        return
    
    # Confirm and execute
    click.echo(f"\nExecuting: {cmd}")
    
    if click.confirm("Continue?"):
        from prevcarga.cli.commands.interactive import execute_interactive_command
        execute_interactive_command(cmd, click.get_current_context())
    else:
        click.echo("Cancelled")


def guided_backtest_workflow():
    """Guide user through backtest workflow."""
    click.echo("\n╔═════════════════════════════════════════════════╗")
    click.echo("║      Guided Backtest Workflow                   ║")
    click.echo("╚═════════════════════════════════════════════════╝")
    click.echo()
    
    click.echo("Step 1: Backtest Period")
    start_date = prompt("Start date (YYYY-MM-DD): ", default='2024-01-01')
    end_date = prompt("End date (YYYY-MM-DD): ", default='2024-03-31')
    
    click.echo("\nStep 2: Retraining Intervals")
    click.echo("  Enter comma-separated intervals in days")
    intervals = prompt("Intervals: ", default='7,14,30')
    
    # Construct command
    cmd = f"evaluate backtest --start-date {start_date} --end-date {end_date}"
    
    for interval in intervals.split(','):
        interval = interval.strip()
        if interval:
            cmd += f" --retraining-interval {interval}"
    
    # Confirm and execute
    click.echo(f"\nExecuting: {cmd}")
    
    if click.confirm("Continue? (This may take a while)"):
        from prevcarga.cli.commands.interactive import execute_interactive_command
        execute_interactive_command(cmd, click.get_current_context())
    else:
        click.echo("Cancelled")


def guided_feature_workflow():
    """Guide user through feature generation workflow."""
    click.echo("\n╔═════════════════════════════════════════════════╗")
    click.echo("║   Guided Feature Engineering Workflow           ║")
    click.echo("╚═════════════════════════════════════════════════╝")
    click.echo()
    
    click.echo("Step 1: Feature Plugins")
    click.echo("  Available: temporal, calendar, weather, lag, rolling, fourier")
    click.echo("  Enter comma-separated plugins or 'all'")
    plugins = prompt("Plugins: ", default='all')
    
    click.echo("\nStep 2: Date Range")
    start_date = prompt("Start date (YYYY-MM-DD): ", default='2023-01-01')
    end_date = prompt("End date (YYYY-MM-DD): ", default='2023-12-31')
    
    click.echo("\nStep 3: Output Configuration")
    output_dir = prompt("Output directory: ", default='features/')
    parallel = prompt("Parallel workers: ", default='4')
    
    # Construct command
    cmd = f"features generate --start-date {start_date} --end-date {end_date} --parallel {parallel} --output-dir {output_dir}"
    
    if plugins.lower() != 'all':
        for plugin in plugins.split(','):
            plugin = plugin.strip()
            if plugin:
                cmd += f" --plugin {plugin}"
    
    # Confirm and execute
    click.echo(f"\nExecuting: {cmd}")
    
    if click.confirm("Continue?"):
        from prevcarga.cli.commands.interactive import execute_interactive_command
        execute_interactive_command(cmd, click.get_current_context())
    else:
        click.echo("Cancelled")


def guided_comparison_workflow():
    """Guide user through model comparison workflow."""
    click.echo("\n╔═════════════════════════════════════════════════╗")
    click.echo("║   Guided Model Comparison Workflow              ║")
    click.echo("╚═════════════════════════════════════════════════╝")
    click.echo()
    
    click.echo("Step 1: Evaluation Period")
    start_date = prompt("Start date (YYYY-MM-DD): ", default='2024-01-01')
    end_date = prompt("End date (YYYY-MM-DD): ", default='2024-03-31')
    
    click.echo("\nStep 2: Combination Strategies to Compare")
    click.echo("  Available: simple_avg, weighted_avg, stacking, markov")
    click.echo("  Enter comma-separated strategies or 'all'")
    strategies = prompt("Strategies: ", default='all')
    
    click.echo("\nStep 3: Cross-Validation")
    cv_folds = prompt("CV folds: ", default='5')
    
    # Construct command
    cmd = f"evaluate combination --start-date {start_date} --end-date {end_date} --cross-validation {cv_folds}"
    
    if strategies.lower() != 'all':
        for strategy in strategies.split(','):
            strategy = strategy.strip()
            if strategy:
                cmd += f" --strategy {strategy}"
    
    # Confirm and execute
    click.echo(f"\nExecuting: {cmd}")
    
    if click.confirm("Continue?"):
        from prevcarga.cli.commands.interactive import execute_interactive_command
        execute_interactive_command(cmd, click.get_current_context())
    else:
        click.echo("Cancelled")
```

---

## 🧪 Testing Requirements

```python
# tests/cli/test_workflows.py
def test_guided_workflow_manager():
    """Test workflow manager."""
    manager = GuidedWorkflowManager()
    workflows = manager.list_workflows()
    assert len(workflows) == 5
    assert '1' in workflows


def test_workflow_selection():
    """Test workflow selection in interactive mode."""
    runner = CliRunner()
    result = runner.invoke(cli, ['interactive'], input='workflows\ncancel\nexit\n')
    assert 'Guided Workflows' in result.output
```

---

## ✅ Definition of Done

- [ ] All 5 workflows implemented
- [ ] Menu-driven selection working
- [ ] Input validation at each step
- [ ] Command construction correct
- [ ] Error handling robust
- [ ] Unit tests pass (>80% coverage)
- [ ] Documentation complete

---

**Assignee:** Backend Team  
**Estimated Hours:** 10-14 hours  
**Target Completion:** Week 25, Day 3
