# Epic-09B: Interactive Mode & Advanced Features

**Epic ID:** Epic-09B  
**Epic Name:** Interactive Mode & Advanced Features  
**Phase:** Phase 9B  
**Duration:** 1 week (Week 25)  
**Dependencies:** Epic-09A (Core CLI Commands)  
**Priority:** High  
**Status:** Not Started

---

## 🎯 Epic Overview

**Goal:** Provide interactive CLI mode with guided workflows, comprehensive evaluation commands, and advanced configuration management for the unified electric load forecasting system.

**Problem Statement:**
Beyond basic CLI commands, users need:
- Comprehensive evaluation and backtesting capabilities
- Interactive mode with command completion and history
- Guided workflows for complex multi-step operations
- Advanced configuration management
- Model comparison and optimization tools
- Enhanced user experience for exploratory tasks

**Success Criteria:**
✅ Evaluation commands (model, combination, backtest) operational  
✅ Interactive mode with guided workflows functional  
✅ Configuration management commands complete  
✅ Auto-completion for commands and parameters  
✅ Command history persistent across sessions  
✅ E2E tests cover main user workflows  
✅ User experience optimized (<2s startup, <1s response)

---

## 📋 User Stories

### **User Story 1: Evaluation and Backtesting Commands**
**As a** data scientist  
**I want** comprehensive evaluation commands  
**So that** I can assess model performance and optimize system configurations

**Acceptance Criteria:**
- [x] Model evaluation with multiple metrics (MAPE, MAE, RMSE)
- [x] Model comparison across different configurations
- [x] Combination strategy evaluation and comparison
- [x] Backtesting with walk-forward validation
- [x] Drift detection and performance monitoring
- [x] Report generation in multiple formats

**Technical Implementation:**
```python
from datetime import datetime
from pathlib import Path
import click

@cli.group()
def evaluate():
    """Model and system evaluation commands."""
    pass

@evaluate.command('model')
@click.option('--model', type=click.Choice(['lgbm', 'rf', 'regdin_svm', 'holt_winters']),
              required=True, help='Model to evaluate')
@click.option('--area', multiple=True, help='Areas to evaluate')
@click.option('--start-date', type=click.DateTime(['%Y-%m-%d']), required=True)
@click.option('--end-date', type=click.DateTime(['%Y-%m-%d']), required=True)
@click.option('--metrics', multiple=True, 
              type=click.Choice(['mape', 'mae', 'rmse', 'percentiles']),
              default=['mape', 'mae'], help='Evaluation metrics')
@click.option('--output', type=Path, help='Evaluation report output')
def model(model: str, area: tuple, start_date: datetime, end_date: datetime,
          metrics: tuple, output: Path):
    """Evaluate individual model performance across specified period.
    
    Examples:
        prevcarga evaluate model --model lgbm --area SE --start-date 2024-01-01 --end-date 2024-03-31
        prevcarga evaluate model --model rf --metrics mape --metrics percentiles
    """
    config = load_config()
    areas = list(area) if area else config.data.areas
    metric_list = list(metrics)
    
    click.echo(f"Evaluating {model} model for {len(areas)} areas")
    click.echo(f"  Metrics: {', '.join(metric_list)}")
    click.echo(f"  Period: {start_date.date()} to {end_date.date()}")
    
    with EvaluationProgressBar(areas=areas, metrics=metric_list) as progress:
        evaluator = ModelEvaluator(config)
        evaluation_result = evaluator.evaluate_model(
            model_type=model,
            areas=areas,
            start_date=start_date,
            end_date=end_date,
            metrics=metric_list,
            progress_callback=progress.update
        )
    
    display_model_evaluation(evaluation_result)
    
    if output:
        save_evaluation_report(evaluation_result, output)
        click.echo(f"Report saved to {output}")

@evaluate.command('combination')
@click.option('--strategy', multiple=True, 
              type=click.Choice(['simple_avg', 'weighted_avg', 'stacking', 'markov']),
              help='Combination strategies to evaluate')
@click.option('--models', multiple=True, help='Models to include in combination')
@click.option('--start-date', type=click.DateTime(['%Y-%m-%d']), required=True)
@click.option('--end-date', type=click.DateTime(['%Y-%m-%d']), required=True)
@click.option('--cross-validation', type=int, default=5, help='Cross-validation folds')
@click.option('--output', type=Path, help='Comparison report output')
def combination(strategy: tuple, models: tuple, start_date: datetime,
               end_date: datetime, cross_validation: int, output: Path):
    """Evaluate and compare model combination strategies.
    
    Performs cross-validation comparison of different combination
    methods and provides recommendations for optimal strategy.
    
    Examples:
        prevcarga evaluate combination --start-date 2024-01-01 --end-date 2024-03-31
        prevcarga evaluate combination --strategy stacking --strategy markov --cross-validation 10
    """
    config = load_config()
    strategies = list(strategy) if strategy else config.combination.available_strategies
    model_list = list(models) if models else config.models.default_models
    
    click.echo(f"Evaluating {len(strategies)} combination strategies")
    click.echo(f"  Models: {', '.join(model_list)}")
    click.echo(f"  CV folds: {cross_validation}")
    
    evaluator = CombinationEvaluator(config)
    comparison_result = evaluator.compare_strategies(
        strategies=strategies,
        models=model_list,
        start_date=start_date,
        end_date=end_date,
        cv_folds=cross_validation
    )
    
    display_combination_comparison(comparison_result)
    
    if output:
        save_comparison_report(comparison_result, output)

@evaluate.command('backtest')
@click.option('--start-date', type=click.DateTime(['%Y-%m-%d']), required=True)
@click.option('--end-date', type=click.DateTime(['%Y-%m-%d']), required=True)
@click.option('--retraining-interval', multiple=True, type=int, default=[7, 14, 30],
              help='Retraining intervals to test (days)')
@click.option('--models', multiple=True, help='Models to include in backtest')
@click.option('--areas', multiple=True, help='Areas to backtest')
@click.option('--output', type=Path, help='Backtest results output directory')
def backtest(start_date: datetime, end_date: datetime, retraining_interval: tuple,
            models: tuple, areas: tuple, output: Path):
    """Execute comprehensive backtesting with retraining evaluation.
    
    Performs walk-forward validation across specified period,
    evaluating different retraining intervals and model configurations.
    
    Examples:
        prevcarga evaluate backtest --start-date 2024-01-01 --end-date 2024-12-31
        prevcarga evaluate backtest --start-date 2024-01-01 --end-date 2024-03-31 --retraining-interval 7
    """
    config = load_config()
    intervals = list(retraining_interval)
    model_list = list(models) if models else config.models.default_models
    area_list = list(areas) if areas else config.data.areas
    
    total_days = (end_date - start_date).days
    click.echo(f"Starting backtest: {total_days} days with {len(intervals)} intervals")
    click.echo(f"  Models: {', '.join(model_list)}")
    click.echo(f"  Areas: {len(area_list)}")
    click.echo(f"  Retraining intervals: {intervals}")
    
    with BacktestProgressBar(days=total_days, intervals=intervals) as progress:
        backtester = BacktestingWorkflow(config)
        backtest_result = backtester.execute_backtest(
            start_date=start_date,
            end_date=end_date,
            retraining_intervals=intervals,
            models=model_list,
            areas=area_list,
            progress_callback=progress.update
        )
    
    display_backtest_results(backtest_result)
    
    if output:
        save_backtest_results(backtest_result, output)
        click.echo(f"Results saved to {output}")
```

**Definition of Done:**
- Model evaluation covers all key performance metrics
- Combination evaluation provides clear strategy comparisons
- Backtesting validates system performance over time
- Reports are comprehensive and actionable
- Progress monitoring shows detailed evaluation status

---

### **User Story 2: Interactive Mode with Guided Workflows**
**As a** system user  
**I want** an interactive mode with guided workflows  
**So that** I can explore the system capabilities and get step-by-step assistance

**Acceptance Criteria:**
- [x] Interactive CLI mode with prompt_toolkit
- [x] Command history and auto-completion
- [x] Guided workflows for common tasks
- [x] Context-aware help and suggestions
- [x] Persistent command history across sessions
- [x] Clean exit handling

**Technical Implementation:**
```python
from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style

@cli.command()
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
    
    # Setup auto-completion
    completer = WordCompleter([
        'train', 'predict', 'evaluate', 'features', 'status',
        'config', 'help', 'exit', 'workflows', 'history', 'clear'
    ], ignore_case=True)
    
    # Setup command history
    history = FileHistory('.prevcarga_history')
    
    # Interactive loop
    while True:
        try:
            user_input = prompt(
                'prevcarga> ',
                completer=completer,
                history=history,
                complete_while_typing=True
            )
            
            if not user_input.strip():
                continue
                
            if user_input.lower() in ['exit', 'quit']:
                click.echo("Goodbye!")
                break
            elif user_input.lower() == 'help':
                display_interactive_help()
            elif user_input.lower() == 'workflows':
                start_guided_workflow()
            elif user_input.lower() == 'status':
                display_system_status()
            elif user_input.lower() == 'history':
                display_command_history(history)
            elif user_input.lower() == 'clear':
                click.clear()
            else:
                # Execute command through CLI parser
                execute_interactive_command(user_input, ctx)
                
        except (KeyboardInterrupt, EOFError):
            click.echo("\nGoodbye!")
            break
        except Exception as e:
            click.echo(f"Error: {e}", err=True)

def display_interactive_help():
    """Display interactive mode help."""
    click.echo("\nInteractive Mode Commands:")
    click.echo("  train       - Model training commands")
    click.echo("  predict     - Prediction generation")
    click.echo("  evaluate    - Model evaluation")
    click.echo("  features    - Feature engineering")
    click.echo("  config      - Configuration management")
    click.echo("  status      - System status check")
    click.echo("  workflows   - Guided workflows")
    click.echo("  history     - Show command history")
    click.echo("  clear       - Clear screen")
    click.echo("  help        - Show this help")
    click.echo("  exit/quit   - Exit interactive mode")
    click.echo("\nType any command followed by --help for detailed usage.")

def start_guided_workflow():
    """Start guided workflow selection."""
    workflows = {
        '1': 'Complete model training',
        '2': 'Generate predictions',
        '3': 'Run backtest evaluation',
        '4': 'Feature engineering pipeline',
        '5': 'Model comparison analysis'
    }
    
    click.echo("\n╔═════════════════════════════════════╗")
    click.echo("║    Guided Workflows                 ║")
    click.echo("╚═════════════════════════════════════╝")
    
    for key, description in workflows.items():
        click.echo(f"  {key}. {description}")
    
    choice = prompt("\nSelect workflow (1-5) or 'cancel': ")
    
    if choice == '1':
        guided_training_workflow()
    elif choice == '2':
        guided_prediction_workflow()
    elif choice == '3':
        guided_backtest_workflow()
    elif choice == '4':
        guided_feature_workflow()
    elif choice == '5':
        guided_comparison_workflow()
    elif choice.lower() == 'cancel':
        click.echo("Cancelled")
    else:
        click.echo("Invalid choice")

def guided_training_workflow():
    """Guide user through training workflow."""
    click.echo("\n=== Guided Training Workflow ===")
    
    # Step 1: Select model
    model = prompt("Select model (lgbm/rf/regdin_svm/holt_winters/all): ")
    
    # Step 2: Select areas
    areas_input = prompt("Enter areas (comma-separated, or 'all'): ")
    areas = areas_input.split(',') if areas_input.lower() != 'all' else None
    
    # Step 3: Date range
    start_date = prompt("Start date (YYYY-MM-DD): ")
    end_date = prompt("End date (YYYY-MM-DD): ")
    
    # Step 4: Parallel workers
    parallel = prompt("Parallel workers (default: 4): ") or "4"
    
    # Construct and execute command
    cmd = f"train model --model {model} --start-date {start_date} --end-date {end_date} --parallel {parallel}"
    if areas:
        for area in areas:
            cmd += f" --area {area.strip()}"
    
    click.echo(f"\nExecuting: {cmd}")
    execute_interactive_command(cmd, click.get_current_context())

def guided_prediction_workflow():
    """Guide user through prediction workflow."""
    click.echo("\n=== Guided Prediction Workflow ===")
    
    # Step 1: Prediction type
    pred_type = prompt("Prediction type (batch/intraday): ")
    
    if pred_type.lower() == 'batch':
        date = prompt("Prediction date (YYYY-MM-DD): ")
        horizons = prompt("Horizons (comma-separated, or 'all'): ")
        combination = prompt("Combination method (simple_avg/weighted_avg/stacking/markov): ") or "weighted_avg"
        format_type = prompt("Output format (csv/json/parquet): ") or "csv"
        
        cmd = f"predict batch --date {date} --combination {combination} --format {format_type}"
        if horizons.lower() != 'all':
            for h in horizons.split(','):
                cmd += f" --horizon {h.strip()}"
    else:
        datetime_str = prompt("Current datetime (YYYY-MM-DD HH:MM): ")
        cmd = f"predict intraday --datetime \"{datetime_str}\""
    
    click.echo(f"\nExecuting: {cmd}")
    execute_interactive_command(cmd, click.get_current_context())

def guided_backtest_workflow():
    """Guide user through backtest workflow."""
    click.echo("\n=== Guided Backtest Workflow ===")
    
    start_date = prompt("Start date (YYYY-MM-DD): ")
    end_date = prompt("End date (YYYY-MM-DD): ")
    intervals = prompt("Retraining intervals in days (comma-separated, default: 7,14,30): ") or "7,14,30"
    
    cmd = f"evaluate backtest --start-date {start_date} --end-date {end_date}"
    for interval in intervals.split(','):
        cmd += f" --retraining-interval {interval.strip()}"
    
    click.echo(f"\nExecuting: {cmd}")
    execute_interactive_command(cmd, click.get_current_context())

def execute_interactive_command(command_str: str, ctx):
    """Execute a command string in interactive mode."""
    import shlex
    
    try:
        # Parse command string
        args = shlex.split(command_str)
        
        # Execute through CLI
        ctx.invoke(cli, args)
    except Exception as e:
        click.echo(f"Error executing command: {e}", err=True)
```

**Definition of Done:**
- Interactive mode starts and accepts commands
- Auto-completion works for all commands
- Command history persists across sessions
- Guided workflows provide step-by-step assistance
- Error handling is robust
- Clean exit on Ctrl+C or exit command

---

### **User Story 3: Configuration Management**
**As a** system administrator  
**I want** comprehensive configuration management commands  
**So that** I can validate, view, and manage system configuration

**Acceptance Criteria:**
- [x] Show configuration (full or by section)
- [x] Validate configuration with detailed error reporting
- [x] Initialize configuration from templates
- [x] Environment-specific configuration support
- [x] Configuration diff and comparison
- [x] Hot-reload capability indication

**Technical Implementation:**
```python
@cli.group()
def config():
    """Configuration management commands."""
    pass

@config.command('show')
@click.option('--section', help='Configuration section to display')
@click.option('--format', type=click.Choice(['yaml', 'json', 'table']), 
              default='yaml', help='Output format')
def show_config(section: str, format: str):
    """Display current system configuration.
    
    Examples:
        prevcarga config show
        prevcarga config show --section models
        prevcarga config show --section data --format json
    """
    config = load_config()
    
    if section:
        display_config_section(config, section, format)
    else:
        display_full_config(config, format)

@config.command('validate')
@click.option('--config-file', type=Path, help='Configuration file to validate')
@click.option('--strict', is_flag=True, help='Enable strict validation')
def validate_config(config_file: Path, strict: bool):
    """Validate current configuration for completeness and correctness.
    
    Checks for:
    - Required fields presence
    - Value type correctness
    - Value range validity
    - Cross-field consistency
    - Resource availability (S3, etc.)
    """
    try:
        if config_file:
            config = load_config(config_file)
        else:
            config = load_config()
        
        validation_result = validate_system_config(config, strict=strict)
        
        if validation_result.is_valid:
            click.echo("✓ Configuration is valid")
            click.echo(f"  Validated {validation_result.checks_performed} checks")
        else:
            click.echo("✗ Configuration validation failed:")
            for error in validation_result.errors:
                click.echo(f"  - {error}")
            
            if validation_result.warnings:
                click.echo("\nWarnings:")
                for warning in validation_result.warnings:
                    click.echo(f"  - {warning}")
            
            raise click.ClickException("Configuration is invalid")
                
    except Exception as e:
        raise click.ClickException(f"Configuration error: {e}")

@config.command('init')
@click.option('--template', type=click.Choice(['development', 'production', 'testing']),
              default='development', help='Configuration template')
@click.option('--output', type=Path, help='Output configuration file path')
def init_config(template: str, output: Path):
    """Initialize configuration from template.
    
    Creates a new configuration file with appropriate defaults
    for the specified environment.
    
    Examples:
        prevcarga config init --template production
        prevcarga config init --template development --output custom-config.yaml
    """
    config_path = output or Path('prevcarga.yaml')
    
    if config_path.exists():
        if not click.confirm(f"Configuration file {config_path} exists. Overwrite?"):
            return
    
    create_config_from_template(config_path, template)
    click.echo(f"✓ Configuration initialized: {config_path}")
    click.echo(f"  Template: {template}")
    click.echo("\nNext steps:")
    click.echo("  1. Review and customize the configuration")
    click.echo(f"  2. Validate: prevcarga config validate --config-file {config_path}")
    click.echo("  3. Use: prevcarga --config {config_path} <command>")

@config.command('diff')
@click.option('--config1', type=Path, required=True, help='First configuration file')
@click.option('--config2', type=Path, required=True, help='Second configuration file')
@click.option('--output', type=Path, help='Save diff to file')
def diff_config(config1: Path, config2: Path, output: Path):
    """Compare two configuration files.
    
    Shows differences between configuration files,
    useful for comparing environments or tracking changes.
    """
    cfg1 = load_config(config1)
    cfg2 = load_config(config2)
    
    diff_result = compare_configs(cfg1, cfg2)
    
    display_config_diff(diff_result)
    
    if output:
        save_config_diff(diff_result, output)
        click.echo(f"\nDiff saved to {output}")

@config.command('export')
@click.option('--format', type=click.Choice(['yaml', 'json', 'env']),
              default='yaml', help='Export format')
@click.option('--output', type=Path, help='Output file path')
def export_config(format: str, output: Path):
    """Export current configuration to specified format.
    
    Useful for documentation, backups, or conversion between formats.
    """
    config = load_config()
    
    if format == 'env':
        env_vars = config_to_env_vars(config)
        content = '\n'.join(f"{k}={v}" for k, v in env_vars.items())
    elif format == 'json':
        content = config_to_json(config)
    else:
        content = config_to_yaml(config)
    
    if output:
        output.write_text(content)
        click.echo(f"Configuration exported to {output}")
    else:
        click.echo(content)
```

**Definition of Done:**
- All configuration commands operational
- Configuration validation catches errors
- Template initialization works for all environments
- Configuration diff shows meaningful differences
- Export supports multiple formats

---

## 🏗️ Technical Architecture

### **Interactive Mode Architecture**
```
┌─────────────────────────────────────────────────────────┐
│         Interactive Mode (Epic-09B)                      │
├─────────────────────────────────────────────────────────┤
│  prompt_toolkit  │  History  │  Auto-completion          │
├─────────────────────────────────────────────────────────┤
│  Guided Workflows │  Context  │  Command Parser          │
└─────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────┼─────────────────────────────┐
│     Advanced Commands       │     Configuration Mgmt       │
├─────────────────────────────┼─────────────────────────────┤
│ • evaluate (model, combo)   │ • show / validate           │
│ • backtest                  │ • init / diff / export      │
│ • guided workflows          │ • template management       │
└─────────────────────────────┼─────────────────────────────┘
                              │
┌─────────────────────────────┼─────────────────────────────┐
│      User Experience        │      Integration Layer       │
├─────────────────────────────┼─────────────────────────────┤
│ • Command history           │ • Epic-09A commands         │
│ • Context-aware help        │ • Epic-07A/B evaluation     │
│ • Error recovery            │ • Epic-08A workflows        │
│ • Session persistence       │ • Configuration system      │
└─────────────────────────────┼─────────────────────────────┘
```

### **Guided Workflow System**
```python
class GuidedWorkflowManager:
    """Manage guided workflows for complex operations."""
    
    def __init__(self):
        self.workflows = {
            'training': guided_training_workflow,
            'prediction': guided_prediction_workflow,
            'backtest': guided_backtest_workflow,
            'features': guided_feature_workflow,
            'comparison': guided_comparison_workflow
        }
    
    def list_workflows(self):
        """List available guided workflows."""
        return list(self.workflows.keys())
    
    def execute_workflow(self, workflow_name: str):
        """Execute specified guided workflow."""
        if workflow_name not in self.workflows:
            raise ValueError(f"Unknown workflow: {workflow_name}")
        
        workflow_func = self.workflows[workflow_name]
        workflow_func()
```

---

## 📊 Implementation Plan

### **Week 25: Interactive Mode and Advanced Features**

**Day 1: Evaluation Commands**
- Model evaluation command
- Combination evaluation command
- Backtest command
- Progress bars for evaluation
- Report generation integration

**Day 2: Interactive Mode Core**
- prompt_toolkit integration
- Command history setup
- Auto-completion implementation
- Interactive loop and error handling
- Help system for interactive mode

**Day 3: Guided Workflows**
- Guided training workflow
- Guided prediction workflow
- Guided backtest workflow
- Guided feature workflow
- Workflow selection menu

**Day 4: Configuration Management**
- Show configuration command
- Validate configuration command
- Initialize from template
- Configuration diff
- Export functionality

**Day 5: Testing and Polish**
- End-to-end interactive tests
- User experience refinements
- Documentation completion
- Performance optimization
- Final bug fixes

---

## 🧪 Testing Strategy

### **Unit Testing (80% Coverage Target)**
- Evaluation command parsing
- Interactive mode command execution
- Configuration validation logic
- Guided workflow step handling
- Auto-completion functionality

### **Integration Testing**
- End-to-end evaluation workflows
- Interactive mode with real commands
- Guided workflows execution
- Configuration management operations
- Session persistence

### **User Experience Testing**
- Interactive mode usability
- Guided workflow clarity
- Auto-completion accuracy
- Error message helpfulness
- Response time for commands

### **Performance Testing**
- Interactive mode startup (<2 seconds)
- Command execution response (<1 second)
- Evaluation operation timing
- Memory usage during sessions
- History file performance

---

## 📈 Success Metrics

### **Functionality Metrics**
- **Evaluation Commands:** 100% of evaluation functions accessible
- **Interactive Mode:** All commands executable interactively
- **Guided Workflows:** 5+ workflows for common operations
- **Configuration Management:** Full lifecycle support

### **User Experience Metrics**
- **Interactive Startup:** <2 seconds to ready state
- **Command Response:** <1 second for non-computational commands
- **Workflow Completion:** <3 minutes for guided workflows
- **Error Recovery:** Clear guidance for all failure modes

### **Quality Metrics**
- **Test Coverage:** 80%+ for new code
- **Auto-completion Accuracy:** >95% for valid commands
- **History Reliability:** 100% command persistence
- **Session Stability:** No crashes during normal use

---

## 🔗 Integration Points

### **Upstream Dependencies**
- **Epic-09A:** Core CLI commands and infrastructure
- **Epic-08A:** BacktestingWorkflow, configuration system
- **Epic-07A/B:** ModelEvaluator, CombinationEvaluator, report generation
- **Epic-05A/B:** Combination strategies

### **Downstream Consumers**
- **Epic-10:** CLI-based comprehensive testing
- **Epic-11:** Production operational commands
- **Users:** Interactive exploration and analysis
- **Automation:** Scripted evaluation pipelines

---

## 📚 Documentation Requirements

### **User Documentation**
- Interactive mode user guide
- Guided workflow tutorials
- Evaluation command examples
- Configuration management guide
- Troubleshooting for interactive mode

### **Developer Documentation**
- Adding new guided workflows
- Extending evaluation commands
- Custom auto-completion
- Interactive command patterns
- Configuration template creation

---

## 🎯 Definition of Done

- [ ] All evaluation commands implemented
- [ ] Interactive mode operational with history
- [ ] Guided workflows for 5+ common tasks
- [ ] Configuration management complete
- [ ] Auto-completion works accurately
- [ ] 80%+ test coverage
- [ ] E2E tests pass
- [ ] Documentation complete
- [ ] Performance targets met
- [ ] Ready for Epic-10 (Testing)

---

**Epic Owner:** Backend Engineering Team  
**Technical Reviewers:** ML Engineering Team, UX Team  
**Stakeholders:** Data Scientists, System Operators, End Users

---

**Related Epics:**
- **Epic-09A:** Core CLI commands foundation
- **Epic-08A/B:** Workflow orchestration and execution
- **Epic-07A/B:** Evaluation and monitoring
- **Epic-10:** Comprehensive testing and validation
