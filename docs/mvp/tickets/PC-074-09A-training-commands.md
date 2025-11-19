# PC-074-09A: Training Commands

**Ticket ID:** PC-074-09A  
**Epic:** Epic-09A - Core CLI Commands  
**Parent Epic:** Epic-09 - CLI & User Interface  
**Story Points:** 8  
**Priority:** High  
**Status:** Not Started  

---

## 📋 Description

Implement training commands for the CLI that enable model training for individual models or batch training across multiple models, areas, and date ranges. Commands will integrate with TrainingWorkflow from Epic-08A and provide real-time progress monitoring.

---

## 🎯 Acceptance Criteria

- [ ] `train model` command implemented with all options
- [ ] `train batch` command implemented for configuration-based training
- [ ] Model type selection supports: lgbm, rf, regdin_svm, holt_winters, all
- [ ] Area selection with multiple areas support
- [ ] Date range validation (start-date, end-date)
- [ ] Parallel execution support (default 4 workers, configurable)
- [ ] Force retrain option to override existing models
- [ ] Progress monitoring with TrainingProgressBar
- [ ] Training results display with summary statistics
- [ ] Integration with TrainingWorkflow from Epic-08A
- [ ] Error handling for training failures
- [ ] Dry-run mode for batch training

---

## 🔧 Technical Implementation

### **Files to Create/Modify**

```
prevcarga/cli/commands/
├── __init__.py
├── train.py              # Training command group
└── training_display.py   # Training result display utilities
```

### **Training Commands Implementation**

```python
# prevcarga/cli/commands/train.py
import click
from datetime import datetime
from pathlib import Path
from typing import List, Optional
import yaml

from prevcarga.workflows import TrainingWorkflow
from prevcarga.cli.utils import load_config
from prevcarga.cli.validators import CLIValidators
from prevcarga.cli.progress import TrainingProgressBar
from prevcarga.cli.commands.training_display import (
    display_training_plan,
    display_training_results
)


@click.group()
def train():
    """Model training commands.
    
    Train individual models or execute batch training across
    multiple models, areas, and time periods.
    """
    pass


@train.command('model')
@click.option('--model', '-m', 
              type=click.Choice(['lgbm', 'rf', 'regdin_svm', 'holt_winters', 'all']),
              required=True, 
              help='Model type to train')
@click.option('--area', '-a', multiple=True, 
              help='Areas to train (default: all areas from config)')
@click.option('--start-date', type=click.DateTime(['%Y-%m-%d']), required=True,
              help='Training start date (YYYY-MM-DD)')
@click.option('--end-date', type=click.DateTime(['%Y-%m-%d']), required=True,
              help='Training end date (YYYY-MM-DD)')
@click.option('--parallel', '-p', type=int, default=4, 
              help='Number of parallel training workers (default: 4)')
@click.option('--force', is_flag=True, 
              help='Force retrain existing models')
@click.option('--dry-run', is_flag=True,
              help='Show training plan without execution')
@click.pass_context
def train_model(ctx, model: str, area: tuple, start_date: datetime, 
                end_date: datetime, parallel: int, force: bool, dry_run: bool):
    """Train individual or all models for specified areas and date range.
    
    This command trains machine learning models for electric load forecasting
    across specified areas and date ranges. Supports parallel execution for
    faster training across multiple time series.
    
    Examples:
        # Train LGBM for SE and S regions
        prevcarga train model --model lgbm --area SE --area S \\
            --start-date 2023-01-01 --end-date 2023-12-31
        
        # Train all models with 8 parallel workers
        prevcarga train model --model all \\
            --start-date 2023-01-01 --end-date 2023-12-31 --parallel 8
        
        # Force retrain for specific area
        prevcarga train model --model rf --area N --force \\
            --start-date 2023-01-01 --end-date 2023-12-31
        
        # Dry run to see training plan
        prevcarga train model --model all --dry-run \\
            --start-date 2023-01-01 --end-date 2023-12-31
    """
    # Load configuration
    config = ctx.obj.get('system_config') or load_config(ctx.obj.get('config'))
    
    # Validate inputs
    CLIValidators.validate_date_range(start_date, end_date)
    
    # Determine areas to train
    areas = list(area) if area else config.data.areas
    CLIValidators.validate_area_selection(areas, config)
    
    # Determine models to train
    models_to_train = [model] if model != 'all' else config.models.types
    
    # Display training plan
    click.echo("╔════════════════════════════════════════════════╗")
    click.echo("║         Training Plan                          ║")
    click.echo("╚════════════════════════════════════════════════╝")
    click.echo(f"Models:          {', '.join(models_to_train)}")
    click.echo(f"Areas:           {len(areas)} areas")
    click.echo(f"Period:          {start_date.date()} to {end_date.date()}")
    click.echo(f"Parallel:        {parallel} workers")
    click.echo(f"Force retrain:   {'Yes' if force else 'No'}")
    click.echo(f"Total tasks:     {len(models_to_train) * len(areas)}")
    click.echo()
    
    if dry_run:
        display_training_plan(models_to_train, areas, start_date, end_date)
        return
    
    # Confirm execution for large training jobs
    total_tasks = len(models_to_train) * len(areas)
    if total_tasks > 20 and not click.confirm("Proceed with training?"):
        click.echo("Training cancelled")
        return
    
    # Execute training with progress bar
    click.echo("Starting training...")
    click.echo()
    
    try:
        with TrainingProgressBar(models=models_to_train, areas=areas) as progress:
            training_workflow = TrainingWorkflow(config)
            result = training_workflow.execute_training(
                models=models_to_train,
                areas=areas,
                start_date=start_date,
                end_date=end_date,
                parallel_workers=parallel,
                force_retrain=force,
                progress_callback=progress.update
            )
        
        # Display results
        click.echo()
        display_training_results(result)
        
        if result.failed_models:
            click.secho(f"\n⚠ Warning: {len(result.failed_models)} model(s) failed", 
                       fg='yellow', bold=True)
            sys.exit(1)
        else:
            click.secho("\n✓ Training completed successfully", fg='green', bold=True)
            
    except KeyboardInterrupt:
        click.echo("\n\nTraining interrupted by user")
        sys.exit(130)
    except Exception as e:
        click.secho(f"\n✗ Training failed: {e}", fg='red', bold=True)
        raise click.ClickException(str(e))


@train.command('batch')
@click.option('--config-file', type=Path, required=True,
              help='Training configuration YAML file')
@click.option('--dry-run', is_flag=True, 
              help='Show training plan without execution')
@click.pass_context
def train_batch(ctx, config_file: Path, dry_run: bool):
    """Execute batch training from configuration file.
    
    Configuration file should specify models, areas, date ranges,
    and training parameters for comprehensive training runs.
    
    Example configuration (training-config.yaml):
        models:
          - lgbm
          - rf
          - regdin_svm
        areas:
          - SE
          - S
          - NE
          - N
        training_periods:
          - start_date: "2023-01-01"
            end_date: "2023-12-31"
        parallel_workers: 8
        force_retrain: false
    
    Examples:
        # Execute batch training
        prevcarga train batch --config-file training-config.yaml
        
        # Preview training plan
        prevcarga train batch --config-file training-config.yaml --dry-run
    """
    # Validate config file exists
    if not config_file.exists():
        raise click.ClickException(f"Configuration file not found: {config_file}")
    
    # Load training configuration
    with open(config_file) as f:
        training_config = yaml.safe_load(f)
    
    # Load system configuration
    config = ctx.obj.get('system_config') or load_config(ctx.obj.get('config'))
    
    # Validate training configuration
    validate_training_config(training_config, config)
    
    # Display training plan
    display_batch_training_plan(training_config)
    
    if dry_run:
        click.echo("\n[DRY RUN] Training plan displayed above. No training executed.")
        return
    
    # Confirm execution
    if not click.confirm("\nProceed with batch training?"):
        click.echo("Training cancelled")
        return
    
    # Execute batch training
    click.echo("\nStarting batch training...")
    click.echo()
    
    try:
        results = []
        for period in training_config['training_periods']:
            start_date = datetime.strptime(period['start_date'], '%Y-%m-%d')
            end_date = datetime.strptime(period['end_date'], '%Y-%m-%d')
            
            click.echo(f"Training period: {start_date.date()} to {end_date.date()}")
            
            with TrainingProgressBar(
                models=training_config['models'], 
                areas=training_config['areas']
            ) as progress:
                training_workflow = TrainingWorkflow(config)
                result = training_workflow.execute_training(
                    models=training_config['models'],
                    areas=training_config['areas'],
                    start_date=start_date,
                    end_date=end_date,
                    parallel_workers=training_config.get('parallel_workers', 4),
                    force_retrain=training_config.get('force_retrain', False),
                    progress_callback=progress.update
                )
                results.append(result)
            
            click.echo()
        
        # Display aggregate results
        display_batch_results(results)
        
        total_failed = sum(len(r.failed_models) for r in results)
        if total_failed > 0:
            click.secho(f"\n⚠ Warning: {total_failed} model(s) failed across all periods", 
                       fg='yellow', bold=True)
            sys.exit(1)
        else:
            click.secho("\n✓ Batch training completed successfully", fg='green', bold=True)
            
    except KeyboardInterrupt:
        click.echo("\n\nBatch training interrupted by user")
        sys.exit(130)
    except Exception as e:
        click.secho(f"\n✗ Batch training failed: {e}", fg='red', bold=True)
        raise click.ClickException(str(e))


def validate_training_config(training_config: dict, system_config):
    """Validate batch training configuration."""
    required_fields = ['models', 'areas', 'training_periods']
    
    for field in required_fields:
        if field not in training_config:
            raise click.ClickException(f"Missing required field in config: {field}")
    
    # Validate models
    valid_models = set(system_config.models.types)
    for model in training_config['models']:
        if model not in valid_models:
            raise click.ClickException(f"Invalid model type: {model}")
    
    # Validate areas
    valid_areas = set(system_config.data.areas)
    for area in training_config['areas']:
        if area not in valid_areas:
            raise click.ClickException(f"Invalid area: {area}")
    
    # Validate training periods
    for period in training_config['training_periods']:
        if 'start_date' not in period or 'end_date' not in period:
            raise click.ClickException("Training period must have start_date and end_date")


def display_batch_training_plan(training_config: dict):
    """Display batch training plan."""
    click.echo("\n╔════════════════════════════════════════════════╗")
    click.echo("║      Batch Training Configuration              ║")
    click.echo("╚════════════════════════════════════════════════╝")
    click.echo(f"Models:          {', '.join(training_config['models'])}")
    click.echo(f"Areas:           {', '.join(training_config['areas'])}")
    click.echo(f"Periods:         {len(training_config['training_periods'])}")
    click.echo(f"Parallel:        {training_config.get('parallel_workers', 4)} workers")
    click.echo(f"Force retrain:   {training_config.get('force_retrain', False)}")
    
    click.echo("\nTraining Periods:")
    for i, period in enumerate(training_config['training_periods'], 1):
        click.echo(f"  {i}. {period['start_date']} to {period['end_date']}")


def display_batch_results(results: List):
    """Display aggregate batch training results."""
    click.echo("\n╔════════════════════════════════════════════════╗")
    click.echo("║      Batch Training Results                    ║")
    click.echo("╚════════════════════════════════════════════════╝")
    
    total_models = sum(len(r.trained_models) for r in results)
    total_failed = sum(len(r.failed_models) for r in results)
    total_time = sum(r.training_time for r in results)
    
    click.echo(f"Total periods:      {len(results)}")
    click.echo(f"Models trained:     {total_models}")
    click.echo(f"Models failed:      {total_failed}")
    click.echo(f"Total time:         {total_time:.1f}s")
```

### **Training Display Utilities**

```python
# prevcarga/cli/commands/training_display.py
import click
from typing import List
from datetime import datetime

def display_training_plan(models: List[str], areas: List[str], 
                         start_date: datetime, end_date: datetime):
    """Display detailed training plan."""
    click.echo("Detailed Training Plan:")
    click.echo("=" * 60)
    
    for model in models:
        click.echo(f"\nModel: {model}")
        for area in areas:
            click.echo(f"  - Area: {area:10} | Period: {start_date.date()} to {end_date.date()}")


def display_training_results(result):
    """Display training results summary."""
    click.echo("╔════════════════════════════════════════════════╗")
    click.echo("║         Training Results                       ║")
    click.echo("╚════════════════════════════════════════════════╝")
    
    click.echo(f"Models trained:     {len(result.trained_models)}")
    click.echo(f"Models failed:      {len(result.failed_models)}")
    click.echo(f"Training time:      {result.training_time:.1f}s")
    click.echo(f"Avg time/model:     {result.avg_time_per_model:.1f}s")
    
    if result.trained_models:
        click.echo("\n✓ Successfully trained models:")
        for model_info in result.trained_models[:10]:  # Show first 10
            click.echo(f"  • {model_info.model_type} - {model_info.area} "
                      f"(MAPE: {model_info.mape:.2f}%)")
        
        if len(result.trained_models) > 10:
            click.echo(f"  ... and {len(result.trained_models) - 10} more")
    
    if result.failed_models:
        click.echo("\n✗ Failed models:")
        for model_info in result.failed_models:
            click.echo(f"  • {model_info.model_type} - {model_info.area}: {model_info.error}")
```

---

## 🧪 Testing Requirements

### **Unit Tests**

```python
# tests/cli/commands/test_train.py
import pytest
from click.testing import CliRunner
from prevcarga.cli.main import cli
from unittest.mock import Mock, patch

def test_train_model_basic(mock_config, mock_training_workflow):
    """Test basic model training command."""
    runner = CliRunner()
    result = runner.invoke(cli, [
        'train', 'model',
        '--model', 'lgbm',
        '--area', 'SE',
        '--start-date', '2023-01-01',
        '--end-date', '2023-12-31'
    ])
    assert result.exit_code == 0
    assert 'Training completed successfully' in result.output


def test_train_all_models(mock_config):
    """Test training all models."""
    runner = CliRunner()
    result = runner.invoke(cli, [
        'train', 'model',
        '--model', 'all',
        '--start-date', '2023-01-01',
        '--end-date', '2023-12-31'
    ], input='y\n')  # Confirm large job
    assert 'lgbm' in result.output
    assert 'rf' in result.output


def test_train_dry_run():
    """Test dry run mode."""
    runner = CliRunner()
    result = runner.invoke(cli, [
        'train', 'model',
        '--model', 'lgbm',
        '--dry-run',
        '--start-date', '2023-01-01',
        '--end-date', '2023-12-31'
    ])
    assert result.exit_code == 0
    assert 'Training Plan' in result.output


def test_train_batch_from_config(tmp_path):
    """Test batch training from config file."""
    config_file = tmp_path / "training-config.yaml"
    config_file.write_text("""
models:
  - lgbm
areas:
  - SE
training_periods:
  - start_date: "2023-01-01"
    end_date: "2023-12-31"
parallel_workers: 4
    """)
    
    runner = CliRunner()
    result = runner.invoke(cli, [
        'train', 'batch',
        '--config-file', str(config_file)
    ], input='y\n')
    assert result.exit_code == 0
```

---

## 📦 Dependencies

### **Depends On**
- PC-073-09A: Core CLI Structure
- Epic-08A: TrainingWorkflow implementation
- PC-078-09A: Progress Monitoring (TrainingProgressBar)
- PC-077-09A: Input Validation (CLIValidators)

### **Integration Points**
- `TrainingWorkflow` from Epic-08A
- `TrainingProgressBar` from PC-078-09A
- `CLIValidators` from PC-077-09A

---

## ✅ Definition of Done

- [ ] Both training commands implemented and tested
- [ ] All command options working correctly
- [ ] Progress monitoring integrated
- [ ] Error handling robust
- [ ] Unit tests pass (>80% coverage)
- [ ] Integration tests pass
- [ ] Documentation complete
- [ ] Code reviewed
- [ ] Ready for production use

---

**Assignee:** Backend Team  
**Reviewer:** Tech Lead  
**Estimated Hours:** 12-16 hours  
**Target Completion:** Week 24, Day 2
