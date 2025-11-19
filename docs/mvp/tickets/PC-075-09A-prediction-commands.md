# PC-075-09A: Prediction Commands

**Ticket ID:** PC-075-09A  
**Epic:** Epic-09A - Core CLI Commands  
**Parent Epic:** Epic-09 - CLI & User Interface  
**Story Points:** 8  
**Priority:** High  
**Status:** Not Started  

---

## 📋 Description

Implement prediction commands for batch (D+0 to D+8) and intraday forecasting scenarios with model combination, hierarchical reconciliation, and multiple output formats.

---

## 🎯 Acceptance Criteria

- [ ] `predict batch` command for all horizons (D+0 to D+8)
- [ ] `predict intraday` command with BLF strategy
- [ ] Horizon selection (single or multiple)
- [ ] Model selection support
- [ ] Combination method selection (simple_avg, weighted_avg, stacking, markov)
- [ ] Reconciliation method selection (mint, ols, wls, shrinkage)
- [ ] Output format selection (CSV, JSON, Parquet)
- [ ] Output file path specification
- [ ] Progress monitoring with PredictionProgressBar
- [ ] Prediction result display with summary
- [ ] Integration with PredictionWorkflow from Epic-08A
- [ ] Uncertainty quantification in output

---

## 🔧 Technical Implementation

### **Files to Create/Modify**

```
prevcarga/cli/commands/
├── predict.py              # Prediction command group
└── prediction_display.py   # Prediction result display utilities
```

### **Implementation**

```python
# prevcarga/cli/commands/predict.py
import click
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from prevcarga.workflows import PredictionWorkflow
from prevcarga.cli.utils import load_config
from prevcarga.cli.validators import CLIValidators
from prevcarga.cli.progress import PredictionProgressBar
from prevcarga.cli.commands.prediction_display import (
    display_prediction_summary,
    display_intraday_summary,
    save_predictions
)


@click.group()
def predict():
    """Prediction generation commands.
    
    Generate batch predictions for multiple horizons (D+0 to D+8) or
    intraday predictions with BLF (Best Linear Forecast) strategy.
    """
    pass


@predict.command('batch')
@click.option('--date', type=click.DateTime(['%Y-%m-%d']), required=True,
              help='Prediction date (YYYY-MM-DD)')
@click.option('--horizon', multiple=True, type=click.IntRange(0, 8),
              help='Forecast horizons (0-8, default: all)')
@click.option('--model', multiple=True, 
              type=click.Choice(['lgbm', 'rf', 'regdin_svm', 'holt_winters']),
              help='Models to use (default: all)')
@click.option('--combination', 
              type=click.Choice(['simple_avg', 'weighted_avg', 'stacking', 'markov']),
              default='weighted_avg', 
              help='Model combination method (default: weighted_avg)')
@click.option('--reconciliation', 
              type=click.Choice(['mint', 'ols', 'wls', 'shrinkage']),
              default='mint', 
              help='Hierarchical reconciliation method (default: mint)')
@click.option('--output', '-o', type=Path, 
              help='Output file path')
@click.option('--format', 'output_format',
              type=click.Choice(['csv', 'json', 'parquet']),
              default='csv', 
              help='Output format (default: csv)')
@click.option('--include-uncertainty', is_flag=True,
              help='Include uncertainty quantification in output')
@click.pass_context
def batch_predict(ctx, date: datetime, horizon: tuple, model: tuple, 
                 combination: str, reconciliation: str, output: Path, 
                 output_format: str, include_uncertainty: bool):
    """Generate batch predictions for specified date and horizons.
    
    Generates forecasts across multiple horizons (D+0 to D+8) using
    selected models, combination strategies, and hierarchical reconciliation.
    
    Examples:
        # Generate D+0 and D+1 predictions
        prevcarga predict batch --date 2024-01-15 --horizon 0 --horizon 1
        
        # All horizons with stacking and MinT reconciliation
        prevcarga predict batch --date 2024-01-15 \\
            --combination stacking --reconciliation mint
        
        # Specific models with JSON output
        prevcarga predict batch --date 2024-01-15 \\
            --model lgbm --model rf --format json --output predictions.json
        
        # Include uncertainty quantification
        prevcarga predict batch --date 2024-01-15 --include-uncertainty
    """
    config = ctx.obj.get('system_config') or load_config(ctx.obj.get('config'))
    
    # Determine horizons
    horizons = list(horizon) if horizon else list(range(9))
    
    # Determine models
    models = list(model) if model else config.models.default_models
    
    # Validate output path if specified
    if output:
        CLIValidators.validate_output_path(output, output_format)
    
    # Display prediction plan
    click.echo("╔════════════════════════════════════════════════╗")
    click.echo("║         Prediction Plan                        ║")
    click.echo("╚════════════════════════════════════════════════╝")
    click.echo(f"Date:            {date.date()}")
    click.echo(f"Horizons:        D+{min(horizons)} to D+{max(horizons)}")
    click.echo(f"Models:          {', '.join(models)}")
    click.echo(f"Combination:     {combination}")
    click.echo(f"Reconciliation:  {reconciliation}")
    click.echo(f"Output format:   {output_format}")
    click.echo(f"Areas:           {len(config.data.areas)}")
    click.echo()
    
    # Execute prediction with progress bar
    click.echo("Generating predictions...")
    
    try:
        with PredictionProgressBar(horizons=horizons, areas=config.data.areas) as progress:
            prediction_workflow = PredictionWorkflow(config)
            result = prediction_workflow.execute_batch_prediction(
                prediction_date=date,
                horizons=horizons,
                models=models,
                combination_method=combination,
                reconciliation_method=reconciliation,
                include_uncertainty=include_uncertainty,
                progress_callback=progress.update
            )
        
        # Save predictions
        if output:
            save_predictions(result, output, output_format)
            click.echo(f"\n✓ Predictions saved to: {output}")
        
        # Display summary
        click.echo()
        display_prediction_summary(result)
        
        click.secho("\n✓ Batch prediction completed successfully", fg='green', bold=True)
        
    except KeyboardInterrupt:
        click.echo("\n\nPrediction interrupted by user")
        sys.exit(130)
    except Exception as e:
        click.secho(f"\n✗ Prediction failed: {e}", fg='red', bold=True)
        raise click.ClickException(str(e))


@predict.command('intraday')
@click.option('--datetime', type=click.DateTime(['%Y-%m-%d %H:%M']), required=True,
              help='Current datetime for intraday update (YYYY-MM-DD HH:MM)')
@click.option('--data-file', type=Path, 
              help='Updated intraday data file (CSV with latest load measurements)')
@click.option('--output', '-o', type=Path, 
              help='Output file path')
@click.option('--format', 'output_format',
              type=click.Choice(['csv', 'json', 'parquet']),
              default='csv', 
              help='Output format (default: csv)')
@click.pass_context
def intraday_predict(ctx, datetime: datetime, data_file: Path, 
                    output: Path, output_format: str):
    """Generate intraday predictions with latest data updates.
    
    Updates D+0 forecasts using LGBM BLF (Best Linear Forecast) strategy
    with latest load measurements. Reconciles with existing D+1 to D+8 forecasts.
    
    The BLF strategy combines:
    - LGBM model predictions
    - Latest observed load values
    - Optimal blending weights
    
    Examples:
        # Standard intraday update
        prevcarga predict intraday --datetime "2024-01-15 14:30"
        
        # With custom data file
        prevcarga predict intraday --datetime "2024-01-15 14:30" \\
            --data-file latest_loads.csv
        
        # Save to specific location
        prevcarga predict intraday --datetime "2024-01-15 14:30" \\
            --output intraday_forecast.csv
    """
    config = ctx.obj.get('system_config') or load_config(ctx.obj.get('config'))
    
    # Validate model compatibility
    CLIValidators.validate_model_compatibility(['lgbm'], 'intraday')
    
    # Load updated data if provided
    updated_data = None
    if data_file:
        if not data_file.exists():
            raise click.ClickException(f"Data file not found: {data_file}")
        updated_data = load_intraday_data(data_file)
        click.echo(f"Loaded updated data from: {data_file}")
    
    # Validate output path if specified
    if output:
        CLIValidators.validate_output_path(output, output_format)
    
    # Display prediction plan
    click.echo("╔════════════════════════════════════════════════╗")
    click.echo("║      Intraday Prediction Plan                  ║")
    click.echo("╚════════════════════════════════════════════════╝")
    click.echo(f"DateTime:        {datetime}")
    click.echo(f"Strategy:        LGBM BLF")
    click.echo(f"Horizon:         D+0 (current day)")
    click.echo(f"Data source:     {data_file if data_file else 'S3 latest'}")
    click.echo()
    
    # Execute intraday prediction
    click.echo("Generating intraday prediction...")
    
    try:
        prediction_workflow = PredictionWorkflow(config)
        result = prediction_workflow.execute_intraday_prediction(
            current_datetime=datetime,
            updated_data=updated_data
        )
        
        # Save predictions
        if output:
            save_predictions(result, output, output_format)
            click.echo(f"\n✓ Predictions saved to: {output}")
        
        # Display summary
        click.echo()
        display_intraday_summary(result)
        
        click.secho("\n✓ Intraday prediction completed successfully", fg='green', bold=True)
        
    except Exception as e:
        click.secho(f"\n✗ Intraday prediction failed: {e}", fg='red', bold=True)
        raise click.ClickException(str(e))


def load_intraday_data(data_file: Path):
    """Load intraday data from CSV file."""
    import pandas as pd
    
    try:
        data = pd.read_csv(data_file, parse_dates=['timestamp'])
        
        # Validate required columns
        required_cols = ['timestamp', 'area', 'load']
        missing_cols = set(required_cols) - set(data.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        return data
    except Exception as e:
        raise click.ClickException(f"Error loading intraday data: {e}")
```

### **Display Utilities**

```python
# prevcarga/cli/commands/prediction_display.py
import click
from pathlib import Path
import pandas as pd

def display_prediction_summary(result):
    """Display batch prediction summary."""
    click.echo("╔════════════════════════════════════════════════╗")
    click.echo("║      Batch Prediction Results                  ║")
    click.echo("╚════════════════════════════════════════════════╝")
    
    click.echo(f"Horizons:        {len(result.horizons)}")
    click.echo(f"Areas:           {len(result.areas)}")
    click.echo(f"Total forecasts: {result.total_forecasts}")
    click.echo(f"Execution time:  {result.execution_time:.1f}s")
    
    # Summary statistics
    if hasattr(result, 'statistics'):
        click.echo("\nForecast Statistics:")
        click.echo(f"  Mean load:     {result.statistics.mean_load:.2f} MW")
        click.echo(f"  Max load:      {result.statistics.max_load:.2f} MW")
        click.echo(f"  Min load:      {result.statistics.min_load:.2f} MW")
    
    # Uncertainty info
    if hasattr(result, 'uncertainty'):
        click.echo("\nUncertainty Quantification:")
        click.echo(f"  Avg PI width:  {result.uncertainty.avg_pi_width:.2f} MW")
        click.echo(f"  Coverage:      {result.uncertainty.coverage:.1f}%")


def display_intraday_summary(result):
    """Display intraday prediction summary."""
    click.echo("╔════════════════════════════════════════════════╗")
    click.echo("║      Intraday Prediction Results               ║")
    click.echo("╚════════════════════════════════════════════════╝")
    
    click.echo(f"Update time:     {result.update_datetime}")
    click.echo(f"Strategy:        LGBM BLF")
    click.echo(f"Areas updated:   {len(result.areas)}")
    click.echo(f"Execution time:  {result.execution_time:.1f}s")
    
    # BLF blending info
    if hasattr(result, 'blf_weights'):
        click.echo("\nBLF Blending Weights:")
        click.echo(f"  Model weight:  {result.blf_weights.model:.3f}")
        click.echo(f"  Observed weight: {result.blf_weights.observed:.3f}")
    
    # Forecast improvement
    if hasattr(result, 'improvement'):
        click.echo(f"\nForecast Improvement:")
        click.echo(f"  MAPE reduction: {result.improvement.mape_reduction:.2f}%")


def save_predictions(result, output_path: Path, format: str):
    """Save predictions to file."""
    df = result.to_dataframe()
    
    if format == 'csv':
        df.to_csv(output_path, index=False)
    elif format == 'json':
        df.to_json(output_path, orient='records', date_format='iso')
    elif format == 'parquet':
        df.to_parquet(output_path, index=False)
```

---

## 🧪 Testing Requirements

```python
# tests/cli/commands/test_predict.py
def test_batch_predict_basic():
    """Test basic batch prediction."""
    runner = CliRunner()
    result = runner.invoke(cli, [
        'predict', 'batch',
        '--date', '2024-01-15'
    ])
    assert result.exit_code == 0


def test_batch_predict_specific_horizons():
    """Test batch prediction with specific horizons."""
    runner = CliRunner()
    result = runner.invoke(cli, [
        'predict', 'batch',
        '--date', '2024-01-15',
        '--horizon', '0',
        '--horizon', '1'
    ])
    assert 'D+0 to D+1' in result.output


def test_intraday_predict():
    """Test intraday prediction."""
    runner = CliRunner()
    result = runner.invoke(cli, [
        'predict', 'intraday',
        '--datetime', '2024-01-15 14:30'
    ])
    assert result.exit_code == 0
    assert 'LGBM BLF' in result.output
```

---

## ✅ Definition of Done

- [ ] Both prediction commands implemented
- [ ] All combination and reconciliation methods supported
- [ ] Output formats working correctly
- [ ] Progress monitoring integrated
- [ ] Unit tests pass (>80% coverage)
- [ ] Integration tests pass
- [ ] Documentation complete
- [ ] Code reviewed

---

**Assignee:** Backend Team  
**Estimated Hours:** 12-16 hours  
**Target Completion:** Week 24, Day 3
