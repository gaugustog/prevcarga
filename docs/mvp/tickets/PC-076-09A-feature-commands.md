# PC-076-09A: Feature Commands

**Ticket ID:** PC-076-09A  
**Epic:** Epic-09A - Core CLI Commands  
**Parent Epic:** Epic-09 - CLI & User Interface  
**Story Points:** 6  
**Priority:** High  
**Status:** Not Started  

---

## 📋 Description

Implement feature engineering commands for generation, evaluation, and validation of feature sets used in model training.

---

## 🎯 Acceptance Criteria

- [ ] `features generate` command with plugin selection
- [ ] `features evaluate` command for importance analysis
- [ ] `features validate` command for quality checks
- [ ] Plugin selection (temporal, calendar, weather, lag, rolling, etc.)
- [ ] Area selection support
- [ ] Date range specification
- [ ] Parallel processing configuration
- [ ] Output directory management
- [ ] Multiple output formats
- [ ] Progress monitoring with FeatureProgressBar
- [ ] Integration with FeaturePipeline from Epic-02A/B
- [ ] Schema-based validation

---

## 🔧 Technical Implementation

```python
# prevcarga/cli/commands/features.py
import click
from datetime import datetime
from pathlib import Path

from prevcarga.features import FeaturePipeline, FeatureEvaluator, FeatureValidator
from prevcarga.cli.progress import FeatureProgressBar


@click.group()
def features():
    """Feature engineering and evaluation commands."""
    pass


@features.command('generate')
@click.option('--plugin', multiple=True, 
              help='Feature plugins to use (default: all)')
@click.option('--area', multiple=True, 
              help='Areas to generate features for')
@click.option('--start-date', type=click.DateTime(['%Y-%m-%d']), required=True)
@click.option('--end-date', type=click.DateTime(['%Y-%m-%d']), required=True)
@click.option('--output-dir', type=Path, 
              help='Output directory for features')
@click.option('--parallel', type=int, default=4, 
              help='Parallel processing workers')
@click.option('--format', type=click.Choice(['parquet', 'csv', 'feather']),
              default='parquet',
              help='Output format (default: parquet)')
@click.pass_context
def generate_features(ctx, plugin: tuple, area: tuple, start_date: datetime,
                     end_date: datetime, output_dir: Path, parallel: int, format: str):
    """Generate features using specified plugins and date range.
    
    Available plugins:
    - temporal: Hour, day, month, year features
    - calendar: Holiday, weekend, business day indicators
    - weather: Temperature, humidity, precipitation
    - lag: Historical load lag features
    - rolling: Rolling statistics (mean, std, min, max)
    - fourier: Fourier terms for seasonality
    - interaction: Feature interactions and combinations
    
    Examples:
        # Generate temporal and calendar features
        prevcarga features generate --plugin temporal --plugin calendar \\
            --area SE --start-date 2023-01-01 --end-date 2023-12-31
        
        # All features with 8 parallel workers
        prevcarga features generate --start-date 2023-01-01 \\
            --end-date 2023-12-31 --parallel 8 --output-dir features/
    """
    config = ctx.obj.get('system_config')
    plugins = list(plugin) if plugin else config.features.default_plugins
    areas = list(area) if area else config.data.areas
    
    # Validate date range
    CLIValidators.validate_date_range(start_date, end_date)
    
    # Setup output directory
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path('features')
        output_dir.mkdir(exist_ok=True)
    
    click.echo("╔════════════════════════════════════════════════╗")
    click.echo("║      Feature Generation Plan                   ║")
    click.echo("╚════════════════════════════════════════════════╝")
    click.echo(f"Plugins:         {', '.join(plugins)}")
    click.echo(f"Areas:           {len(areas)}")
    click.echo(f"Period:          {start_date.date()} to {end_date.date()}")
    click.echo(f"Parallel:        {parallel} workers")
    click.echo(f"Output dir:      {output_dir}")
    click.echo(f"Output format:   {format}")
    click.echo()
    
    try:
        with FeatureProgressBar(plugins=plugins, areas=areas) as progress:
            feature_pipeline = FeaturePipeline(config)
            features = feature_pipeline.generate_features(
                plugins=plugins,
                areas=areas,
                start_date=start_date,
                end_date=end_date,
                parallel_workers=parallel,
                output_dir=output_dir,
                output_format=format,
                progress_callback=progress.update
            )
        
        display_feature_summary(features, output_dir)
        click.secho("\n✓ Feature generation completed", fg='green', bold=True)
        
    except Exception as e:
        click.secho(f"\n✗ Feature generation failed: {e}", fg='red', bold=True)
        raise click.ClickException(str(e))


@features.command('evaluate')
@click.option('--feature-set', type=Path, required=True, 
              help='Feature set directory to evaluate')
@click.option('--model', type=click.Choice(['lgbm', 'rf']), default='lgbm',
              help='Model for feature importance')
@click.option('--area', help='Area for evaluation (default: nacional)')
@click.option('--output', type=Path, help='Evaluation report output')
@click.option('--top-n', type=int, default=20,
              help='Number of top features to display')
@click.pass_context
def evaluate_features(ctx, feature_set: Path, model: str, area: str, 
                     output: Path, top_n: int):
    """Evaluate feature importance and quality metrics.
    
    Generates:
    - Feature importance rankings
    - Correlation analysis
    - Predictive power assessment
    - Feature stability metrics
    
    Examples:
        prevcarga features evaluate --feature-set features/ --model lgbm
        prevcarga features evaluate --feature-set features/ --area SE --top-n 30
    """
    if not feature_set.exists():
        raise click.ClickException(f"Feature set not found: {feature_set}")
    
    config = ctx.obj.get('system_config')
    
    click.echo(f"Evaluating features using {model} model")
    click.echo(f"Feature set: {feature_set}")
    click.echo(f"Area: {area or 'nacional'}")
    click.echo()
    
    features = load_feature_set(feature_set)
    
    evaluator = FeatureEvaluator(config)
    evaluation_result = evaluator.evaluate_feature_set(
        features=features,
        model_type=model,
        area=area or 'nacional'
    )
    
    display_feature_evaluation(evaluation_result, top_n)
    
    if output:
        save_evaluation_report(evaluation_result, output)
        click.echo(f"\n✓ Report saved to: {output}")


@features.command('validate')
@click.option('--feature-dir', type=Path, required=True, 
              help='Feature directory to validate')
@click.option('--schema-file', type=Path, 
              help='Feature schema file')
@click.option('--strict', is_flag=True,
              help='Enable strict validation')
@click.pass_context
def validate_features(ctx, feature_dir: Path, schema_file: Path, strict: bool):
    """Validate feature sets for completeness and quality.
    
    Checks:
    - Missing values
    - Data types
    - Value ranges
    - Schema compliance
    - Temporal consistency
    - Cross-area consistency
    
    Examples:
        prevcarga features validate --feature-dir features/
        prevcarga features validate --feature-dir features/ --schema-file schema.yaml --strict
    """
    if not feature_dir.exists():
        raise click.ClickException(f"Feature directory not found: {feature_dir}")
    
    click.echo(f"Validating features in {feature_dir}")
    if strict:
        click.echo("Strict mode: Enabled")
    click.echo()
    
    validator = FeatureValidator(schema_file)
    validation_result = validator.validate_directory(feature_dir, strict=strict)
    
    display_validation_results(validation_result)
    
    if not validation_result.is_valid:
        click.secho("\n✗ Feature validation failed", fg='red', bold=True)
        raise click.ClickException("Feature validation errors found")
    else:
        click.secho("\n✓ Feature validation passed", fg='green', bold=True)
```

---

## ✅ Definition of Done

- [ ] All three feature commands implemented
- [ ] Plugin system integrated
- [ ] Progress monitoring working
- [ ] Output formats supported
- [ ] Unit tests pass (>80% coverage)
- [ ] Documentation complete

---

**Assignee:** Backend Team  
**Estimated Hours:** 10-14 hours  
**Target Completion:** Week 24, Day 4
