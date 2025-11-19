# Epic-09A: Core CLI Commands

**Epic ID:** Epic-09A  
**Epic Name:** Core CLI Commands  
**Phase:** Phase 9A  
**Duration:** 1 week (Week 24)  
**Dependencies:** Epic-08B (Execution Engine & Monitoring)  
**Priority:** High  
**Status:** Not Started

---

## 🎯 Epic Overview

**Goal:** Implement core command-line interface with essential commands for training, prediction, and feature engineering in the unified electric load forecasting system.

**Problem Statement:**
The PrevCarga system requires a user-friendly CLI interface for:
- Training models across 26 time series with 5 different algorithms
- Generating predictions for batch and intraday scenarios
- Managing feature generation and evaluation pipelines
- Input validation to prevent configuration errors
- Progress monitoring for long-running operations
- Clear help texts and error handling

**Success Criteria:**
✅ CLI application structure operational with Click framework  
✅ Training commands support all 5 model types  
✅ Prediction commands with combination/reconciliation integration  
✅ Feature commands for generation, evaluation, and validation  
✅ Input validation prevents configuration errors  
✅ Progress bars show real-time status for long operations  
✅ Help texts are clear and complete

---

## 📋 User Stories

### **User Story 1: Core CLI Application Structure**
**As a** system administrator  
**I want** a well-structured CLI application  
**So that** I can access all system functions through a consistent command interface

**Acceptance Criteria:**
- [x] Click-based CLI application with hierarchical command structure
- [x] Global configuration options (config file, log level, verbose mode)
- [x] Consistent command naming and parameter conventions
- [x] Error handling with user-friendly messages
- [x] Version information and system status commands
- [x] Auto-completion support for commands and parameters

**Technical Implementation:**
```python
import click
from pathlib import Path
from typing import Optional
import sys
import platform

@click.group()
@click.option('--config', '-c', type=Path, help='Configuration file path')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--log-level', type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR']), 
              default='INFO', help='Set logging level')
@click.pass_context
def cli(ctx: click.Context, config: Optional[Path], verbose: bool, log_level: str):
    """PrevCarga Unified Electric Load Forecasting System
    
    A comprehensive CLI for training, prediction, and evaluation of
    electric load forecasting models across 26 time series.
    """
    ctx.ensure_object(dict)
    ctx.obj['config'] = config
    ctx.obj['verbose'] = verbose
    ctx.obj['log_level'] = log_level
    
    # Initialize logging and configuration
    setup_logging(log_level, verbose)
    load_configuration(config)

@cli.command()
def version():
    """Display version and system information."""
    from prevcarga import __version__
    click.echo(f"PrevCarga CLI v{__version__}")
    click.echo(f"Python: {sys.version}")
    click.echo(f"Platform: {platform.platform()}")

@cli.command()
def status():
    """Check system status and connectivity."""
    # Check S3 connectivity, model registry, etc.
    with click.progressbar(length=5, label='Checking system status') as bar:
        # Check S3 connection
        bar.update(1)
        # Check model registry
        bar.update(1)
        # Check configuration
        bar.update(1)
        # Check data availability
        bar.update(1)
        # Final status
        bar.update(1)
    
    click.echo("✓ System operational")
```

**Definition of Done:**
- CLI application structure supports hierarchical commands
- Global options work across all subcommands
- Error messages are informative and actionable
- Auto-completion works for commands and common parameters
- Version and status commands provide useful information

---

### **User Story 2: Model Training Commands**
**As a** ML engineer  
**I want** comprehensive training commands  
**So that** I can train models efficiently with proper configuration and monitoring

**Acceptance Criteria:**
- [x] Train command for individual models with area selection
- [x] Train-all command for complete system training
- [x] Model type selection (LGBM, RF, RegDin+SVM, Holt-Winters)
- [x] Date range specification for training data
- [x] Parallel training configuration and monitoring
- [x] Training progress visualization and logging

**Technical Implementation:**
```python
from datetime import datetime

@cli.group()
def train():
    """Model training commands."""
    pass

@train.command()
@click.option('--model', '-m', type=click.Choice(['lgbm', 'rf', 'regdin_svm', 'holt_winters', 'all']),
              required=True, help='Model type to train')
@click.option('--area', '-a', multiple=True, help='Areas to train (default: all)')
@click.option('--start-date', type=click.DateTime(['%Y-%m-%d']), required=True,
              help='Training start date (YYYY-MM-DD)')
@click.option('--end-date', type=click.DateTime(['%Y-%m-%d']), required=True,
              help='Training end date (YYYY-MM-DD)')
@click.option('--parallel', '-p', type=int, default=4, help='Parallel training workers')
@click.option('--force', is_flag=True, help='Force retrain existing models')
@click.pass_context
def model(ctx, model: str, area: tuple, start_date: datetime, end_date: datetime, 
          parallel: int, force: bool):
    """Train individual or all models for specified areas and date range.
    
    Examples:
        prevcarga train model --model lgbm --area SE --area S --start-date 2023-01-01 --end-date 2023-12-31
        prevcarga train model --model all --start-date 2023-01-01 --end-date 2023-12-31 --parallel 8
    """
    # Validate inputs
    CLIValidators.validate_date_range(start_date, end_date)
    
    config = load_config(ctx.obj['config'])
    areas = list(area) if area else config.data.areas
    
    CLIValidators.validate_area_selection(areas, config)
    
    # Display training plan
    models_to_train = [model] if model != 'all' else config.models.types
    click.echo(f"Training plan:")
    click.echo(f"  Models: {', '.join(models_to_train)}")
    click.echo(f"  Areas: {len(areas)} areas")
    click.echo(f"  Period: {start_date.date()} to {end_date.date()}")
    click.echo(f"  Parallel workers: {parallel}")
    
    # Execute training with progress bar
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
    
    display_training_results(result)

@train.command()
@click.option('--config-file', type=Path, help='Training configuration YAML file')
@click.option('--dry-run', is_flag=True, help='Show training plan without execution')
def batch(config_file: Path, dry_run: bool):
    """Execute batch training from configuration file.
    
    Configuration file should specify models, areas, date ranges,
    and training parameters for comprehensive training runs.
    """
    training_config = load_training_config(config_file)
    
    if dry_run:
        display_training_plan(training_config)
        return
    
    execute_batch_training(training_config)
```

**Definition of Done:**
- Training commands support all 5 model types
- Area selection works for individual and multiple areas
- Date range validation prevents invalid configurations
- Progress bars show training status in real-time
- Training results are clearly displayed with summaries

---

### **User Story 3: Prediction Commands (Batch & Intraday)**
**As a** system operator  
**I want** flexible prediction commands  
**So that** I can generate forecasts for operational and planning scenarios

**Acceptance Criteria:**
- [x] Batch prediction for all horizons (D+0 to D+8)
- [x] Intraday prediction with BLF strategy
- [x] Output format selection (CSV, JSON, Parquet)
- [x] Model combination and reconciliation integration
- [x] Prediction validation and error handling
- [x] Real-time progress monitoring

**Technical Implementation:**
```python
@cli.group()
def predict():
    """Prediction generation commands."""
    pass

@predict.command()
@click.option('--date', type=click.DateTime(['%Y-%m-%d']), required=True,
              help='Prediction date (YYYY-MM-DD)')
@click.option('--horizon', multiple=True, type=click.IntRange(0, 8),
              help='Forecast horizons (0-8, default: all)')
@click.option('--model', multiple=True, type=click.Choice(['lgbm', 'rf', 'regdin_svm', 'holt_winters']),
              help='Models to use (default: all)')
@click.option('--combination', type=click.Choice(['simple_avg', 'weighted_avg', 'stacking', 'markov']),
              default='weighted_avg', help='Model combination method')
@click.option('--reconciliation', type=click.Choice(['mint', 'ols', 'wls', 'shrinkage']),
              default='mint', help='Hierarchical reconciliation method')
@click.option('--output', '-o', type=Path, help='Output file path')
@click.option('--format', type=click.Choice(['csv', 'json', 'parquet']),
              default='csv', help='Output format')
def batch(date: datetime, horizon: tuple, model: tuple, combination: str,
          reconciliation: str, output: Path, format: str):
    """Generate batch predictions for specified date and horizons.
    
    Examples:
        prevcarga predict batch --date 2024-01-15 --horizon 0 --horizon 1 --format json
        prevcarga predict batch --date 2024-01-15 --combination stacking --reconciliation ols
    """
    config = load_config()
    horizons = list(horizon) if horizon else list(range(9))
    models = list(model) if model else config.models.default_models
    
    click.echo(f"Generating predictions for {date.date()}")
    click.echo(f"  Horizons: D+{min(horizons)} to D+{max(horizons)}")
    click.echo(f"  Models: {', '.join(models)}")
    click.echo(f"  Combination: {combination}")
    click.echo(f"  Reconciliation: {reconciliation}")
    
    with PredictionProgressBar(horizons=horizons, areas=config.data.areas) as progress:
        prediction_workflow = PredictionWorkflow(config)
        result = prediction_workflow.execute_batch_prediction(
            prediction_date=date,
            horizons=horizons,
            models=models,
            combination_method=combination,
            reconciliation_method=reconciliation,
            progress_callback=progress.update
        )
    
    save_predictions(result, output, format)
    display_prediction_summary(result)

@predict.command()
@click.option('--datetime', type=click.DateTime(['%Y-%m-%d %H:%M']), required=True,
              help='Current datetime for intraday update')
@click.option('--data-file', type=Path, help='Updated intraday data file')
@click.option('--output', '-o', type=Path, help='Output file path')
def intraday(datetime: datetime, data_file: Path, output: Path):
    """Generate intraday predictions with latest data updates.
    
    Updates D+0 forecasts using LGBM BLF strategy with latest
    load measurements and reconciles with existing D+1 to D+8 forecasts.
    """
    config = load_config()
    updated_data = load_intraday_data(data_file) if data_file else None
    
    click.echo(f"Generating intraday prediction for {datetime}")
    
    prediction_workflow = PredictionWorkflow(config)
    result = prediction_workflow.execute_intraday_prediction(
        current_datetime=datetime,
        updated_data=updated_data
    )
    
    save_predictions(result, output, 'csv')
    display_intraday_summary(result)
```

**Definition of Done:**
- Batch predictions support all horizons and model combinations
- Intraday predictions integrate BLF strategy correctly
- Output formats are properly validated and generated
- Progress monitoring shows detailed prediction status
- Prediction results include uncertainty quantification

---

### **User Story 4: Feature Engineering Commands**
**As a** data scientist  
**I want** feature engineering commands  
**So that** I can generate, evaluate, and optimize feature sets for model training

**Acceptance Criteria:**
- [x] Feature generation command with plugin selection
- [x] Feature evaluation and importance analysis
- [x] Feature validation and quality checks
- [x] Feature pipeline configuration and testing
- [x] Performance benchmarking for feature generation
- [x] Feature export in multiple formats

**Technical Implementation:**
```python
@cli.group()
def features():
    """Feature engineering and evaluation commands."""
    pass

@features.command('generate')
@click.option('--plugin', multiple=True, help='Feature plugins to use (default: all)')
@click.option('--area', multiple=True, help='Areas to generate features for')
@click.option('--start-date', type=click.DateTime(['%Y-%m-%d']), required=True)
@click.option('--end-date', type=click.DateTime(['%Y-%m-%d']), required=True)
@click.option('--output-dir', type=Path, help='Output directory for features')
@click.option('--parallel', type=int, default=4, help='Parallel processing workers')
def generate_features(plugin: tuple, area: tuple, start_date: datetime,
                     end_date: datetime, output_dir: Path, parallel: int):
    """Generate features using specified plugins and date range.
    
    Examples:
        prevcarga features generate --plugin temporal --plugin calendar --area SE --start-date 2023-01-01 --end-date 2023-12-31
        prevcarga features generate --start-date 2023-01-01 --end-date 2023-12-31 --parallel 8
    """
    config = load_config()
    plugins = list(plugin) if plugin else config.features.default_plugins
    areas = list(area) if area else config.data.areas
    
    click.echo(f"Generating features with plugins: {', '.join(plugins)}")
    click.echo(f"  Areas: {len(areas)}")
    click.echo(f"  Period: {start_date.date()} to {end_date.date()}")
    
    with FeatureProgressBar(plugins=plugins, areas=areas) as progress:
        feature_pipeline = FeaturePipeline(config)
        features = feature_pipeline.generate_features(
            plugins=plugins,
            areas=areas,
            start_date=start_date,
            end_date=end_date,
            parallel_workers=parallel,
            output_dir=output_dir,
            progress_callback=progress.update
        )
    
    display_feature_summary(features)

@features.command('evaluate')
@click.option('--feature-set', type=Path, required=True, help='Feature set to evaluate')
@click.option('--model', type=click.Choice(['lgbm', 'rf']), default='lgbm',
              help='Model for feature importance')
@click.option('--area', help='Area for evaluation (default: national)')
@click.option('--output', type=Path, help='Evaluation report output')
def evaluate_features(feature_set: Path, model: str, area: str, output: Path):
    """Evaluate feature importance and quality metrics.
    
    Generates feature importance rankings, correlation analysis,
    and predictive power assessment using specified model.
    """
    config = load_config()
    features = load_feature_set(feature_set)
    
    click.echo(f"Evaluating features using {model} model")
    click.echo(f"  Area: {area or 'nacional'}")
    
    evaluator = FeatureEvaluator(config)
    evaluation_result = evaluator.evaluate_feature_set(
        features=features,
        model_type=model,
        area=area or 'nacional'
    )
    
    if output:
        save_evaluation_report(evaluation_result, output)
    
    display_feature_evaluation(evaluation_result)

@features.command('validate')
@click.option('--feature-dir', type=Path, required=True, help='Feature directory to validate')
@click.option('--schema-file', type=Path, help='Feature schema file')
def validate_features(feature_dir: Path, schema_file: Path):
    """Validate feature sets for completeness and quality.
    
    Checks for missing values, data types, value ranges,
    and schema compliance across all feature files.
    """
    click.echo(f"Validating features in {feature_dir}")
    
    validator = FeatureValidator(schema_file)
    validation_result = validator.validate_directory(feature_dir)
    
    display_validation_results(validation_result)
    
    if not validation_result.is_valid:
        raise click.ClickException("Feature validation failed")
```

**Definition of Done:**
- Feature generation supports all available plugins
- Feature evaluation provides actionable insights
- Feature validation catches common data quality issues
- Performance metrics help optimize feature pipelines
- Output formats support downstream model training

---

## 🏗️ Technical Architecture

### **CLI Application Structure**
```
┌─────────────────────────────────────────────────────────┐
│              Core CLI Application (Epic-09A)             │
├─────────────────────────────────────────────────────────┤
│  Click Framework  │  Global Options  │  Input Validation │
├─────────────────────────────────────────────────────────┤
│  Command Groups   │  Progress Bars   │  Error Handling   │
└─────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────┼─────────────────────────────┐
│       Core Commands         │        User Experience       │
├─────────────────────────────┼─────────────────────────────┤
│ • version / status          │ • Real-time Progress Bars   │
│ • train (model, batch)      │ • Input Validation          │
│ • predict (batch, intraday) │ • Error Messages            │
│ • features (gen, eval, val) │ • Help Texts                │
└─────────────────────────────┼─────────────────────────────┘
                              │
┌─────────────────────────────┼─────────────────────────────┐
│      Integration Layer      │      Output Handling         │
├─────────────────────────────┼─────────────────────────────┤
│ • Epic-08A/B Workflows      │ • Multiple Output Formats   │
│ • Configuration Management  │ • Console Output            │
│ • Progress Callbacks        │ • Result Summaries          │
└─────────────────────────────┼─────────────────────────────┘
```

### **Progress Monitoring System**
```python
class ProgressBarManager:
    """Centralized progress bar management for CLI commands."""
    
    def __init__(self):
        self.progress_bars = {}
    
    def create_training_progress(self, models: List[str], areas: List[str]):
        """Create progress bar for training operations."""
        total_tasks = len(models) * len(areas)
        return click.progressbar(
            length=total_tasks,
            label='Training models',
            show_eta=True,
            show_percent=True
        )
    
    def create_prediction_progress(self, horizons: List[int], areas: List[str]):
        """Create progress bar for prediction operations."""
        total_tasks = len(horizons) * len(areas)
        return click.progressbar(
            length=total_tasks,
            label='Generating predictions',
            show_eta=True,
            show_percent=True
        )
    
    def create_feature_progress(self, plugins: List[str], areas: List[str]):
        """Create progress bar for feature generation."""
        total_tasks = len(plugins) * len(areas)
        return click.progressbar(
            length=total_tasks,
            label='Generating features',
            show_eta=True
        )
```

### **Input Validation Framework**
```python
class CLIValidators:
    """Input validation utilities for CLI commands."""
    
    @staticmethod
    def validate_date_range(start_date: datetime, end_date: datetime):
        """Validate date range parameters."""
        if start_date >= end_date:
            raise click.BadParameter("Start date must be before end date")
        
        if (end_date - start_date).days > 730:  # 2 years
            click.confirm("Date range exceeds 2 years. Continue?", abort=True)
    
    @staticmethod
    def validate_area_selection(areas: List[str], config):
        """Validate area selection against configuration."""
        valid_areas = set(config.data.areas)
        invalid_areas = set(areas) - valid_areas
        
        if invalid_areas:
            raise click.BadParameter(f"Invalid areas: {', '.join(invalid_areas)}")
    
    @staticmethod
    def validate_model_compatibility(models: List[str], operation: str):
        """Validate model compatibility for specific operations."""
        if operation == 'intraday' and 'lgbm' not in models:
            raise click.BadParameter("Intraday predictions require LGBM model")
    
    @staticmethod
    def validate_output_path(path: Path, format: str):
        """Validate output path and format compatibility."""
        if path and not path.suffix:
            raise click.BadParameter(f"Output path must include file extension (.{format})")
        
        if path and path.exists():
            click.confirm(f"File {path} exists. Overwrite?", abort=True)
```

---

## 📊 Implementation Plan

### **Week 24: Core CLI Implementation**

**Day 1: CLI Application Structure**
- Click application setup with entry point
- Global options and context management
- Version and status commands
- Basic error handling framework
- Auto-completion setup

**Day 2: Training Commands**
- Train model command implementation
- Train batch command implementation
- Input validation for training
- Progress bar integration
- Training result display

**Day 3: Prediction Commands**
- Batch prediction command
- Intraday prediction command
- Output format handling
- Prediction result display
- Integration with Epic-08A workflows

**Day 4: Feature Commands**
- Feature generation command
- Feature evaluation command
- Feature validation command
- Feature result display
- Performance monitoring

**Day 5: Testing and Documentation**
- Unit tests for command parsing
- Integration tests for workflows
- Help text completion
- Usage examples documentation
- Bug fixes and refinements

---

## 🧪 Testing Strategy

### **Unit Testing (80% Coverage Target)**
- Command parsing and parameter validation
- Input validation functions
- Progress bar creation and updates
- Error handling and messages
- Output formatting functions

### **Integration Testing**
- End-to-end training workflow
- End-to-end prediction workflow
- End-to-end feature generation
- Configuration loading and validation
- Progress bar integration with workflows

### **User Experience Testing**
- Command discoverability
- Help text clarity
- Error message usefulness
- Progress bar accuracy
- Result display readability

### **Performance Testing**
- Command startup time (<2 seconds)
- Progress bar update overhead (<5%)
- Memory usage during operations
- Response time for short commands (<1 second)

---

## 📈 Success Metrics

### **Functionality Metrics**
- **Command Coverage:** 100% of core functions (train, predict, features)
- **Input Validation:** 100% prevention of invalid configurations
- **Progress Monitoring:** Real-time updates for all operations >30 seconds
- **Help System:** Complete help text for every command

### **User Experience Metrics**
- **Command Discovery:** <30 seconds to find relevant command
- **Task Completion:** <5 minutes for basic operations
- **Error Recovery:** Clear guidance for all error scenarios
- **Learning Curve:** New users productive within 30 minutes

### **Performance Metrics**
- **Command Startup:** <2 seconds for any command
- **Progress Updates:** <5% overhead on operation time
- **Memory Usage:** <200MB for CLI operations
- **Response Time:** <1 second for informational commands

---

## 🔗 Integration Points

### **Upstream Dependencies**
- **Epic-08A:** TrainingWorkflow, PredictionWorkflow, BacktestingWorkflow
- **Epic-08B:** ParallelExecutor, StructuredLogger
- **Epic-07A:** MetricsCalculator, PercentileAnalyzer
- **Epic-02A/B:** FeaturePipeline, FeatureEvaluator

### **Downstream Consumers**
- **Epic-09B:** Interactive mode and advanced features
- **Epic-10:** CLI-based testing workflows
- **Users:** Data scientists, ML engineers, system operators
- **Automation:** Scheduled jobs and scripts

---

## 📚 Documentation Requirements

### **User Documentation**
- CLI command reference with all options
- Common workflow examples (train, predict, features)
- Input validation rules and constraints
- Error messages and troubleshooting
- Output format specifications

### **Developer Documentation**
- Adding new CLI commands
- Extending command groups
- Custom progress bar implementations
- Input validator patterns
- Integration with workflow systems

---

## 🎯 Definition of Done

- [ ] CLI application structure operational
- [ ] All core commands implemented (train, predict, features)
- [ ] Input validation prevents configuration errors
- [ ] Progress bars show real-time status
- [ ] Help texts complete and accurate
- [ ] 80%+ unit test coverage
- [ ] Integration tests pass
- [ ] Documentation complete
- [ ] Ready for Epic-09B (Interactive Mode)

---

**Epic Owner:** Backend Engineering Team  
**Technical Reviewers:** ML Engineering Team  
**Stakeholders:** Data Scientists, System Operators

---

**Related Epics:**
- **Epic-08A/B:** Workflow orchestration and execution
- **Epic-09B:** Interactive mode and advanced features
- **Epic-10:** Testing and validation
