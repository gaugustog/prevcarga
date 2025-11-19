# PC-079-09B: Evaluation Commands

**Ticket ID:** PC-079-09B  
**Epic:** Epic-09B - Interactive Mode & Advanced Features  
**Parent Epic:** Epic-09 - CLI & User Interface  
**Story Points:** 8  
**Priority:** High  
**Status:** Not Started  

---

## 📋 Description

Implement comprehensive evaluation commands for model assessment, combination strategy comparison, and backtesting with walk-forward validation.

---

## 🎯 Acceptance Criteria

- [ ] `evaluate model` command with multiple metrics
- [ ] `evaluate combination` command for strategy comparison
- [ ] `evaluate backtest` command with walk-forward validation
- [ ] Metrics support: MAPE, MAE, RMSE, percentiles
- [ ] Cross-validation for combination evaluation
- [ ] Retraining intervals: 7, 14, 30 days (configurable)
- [ ] Report generation in multiple formats
- [ ] Progress monitoring with EvaluationProgressBar/BacktestProgressBar
- [ ] Integration with ModelEvaluator from Epic-07A/B
- [ ] Integration with CombinationEvaluator from Epic-07A/B
- [ ] Integration with BacktestingWorkflow from Epic-08A
- [ ] Result display with detailed statistics

---

## 🔧 Technical Implementation

```python
# prevcarga/cli/commands/evaluate.py
import click
from datetime import datetime
from pathlib import Path

from prevcarga.evaluation import ModelEvaluator, CombinationEvaluator
from prevcarga.workflows import BacktestingWorkflow
from prevcarga.cli.progress import EvaluationProgressBar, BacktestProgressBar


@click.group()
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
@click.pass_context
def evaluate_model(ctx, model: str, area: tuple, start_date: datetime, 
                  end_date: datetime, metrics: tuple, output: Path):
    """Evaluate individual model performance across specified period.
    
    Examples:
        prevcarga evaluate model --model lgbm --area SE --start-date 2024-01-01 --end-date 2024-03-31
        prevcarga evaluate model --model rf --metrics mape --metrics percentiles
    """
    config = ctx.obj.get('system_config')
    areas = list(area) if area else config.data.areas
    metric_list = list(metrics)
    
    click.echo(f"Evaluating {model} model for {len(areas)} areas")
    click.echo(f"  Metrics: {', '.join(metric_list)}")
    click.echo(f"  Period: {start_date.date()} to {end_date.date()}")
    click.echo()
    
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
        click.echo(f"\nReport saved to {output}")


@evaluate.command('combination')
@click.option('--strategy', multiple=True, 
              type=click.Choice(['simple_avg', 'weighted_avg', 'stacking', 'markov']),
              help='Combination strategies to evaluate')
@click.option('--models', multiple=True, help='Models to include in combination')
@click.option('--start-date', type=click.DateTime(['%Y-%m-%d']), required=True)
@click.option('--end-date', type=click.DateTime(['%Y-%m-%d']), required=True)
@click.option('--cross-validation', type=int, default=5, help='Cross-validation folds')
@click.option('--output', type=Path, help='Comparison report output')
@click.pass_context
def evaluate_combination(ctx, strategy: tuple, models: tuple, start_date: datetime,
                        end_date: datetime, cross_validation: int, output: Path):
    """Evaluate and compare model combination strategies.
    
    Examples:
        prevcarga evaluate combination --start-date 2024-01-01 --end-date 2024-03-31
        prevcarga evaluate combination --strategy stacking --strategy markov --cross-validation 10
    """
    config = ctx.obj.get('system_config')
    strategies = list(strategy) if strategy else config.combination.available_strategies
    model_list = list(models) if models else config.models.default_models
    
    click.echo(f"Evaluating {len(strategies)} combination strategies")
    click.echo(f"  Models: {', '.join(model_list)}")
    click.echo(f"  CV folds: {cross_validation}")
    click.echo()
    
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
@click.pass_context
def backtest(ctx, start_date: datetime, end_date: datetime, retraining_interval: tuple,
            models: tuple, areas: tuple, output: Path):
    """Execute comprehensive backtesting with retraining evaluation.
    
    Examples:
        prevcarga evaluate backtest --start-date 2024-01-01 --end-date 2024-12-31
        prevcarga evaluate backtest --start-date 2024-01-01 --end-date 2024-03-31 --retraining-interval 7
    """
    config = ctx.obj.get('system_config')
    intervals = list(retraining_interval)
    model_list = list(models) if models else config.models.default_models
    area_list = list(areas) if areas else config.data.areas
    
    total_days = (end_date - start_date).days
    click.echo(f"Starting backtest: {total_days} days with {len(intervals)} intervals")
    click.echo(f"  Models: {', '.join(model_list)}")
    click.echo(f"  Areas: {len(area_list)}")
    click.echo(f"  Retraining intervals: {intervals}")
    click.echo()
    
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
        click.echo(f"\nResults saved to {output}")
```

---

## ✅ Definition of Done

- [ ] All three evaluation commands implemented
- [ ] Metrics calculation accurate
- [ ] Cross-validation working correctly
- [ ] Backtest walk-forward validation functional
- [ ] Progress monitoring integrated
- [ ] Report generation working
- [ ] Unit tests pass (>80% coverage)
- [ ] Documentation complete

---

**Assignee:** Backend Team  
**Estimated Hours:** 12-16 hours  
**Target Completion:** Week 25, Day 1
