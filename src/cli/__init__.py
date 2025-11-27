"""Command-line interface for PrevCarga.

This package provides the command-line interface for the PrevCarga
electric load forecasting system.

Key Components:
- cli: Main CLI application
- train: Training command with model and batch subcommands
- predict: Prediction command
- backtest: Backtesting command
- status: System status command
- config: Configuration management commands

Example:
    ```bash
    # Show help
    prevcarga --help

    # Train models (advanced)
    prevcarga train model --model lgbm --area SECO \\
        --start-date 2023-01-01 --end-date 2023-12-31

    # Batch training from config
    prevcarga train batch --config-file training.yaml

    # Generate predictions
    prevcarga predict --areas SECO,S

    # Run backtesting
    prevcarga backtest --start 2023-01-01 --end 2023-12-31
    ```
"""

from src.cli.core import (
    CLIContext,
    OutputFormat,
    OutputHelper,
    Verbosity,
    cli,
    main,
)

__all__ = [
    "cli",
    "main",
    "CLIContext",
    "OutputFormat",
    "OutputHelper",
    "Verbosity",
]
