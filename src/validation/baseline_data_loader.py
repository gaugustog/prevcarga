"""Baseline data loader for PrevCargaDESSEM historical data.

This module provides functionality to load and process historical prediction
data from the PrevCargaDESSEM R system for baseline validation.

Key Features:
- Load historical predictions from CSV files
- Calculate baseline metrics (MAPE, MAE, RMSE)
- Discover and filter baseline files by date range
- Caching support for performance

Example:
    ```python
    from src.validation.baseline_data_loader import BaselineDataLoader
    from datetime import datetime

    loader = BaselineDataLoader("/data/baseline")
    data = loader.load_historical_data(
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 1, 31)
    )
    ```
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


@dataclass
class BaselineDataset:
    """Container for baseline prediction data.

    Attributes:
        data: DataFrame with predictions, actuals, area, model columns.
        start_date: Start of data range.
        end_date: End of data range.
        models: List of models in dataset.
        areas: List of areas in dataset.
        metadata: Additional metadata.
    """

    data: pd.DataFrame
    start_date: datetime
    end_date: datetime
    models: list[str] = field(default_factory=list)
    areas: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_predictions(self, area: str, model: str) -> np.ndarray:
        """Get predictions for specific area and model.

        Args:
            area: Area name.
            model: Model name.

        Returns:
            Numpy array of predictions.
        """
        mask = (self.data["area"] == area) & (self.data["model"] == model)
        return self.data.loc[mask, "prediction"].values

    def get_actuals(self, area: str) -> np.ndarray:
        """Get actual values for area.

        Args:
            area: Area name.

        Returns:
            Numpy array of actual values.
        """
        mask = self.data["area"] == area
        return self.data.loc[mask, "actual"].values

    def get_mape(self, area: str, model: str) -> float:
        """Get MAPE for specific area and model.

        Args:
            area: Area name.
            model: Model name.

        Returns:
            MAPE value as percentage.
        """
        predictions = self.get_predictions(area, model)
        actuals = self.get_actuals(area)

        if len(predictions) != len(actuals):
            # Align by deduplicating actuals if needed
            unique_actuals = self.data.loc[
                (self.data["area"] == area) & (self.data["model"] == model), "actual"
            ].values
            actuals = unique_actuals

        return _calculate_mape(predictions, actuals)


@dataclass
class PredictionDataset:
    """Container for system prediction data.

    Attributes:
        data: DataFrame with predictions, actuals, area, model columns.
        start_date: Start of data range.
        end_date: End of data range.
        models: List of models in dataset.
        areas: List of areas in dataset.
        metadata: Additional metadata.
    """

    data: pd.DataFrame
    start_date: datetime
    end_date: datetime
    models: list[str] = field(default_factory=list)
    areas: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_predictions(self, area: str, model: str) -> np.ndarray:
        """Get predictions for specific area and model.

        Args:
            area: Area name.
            model: Model name.

        Returns:
            Numpy array of predictions.
        """
        mask = (self.data["area"] == area) & (self.data["model"] == model)
        return self.data.loc[mask, "prediction"].values

    def get_actuals(self, area: str) -> np.ndarray:
        """Get actual values for area.

        Args:
            area: Area name.

        Returns:
            Numpy array of actual values.
        """
        mask = self.data["area"] == area
        # Return unique actuals to avoid duplication
        return self.data.loc[mask, "actual"].drop_duplicates().values


def _calculate_mape(predictions: np.ndarray, actuals: np.ndarray) -> float:
    """Calculate Mean Absolute Percentage Error.

    Args:
        predictions: Predicted values.
        actuals: Actual values.

    Returns:
        MAPE as percentage.
    """
    if len(predictions) == 0 or len(actuals) == 0:
        return np.nan

    # Handle mismatched lengths
    min_len = min(len(predictions), len(actuals))
    predictions = predictions[:min_len]
    actuals = actuals[:min_len]

    # Avoid division by zero
    mask = actuals != 0
    if not mask.any():
        return np.nan

    return float(np.mean(np.abs((actuals[mask] - predictions[mask]) / actuals[mask])) * 100)


def _calculate_mae(predictions: np.ndarray, actuals: np.ndarray) -> float:
    """Calculate Mean Absolute Error.

    Args:
        predictions: Predicted values.
        actuals: Actual values.

    Returns:
        MAE value.
    """
    if len(predictions) == 0 or len(actuals) == 0:
        return np.nan

    min_len = min(len(predictions), len(actuals))
    predictions = predictions[:min_len]
    actuals = actuals[:min_len]

    return float(np.mean(np.abs(actuals - predictions)))


def _calculate_rmse(predictions: np.ndarray, actuals: np.ndarray) -> float:
    """Calculate Root Mean Squared Error.

    Args:
        predictions: Predicted values.
        actuals: Actual values.

    Returns:
        RMSE value.
    """
    if len(predictions) == 0 or len(actuals) == 0:
        return np.nan

    min_len = min(len(predictions), len(actuals))
    predictions = predictions[:min_len]
    actuals = actuals[:min_len]

    return float(np.sqrt(np.mean((actuals - predictions) ** 2)))


class BaselineDataLoader:
    """Loads and processes PrevCargaDESSEM baseline data.

    This class handles loading historical predictions from the PrevCargaDESSEM
    R system and calculating baseline metrics for validation purposes.

    Attributes:
        baseline_path: Path to baseline data directory.
        cache: Cache for loaded data.
    """

    def __init__(self, baseline_path: str | Path) -> None:
        """Initialize loader.

        Args:
            baseline_path: Path to baseline data directory.
        """
        self.baseline_path = Path(baseline_path)
        self.cache: dict[str, pd.DataFrame] = {}
        self._metrics_cache: dict[str, dict[str, dict[str, float]]] = {}

    def load_historical_data(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """Load historical predictions from PrevCargaDESSEM.

        Args:
            start_date: Start of period.
            end_date: End of period.

        Returns:
            DataFrame with historical predictions.
        """
        logger.info(f"Loading baseline data from {start_date} to {end_date}")

        cache_key = f"{start_date.isoformat()}_{end_date.isoformat()}"
        if cache_key in self.cache:
            logger.debug("Returning cached baseline data")
            return self.cache[cache_key]

        # Load from R system output files
        baseline_files = self._discover_baseline_files(start_date, end_date)

        if not baseline_files:
            logger.warning("No baseline files found, returning empty DataFrame")
            return pd.DataFrame(
                columns=["date", "area", "model", "prediction", "actual", "horizon"]
            )

        dataframes = []
        for file_path in baseline_files:
            df = self._load_baseline_file(file_path)
            if df is not None and not df.empty:
                dataframes.append(df)

        if not dataframes:
            logger.warning("No data loaded from baseline files")
            return pd.DataFrame(
                columns=["date", "area", "model", "prediction", "actual", "horizon"]
            )

        # Concatenate and filter
        baseline_data = pd.concat(dataframes, ignore_index=True)

        # Ensure date column is datetime
        if "date" in baseline_data.columns:
            baseline_data["date"] = pd.to_datetime(baseline_data["date"])
            baseline_data = baseline_data[
                (baseline_data["date"] >= start_date)
                & (baseline_data["date"] <= end_date)
            ]

        # Cache result
        self.cache[cache_key] = baseline_data

        logger.info(f"Loaded {len(baseline_data)} baseline records")
        return baseline_data

    def calculate_metrics(
        self,
        start_date: datetime,
        end_date: datetime,
        models: list[str],
    ) -> dict[str, dict[str, float]]:
        """Calculate baseline metrics for specified period and models.

        Args:
            start_date: Start of period.
            end_date: End of period.
            models: List of model names.

        Returns:
            Dictionary of metrics by area and model.
        """
        cache_key = f"{start_date.isoformat()}_{end_date.isoformat()}_{'_'.join(sorted(models))}"
        if cache_key in self._metrics_cache:
            return self._metrics_cache[cache_key]

        baseline_data = self.load_historical_data(start_date, end_date)

        if baseline_data.empty:
            logger.warning("No baseline data to calculate metrics")
            return {}

        metrics: dict[str, dict[str, float]] = {}

        # Get areas from data
        areas = baseline_data["area"].unique().tolist()

        for model in models:
            for area in areas:
                # Get predictions and actuals for this model-area combination
                mask = (baseline_data["model"] == model) & (baseline_data["area"] == area)
                subset = baseline_data[mask]

                if subset.empty:
                    continue

                predictions = subset["prediction"].values
                actuals = subset["actual"].values

                key = f"{area}_{model}"
                metrics[key] = {
                    "mape": _calculate_mape(predictions, actuals),
                    "mae": _calculate_mae(predictions, actuals),
                    "rmse": _calculate_rmse(predictions, actuals),
                }

        self._metrics_cache[cache_key] = metrics
        return metrics

    def to_reference_format(
        self,
        metrics: dict[str, dict[str, float]],
    ) -> dict[str, Any]:
        """Convert metrics to reference format for storage.

        Args:
            metrics: Calculated metrics dictionary.

        Returns:
            Reference format dictionary.
        """
        return {
            "metrics": metrics,
            "generated_at": datetime.now().isoformat(),
            "format_version": "1.0",
        }

    def create_baseline_dataset(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> BaselineDataset:
        """Create a BaselineDataset from loaded data.

        Args:
            start_date: Start of period.
            end_date: End of period.

        Returns:
            BaselineDataset instance.
        """
        data = self.load_historical_data(start_date, end_date)

        models = data["model"].unique().tolist() if not data.empty else []
        areas = data["area"].unique().tolist() if not data.empty else []

        return BaselineDataset(
            data=data,
            start_date=start_date,
            end_date=end_date,
            models=models,
            areas=areas,
            metadata={"source": str(self.baseline_path)},
        )

    def _discover_baseline_files(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> list[Path]:
        """Discover baseline files for date range.

        Args:
            start_date: Start date.
            end_date: End date.

        Returns:
            List of file paths.
        """
        if not self.baseline_path.exists():
            logger.warning(f"Baseline path does not exist: {self.baseline_path}")
            return []

        files = []
        for file_path in self.baseline_path.glob("*.csv"):
            # Try to parse date from filename
            file_date = self._extract_date_from_filename(file_path.name)
            if file_date is None:
                # If no date in filename, include the file
                files.append(file_path)
            elif start_date <= file_date <= end_date:
                files.append(file_path)

        return sorted(files)

    def _load_baseline_file(self, file_path: Path) -> pd.DataFrame | None:
        """Load single baseline file.

        Args:
            file_path: Path to CSV file.

        Returns:
            DataFrame or None if loading fails.
        """
        try:
            df = pd.read_csv(file_path)

            # Try to parse date column if present
            date_columns = ["date", "datetime", "timestamp", "Date", "DateTime"]
            for col in date_columns:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
                    # Standardize to 'date' column
                    if col != "date":
                        df["date"] = df[col]
                    break

            return df

        except Exception as e:
            logger.warning(f"Failed to load {file_path}: {e}")
            return None

    def _load_actuals(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """Load actual load data for period.

        Args:
            start_date: Start of period.
            end_date: End of period.

        Returns:
            DataFrame with actual values.
        """
        actuals_path = self.baseline_path.parent / "actuals"

        if not actuals_path.exists():
            logger.warning(f"Actuals path does not exist: {actuals_path}")
            return pd.DataFrame(columns=["date", "area", "load"])

        # Look for actuals file
        actuals_file = actuals_path / "actuals.csv"
        if not actuals_file.exists():
            # Try glob for any CSV
            csv_files = list(actuals_path.glob("*.csv"))
            if csv_files:
                actuals_file = csv_files[0]
            else:
                return pd.DataFrame(columns=["date", "area", "load"])

        try:
            df = pd.read_csv(actuals_file)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
            return df
        except Exception as e:
            logger.warning(f"Failed to load actuals: {e}")
            return pd.DataFrame(columns=["date", "area", "load"])

    @staticmethod
    def _extract_date_from_filename(filename: str) -> datetime | None:
        """Extract date from filename.

        Args:
            filename: Filename string.

        Returns:
            Datetime or None if no date found.
        """
        # Try common date patterns
        patterns = [
            r"(\d{4}-\d{2}-\d{2})",  # YYYY-MM-DD
            r"(\d{4}_\d{2}_\d{2})",  # YYYY_MM_DD
            r"(\d{8})",  # YYYYMMDD
        ]

        for pattern in patterns:
            match = re.search(pattern, filename)
            if match:
                date_str = match.group(1)
                try:
                    if "-" in date_str:
                        return datetime.strptime(date_str, "%Y-%m-%d")
                    elif "_" in date_str:
                        return datetime.strptime(date_str, "%Y_%m_%d")
                    else:
                        return datetime.strptime(date_str, "%Y%m%d")
                except ValueError:
                    continue

        return None
