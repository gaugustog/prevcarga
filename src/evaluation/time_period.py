"""Time period analyzer for segmenting forecast performance by temporal patterns.

This module provides the TimePeriodAnalyzer class that breaks down forecast
performance metrics by different time periods (weekday/weekend, seasons,
holidays, hours, etc.) to identify systematic errors and temporal patterns.

Key Features:
- Weekday vs weekend analysis
- Seasonal patterns (Southern Hemisphere)
- Hour of day analysis
- Brazilian holidays support
- Statistical testing (ANOVA, Kruskal-Wallis)
- Worst/best period identification

Example:
    ```python
    from src.evaluation.time_period import TimePeriodAnalyzer, TimePeriodConfig
    import pandas as pd
    import numpy as np

    # Prepare data
    timestamps = pd.date_range("2023-01-01", periods=8760, freq="h")
    actual = np.random.randn(8760) * 100 + 1000
    forecast = actual + np.random.randn(8760) * 50

    # Configure analyzer
    config = TimePeriodConfig(
        period_types=["weekday", "season", "hour"],
        metrics=["mape", "rmse"],
        min_samples=30,
        timezone="America/Sao_Paulo"
    )

    # Analyze by weekday
    analyzer = TimePeriodAnalyzer(config=config)
    result = analyzer.analyze_by_weekday(actual, forecast, timestamps)

    print(f"Monday MAPE: {result.metrics_by_period['Monday'].mape:.2f}%")
    print(f"Worst day: {result.worst_periods[0]}")
    ```
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator
from scipy import stats

from src.evaluation.metrics import MetricsCalculator, MetricsConfig, MetricsResult
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Brazilian national holidays (fixed dates)
BRAZILIAN_HOLIDAYS = {
    "New Year": (1, 1),
    "Tiradentes": (4, 21),
    "Labor Day": (5, 1),
    "Independence Day": (9, 7),
    "Nossa Senhora Aparecida": (10, 12),
    "All Souls' Day": (11, 2),
    "Republic Day": (11, 15),
    "Black Consciousness Day": (11, 20),
    "Christmas": (12, 25),
}

# Moveable holidays (would need calculation - simplified here)
# Carnival, Corpus Christi, Good Friday vary by year


class TimePeriodConfig(BaseModel):
    """Configuration for time period analysis.

    Attributes:
        period_types: List of period types to analyze.
            Options: "weekday", "season", "hour", "month", "holiday_proximity"
        metrics: List of metrics to compute per period.
            Options: "mape", "mae", "rmse", "mse", "r2"
        min_samples: Minimum samples required per period for valid analysis.
        include_statistical_tests: Whether to run ANOVA/Kruskal-Wallis tests.
        timezone: Timezone for timestamp interpretation.
        alpha: Significance level for statistical tests.

    Example:
        >>> config = TimePeriodConfig(
        ...     period_types=["weekday", "season"],
        ...     metrics=["mape", "rmse"],
        ...     min_samples=30,
        ...     timezone="America/Sao_Paulo"
        ... )
    """

    period_types: list[str] = Field(
        default=["weekday", "season", "hour"],
        description="Period types to analyze",
    )
    metrics: list[str] = Field(
        default=["mape", "mae", "rmse"],
        description="Metrics to compute per period",
    )
    min_samples: int = Field(
        default=30,
        ge=1,
        description="Minimum samples per period",
    )
    include_statistical_tests: bool = Field(
        default=True,
        description="Include statistical tests",
    )
    timezone: str = Field(
        default="America/Sao_Paulo",
        description="Timezone for analysis",
    )
    alpha: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Significance level for tests",
    )

    @field_validator("period_types")
    @classmethod
    def validate_period_types(cls, v: list[str]) -> list[str]:
        """Validate period types.

        Args:
            v: List of period types.

        Returns:
            Validated list.

        Raises:
            ValueError: If invalid period type.
        """
        valid_types = {
            "weekday",
            "season",
            "hour",
            "month",
            "holiday_proximity",
            "day_type",
            "quarter",
        }
        for period_type in v:
            if period_type not in valid_types:
                msg = f"Invalid period_type '{period_type}'. Must be one of {valid_types}"
                raise ValueError(msg)
        return v

    @field_validator("metrics")
    @classmethod
    def validate_metrics(cls, v: list[str]) -> list[str]:
        """Validate metrics.

        Args:
            v: List of metrics.

        Returns:
            Validated list.

        Raises:
            ValueError: If invalid metric.
        """
        valid_metrics = {"mape", "mae", "rmse", "mse", "r2"}
        for metric in v:
            if metric not in valid_metrics:
                msg = f"Invalid metric '{metric}'. Must be one of {valid_metrics}"
                raise ValueError(msg)
        return v


@dataclass
class TimePeriodResult:
    """Results from time period analysis.

    Attributes:
        period_type: Type of period analyzed (e.g., "weekday", "season").
        metrics_by_period: Dictionary mapping period names to MetricsResult.
        worst_periods: List of (period_name, error_value) tuples, worst first.
        best_periods: List of (period_name, error_value) tuples, best first.
        statistical_tests: Results from statistical tests (ANOVA, etc.).
        metadata: Additional metadata about the analysis.

    Example:
        >>> result = TimePeriodResult(
        ...     period_type="weekday",
        ...     metrics_by_period={"Monday": metrics_mon, "Tuesday": metrics_tue},
        ...     worst_periods=[("Sunday", 8.5), ("Saturday", 7.2)],
        ...     best_periods=[("Wednesday", 3.2), ("Thursday", 3.5)],
        ...     statistical_tests={"anova": {"p_value": 0.001}},
        ...     metadata={"timezone": "America/Sao_Paulo"}
        ... )
    """

    period_type: str
    metrics_by_period: dict[str, MetricsResult]
    worst_periods: list[tuple[str, float]]
    best_periods: list[tuple[str, float]]
    statistical_tests: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_periods(self) -> int:
        """Get number of periods analyzed.

        Returns:
            Number of periods.
        """
        return len(self.metrics_by_period)

    @property
    def worst_period(self) -> tuple[str, float] | None:
        """Get worst performing period.

        Returns:
            Tuple of (period_name, error_value) or None.
        """
        if self.worst_periods:
            return self.worst_periods[0]
        return None

    @property
    def best_period(self) -> tuple[str, float] | None:
        """Get best performing period.

        Returns:
            Tuple of (period_name, error_value) or None.
        """
        if self.best_periods:
            return self.best_periods[0]
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "period_type": self.period_type,
            "metrics_by_period": {
                name: result.to_dict() for name, result in self.metrics_by_period.items()
            },
            "worst_periods": self.worst_periods.copy(),
            "best_periods": self.best_periods.copy(),
            "statistical_tests": self.statistical_tests.copy(),
            "metadata": self.metadata.copy(),
        }

    def __repr__(self) -> str:
        """Return string representation.

        Returns:
            String representation.
        """
        return (
            f"TimePeriodResult("
            f"period_type={self.period_type!r}, "
            f"n_periods={self.n_periods}, "
            f"worst={self.worst_period}, "
            f"best={self.best_period})"
        )


class TimePeriodAnalyzer:
    """Time period analyzer for forecast performance segmentation.

    Analyzes forecast performance across different time periods to identify
    systematic errors related to specific days, seasons, holidays, or hours.

    Supported Period Types:
        - weekday: Monday through Sunday analysis
        - season: Summer, Autumn, Winter, Spring (Southern Hemisphere)
        - hour: Hour of day analysis (0-23)
        - month: Monthly analysis (January-December)
        - quarter: Quarterly analysis (Q1-Q4)
        - holiday_proximity: Pre-holiday, post-holiday analysis
        - day_type: Weekday vs weekend

    Features:
        - Brazilian-specific patterns (holidays, seasons, timezone)
        - Statistical testing (ANOVA, Kruskal-Wallis)
        - Worst/best period identification
        - Configurable metrics and thresholds

    Example:
        >>> config = TimePeriodConfig(
        ...     period_types=["weekday", "season"],
        ...     metrics=["mape", "rmse"],
        ...     min_samples=30
        ... )
        >>> analyzer = TimePeriodAnalyzer(config=config)
        >>> result = analyzer.analyze_by_weekday(actual, forecast, timestamps)
        >>> print(result.worst_period)
    """

    def __init__(
        self,
        config: TimePeriodConfig | None = None,
        metrics_config: MetricsConfig | None = None,
    ) -> None:
        """Initialize analyzer.

        Args:
            config: Configuration for time period analysis.
            metrics_config: Configuration for metrics calculation.
        """
        self.config = config or TimePeriodConfig()
        self.metrics_calculator = MetricsCalculator(metrics_config or MetricsConfig())

        logger.debug(
            "Initialized TimePeriodAnalyzer with timezone=%s, min_samples=%d",
            self.config.timezone,
            self.config.min_samples,
        )

    def analyze_by_weekday(
        self,
        actual: np.ndarray | pd.Series,
        forecast: np.ndarray | pd.Series,
        timestamps: pd.DatetimeIndex | pd.Series,
    ) -> TimePeriodResult:
        """Analyze forecast performance by day of week.

        Args:
            actual: Actual values.
            forecast: Forecast values.
            timestamps: Timestamps for each value.

        Returns:
            TimePeriodResult with metrics by weekday.

        Raises:
            ValueError: If input arrays have different lengths.
        """
        # Convert to DataFrame with timestamps
        df = self._prepare_dataframe(actual, forecast, timestamps)

        # Extract weekday names
        df["period"] = df["timestamp"].dt.day_name()

        # Define weekday order
        weekday_order = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]

        return self._analyze_by_period(
            df,
            period_type="weekday",
            period_order=weekday_order,
        )

    def analyze_by_season(
        self,
        actual: np.ndarray | pd.Series,
        forecast: np.ndarray | pd.Series,
        timestamps: pd.DatetimeIndex | pd.Series,
    ) -> TimePeriodResult:
        """Analyze forecast performance by season (Southern Hemisphere).

        Seasons:
            - Summer: December, January, February
            - Autumn: March, April, May
            - Winter: June, July, August
            - Spring: September, October, November

        Args:
            actual: Actual values.
            forecast: Forecast values.
            timestamps: Timestamps for each value.

        Returns:
            TimePeriodResult with metrics by season.

        Raises:
            ValueError: If input arrays have different lengths.
        """
        # Convert to DataFrame with timestamps
        df = self._prepare_dataframe(actual, forecast, timestamps)

        # Extract season (Southern Hemisphere)
        df["period"] = df["timestamp"].apply(self._extract_season)

        # Define season order
        season_order = ["Summer", "Autumn", "Winter", "Spring"]

        return self._analyze_by_period(
            df,
            period_type="season",
            period_order=season_order,
        )

    def analyze_by_hour(
        self,
        actual: np.ndarray | pd.Series,
        forecast: np.ndarray | pd.Series,
        timestamps: pd.DatetimeIndex | pd.Series,
    ) -> TimePeriodResult:
        """Analyze forecast performance by hour of day.

        Args:
            actual: Actual values.
            forecast: Forecast values.
            timestamps: Timestamps for each value.

        Returns:
            TimePeriodResult with metrics by hour (0-23).

        Raises:
            ValueError: If input arrays have different lengths.
        """
        # Convert to DataFrame with timestamps
        df = self._prepare_dataframe(actual, forecast, timestamps)

        # Extract hour
        df["period"] = df["timestamp"].dt.hour.astype(str)

        # Define hour order
        hour_order = [str(h) for h in range(24)]

        return self._analyze_by_period(
            df,
            period_type="hour",
            period_order=hour_order,
        )

    def analyze_by_month(
        self,
        actual: np.ndarray | pd.Series,
        forecast: np.ndarray | pd.Series,
        timestamps: pd.DatetimeIndex | pd.Series,
    ) -> TimePeriodResult:
        """Analyze forecast performance by month.

        Args:
            actual: Actual values.
            forecast: Forecast values.
            timestamps: Timestamps for each value.

        Returns:
            TimePeriodResult with metrics by month.

        Raises:
            ValueError: If input arrays have different lengths.
        """
        # Convert to DataFrame with timestamps
        df = self._prepare_dataframe(actual, forecast, timestamps)

        # Extract month name
        df["period"] = df["timestamp"].dt.month_name()

        # Define month order
        month_order = [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ]

        return self._analyze_by_period(
            df,
            period_type="month",
            period_order=month_order,
        )

    def analyze_by_quarter(
        self,
        actual: np.ndarray | pd.Series,
        forecast: np.ndarray | pd.Series,
        timestamps: pd.DatetimeIndex | pd.Series,
    ) -> TimePeriodResult:
        """Analyze forecast performance by quarter.

        Args:
            actual: Actual values.
            forecast: Forecast values.
            timestamps: Timestamps for each value.

        Returns:
            TimePeriodResult with metrics by quarter (Q1-Q4).

        Raises:
            ValueError: If input arrays have different lengths.
        """
        # Convert to DataFrame with timestamps
        df = self._prepare_dataframe(actual, forecast, timestamps)

        # Extract quarter
        df["period"] = "Q" + df["timestamp"].dt.quarter.astype(str)

        # Define quarter order
        quarter_order = ["Q1", "Q2", "Q3", "Q4"]

        return self._analyze_by_period(
            df,
            period_type="quarter",
            period_order=quarter_order,
        )

    def analyze_by_day_type(
        self,
        actual: np.ndarray | pd.Series,
        forecast: np.ndarray | pd.Series,
        timestamps: pd.DatetimeIndex | pd.Series,
    ) -> TimePeriodResult:
        """Analyze forecast performance by day type (weekday vs weekend).

        Args:
            actual: Actual values.
            forecast: Forecast values.
            timestamps: Timestamps for each value.

        Returns:
            TimePeriodResult with metrics by day type.

        Raises:
            ValueError: If input arrays have different lengths.
        """
        # Convert to DataFrame with timestamps
        df = self._prepare_dataframe(actual, forecast, timestamps)

        # Extract day type
        # Monday=0, ..., Friday=4, Saturday=5, Sunday=6
        weekend_start_day = 5
        df["period"] = df["timestamp"].dt.dayofweek.apply(
            lambda x: "Weekend" if x >= weekend_start_day else "Weekday"
        )

        # Define day type order
        day_type_order = ["Weekday", "Weekend"]

        return self._analyze_by_period(
            df,
            period_type="day_type",
            period_order=day_type_order,
        )

    def analyze_by_holiday_proximity(
        self,
        actual: np.ndarray | pd.Series,
        forecast: np.ndarray | pd.Series,
        timestamps: pd.DatetimeIndex | pd.Series,
    ) -> TimePeriodResult:
        """Analyze forecast performance by holiday proximity.

        Categories:
            - Holiday: The holiday itself
            - Pre-Holiday: Day before holiday
            - Post-Holiday: Day after holiday
            - Regular: All other days

        Args:
            actual: Actual values.
            forecast: Forecast values.
            timestamps: Timestamps for each value.

        Returns:
            TimePeriodResult with metrics by holiday proximity.

        Raises:
            ValueError: If input arrays have different lengths.
        """
        # Convert to DataFrame with timestamps
        df = self._prepare_dataframe(actual, forecast, timestamps)

        # Extract holiday proximity
        df["period"] = df["timestamp"].apply(self._extract_holiday_proximity)

        # Define proximity order
        proximity_order = ["Holiday", "Pre-Holiday", "Post-Holiday", "Regular"]

        return self._analyze_by_period(
            df,
            period_type="holiday_proximity",
            period_order=proximity_order,
        )

    def compare_periods(
        self,
        actual: np.ndarray | pd.Series,
        forecast: np.ndarray | pd.Series,
        timestamps: pd.DatetimeIndex | pd.Series,
        period_type: str,
    ) -> TimePeriodResult:
        """Compare forecast performance across periods with statistical tests.

        Args:
            actual: Actual values.
            forecast: Forecast values.
            timestamps: Timestamps for each value.
            period_type: Type of period to analyze.

        Returns:
            TimePeriodResult with comprehensive comparison.

        Raises:
            ValueError: If invalid period_type.
        """
        # Route to appropriate analysis method
        period_type_mapping = {
            "weekday": self.analyze_by_weekday,
            "season": self.analyze_by_season,
            "hour": self.analyze_by_hour,
            "month": self.analyze_by_month,
            "quarter": self.analyze_by_quarter,
            "day_type": self.analyze_by_day_type,
            "holiday_proximity": self.analyze_by_holiday_proximity,
        }

        if period_type not in period_type_mapping:
            msg = f"Invalid period_type: {period_type}"
            raise ValueError(msg)

        return period_type_mapping[period_type](actual, forecast, timestamps)

    def identify_worst_periods(
        self,
        result: TimePeriodResult,
        n: int = 3,
    ) -> list[tuple[str, float]]:
        """Identify the N worst performing periods.

        Args:
            result: TimePeriodResult from analysis.
            n: Number of worst periods to return.

        Returns:
            List of (period_name, mape) tuples.
        """
        return result.worst_periods[:n]

    def generate_period_report(
        self,
        result: TimePeriodResult,
    ) -> str:
        """Generate text summary report of period analysis.

        Args:
            result: TimePeriodResult from analysis.

        Returns:
            Formatted text report.
        """
        lines = []
        lines.append(f"Time Period Analysis Report: {result.period_type}")
        lines.append("=" * 60)
        lines.append(f"Number of periods analyzed: {result.n_periods}")
        lines.append(f"Total samples: {result.metadata.get('total_samples', 'N/A')}")
        lines.append("")

        # Best and worst periods
        if result.best_period:
            best_name, best_value = result.best_period
            lines.append(f"Best period: {best_name} (MAPE: {best_value:.2f}%)")

        if result.worst_period:
            worst_name, worst_value = result.worst_period
            lines.append(f"Worst period: {worst_name} (MAPE: {worst_value:.2f}%)")

        lines.append("")

        # Period-by-period breakdown
        lines.append("Period-by-Period Metrics:")
        lines.append("-" * 60)
        for period_name, metrics in result.metrics_by_period.items():
            lines.append(
                f"  {period_name:15s} | "
                f"MAPE: {metrics.mape:6.2f}% | "
                f"MAE: {metrics.mae:8.2f} | "
                f"RMSE: {metrics.rmse:8.2f} | "
                f"n={metrics.sample_size}"
            )

        # Statistical tests
        if result.statistical_tests:
            lines.append("")
            lines.append("Statistical Tests:")
            lines.append("-" * 60)

            if "anova" in result.statistical_tests:
                anova = result.statistical_tests["anova"]
                lines.append(
                    f"  ANOVA: F={anova['f_statistic']:.4f}, " f"p-value={anova['p_value']:.4f}"
                )
                if anova.get("significant", False):
                    lines.append("  → Significant difference between periods detected!")

            if "kruskal_wallis" in result.statistical_tests:
                kw = result.statistical_tests["kruskal_wallis"]
                lines.append(
                    f"  Kruskal-Wallis: H={kw['h_statistic']:.4f}, " f"p-value={kw['p_value']:.4f}"
                )
                if kw.get("significant", False):
                    lines.append("  → Significant difference between periods detected!")

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)

    # Private helper methods

    def _prepare_dataframe(
        self,
        actual: np.ndarray | pd.Series,
        forecast: np.ndarray | pd.Series,
        timestamps: pd.DatetimeIndex | pd.Series,
    ) -> pd.DataFrame:
        """Prepare DataFrame with actual, forecast, and timestamps.

        Args:
            actual: Actual values.
            forecast: Forecast values.
            timestamps: Timestamps for each value.

        Returns:
            DataFrame with columns: actual, forecast, timestamp.

        Raises:
            ValueError: If arrays have different lengths.
        """
        # Convert to arrays
        actual_arr = np.asarray(actual, dtype=np.float64)
        forecast_arr = np.asarray(forecast, dtype=np.float64)

        # Validate shapes
        if actual_arr.shape != forecast_arr.shape:
            msg = f"Shape mismatch: actual {actual_arr.shape} vs forecast {forecast_arr.shape}"
            raise ValueError(msg)

        # Convert timestamps to DatetimeIndex
        if isinstance(timestamps, pd.Series) or not isinstance(timestamps, pd.DatetimeIndex):
            timestamps = pd.DatetimeIndex(timestamps)

        # Validate length
        if len(timestamps) != len(actual_arr):
            msg = f"Length mismatch: timestamps {len(timestamps)} vs actual {len(actual_arr)}"
            raise ValueError(msg)

        # Localize to timezone if needed
        if timestamps.tz is None:
            timestamps = timestamps.tz_localize(self.config.timezone)
        else:
            timestamps = timestamps.tz_convert(self.config.timezone)

        # Create DataFrame
        df = pd.DataFrame(
            {
                "actual": actual_arr,
                "forecast": forecast_arr,
                "timestamp": timestamps,
            }
        )

        # Remove NaN values
        df = df.dropna()

        if len(df) == 0:
            msg = "All values are NaN after cleaning"
            raise ValueError(msg)

        return df

    def _analyze_by_period(
        self,
        df: pd.DataFrame,
        period_type: str,
        period_order: list[str] | None = None,
    ) -> TimePeriodResult:
        """Generic period analysis method.

        Args:
            df: DataFrame with columns: actual, forecast, timestamp, period.
            period_type: Type of period being analyzed.
            period_order: Optional order for periods (for consistent sorting).

        Returns:
            TimePeriodResult with analysis.
        """
        # Compute metrics for each period
        metrics_by_period = {}
        sample_counts = {}

        periods = df["period"].unique()
        if period_order is not None:
            # Filter to only existing periods and maintain order
            periods = [p for p in period_order if p in periods]

        for period_name in periods:
            mask = df["period"] == period_name
            n_samples = mask.sum()

            if n_samples >= self.config.min_samples:
                actual_period = df.loc[mask, "actual"].to_numpy()
                forecast_period = df.loc[mask, "forecast"].to_numpy()

                metrics = self.metrics_calculator.calculate_single(
                    actual_period,
                    forecast_period,
                )
                metrics_by_period[period_name] = metrics
                sample_counts[period_name] = n_samples
            else:
                logger.debug(
                    "Skipping period %s with only %d samples (min: %d)",
                    period_name,
                    n_samples,
                    self.config.min_samples,
                )

        # Rank periods by MAPE
        worst_periods = self._rank_periods(metrics_by_period, ascending=False)
        best_periods = self._rank_periods(metrics_by_period, ascending=True)

        # Run statistical tests
        min_groups_for_testing = 2
        statistical_tests = {}
        if (
            self.config.include_statistical_tests
            and len(metrics_by_period) >= min_groups_for_testing
        ):
            statistical_tests = self._run_statistical_tests(df, period_order or list(periods))

        # Build metadata
        metadata = {
            "timezone": str(self.config.timezone),
            "total_samples": len(df),
            "n_periods_analyzed": len(metrics_by_period),
            "sample_counts": sample_counts,
            "date_range": (
                str(df["timestamp"].min()),
                str(df["timestamp"].max()),
            ),
        }

        return TimePeriodResult(
            period_type=period_type,
            metrics_by_period=metrics_by_period,
            worst_periods=worst_periods,
            best_periods=best_periods,
            statistical_tests=statistical_tests,
            metadata=metadata,
        )

    def _rank_periods(
        self,
        metrics_by_period: dict[str, MetricsResult],
        ascending: bool = True,
    ) -> list[tuple[str, float]]:
        """Rank periods by MAPE.

        Args:
            metrics_by_period: Dictionary of period name to MetricsResult.
            ascending: If True, rank from best (lowest MAPE) to worst.

        Returns:
            List of (period_name, mape) tuples, sorted.
        """
        if not metrics_by_period:
            return []

        periods_with_mape = [(name, result.mape) for name, result in metrics_by_period.items()]

        # Sort by MAPE
        periods_with_mape.sort(key=lambda x: x[1], reverse=not ascending)

        return periods_with_mape

    def _run_statistical_tests(
        self,
        df: pd.DataFrame,
        period_order: list[str],
    ) -> dict[str, Any]:
        """Run statistical tests to compare periods.

        Args:
            df: DataFrame with actual, forecast, and period columns.
            period_order: List of period names to include.

        Returns:
            Dictionary with test results.
        """
        tests = {}

        # Prepare groups for testing
        min_groups_for_testing = 2
        groups = []
        group_labels = []

        for period_name in period_order:
            mask = df["period"] == period_name
            if mask.sum() >= self.config.min_samples:
                actual = df.loc[mask, "actual"].to_numpy()
                forecast = df.loc[mask, "forecast"].to_numpy()
                errors = np.abs(actual - forecast)
                groups.append(errors)
                group_labels.append(period_name)

        if len(groups) < min_groups_for_testing:
            return {"note": "Insufficient groups for statistical testing"}

        # ANOVA test (parametric)
        try:
            f_stat, p_value = stats.f_oneway(*groups)
            tests["anova"] = {
                "f_statistic": float(f_stat),
                "p_value": float(p_value),
                "significant": p_value < self.config.alpha,
                "n_groups": len(groups),
                "alpha": self.config.alpha,
            }
        except Exception as e:
            logger.warning("ANOVA test failed: %s", e)
            tests["anova"] = {"error": str(e)}

        # Kruskal-Wallis test (non-parametric)
        try:
            h_stat, p_value = stats.kruskal(*groups)
            tests["kruskal_wallis"] = {
                "h_statistic": float(h_stat),
                "p_value": float(p_value),
                "significant": p_value < self.config.alpha,
                "n_groups": len(groups),
                "alpha": self.config.alpha,
            }
        except Exception as e:
            logger.warning("Kruskal-Wallis test failed: %s", e)
            tests["kruskal_wallis"] = {"error": str(e)}

        return tests

    def _extract_season(self, timestamp: pd.Timestamp) -> str:
        """Extract season for Southern Hemisphere.

        Seasons:
            - Summer: December (12), January (1), February (2)
            - Autumn: March (3), April (4), May (5)
            - Winter: June (6), July (7), August (8)
            - Spring: September (9), October (10), November (11)

        Args:
            timestamp: Timestamp to extract season from.

        Returns:
            Season name.
        """
        month = timestamp.month

        if month in [12, 1, 2]:
            return "Summer"
        if month in [3, 4, 5]:
            return "Autumn"
        if month in [6, 7, 8]:
            return "Winter"
        # 9, 10, 11
        return "Spring"

    def _extract_holiday_proximity(self, timestamp: pd.Timestamp) -> str:
        """Extract holiday proximity category.

        Args:
            timestamp: Timestamp to classify.

        Returns:
            Holiday proximity category.
        """
        month = timestamp.month
        day = timestamp.day

        # Check if it's a Brazilian holiday
        if (month, day) in BRAZILIAN_HOLIDAYS.values():
            return "Holiday"

        # Check if it's pre-holiday (day before)
        next_day = timestamp + pd.Timedelta(days=1)
        if (next_day.month, next_day.day) in BRAZILIAN_HOLIDAYS.values():
            return "Pre-Holiday"

        # Check if it's post-holiday (day after)
        prev_day = timestamp - pd.Timedelta(days=1)
        if (prev_day.month, prev_day.day) in BRAZILIAN_HOLIDAYS.values():
            return "Post-Holiday"

        return "Regular"

    def __repr__(self) -> str:
        """Return string representation.

        Returns:
            String representation.
        """
        return (
            f"TimePeriodAnalyzer("
            f"timezone={self.config.timezone!r}, "
            f"min_samples={self.config.min_samples})"
        )
