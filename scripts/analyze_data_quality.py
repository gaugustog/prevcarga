#!/usr/bin/env python3
"""
Data Quality Analysis Script

Analyzes the data structure and provides:
- Summary statistics
- Timeline coverage and gaps
- Missing data analysis
- Outlier detection

Usage:
    python scripts/analyze_data_quality.py --source ./data --areas SP
    python scripts/analyze_data_quality.py --source ./data --all-areas
    python scripts/analyze_data_quality.py --source ./data --areas SP --output report.html
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def get_available_areas(data_path: Path) -> list[str]:
    """Get list of available areas from the processed directory."""
    load_dir = data_path / "processed" / "load"
    if not load_dir.exists():
        # Fallback to raw_data
        load_dir = data_path / "raw_data" / "load"
    if not load_dir.exists():
        return []

    areas = []
    for item in load_dir.iterdir():
        if item.is_dir():
            # Handle Hive-style partitioning (area_code=SP)
            if item.name.startswith("area_code="):
                areas.append(item.name.split("=")[1])
            else:
                areas.append(item.name)
    return sorted(areas)


def load_processed_data(data_path: Path, data_type: str, area: str) -> pd.DataFrame:
    """Load processed (consolidated) data for an area."""
    if data_type == "load":
        base_path = data_path / "processed" / "load" / f"area_code={area}"
    elif data_type == "weather_observed":
        base_path = data_path / "processed" / "weather" / "observed" / f"area_code={area}"
    elif data_type == "weather_forecast":
        base_path = data_path / "processed" / "weather" / "forecast" / f"area_code={area}"
    else:
        raise ValueError(f"Unknown data type: {data_type}")

    if not base_path.exists():
        return pd.DataFrame()

    # Read all year partitions
    dfs = []
    for year_dir in sorted(base_path.glob("year=*")):
        parquet_file = year_dir / "data.parquet"
        if parquet_file.exists():
            df = pd.read_parquet(parquet_file)
            dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)


def detect_outliers_iqr(series: pd.Series, multiplier: float = 1.5) -> pd.Series:
    """Detect outliers using IQR method."""
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    return (series < lower_bound) | (series > upper_bound)


def detect_outliers_zscore(series: pd.Series, threshold: float = 3.0) -> pd.Series:
    """Detect outliers using Z-score method."""
    z_scores = np.abs((series - series.mean()) / series.std())
    return z_scores > threshold


def analyze_timeline_gaps(df: pd.DataFrame, time_col: str, expected_freq: str = "30min") -> dict:
    """Analyze timeline for gaps and missing periods."""
    if df.empty or time_col not in df.columns:
        return {"error": "No data or missing time column"}

    df = df.sort_values(time_col)

    # Get time range
    min_time = df[time_col].min()
    max_time = df[time_col].max()

    # Create expected timeline
    expected_times = pd.date_range(start=min_time, end=max_time, freq=expected_freq)
    actual_times = set(df[time_col].dropna())

    # Find missing times
    missing_times = sorted(set(expected_times) - actual_times)

    # Group consecutive missing times into gaps
    gaps = []
    if missing_times:
        gap_start = missing_times[0]
        gap_end = missing_times[0]

        for t in missing_times[1:]:
            if t - gap_end <= pd.Timedelta(expected_freq):
                gap_end = t
            else:
                gaps.append({
                    "start": gap_start,
                    "end": gap_end,
                    "duration": gap_end - gap_start + pd.Timedelta(expected_freq),
                    "missing_points": int((gap_end - gap_start) / pd.Timedelta(expected_freq)) + 1
                })
                gap_start = t
                gap_end = t

        # Don't forget the last gap
        gaps.append({
            "start": gap_start,
            "end": gap_end,
            "duration": gap_end - gap_start + pd.Timedelta(expected_freq),
            "missing_points": int((gap_end - gap_start) / pd.Timedelta(expected_freq)) + 1
        })

    return {
        "time_range": {
            "start": min_time,
            "end": max_time,
            "total_days": (max_time - min_time).days
        },
        "expected_points": len(expected_times),
        "actual_points": len(actual_times),
        "missing_points": len(missing_times),
        "missing_percentage": 100 * len(missing_times) / len(expected_times) if expected_times.size > 0 else 0,
        "gaps_count": len(gaps),
        "gaps": gaps[:20] if len(gaps) > 20 else gaps,  # Limit to first 20 gaps
        "gaps_truncated": len(gaps) > 20
    }


def analyze_load_data(df: pd.DataFrame, area: str) -> dict:
    """Analyze load data quality."""
    if df.empty:
        return {"error": "No data available"}

    results = {
        "area": area,
        "data_type": "load",
        "total_rows": len(df),
        "columns": list(df.columns),
    }

    # Timeline analysis
    results["timeline"] = analyze_timeline_gaps(df, "timestamp", "30min")

    # Basic statistics for load_mwh
    if "load_mwh" in df.columns:
        load_series = df["load_mwh"].dropna()
        results["load_mwh"] = {
            "count": len(load_series),
            "mean": float(load_series.mean()),
            "std": float(load_series.std()),
            "min": float(load_series.min()),
            "max": float(load_series.max()),
            "median": float(load_series.median()),
            "q1": float(load_series.quantile(0.25)),
            "q3": float(load_series.quantile(0.75)),
            "null_count": int(df["load_mwh"].isna().sum()),
            "null_percentage": 100 * df["load_mwh"].isna().sum() / len(df),
        }

        # Outlier detection
        outliers_iqr = detect_outliers_iqr(load_series)
        outliers_zscore = detect_outliers_zscore(load_series)

        results["load_mwh"]["outliers"] = {
            "iqr_method": {
                "count": int(outliers_iqr.sum()),
                "percentage": 100 * outliers_iqr.sum() / len(load_series),
                "samples": df[outliers_iqr.values]["load_mwh"].head(10).tolist() if outliers_iqr.sum() > 0 else []
            },
            "zscore_method": {
                "count": int(outliers_zscore.sum()),
                "percentage": 100 * outliers_zscore.sum() / len(load_series),
            }
        }

        # Negative values check
        negative_count = (load_series < 0).sum()
        results["load_mwh"]["negative_values"] = {
            "count": int(negative_count),
            "percentage": 100 * negative_count / len(load_series) if len(load_series) > 0 else 0
        }

        # Zero values check
        zero_count = (load_series == 0).sum()
        results["load_mwh"]["zero_values"] = {
            "count": int(zero_count),
            "percentage": 100 * zero_count / len(load_series) if len(load_series) > 0 else 0
        }

        # Daily pattern check
        df_copy = df.copy()
        df_copy["hour"] = df_copy["timestamp"].dt.hour
        hourly_avg = df_copy.groupby("hour")["load_mwh"].mean()
        results["load_mwh"]["hourly_pattern"] = {
            "min_hour": int(hourly_avg.idxmin()),
            "max_hour": int(hourly_avg.idxmax()),
            "min_avg": float(hourly_avg.min()),
            "max_avg": float(hourly_avg.max()),
        }

    return results


def analyze_weather_observed(df: pd.DataFrame, area: str) -> dict:
    """Analyze observed weather data quality."""
    if df.empty:
        return {"error": "No data available"}

    results = {
        "area": area,
        "data_type": "weather_observed",
        "total_rows": len(df),
        "columns": list(df.columns),
    }

    # Timeline analysis
    results["timeline"] = analyze_timeline_gaps(df, "timestamp", "30min")

    # Temperature statistics
    if "temperature" in df.columns:
        temp_series = df["temperature"].dropna()
        results["temperature"] = {
            "count": len(temp_series),
            "mean": float(temp_series.mean()),
            "std": float(temp_series.std()),
            "min": float(temp_series.min()),
            "max": float(temp_series.max()),
            "median": float(temp_series.median()),
            "null_count": int(df["temperature"].isna().sum()),
            "null_percentage": 100 * df["temperature"].isna().sum() / len(df),
        }

        # Outlier detection
        outliers_iqr = detect_outliers_iqr(temp_series)
        outliers_zscore = detect_outliers_zscore(temp_series)

        results["temperature"]["outliers"] = {
            "iqr_method": {
                "count": int(outliers_iqr.sum()),
                "percentage": 100 * outliers_iqr.sum() / len(temp_series),
            },
            "zscore_method": {
                "count": int(outliers_zscore.sum()),
                "percentage": 100 * outliers_zscore.sum() / len(temp_series),
            }
        }

        # Climate range check (Brazil: -10 to 50°C)
        out_of_range = ((temp_series < -10) | (temp_series > 50)).sum()
        results["temperature"]["out_of_climate_range"] = {
            "count": int(out_of_range),
            "percentage": 100 * out_of_range / len(temp_series) if len(temp_series) > 0 else 0
        }

    return results


def analyze_weather_forecast(df: pd.DataFrame, area: str) -> dict:
    """Analyze forecast weather data quality."""
    if df.empty:
        return {"error": "No data available"}

    results = {
        "area": area,
        "data_type": "weather_forecast",
        "total_rows": len(df),
        "columns": list(df.columns),
    }

    # Issue time analysis
    if "issue_time" in df.columns:
        issue_times = df["issue_time"].dropna().unique()
        results["issue_time"] = {
            "unique_count": len(issue_times),
            "min": str(pd.Timestamp(min(issue_times))),
            "max": str(pd.Timestamp(max(issue_times))),
        }

    # Valid time analysis
    if "valid_time" in df.columns:
        valid_times = df["valid_time"].dropna().unique()
        results["valid_time"] = {
            "unique_count": len(valid_times),
            "min": str(pd.Timestamp(min(valid_times))),
            "max": str(pd.Timestamp(max(valid_times))),
        }

    # Forecast horizon analysis (valid_time - issue_time)
    if "issue_time" in df.columns and "valid_time" in df.columns:
        df_copy = df.copy()
        df_copy["horizon_hours"] = (df_copy["valid_time"] - df_copy["issue_time"]).dt.total_seconds() / 3600
        results["forecast_horizon"] = {
            "min_hours": float(df_copy["horizon_hours"].min()),
            "max_hours": float(df_copy["horizon_hours"].max()),
            "mean_hours": float(df_copy["horizon_hours"].mean()),
        }

    # Temperature statistics
    if "temperature" in df.columns:
        temp_series = df["temperature"].dropna()
        results["temperature"] = {
            "count": len(temp_series),
            "mean": float(temp_series.mean()),
            "std": float(temp_series.std()),
            "min": float(temp_series.min()),
            "max": float(temp_series.max()),
            "null_count": int(df["temperature"].isna().sum()),
            "null_percentage": 100 * df["temperature"].isna().sum() / len(df),
        }

        # Outlier detection
        outliers_iqr = detect_outliers_iqr(temp_series)
        results["temperature"]["outliers"] = {
            "iqr_method": {
                "count": int(outliers_iqr.sum()),
                "percentage": 100 * outliers_iqr.sum() / len(temp_series),
            }
        }

    return results


def print_report(results: dict, area: str):
    """Print a formatted report to console."""
    print(f"\n{'='*80}")
    print(f"DATA QUALITY REPORT: {area}")
    print(f"{'='*80}")

    for data_type, analysis in results.items():
        if "error" in analysis:
            print(f"\n{data_type.upper()}: {analysis['error']}")
            continue

        print(f"\n{'-'*40}")
        print(f"{data_type.upper()}")
        print(f"{'-'*40}")
        print(f"Total rows: {analysis.get('total_rows', 'N/A'):,}")

        # Timeline
        if "timeline" in analysis:
            tl = analysis["timeline"]
            if "error" not in tl:
                print(f"\nTimeline:")
                print(f"  Range: {tl['time_range']['start']} to {tl['time_range']['end']}")
                print(f"  Total days: {tl['time_range']['total_days']:,}")
                print(f"  Expected points: {tl['expected_points']:,}")
                print(f"  Actual points: {tl['actual_points']:,}")
                print(f"  Missing points: {tl['missing_points']:,} ({tl['missing_percentage']:.2f}%)")
                print(f"  Gaps count: {tl['gaps_count']}")

                if tl['gaps'] and tl['gaps_count'] > 0:
                    print(f"\n  Top gaps (by duration):")
                    sorted_gaps = sorted(tl['gaps'], key=lambda x: x['duration'], reverse=True)[:5]
                    for i, gap in enumerate(sorted_gaps, 1):
                        print(f"    {i}. {gap['start']} to {gap['end']} ({gap['duration']}, {gap['missing_points']} points)")

        # Load statistics
        if "load_mwh" in analysis:
            lm = analysis["load_mwh"]
            print(f"\nLoad (MWh):")
            print(f"  Mean: {lm['mean']:,.2f}")
            print(f"  Std: {lm['std']:,.2f}")
            print(f"  Min: {lm['min']:,.2f}")
            print(f"  Max: {lm['max']:,.2f}")
            print(f"  Median: {lm['median']:,.2f}")
            print(f"  Nulls: {lm['null_count']:,} ({lm['null_percentage']:.2f}%)")

            if "outliers" in lm:
                print(f"\n  Outliers (IQR): {lm['outliers']['iqr_method']['count']:,} ({lm['outliers']['iqr_method']['percentage']:.2f}%)")
                print(f"  Outliers (Z-score): {lm['outliers']['zscore_method']['count']:,} ({lm['outliers']['zscore_method']['percentage']:.2f}%)")

            if "negative_values" in lm:
                print(f"  Negative values: {lm['negative_values']['count']:,} ({lm['negative_values']['percentage']:.2f}%)")

            if "zero_values" in lm:
                print(f"  Zero values: {lm['zero_values']['count']:,} ({lm['zero_values']['percentage']:.2f}%)")

            if "hourly_pattern" in lm:
                hp = lm["hourly_pattern"]
                print(f"\n  Daily pattern:")
                print(f"    Min load hour: {hp['min_hour']:02d}:00 (avg: {hp['min_avg']:,.2f} MWh)")
                print(f"    Max load hour: {hp['max_hour']:02d}:00 (avg: {hp['max_avg']:,.2f} MWh)")

        # Temperature statistics
        if "temperature" in analysis:
            temp = analysis["temperature"]
            print(f"\nTemperature (°C):")
            print(f"  Mean: {temp['mean']:.2f}")
            print(f"  Std: {temp['std']:.2f}")
            print(f"  Min: {temp['min']:.2f}")
            print(f"  Max: {temp['max']:.2f}")
            print(f"  Nulls: {temp['null_count']:,} ({temp['null_percentage']:.2f}%)")

            if "outliers" in temp:
                print(f"  Outliers (IQR): {temp['outliers']['iqr_method']['count']:,} ({temp['outliers']['iqr_method']['percentage']:.2f}%)")

            if "out_of_climate_range" in temp:
                print(f"  Out of climate range (-10 to 50°C): {temp['out_of_climate_range']['count']:,}")

        # Forecast-specific
        if "issue_time" in analysis:
            print(f"\nForecast origins (issue_time):")
            print(f"  Unique: {analysis['issue_time']['unique_count']:,}")
            print(f"  Range: {analysis['issue_time']['min']} to {analysis['issue_time']['max']}")

        if "forecast_horizon" in analysis:
            fh = analysis["forecast_horizon"]
            print(f"\nForecast horizon:")
            print(f"  Min: {fh['min_hours']:.1f} hours ({fh['min_hours']/24:.1f} days)")
            print(f"  Max: {fh['max_hours']:.1f} hours ({fh['max_hours']/24:.1f} days)")
            print(f"  Mean: {fh['mean_hours']:.1f} hours ({fh['mean_hours']/24:.1f} days)")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze data quality and generate report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("./data"),
        help="Data directory containing processed/ (default: ./data)",
    )
    parser.add_argument(
        "--areas",
        type=str,
        default=None,
        help="Comma-separated list of areas to analyze (e.g., SP,RJ,SECO)",
    )
    parser.add_argument(
        "--all-areas",
        action="store_true",
        help="Analyze all available areas",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file for JSON report (optional)",
    )

    args = parser.parse_args()

    # Validate source directory
    if not args.source.exists():
        logger.error(f"Source directory not found: {args.source}")
        sys.exit(1)

    # Determine areas to analyze
    available_areas = get_available_areas(args.source)
    logger.info(f"Available areas: {available_areas}")

    if args.all_areas:
        areas = available_areas
    elif args.areas:
        areas = [a.strip() for a in args.areas.split(",")]
        invalid = set(areas) - set(available_areas)
        if invalid:
            logger.error(f"Invalid areas: {invalid}")
            sys.exit(1)
    else:
        logger.error("Must specify --areas or --all-areas")
        sys.exit(1)

    logger.info(f"Analyzing areas: {areas}")

    all_results = {}

    for area in areas:
        logger.info(f"\nAnalyzing area: {area}")

        area_results = {}

        # Load data
        logger.info(f"  Loading load data...")
        df_load = load_processed_data(args.source, "load", area)
        area_results["load"] = analyze_load_data(df_load, area)

        logger.info(f"  Loading weather observed data...")
        df_weather_obs = load_processed_data(args.source, "weather_observed", area)
        area_results["weather_observed"] = analyze_weather_observed(df_weather_obs, area)

        logger.info(f"  Loading weather forecast data...")
        df_weather_fcst = load_processed_data(args.source, "weather_forecast", area)
        area_results["weather_forecast"] = analyze_weather_forecast(df_weather_fcst, area)

        # Print report
        print_report(area_results, area)

        all_results[area] = area_results

    # Save JSON report if requested
    if args.output:
        import json

        # Convert datetime objects to strings for JSON serialization
        def serialize(obj):
            if isinstance(obj, (pd.Timestamp, datetime)):
                return str(obj)
            if isinstance(obj, timedelta):
                return str(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            return obj

        with open(args.output, "w") as f:
            json.dump(all_results, f, default=serialize, indent=2)
        logger.info(f"\nJSON report saved to: {args.output}")


if __name__ == "__main__":
    main()
