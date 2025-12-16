#!/usr/bin/env python3
"""
Heat Index Population Script

Populates heat_index and rh (relative humidity) columns in weather parquet files.

For observed data:
    - Extracts data from heatindex.zip (ERA5-based calculations)
    - Interpolates hourly → semi-hourly using cubic spline
    - Updates weather/observed parquet files with heat_index and rh

For forecast data:
    - Uses observed RH at the same valid_time
    - Calculates heat_index using NOAA formula

Usage:
    python scripts/populate_heat_index.py --heatindex-zip ./data/heatindex.zip --data-dir ./data
    python scripts/populate_heat_index.py --data-dir ./data --observed-only
    python scripts/populate_heat_index.py --data-dir ./data --forecast-only
"""

import argparse
import io
import logging
import math
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Area code mapping: ZIP name -> parquet name
AREA_MAPPING = {"TO": "TON"}

# Areas to skip (aggregates that use reconciliation)
SKIP_AREAS = {"SECO", "S", "N", "NE", "PEN", "PES", "PENE", "PESE"}


def parse_brazilian_number(value: str) -> float:
    """Parse Brazilian number format (comma as decimal separator)."""
    if pd.isna(value) or value == "":
        return float("nan")
    if isinstance(value, (int, float)):
        return float(value)
    return float(str(value).replace(",", "."))


def calc_heat_index(temp_f: float, rh: float) -> float:
    """
    Calculate heat index from temperature (Fahrenheit) and relative humidity (%).

    Uses NOAA formula with adjustments for different temperature/humidity ranges.
    Returns heat index in Celsius.
    """
    if np.isnan(temp_f) or np.isnan(rh):
        return np.nan

    if temp_f < 80:
        hi = 0.5 * (temp_f + 61.0 + ((temp_f - 68.0) * 1.2) + (rh * 0.094))
    else:
        hi = (
            -42.379
            + 2.04901523 * temp_f
            + 10.14333127 * rh
            - 0.22475541 * temp_f * rh
            - 6.83783e-3 * temp_f**2
            - 5.481717e-2 * rh**2
            + 1.22874e-3 * temp_f**2 * rh
            + 8.5282e-4 * temp_f * rh**2
            - 1.99e-6 * temp_f**2 * rh**2
        )

        if rh < 13 and 80 <= temp_f <= 112:
            adjustment = ((13 - rh) / 4) * math.sqrt((17 - abs(temp_f - 95)) / 17)
            hi -= adjustment
        elif rh > 85 and 80 <= temp_f <= 87:
            adjustment = ((rh - 85) / 10) * ((87 - temp_f) / 5)
            hi += adjustment

    # Convert Fahrenheit to Celsius
    return (hi - 32) * 5 / 9


def calc_heat_index_vec(temp_c: np.ndarray, rh: np.ndarray) -> np.ndarray:
    """Vectorized heat index calculation from Celsius temperature and RH."""
    # Convert Celsius to Fahrenheit
    temp_f = temp_c * 9 / 5 + 32
    return np.array([calc_heat_index(t, r) for t, r in zip(temp_f, rh)])


def interpolate_to_semihourly(
    df: pd.DataFrame,
    time_col: str,
    value_cols: list[str],
) -> pd.DataFrame:
    """
    Interpolate hourly data to semi-hourly using cubic spline.

    Args:
        df: DataFrame with hourly data
        time_col: Name of the timestamp column
        value_cols: List of columns to interpolate

    Returns:
        DataFrame with semi-hourly data
    """
    df = df.sort_values(time_col).copy()

    if len(df) < 2:
        return df

    # Get time range
    min_time = df[time_col].min()
    max_time = df[time_col].max()

    # Create semi-hourly time index
    semihourly_times = pd.date_range(start=min_time, end=max_time, freq="30min")

    # Convert timestamps to numeric for interpolation
    time_numeric = (df[time_col] - min_time).dt.total_seconds().values
    new_time_numeric = (semihourly_times - min_time).total_seconds().values

    # Prepare result DataFrame
    result = pd.DataFrame({time_col: semihourly_times})

    # Interpolate each value column
    for col in value_cols:
        if col in df.columns:
            values = df[col].values

            # Handle NaN values
            valid_mask = ~np.isnan(values)
            if valid_mask.sum() < 2:
                result[col] = np.nan
                continue

            valid_times = time_numeric[valid_mask]
            valid_values = values[valid_mask]

            try:
                cs = CubicSpline(valid_times, valid_values, extrapolate=False)
                result[col] = cs(new_time_numeric)
            except Exception as e:
                logger.warning(f"Cubic spline failed for {col}, using linear: {e}")
                result[col] = np.interp(new_time_numeric, valid_times, valid_values)

    return result


def read_heatindex_zip(zip_path: Path) -> dict[str, pd.DataFrame]:
    """
    Read all heatindex CSV files from the ZIP archive.

    Returns:
        Dictionary mapping area_code -> DataFrame with heat index data
    """
    logger.info(f"Reading heat index data from {zip_path}")

    area_data = {}

    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            if name.endswith(".csv") and "heatindex_" in name:
                # Extract area code from filename (e.g., "heatindex_SP.csv" -> "SP")
                area = name.split("heatindex_")[-1].replace(".csv", "")

                # Map area code if needed
                area = AREA_MAPPING.get(area, area)

                logger.info(f"  Reading {name} -> area {area}")

                with zf.open(name) as f:
                    content = f.read().decode("utf-8")
                    df = pd.read_csv(io.StringIO(content), sep=";", low_memory=False)

                # Convert numeric columns (Brazilian format)
                for col in ["temperatura", "rh", "heatindex"]:
                    if col in df.columns:
                        df[col] = df[col].apply(parse_brazilian_number)

                # Convert timestamp
                df["timestamp"] = pd.to_datetime(df["din_referenciautc"])
                df["area_code"] = area

                # Select and rename columns
                df = df[["timestamp", "area_code", "temperatura", "rh", "heatindex"]]
                df = df.rename(columns={"temperatura": "temperature", "heatindex": "heat_index"})

                area_data[area] = df
                logger.info(f"    Loaded {len(df):,} rows for {area}")

    logger.info(f"Loaded heat index data for {len(area_data)} areas")
    return area_data


def update_observed_parquet(
    data_dir: Path,
    heatindex_data: dict[str, pd.DataFrame],
) -> int:
    """
    Update weather/observed parquet files with heat_index and rh.

    Returns number of files updated.
    """
    logger.info("Updating observed weather parquet files...")

    files_updated = 0
    observed_dir = data_dir / "weather" / "observed"

    for area_dir in observed_dir.iterdir():
        if not area_dir.is_dir():
            continue

        # Extract area code from directory name
        area = area_dir.name.replace("area_code=", "")

        # Skip aggregate areas
        if area in SKIP_AREAS:
            logger.info(f"  Skipping aggregate area: {area}")
            continue

        # Check if we have heat index data for this area
        if area not in heatindex_data:
            logger.warning(f"  No heat index data for area: {area}")
            continue

        hi_df = heatindex_data[area].copy()

        # Interpolate to semi-hourly
        logger.info(f"  Interpolating heat index for {area}...")
        hi_interpolated = interpolate_to_semihourly(hi_df, "timestamp", ["heat_index", "rh"])
        hi_interpolated["area_code"] = area

        # Process each year directory
        for year_dir in area_dir.iterdir():
            if not year_dir.is_dir():
                continue

            year = int(year_dir.name.replace("year=", ""))
            parquet_file = year_dir / "data.parquet"

            if not parquet_file.exists():
                continue

            # Read existing parquet
            df = pd.read_parquet(parquet_file)
            original_len = len(df)

            # Filter heat index data for this year
            hi_year = hi_interpolated[hi_interpolated["timestamp"].dt.year == year].copy()

            if len(hi_year) == 0:
                logger.info(f"    {area}/{year}: No heat index data for this year")
                continue

            # Merge on timestamp
            df = df.drop(columns=["heat_index", "rh"], errors="ignore")

            # Ensure timestamp columns are compatible (remove timezone info for merge)
            df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
            hi_year["timestamp"] = pd.to_datetime(hi_year["timestamp"]).dt.tz_localize(None)

            # Merge
            df = df.merge(
                hi_year[["timestamp", "heat_index", "rh"]],
                on="timestamp",
                how="left",
            )

            # Ensure column order
            columns = ["timestamp", "area_code", "temperature", "heat_index", "rh"]
            df = df[columns]

            # Write back
            df.to_parquet(parquet_file, index=False)

            # Count non-null heat_index
            non_null = df["heat_index"].notna().sum()
            logger.info(f"    {area}/{year}: {original_len:,} rows, {non_null:,} with heat_index")

            files_updated += 1

    logger.info(f"Updated {files_updated} observed parquet files")
    return files_updated


def update_forecast_parquet(
    data_dir: Path,
    heatindex_data: dict[str, pd.DataFrame],
) -> int:
    """
    Update weather/forecast parquet files with calculated heat_index.

    Uses observed RH at same valid_time to calculate heat_index.

    Returns number of files updated.
    """
    logger.info("Updating forecast weather parquet files...")

    # First, build lookup table for observed RH (interpolated to semi-hourly)
    logger.info("Building observed RH lookup table...")
    rh_lookup = {}

    for area, hi_df in heatindex_data.items():
        hi_copy = hi_df.copy()
        # Remove timezone info for matching
        hi_copy["timestamp"] = pd.to_datetime(hi_copy["timestamp"]).dt.tz_localize(None)
        # Interpolate to semi-hourly
        hi_interpolated = interpolate_to_semihourly(hi_copy, "timestamp", ["rh"])
        hi_interpolated = hi_interpolated.set_index("timestamp")
        rh_lookup[area] = hi_interpolated["rh"]

    files_updated = 0
    forecast_dir = data_dir / "weather" / "forecast"

    for area_dir in forecast_dir.iterdir():
        if not area_dir.is_dir():
            continue

        area = area_dir.name.replace("area_code=", "")

        if area in SKIP_AREAS:
            logger.info(f"  Skipping aggregate area: {area}")
            continue

        if area not in rh_lookup:
            logger.warning(f"  No RH data for area: {area}")
            continue

        area_rh = rh_lookup[area]

        for year_dir in area_dir.iterdir():
            if not year_dir.is_dir():
                continue

            year = int(year_dir.name.replace("year=", ""))
            parquet_file = year_dir / "data.parquet"

            if not parquet_file.exists():
                continue

            # Read existing parquet
            df = pd.read_parquet(parquet_file)
            original_len = len(df)

            # Ensure valid_time is datetime (timezone-naive)
            df["valid_time"] = pd.to_datetime(df["valid_time"]).dt.tz_localize(None)

            # Look up RH for each valid_time
            df["rh"] = df["valid_time"].map(lambda t: area_rh.get(t) if t in area_rh.index else np.nan)

            # Calculate heat_index
            df["heat_index"] = calc_heat_index_vec(df["temperature"].values, df["rh"].values)

            # Drop rh column (not stored in forecast)
            df = df.drop(columns=["rh"])

            # Ensure column order
            columns = ["issue_time", "valid_time", "area_code", "temperature", "heat_index"]
            df = df[columns]

            # Write back
            df.to_parquet(parquet_file, index=False)

            non_null = df["heat_index"].notna().sum()
            logger.info(f"    {area}/{year}: {original_len:,} rows, {non_null:,} with heat_index")

            files_updated += 1

    logger.info(f"Updated {files_updated} forecast parquet files")
    return files_updated


def main():
    parser = argparse.ArgumentParser(
        description="Populate heat index in weather parquet files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--heatindex-zip",
        type=Path,
        default=Path("./data/heatindex.zip"),
        help="Path to heatindex.zip file (default: ./data/heatindex.zip)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("./data"),
        help="Data directory with parquet files (default: ./data)",
    )
    parser.add_argument(
        "--observed-only",
        action="store_true",
        help="Only update observed weather data",
    )
    parser.add_argument(
        "--forecast-only",
        action="store_true",
        help="Only update forecast weather data",
    )
    parser.add_argument(
        "--areas",
        type=str,
        default=None,
        help="Comma-separated list of areas to process (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )

    args = parser.parse_args()

    # Validate paths
    if not args.heatindex_zip.exists():
        logger.error(f"Heat index ZIP not found: {args.heatindex_zip}")
        sys.exit(1)

    if not args.data_dir.exists():
        logger.error(f"Data directory not found: {args.data_dir}")
        sys.exit(1)

    # Read heat index data
    heatindex_data = read_heatindex_zip(args.heatindex_zip)

    # Filter areas if specified
    if args.areas:
        areas = [a.strip() for a in args.areas.split(",")]
        heatindex_data = {k: v for k, v in heatindex_data.items() if k in areas}
        logger.info(f"Filtered to {len(heatindex_data)} areas: {list(heatindex_data.keys())}")

    if args.dry_run:
        logger.info("DRY RUN - no files will be modified")
        logger.info(f"Would process {len(heatindex_data)} areas")
        return

    stats = {"observed": 0, "forecast": 0}

    # Update observed data
    if not args.forecast_only:
        stats["observed"] = update_observed_parquet(args.data_dir, heatindex_data)

    # Update forecast data
    if not args.observed_only:
        stats["forecast"] = update_forecast_parquet(args.data_dir, heatindex_data)

    # Summary
    logger.info("=" * 60)
    logger.info("COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Observed files updated: {stats['observed']}")
    logger.info(f"Forecast files updated: {stats['forecast']}")


if __name__ == "__main__":
    main()
