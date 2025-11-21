#!/usr/bin/env python3
"""
Consolidation Script for ML-Ready Data

Consolidates daily raw parquet files into yearly partitioned files optimized for ML training.
This improves read performance by reducing I/O overhead and leveraging Parquet's columnar efficiency.

Input structure (raw_data/):
    raw_data/load/{area}/{YYYY}/{MM}/{YYYYMMDD}.parquet
    raw_data/weather/observed/{area}/{YYYY}/{MM}/{YYYYMMDD}.parquet
    raw_data/weather/forecast/{area}/{YYYY}/{MM}/{YYYYMMDD}.parquet

Output structure (processed/):
    processed/load/area_code={area}/year={YYYY}/data.parquet
    processed/weather/observed/area_code={area}/year={YYYY}/data.parquet
    processed/weather/forecast/area_code={area}/year={YYYY}/data.parquet

Usage:
    python scripts/consolidate_for_ml.py --source ./data --areas SP,RJ
    python scripts/consolidate_for_ml.py --source ./data --all-areas
    python scripts/consolidate_for_ml.py --source ./data --all-areas --years 2023,2024
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def get_available_areas(data_path: Path) -> list[str]:
    """Get list of available areas from the load directory."""
    load_dir = data_path / "raw_data" / "load"
    if not load_dir.exists():
        return []
    return sorted([d.name for d in load_dir.iterdir() if d.is_dir()])


def get_available_years(data_path: Path, area: str, data_type: str = "load") -> list[int]:
    """Get list of available years for an area."""
    if data_type == "load":
        area_dir = data_path / "raw_data" / "load" / area
    elif data_type == "weather_observed":
        area_dir = data_path / "raw_data" / "weather" / "observed" / area
    elif data_type == "weather_forecast":
        area_dir = data_path / "raw_data" / "weather" / "forecast" / area
    else:
        return []

    if not area_dir.exists():
        return []

    years = []
    for item in area_dir.iterdir():
        if item.is_dir() and item.name.isdigit():
            years.append(int(item.name))
    return sorted(years)


def consolidate_load(
    data_path: Path,
    area: str,
    year: int,
    output_path: Optional[Path] = None,
) -> int:
    """
    Consolidate daily load files for an area/year into a single partitioned file.

    Returns number of rows written.
    """
    raw_dir = data_path / "raw_data" / "load" / area / str(year)
    if not raw_dir.exists():
        logger.warning(f"  No load data for {area}/{year}")
        return 0

    # Find all parquet files for this year
    parquet_files = list(raw_dir.glob("**/*.parquet"))
    if not parquet_files:
        logger.warning(f"  No parquet files found for {area}/{year}")
        return 0

    # Read all files
    dfs = []
    for f in sorted(parquet_files):
        try:
            df = pd.read_parquet(f)
            dfs.append(df)
        except Exception as e:
            logger.warning(f"  Error reading {f}: {e}")

    if not dfs:
        return 0

    # Combine all data
    df_combined = pd.concat(dfs, ignore_index=True)

    # Sort by timestamp
    df_combined = df_combined.sort_values("timestamp").reset_index(drop=True)

    # Remove duplicates (keep last)
    df_combined = df_combined.drop_duplicates(subset=["timestamp", "area_code"], keep="last")

    # Write to partitioned output
    output_dir = (output_path or data_path) / "processed" / "load" / f"area_code={area}" / f"year={year}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "data.parquet"

    df_combined.to_parquet(output_file, index=False)

    logger.info(f"  Wrote {len(df_combined):,} rows to {output_file}")
    return len(df_combined)


def consolidate_weather_observed(
    data_path: Path,
    area: str,
    year: int,
    output_path: Optional[Path] = None,
) -> int:
    """
    Consolidate daily weather observed files for an area/year into a single partitioned file.

    Returns number of rows written.
    """
    raw_dir = data_path / "raw_data" / "weather" / "observed" / area / str(year)
    if not raw_dir.exists():
        logger.warning(f"  No weather observed data for {area}/{year}")
        return 0

    # Find all parquet files for this year
    parquet_files = list(raw_dir.glob("**/*.parquet"))
    if not parquet_files:
        logger.warning(f"  No parquet files found for {area}/{year}")
        return 0

    # Read all files
    dfs = []
    for f in sorted(parquet_files):
        try:
            df = pd.read_parquet(f)
            dfs.append(df)
        except Exception as e:
            logger.warning(f"  Error reading {f}: {e}")

    if not dfs:
        return 0

    # Combine all data
    df_combined = pd.concat(dfs, ignore_index=True)

    # Sort by timestamp
    df_combined = df_combined.sort_values("timestamp").reset_index(drop=True)

    # Remove duplicates (keep last)
    df_combined = df_combined.drop_duplicates(subset=["timestamp", "area_code"], keep="last")

    # Write to partitioned output
    output_dir = (output_path or data_path) / "processed" / "weather" / "observed" / f"area_code={area}" / f"year={year}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "data.parquet"

    df_combined.to_parquet(output_file, index=False)

    logger.info(f"  Wrote {len(df_combined):,} rows to {output_file}")
    return len(df_combined)


def consolidate_weather_forecast(
    data_path: Path,
    area: str,
    year: int,
    output_path: Optional[Path] = None,
) -> int:
    """
    Consolidate daily weather forecast files for an area/year into a single partitioned file.

    Returns number of rows written.
    """
    raw_dir = data_path / "raw_data" / "weather" / "forecast" / area / str(year)
    if not raw_dir.exists():
        logger.warning(f"  No weather forecast data for {area}/{year}")
        return 0

    # Find all parquet files for this year
    parquet_files = list(raw_dir.glob("**/*.parquet"))
    if not parquet_files:
        logger.warning(f"  No parquet files found for {area}/{year}")
        return 0

    # Read all files
    dfs = []
    for f in sorted(parquet_files):
        try:
            df = pd.read_parquet(f)
            dfs.append(df)
        except Exception as e:
            logger.warning(f"  Error reading {f}: {e}")

    if not dfs:
        return 0

    # Combine all data
    df_combined = pd.concat(dfs, ignore_index=True)

    # Sort by issue_time, then valid_time
    df_combined = df_combined.sort_values(["issue_time", "valid_time"]).reset_index(drop=True)

    # Remove exact duplicates only - keep all (issue_time, valid_time) pairs
    # Each issue_time represents a different forecast origin, we need to keep them all
    df_combined = df_combined.drop_duplicates(
        subset=["issue_time", "valid_time", "area_code"],
        keep="last"
    )

    # Write to partitioned output
    output_dir = (output_path or data_path) / "processed" / "weather" / "forecast" / f"area_code={area}" / f"year={year}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "data.parquet"

    df_combined.to_parquet(output_file, index=False)

    logger.info(f"  Wrote {len(df_combined):,} rows to {output_file}")
    return len(df_combined)


def main():
    parser = argparse.ArgumentParser(
        description="Consolidate daily files into ML-ready yearly partitions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("./data"),
        help="Data directory containing raw_data/ (default: ./data)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory for processed/ (default: same as source)",
    )
    parser.add_argument(
        "--areas",
        type=str,
        default=None,
        help="Comma-separated list of areas to consolidate (e.g., SP,RJ,SECO)",
    )
    parser.add_argument(
        "--all-areas",
        action="store_true",
        help="Consolidate all available areas",
    )
    parser.add_argument(
        "--years",
        type=str,
        default=None,
        help="Comma-separated list of years to consolidate (e.g., 2023,2024). Default: all years",
    )
    parser.add_argument(
        "--skip-load",
        action="store_true",
        help="Skip load data consolidation",
    )
    parser.add_argument(
        "--skip-weather",
        action="store_true",
        help="Skip weather data consolidation",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be consolidated without writing files",
    )

    args = parser.parse_args()

    # Validate source directory
    if not args.source.exists():
        logger.error(f"Source directory not found: {args.source}")
        sys.exit(1)

    # Determine areas to consolidate
    available_areas = get_available_areas(args.source)
    logger.info(f"Available areas: {available_areas}")

    if args.all_areas:
        areas = available_areas
    elif args.areas:
        areas = [a.strip() for a in args.areas.split(",")]
        # Validate areas
        invalid = set(areas) - set(available_areas)
        if invalid:
            logger.error(f"Invalid areas: {invalid}")
            sys.exit(1)
    else:
        logger.error("Must specify --areas or --all-areas")
        sys.exit(1)

    # Parse years filter
    year_filter = None
    if args.years:
        year_filter = [int(y.strip()) for y in args.years.split(",")]

    logger.info(f"Consolidating areas: {areas}")
    if year_filter:
        logger.info(f"Year filter: {year_filter}")

    if args.dry_run:
        logger.info("DRY RUN - no files will be written")
        return

    # Statistics
    stats = {
        "load_rows": 0,
        "load_files": 0,
        "weather_obs_rows": 0,
        "weather_obs_files": 0,
        "weather_fcst_rows": 0,
        "weather_fcst_files": 0,
    }

    # Process each area
    for area in areas:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing area: {area}")
        logger.info(f"{'='*60}")

        # Get available years for this area
        if not args.skip_load:
            load_years = get_available_years(args.source, area, "load")
            if year_filter:
                load_years = [y for y in load_years if y in year_filter]

            for year in load_years:
                rows = consolidate_load(args.source, area, year, args.output)
                if rows > 0:
                    stats["load_rows"] += rows
                    stats["load_files"] += 1

        if not args.skip_weather:
            # Weather observed
            obs_years = get_available_years(args.source, area, "weather_observed")
            if year_filter:
                obs_years = [y for y in obs_years if y in year_filter]

            for year in obs_years:
                rows = consolidate_weather_observed(args.source, area, year, args.output)
                if rows > 0:
                    stats["weather_obs_rows"] += rows
                    stats["weather_obs_files"] += 1

            # Weather forecast
            fcst_years = get_available_years(args.source, area, "weather_forecast")
            if year_filter:
                fcst_years = [y for y in fcst_years if y in year_filter]

            for year in fcst_years:
                rows = consolidate_weather_forecast(args.source, area, year, args.output)
                if rows > 0:
                    stats["weather_fcst_rows"] += rows
                    stats["weather_fcst_files"] += 1

    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("CONSOLIDATION COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Load data:            {stats['load_files']:,} files, {stats['load_rows']:,} rows")
    logger.info(f"Weather observed:     {stats['weather_obs_files']:,} files, {stats['weather_obs_rows']:,} rows")
    logger.info(f"Weather forecast:     {stats['weather_fcst_files']:,} files, {stats['weather_fcst_rows']:,} rows")
    logger.info(f"\nTotal files: {stats['load_files'] + stats['weather_obs_files'] + stats['weather_fcst_files']:,}")
    logger.info(f"Total rows:  {stats['load_rows'] + stats['weather_obs_rows'] + stats['weather_fcst_rows']:,}")


if __name__ == "__main__":
    main()
