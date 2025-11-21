#!/usr/bin/env python3
"""
Legacy Data Migration Script

Converts legacy data format (old-data-structure/) to new unified format.
All time series data is standardized to semi-hourly (30-minute) resolution.
Output is consolidated by area/year using Hive-style partitioning.

Legacy format (per area):
    {area}/CARGASHHIST.csv.gz    -> data/load/area_code={area}/year={YYYY}/data.parquet
    {area}/TEMPHIST.csv.gz       -> data/weather/observed/area_code={area}/year={YYYY}/data.parquet
    {area}/TEMPPREVHIST.csv.gz   -> data/weather/forecast/area_code={area}/year={YYYY}/data.parquet
    {area}/FERIADOS.csv.gz       -> data/auxiliary/holidays/area_code={area}/year={YYYY}/data.parquet
    {area}/PATAMARES.csv.gz      -> data/auxiliary/load_tiers/area_code={area}/data.parquet
    {area}/HORAVERAO.csv.gz      -> data/auxiliary/dst_periods/data.parquet

Note: CARGAHIST.csv.gz (hourly) is ignored - we use semi-hourly as source of truth.

Usage:
    python scripts/migrate_legacy_data.py --source ./old-data-structure --dest ./data --areas SP,RJ,SECO
    python scripts/migrate_legacy_data.py --source ./old-data-structure --dest ./data --all-areas
"""

import argparse
import gzip
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

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

# Field name mapping: Legacy (Portuguese) -> New (English)
FIELD_MAPPING = {
    # Load fields
    "cod_areacarga": "area_code",
    "dat_referencia": "reference_date",
    "din_atualizacao": "updated_at",
    "din_referencia": "timestamp",
    "val_carga": "load_mwh",
    # Weather fields
    "din_inclusaodl": "ingestion_time",
    "din_origemprevisaoutc": "issue_time",
    "val_tmp": "temperature",
    # Holiday fields
    "dat_diaespecial": "date",
    "id_tipodiaespecial": "holiday_type_id",
    "cod_prevcarga": "prevcarga_code",
    "cod_simplificado": "simplified_code",
    # Model prediction fields
    "cod_associacaoexogenacarga": "exogenous_association_code",
    "cod_modeloprevisaocarga": "model_code",
    "din_referenciautc": "target_time",
    "val_previsaocarga": "predicted_load",
}


def parse_brazilian_number(value: str) -> float:
    """Parse Brazilian number format (comma as decimal separator)."""
    if pd.isna(value) or value == "":
        return float("nan")
    if isinstance(value, (int, float)):
        return float(value)
    return float(str(value).replace(",", "."))


def read_legacy_csv(filepath: Path) -> pd.DataFrame:
    """Read a legacy CSV.gz file with proper encoding and parsing."""
    logger.info(f"Reading {filepath}")

    with gzip.open(filepath, "rt", encoding="utf-8") as f:
        df = pd.read_csv(f, sep=";", low_memory=False)

    logger.info(f"  Loaded {len(df):,} rows, columns: {list(df.columns)}")
    return df


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns from Portuguese to English."""
    rename_map = {col: FIELD_MAPPING.get(col, col) for col in df.columns}
    return df.rename(columns=rename_map)


def convert_numeric_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Convert Brazilian number format to float."""
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = df[col].apply(parse_brazilian_number)
    return df


def convert_datetime_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Convert datetime columns to proper datetime type."""
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def interpolate_to_semihourly(
    df: pd.DataFrame,
    time_col: str,
    value_cols: list[str],
    group_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Interpolate hourly data to semi-hourly using cubic spline.

    Args:
        df: DataFrame with hourly data
        time_col: Name of the timestamp column
        value_cols: List of columns to interpolate
        group_cols: Optional columns to group by before interpolating

    Returns:
        DataFrame with semi-hourly data (original hourly values preserved at :00,
        interpolated values at :30)
    """
    def interpolate_group(group_df: pd.DataFrame) -> pd.DataFrame:
        """Interpolate a single group to semi-hourly."""
        group_df = group_df.sort_values(time_col).copy()

        if len(group_df) < 2:
            # Not enough points for interpolation, just return as-is
            return group_df

        # Get time range
        min_time = group_df[time_col].min()
        max_time = group_df[time_col].max()

        # Create semi-hourly time index
        semihourly_times = pd.date_range(start=min_time, end=max_time, freq="30min")

        # Convert timestamps to numeric for interpolation
        time_numeric = (group_df[time_col] - min_time).dt.total_seconds().values
        new_time_numeric = (semihourly_times - min_time).total_seconds().values

        # Prepare result DataFrame
        result = pd.DataFrame({time_col: semihourly_times})

        # Interpolate each value column
        for col in value_cols:
            if col in group_df.columns:
                values = group_df[col].values

                # Handle NaN values - use linear interpolation to fill gaps first
                valid_mask = ~np.isnan(values)
                if valid_mask.sum() < 2:
                    # Not enough valid points
                    result[col] = np.nan
                    continue

                valid_times = time_numeric[valid_mask]
                valid_values = values[valid_mask]

                try:
                    # Use cubic spline interpolation
                    cs = CubicSpline(valid_times, valid_values, extrapolate=False)
                    result[col] = cs(new_time_numeric)
                except Exception as e:
                    logger.warning(f"Cubic spline failed for {col}, using linear: {e}")
                    # Fallback to linear interpolation
                    result[col] = np.interp(new_time_numeric, valid_times, valid_values)

        return result

    if group_cols:
        # Apply interpolation per group
        results = []
        for group_keys, group_df in df.groupby(group_cols):
            interpolated = interpolate_group(group_df)

            # Add group columns back
            if isinstance(group_keys, tuple):
                for col, val in zip(group_cols, group_keys):
                    interpolated[col] = val
            else:
                interpolated[group_cols[0]] = group_keys

            results.append(interpolated)

        return pd.concat(results, ignore_index=True)
    else:
        return interpolate_group(df)


def migrate_load_data(
    source_path: Path,
    dest_path: Path,
    area: str,
) -> int:
    """
    Migrate semi-hourly load data (CARGASHHIST) to new format.
    Consolidates data by area/year using Hive-style partitioning.

    Returns number of files written.
    """
    source_file = source_path / area / "CARGASHHIST.csv.gz"

    if not source_file.exists():
        logger.warning(f"  CARGASHHIST.csv.gz not found for {area}")
        return 0

    # Read legacy data
    df = read_legacy_csv(source_file)

    # Rename columns
    df = rename_columns(df)

    # Convert numeric columns
    df = convert_numeric_columns(df, ["load_mwh"])

    # Convert datetime columns
    df = convert_datetime_columns(df, ["timestamp", "updated_at", "reference_date"])

    # Sort by timestamp and remove duplicates
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp", "area_code"], keep="last")

    # Extract year for partitioning
    df["_year"] = df["timestamp"].dt.year

    # Select final columns
    output_columns = ["timestamp", "area_code", "load_mwh"]

    # Write partitioned by area/year
    files_written = 0
    for year, group in df.groupby("_year"):
        output_dir = dest_path / "load" / f"area_code={area}" / f"year={year}"
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / "data.parquet"
        group[output_columns].reset_index(drop=True).to_parquet(output_file, index=False)
        files_written += 1

    logger.info(f"  Wrote {files_written} load files for {area}")
    return files_written


def migrate_weather_observed(
    source_path: Path,
    dest_path: Path,
    area: str,
) -> int:
    """
    Migrate observed weather data (TEMPHIST) to new format.
    Interpolates hourly data to semi-hourly using cubic spline.
    Consolidates data by area/year using Hive-style partitioning.

    Interpolates across day boundaries to ensure 23:30 values are computed
    using both 23:00 of current day and 00:00 of next day.

    Returns number of files written.
    """
    source_file = source_path / area / "TEMPHIST.csv.gz"

    if not source_file.exists():
        logger.warning(f"  TEMPHIST.csv.gz not found for {area}")
        return 0

    # Read legacy data
    df = read_legacy_csv(source_file)

    # Rename columns
    df = rename_columns(df)

    # Convert numeric columns
    df = convert_numeric_columns(df, ["temperature"])

    # Convert datetime columns
    df = convert_datetime_columns(df, ["timestamp", "ingestion_time", "reference_date"])

    # Add heat_index as None (not in legacy data)
    df["heat_index"] = np.nan

    # Sort by timestamp for continuous interpolation
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)

    # Interpolate entire series at once (not grouped by date)
    # This allows 23:30 to be interpolated using 23:00 and next day's 00:00
    logger.info(f"  Interpolating to semi-hourly (across day boundaries)...")
    df_interpolated = interpolate_to_semihourly(
        df,
        time_col="timestamp",
        value_cols=["temperature", "heat_index"],
        group_cols=None,  # No grouping - interpolate entire series
    )

    # Add area_code back
    df_interpolated["area_code"] = area

    # Extract year for partitioning
    df_interpolated["_year"] = df_interpolated["timestamp"].dt.year

    # Select final columns
    output_columns = ["timestamp", "area_code", "temperature", "heat_index"]

    # Write partitioned by area/year
    files_written = 0
    for year, group in df_interpolated.groupby("_year"):
        output_dir = dest_path / "weather" / "observed" / f"area_code={area}" / f"year={year}"
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / "data.parquet"
        group[output_columns].reset_index(drop=True).to_parquet(output_file, index=False)
        files_written += 1

    logger.info(f"  Wrote {files_written} weather observed files for {area}")
    return files_written


def migrate_weather_forecast(
    source_path: Path,
    dest_path: Path,
    area: str,
) -> int:
    """
    Migrate forecast weather data (TEMPPREVHIST) to new format.
    Interpolates hourly data to semi-hourly using cubic spline.
    Consolidates data by area/year using Hive-style partitioning.

    Interpolates across day boundaries within each issue_time to ensure
    23:30 values are computed using both 23:00 and next day's 00:00.

    Note: We use reference_date (dat_referencia, local date) as issue_time instead of
    din_origemprevisaoutc (UTC date) to avoid negative forecast horizons caused by
    timezone conversion issues.

    Returns number of files written.
    """
    source_file = source_path / area / "TEMPPREVHIST.csv.gz"

    if not source_file.exists():
        logger.warning(f"  TEMPPREVHIST.csv.gz not found for {area}")
        return 0

    # Read legacy data
    df = read_legacy_csv(source_file)

    # Rename columns
    df = rename_columns(df)

    # Convert numeric columns
    df = convert_numeric_columns(df, ["temperature"])

    # Convert datetime columns
    df = convert_datetime_columns(df, ["timestamp", "issue_time", "ingestion_time", "reference_date"])

    # Rename timestamp to valid_time for forecasts
    df = df.rename(columns={"timestamp": "valid_time"})

    # IMPORTANT: Use reference_date (local date) as issue_time instead of din_origemprevisaoutc (UTC)
    # This avoids negative forecast horizons caused by timezone conversion
    # The original issue_time (din_origemprevisaoutc) is UTC date only, while reference_date
    # (dat_referencia) is the local date when the forecast was made
    df["issue_time"] = pd.to_datetime(df["reference_date"].dt.date)

    # Add heat_index as None (not in legacy data)
    df["heat_index"] = np.nan

    # Interpolate to semi-hourly per issue_time group
    # (each issue_time represents a separate forecast, interpolate across all valid_times)
    logger.info(f"  Interpolating to semi-hourly (across day boundaries per issue_time)...")

    results = []
    issue_times = df["issue_time"].unique()
    total_issue_times = len(issue_times)

    for i, issue_time in enumerate(issue_times):
        if (i + 1) % 500 == 0:
            logger.info(f"    Processing issue_time {i + 1}/{total_issue_times}...")

        issue_group = df[df["issue_time"] == issue_time].copy()
        issue_group = issue_group.sort_values("valid_time").drop_duplicates(subset=["valid_time"])

        if len(issue_group) < 2:
            # Not enough points, keep as-is
            results.append(issue_group)
            continue

        # Interpolate entire forecast series (not grouped by date)
        interpolated = interpolate_to_semihourly(
            issue_group,
            time_col="valid_time",
            value_cols=["temperature", "heat_index"],
            group_cols=None,  # No grouping - interpolate entire series
        )
        interpolated["issue_time"] = issue_time
        interpolated["area_code"] = area
        results.append(interpolated)

    if not results:
        logger.warning(f"  No data to interpolate for {area}")
        return 0

    df_interpolated = pd.concat(results, ignore_index=True)

    # Sort by issue_time, valid_time and remove exact duplicates
    df_interpolated = df_interpolated.sort_values(["issue_time", "valid_time"])
    df_interpolated = df_interpolated.drop_duplicates(
        subset=["issue_time", "valid_time", "area_code"], keep="last"
    )

    # Extract year for partitioning (by valid_time)
    df_interpolated["_year"] = df_interpolated["valid_time"].dt.year

    # Select final columns
    output_columns = ["issue_time", "valid_time", "area_code", "temperature", "heat_index"]

    # Write partitioned by area/year
    files_written = 0
    for year, group in df_interpolated.groupby("_year"):
        output_dir = dest_path / "weather" / "forecast" / f"area_code={area}" / f"year={year}"
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / "data.parquet"
        group[output_columns].reset_index(drop=True).to_parquet(output_file, index=False)
        files_written += 1

    logger.info(f"  Wrote {files_written} weather forecast files for {area}")
    return files_written


def migrate_holidays(
    source_path: Path,
    dest_path: Path,
    area: str,
) -> int:
    """
    Migrate holidays data (FERIADOS) to new format per area.
    Consolidates data by area/year using Hive-style partitioning.

    Returns number of files written.
    """
    source_file = source_path / area / "FERIADOS.csv.gz"

    if not source_file.exists():
        logger.warning(f"  FERIADOS.csv.gz not found for {area}")
        return 0

    df = read_legacy_csv(source_file)
    df = rename_columns(df)

    # Convert date column
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date

    # Remove duplicates (same date)
    df = df.drop_duplicates(subset=["date"])

    # Extract year for partitioning
    df["_year"] = pd.to_datetime(df["date"]).dt.year

    # Select final columns
    output_columns = ["date", "area_code", "holiday_type_id", "prevcarga_code", "simplified_code"]

    # Write per area/year files
    files_written = 0
    for year, group in df.groupby("_year"):
        output_dir = dest_path / "auxiliary" / "holidays" / f"area_code={area}" / f"year={year}"
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / "data.parquet"
        group[output_columns].reset_index(drop=True).to_parquet(output_file, index=False)
        files_written += 1

    logger.info(f"  Wrote {files_written} holiday files for {area}")
    return files_written


def migrate_load_tiers(
    source_path: Path,
    dest_path: Path,
    area: str,
) -> int:
    """
    Migrate load tiers (PATAMARES) to new format.
    Uses Hive-style partitioning by area.

    Returns 1 if file written, 0 otherwise.
    """
    source_file = source_path / area / "PATAMARES.csv.gz"

    if not source_file.exists():
        logger.warning(f"  PATAMARES.csv.gz not found for {area}")
        return 0

    # Read legacy data
    df = read_legacy_csv(source_file)

    # Rename columns (keep structure as-is, just translate)
    column_mapping = {
        "Hora": "hour",
        "dia.util.inverno": "weekday_winter",
        "fds.inverno": "weekend_winter",
        "dia.util.intermediario": "weekday_intermediate",
        "fds.intermediario": "weekend_intermediate",
        "dia.util.verao": "weekday_summer",
        "fds.verao": "weekend_summer",
    }
    df = df.rename(columns=column_mapping)

    # Add area_code
    df["area_code"] = area

    # Write output with Hive-style partitioning
    output_dir = dest_path / "auxiliary" / "load_tiers" / f"area_code={area}"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "data.parquet"
    df.to_parquet(output_file, index=False)

    logger.info(f"  Wrote load_tiers for {area}")
    return 1


def migrate_dst_periods(
    source_path: Path,
    dest_path: Path,
    areas: list[str],
) -> int:
    """
    Migrate DST periods (HORAVERAO) to new format.

    Returns 1 if file written, 0 otherwise.
    """
    # DST periods are the same for all areas, just read from first available
    df = None
    for area in areas:
        source_file = source_path / area / "HORAVERAO.csv.gz"
        if source_file.exists():
            df = read_legacy_csv(source_file)
            break

    if df is None:
        logger.warning("No HORAVERAO.csv.gz found in any area")
        return 0

    # Rename columns
    column_mapping = {
        "Data.inicial": "start_date",
        "Data.final": "end_date",
    }
    df = df.rename(columns=column_mapping)

    # Parse dates (Brazilian format dd/mm/yyyy)
    df["start_date"] = pd.to_datetime(df["start_date"], format="%d/%m/%Y", errors="coerce")
    df["end_date"] = pd.to_datetime(df["end_date"], format="%d/%m/%Y", errors="coerce")

    # Write output
    output_dir = dest_path / "auxiliary" / "dst_periods"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "data.parquet"
    df.to_parquet(output_file, index=False)

    logger.info(f"  Wrote dst_periods with {len(df)} periods")
    return 1


def get_available_areas(source_path: Path) -> list[str]:
    """Get list of available areas from source directory."""
    areas = []
    for item in source_path.iterdir():
        if item.is_dir() and item.name not in ["source", ".git"]:
            areas.append(item.name)
    return sorted(areas)


def main():
    parser = argparse.ArgumentParser(
        description="Migrate legacy data to new unified format (semi-hourly)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("./old-data-structure"),
        help="Source directory with legacy data (default: ./old-data-structure)",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path("./data"),
        help="Destination directory for new format (default: ./data)",
    )
    parser.add_argument(
        "--areas",
        type=str,
        default=None,
        help="Comma-separated list of areas to migrate (e.g., SP,RJ,SECO)",
    )
    parser.add_argument(
        "--all-areas",
        action="store_true",
        help="Migrate all available areas",
    )
    parser.add_argument(
        "--skip-load",
        action="store_true",
        help="Skip load data migration",
    )
    parser.add_argument(
        "--skip-weather",
        action="store_true",
        help="Skip weather data migration",
    )
    parser.add_argument(
        "--skip-auxiliary",
        action="store_true",
        help="Skip auxiliary data migration (holidays, load_tiers, dst)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be migrated without writing files",
    )

    args = parser.parse_args()

    # Validate source directory
    if not args.source.exists():
        logger.error(f"Source directory not found: {args.source}")
        sys.exit(1)

    # Determine areas to migrate
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

    logger.info(f"Migrating areas: {areas}")

    if args.dry_run:
        logger.info("DRY RUN - no files will be written")
        return

    # Create destination directory
    args.dest.mkdir(parents=True, exist_ok=True)

    # Statistics
    stats = {
        "load": 0,
        "weather_observed": 0,
        "weather_forecast": 0,
        "holidays": 0,
        "load_tiers": 0,
        "dst_periods": 0,
    }

    # Migrate data for each area
    for area in areas:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing area: {area}")
        logger.info(f"{'='*60}")

        if not args.skip_load:
            stats["load"] += migrate_load_data(args.source, args.dest, area)

        if not args.skip_weather:
            stats["weather_observed"] += migrate_weather_observed(args.source, args.dest, area)
            stats["weather_forecast"] += migrate_weather_forecast(args.source, args.dest, area)

        if not args.skip_auxiliary:
            stats["holidays"] += migrate_holidays(args.source, args.dest, area)
            stats["load_tiers"] += migrate_load_tiers(args.source, args.dest, area)

    # Migrate shared auxiliary data (DST periods are same for all areas)
    if not args.skip_auxiliary:
        logger.info(f"\n{'='*60}")
        logger.info("Processing shared auxiliary data")
        logger.info(f"{'='*60}")

        stats["dst_periods"] = migrate_dst_periods(args.source, args.dest, areas)

    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("MIGRATION COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Load (semi-hourly) files: {stats['load']:,}")
    logger.info(f"Weather observed files:   {stats['weather_observed']:,}")
    logger.info(f"Weather forecast files:   {stats['weather_forecast']:,}")
    logger.info(f"Holiday files:            {stats['holidays']:,}")
    logger.info(f"Load tier files:          {stats['load_tiers']:,}")
    logger.info(f"DST period files:         {stats['dst_periods']:,}")
    logger.info(f"\nTotal files written: {sum(stats.values()):,}")


if __name__ == "__main__":
    main()
