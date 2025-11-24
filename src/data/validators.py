"""Data validation schemas and validators for PrevCarga.

This module provides Pydantic schemas for validating data records,
along with utility classes for data validation workflows including
timestamp continuity checks and duplicate detection.

Example:
    ```python
    import pandas as pd
    from src.data.validators import CargaSchema, DataValidator

    # Validate a DataFrame of carga records
    validator = DataValidator(CargaSchema)
    df = pd.DataFrame({
        "timestamp": ["2024-01-01 00:00:00"],
        "cod_area": ["SP"],
        "val_cargaglobalcons": [100.0],
        "val_cargammgd": [10.0],
    })
    validated_df, report = validator.validate(df)
    report.log_summary()
    ```
"""

from datetime import date, datetime
from enum import Enum
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.utils import get_logger

logger = get_logger(__name__)

# Valid area codes for the Brazilian electric system
VALID_AREAS = [
    "RJ", "SP", "MG", "ES", "MT", "MS", "AC", "RO", "DF", "GO",
    "PR", "SC", "RS", "ALPE", "PBRN", "BASE", "CE", "PI", "BAOE",
    "AM", "PA", "MA", "TO", "RR", "AP",
    "SECO", "S", "NE", "N",
    "PESE", "PES", "PENE", "PEN"
]


class TemperatureSource(str, Enum):
    """Source of temperature data.

    Attributes:
        D_PLUS_1: Day-ahead forecast (D+1).
        D_PLUS_2: Two-day ahead forecast (D+2).
        ECMWF: European Centre for Medium-Range Weather Forecasts.
        WEIGHTED: Weighted combination of sources.
    """

    D_PLUS_1 = "D+1"
    D_PLUS_2 = "D+2"
    ECMWF = "ECMWF"
    WEIGHTED = "weighted"


class HolidayType(str, Enum):
    """Type of holiday classification.

    Attributes:
        NATIONAL: National holiday affecting all areas.
        REGIONAL: Regional holiday affecting a state or region.
        MUNICIPAL: Municipal holiday affecting a specific city.
    """

    NATIONAL = "National"
    REGIONAL = "Regional"
    MUNICIPAL = "Municipal"


class CargaSchema(BaseModel):
    """Schema for electric load (carga) data records.

    Validates individual records of electric load data from the
    Brazilian National Interconnected System (SIN).

    Attributes:
        timestamp: Date and time of the measurement.
        cod_area: Area code (must be in VALID_AREAS).
        val_cargaglobalcons: Global consolidated load value in MWh (must be > 0).
        val_cargammgd: Micro and mini distributed generation value (must be >= 0).

    Example:
        ```python
        from datetime import datetime
        from src.data.validators import CargaSchema

        record = CargaSchema(
            timestamp=datetime(2024, 1, 1, 12, 0),
            cod_area="SP",
            val_cargaglobalcons=1500.5,
            val_cargammgd=25.0,
        )
        ```
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    timestamp: datetime
    cod_area: str
    val_cargaglobalcons: float = Field(gt=0)
    val_cargammgd: float | None = Field(default=None, ge=0)

    @field_validator("cod_area")
    @classmethod
    def validate_area_code(cls, v: str) -> str:
        """Validate that area code is in the allowed list.

        Args:
            v: The area code to validate.

        Returns:
            The validated area code.

        Raises:
            ValueError: If area code is not in VALID_AREAS.
        """
        if v not in VALID_AREAS:
            msg = f"Invalid area code '{v}'. Must be one of: {', '.join(VALID_AREAS)}"
            raise ValueError(msg)
        return v


class TemperaturaSchema(BaseModel):
    """Schema for temperature data records.

    Validates temperature measurements used for load forecasting.

    Attributes:
        timestamp: Date and time of the temperature measurement.
        cod_area: Area code for the temperature reading.
        temp_celsius: Temperature in Celsius (must be between -10 and 50).
        source: Source of the temperature data.

    Example:
        ```python
        from datetime import datetime
        from src.data.validators import TemperaturaSchema, TemperatureSource

        record = TemperaturaSchema(
            timestamp=datetime(2024, 1, 1, 12, 0),
            cod_area="SP",
            temp_celsius=28.5,
            source=TemperatureSource.ECMWF,
        )
        ```
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    timestamp: datetime
    cod_area: str
    temp_celsius: float = Field(ge=-10, le=50)
    source: TemperatureSource


class HeatIndexSchema(BaseModel):
    """Schema for heat index data records.

    Validates heat index values which combine temperature and humidity effects.

    Attributes:
        timestamp: Date and time of the heat index measurement.
        cod_area: Area code for the heat index reading.
        heat_index: Heat index value (must be between 15 and 55).

    Example:
        ```python
        from datetime import datetime
        from src.data.validators import HeatIndexSchema

        record = HeatIndexSchema(
            timestamp=datetime(2024, 1, 1, 12, 0),
            cod_area="SP",
            heat_index=32.5,
        )
        ```
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    timestamp: datetime
    cod_area: str
    heat_index: float = Field(ge=15, le=55)


class FeriadoSchema(BaseModel):
    """Schema for holiday data records.

    Validates holiday information used for load forecasting adjustments.

    Attributes:
        date: Date of the holiday.
        holiday_name: Name of the holiday.
        id_tipodiaespecial: Special day type identifier (0-10).
        holiday_type: Type of holiday (National, Regional, Municipal).
        affected_areas: List of area codes affected by the holiday.

    Example:
        ```python
        from datetime import date
        from src.data.validators import FeriadoSchema, HolidayType

        record = FeriadoSchema(
            date=date(2024, 1, 1),
            holiday_name="Ano Novo",
            id_tipodiaespecial=1,
            holiday_type=HolidayType.NATIONAL,
            affected_areas=["SP", "RJ", "MG"],
        )
        ```
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    date: date
    holiday_name: str
    id_tipodiaespecial: int = Field(ge=0, le=10)
    holiday_type: HolidayType
    affected_areas: list[str]

    @field_validator("affected_areas")
    @classmethod
    def validate_affected_areas(cls, v: list[str]) -> list[str]:
        """Validate that all affected areas are valid area codes.

        Args:
            v: List of area codes to validate.

        Returns:
            The validated list of area codes.

        Raises:
            ValueError: If any area code is not in VALID_AREAS.
        """
        invalid_areas = [area for area in v if area not in VALID_AREAS]
        if invalid_areas:
            msg = (
                f"Invalid area codes: {invalid_areas}. "
                f"Must be one of: {', '.join(VALID_AREAS)}"
            )
            raise ValueError(msg)
        return v


class ValidationReport:
    """Report containing validation results and statistics.

    Collects and summarizes validation outcomes including counts of
    valid/invalid records, error distributions, and sample invalid records.

    Attributes:
        total_records: Total number of records validated.
        valid_records: Number of records that passed validation.
        invalid_records: Number of records that failed validation.
        errors_by_field: Count of errors grouped by field name.
        invalid_samples: Sample of invalid records for debugging.

    Example:
        ```python
        from src.data.validators import ValidationReport

        report = ValidationReport()
        report.total_records = 1000
        report.valid_records = 990
        report.invalid_records = 10
        report.errors_by_field = {"cod_area": 5, "val_cargaglobalcons": 5}
        report.log_summary()
        ```
    """

    def __init__(self) -> None:
        """Initialize an empty validation report."""
        self.total_records: int = 0
        self.valid_records: int = 0
        self.invalid_records: int = 0
        self.errors_by_field: dict[str, int] = {}
        self.invalid_samples: list[dict[str, Any]] = []

    def to_dict(self) -> dict[str, Any]:
        """Convert the report to a dictionary representation.

        Returns:
            Dictionary containing all report attributes.

        Example:
            ```python
            report = ValidationReport()
            report.total_records = 100
            report_dict = report.to_dict()
            print(report_dict["total_records"])  # 100
            ```
        """
        return {
            "total_records": self.total_records,
            "valid_records": self.valid_records,
            "invalid_records": self.invalid_records,
            "errors_by_field": self.errors_by_field,
            "invalid_samples": self.invalid_samples,
        }

    def log_summary(self) -> None:
        """Log a summary of the validation report.

        Outputs validation statistics to the logger at INFO level,
        including total, valid, and invalid record counts, plus
        error distribution by field.

        Example:
            ```python
            report = ValidationReport()
            report.total_records = 1000
            report.valid_records = 990
            report.invalid_records = 10
            report.log_summary()
            # Logs: "Validation complete: 1000 total, 990 valid, 10 invalid"
            ```
        """
        logger.info(
            "Validation complete: %d total, %d valid, %d invalid",
            self.total_records,
            self.valid_records,
            self.invalid_records,
        )
        if self.errors_by_field:
            logger.info("Errors by field: %s", self.errors_by_field)
        if self.invalid_samples:
            logger.debug("Sample invalid records: %s", self.invalid_samples[:5])


class DataValidator:
    """Validator for DataFrame records using Pydantic schemas.

    Validates each row of a DataFrame against a specified Pydantic schema,
    collecting validation results and optionally filtering out invalid rows.

    Attributes:
        schema: The Pydantic BaseModel class used for validation.

    Example:
        ```python
        import pandas as pd
        from src.data.validators import CargaSchema, DataValidator

        validator = DataValidator(CargaSchema)
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(["2024-01-01 00:00:00"]),
            "cod_area": ["SP"],
            "val_cargaglobalcons": [100.0],
            "val_cargammgd": [10.0],
        })

        # Non-strict mode: filters invalid rows
        validated_df, report = validator.validate(df)

        # Strict mode: raises on first invalid row
        validated_df, report = validator.validate(df, strict=True)
        ```
    """

    def __init__(self, schema: type[BaseModel]) -> None:
        """Initialize the validator with a schema.

        Args:
            schema: Pydantic BaseModel class to validate records against.
        """
        self.schema = schema

    def validate(
        self,
        df: pd.DataFrame,
        strict: bool = False,
    ) -> tuple[pd.DataFrame, ValidationReport]:
        """Validate all rows in a DataFrame against the schema.

        Iterates through each row, validating against the configured schema.
        In non-strict mode, invalid rows are filtered out. In strict mode,
        validation stops at the first error and raises an exception.

        Args:
            df: DataFrame to validate.
            strict: If True, raise ValidationError on first invalid row.
                If False (default), filter out invalid rows and continue.

        Returns:
            Tuple of (validated_df, report) where validated_df contains
            only valid rows (in non-strict mode) and report contains
            validation statistics.

        Raises:
            ValueError: In strict mode, if any row fails validation.

        Example:
            ```python
            validator = DataValidator(CargaSchema)
            validated_df, report = validator.validate(df, strict=False)
            print(f"Filtered {report.invalid_records} invalid rows")
            ```
        """
        report = ValidationReport()
        report.total_records = len(df)

        valid_indices: list[int] = []
        max_samples = 10  # Maximum number of invalid samples to collect

        for idx, row in df.iterrows():
            row_dict = row.to_dict()
            try:
                self.schema.model_validate(row_dict)
                valid_indices.append(idx)  # type: ignore[arg-type]
            except Exception as e:
                report.invalid_records += 1

                # Extract field names from validation error
                error_str = str(e)
                self._update_errors_by_field(report, error_str)

                # Collect sample of invalid records
                if len(report.invalid_samples) < max_samples:
                    report.invalid_samples.append({
                        "index": idx,
                        "data": row_dict,
                        "error": error_str,
                    })

                if strict:
                    msg = f"Validation failed at row {idx}: {error_str}"
                    raise ValueError(msg) from e

        report.valid_records = len(valid_indices)

        # Filter to only valid rows
        validated_df = (
            df.loc[valid_indices].copy() if valid_indices else df.iloc[:0].copy()
        )

        logger.debug(
            "Validated %d rows: %d valid, %d invalid",
            report.total_records,
            report.valid_records,
            report.invalid_records,
        )

        return validated_df, report

    def _update_errors_by_field(
        self,
        report: ValidationReport,
        error_str: str,
    ) -> None:
        """Update errors_by_field from validation error string.

        Parses the error string to extract field names and increment
        their error counts in the report.

        Args:
            report: ValidationReport to update.
            error_str: Error string from Pydantic validation.
        """
        # Extract field names from Pydantic v2 error format
        # Errors typically contain field names in the message
        for field in self.schema.model_fields:
            if field in error_str:
                report.errors_by_field[field] = (
                    report.errors_by_field.get(field, 0) + 1
                )


class TimestampContinuityValidator:
    """Validator for checking timestamp continuity in time series data.

    Detects gaps in time series data where expected timestamps are missing,
    grouped by an optional grouping column (e.g., area code).

    Attributes:
        expected_freq: Expected frequency of timestamps (e.g., "30min", "1h").

    Example:
        ```python
        import pandas as pd
        from src.data.validators import TimestampContinuityValidator

        validator = TimestampContinuityValidator(expected_freq="30min")
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="30min"),
            "cod_area": ["SP"] * 10,
        })

        is_continuous, gaps = validator.validate(df)
        if not is_continuous:
            print(f"Found gaps: {gaps}")
        ```
    """

    def __init__(self, expected_freq: str = "30min") -> None:
        """Initialize the validator with expected timestamp frequency.

        Args:
            expected_freq: Expected frequency string (e.g., "30min", "1h", "1D").
                Uses pandas frequency string format.
        """
        self.expected_freq = expected_freq

    def validate(
        self,
        df: pd.DataFrame,
        timestamp_col: str = "timestamp",
        group_col: str | None = "cod_area",
    ) -> tuple[bool, list[str]]:
        """Check for gaps in timestamp continuity.

        Validates that timestamps form a continuous series at the expected
        frequency, optionally checking each group separately.

        Args:
            df: DataFrame containing timestamp data.
            timestamp_col: Name of the timestamp column.
            group_col: Column to group by (e.g., "cod_area"). If None,
                validates the entire DataFrame as one series.

        Returns:
            Tuple of (is_continuous, gaps) where is_continuous is True
            if no gaps were found, and gaps is a list of gap descriptions.

        Example:
            ```python
            validator = TimestampContinuityValidator(expected_freq="1h")
            is_continuous, gaps = validator.validate(
                df,
                timestamp_col="timestamp",
                group_col="cod_area"
            )
            ```
        """
        if df.empty:
            return True, []

        gaps: list[str] = []

        if group_col is not None and group_col in df.columns:
            # Validate each group separately
            for group_name, group_df in df.groupby(group_col):
                group_gaps = self._check_group_continuity(
                    group_df, timestamp_col, str(group_name)
                )
                gaps.extend(group_gaps)
        else:
            # Validate entire DataFrame as one series
            gaps = self._check_group_continuity(df, timestamp_col, "all")

        is_continuous = len(gaps) == 0

        if not is_continuous:
            logger.warning("Found %d timestamp gaps", len(gaps))

        return is_continuous, gaps

    def _check_group_continuity(
        self,
        df: pd.DataFrame,
        timestamp_col: str,
        group_name: str,
    ) -> list[str]:
        """Check timestamp continuity for a single group.

        Args:
            df: DataFrame for a single group.
            timestamp_col: Name of the timestamp column.
            group_name: Name of the group for gap descriptions.

        Returns:
            List of gap descriptions for this group.
        """
        gaps: list[str] = []

        min_records_for_gap_check = 2
        if len(df) < min_records_for_gap_check:
            return gaps

        # Sort by timestamp and get expected vs actual
        sorted_df = df.sort_values(timestamp_col)
        timestamps = pd.to_datetime(sorted_df[timestamp_col])

        # Generate expected range
        expected_range = pd.date_range(
            start=timestamps.min(),
            end=timestamps.max(),
            freq=self.expected_freq,
        )

        # Find missing timestamps
        actual_set = set(timestamps)
        expected_set = set(expected_range)
        missing = expected_set - actual_set

        for missing_ts in sorted(missing):
            gaps.append(
                f"Group '{group_name}': missing timestamp {missing_ts}"
            )

        return gaps


class DuplicateDetector:
    """Detector for duplicate records based on key columns.

    Identifies duplicate rows in a DataFrame based on specified
    key columns that should uniquely identify each record.

    Attributes:
        key_columns: List of column names that form the unique key.

    Example:
        ```python
        import pandas as pd
        from src.data.validators import DuplicateDetector

        detector = DuplicateDetector(key_columns=["timestamp", "cod_area"])
        df = pd.DataFrame({
            "timestamp": ["2024-01-01", "2024-01-01", "2024-01-02"],
            "cod_area": ["SP", "SP", "SP"],
            "value": [100, 100, 200],
        })

        has_duplicates, duplicate_df = detector.detect(df)
        if has_duplicates:
            print(f"Found {len(duplicate_df)} duplicate rows")
        ```
    """

    def __init__(self, key_columns: list[str]) -> None:
        """Initialize the detector with key columns.

        Args:
            key_columns: List of column names that should uniquely
                identify each record.
        """
        self.key_columns = key_columns

    def detect(self, df: pd.DataFrame) -> tuple[bool, pd.DataFrame]:
        """Detect duplicate rows based on key columns.

        Args:
            df: DataFrame to check for duplicates.

        Returns:
            Tuple of (has_duplicates, duplicate_df) where has_duplicates
            is True if duplicates were found, and duplicate_df contains
            all rows that are duplicates (including first occurrences).

        Example:
            ```python
            detector = DuplicateDetector(["timestamp", "cod_area"])
            has_duplicates, duplicates = detector.detect(df)
            if has_duplicates:
                print(duplicates)
            ```
        """
        if df.empty:
            return False, df.iloc[:0].copy()

        # Check which columns exist
        existing_cols = [col for col in self.key_columns if col in df.columns]

        if not existing_cols:
            logger.warning(
                "None of the key columns %s exist in DataFrame",
                self.key_columns,
            )
            return False, df.iloc[:0].copy()

        # Find duplicates - keep='False' marks all duplicates (including first)
        duplicate_mask = df.duplicated(subset=existing_cols, keep=False)
        duplicate_df = df[duplicate_mask].copy()

        has_duplicates = len(duplicate_df) > 0

        if has_duplicates:
            logger.info(
                "Found %d duplicate rows based on columns %s",
                len(duplicate_df),
                existing_cols,
            )

        return has_duplicates, duplicate_df
