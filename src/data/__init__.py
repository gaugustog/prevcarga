"""Data loading, validation, and preprocessing modules.

This module provides the core data loading functionality for the PrevCarga
system, including loaders for raw electric load data from various storage
backends, as well as validation schemas and utilities.

Example:
    ```python
    from datetime import date
    from src.data import DataLoader, DateRangeGenerator, CargaSchema, DataValidator

    # Load electric load data
    loader = DataLoader()
    df = loader.load_carga(
        areas=["SP", "RJ"],
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 31),
    )

    # Generate date ranges
    for d in DateRangeGenerator(date(2024, 1, 1), date(2024, 1, 7)):
        print(f"Processing {d}")

    # Validate data
    validator = DataValidator(CargaSchema)
    validated_df, report = validator.validate(df)
    report.log_summary()
    ```
"""

from src.data.loaders import DataLoader
from src.data.utils import DateRangeGenerator
from src.data.validators import (
    VALID_AREAS,
    CargaSchema,
    DataValidator,
    DuplicateDetector,
    FeriadoSchema,
    HeatIndexSchema,
    HolidayType,
    TemperaturaSchema,
    TemperatureSource,
    TimestampContinuityValidator,
    ValidationReport,
)

__all__ = [
    "VALID_AREAS",
    "CargaSchema",
    "DataLoader",
    "DataValidator",
    "DateRangeGenerator",
    "DuplicateDetector",
    "FeriadoSchema",
    "HeatIndexSchema",
    "HolidayType",
    "TemperaturaSchema",
    "TemperatureSource",
    "TimestampContinuityValidator",
    "ValidationReport",
]
