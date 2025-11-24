"""Data loading, validation, and preprocessing modules.

This module provides the core data loading functionality for the PrevCarga
system, including loaders for raw electric load data from various storage
backends, as well as validation schemas, utilities, and preprocessing tools.

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

    # Impute missing values using triple-pass algorithm
    from src.data import TriplePassImputer
    imputer = TriplePassImputer(date_col="timestamp")
    df = imputer.fit_transform(df, value_col="load")

    # Or chain multiple imputers
    from src.data import ImputerChain, ForwardFillImputer, InterpolationImputer
    chain = ImputerChain([
        ForwardFillImputer(limit=2),
        InterpolationImputer(method="linear"),
    ])
    df = chain.fit_transform(df, value_col="load")
    ```
"""

from src.data.loaders import DataLoader
from src.data.preprocessors import (
    BackwardFillImputer,
    BaseImputer,
    ForwardFillImputer,
    ImputerChain,
    InterpolationImputer,
    LagFillImputer,
    ResamplerMixin,
    TimezoneHandler,
    TriplePassImputer,
)
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
    # Imputers (preprocessors)
    "BackwardFillImputer",
    "BaseImputer",
    "ForwardFillImputer",
    "ImputerChain",
    "InterpolationImputer",
    "LagFillImputer",
    "ResamplerMixin",
    "TimezoneHandler",
    "TriplePassImputer",
    # Loaders
    "DataLoader",
    # Utilities
    "DateRangeGenerator",
    # Validators
    "VALID_AREAS",
    "CargaSchema",
    "DataValidator",
    "DuplicateDetector",
    "FeriadoSchema",
    "HeatIndexSchema",
    "HolidayType",
    "TemperaturaSchema",
    "TemperatureSource",
    "TimestampContinuityValidator",
    "ValidationReport",
]
