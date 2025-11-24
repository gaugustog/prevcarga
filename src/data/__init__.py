"""Data loading, validation, and preprocessing modules.

This module provides the core data loading functionality for the PrevCarga
system, including loaders for raw electric load data from various storage
backends.

Example:
    ```python
    from datetime import date
    from src.data import DataLoader, DateRangeGenerator

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
    ```
"""

from src.data.loaders import DataLoader
from src.data.utils import DateRangeGenerator

__all__ = [
    "DataLoader",
    "DateRangeGenerator",
]
