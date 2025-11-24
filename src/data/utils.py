"""Utility classes and functions for data loading.

This module provides utility classes used by the data loading components,
including date range generation for iterating over date ranges.

Example:
    ```python
    from datetime import date
    from src.data.utils import DateRangeGenerator

    # Generate dates from Jan 1 to Jan 5
    date_range = DateRangeGenerator(date(2024, 1, 1), date(2024, 1, 5))
    for d in date_range:
        print(d)  # Prints dates from 2024-01-01 to 2024-01-05 inclusive
    ```
"""

from collections.abc import Iterator
from datetime import date, timedelta


class DateRangeGenerator:
    """Generator for iterating over a range of dates.

    This class provides an iterable interface for generating dates
    within a specified range (inclusive of both start and end dates).

    Attributes:
        start_date: The start date of the range (inclusive).
        end_date: The end date of the range (inclusive).

    Example:
        ```python
        from datetime import date
        from src.data.utils import DateRangeGenerator

        # Iterate over a week
        for d in DateRangeGenerator(date(2024, 1, 1), date(2024, 1, 7)):
            print(f"Processing {d}")

        # Convert to list
        dates = list(DateRangeGenerator(date(2024, 1, 1), date(2024, 1, 3)))
        # [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)]
        ```
    """

    def __init__(self, start_date: date, end_date: date) -> None:
        """Initialize the date range generator.

        Args:
            start_date: The start date of the range (inclusive).
            end_date: The end date of the range (inclusive).

        Raises:
            ValueError: If start_date is after end_date.
        """
        if start_date > end_date:
            msg = f"start_date ({start_date}) cannot be after end_date ({end_date})"
            raise ValueError(msg)

        self.start_date = start_date
        self.end_date = end_date

    def __iter__(self) -> Iterator[date]:
        """Iterate over dates in the range.

        Yields:
            Dates from start_date to end_date, inclusive.

        Example:
            ```python
            date_range = DateRangeGenerator(date(2024, 1, 1), date(2024, 1, 3))
            dates = list(date_range)
            # [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)]
            ```
        """
        current = self.start_date
        while current <= self.end_date:
            yield current
            current += timedelta(days=1)

    def __len__(self) -> int:
        """Return the number of dates in the range.

        Returns:
            Number of days in the range (inclusive of both ends).

        Example:
            ```python
            date_range = DateRangeGenerator(date(2024, 1, 1), date(2024, 1, 5))
            len(date_range)  # Returns 5
            ```
        """
        return (self.end_date - self.start_date).days + 1

    def __repr__(self) -> str:
        """Return a string representation of the date range.

        Returns:
            String representation showing start and end dates.
        """
        return f"DateRangeGenerator({self.start_date}, {self.end_date})"
