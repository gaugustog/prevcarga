"""Tests for data validation schemas and validators.

This module contains comprehensive tests for the data validation functionality
in the PrevCarga system, including Pydantic schemas, validation reports,
and utility validators for timestamp continuity and duplicate detection.
"""

from datetime import date, datetime

import pandas as pd
import pytest
from pydantic import ValidationError

from src.data.validators import (
    VALID_AREAS,
    CargaSchema,
    DataValidator,
    DuplicateDetector,
    FeriadoSchema,
    HeatIndexSchema,
    HolidayType,
    TemperatureSource,
    TemperaturaSchema,
    TimestampContinuityValidator,
    ValidationReport,
)

# ruff: noqa: I001


# =============================================================================
# CargaSchema Tests
# =============================================================================


class TestCargaSchema:
    """Tests for CargaSchema validation."""

    def test_valid_carga_record(self):
        """Test that a valid carga record passes validation."""
        record = CargaSchema(
            timestamp=datetime(2024, 1, 1, 12, 0),
            cod_area="SP",
            val_cargaglobalcons=1500.5,
            val_cargammgd=25.0,
        )

        assert record.timestamp == datetime(2024, 1, 1, 12, 0)
        assert record.cod_area == "SP"
        assert record.val_cargaglobalcons == 1500.5
        assert record.val_cargammgd == 25.0

    def test_valid_carga_without_optional_field(self):
        """Test that val_cargammgd is optional."""
        record = CargaSchema(
            timestamp=datetime(2024, 1, 1, 12, 0),
            cod_area="RJ",
            val_cargaglobalcons=1000.0,
        )

        assert record.val_cargammgd is None

    def test_invalid_area_code(self):
        """Test that invalid area code raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            CargaSchema(
                timestamp=datetime(2024, 1, 1, 12, 0),
                cod_area="INVALID",
                val_cargaglobalcons=1000.0,
            )

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("cod_area",)
        assert "Invalid area code" in errors[0]["msg"]

    def test_invalid_load_negative(self):
        """Test that negative load value raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            CargaSchema(
                timestamp=datetime(2024, 1, 1, 12, 0),
                cod_area="SP",
                val_cargaglobalcons=-100.0,
            )

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("val_cargaglobalcons",)

    def test_invalid_load_zero(self):
        """Test that zero load value raises ValidationError (gt=0)."""
        with pytest.raises(ValidationError) as exc_info:
            CargaSchema(
                timestamp=datetime(2024, 1, 1, 12, 0),
                cod_area="SP",
                val_cargaglobalcons=0.0,
            )

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("val_cargaglobalcons",) for e in errors)

    def test_invalid_mmgd_negative(self):
        """Test that negative MMGD value raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            CargaSchema(
                timestamp=datetime(2024, 1, 1, 12, 0),
                cod_area="SP",
                val_cargaglobalcons=100.0,
                val_cargammgd=-5.0,
            )

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("val_cargammgd",) for e in errors)

    def test_valid_mmgd_zero(self):
        """Test that zero MMGD value is valid (ge=0)."""
        record = CargaSchema(
            timestamp=datetime(2024, 1, 1, 12, 0),
            cod_area="SP",
            val_cargaglobalcons=100.0,
            val_cargammgd=0.0,
        )

        assert record.val_cargammgd == 0.0

    def test_all_valid_areas(self):
        """Test that all VALID_AREAS are accepted."""
        for area in VALID_AREAS:
            record = CargaSchema(
                timestamp=datetime(2024, 1, 1, 12, 0),
                cod_area=area,
                val_cargaglobalcons=100.0,
            )
            assert record.cod_area == area

    def test_whitespace_stripping(self):
        """Test that whitespace is stripped from string fields."""
        record = CargaSchema(
            timestamp=datetime(2024, 1, 1, 12, 0),
            cod_area="  SP  ",
            val_cargaglobalcons=100.0,
        )
        assert record.cod_area == "SP"


# =============================================================================
# TemperaturaSchema Tests
# =============================================================================


class TestTemperaturaSchema:
    """Tests for TemperaturaSchema validation."""

    def test_valid_temperature_record(self):
        """Test that a valid temperature record passes validation."""
        record = TemperaturaSchema(
            timestamp=datetime(2024, 1, 1, 12, 0),
            cod_area="SP",
            temp_celsius=28.5,
            source=TemperatureSource.ECMWF,
        )

        assert record.timestamp == datetime(2024, 1, 1, 12, 0)
        assert record.cod_area == "SP"
        assert record.temp_celsius == 28.5
        assert record.source == TemperatureSource.ECMWF

    def test_temperature_at_lower_bound(self):
        """Test temperature at lower bound (-10)."""
        record = TemperaturaSchema(
            timestamp=datetime(2024, 1, 1, 12, 0),
            cod_area="RS",
            temp_celsius=-10.0,
            source=TemperatureSource.D_PLUS_1,
        )

        assert record.temp_celsius == -10.0

    def test_temperature_at_upper_bound(self):
        """Test temperature at upper bound (50)."""
        record = TemperaturaSchema(
            timestamp=datetime(2024, 1, 1, 12, 0),
            cod_area="AM",
            temp_celsius=50.0,
            source=TemperatureSource.D_PLUS_2,
        )

        assert record.temp_celsius == 50.0

    def test_temperature_out_of_range_low(self):
        """Test that temperature below -10 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            TemperaturaSchema(
                timestamp=datetime(2024, 1, 1, 12, 0),
                cod_area="SP",
                temp_celsius=-11.0,
                source=TemperatureSource.ECMWF,
            )

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("temp_celsius",) for e in errors)

    def test_temperature_out_of_range_high(self):
        """Test that temperature above 50 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            TemperaturaSchema(
                timestamp=datetime(2024, 1, 1, 12, 0),
                cod_area="SP",
                temp_celsius=51.0,
                source=TemperatureSource.ECMWF,
            )

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("temp_celsius",) for e in errors)

    def test_all_temperature_sources(self):
        """Test that all temperature sources are valid."""
        for source in TemperatureSource:
            record = TemperaturaSchema(
                timestamp=datetime(2024, 1, 1, 12, 0),
                cod_area="SP",
                temp_celsius=25.0,
                source=source,
            )
            assert record.source == source

    def test_invalid_temperature_source(self):
        """Test that invalid source raises ValidationError."""
        with pytest.raises(ValidationError):
            TemperaturaSchema(
                timestamp=datetime(2024, 1, 1, 12, 0),
                cod_area="SP",
                temp_celsius=25.0,
                source="INVALID",  # type: ignore[arg-type]
            )


# =============================================================================
# HeatIndexSchema Tests
# =============================================================================


class TestHeatIndexSchema:
    """Tests for HeatIndexSchema validation."""

    def test_valid_heat_index_record(self):
        """Test that a valid heat index record passes validation."""
        record = HeatIndexSchema(
            timestamp=datetime(2024, 1, 1, 12, 0),
            cod_area="SP",
            heat_index=32.5,
        )

        assert record.timestamp == datetime(2024, 1, 1, 12, 0)
        assert record.cod_area == "SP"
        assert record.heat_index == 32.5

    def test_heat_index_at_lower_bound(self):
        """Test heat index at lower bound (15)."""
        record = HeatIndexSchema(
            timestamp=datetime(2024, 1, 1, 12, 0),
            cod_area="RS",
            heat_index=15.0,
        )

        assert record.heat_index == 15.0

    def test_heat_index_at_upper_bound(self):
        """Test heat index at upper bound (55)."""
        record = HeatIndexSchema(
            timestamp=datetime(2024, 1, 1, 12, 0),
            cod_area="AM",
            heat_index=55.0,
        )

        assert record.heat_index == 55.0

    def test_heat_index_out_of_range_low(self):
        """Test that heat index below 15 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            HeatIndexSchema(
                timestamp=datetime(2024, 1, 1, 12, 0),
                cod_area="SP",
                heat_index=14.9,
            )

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("heat_index",) for e in errors)

    def test_heat_index_out_of_range_high(self):
        """Test that heat index above 55 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            HeatIndexSchema(
                timestamp=datetime(2024, 1, 1, 12, 0),
                cod_area="SP",
                heat_index=55.1,
            )

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("heat_index",) for e in errors)


# =============================================================================
# FeriadoSchema Tests
# =============================================================================


class TestFeriadoSchema:
    """Tests for FeriadoSchema validation."""

    def test_valid_feriado_record(self):
        """Test that a valid feriado record passes validation."""
        record = FeriadoSchema(
            date=date(2024, 1, 1),
            holiday_name="Ano Novo",
            id_tipodiaespecial=1,
            holiday_type=HolidayType.NATIONAL,
            affected_areas=["SP", "RJ", "MG"],
        )

        assert record.date == date(2024, 1, 1)
        assert record.holiday_name == "Ano Novo"
        assert record.id_tipodiaespecial == 1
        assert record.holiday_type == HolidayType.NATIONAL
        assert record.affected_areas == ["SP", "RJ", "MG"]

    def test_invalid_affected_area(self):
        """Test that invalid affected area raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            FeriadoSchema(
                date=date(2024, 1, 1),
                holiday_name="Test Holiday",
                id_tipodiaespecial=1,
                holiday_type=HolidayType.REGIONAL,
                affected_areas=["SP", "INVALID_AREA"],
            )

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("affected_areas",)
        assert "Invalid area codes" in errors[0]["msg"]

    def test_all_holiday_types(self):
        """Test that all holiday types are valid."""
        for holiday_type in HolidayType:
            record = FeriadoSchema(
                date=date(2024, 1, 1),
                holiday_name="Test Holiday",
                id_tipodiaespecial=1,
                holiday_type=holiday_type,
                affected_areas=["SP"],
            )
            assert record.holiday_type == holiday_type

    def test_id_tipodiaespecial_bounds(self):
        """Test id_tipodiaespecial boundaries (0-10)."""
        # Valid at lower bound
        record = FeriadoSchema(
            date=date(2024, 1, 1),
            holiday_name="Test",
            id_tipodiaespecial=0,
            holiday_type=HolidayType.NATIONAL,
            affected_areas=["SP"],
        )
        assert record.id_tipodiaespecial == 0

        # Valid at upper bound
        record = FeriadoSchema(
            date=date(2024, 1, 1),
            holiday_name="Test",
            id_tipodiaespecial=10,
            holiday_type=HolidayType.NATIONAL,
            affected_areas=["SP"],
        )
        assert record.id_tipodiaespecial == 10

    def test_id_tipodiaespecial_out_of_range(self):
        """Test that id_tipodiaespecial outside 0-10 raises ValidationError."""
        with pytest.raises(ValidationError):
            FeriadoSchema(
                date=date(2024, 1, 1),
                holiday_name="Test",
                id_tipodiaespecial=11,
                holiday_type=HolidayType.NATIONAL,
                affected_areas=["SP"],
            )

        with pytest.raises(ValidationError):
            FeriadoSchema(
                date=date(2024, 1, 1),
                holiday_name="Test",
                id_tipodiaespecial=-1,
                holiday_type=HolidayType.NATIONAL,
                affected_areas=["SP"],
            )

    def test_empty_affected_areas(self):
        """Test that empty affected_areas list is valid."""
        record = FeriadoSchema(
            date=date(2024, 1, 1),
            holiday_name="Test",
            id_tipodiaespecial=1,
            holiday_type=HolidayType.NATIONAL,
            affected_areas=[],
        )
        assert record.affected_areas == []


# =============================================================================
# ValidationReport Tests
# =============================================================================


class TestValidationReport:
    """Tests for ValidationReport class."""

    def test_to_dict(self):
        """Test that to_dict returns correct dictionary representation."""
        report = ValidationReport()
        report.total_records = 100
        report.valid_records = 95
        report.invalid_records = 5
        report.errors_by_field = {"cod_area": 3, "val_cargaglobalcons": 2}
        report.invalid_samples = [{"index": 0, "error": "test error"}]

        result = report.to_dict()

        assert result["total_records"] == 100
        assert result["valid_records"] == 95
        assert result["invalid_records"] == 5
        assert result["errors_by_field"] == {"cod_area": 3, "val_cargaglobalcons": 2}
        assert len(result["invalid_samples"]) == 1

    def test_to_dict_empty_report(self):
        """Test to_dict with empty report."""
        report = ValidationReport()
        result = report.to_dict()

        assert result["total_records"] == 0
        assert result["valid_records"] == 0
        assert result["invalid_records"] == 0
        assert result["errors_by_field"] == {}
        assert result["invalid_samples"] == []

    def test_log_summary(self):
        """Test that log_summary executes without error."""
        report = ValidationReport()
        report.total_records = 100
        report.valid_records = 90
        report.invalid_records = 10
        report.errors_by_field = {"cod_area": 5}
        report.invalid_samples = [{"index": 0, "error": "test"}]

        # log_summary should execute without raising an exception
        report.log_summary()

        # Verify the report state is unchanged
        assert report.total_records == 100
        assert report.valid_records == 90
        assert report.invalid_records == 10

    def test_log_summary_no_errors(self):
        """Test log_summary with no errors executes without error."""
        report = ValidationReport()
        report.total_records = 100
        report.valid_records = 100
        report.invalid_records = 0

        # log_summary should execute without raising an exception
        report.log_summary()

        # Verify the report state is unchanged
        assert report.total_records == 100
        assert report.valid_records == 100
        assert report.invalid_records == 0


# =============================================================================
# DataValidator Tests
# =============================================================================


class TestDataValidator:
    """Tests for DataValidator class."""

    @pytest.fixture
    def valid_carga_df(self) -> pd.DataFrame:
        """Create a valid carga DataFrame for testing."""
        return pd.DataFrame({
            "timestamp": pd.to_datetime([
                "2024-01-01 00:00:00",
                "2024-01-01 00:30:00",
                "2024-01-01 01:00:00",
            ]),
            "cod_area": ["SP", "SP", "SP"],
            "val_cargaglobalcons": [100.0, 110.0, 120.0],
            "val_cargammgd": [10.0, 11.0, 12.0],
        })

    @pytest.fixture
    def mixed_carga_df(self) -> pd.DataFrame:
        """Create a DataFrame with mixed valid and invalid records."""
        return pd.DataFrame({
            "timestamp": pd.to_datetime([
                "2024-01-01 00:00:00",
                "2024-01-01 00:30:00",
                "2024-01-01 01:00:00",
            ]),
            "cod_area": ["SP", "INVALID", "RJ"],
            "val_cargaglobalcons": [100.0, 110.0, -50.0],
            "val_cargammgd": [10.0, 11.0, 12.0],
        })

    def test_validate_all_valid(self, valid_carga_df: pd.DataFrame):
        """Test validation with all valid records."""
        validator = DataValidator(CargaSchema)
        validated_df, report = validator.validate(valid_carga_df)

        assert len(validated_df) == 3
        assert report.total_records == 3
        assert report.valid_records == 3
        assert report.invalid_records == 0
        assert len(report.invalid_samples) == 0

    def test_validate_filters_invalid(self, mixed_carga_df: pd.DataFrame):
        """Test that invalid records are filtered in non-strict mode."""
        validator = DataValidator(CargaSchema)
        validated_df, report = validator.validate(mixed_carga_df, strict=False)

        assert len(validated_df) == 1  # Only first record is valid
        assert report.total_records == 3
        assert report.valid_records == 1
        assert report.invalid_records == 2
        assert len(report.invalid_samples) == 2

    def test_validate_strict_mode_raises(self, mixed_carga_df: pd.DataFrame):
        """Test that strict mode raises ValueError on first invalid record."""
        validator = DataValidator(CargaSchema)

        with pytest.raises(ValueError) as exc_info:
            validator.validate(mixed_carga_df, strict=True)

        assert "Validation failed" in str(exc_info.value)

    def test_validate_empty_dataframe(self):
        """Test validation with empty DataFrame."""
        validator = DataValidator(CargaSchema)
        empty_df = pd.DataFrame(columns=[
            "timestamp", "cod_area", "val_cargaglobalcons", "val_cargammgd"
        ])

        validated_df, report = validator.validate(empty_df)

        assert len(validated_df) == 0
        assert report.total_records == 0
        assert report.valid_records == 0
        assert report.invalid_records == 0

    def test_validate_collects_errors_by_field(self, mixed_carga_df: pd.DataFrame):
        """Test that errors are collected by field."""
        validator = DataValidator(CargaSchema)
        _validated_df, report = validator.validate(mixed_carga_df)

        # Should have errors for cod_area (INVALID) and val_cargaglobalcons (-50)
        assert len(report.errors_by_field) > 0

    def test_validate_limits_invalid_samples(self):
        """Test that invalid samples are limited to max_samples."""
        # Create DataFrame with many invalid records
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(["2024-01-01"] * 20),
            "cod_area": ["INVALID"] * 20,
            "val_cargaglobalcons": [100.0] * 20,
        })

        validator = DataValidator(CargaSchema)
        _validated_df, report = validator.validate(df)

        # Should be limited to 10 samples
        assert len(report.invalid_samples) <= 10


# =============================================================================
# TimestampContinuityValidator Tests
# =============================================================================


class TestTimestampContinuityValidator:
    """Tests for TimestampContinuityValidator class."""

    def test_valid_continuous_series(self):
        """Test validation with continuous timestamp series."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="30min"),
            "cod_area": ["SP"] * 10,
        })

        validator = TimestampContinuityValidator(expected_freq="30min")
        is_continuous, gaps = validator.validate(df)

        assert is_continuous is True
        assert len(gaps) == 0

    def test_detects_gaps(self):
        """Test that gaps are detected in time series."""
        # Create series with a gap (missing 01:00)
        timestamps = [
            "2024-01-01 00:00:00",
            "2024-01-01 00:30:00",
            # Missing 01:00:00
            "2024-01-01 01:30:00",
            "2024-01-01 02:00:00",
        ]
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(timestamps),
            "cod_area": ["SP"] * 4,
        })

        validator = TimestampContinuityValidator(expected_freq="30min")
        is_continuous, gaps = validator.validate(df)

        assert is_continuous is False
        assert len(gaps) == 1
        assert "missing timestamp" in gaps[0]

    def test_validates_by_group(self):
        """Test validation by group column."""
        df = pd.DataFrame({
            "timestamp": pd.to_datetime([
                "2024-01-01 00:00:00", "2024-01-01 00:30:00",  # SP complete
                "2024-01-01 00:00:00", "2024-01-01 01:00:00",  # RJ has gap
            ]),
            "cod_area": ["SP", "SP", "RJ", "RJ"],
        })

        validator = TimestampContinuityValidator(expected_freq="30min")
        is_continuous, gaps = validator.validate(df, group_col="cod_area")

        assert is_continuous is False
        # RJ should have a gap
        rj_gaps = [g for g in gaps if "'RJ'" in g]
        assert len(rj_gaps) > 0

    def test_empty_dataframe(self):
        """Test validation with empty DataFrame."""
        df = pd.DataFrame(columns=["timestamp", "cod_area"])

        validator = TimestampContinuityValidator(expected_freq="30min")
        is_continuous, gaps = validator.validate(df)

        assert is_continuous is True
        assert len(gaps) == 0

    def test_single_record(self):
        """Test validation with single record (no gaps possible)."""
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(["2024-01-01 00:00:00"]),
            "cod_area": ["SP"],
        })

        validator = TimestampContinuityValidator(expected_freq="30min")
        is_continuous, gaps = validator.validate(df)

        assert is_continuous is True
        assert len(gaps) == 0

    def test_custom_timestamp_column(self):
        """Test validation with custom timestamp column name."""
        df = pd.DataFrame({
            "datetime": pd.date_range("2024-01-01", periods=5, freq="1h"),
            "area": ["SP"] * 5,
        })

        validator = TimestampContinuityValidator(expected_freq="1h")
        is_continuous, gaps = validator.validate(
            df, timestamp_col="datetime", group_col="area"
        )

        assert is_continuous is True
        assert len(gaps) == 0

    def test_no_group_column(self):
        """Test validation without grouping."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="30min"),
            "cod_area": ["SP"] * 5,
        })

        validator = TimestampContinuityValidator(expected_freq="30min")
        is_continuous, gaps = validator.validate(df, group_col=None)

        assert is_continuous is True
        assert len(gaps) == 0


# =============================================================================
# DuplicateDetector Tests
# =============================================================================


class TestDuplicateDetector:
    """Tests for DuplicateDetector class."""

    def test_no_duplicates(self):
        """Test detection with no duplicates."""
        df = pd.DataFrame({
            "timestamp": pd.to_datetime([
                "2024-01-01 00:00:00",
                "2024-01-01 00:30:00",
                "2024-01-01 01:00:00",
            ]),
            "cod_area": ["SP", "SP", "SP"],
            "value": [100, 110, 120],
        })

        detector = DuplicateDetector(key_columns=["timestamp", "cod_area"])
        has_duplicates, duplicate_df = detector.detect(df)

        assert has_duplicates is False
        assert len(duplicate_df) == 0

    def test_finds_duplicates(self):
        """Test detection finds duplicate rows."""
        df = pd.DataFrame({
            "timestamp": pd.to_datetime([
                "2024-01-01 00:00:00",
                "2024-01-01 00:00:00",  # Duplicate
                "2024-01-01 01:00:00",
            ]),
            "cod_area": ["SP", "SP", "SP"],
            "value": [100, 100, 120],
        })

        detector = DuplicateDetector(key_columns=["timestamp", "cod_area"])
        has_duplicates, duplicate_df = detector.detect(df)

        assert has_duplicates is True
        assert len(duplicate_df) == 2  # Both duplicate rows returned

    def test_multiple_duplicates(self):
        """Test detection with multiple sets of duplicates."""
        df = pd.DataFrame({
            "timestamp": pd.to_datetime([
                "2024-01-01 00:00:00",
                "2024-01-01 00:00:00",
                "2024-01-01 00:00:00",  # Three duplicates
                "2024-01-01 01:00:00",
                "2024-01-01 01:00:00",  # Two duplicates
            ]),
            "cod_area": ["SP", "SP", "SP", "RJ", "RJ"],
            "value": [100, 100, 100, 200, 200],
        })

        detector = DuplicateDetector(key_columns=["timestamp", "cod_area"])
        has_duplicates, duplicate_df = detector.detect(df)

        assert has_duplicates is True
        assert len(duplicate_df) == 5  # All duplicate rows

    def test_empty_dataframe(self):
        """Test detection with empty DataFrame."""
        df = pd.DataFrame(columns=["timestamp", "cod_area", "value"])

        detector = DuplicateDetector(key_columns=["timestamp", "cod_area"])
        has_duplicates, duplicate_df = detector.detect(df)

        assert has_duplicates is False
        assert len(duplicate_df) == 0

    def test_single_key_column(self):
        """Test detection with single key column."""
        df = pd.DataFrame({
            "id": [1, 2, 2, 3],
            "value": [100, 200, 200, 300],
        })

        detector = DuplicateDetector(key_columns=["id"])
        has_duplicates, duplicate_df = detector.detect(df)

        assert has_duplicates is True
        assert len(duplicate_df) == 2

    def test_missing_key_column(self):
        """Test detection when key column doesn't exist."""
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(["2024-01-01"]),
            "value": [100],
        })

        detector = DuplicateDetector(key_columns=["nonexistent_col"])
        has_duplicates, duplicate_df = detector.detect(df)

        assert has_duplicates is False
        assert len(duplicate_df) == 0

    def test_partial_key_columns_exist(self):
        """Test detection when only some key columns exist."""
        df = pd.DataFrame({
            "timestamp": pd.to_datetime([
                "2024-01-01 00:00:00",
                "2024-01-01 00:00:00",
            ]),
            "value": [100, 100],
        })

        detector = DuplicateDetector(key_columns=["timestamp", "nonexistent"])
        has_duplicates, duplicate_df = detector.detect(df)

        # Should use only existing columns
        assert has_duplicates is True
        assert len(duplicate_df) == 2


# =============================================================================
# VALID_AREAS Tests
# =============================================================================


class TestValidAreas:
    """Tests for VALID_AREAS constant."""

    def test_valid_areas_not_empty(self):
        """Test that VALID_AREAS is not empty."""
        assert len(VALID_AREAS) > 0

    def test_valid_areas_contains_subsystems(self):
        """Test that VALID_AREAS contains subsystem codes."""
        subsystems = ["SECO", "S", "NE", "N"]
        for subsystem in subsystems:
            assert subsystem in VALID_AREAS

    def test_valid_areas_contains_major_states(self):
        """Test that VALID_AREAS contains major state codes."""
        major_states = ["SP", "RJ", "MG", "RS", "PR"]
        for state in major_states:
            assert state in VALID_AREAS

    def test_valid_areas_contains_loss_components(self):
        """Test that VALID_AREAS contains loss component codes."""
        loss_components = ["PESE", "PES", "PENE", "PEN"]
        for component in loss_components:
            assert component in VALID_AREAS


# =============================================================================
# Enum Tests
# =============================================================================


class TestEnums:
    """Tests for enum types."""

    def test_temperature_source_values(self):
        """Test TemperatureSource enum values."""
        assert TemperatureSource.D_PLUS_1.value == "D+1"
        assert TemperatureSource.D_PLUS_2.value == "D+2"
        assert TemperatureSource.ECMWF.value == "ECMWF"
        assert TemperatureSource.WEIGHTED.value == "weighted"

    def test_holiday_type_values(self):
        """Test HolidayType enum values."""
        assert HolidayType.NATIONAL.value == "National"
        assert HolidayType.REGIONAL.value == "Regional"
        assert HolidayType.MUNICIPAL.value == "Municipal"
