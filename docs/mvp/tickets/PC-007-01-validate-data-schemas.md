# PC-007-01: Validate Data Schemas

**Ticket ID:** PC-007-01  
**Epic:** [Epic-01: Data Infrastructure Layer](../epics/Epic-01.md)  
**User Story:** US-1.2  
**Story Points:** 5  
**Priority:** Critical  
**Assignee:** TBD  
**Status:** 📝 To Do

---

## 📋 Description

Implement comprehensive data validation using Pydantic V2 schemas for all data types (load, temperature, heat index, holidays) with business rules, range checks, and continuity validation to ensure data quality throughout the pipeline.

**As a** system  
**I want to** validate all incoming data against predefined schemas  
**So that** downstream components can assume data quality

---

## ✅ Acceptance Criteria

- [ ] Pydantic schemas defined for:
  - Load (`CargaSchema`): `val_cargaglobalcons`, `val_cargammgd`
  - Temperature (`TemperaturaSchema`): Multiple sources (D+1, D+2, ECMWF, weighted)
  - Heat Index (`HeatIndexSchema`): Apparent temperature values
  - Holidays (`FeriadoSchema`): `id_tipodiaespecial`, `dat_diaespecial`
- [ ] Validation checks include:
  - Data types and required columns
  - Value ranges: load > 0, temperature -10°C to 50°C, heat index 15°C to 55°C
  - Timestamp continuity (30-min or 60-min intervals)
  - Complete day validation (48 semi-hourly or 24 hourly records per day)
- [ ] Business rules validated:
  - Load values non-negative
  - Temperatures within Brazil climate range
  - No duplicate (timestamp, cod_area) pairs
  - Chronological ordering within area
- [ ] Invalid records logged with details but don't stop pipeline
- [ ] Validation summary report generated (counts of valid/invalid records)
- [ ] Performance: <2s validation overhead for 100k records

---

## 🔧 Implementation Tasks

### 1. Define CargaSchema
- [ ] Create `src/data/validators.py`
- [ ] Define `CargaSchema` with Pydantic BaseModel:
  - `timestamp: datetime` (required)
  - `cod_area: str` (pattern validation for area codes)
  - `val_cargaglobalcons: float` (must be > 0)
  - `val_cargammgd: float` (must be >= 0)
- [ ] Add field validators for area code whitelist (26 valid areas)
- [ ] Add validator for load value positivity
- [ ] Add docstrings with field descriptions

### 2. Define TemperaturaSchema
- [ ] Define `TemperaturaSchema`:
  - `timestamp: datetime`
  - `cod_area: str`
  - `temp_celsius: float` (range: -10 to 50)
  - `source: str` (enum: "D+1", "D+2", "ECMWF", "weighted")
- [ ] Add temperature range validator
- [ ] Add source enumeration validator
- [ ] Support multiple temperature formats

### 3. Define HeatIndexSchema
- [ ] Define `HeatIndexSchema`:
  - `timestamp: datetime`
  - `cod_area: str`
  - `heat_index: float` (range: 15 to 55)
- [ ] Add heat index range validator
- [ ] Add relationship validator (heat_index >= temperature)

### 4. Define FeriadoSchema
- [ ] Define `FeriadoSchema`:
  - `date: date`
  - `holiday_name: str`
  - `id_tipodiaespecial: int` (enum: 0-10)
  - `holiday_type: str` (enum: National, Regional, Municipal)
  - `affected_areas: List[str]`
- [ ] Add holiday type enumeration
- [ ] Add special day ID validation
- [ ] Add affected areas validator

### 5. Implement DataValidator Class
- [ ] Create `DataValidator` class with schema parameter
- [ ] Implement `validate()` method for bulk DataFrame validation
- [ ] Add row-by-row validation with error collection
- [ ] Filter out invalid rows (don't stop pipeline)
- [ ] Log validation errors with row details
- [ ] Return validated DataFrame with clean data

### 6. Add Custom Validators
- [ ] Create `TimestampContinuityValidator`:
  - Check 30-min or 60-min intervals
  - Detect gaps and overlaps
  - Group by area for multi-area DataFrames
- [ ] Create `CompleteDayValidator`:
  - Check 48 semi-hourly or 24 hourly records per day
  - Group by area and date
- [ ] Create `DuplicateDetector`:
  - Find duplicate (timestamp, cod_area) pairs
  - Log duplicates with details
- [ ] Create `ChronologicalOrderValidator`:
  - Verify timestamps are sorted within area
  - Detect out-of-order records

### 7. Implement Validation Report Generator
- [ ] Create `ValidationReport` class:
  - Total records processed
  - Valid records count
  - Invalid records count
  - Breakdown by validation rule
  - Invalid row samples (first 10)
- [ ] Generate summary statistics
- [ ] Format report as structured dict/JSON
- [ ] Add logging integration

### 8. Write Comprehensive Tests
- [ ] Create `tests/data/test_validators.py`
- [ ] Test each schema with valid data
- [ ] Test each schema with invalid data:
  - Out of range values
  - Invalid area codes
  - Invalid types
  - Missing required fields
- [ ] Test DataValidator bulk validation
- [ ] Test custom validators (continuity, duplicates, order)
- [ ] Test validation report generation
- [ ] Test performance with 100k records

---

## 💻 Implementation Details

### Pydantic Schemas

```python
"""Data validation schemas using Pydantic V2."""
from datetime import datetime, date
from typing import List, Literal, Optional
from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator

from src.utils.logger import get_logger

logger = get_logger(__name__)


# Valid area codes for Brazil electric system
VALID_AREAS = [
    # States
    "RJ", "SP", "MG", "ES", "MT", "MS", "AC", "RO", "DF", "GO",
    "PR", "SC", "RS", "ALPE", "PBRN", "BASE", "CE", "PI", "BAOE",
    "AM", "PA", "MA", "TO", "RR", "AP",
    # Subsystems
    "SECO", "S", "NE", "N",
    # Losses
    "PESE", "PES", "PENE", "PEN"
]


class CargaSchema(BaseModel):
    """Schema for electric load data validation."""
    
    timestamp: datetime = Field(..., description="Timestamp of measurement (America/Sao_Paulo)")
    cod_area: str = Field(..., pattern=r"^[A-Z]{2,6}$", description="Area code")
    val_cargaglobalcons: float = Field(..., gt=0, description="Global consumption load (MWh)")
    val_cargammgd: Optional[float] = Field(None, ge=0, description="MMGD load (MWh)")
    
    @field_validator("cod_area")
    @classmethod
    def validate_area_code(cls, v: str) -> str:
        """Validate area code is in whitelist."""
        if v not in VALID_AREAS:
            raise ValueError(f"Invalid area code: {v}. Must be one of {VALID_AREAS}")
        return v
    
    class Config:
        str_strip_whitespace = True


class TemperatureSource(str, Enum):
    """Temperature forecast source."""
    D1 = "D+1"
    D2 = "D+2"
    ECMWF = "ECMWF"
    WEIGHTED = "weighted"


class TemperaturaSchema(BaseModel):
    """Schema for temperature forecast data validation."""
    
    timestamp: datetime = Field(..., description="Timestamp of forecast")
    cod_area: str = Field(..., pattern=r"^[A-Z]{2,6}$", description="Area code")
    temp_celsius: float = Field(..., ge=-10, le=50, description="Temperature in Celsius")
    source: TemperatureSource = Field(..., description="Forecast source")
    
    @field_validator("cod_area")
    @classmethod
    def validate_area_code(cls, v: str) -> str:
        """Validate area code."""
        if v not in VALID_AREAS:
            raise ValueError(f"Invalid area code: {v}")
        return v


class HeatIndexSchema(BaseModel):
    """Schema for heat index (apparent temperature) data validation."""
    
    timestamp: datetime = Field(..., description="Timestamp of measurement")
    cod_area: str = Field(..., pattern=r"^[A-Z]{2,6}$", description="Area code")
    heat_index: float = Field(..., ge=15, le=55, description="Heat index in Celsius")
    
    @field_validator("cod_area")
    @classmethod
    def validate_area_code(cls, v: str) -> str:
        """Validate area code."""
        if v not in VALID_AREAS:
            raise ValueError(f"Invalid area code: {v}")
        return v


class HolidayType(str, Enum):
    """Holiday type enumeration."""
    NATIONAL = "National"
    REGIONAL = "Regional"
    MUNICIPAL = "Municipal"


class FeriadoSchema(BaseModel):
    """Schema for holiday calendar data validation."""
    
    date: date = Field(..., description="Holiday date")
    dat_diaespecial: date = Field(..., description="Special day date (same as date)")
    holiday_name: str = Field(..., min_length=1, description="Holiday name")
    id_tipodiaespecial: int = Field(..., ge=0, le=10, description="Special day type ID")
    holiday_type: HolidayType = Field(..., description="Holiday type")
    affected_areas: List[str] = Field(default_factory=list, description="Affected area codes")
    
    @field_validator("affected_areas")
    @classmethod
    def validate_affected_areas(cls, v: List[str]) -> List[str]:
        """Validate all affected areas are valid."""
        for area in v:
            if area not in VALID_AREAS:
                raise ValueError(f"Invalid area in affected_areas: {area}")
        return v
    
    @model_validator(mode="after")
    def validate_dates_match(self) -> "FeriadoSchema":
        """Ensure date and dat_diaespecial match."""
        if self.date != self.dat_diaespecial:
            raise ValueError("date and dat_diaespecial must match")
        return self
```

### DataValidator Class

```python
"""DataFrame validation using Pydantic schemas."""
import pandas as pd
from typing import Type, Dict, Any, List
from pydantic import BaseModel, ValidationError


class ValidationReport:
    """Validation report with statistics."""
    
    def __init__(self) -> None:
        self.total_records: int = 0
        self.valid_records: int = 0
        self.invalid_records: int = 0
        self.errors_by_field: Dict[str, int] = {}
        self.invalid_samples: List[Dict[str, Any]] = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "total_records": self.total_records,
            "valid_records": self.valid_records,
            "invalid_records": self.invalid_records,
            "validation_rate": self.valid_records / self.total_records if self.total_records > 0 else 0,
            "errors_by_field": self.errors_by_field,
            "invalid_samples": self.invalid_samples[:10]  # First 10
        }
    
    def log_summary(self) -> None:
        """Log validation summary."""
        logger.info(
            f"Validation complete: {self.valid_records}/{self.total_records} valid "
            f"({self.validation_rate:.1%}), {self.invalid_records} invalid"
        )
        if self.errors_by_field:
            logger.info(f"Errors by field: {self.errors_by_field}")


class DataValidator:
    """Validate DataFrames against Pydantic schemas."""
    
    def __init__(self, schema: Type[BaseModel]) -> None:
        """
        Initialize validator with schema.
        
        Args:
            schema: Pydantic model class for validation
        """
        self.schema = schema
    
    def validate(self, df: pd.DataFrame, strict: bool = False) -> pd.DataFrame:
        """
        Validate DataFrame rows against schema.
        
        Args:
            df: DataFrame to validate
            strict: If True, raise on any invalid row. If False, filter invalid rows.
        
        Returns:
            Validated DataFrame with invalid rows removed (if not strict)
        
        Raises:
            ValueError: If strict=True and invalid rows found
        """
        report = ValidationReport()
        report.total_records = len(df)
        
        valid_indices = []
        
        for idx, row in df.iterrows():
            try:
                # Validate row
                self.schema(**row.to_dict())
                valid_indices.append(idx)
                report.valid_records += 1
                
            except ValidationError as e:
                report.invalid_records += 1
                
                # Track errors by field
                for error in e.errors():
                    field = error["loc"][0] if error["loc"] else "unknown"
                    report.errors_by_field[field] = report.errors_by_field.get(field, 0) + 1
                
                # Store sample
                if len(report.invalid_samples) < 10:
                    report.invalid_samples.append({
                        "row": idx,
                        "data": row.to_dict(),
                        "errors": [err["msg"] for err in e.errors()]
                    })
                
                # Log warning
                logger.warning(f"Invalid row {idx}: {e}")
        
        # Log summary
        report.log_summary()
        
        # Handle strict mode
        if strict and report.invalid_records > 0:
            raise ValueError(
                f"Validation failed: {report.invalid_records} invalid records. "
                f"Report: {report.to_dict()}"
            )
        
        # Return validated DataFrame
        return df.loc[valid_indices].reset_index(drop=True)
```

### Custom Validators

```python
"""Custom validators for data quality checks."""
import pandas as pd
from typing import Tuple


class TimestampContinuityValidator:
    """Validate timestamp continuity in time series."""
    
    def __init__(self, expected_freq: str = "30T") -> None:
        """
        Initialize continuity validator.
        
        Args:
            expected_freq: Expected frequency ("30T" for 30-min, "1H" for hourly)
        """
        self.expected_freq = expected_freq
        self.expected_delta = pd.Timedelta(expected_freq)
    
    def validate(self, df: pd.DataFrame, timestamp_col: str = "timestamp") -> Tuple[bool, List[str]]:
        """
        Validate timestamp continuity.
        
        Returns:
            (is_valid, error_messages)
        """
        errors = []
        
        if len(df) < 2:
            return True, []
        
        # Sort by timestamp
        df_sorted = df.sort_values(timestamp_col)
        
        # Check intervals
        intervals = df_sorted[timestamp_col].diff()
        irregular = intervals[intervals != self.expected_delta].dropna()
        
        if len(irregular) > 0:
            errors.append(
                f"Found {len(irregular)} irregular intervals "
                f"(expected {self.expected_freq})"
            )
            for idx in irregular.index[:5]:  # First 5
                errors.append(
                    f"  Row {idx}: interval = {intervals[idx]}"
                )
        
        is_valid = len(errors) == 0
        return is_valid, errors


class DuplicateDetector:
    """Detect duplicate records."""
    
    def __init__(self, key_columns: List[str]) -> None:
        """
        Initialize duplicate detector.
        
        Args:
            key_columns: Columns that define uniqueness
        """
        self.key_columns = key_columns
    
    def detect(self, df: pd.DataFrame) -> Tuple[bool, pd.DataFrame]:
        """
        Detect duplicates.
        
        Returns:
            (has_duplicates, duplicate_rows)
        """
        duplicates = df[df.duplicated(subset=self.key_columns, keep=False)]
        
        if len(duplicates) > 0:
            logger.warning(
                f"Found {len(duplicates)} duplicate records "
                f"on columns {self.key_columns}"
            )
        
        return len(duplicates) > 0, duplicates
```

---

## 🧪 Testing & Validation

### Schema Tests

```python
"""Tests for Pydantic schemas."""
import pytest
from datetime import datetime, date
from pydantic import ValidationError

from src.data.validators import CargaSchema, TemperaturaSchema, HeatIndexSchema, FeriadoSchema


def test_carga_schema_valid():
    """Test CargaSchema with valid data."""
    data = {
        "timestamp": datetime(2024, 1, 1, 0, 0),
        "cod_area": "RJ",
        "val_cargaglobalcons": 1500.5,
        "val_cargammgd": 100.0
    }
    schema = CargaSchema(**data)
    assert schema.cod_area == "RJ"
    assert schema.val_cargaglobalcons == 1500.5


def test_carga_schema_invalid_area():
    """Test CargaSchema rejects invalid area code."""
    data = {
        "timestamp": datetime(2024, 1, 1, 0, 0),
        "cod_area": "INVALID",
        "val_cargaglobalcons": 1500.5
    }
    with pytest.raises(ValidationError) as exc_info:
        CargaSchema(**data)
    
    assert "Invalid area code" in str(exc_info.value)


def test_carga_schema_invalid_load():
    """Test CargaSchema rejects non-positive load."""
    data = {
        "timestamp": datetime(2024, 1, 1, 0, 0),
        "cod_area": "RJ",
        "val_cargaglobalcons": -100.0
    }
    with pytest.raises(ValidationError) as exc_info:
        CargaSchema(**data)
    
    assert "greater than 0" in str(exc_info.value)


def test_temperatura_schema_valid():
    """Test TemperaturaSchema with valid data."""
    data = {
        "timestamp": datetime(2024, 1, 1, 0, 0),
        "cod_area": "RJ",
        "temp_celsius": 25.5,
        "source": "D+1"
    }
    schema = TemperaturaSchema(**data)
    assert schema.temp_celsius == 25.5
    assert schema.source == "D+1"


def test_temperatura_schema_out_of_range():
    """Test TemperaturaSchema rejects out of range temperature."""
    data = {
        "timestamp": datetime(2024, 1, 1, 0, 0),
        "cod_area": "RJ",
        "temp_celsius": 60.0,  # Too hot for Brazil
        "source": "D+1"
    }
    with pytest.raises(ValidationError):
        TemperaturaSchema(**data)
```

### DataValidator Tests

```python
"""Tests for DataValidator."""
import pandas as pd
from src.data.validators import DataValidator, CargaSchema


def test_validator_all_valid():
    """Test validator with all valid rows."""
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=3, freq="30T"),
        "cod_area": ["RJ", "RJ", "RJ"],
        "val_cargaglobalcons": [1000.0, 1100.0, 1200.0],
        "val_cargammgd": [100.0, 110.0, 120.0]
    })
    
    validator = DataValidator(CargaSchema)
    result = validator.validate(df)
    
    assert len(result) == 3
    assert list(result.columns) == list(df.columns)


def test_validator_filters_invalid():
    """Test validator filters out invalid rows."""
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=3, freq="30T"),
        "cod_area": ["RJ", "INVALID", "RJ"],
        "val_cargaglobalcons": [1000.0, 1100.0, 1200.0],
        "val_cargammgd": [100.0, 110.0, 120.0]
    })
    
    validator = DataValidator(CargaSchema)
    result = validator.validate(df, strict=False)
    
    assert len(result) == 2  # Invalid row filtered out
    assert all(result["cod_area"] == "RJ")


def test_validator_strict_mode():
    """Test validator raises in strict mode."""
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=3, freq="30T"),
        "cod_area": ["RJ", "INVALID", "RJ"],
        "val_cargaglobalcons": [1000.0, 1100.0, 1200.0]
    })
    
    validator = DataValidator(CargaSchema)
    
    with pytest.raises(ValueError) as exc_info:
        validator.validate(df, strict=True)
    
    assert "Validation failed" in str(exc_info.value)
```

---

## 📝 Technical Notes

- Use Pydantic V2 for better performance and features
- Field validators run before model validators
- Use `@classmethod` for field validators
- Enum classes provide type safety for categorical fields
- ValidationError contains detailed error information
- Non-strict mode allows pipeline to continue with clean data
- Validation overhead should be minimal (<1% of load time)

---

## 🔗 Dependencies

**Depends On:**
- PC-001-00: Repository Structure Setup
- PC-002-00: Python Environment with uv
- PC-006-01: Load Raw Load Data from S3

**Blocks:**
- PC-008-01: Preprocess and Impute Missing Data
- PC-010-01: Filter Complete Days

**Related:**
- Epic-02: Feature Engineering (depends on validated data)

---

## 📊 Definition of Done

- [ ] All acceptance criteria met
- [ ] All 4 Pydantic schemas implemented
- [ ] DataValidator class with bulk validation
- [ ] Custom validators for continuity and duplicates
- [ ] ValidationReport with statistics
- [ ] Unit tests pass with >85% coverage
- [ ] Tests cover valid and invalid data
- [ ] Performance test passes (<2s for 100k records)
- [ ] Code reviewed and approved
- [ ] Documentation complete
- [ ] Integration with PC-006-01 verified
- [ ] Ready for PC-008-01 (preprocessing)

---

## 🔄 Change Log

| Date | Version | Changes | Author |
|------|---------|---------|--------|
| 2025-11-18 | 1.0.0 | Initial ticket created | System |

---

**Next Ticket:** [PC-008-01: Preprocess and Impute Missing Data](PC-008-01-preprocess-impute-data.md)
