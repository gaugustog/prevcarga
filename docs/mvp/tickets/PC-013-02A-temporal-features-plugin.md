# PC-013-02A: Temporal Features Plugin

**Ticket ID:** PC-013-02A  
**Epic:** [Epic-02A: Core Feature Engineering](../epics/Epic-02A.md)  
**User Story:** US-2  
**Story Points:** 5  
**Priority:** High  
**Assignee:** TBD  
**Status:** 📝 To Do

---

## 📋 Description

Implement a plugin for generating temporal features including hour, day of week, month, quarter, season, and year. Features should properly handle Brazilian timezone and include cyclical encoding for periodic features.

**As a** data scientist  
**I want** temporal features extracted from datetime index  
**So that** the model can learn daily, weekly, and seasonal patterns

---

## ✅ Acceptance Criteria

- [ ] `TemporalFeaturesPlugin` extracts hour, day, month, quarter, season, year
- [ ] Timezone handling for America/Sao_Paulo
- [ ] Cyclical encoding (sin/cos) for hour, day_of_week, month
- [ ] Business day indicator (is_business_day)
- [ ] Weekend indicator (is_weekend)
- [ ] Season determination for Southern Hemisphere
- [ ] Configurable feature selection
- [ ] Unit tests with timezone edge cases
- [ ] Performance: <100ms for 1 year of hourly data

---

## 🔧 Implementation Tasks

### 1. Create Plugin Module
- [ ] Create `src/features/plugins/temporal.py`
- [ ] Import BaseFeaturePlugin and dependencies
- [ ] Add module docstring with feature descriptions

### 2. Implement TemporalFeaturesPlugin Class
- [ ] Inherit from `BaseFeaturePlugin`
- [ ] Implement `name` property returning "temporal_features"
- [ ] Implement `version` property with semantic versioning
- [ ] Add class docstring with usage examples

### 3. Implement Basic Temporal Features
- [ ] Extract hour (0-23)
- [ ] Extract day of week (0=Monday to 6=Sunday)
- [ ] Extract day of month (1-31)
- [ ] Extract month (1-12)
- [ ] Extract quarter (1-4)
- [ ] Extract year
- [ ] Add is_weekend flag (Saturday/Sunday)
- [ ] Add is_business_day flag (Monday-Friday)

### 4. Implement Season Features
- [ ] Create `_get_season()` helper method
- [ ] Map months to Southern Hemisphere seasons:
  - Summer: Dec, Jan, Feb (12, 1, 2)
  - Autumn: Mar, Apr, May (3, 4, 5)
  - Winter: Jun, Jul, Aug (6, 7, 8)
  - Spring: Sep, Oct, Nov (9, 10, 11)
- [ ] Return numeric season codes (1-4)
- [ ] Add season name as optional feature

### 5. Implement Cyclical Encoding
- [ ] Create `_cyclical_encode()` helper method
- [ ] Add sin/cos encoding for hour (24-hour cycle)
- [ ] Add sin/cos encoding for day_of_week (7-day cycle)
- [ ] Add sin/cos encoding for month (12-month cycle)
- [ ] Preserve original features alongside encoded versions
- [ ] Formula: sin(2π × value / max_value), cos(2π × value / max_value)

### 6. Implement Configuration Schema
- [ ] Create `TemporalFeaturesConfig` Pydantic model
- [ ] Add `include_cyclical` boolean flag (default: True)
- [ ] Add `include_season` boolean flag (default: True)
- [ ] Add `timezone` string field (default: "America/Sao_Paulo")
- [ ] Add `features` list for selective feature generation
- [ ] Validate timezone string against pytz

### 7. Implement generate_features() Method
- [ ] Validate input DataFrame has DatetimeIndex
- [ ] Convert timezone if needed
- [ ] Generate all requested features
- [ ] Apply cyclical encoding if configured
- [ ] Return DataFrame with same index as input
- [ ] Add error handling for invalid inputs

### 8. Implement get_feature_names() Method
- [ ] Return list based on configuration
- [ ] Include base features (hour, day_of_week, etc.)
- [ ] Include cyclical features if enabled
- [ ] Include season features if enabled
- [ ] Maintain consistent naming convention

### 9. Write Comprehensive Tests
- [ ] Create `tests/features/plugins/test_temporal.py`
- [ ] Test all basic temporal features
- [ ] Test Southern Hemisphere seasons
- [ ] Test cyclical encoding correctness
- [ ] Test timezone conversion
- [ ] Test configuration validation
- [ ] Test feature name generation
- [ ] Test edge cases (leap years, DST transitions)
- [ ] Test performance with large datasets

### 10. Create Usage Examples
- [ ] Add example in docstring
- [ ] Create `examples/temporal_features_demo.py`
- [ ] Document configuration options
- [ ] Show visualization of cyclical features

---

## 💻 Implementation Details

### TemporalFeaturesPlugin Implementation

```python
"""Temporal features plugin for time-based feature extraction."""
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator
import pytz

from src.features.base.plugin import BaseFeaturePlugin
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TemporalFeaturesConfig(BaseModel):
    """Configuration for temporal features plugin."""
    
    include_cyclical: bool = Field(
        default=True,
        description="Include sin/cos cyclical encoding"
    )
    include_season: bool = Field(
        default=True,
        description="Include season features"
    )
    timezone: str = Field(
        default="America/Sao_Paulo",
        description="Timezone for datetime localization"
    )
    features: List[str] = Field(
        default_factory=lambda: [
            "hour", "day_of_week", "day_of_month",
            "month", "quarter", "year",
            "is_weekend", "is_business_day"
        ],
        description="List of features to generate"
    )
    
    @field_validator('timezone')
    @classmethod
    def validate_timezone(cls, v):
        """Validate timezone string."""
        if v not in pytz.all_timezones:
            raise ValueError(f"Invalid timezone: {v}")
        return v


class TemporalFeaturesPlugin(BaseFeaturePlugin):
    """
    Plugin for generating temporal features from datetime index.
    
    Features include:
    - Basic: hour, day_of_week, day_of_month, month, quarter, year
    - Flags: is_weekend, is_business_day
    - Season: Southern Hemisphere seasons (if enabled)
    - Cyclical: sin/cos encoding for periodic features (if enabled)
    
    Example:
        >>> plugin = TemporalFeaturesPlugin()
        >>> config = {
        ...     "include_cyclical": True,
        ...     "include_season": True,
        ...     "timezone": "America/Sao_Paulo"
        ... }
        >>> features = plugin.generate_features(df, config)
    """
    
    @property
    def name(self) -> str:
        return "temporal_features"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    def validate_config(self, config: Dict[str, Any]) -> None:
        """Validate configuration using Pydantic model."""
        TemporalFeaturesConfig(**config)
        logger.debug("Temporal features config validated")
    
    def generate_features(
        self,
        df: pd.DataFrame,
        config: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Generate temporal features from DataFrame datetime index.
        
        Args:
            df: Input DataFrame with DatetimeIndex
            config: Plugin configuration
        
        Returns:
            DataFrame with temporal features
        
        Raises:
            ValueError: If DataFrame doesn't have DatetimeIndex
        """
        # Validate config
        validated_config = TemporalFeaturesConfig(**config)
        
        # Validate input
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have DatetimeIndex")
        
        logger.info(f"Generating temporal features for {len(df)} records")
        
        # Ensure timezone
        dt_index = df.index
        if dt_index.tz is None:
            tz = pytz.timezone(validated_config.timezone)
            dt_index = dt_index.tz_localize(tz)
        
        # Initialize feature dictionary
        features = {}
        
        # Generate basic features
        if "hour" in validated_config.features:
            features["hour"] = dt_index.hour
        
        if "day_of_week" in validated_config.features:
            features["day_of_week"] = dt_index.dayofweek
        
        if "day_of_month" in validated_config.features:
            features["day_of_month"] = dt_index.day
        
        if "month" in validated_config.features:
            features["month"] = dt_index.month
        
        if "quarter" in validated_config.features:
            features["quarter"] = dt_index.quarter
        
        if "year" in validated_config.features:
            features["year"] = dt_index.year
        
        # Generate flag features
        if "is_weekend" in validated_config.features:
            features["is_weekend"] = (dt_index.dayofweek >= 5).astype(int)
        
        if "is_business_day" in validated_config.features:
            features["is_business_day"] = (dt_index.dayofweek < 5).astype(int)
        
        # Generate season features (Southern Hemisphere)
        if validated_config.include_season:
            features["season"] = dt_index.month.map(self._get_season)
        
        # Generate cyclical encoding
        if validated_config.include_cyclical:
            if "hour" in validated_config.features:
                features["hour_sin"] = np.sin(2 * np.pi * dt_index.hour / 24)
                features["hour_cos"] = np.cos(2 * np.pi * dt_index.hour / 24)
            
            if "day_of_week" in validated_config.features:
                features["day_of_week_sin"] = np.sin(2 * np.pi * dt_index.dayofweek / 7)
                features["day_of_week_cos"] = np.cos(2 * np.pi * dt_index.dayofweek / 7)
            
            if "month" in validated_config.features:
                features["month_sin"] = np.sin(2 * np.pi * dt_index.month / 12)
                features["month_cos"] = np.cos(2 * np.pi * dt_index.month / 12)
        
        # Create DataFrame
        result = pd.DataFrame(features, index=df.index)
        
        logger.info(f"Generated {len(result.columns)} temporal features")
        return result
    
    def get_feature_names(self, config: Dict[str, Any]) -> List[str]:
        """Get list of feature names based on configuration."""
        validated_config = TemporalFeaturesConfig(**config)
        
        feature_names = validated_config.features.copy()
        
        # Add season if enabled
        if validated_config.include_season:
            feature_names.append("season")
        
        # Add cyclical encodings if enabled
        if validated_config.include_cyclical:
            cyclical_features = []
            if "hour" in validated_config.features:
                cyclical_features.extend(["hour_sin", "hour_cos"])
            if "day_of_week" in validated_config.features:
                cyclical_features.extend(["day_of_week_sin", "day_of_week_cos"])
            if "month" in validated_config.features:
                cyclical_features.extend(["month_sin", "month_cos"])
            feature_names.extend(cyclical_features)
        
        return feature_names
    
    @staticmethod
    def _get_season(month: int) -> int:
        """
        Map month to Southern Hemisphere season.
        
        Args:
            month: Month number (1-12)
        
        Returns:
            Season code: 1=Summer, 2=Autumn, 3=Winter, 4=Spring
        """
        if month in [12, 1, 2]:
            return 1  # Summer
        elif month in [3, 4, 5]:
            return 2  # Autumn
        elif month in [6, 7, 8]:
            return 3  # Winter
        else:  # month in [9, 10, 11]
            return 4  # Spring
```

---

## 🧪 Testing & Validation

### Unit Tests

```python
"""Tests for temporal features plugin."""
import pytest
import pandas as pd
import numpy as np
import pytz
from datetime import datetime

from src.features.plugins.temporal import TemporalFeaturesPlugin, TemporalFeaturesConfig


@pytest.fixture
def sample_df():
    """Create sample DataFrame with datetime index."""
    dates = pd.date_range(
        start="2024-01-01",
        end="2024-12-31",
        freq="h",
        tz="America/Sao_Paulo"
    )
    return pd.DataFrame({"value": range(len(dates))}, index=dates)


@pytest.fixture
def plugin():
    """Create plugin instance."""
    return TemporalFeaturesPlugin()


def test_basic_temporal_features(plugin, sample_df):
    """Test basic temporal feature extraction."""
    config = {"include_cyclical": False, "include_season": False}
    result = plugin.generate_features(sample_df, config)
    
    assert "hour" in result.columns
    assert "day_of_week" in result.columns
    assert "month" in result.columns
    assert result["hour"].min() == 0
    assert result["hour"].max() == 23
    assert result["day_of_week"].min() == 0
    assert result["day_of_week"].max() == 6


def test_cyclical_encoding(plugin, sample_df):
    """Test cyclical encoding for periodic features."""
    config = {"include_cyclical": True}
    result = plugin.generate_features(sample_df, config)
    
    assert "hour_sin" in result.columns
    assert "hour_cos" in result.columns
    assert "day_of_week_sin" in result.columns
    assert "day_of_week_cos" in result.columns
    
    # Check values are in [-1, 1]
    assert result["hour_sin"].between(-1, 1).all()
    assert result["hour_cos"].between(-1, 1).all()
    
    # Check sin^2 + cos^2 = 1
    hour_sum = result["hour_sin"]**2 + result["hour_cos"]**2
    np.testing.assert_allclose(hour_sum, 1.0, rtol=1e-10)


def test_southern_hemisphere_seasons(plugin, sample_df):
    """Test season mapping for Southern Hemisphere."""
    config = {"include_season": True}
    result = plugin.generate_features(sample_df, config)
    
    assert "season" in result.columns
    
    # Check season values
    january_season = result[result.index.month == 1]["season"].iloc[0]
    assert january_season == 1  # Summer
    
    june_season = result[result.index.month == 6]["season"].iloc[0]
    assert june_season == 3  # Winter


def test_weekend_flags(plugin, sample_df):
    """Test weekend and business day flags."""
    config = {}
    result = plugin.generate_features(sample_df, config)
    
    assert "is_weekend" in result.columns
    assert "is_business_day" in result.columns
    
    # Check mutual exclusivity
    assert ((result["is_weekend"] + result["is_business_day"]) == 1).all()


def test_timezone_handling(plugin):
    """Test timezone conversion."""
    # Create DataFrame without timezone
    dates = pd.date_range(start="2024-01-01", periods=24, freq="h")
    df = pd.DataFrame({"value": range(24)}, index=dates)
    
    config = {"timezone": "America/Sao_Paulo"}
    result = plugin.generate_features(df, config)
    
    assert len(result) == 24
    assert "hour" in result.columns


def test_feature_names(plugin):
    """Test feature name generation."""
    config = {
        "include_cyclical": True,
        "include_season": True,
        "features": ["hour", "month"]
    }
    names = plugin.get_feature_names(config)
    
    assert "hour" in names
    assert "month" in names
    assert "season" in names
    assert "hour_sin" in names
    assert "hour_cos" in names
    assert "month_sin" in names
    assert "month_cos" in names


def test_invalid_timezone():
    """Test invalid timezone raises error."""
    with pytest.raises(ValueError, match="Invalid timezone"):
        TemporalFeaturesConfig(timezone="Invalid/Timezone")


def test_performance(plugin, sample_df):
    """Test performance with large dataset."""
    import time
    
    start = time.time()
    result = plugin.generate_features(sample_df, {})
    elapsed = time.time() - start
    
    # Should process 1 year of hourly data in <100ms
    assert elapsed < 0.1
    assert len(result) == len(sample_df)
```

---

## 📝 Technical Notes

- Use numpy for efficient cyclical encoding calculations
- Timezone handling critical for Brazilian DST transitions
- Southern Hemisphere seasons different from Northern Hemisphere
- Cyclical encoding preserves periodic relationships (e.g., 23:00 close to 00:00)
- sin²(x) + cos²(x) = 1 property useful for validation

---

## 🔗 Dependencies

**Depends On:**
- PC-012-02A: Plugin Architecture Foundation

**Blocks:**
- PC-017-02A: Feature Pipeline Composer

---

## 📊 Definition of Done

- [ ] All acceptance criteria met
- [ ] TemporalFeaturesPlugin implemented
- [ ] Configuration schema with validation
- [ ] Cyclical encoding working correctly
- [ ] Southern Hemisphere seasons correct
- [ ] Unit tests pass with >90% coverage
- [ ] Performance benchmark met (<100ms for 1 year hourly data)
- [ ] Code reviewed and approved
- [ ] Plugin registered in example code

---

## 🔄 Change Log

| Date | Version | Changes | Author |
|------|---------|---------|--------|
| 2025-11-18 | 1.0.0 | Initial ticket created | System |

---

**Previous:** [PC-012-02A: Plugin Architecture Foundation](PC-012-02A-plugin-architecture-foundation.md)  
**Next:** [PC-014-02A: Calendar/Holiday Features Plugin](PC-014-02A-calendar-holiday-features-plugin.md)
