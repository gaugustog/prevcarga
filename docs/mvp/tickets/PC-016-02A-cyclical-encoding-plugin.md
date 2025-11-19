# PC-016-02A: Cyclical Encoding Plugin

**Ticket ID:** PC-016-02A  
**Epic:** [Epic-02A: Core Feature Engineering](../epics/Epic-02A.md)  
**User Story:** US-5  
**Story Points:** 3  
**Priority:** Medium  
**Assignee:** TBD  
**Status:** 📝 To Do

---

## 📋 Description

Implement a standalone plugin for cyclical encoding of periodic features using sine/cosine transformations. Provides reusable encoding for hour, day, month, and any custom periodic features with configurable period lengths.

**As a** data scientist  
**I want** standalone cyclical encoding capabilities  
**So that** I can encode any periodic feature without code duplication

---

## ✅ Acceptance Criteria

- [ ] `CyclicalEncodingPlugin` encodes periodic features
- [ ] Support configurable period lengths (24, 7, 12, 365, etc.)
- [ ] Generate sin/cos pair for each feature
- [ ] Preserve original features optionally
- [ ] Validate encoded values in [-1, 1]
- [ ] Verify sin² + cos² = 1 property
- [ ] Support multiple features in single call
- [ ] Unit tests with various periods
- [ ] Performance: <10ms for 1 year of hourly data

---

## 🔧 Implementation Tasks

### 1. Create Plugin Module
- [ ] Create `src/features/plugins/cyclical.py`
- [ ] Import BaseFeaturePlugin and dependencies
- [ ] Add module docstring explaining cyclical encoding

### 2. Implement CyclicalEncodingPlugin Class
- [ ] Inherit from `BaseFeaturePlugin`
- [ ] Implement `name` property returning "cyclical_encoding"
- [ ] Implement `version` property with semantic versioning
- [ ] Add class docstring with mathematical explanation

### 3. Implement Configuration Schema
- [ ] Create `CyclicalEncodingConfig` Pydantic model
- [ ] Add `features` dictionary mapping feature name to period
- [ ] Add `keep_original` boolean flag (default: True)
- [ ] Add `suffix_sin` string (default: "_sin")
- [ ] Add `suffix_cos` string (default: "_cos")
- [ ] Validate periods are positive integers
- [ ] Validate feature names exist in input

### 4. Implement Cyclical Encoding Logic
- [ ] Create `_encode_cyclical()` static method
- [ ] Implement formula: sin(2π × value / period)
- [ ] Implement formula: cos(2π × value / period)
- [ ] Use numpy for vectorized computation
- [ ] Handle edge cases (value = 0, value = period)
- [ ] Return sin and cos arrays

### 5. Implement generate_features() Method
- [ ] Validate input DataFrame structure
- [ ] Check configured features exist in DataFrame
- [ ] Encode each configured feature
- [ ] Generate sin/cos pairs
- [ ] Optionally keep original features
- [ ] Ensure output index matches input
- [ ] Add error handling for invalid values
- [ ] Log encoding summary

### 6. Implement get_feature_names() Method
- [ ] Generate sin/cos feature names
- [ ] Include original feature names if configured
- [ ] Return complete list
- [ ] Maintain naming convention

### 7. Add Validation Utilities
- [ ] Create `_validate_encoding()` method
- [ ] Check sin values in [-1, 1]
- [ ] Check cos values in [-1, 1]
- [ ] Verify sin² + cos² ≈ 1 (within tolerance)
- [ ] Log validation warnings if checks fail

### 8. Write Comprehensive Tests
- [ ] Create `tests/features/plugins/test_cyclical.py`
- [ ] Test basic sin/cos encoding
- [ ] Test multiple features
- [ ] Test various period lengths
- [ ] Test value range validation
- [ ] Test Pythagorean identity (sin² + cos² = 1)
- [ ] Test keep_original flag
- [ ] Test custom suffix naming
- [ ] Test edge cases (zero, max values)
- [ ] Test performance benchmark

### 9. Create Mathematical Documentation
- [ ] Document cyclical encoding theory
- [ ] Explain why sine/cosine for periodic features
- [ ] Provide visualization examples
- [ ] Include use cases (hour, day, month, angle)
- [ ] Add references to literature

---

## 💻 Implementation Details

### Configuration Schema

```python
"""Configuration for cyclical encoding plugin."""
from typing import Dict
from pydantic import BaseModel, Field, field_validator


class CyclicalEncodingConfig(BaseModel):
    """Configuration for cyclical encoding plugin."""
    
    features: Dict[str, int] = Field(
        default_factory=dict,
        description="Mapping of feature names to their period lengths"
    )
    keep_original: bool = Field(
        default=True,
        description="Whether to keep original features in output"
    )
    suffix_sin: str = Field(
        default="_sin",
        description="Suffix for sine-encoded features"
    )
    suffix_cos: str = Field(
        default="_cos",
        description="Suffix for cosine-encoded features"
    )
    
    @field_validator('features')
    @classmethod
    def validate_features(cls, v):
        """Validate period values are positive integers."""
        for feature, period in v.items():
            if not isinstance(period, int) or period <= 0:
                raise ValueError(
                    f"Period for '{feature}' must be positive integer, got {period}"
                )
        return v
```

### CyclicalEncodingPlugin Implementation

```python
"""Cyclical encoding plugin for periodic features."""
from typing import Dict, Any, List
import pandas as pd
import numpy as np

from src.features.base.plugin import BaseFeaturePlugin
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CyclicalEncodingPlugin(BaseFeaturePlugin):
    """
    Plugin for cyclical encoding of periodic features.
    
    Transforms periodic features using sine/cosine encoding to preserve
    cyclic relationships. This is particularly useful for features like:
    - Hour of day (period = 24)
    - Day of week (period = 7)
    - Month of year (period = 12)
    - Day of year (period = 365)
    
    Mathematical transformation:
    - sin_feature = sin(2π × value / period)
    - cos_feature = cos(2π × value / period)
    
    Properties:
    - Preserves cyclic distance (e.g., 23:00 is close to 00:00)
    - Values bounded in [-1, 1]
    - Satisfies Pythagorean identity: sin² + cos² = 1
    
    Example:
        >>> plugin = CyclicalEncodingPlugin()
        >>> config = {
        ...     "features": {
        ...         "hour": 24,
        ...         "day_of_week": 7,
        ...         "month": 12
        ...     },
        ...     "keep_original": True
        ... }
        >>> encoded = plugin.generate_features(df, config)
    """
    
    @property
    def name(self) -> str:
        return "cyclical_encoding"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    def validate_config(self, config: Dict[str, Any]) -> None:
        """Validate configuration using Pydantic model."""
        CyclicalEncodingConfig(**config)
        logger.debug("Cyclical encoding config validated")
    
    def generate_features(
        self,
        df: pd.DataFrame,
        config: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Generate cyclically encoded features.
        
        Args:
            df: Input DataFrame with features to encode
            config: Plugin configuration
        
        Returns:
            DataFrame with sin/cos encoded features
        
        Raises:
            ValueError: If configured feature not in DataFrame
        """
        # Validate config
        validated_config = CyclicalEncodingConfig(**config)
        
        if not validated_config.features:
            logger.warning("No features configured for encoding, returning empty DataFrame")
            return pd.DataFrame(index=df.index)
        
        logger.info(
            f"Encoding {len(validated_config.features)} features: "
            f"{list(validated_config.features.keys())}"
        )
        
        # Initialize result dictionary
        result_dict = {}
        
        # Optionally keep original features
        if validated_config.keep_original:
            for feature_name in validated_config.features.keys():
                if feature_name in df.columns:
                    result_dict[feature_name] = df[feature_name]
        
        # Encode each feature
        for feature_name, period in validated_config.features.items():
            # Validate feature exists
            if feature_name not in df.columns:
                raise ValueError(
                    f"Feature '{feature_name}' not found in DataFrame. "
                    f"Available columns: {list(df.columns)}"
                )
            
            # Get feature values
            values = df[feature_name].values
            
            # Encode using sin/cos
            sin_values, cos_values = self._encode_cyclical(values, period)
            
            # Create feature names
            sin_name = f"{feature_name}{validated_config.suffix_sin}"
            cos_name = f"{feature_name}{validated_config.suffix_cos}"
            
            # Store encoded features
            result_dict[sin_name] = sin_values
            result_dict[cos_name] = cos_values
            
            # Validate encoding
            self._validate_encoding(sin_values, cos_values, feature_name)
        
        # Create result DataFrame
        result = pd.DataFrame(result_dict, index=df.index)
        
        logger.info(f"Generated {len(result.columns)} encoded features")
        return result
    
    def get_feature_names(self, config: Dict[str, Any]) -> List[str]:
        """Get list of feature names based on configuration."""
        validated_config = CyclicalEncodingConfig(**config)
        
        feature_names = []
        
        for feature_name in validated_config.features.keys():
            # Add original if configured
            if validated_config.keep_original:
                feature_names.append(feature_name)
            
            # Add sin/cos encoded features
            feature_names.append(f"{feature_name}{validated_config.suffix_sin}")
            feature_names.append(f"{feature_name}{validated_config.suffix_cos}")
        
        return feature_names
    
    @staticmethod
    def _encode_cyclical(values: np.ndarray, period: int) -> tuple:
        """
        Encode values using cyclical sin/cos transformation.
        
        Args:
            values: Array of values to encode
            period: Period length (e.g., 24 for hours)
        
        Returns:
            Tuple of (sin_values, cos_values)
        """
        # Normalize to [0, 1] range
        normalized = values / period
        
        # Apply sin/cos transformation
        sin_values = np.sin(2 * np.pi * normalized)
        cos_values = np.cos(2 * np.pi * normalized)
        
        return sin_values, cos_values
    
    @staticmethod
    def _validate_encoding(
        sin_values: np.ndarray,
        cos_values: np.ndarray,
        feature_name: str,
        tolerance: float = 1e-6
    ) -> None:
        """
        Validate cyclical encoding properties.
        
        Args:
            sin_values: Sine-encoded values
            cos_values: Cosine-encoded values
            feature_name: Name of feature (for logging)
            tolerance: Tolerance for Pythagorean identity check
        """
        # Check value ranges
        if not np.all((sin_values >= -1) & (sin_values <= 1)):
            logger.warning(
                f"Feature '{feature_name}': sin values outside [-1, 1] range"
            )
        
        if not np.all((cos_values >= -1) & (cos_values <= 1)):
            logger.warning(
                f"Feature '{feature_name}': cos values outside [-1, 1] range"
            )
        
        # Check Pythagorean identity: sin² + cos² = 1
        pythagorean_sum = sin_values**2 + cos_values**2
        if not np.allclose(pythagorean_sum, 1.0, atol=tolerance):
            max_deviation = np.max(np.abs(pythagorean_sum - 1.0))
            logger.warning(
                f"Feature '{feature_name}': Pythagorean identity violated. "
                f"Max deviation: {max_deviation}"
            )
```

---

## 🧪 Testing & Validation

### Unit Tests

```python
"""Tests for cyclical encoding plugin."""
import pytest
import pandas as pd
import numpy as np

from src.features.plugins.cyclical import (
    CyclicalEncodingPlugin,
    CyclicalEncodingConfig
)


@pytest.fixture
def plugin():
    """Create plugin instance."""
    return CyclicalEncodingPlugin()


@pytest.fixture
def sample_df():
    """Create sample DataFrame with periodic features."""
    return pd.DataFrame({
        "hour": np.arange(24),
        "day_of_week": np.arange(7),
        "month": np.arange(1, 13)
    })


def test_basic_encoding(plugin, sample_df):
    """Test basic sin/cos encoding."""
    config = {
        "features": {"hour": 24},
        "keep_original": False
    }
    result = plugin.generate_features(sample_df, config)
    
    assert "hour_sin" in result.columns
    assert "hour_cos" in result.columns
    assert "hour" not in result.columns  # Not kept


def test_keep_original(plugin, sample_df):
    """Test keeping original features."""
    config = {
        "features": {"hour": 24},
        "keep_original": True
    }
    result = plugin.generate_features(sample_df, config)
    
    assert "hour" in result.columns
    assert "hour_sin" in result.columns
    assert "hour_cos" in result.columns


def test_multiple_features(plugin, sample_df):
    """Test encoding multiple features."""
    config = {
        "features": {
            "hour": 24,
            "day_of_week": 7,
            "month": 12
        },
        "keep_original": False
    }
    result = plugin.generate_features(sample_df, config)
    
    assert len(result.columns) == 6  # 3 features × 2 encodings
    assert "hour_sin" in result.columns
    assert "day_of_week_cos" in result.columns
    assert "month_sin" in result.columns


def test_value_ranges(plugin, sample_df):
    """Test encoded values are in [-1, 1]."""
    config = {"features": {"hour": 24}}
    result = plugin.generate_features(sample_df, config)
    
    assert result["hour_sin"].between(-1, 1).all()
    assert result["hour_cos"].between(-1, 1).all()


def test_pythagorean_identity(plugin, sample_df):
    """Test sin² + cos² = 1."""
    config = {"features": {"hour": 24}}
    result = plugin.generate_features(sample_df, config)
    
    pythagorean_sum = result["hour_sin"]**2 + result["hour_cos"]**2
    np.testing.assert_allclose(pythagorean_sum, 1.0, rtol=1e-10)


def test_cyclic_distance(plugin):
    """Test that encoding preserves cyclic distance."""
    # Create DataFrame with hour 0 and hour 23
    df = pd.DataFrame({"hour": [0, 23]})
    
    config = {"features": {"hour": 24}}
    result = plugin.generate_features(df, config)
    
    # Hour 0 and hour 23 should be close in encoded space
    sin_0 = result["hour_sin"].iloc[0]
    sin_23 = result["hour_sin"].iloc[1]
    cos_0 = result["hour_cos"].iloc[0]
    cos_23 = result["hour_cos"].iloc[1]
    
    # Distance should be small
    distance = np.sqrt((sin_0 - sin_23)**2 + (cos_0 - cos_23)**2)
    assert distance < 0.3  # Close in 2D space


def test_custom_suffixes(plugin, sample_df):
    """Test custom suffix naming."""
    config = {
        "features": {"hour": 24},
        "suffix_sin": "_sine",
        "suffix_cos": "_cosine"
    }
    result = plugin.generate_features(sample_df, config)
    
    assert "hour_sine" in result.columns
    assert "hour_cosine" in result.columns


def test_feature_names(plugin):
    """Test feature name generation."""
    config = {
        "features": {"hour": 24, "day": 7},
        "keep_original": True
    }
    names = plugin.get_feature_names(config)
    
    assert "hour" in names
    assert "hour_sin" in names
    assert "hour_cos" in names
    assert "day" in names
    assert "day_sin" in names
    assert "day_cos" in names


def test_missing_feature_error(plugin, sample_df):
    """Test error when configured feature doesn't exist."""
    config = {"features": {"nonexistent": 10}}
    
    with pytest.raises(ValueError, match="not found in DataFrame"):
        plugin.generate_features(sample_df, config)


def test_invalid_period():
    """Test that invalid periods are rejected."""
    with pytest.raises(ValueError, match="positive integer"):
        CyclicalEncodingConfig(features={"hour": -1})
    
    with pytest.raises(ValueError, match="positive integer"):
        CyclicalEncodingConfig(features={"hour": 0})


def test_edge_case_zero_value(plugin):
    """Test encoding of zero value."""
    df = pd.DataFrame({"value": [0]})
    config = {"features": {"value": 10}}
    
    result = plugin.generate_features(df, config)
    
    # sin(0) = 0, cos(0) = 1
    np.testing.assert_almost_equal(result["value_sin"].iloc[0], 0.0)
    np.testing.assert_almost_equal(result["value_cos"].iloc[0], 1.0)


def test_edge_case_max_value(plugin):
    """Test encoding of max period value."""
    df = pd.DataFrame({"value": [24]})
    config = {"features": {"value": 24}}
    
    result = plugin.generate_features(df, config)
    
    # sin(2π) = 0, cos(2π) = 1
    np.testing.assert_almost_equal(result["value_sin"].iloc[0], 0.0, decimal=10)
    np.testing.assert_almost_equal(result["value_cos"].iloc[0], 1.0, decimal=10)


def test_performance(plugin):
    """Test performance with large dataset."""
    import time
    
    # 1 year of hourly data
    large_df = pd.DataFrame({
        "hour": np.tile(np.arange(24), 365),
        "day": np.repeat(np.arange(365), 24)
    })
    
    config = {
        "features": {"hour": 24, "day": 365}
    }
    
    start = time.time()
    result = plugin.generate_features(large_df, config)
    elapsed = time.time() - start
    
    # Should complete in <10ms
    assert elapsed < 0.01
    assert len(result) == len(large_df)
```

---

## 📝 Technical Notes

### Why Cyclical Encoding?

1. **Preserves Cyclic Relationships**: Hour 23 is close to hour 0
2. **Continuous Representation**: No arbitrary jumps in encoded space
3. **Bounded Values**: Always in [-1, 1] range
4. **Mathematical Properties**: Satisfies Pythagorean identity

### Use Cases

- **Hour**: Encodes daily cycle (period = 24)
- **Day of Week**: Encodes weekly cycle (period = 7)
- **Month**: Encodes annual cycle (period = 12)
- **Day of Year**: Encodes seasonal cycle (period = 365)
- **Angle**: Encodes directional data (period = 360)

### Visualization

For hour encoding (period = 24):
- Hour 0 → (sin=0, cos=1) → 0°
- Hour 6 → (sin=1, cos=0) → 90°
- Hour 12 → (sin=0, cos=-1) → 180°
- Hour 18 → (sin=-1, cos=0) → 270°

---

## 🔗 Dependencies

**Depends On:**
- PC-012-02A: Plugin Architecture Foundation

**Blocks:**
- PC-017-02A: Feature Pipeline Composer

**Related:**
- PC-013-02A: Temporal Features Plugin (uses cyclical encoding internally)

---

## 📊 Definition of Done

- [ ] All acceptance criteria met
- [ ] CyclicalEncodingPlugin implemented
- [ ] Sin/cos encoding working correctly
- [ ] Pythagorean identity validated
- [ ] Multiple period support working
- [ ] Unit tests pass with >95% coverage
- [ ] Performance benchmark met (<10ms for 1 year hourly)
- [ ] Mathematical documentation complete
- [ ] Code reviewed and approved

---

## 🔄 Change Log

| Date | Version | Changes | Author |
|------|---------|---------|--------|
| 2025-11-18 | 1.0.0 | Initial ticket created | System |

---

**Previous:** [PC-015-02A: Lag Features Plugin](PC-015-02A-lag-features-plugin.md)  
**Next:** [PC-017-02A: Feature Pipeline Composer](PC-017-02A-feature-pipeline-composer.md)
