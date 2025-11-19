# PC-019-02B: Wavelet Transform Plugin

**Ticket ID:** PC-019-02B  
**Epic:** [Epic-02B: Advanced Feature Transformations](../epics/Epic-02B.md)  
**User Story:** US-2  
**Story Points:** 8  
**Priority:** High  
**Assignee:** TBD  
**Status:** 📝 To Do

---

## 📋 Description

Implement a Discrete Wavelet Transform (DWT) plugin for multi-scale time-frequency analysis of electric load signals. Extracts approximation and detail coefficients at multiple decomposition levels to capture patterns at different time scales.

**As a** signal processing engineer  
**I want** wavelet decomposition features  
**So that** models can analyze load patterns across multiple time-frequency scales

---

## ✅ Acceptance Criteria

- [ ] `WaveletFeaturePlugin` implements Discrete Wavelet Transform
- [ ] Supports multiple wavelet families (db4, haar, coif2)
- [ ] Generates coefficients for approximation and detail levels (1-6)
- [ ] Handles boundary conditions appropriately (padding)
- [ ] Provides coefficient statistics (energy, entropy)
- [ ] Includes inverse transform validation for correctness
- [ ] Unit tests validate mathematical properties
- [ ] Performance: <10 seconds for 1 year hourly data

---

## 🔧 Implementation Tasks

### 1. Install Wavelet Dependencies
- [ ] Add `pywt>=1.4.0` to `pyproject.toml`
- [ ] Document PyWavelets installation
- [ ] Verify wavelet availability across platforms

### 2. Create Wavelet Plugin Module
- [ ] Create `src/features/plugins/wavelet.py`
- [ ] Import PyWavelets (`pywt`)
- [ ] Add module docstring with wavelet theory

### 3. Implement Configuration Schema
- [ ] Create `WaveletFeaturesConfig` Pydantic model
- [ ] Add `target_column` string field
- [ ] Add `wavelet` string (default: "db4")
- [ ] Add `levels` integer (decomposition levels, default: 4)
- [ ] Add `mode` string (boundary condition, default: "symmetric")
- [ ] Add `include_statistics` boolean (default: True)
- [ ] Add `include_energy` boolean (default: True)
- [ ] Add `include_entropy` boolean (default: True)
- [ ] Validate wavelet name against PyWavelets families
- [ ] Validate levels >= 1

### 4. Implement WaveletFeaturePlugin Class
- [ ] Inherit from `AdvancedFeaturePlugin`
- [ ] Implement `name` property returning "wavelet_features"
- [ ] Implement `version` property
- [ ] Implement `computational_complexity` returning "medium"
- [ ] Add class docstring with mathematical background

### 5. Implement Signal Padding
- [ ] Create `_pad_signal()` method
- [ ] Calculate next power of 2 for efficient DWT
- [ ] Pad signal using edge values
- [ ] Return padded signal and original length
- [ ] Log padding information

### 6. Implement Multi-Level DWT
- [ ] Create `_apply_dwt()` method
- [ ] Use `pywt.wavedec()` for multi-level decomposition
- [ ] Extract approximation coefficients (cA)
- [ ] Extract detail coefficients (cD) for each level
- [ ] Store coefficients hierarchy
- [ ] Handle decomposition errors

### 7. Implement Coefficient Resizing
- [ ] Create `_resize_coeffs()` method
- [ ] Interpolate coefficients to match original length
- [ ] Use linear interpolation for upsampling
- [ ] Maintain temporal alignment
- [ ] Handle edge cases (single coefficient)

### 8. Implement Wavelet Energy Calculation
- [ ] Create `_calculate_energy()` method
- [ ] Calculate energy: sum of squared coefficients
- [ ] Normalize by total signal energy
- [ ] Return energy per decomposition level
- [ ] Implement relative energy percentages

### 9. Implement Wavelet Entropy Calculation
- [ ] Create `_calculate_entropy()` method
- [ ] Calculate Shannon entropy from coefficient energy
- [ ] Formula: -Σ(p * log2(p)) where p = energy/total_energy
- [ ] Handle zero probabilities
- [ ] Return entropy as complexity measure

### 10. Implement Inverse DWT Validation
- [ ] Create `_validate_reconstruction()` method
- [ ] Use `pywt.waverec()` for inverse transform
- [ ] Calculate reconstruction error
- [ ] Compare reconstructed signal with original
- [ ] Log reconstruction RMSE
- [ ] Warn if error exceeds threshold (1%)

### 11. Implement generate_features() Method
- [ ] Validate input DataFrame
- [ ] Check target column exists
- [ ] Get signal and pad if needed
- [ ] Apply multi-level DWT
- [ ] Resize approximation coefficients
- [ ] Resize detail coefficients for each level
- [ ] Calculate energy features if configured
- [ ] Calculate entropy features if configured
- [ ] Validate reconstruction
- [ ] Return DataFrame with all features

### 12. Implement get_feature_names() Method
- [ ] Generate names for approximation features
- [ ] Generate names for detail features per level
- [ ] Generate names for energy features if enabled
- [ ] Generate names for entropy features if enabled
- [ ] Return complete list

### 13. Implement Performance Estimation
- [ ] Implement `estimate_compute_time()`
- [ ] DWT complexity: O(n log n)
- [ ] Factor in number of decomposition levels
- [ ] Return estimated seconds

### 14. Write Comprehensive Tests
- [ ] Create `tests/features/plugins/test_wavelet.py`
- [ ] Test basic DWT decomposition
- [ ] Test multi-level decomposition (1-6 levels)
- [ ] Test different wavelet families
- [ ] Test reconstruction accuracy (<1% error)
- [ ] Test energy calculation correctness
- [ ] Test entropy calculation
- [ ] Test coefficient resizing
- [ ] Test with synthetic signals (known frequencies)
- [ ] Test boundary condition handling
- [ ] Test performance benchmark

### 15. Create Wavelet Theory Documentation
- [ ] Document wavelet transform theory
- [ ] Explain approximation vs detail coefficients
- [ ] Describe frequency bands per level
- [ ] Include wavelet selection guide
- [ ] Add visualization examples
- [ ] Document use cases

---

## 💻 Implementation Details

### Configuration Schema

```python
"""Configuration for wavelet features plugin."""
from pydantic import BaseModel, Field, field_validator
import pywt


class WaveletFeaturesConfig(BaseModel):
    """Configuration for wavelet transform plugin."""
    
    target_column: str = Field(
        default="carga",
        description="Target column for wavelet decomposition"
    )
    wavelet: str = Field(
        default="db4",
        description="Wavelet family (db4, haar, coif2, sym4, etc.)"
    )
    levels: int = Field(
        default=4,
        description="Number of decomposition levels (1-6)"
    )
    mode: str = Field(
        default="symmetric",
        description="Signal extension mode for boundary handling"
    )
    include_statistics: bool = Field(
        default=True,
        description="Include statistical features of coefficients"
    )
    include_energy: bool = Field(
        default=True,
        description="Include energy features"
    )
    include_entropy: bool = Field(
        default=True,
        description="Include entropy features"
    )
    
    @field_validator('wavelet')
    @classmethod
    def validate_wavelet(cls, v):
        """Validate wavelet family exists."""
        available_wavelets = pywt.wavelist(kind='discrete')
        if v not in available_wavelets:
            raise ValueError(
                f"Invalid wavelet: {v}. "
                f"Available: {available_wavelets[:10]}..."
            )
        return v
    
    @field_validator('levels')
    @classmethod
    def validate_levels(cls, v):
        """Validate decomposition levels."""
        if v < 1 or v > 6:
            raise ValueError(f"Levels must be 1-6, got {v}")
        return v
    
    @field_validator('mode')
    @classmethod
    def validate_mode(cls, v):
        """Validate extension mode."""
        valid_modes = pywt.Modes.modes
        if v not in valid_modes:
            raise ValueError(
                f"Invalid mode: {v}. Valid: {valid_modes}"
            )
        return v
```

### WaveletFeaturePlugin Implementation

```python
"""Wavelet transform plugin for multi-scale signal analysis."""
from typing import Dict, Any, List, Literal, Tuple
import pandas as pd
import numpy as np
import pywt

from src.features.base.plugin import AdvancedFeaturePlugin
from src.utils.logger import get_logger

logger = get_logger(__name__)


class WaveletFeaturePlugin(AdvancedFeaturePlugin):
    """
    Plugin for generating wavelet transform features.
    
    Uses Discrete Wavelet Transform (DWT) to decompose time series
    into multi-scale representations, capturing patterns at different
    time-frequency resolutions.
    
    Mathematical Background:
    - DWT decomposes signal into approximation (low-freq) and detail (high-freq)
    - Multi-level decomposition creates frequency hierarchy
    - Level 1: Highest frequency details
    - Level N: Lowest frequency details
    - Approximation: Smoothed trend component
    
    Features Generated:
    - wavelet_approx: Approximation coefficients (trend)
    - wavelet_detail_L{i}: Detail coefficients at level i
    - wavelet_energy_L{i}: Energy at level i
    - wavelet_entropy_L{i}: Entropy (complexity) at level i
    
    Wavelet Selection Guide:
    - db4 (Daubechies 4): Good for general signals, asymmetric
    - haar: Simplest, good for discontinuities
    - coif2: Symmetric, good for smooth signals
    - sym4: Symmetric version of daubechies
    
    Example:
        >>> plugin = WaveletFeaturePlugin()
        >>> config = {
        ...     "target_column": "carga",
        ...     "wavelet": "db4",
        ...     "levels": 4
        ... }
        >>> features = plugin.generate_features(df, config)
    """
    
    @property
    def name(self) -> str:
        return "wavelet_features"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def computational_complexity(self) -> Literal["low", "medium", "high"]:
        return "medium"
    
    def estimate_compute_time(self, data_size: int) -> float:
        """
        Estimate DWT computation time.
        
        DWT complexity is O(n log n).
        
        Args:
            data_size: Number of data points
        
        Returns:
            Estimated seconds
        """
        # Empirical formula
        return 0.000001 * data_size * np.log2(data_size + 1)
    
    def validate_config(self, config: Dict[str, Any]) -> None:
        """Validate configuration."""
        WaveletFeaturesConfig(**config)
        logger.debug("Wavelet features config validated")
    
    def generate_features(
        self,
        df: pd.DataFrame,
        config: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Generate wavelet transform features.
        
        Args:
            df: Input DataFrame
            config: Plugin configuration
        
        Returns:
            DataFrame with wavelet features
        """
        # Validate config
        validated_config = WaveletFeaturesConfig(**config)
        
        # Validate target column
        if validated_config.target_column not in df.columns:
            raise ValueError(
                f"Target column '{validated_config.target_column}' not found"
            )
        
        # Get signal
        signal = df[validated_config.target_column].dropna().values
        
        logger.info(
            f"Generating wavelet features: {validated_config.wavelet}, "
            f"{validated_config.levels} levels, {len(signal)} points"
        )
        
        # Pad signal to power of 2 for efficient DWT
        padded_signal, original_length = self._pad_signal(signal)
        
        # Apply multi-level DWT
        coeffs = pywt.wavedec(
            padded_signal,
            validated_config.wavelet,
            level=validated_config.levels,
            mode=validated_config.mode
        )
        
        # Validate reconstruction
        self._validate_reconstruction(
            coeffs,
            padded_signal,
            validated_config.wavelet,
            validated_config.mode
        )
        
        # Initialize features
        features = {}
        
        # Approximation coefficients (cA)
        approx_coeffs = coeffs[0]
        features['wavelet_approx'] = self._resize_coeffs(
            approx_coeffs,
            original_length
        )
        
        # Detail coefficients (cD) for each level
        for i, detail_coeffs in enumerate(coeffs[1:], 1):
            # Resize to original length
            resized_detail = self._resize_coeffs(detail_coeffs, original_length)
            features[f'wavelet_detail_L{i}'] = resized_detail
            
            # Energy features
            if validated_config.include_energy:
                energy = self._calculate_energy(detail_coeffs)
                # Scalar value, broadcast to all rows
                features[f'wavelet_energy_L{i}'] = energy
            
            # Entropy features
            if validated_config.include_entropy:
                entropy = self._calculate_entropy(detail_coeffs)
                # Scalar value, broadcast to all rows
                features[f'wavelet_entropy_L{i}'] = entropy
        
        # Create DataFrame (align with original non-NaN indices)
        original_index = df[validated_config.target_column].dropna().index
        result = pd.DataFrame(features, index=original_index)
        
        # Reindex to match full DataFrame
        result = result.reindex(df.index)
        
        logger.info(f"Generated {len(result.columns)} wavelet features")
        return result
    
    def get_feature_names(self, config: Dict[str, Any]) -> List[str]:
        """Get feature names."""
        validated_config = WaveletFeaturesConfig(**config)
        
        names = ['wavelet_approx']
        
        for i in range(1, validated_config.levels + 1):
            names.append(f'wavelet_detail_L{i}')
            
            if validated_config.include_energy:
                names.append(f'wavelet_energy_L{i}')
            
            if validated_config.include_entropy:
                names.append(f'wavelet_entropy_L{i}')
        
        return names
    
    def _pad_signal(self, signal: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Pad signal to next power of 2 for efficient DWT.
        
        Args:
            signal: Input signal
        
        Returns:
            Tuple of (padded_signal, original_length)
        """
        original_length = len(signal)
        padded_length = 2 ** int(np.ceil(np.log2(original_length)))
        
        if padded_length > original_length:
            padded_signal = np.pad(
                signal,
                (0, padded_length - original_length),
                mode='edge'
            )
            logger.debug(
                f"Padded signal from {original_length} to {padded_length}"
            )
        else:
            padded_signal = signal
        
        return padded_signal, original_length
    
    def _resize_coeffs(
        self,
        coeffs: np.ndarray,
        target_length: int
    ) -> np.ndarray:
        """
        Resize coefficients to target length using interpolation.
        
        Args:
            coeffs: Wavelet coefficients
            target_length: Target length
        
        Returns:
            Resized coefficients
        """
        if len(coeffs) == target_length:
            return coeffs
        
        # Use linear interpolation
        x_old = np.linspace(0, 1, len(coeffs))
        x_new = np.linspace(0, 1, target_length)
        resized = np.interp(x_new, x_old, coeffs)
        
        return resized
    
    def _calculate_energy(self, coeffs: np.ndarray) -> float:
        """
        Calculate energy of wavelet coefficients.
        
        Energy = sum of squared coefficients
        
        Args:
            coeffs: Wavelet coefficients
        
        Returns:
            Energy value
        """
        return float(np.sum(coeffs ** 2))
    
    def _calculate_entropy(self, coeffs: np.ndarray) -> float:
        """
        Calculate Shannon entropy of wavelet coefficients.
        
        Measures complexity/randomness of signal at this scale.
        
        Args:
            coeffs: Wavelet coefficients
        
        Returns:
            Entropy value
        """
        # Calculate energy distribution
        energy = coeffs ** 2
        total_energy = np.sum(energy)
        
        if total_energy == 0:
            return 0.0
        
        # Probability distribution
        prob = energy / total_energy
        
        # Remove zeros to avoid log(0)
        prob = prob[prob > 0]
        
        # Shannon entropy
        entropy = -np.sum(prob * np.log2(prob))
        
        return float(entropy)
    
    def _validate_reconstruction(
        self,
        coeffs: List[np.ndarray],
        original_signal: np.ndarray,
        wavelet: str,
        mode: str
    ) -> None:
        """
        Validate that inverse DWT reconstructs original signal.
        
        Args:
            coeffs: Wavelet coefficients
            original_signal: Original signal
            wavelet: Wavelet family
            mode: Extension mode
        
        Raises:
            Warning if reconstruction error exceeds threshold
        """
        # Reconstruct signal
        reconstructed = pywt.waverec(coeffs, wavelet, mode=mode)
        
        # Trim to original length
        reconstructed = reconstructed[:len(original_signal)]
        
        # Calculate RMSE
        rmse = np.sqrt(np.mean((original_signal - reconstructed) ** 2))
        relative_error = rmse / (np.std(original_signal) + 1e-10)
        
        logger.debug(f"Wavelet reconstruction RMSE: {rmse:.4f}")
        
        if relative_error > 0.01:  # 1% threshold
            logger.warning(
                f"High reconstruction error: {relative_error:.2%}. "
                f"DWT may not be accurately representing signal."
            )
```

---

## 🧪 Testing & Validation

### Unit Tests

```python
"""Tests for wavelet features plugin."""
import pytest
import pandas as pd
import numpy as np
import pywt

from src.features.plugins.wavelet import WaveletFeaturePlugin, WaveletFeaturesConfig


@pytest.fixture
def plugin():
    """Create plugin instance."""
    return WaveletFeaturePlugin()


@pytest.fixture
def sample_df():
    """Create sample with known frequency components."""
    dates = pd.date_range("2024-01-01", periods=1024, freq="h")
    
    # Generate signal with multiple frequencies
    t = np.arange(1024)
    signal = (
        10 * np.sin(2 * np.pi * t / 128) +      # Low frequency
        5 * np.sin(2 * np.pi * t / 32) +        # Medium frequency
        2 * np.sin(2 * np.pi * t / 8) +         # High frequency
        np.random.randn(1024) * 0.5             # Noise
    )
    
    return pd.DataFrame({"carga": signal}, index=dates)


def test_basic_wavelet_decomposition(plugin, sample_df):
    """Test basic DWT decomposition."""
    config = {
        "target_column": "carga",
        "wavelet": "db4",
        "levels": 3
    }
    
    result = plugin.generate_features(sample_df, config)
    
    assert "wavelet_approx" in result.columns
    assert "wavelet_detail_L1" in result.columns
    assert "wavelet_detail_L2" in result.columns
    assert "wavelet_detail_L3" in result.columns


def test_reconstruction_accuracy(plugin, sample_df):
    """Test that wavelet reconstruction is accurate."""
    signal = sample_df["carga"].values
    
    # Manual reconstruction test
    coeffs = pywt.wavedec(signal, 'db4', level=4)
    reconstructed = pywt.waverec(coeffs, 'db4')
    reconstructed = reconstructed[:len(signal)]
    
    # RMSE should be very small
    rmse = np.sqrt(np.mean((signal - reconstructed) ** 2))
    relative_error = rmse / np.std(signal)
    
    assert relative_error < 0.01  # Less than 1% error


def test_multiple_wavelets(plugin, sample_df):
    """Test different wavelet families."""
    wavelets = ['db4', 'haar', 'coif2', 'sym4']
    
    for wavelet in wavelets:
        config = {
            "target_column": "carga",
            "wavelet": wavelet,
            "levels": 3
        }
        
        result = plugin.generate_features(sample_df, config)
        assert len(result.columns) > 0


def test_energy_calculation(plugin, sample_df):
    """Test wavelet energy features."""
    config = {
        "target_column": "carga",
        "wavelet": "db4",
        "levels": 3,
        "include_energy": True
    }
    
    result = plugin.generate_features(sample_df, config)
    
    # Energy features should exist
    assert "wavelet_energy_L1" in result.columns
    assert "wavelet_energy_L2" in result.columns
    
    # Energy should be positive
    assert result["wavelet_energy_L1"].iloc[0] > 0


def test_entropy_calculation(plugin, sample_df):
    """Test wavelet entropy features."""
    config = {
        "target_column": "carga",
        "wavelet": "db4",
        "levels": 3,
        "include_entropy": True
    }
    
    result = plugin.generate_features(sample_df, config)
    
    # Entropy features should exist
    assert "wavelet_entropy_L1" in result.columns
    
    # Entropy should be positive
    assert result["wavelet_entropy_L1"].iloc[0] > 0


def test_coefficient_resizing(plugin):
    """Test coefficient resizing maintains temporal alignment."""
    # Create small coefficient array
    coeffs = np.array([1.0, 2.0, 3.0, 4.0])
    
    # Resize to larger length
    resized = plugin._resize_coeffs(coeffs, 10)
    
    assert len(resized) == 10
    # First and last values should be approximately preserved
    assert abs(resized[0] - coeffs[0]) < 0.5
    assert abs(resized[-1] - coeffs[-1]) < 0.5


def test_feature_names(plugin):
    """Test feature name generation."""
    config = {
        "levels": 3,
        "include_energy": True,
        "include_entropy": True
    }
    
    names = plugin.get_feature_names(config)
    
    assert "wavelet_approx" in names
    assert "wavelet_detail_L1" in names
    assert "wavelet_energy_L1" in names
    assert "wavelet_entropy_L1" in names


def test_config_validation():
    """Test configuration validation."""
    # Invalid wavelet
    with pytest.raises(ValueError, match="Invalid wavelet"):
        WaveletFeaturesConfig(wavelet="invalid_wavelet")
    
    # Invalid levels
    with pytest.raises(ValueError, match="Levels must be 1-6"):
        WaveletFeaturesConfig(levels=0)
    
    with pytest.raises(ValueError, match="Levels must be 1-6"):
        WaveletFeaturesConfig(levels=7)


def test_signal_padding(plugin):
    """Test signal padding to power of 2."""
    signal = np.random.randn(1000)
    
    padded, original_len = plugin._pad_signal(signal)
    
    assert original_len == 1000
    assert len(padded) == 1024  # Next power of 2
    assert np.log2(len(padded)) == int(np.log2(len(padded)))


def test_frequency_band_separation(plugin):
    """Test that different levels capture different frequencies."""
    # Create signal with distinct frequency components
    t = np.arange(1024)
    low_freq = 10 * np.sin(2 * np.pi * t / 256)  # Very low
    high_freq = 2 * np.sin(2 * np.pi * t / 4)    # Very high
    
    signal = low_freq + high_freq
    df = pd.DataFrame({"carga": signal})
    
    config = {
        "target_column": "carga",
        "wavelet": "db4",
        "levels": 5
    }
    
    result = plugin.generate_features(df, config)
    
    # High level details should have more energy from high frequencies
    # Approximation should capture low frequencies
    energy_L1 = result["wavelet_energy_L1"].iloc[0]
    approx = result["wavelet_approx"].values
    
    # High frequency should dominate in detail
    assert energy_L1 > 0
    
    # Approximation should be smooth (like low frequency)
    approx_smoothness = np.var(np.diff(approx))
    signal_roughness = np.var(np.diff(signal))
    assert approx_smoothness < signal_roughness


def test_performance_benchmark(plugin):
    """Test performance with large dataset."""
    import time
    
    # 1 year of hourly data
    dates = pd.date_range("2024-01-01", periods=8760, freq="h")
    df = pd.DataFrame({
        "carga": np.random.randn(8760) + 1000
    }, index=dates)
    
    config = {
        "target_column": "carga",
        "wavelet": "db4",
        "levels": 4
    }
    
    start = time.time()
    result = plugin.generate_features(df, config)
    elapsed = time.time() - start
    
    # Should complete in <10 seconds
    assert elapsed < 10.0
    assert len(result) == len(df)
```

---

## 📝 Technical Notes

### Wavelet Transform Theory
- **DWT**: Decomposes signal into approximation (low-freq) and details (high-freq)
- **Multi-Level**: Each level halves the frequency range
- **Perfect Reconstruction**: Original signal can be exactly recovered
- **Frequency Bands**: Level 1 = highest frequencies, Level N = lowest

### Wavelet Selection
- **db4 (Daubechies)**: Good general-purpose wavelet, asymmetric
- **haar**: Simplest, good for discontinuities and sharp changes
- **coif2 (Coiflet)**: Symmetric, good for smooth signals
- **sym4 (Symlet)**: Near-symmetric version of Daubechies

### Use Cases
- Multi-scale pattern detection
- Noise reduction at specific frequency bands
- Feature extraction for time series classification
- Capturing both trend and oscillatory components

---

## 🔗 Dependencies

**Depends On:**
- PC-012-02A: Plugin Architecture Foundation
- PC-018-02B: LOESS Smoothing Plugin (AdvancedFeaturePlugin interface)

**External Dependencies:**
- `pywt>=1.4.0` (PyWavelets for DWT)

---

## 📊 Definition of Done

- [ ] All acceptance criteria met
- [ ] WaveletFeaturePlugin implemented
- [ ] DWT decomposition working for multiple wavelets
- [ ] Reconstruction validation passing (<1% error)
- [ ] Energy and entropy calculations correct
- [ ] Coefficient resizing maintains alignment
- [ ] Unit tests pass with >90% coverage
- [ ] Performance benchmark met (<10s for 1 year)
- [ ] Mathematical validation complete
- [ ] Code reviewed and approved

---

## 🔄 Change Log

| Date | Version | Changes | Author |
|------|---------|---------|--------|
| 2025-11-18 | 1.0.0 | Initial ticket created | System |

---

**Previous:** [PC-018-02B: LOESS Smoothing Plugin](PC-018-02B-loess-smoothing-plugin.md)  
**Next:** [PC-020-02B: BLF Strategy Plugin](PC-020-02B-blf-strategy-plugin.md)
