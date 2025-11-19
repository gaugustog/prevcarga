# Epic-07A: Core Metrics & Analysis

**Epic ID:** Epic-07A  
**Epic Name:** Core Metrics & Analysis  
**Phase:** 7A  
**Duration:** 1 week (Week 19)  
**Dependencies:** Epic-05A (Base Combination), Epic-06A (Core Reconciliation)  
**Priority:** Medium  

---

## 🎯 Epic Overview

Implement the foundational metrics calculation and statistical analysis system that provides accurate performance assessment across all models, combinations, and reconciliation methods. This epic establishes the core evaluation capabilities required for understanding model performance across different dimensions (time periods, horizons, percentiles).

### Business Value
- **Performance Assessment:** Accurate calculation of standard forecasting metrics for all models
- **Statistical Analysis:** Deep understanding of error distributions and performance patterns
- **Operational Insights:** Time period and horizon-specific performance for operational planning
- **Data-Driven Decisions:** Statistical foundation for model selection and optimization

---

## 📋 User Stories

### **User Story 1: Core Metrics Calculation Engine**
**As a** system analyst  
**I want** accurate and efficient calculation of standard forecasting metrics  
**So that** I can assess model performance across all time series and horizons  

**Acceptance Criteria:**
- [ ] `MetricsCalculator` computes MAPE, MAE, RMSE with numerical stability
- [ ] Handles missing values, zero targets, and edge cases gracefully
- [ ] Supports weighted metrics by time series importance or load magnitude
- [ ] Calculates confidence intervals and statistical significance tests
- [ ] Performance optimized for 26 time series × 9 horizons × multiple periods

**Technical Requirements:**
- Core metrics: MAPE = mean(|actual - forecast|/|actual|) × 100
- Robust MAPE handling zero/near-zero values: symmetric MAPE alternative
- MAE = mean(|actual - forecast|), RMSE = sqrt(mean((actual - forecast)²))
- Weighted metrics: importance weights by area consumption levels
- Statistical tests: t-tests for significance, bootstrap confidence intervals

**Definition of Done:**
- [ ] `MetricsCalculator` class implemented with all core metrics
- [ ] Edge case handling tested and validated
- [ ] Performance benchmarks met (<5s for full evaluation)
- [ ] Statistical significance testing working
- [ ] Unit tests cover numerical edge cases

---

### **User Story 2: Percentile Analysis Framework**
**As a** forecasting analyst  
**I want** detailed percentile analysis of forecast errors  
**So that** I can understand error distribution and identify outlier performance  

**Acceptance Criteria:**
- [ ] `PercentileAnalyzer` calculates P5, P10, P25, P50, P75, P90, P95 percentiles
- [ ] Error distribution analysis with statistical tests for normality
- [ ] Outlier detection and flagging for extreme error periods
- [ ] Quantile regression analysis for conditional performance assessment
- [ ] Visualization-ready data structures for error distribution plots

**Technical Requirements:**
- Percentile calculation using numpy.percentile with interpolation methods
- Distribution fitting: Shapiro-Wilk test, Anderson-Darling test for normality
- Outlier detection: IQR method, modified Z-score, isolation forest
- Quantile regression: sklearn.linear_model.QuantileRegressor
- Data structures: pandas DataFrames with multi-index (area, horizon, percentile)

**Definition of Done:**
- [ ] `PercentileAnalyzer` class implemented with statistical tests
- [ ] Distribution analysis and outlier detection working
- [ ] Quantile regression providing conditional insights
- [ ] Visualization data structures complete
- [ ] Performance validated across all time series

---

### **User Story 3: Time Period Performance Breakdown**
**As a** grid operations analyst  
**I want** performance metrics segmented by time periods (peak/off-peak, seasons, days)  
**So that** I can understand model performance under different operational conditions  

**Acceptance Criteria:**
- [ ] `TimePeriodAnalyzer` segments performance by peak/off-peak hours
- [ ] Seasonal analysis (summer, winter, transition periods)
- [ ] Day-of-week and weekend/weekday performance breakdown
- [ ] Holiday and special event period analysis
- [ ] Comparative analysis showing performance differences across periods

**Technical Requirements:**
- Time period classification: peak (10-12h, 18-22h), off-peak (other hours)
- Seasonal definition: summer (Dec-Mar), winter (Jun-Sep), transition periods
- Holiday detection: integration with holiday calendar from Epic-01
- Statistical comparison: ANOVA tests for period differences
- Multi-dimensional analysis: period × area × horizon performance matrices

**Definition of Done:**
- [ ] `TimePeriodAnalyzer` class implemented with all period types
- [ ] Statistical comparison framework working
- [ ] Performance differences quantified and tested
- [ ] Integration with holiday calendar complete
- [ ] Multi-dimensional analysis validated

---

### **User Story 4: Horizon-Specific Performance Analysis**
**As a** ML engineer  
**I want** detailed performance analysis across forecast horizons (D+0 to D+8)  
**So that** I can understand how model accuracy degrades with forecast distance  

**Acceptance Criteria:**
- [ ] `HorizonAnalyzer` tracks performance degradation across D+0 to D+8
- [ ] Statistical modeling of accuracy decay with forecast horizon
- [ ] Comparison of horizon performance across different models
- [ ] Identification of models with superior long-term vs short-term performance
- [ ] Horizon-specific confidence intervals and prediction bounds

**Technical Requirements:**
- Horizon performance tracking: metrics by horizon with trend analysis
- Decay modeling: exponential, polynomial, or piecewise linear fits
- Model comparison: performance rankings by horizon
- Confidence bounds: horizon-specific prediction intervals
- Temporal stability: tracking how horizon performance changes over time

**Definition of Done:**
- [ ] `HorizonAnalyzer` class implemented with decay modeling
- [ ] Statistical horizon performance modeling working
- [ ] Model comparison across horizons complete
- [ ] Confidence interval calculations accurate
- [ ] Temporal stability analysis operational

---

## 🏗️ Technical Architecture

### Core Components

```python
# Metrics Calculation Engine
from dataclasses import dataclass
from typing import Dict, Optional, List
import numpy as np
import pandas as pd
from scipy import stats

@dataclass
class MetricsConfig:
    """Configuration for metrics calculation."""
    epsilon: float = 1e-8  # Small value for numerical stability
    use_symmetric_mape: bool = True  # Use sMAPE for zero handling
    confidence_level: float = 0.95  # Confidence level for intervals
    bootstrap_iterations: int = 1000  # Bootstrap iterations
    weighted_metrics: bool = False  # Enable weighted metrics
    
@dataclass
class MetricsResult:
    """Container for calculated metrics."""
    mape: float
    mae: float
    rmse: float
    mse: float
    r2: float
    confidence_interval: Dict[str, tuple]
    sample_size: int
    metadata: Dict[str, any]

class MetricsCalculator:
    """Core metrics calculation with robust error handling."""
    
    def __init__(self, config: MetricsConfig):
        self.config = config
        self.statistical_tests = StatisticalTests()
    
    def calculate_metrics(self, 
                         predictions: Dict[str, np.ndarray], 
                         actuals: Dict[str, np.ndarray],
                         weights: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, MetricsResult]:
        """
        Calculate comprehensive forecast metrics for all time series.
        
        Args:
            predictions: Dictionary mapping series_id to forecast arrays
            actuals: Dictionary mapping series_id to actual value arrays
            weights: Optional weights for weighted metrics
            
        Returns:
            Dictionary mapping series_id to MetricsResult objects
        """
        results = {}
        
        for series_id in predictions.keys():
            if series_id not in actuals:
                raise ValueError(f"Missing actuals for series {series_id}")
            
            pred = predictions[series_id]
            actual = actuals[series_id]
            weight = weights.get(series_id) if weights else None
            
            # Calculate individual metrics
            mape = self.calculate_mape_robust(actual, pred)
            mae = self.calculate_mae(actual, pred, weight)
            rmse = self.calculate_rmse(actual, pred, weight)
            mse = rmse ** 2
            r2 = self.calculate_r2(actual, pred)
            
            # Calculate confidence intervals
            ci = self.calculate_confidence_intervals(actual, pred)
            
            results[series_id] = MetricsResult(
                mape=mape,
                mae=mae,
                rmse=rmse,
                mse=mse,
                r2=r2,
                confidence_interval=ci,
                sample_size=len(actual),
                metadata={'series_id': series_id}
            )
        
        return results
    
    def calculate_mape_robust(self, 
                             actual: np.ndarray, 
                             forecast: np.ndarray) -> float:
        """
        MAPE calculation handling zero/near-zero values.
        
        Uses symmetric MAPE when enabled to handle zeros:
        sMAPE = 100 * mean(|actual - forecast| / ((|actual| + |forecast|) / 2))
        """
        # Remove NaN values
        mask = ~(np.isnan(actual) | np.isnan(forecast))
        actual_clean = actual[mask]
        forecast_clean = forecast[mask]
        
        if len(actual_clean) == 0:
            return np.nan
        
        if self.config.use_symmetric_mape:
            # Symmetric MAPE - handles zeros better
            denominator = (np.abs(actual_clean) + np.abs(forecast_clean)) / 2
            denominator = np.maximum(denominator, self.config.epsilon)
            mape = 100 * np.mean(np.abs(actual_clean - forecast_clean) / denominator)
        else:
            # Standard MAPE with epsilon protection
            denominator = np.maximum(np.abs(actual_clean), self.config.epsilon)
            mape = 100 * np.mean(np.abs(actual_clean - forecast_clean) / denominator)
        
        return float(mape)
    
    def calculate_mae(self, 
                     actual: np.ndarray, 
                     forecast: np.ndarray,
                     weights: Optional[np.ndarray] = None) -> float:
        """Calculate Mean Absolute Error with optional weighting."""
        mask = ~(np.isnan(actual) | np.isnan(forecast))
        actual_clean = actual[mask]
        forecast_clean = forecast[mask]
        
        if len(actual_clean) == 0:
            return np.nan
        
        errors = np.abs(actual_clean - forecast_clean)
        
        if weights is not None:
            weights_clean = weights[mask]
            mae = np.average(errors, weights=weights_clean)
        else:
            mae = np.mean(errors)
        
        return float(mae)
    
    def calculate_rmse(self, 
                      actual: np.ndarray, 
                      forecast: np.ndarray,
                      weights: Optional[np.ndarray] = None) -> float:
        """Calculate Root Mean Squared Error with optional weighting."""
        mask = ~(np.isnan(actual) | np.isnan(forecast))
        actual_clean = actual[mask]
        forecast_clean = forecast[mask]
        
        if len(actual_clean) == 0:
            return np.nan
        
        squared_errors = (actual_clean - forecast_clean) ** 2
        
        if weights is not None:
            weights_clean = weights[mask]
            mse = np.average(squared_errors, weights=weights_clean)
        else:
            mse = np.mean(squared_errors)
        
        return float(np.sqrt(mse))
    
    def calculate_r2(self, actual: np.ndarray, forecast: np.ndarray) -> float:
        """Calculate R² coefficient of determination."""
        mask = ~(np.isnan(actual) | np.isnan(forecast))
        actual_clean = actual[mask]
        forecast_clean = forecast[mask]
        
        if len(actual_clean) == 0:
            return np.nan
        
        ss_res = np.sum((actual_clean - forecast_clean) ** 2)
        ss_tot = np.sum((actual_clean - np.mean(actual_clean)) ** 2)
        
        if ss_tot < self.config.epsilon:
            return np.nan
        
        r2 = 1 - (ss_res / ss_tot)
        return float(r2)
    
    def calculate_confidence_intervals(self, 
                                      actual: np.ndarray, 
                                      forecast: np.ndarray) -> Dict[str, tuple]:
        """
        Calculate confidence intervals for metrics using bootstrap.
        
        Returns:
            Dictionary with metric names and (lower, upper) bounds
        """
        mask = ~(np.isnan(actual) | np.isnan(forecast))
        actual_clean = actual[mask]
        forecast_clean = forecast[mask]
        
        if len(actual_clean) < 10:  # Minimum sample size
            return {}
        
        n_iterations = self.config.bootstrap_iterations
        n_samples = len(actual_clean)
        
        mape_samples = []
        mae_samples = []
        rmse_samples = []
        
        for _ in range(n_iterations):
            # Bootstrap resampling
            indices = np.random.choice(n_samples, n_samples, replace=True)
            actual_boot = actual_clean[indices]
            forecast_boot = forecast_clean[indices]
            
            mape_samples.append(self.calculate_mape_robust(actual_boot, forecast_boot))
            mae_samples.append(self.calculate_mae(actual_boot, forecast_boot))
            rmse_samples.append(self.calculate_rmse(actual_boot, forecast_boot))
        
        alpha = 1 - self.config.confidence_level
        
        return {
            'mape': (np.percentile(mape_samples, alpha/2 * 100),
                    np.percentile(mape_samples, (1 - alpha/2) * 100)),
            'mae': (np.percentile(mae_samples, alpha/2 * 100),
                   np.percentile(mae_samples, (1 - alpha/2) * 100)),
            'rmse': (np.percentile(rmse_samples, alpha/2 * 100),
                    np.percentile(rmse_samples, (1 - alpha/2) * 100))
        }

class StatisticalTests:
    """Statistical significance testing for metric differences."""
    
    def paired_t_test(self, 
                     errors_a: np.ndarray, 
                     errors_b: np.ndarray,
                     alpha: float = 0.05) -> Dict[str, any]:
        """
        Perform paired t-test to compare two sets of errors.
        
        Returns:
            Dictionary with t_statistic, p_value, significant, and interpretation
        """
        # Remove NaN pairs
        mask = ~(np.isnan(errors_a) | np.isnan(errors_b))
        errors_a_clean = errors_a[mask]
        errors_b_clean = errors_b[mask]
        
        if len(errors_a_clean) < 2:
            return {'error': 'Insufficient samples'}
        
        t_stat, p_value = stats.ttest_rel(errors_a_clean, errors_b_clean)
        
        return {
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant': p_value < alpha,
            'interpretation': 'statistically significant' if p_value < alpha else 'not significant',
            'sample_size': len(errors_a_clean)
        }
    
    def wilcoxon_test(self, 
                     errors_a: np.ndarray, 
                     errors_b: np.ndarray,
                     alpha: float = 0.05) -> Dict[str, any]:
        """
        Perform Wilcoxon signed-rank test (non-parametric alternative to t-test).
        """
        mask = ~(np.isnan(errors_a) | np.isnan(errors_b))
        errors_a_clean = errors_a[mask]
        errors_b_clean = errors_b[mask]
        
        if len(errors_a_clean) < 10:
            return {'error': 'Insufficient samples (min 10 required)'}
        
        statistic, p_value = stats.wilcoxon(errors_a_clean, errors_b_clean)
        
        return {
            'statistic': float(statistic),
            'p_value': float(p_value),
            'significant': p_value < alpha,
            'interpretation': 'statistically significant' if p_value < alpha else 'not significant',
            'sample_size': len(errors_a_clean)
        }


# Percentile Analysis Framework
@dataclass
class DistributionAnalysis:
    """Results of error distribution analysis."""
    percentiles: Dict[int, float]  # P5, P10, ..., P95
    mean: float
    std: float
    skewness: float
    kurtosis: float
    normality_test: Dict[str, any]
    outlier_count: int
    outlier_indices: List[int]

@dataclass
class OutlierAnalysis:
    """Results of outlier detection."""
    outlier_indices: List[int]
    outlier_values: np.ndarray
    outlier_scores: np.ndarray
    threshold_used: float
    method: str

class PercentileAnalyzer:
    """Statistical analysis of forecast error distributions."""
    
    def __init__(self, percentiles: List[int] = [5, 10, 25, 50, 75, 90, 95]):
        self.percentiles = percentiles
    
    def analyze_error_distribution(self, 
                                   actual: np.ndarray,
                                   forecast: np.ndarray) -> DistributionAnalysis:
        """
        Comprehensive error distribution analysis.
        
        Args:
            actual: Actual values
            forecast: Forecast values
            
        Returns:
            DistributionAnalysis with percentiles and statistical tests
        """
        # Calculate errors
        errors = actual - forecast
        
        # Remove NaN values
        errors_clean = errors[~np.isnan(errors)]
        
        if len(errors_clean) < 3:
            raise ValueError("Insufficient data for distribution analysis")
        
        # Calculate percentiles
        percentile_values = {
            p: float(np.percentile(errors_clean, p))
            for p in self.percentiles
        }
        
        # Statistical moments
        mean_error = float(np.mean(errors_clean))
        std_error = float(np.std(errors_clean))
        skew = float(stats.skew(errors_clean))
        kurt = float(stats.kurtosis(errors_clean))
        
        # Normality tests
        normality = self._test_normality(errors_clean)
        
        # Outlier detection
        outlier_result = self.detect_outliers(errors_clean, method='iqr')
        
        return DistributionAnalysis(
            percentiles=percentile_values,
            mean=mean_error,
            std=std_error,
            skewness=skew,
            kurtosis=kurt,
            normality_test=normality,
            outlier_count=len(outlier_result.outlier_indices),
            outlier_indices=outlier_result.outlier_indices
        )
    
    def _test_normality(self, errors: np.ndarray) -> Dict[str, any]:
        """
        Test for normality using multiple methods.
        
        Returns:
            Dictionary with Shapiro-Wilk and Anderson-Darling test results
        """
        results = {}
        
        # Shapiro-Wilk test (n < 5000)
        if len(errors) < 5000:
            statistic, p_value = stats.shapiro(errors)
            results['shapiro_wilk'] = {
                'statistic': float(statistic),
                'p_value': float(p_value),
                'normal': p_value > 0.05
            }
        
        # Anderson-Darling test
        result = stats.anderson(errors, dist='norm')
        results['anderson_darling'] = {
            'statistic': float(result.statistic),
            'critical_values': result.critical_values.tolist(),
            'significance_levels': result.significance_level.tolist()
        }
        
        return results
    
    def detect_outliers(self, 
                       errors: np.ndarray, 
                       method: str = 'iqr',
                       threshold: float = 1.5) -> OutlierAnalysis:
        """
        Detect and classify forecast error outliers.
        
        Args:
            errors: Error array
            method: Detection method ('iqr', 'zscore', 'modified_zscore')
            threshold: Threshold for outlier detection
            
        Returns:
            OutlierAnalysis with detected outliers
        """
        if method == 'iqr':
            return self._detect_outliers_iqr(errors, threshold)
        elif method == 'zscore':
            return self._detect_outliers_zscore(errors, threshold)
        elif method == 'modified_zscore':
            return self._detect_outliers_modified_zscore(errors, threshold)
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")
    
    def _detect_outliers_iqr(self, 
                            errors: np.ndarray, 
                            threshold: float = 1.5) -> OutlierAnalysis:
        """Interquartile Range (IQR) method for outlier detection."""
        q1 = np.percentile(errors, 25)
        q3 = np.percentile(errors, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        
        outlier_mask = (errors < lower_bound) | (errors > upper_bound)
        outlier_indices = np.where(outlier_mask)[0].tolist()
        outlier_values = errors[outlier_mask]
        
        # Calculate outlier scores (distance from bounds)
        scores = np.zeros_like(errors)
        scores[errors < lower_bound] = (lower_bound - errors[errors < lower_bound]) / iqr
        scores[errors > upper_bound] = (errors[errors > upper_bound] - upper_bound) / iqr
        
        return OutlierAnalysis(
            outlier_indices=outlier_indices,
            outlier_values=outlier_values,
            outlier_scores=scores[outlier_mask],
            threshold_used=threshold,
            method='iqr'
        )
    
    def _detect_outliers_zscore(self, 
                               errors: np.ndarray, 
                               threshold: float = 3.0) -> OutlierAnalysis:
        """Z-score method for outlier detection."""
        mean = np.mean(errors)
        std = np.std(errors)
        
        z_scores = np.abs((errors - mean) / std)
        outlier_mask = z_scores > threshold
        
        outlier_indices = np.where(outlier_mask)[0].tolist()
        outlier_values = errors[outlier_mask]
        
        return OutlierAnalysis(
            outlier_indices=outlier_indices,
            outlier_values=outlier_values,
            outlier_scores=z_scores[outlier_mask],
            threshold_used=threshold,
            method='zscore'
        )
    
    def _detect_outliers_modified_zscore(self, 
                                        errors: np.ndarray, 
                                        threshold: float = 3.5) -> OutlierAnalysis:
        """Modified Z-score using median absolute deviation (more robust)."""
        median = np.median(errors)
        mad = np.median(np.abs(errors - median))
        
        # Modified z-score
        modified_z_scores = 0.6745 * (errors - median) / mad
        outlier_mask = np.abs(modified_z_scores) > threshold
        
        outlier_indices = np.where(outlier_mask)[0].tolist()
        outlier_values = errors[outlier_mask]
        
        return OutlierAnalysis(
            outlier_indices=outlier_indices,
            outlier_values=outlier_values,
            outlier_scores=np.abs(modified_z_scores[outlier_mask]),
            threshold_used=threshold,
            method='modified_zscore'
        )


# Time Period Analysis
@dataclass
class PeriodAnalysis:
    """Results of time period performance analysis."""
    period_metrics: Dict[str, MetricsResult]
    period_comparison: Dict[str, any]
    statistical_tests: Dict[str, any]

class PeriodClassifier:
    """Classify timestamps into operational periods."""
    
    def __init__(self):
        self.peak_hours = [(10, 12), (18, 22)]  # Peak hour ranges
    
    def classify_period(self, timestamp: pd.Timestamp) -> str:
        """Classify a single timestamp into a period type."""
        hour = timestamp.hour
        
        # Check peak hours
        for start, end in self.peak_hours:
            if start <= hour < end:
                return 'peak'
        
        return 'off_peak'
    
    def classify_season(self, timestamp: pd.Timestamp) -> str:
        """Classify timestamp into season (Southern Hemisphere)."""
        month = timestamp.month
        
        if month in [12, 1, 2, 3]:  # Summer
            return 'summer'
        elif month in [6, 7, 8, 9]:  # Winter
            return 'winter'
        else:  # Transition
            return 'transition'
    
    def classify_day_type(self, 
                         timestamp: pd.Timestamp,
                         holiday_calendar: Optional[any] = None) -> str:
        """Classify day type (weekday, weekend, holiday)."""
        if holiday_calendar and holiday_calendar.is_holiday(timestamp):
            return 'holiday'
        
        if timestamp.dayofweek >= 5:  # Saturday = 5, Sunday = 6
            return 'weekend'
        
        return 'weekday'

class TimePeriodAnalyzer:
    """Performance analysis by time periods and conditions."""
    
    def __init__(self, holiday_calendar: Optional[any] = None):
        self.holiday_calendar = holiday_calendar
        self.period_classifier = PeriodClassifier()
        self.metrics_calculator = MetricsCalculator(MetricsConfig())
    
    def analyze_by_periods(self, 
                          predictions: Dict[str, np.ndarray],
                          actuals: Dict[str, np.ndarray],
                          timestamps: pd.DatetimeIndex) -> Dict[str, PeriodAnalysis]:
        """
        Analyze performance across different time periods.
        
        Args:
            predictions: Dictionary of predictions by series
            actuals: Dictionary of actuals by series
            timestamps: Timestamps for each prediction
            
        Returns:
            Dictionary mapping series_id to PeriodAnalysis
        """
        results = {}
        
        for series_id in predictions.keys():
            # Classify all timestamps
            peak_mask = np.array([
                self.period_classifier.classify_period(ts) == 'peak'
                for ts in timestamps
            ])
            
            season_classes = [
                self.period_classifier.classify_season(ts)
                for ts in timestamps
            ]
            
            day_type_classes = [
                self.period_classifier.classify_day_type(ts, self.holiday_calendar)
                for ts in timestamps
            ]
            
            # Calculate metrics by period
            period_metrics = {}
            
            # Peak vs Off-peak
            for period_type, mask in [('peak', peak_mask), ('off_peak', ~peak_mask)]:
                if mask.sum() > 0:
                    pred_subset = {series_id: predictions[series_id][mask]}
                    actual_subset = {series_id: actuals[series_id][mask]}
                    period_metrics[period_type] = self.metrics_calculator.calculate_metrics(
                        pred_subset, actual_subset
                    )[series_id]
            
            # By season
            for season in ['summer', 'winter', 'transition']:
                season_mask = np.array([s == season for s in season_classes])
                if season_mask.sum() > 0:
                    pred_subset = {series_id: predictions[series_id][season_mask]}
                    actual_subset = {series_id: actuals[series_id][season_mask]}
                    period_metrics[f'season_{season}'] = self.metrics_calculator.calculate_metrics(
                        pred_subset, actual_subset
                    )[series_id]
            
            # By day type
            for day_type in ['weekday', 'weekend', 'holiday']:
                day_mask = np.array([d == day_type for d in day_type_classes])
                if day_mask.sum() > 0:
                    pred_subset = {series_id: predictions[series_id][day_mask]}
                    actual_subset = {series_id: actuals[series_id][day_mask]}
                    period_metrics[f'day_{day_type}'] = self.metrics_calculator.calculate_metrics(
                        pred_subset, actual_subset
                    )[series_id]
            
            # Statistical comparison
            comparison = self._compare_periods(period_metrics)
            statistical_tests = self._perform_statistical_tests(
                predictions[series_id], actuals[series_id], peak_mask
            )
            
            results[series_id] = PeriodAnalysis(
                period_metrics=period_metrics,
                period_comparison=comparison,
                statistical_tests=statistical_tests
            )
        
        return results
    
    def _compare_periods(self, period_metrics: Dict[str, MetricsResult]) -> Dict[str, any]:
        """Compare performance across different periods."""
        comparison = {}
        
        # Compare peak vs off-peak
        if 'peak' in period_metrics and 'off_peak' in period_metrics:
            peak_mape = period_metrics['peak'].mape
            off_peak_mape = period_metrics['off_peak'].mape
            
            comparison['peak_vs_off_peak'] = {
                'peak_mape': peak_mape,
                'off_peak_mape': off_peak_mape,
                'difference': peak_mape - off_peak_mape,
                'relative_difference_pct': 100 * (peak_mape - off_peak_mape) / off_peak_mape
            }
        
        # Compare seasons
        season_mapes = {
            season: period_metrics[f'season_{season}'].mape
            for season in ['summer', 'winter', 'transition']
            if f'season_{season}' in period_metrics
        }
        
        if season_mapes:
            comparison['seasonal'] = {
                'by_season': season_mapes,
                'best_season': min(season_mapes, key=season_mapes.get),
                'worst_season': max(season_mapes, key=season_mapes.get),
                'range': max(season_mapes.values()) - min(season_mapes.values())
            }
        
        return comparison
    
    def _perform_statistical_tests(self,
                                   predictions: np.ndarray,
                                   actuals: np.ndarray,
                                   peak_mask: np.ndarray) -> Dict[str, any]:
        """Perform ANOVA tests for period differences."""
        errors = np.abs(actuals - predictions)
        
        peak_errors = errors[peak_mask]
        off_peak_errors = errors[~peak_mask]
        
        if len(peak_errors) > 0 and len(off_peak_errors) > 0:
            # T-test for peak vs off-peak
            t_stat, p_value = stats.ttest_ind(peak_errors, off_peak_errors)
            
            return {
                'peak_vs_off_peak_ttest': {
                    't_statistic': float(t_stat),
                    'p_value': float(p_value),
                    'significant': p_value < 0.05
                }
            }
        
        return {}


# Horizon Analysis
@dataclass
class HorizonAnalysis:
    """Results of horizon-specific performance analysis."""
    horizon_metrics: Dict[int, MetricsResult]
    decay_model: Dict[str, any]
    model_ranking: List[tuple]

class HorizonAnalyzer:
    """Horizon-specific performance analysis and decay modeling."""
    
    def __init__(self):
        self.metrics_calculator = MetricsCalculator(MetricsConfig())
    
    def analyze_horizon_performance(self,
                                   predictions_by_horizon: Dict[int, Dict[str, np.ndarray]],
                                   actuals_by_horizon: Dict[int, Dict[str, np.ndarray]]) -> Dict[str, HorizonAnalysis]:
        """
        Analyze performance degradation across forecast horizons.
        
        Args:
            predictions_by_horizon: Dict[horizon, Dict[series_id, predictions]]
            actuals_by_horizon: Dict[horizon, Dict[series_id, actuals]]
            
        Returns:
            Dictionary mapping series_id to HorizonAnalysis
        """
        # Get all series IDs
        all_series = set()
        for horizon_preds in predictions_by_horizon.values():
            all_series.update(horizon_preds.keys())
        
        results = {}
        
        for series_id in all_series:
            # Calculate metrics for each horizon
            horizon_metrics = {}
            
            for horizon in sorted(predictions_by_horizon.keys()):
                if series_id in predictions_by_horizon[horizon]:
                    pred = {series_id: predictions_by_horizon[horizon][series_id]}
                    actual = {series_id: actuals_by_horizon[horizon][series_id]}
                    
                    metrics = self.metrics_calculator.calculate_metrics(pred, actual)
                    horizon_metrics[horizon] = metrics[series_id]
            
            # Model accuracy decay
            decay_model = self.model_accuracy_decay(horizon_metrics)
            
            # Rank horizons by performance
            ranking = sorted(
                horizon_metrics.items(),
                key=lambda x: x[1].mape
            )
            
            results[series_id] = HorizonAnalysis(
                horizon_metrics=horizon_metrics,
                decay_model=decay_model,
                model_ranking=ranking
            )
        
        return results
    
    def model_accuracy_decay(self, horizon_metrics: Dict[int, MetricsResult]) -> Dict[str, any]:
        """
        Model forecast accuracy decay with horizon.
        
        Fits exponential, polynomial, and linear models to accuracy decay.
        """
        horizons = np.array(sorted(horizon_metrics.keys()))
        mapes = np.array([horizon_metrics[h].mape for h in horizons])
        
        if len(horizons) < 3:
            return {'error': 'Insufficient horizons for decay modeling'}
        
        decay_models = {}
        
        # Linear fit: mape = a * horizon + b
        linear_fit = np.polyfit(horizons, mapes, 1)
        linear_pred = np.polyval(linear_fit, horizons)
        linear_r2 = 1 - (np.sum((mapes - linear_pred)**2) / np.sum((mapes - np.mean(mapes))**2))
        
        decay_models['linear'] = {
            'coefficients': linear_fit.tolist(),
            'r2': float(linear_r2),
            'formula': f'MAPE = {linear_fit[0]:.4f} * horizon + {linear_fit[1]:.4f}'
        }
        
        # Polynomial fit (degree 2): mape = a * horizon^2 + b * horizon + c
        if len(horizons) >= 4:
            poly_fit = np.polyfit(horizons, mapes, 2)
            poly_pred = np.polyval(poly_fit, horizons)
            poly_r2 = 1 - (np.sum((mapes - poly_pred)**2) / np.sum((mapes - np.mean(mapes))**2))
            
            decay_models['polynomial'] = {
                'coefficients': poly_fit.tolist(),
                'r2': float(poly_r2),
                'degree': 2
            }
        
        # Exponential fit: mape = a * exp(b * horizon)
        try:
            log_mapes = np.log(mapes)
            exp_fit = np.polyfit(horizons, log_mapes, 1)
            exp_pred = np.exp(np.polyval(exp_fit, horizons))
            exp_r2 = 1 - (np.sum((mapes - exp_pred)**2) / np.sum((mapes - np.mean(mapes))**2))
            
            decay_models['exponential'] = {
                'a': float(np.exp(exp_fit[1])),
                'b': float(exp_fit[0]),
                'r2': float(exp_r2),
                'formula': f'MAPE = {np.exp(exp_fit[1]):.4f} * exp({exp_fit[0]:.4f} * horizon)'
            }
        except:
            pass  # Exponential fit may fail for some data
        
        # Select best model
        best_model = max(decay_models.items(), key=lambda x: x[1]['r2'])
        
        return {
            'models': decay_models,
            'best_model': best_model[0],
            'best_r2': best_model[1]['r2']
        }
    
    def compare_models_by_horizon(self,
                                 model_results: Dict[str, Dict[int, MetricsResult]]) -> Dict[int, List[tuple]]:
        """
        Compare different models' performance at each horizon.
        
        Args:
            model_results: Dict[model_name, Dict[horizon, MetricsResult]]
            
        Returns:
            Dict[horizon, List[(model_name, mape)]] sorted by performance
        """
        # Get all horizons
        all_horizons = set()
        for model_metrics in model_results.values():
            all_horizons.update(model_metrics.keys())
        
        comparison = {}
        
        for horizon in sorted(all_horizons):
            horizon_performance = []
            
            for model_name, model_metrics in model_results.items():
                if horizon in model_metrics:
                    mape = model_metrics[horizon].mape
                    horizon_performance.append((model_name, mape))
            
            # Sort by MAPE (lower is better)
            horizon_performance.sort(key=lambda x: x[1])
            comparison[horizon] = horizon_performance
        
        return comparison
```

### Integration Points

```python
# Example usage with Epic-05A/05B outputs
from epic_05a.base_combiner import WeightedAverageCombiner
from epic_05b.stacking_combiner import StackingCombiner
from epic_06a.mint_reconciler import MinTReconciler

# Calculate metrics for combination methods
calculator = MetricsCalculator(MetricsConfig())
percentile_analyzer = PercentileAnalyzer()
horizon_analyzer = HorizonAnalyzer()

# Evaluate base models
base_model_metrics = calculator.calculate_metrics(
    predictions=base_model_predictions,
    actuals=actuals
)

# Evaluate combinations
combination_metrics = calculator.calculate_metrics(
    predictions=combined_predictions,
    actuals=actuals
)

# Evaluate reconciled forecasts
reconciled_metrics = calculator.calculate_metrics(
    predictions=reconciled_predictions,
    actuals=actuals
)

# Analyze by horizon
horizon_analysis = horizon_analyzer.analyze_horizon_performance(
    predictions_by_horizon=predictions_by_horizon,
    actuals_by_horizon=actuals_by_horizon
)

# Percentile analysis
for series_id, predictions in combined_predictions.items():
    distribution = percentile_analyzer.analyze_error_distribution(
        actual=actuals[series_id],
        forecast=predictions
    )
```

---

## 🔧 Implementation Plan

### Week 1: Core Metrics and Analysis (Days 1-5)
- **Day 1:** Implement `MetricsCalculator` with MAPE, MAE, RMSE, R²
  - Robust MAPE with symmetric alternative
  - Weighted metrics support
  - Edge case handling (zeros, NaN values)
  
- **Day 2:** Implement statistical significance testing
  - Bootstrap confidence intervals
  - Paired t-tests
  - Wilcoxon signed-rank tests
  
- **Day 3:** Implement `PercentileAnalyzer`
  - Percentile calculations (P5-P95)
  - Distribution analysis (Shapiro-Wilk, Anderson-Darling)
  - Outlier detection (IQR, Z-score, modified Z-score)
  
- **Day 4:** Implement `TimePeriodAnalyzer`
  - Period classification (peak/off-peak, seasons, day types)
  - Integration with holiday calendar
  - Statistical comparison (ANOVA, t-tests)
  
- **Day 5:** Implement `HorizonAnalyzer`
  - Horizon-specific metrics calculation
  - Accuracy decay modeling (linear, polynomial, exponential)
  - Model comparison by horizon
  - Integration testing with Epic-05A/05B/06A outputs

---

## 📊 Success Metrics

### Performance Targets
- **Calculation Speed:** Full evaluation of 26 series × 9 horizons in <5 seconds
- **Memory Efficiency:** Peak memory usage <500MB for full evaluation
- **Statistical Accuracy:** Confidence intervals within ±2% of true values (validated via simulation)

### Quality Gates
- [ ] All metrics calculated correctly with proper edge case handling
- [ ] Bootstrap confidence intervals accurate and efficient
- [ ] Percentile analysis provides distribution insights
- [ ] Time period analysis reveals operational patterns
- [ ] Horizon decay models fit with R² > 0.8
- [ ] Statistical tests properly control Type I error (α = 0.05)

### Acceptance Criteria
- [ ] Integration with Epic-05A/05B combination outputs successful
- [ ] Integration with Epic-06A reconciliation outputs working
- [ ] All 4 user stories completed with comprehensive testing
- [ ] Performance benchmarks exceeded across all components
- [ ] 80%+ unit test coverage with edge cases
- [ ] Ready for Epic-07B (Monitoring & Reporting)

---

## 🧪 Testing Strategy

### Unit Tests
- Individual metric calculation accuracy (MAPE, MAE, RMSE, R²)
- Edge case handling (zeros, near-zeros, NaN values, single values)
- Bootstrap confidence interval convergence
- Statistical test implementations (t-test, Wilcoxon, ANOVA)
- Outlier detection methods (IQR, Z-score, modified Z-score)
- Period classification logic
- Horizon decay model fitting

### Integration Tests
- End-to-end metrics calculation pipeline
- Integration with Epic-04 model outputs
- Integration with Epic-05A/05B combination outputs
- Integration with Epic-06A reconciliation outputs
- Multi-dimensional analysis (area × horizon × period)

### Performance Tests
- Metrics calculation speed benchmarks
- Memory usage validation
- Bootstrap iteration performance
- Statistical test performance with large samples

### Validation Tests
- Statistical method accuracy validation (via simulation)
- Confidence interval coverage probability
- Outlier detection sensitivity analysis
- Decay model predictive accuracy

---

## 📚 Dependencies & Risks

### External Dependencies
- **Epic-05A Completion:** Base combination strategies for evaluation
- **Epic-06A Completion:** Core reconciliation for hierarchy metrics
- **Epic-01 Holiday Calendar:** For time period classification
- **Historical Data:** Sufficient actuals for statistical tests

### Python Dependencies
```
numpy >= 1.24.0          # Core numerical operations
pandas >= 2.0.0          # Data structures
scipy >= 1.10.0          # Statistical tests
scikit-learn >= 1.3.0    # Quantile regression (optional)
```

### Technical Risks & Mitigation
1. **Numerical Stability with Zeros:** Use symmetric MAPE and epsilon protection
2. **Bootstrap Performance:** Vectorize operations, limit iterations for large datasets
3. **Statistical Test Assumptions:** Provide both parametric and non-parametric alternatives
4. **Memory with Large Datasets:** Process series independently, stream data when possible

### Business Risks & Mitigation
1. **Metric Interpretation:** Provide clear documentation and interpretation guidelines
2. **Statistical Complexity:** Abstract complexity, provide simple summaries
3. **Performance Overhead:** Optimize for production use, make detailed analysis optional

---

## 🔄 Handoff Criteria

### Deliverables for Epic-07B
- [ ] Core metrics calculation engine working reliably
- [ ] Statistical analysis framework providing actionable insights
- [ ] Percentile analysis revealing error distribution patterns
- [ ] Time period analysis identifying operational performance patterns
- [ ] Horizon analysis tracking accuracy degradation
- [ ] All components tested and documented

### Documentation Requirements
- [ ] Metrics calculation methodology and formulas
- [ ] Statistical test descriptions and assumptions
- [ ] Edge case handling documentation
- [ ] API reference for all public classes
- [ ] Integration examples with previous epics

---

## 📈 Success Definition

Epic-07A is successful when:
1. **Metrics calculation system** accurately computes MAPE, MAE, RMSE, R² with proper edge case handling
2. **Statistical framework** provides confidence intervals and significance tests
3. **Percentile analysis** reveals error distribution patterns and outliers
4. **Time period analysis** identifies performance differences across operational conditions
5. **Horizon analysis** models accuracy decay and enables model comparison
6. **Performance benchmarks** consistently met (<5s for full evaluation)
7. **Foundation established** for Epic-07B monitoring and reporting

**Ready for Epic-07B when:** All core metrics working reliably, statistical analysis validated, integration with Epic-05/06 outputs successful, and comprehensive test coverage achieved.
