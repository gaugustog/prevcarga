"""Tests for feature importance tracker."""

from pathlib import Path

import pandas as pd
import pytest

from src.features.performance.importance_tracker import FeatureImportanceTracker


@pytest.fixture
def storage_path(tmp_path):
    """Create temporary storage path."""
    return tmp_path / "importance.parquet"


def test_record_and_retrieve_importance(storage_path):
    """Test recording and retrieving importance."""
    tracker = FeatureImportanceTracker(storage_path=storage_path)

    importance = {"feature1": 0.5, "feature2": 0.3, "feature3": 0.2}

    tracker.record_importance(
        feature_importance=importance,
        model_id="model1",
        importance_type="gain",
    )

    # Get aggregated importance
    agg = tracker.get_aggregated_importance()

    assert len(agg) == 3
    assert agg.iloc[0]["feature_name"] == "feature1"
    assert agg.iloc[0]["aggregated_importance"] == 0.5


def test_aggregate_multiple_models(storage_path):
    """Test aggregating importance across models."""
    tracker = FeatureImportanceTracker(storage_path=storage_path)

    # Record from model 1
    tracker.record_importance(
        feature_importance={"feature1": 0.5, "feature2": 0.3},
        model_id="model1",
    )

    # Record from model 2
    tracker.record_importance(
        feature_importance={"feature1": 0.7, "feature2": 0.2},
        model_id="model2",
    )

    # Aggregate (mean)
    agg = tracker.get_aggregated_importance(method="mean")

    # feature1 should have mean of 0.6
    feature1_importance = agg[agg["feature_name"] == "feature1"][
        "aggregated_importance"
    ].values[0]
    assert abs(feature1_importance - 0.6) < 0.01


def test_get_top_features(storage_path):
    """Test getting top features."""
    tracker = FeatureImportanceTracker(storage_path=storage_path)

    tracker.record_importance(
        feature_importance={
            "feature1": 0.5,
            "feature2": 0.3,
            "feature3": 0.2,
            "feature4": 0.1,
        },
        model_id="model1",
    )

    top_features = tracker.get_top_features(n=2)

    assert len(top_features) == 2
    assert top_features[0] == "feature1"
    assert top_features[1] == "feature2"


def test_persistence(storage_path):
    """Test importance data persists."""
    # First tracker instance
    tracker1 = FeatureImportanceTracker(storage_path=storage_path)
    tracker1.record_importance(
        feature_importance={"feature1": 0.5}, model_id="model1"
    )

    # Create new instance (simulating restart)
    tracker2 = FeatureImportanceTracker(storage_path=storage_path)

    # Should load previous data
    agg = tracker2.get_aggregated_importance()
    assert len(agg) == 1
    assert agg.iloc[0]["feature_name"] == "feature1"


def test_importance_trends(storage_path):
    """Test importance trend tracking."""
    tracker = FeatureImportanceTracker(storage_path=storage_path)

    # Record multiple times
    tracker.record_importance(
        feature_importance={"feature1": 0.5}, model_id="model1"
    )

    tracker.record_importance(
        feature_importance={"feature1": 0.6}, model_id="model2"
    )

    trend = tracker.get_importance_trends("feature1")

    assert len(trend) == 2
    assert trend.iloc[0]["importance_score"] == 0.5
    assert trend.iloc[1]["importance_score"] == 0.6


def test_different_importance_types(storage_path):
    """Test tracking different importance types."""
    tracker = FeatureImportanceTracker(storage_path=storage_path)

    # Record gain importance
    tracker.record_importance(
        feature_importance={"feature1": 0.5},
        model_id="model1",
        importance_type="gain",
    )

    # Record permutation importance
    tracker.record_importance(
        feature_importance={"feature1": 0.3},
        model_id="model2",
        importance_type="permutation",
    )

    # Get aggregated by type
    agg_gain = tracker.get_aggregated_importance(importance_type="gain")
    agg_perm = tracker.get_aggregated_importance(
        importance_type="permutation"
    )

    assert len(agg_gain) == 1
    assert agg_gain.iloc[0]["aggregated_importance"] == 0.5

    assert len(agg_perm) == 1
    assert agg_perm.iloc[0]["aggregated_importance"] == 0.3


def test_aggregate_with_metadata(storage_path):
    """Test recording with metadata."""
    tracker = FeatureImportanceTracker(storage_path=storage_path)

    tracker.record_importance(
        feature_importance={"feature1": 0.5},
        model_id="model1",
        metadata={"horizon": 1, "area": "SE"},
    )

    # Check metadata is stored
    assert len(tracker.importance_records) == 1
    assert tracker.importance_records.iloc[0]["horizon"] == 1
    assert tracker.importance_records.iloc[0]["area"] == "SE"


def test_aggregation_methods(storage_path):
    """Test different aggregation methods."""
    tracker = FeatureImportanceTracker(storage_path=storage_path)

    # Record multiple values
    for score in [0.3, 0.5, 0.4]:
        tracker.record_importance(
            feature_importance={"feature1": score},
            model_id=f"model_{score}",
        )

    # Mean
    agg_mean = tracker.get_aggregated_importance(method="mean")
    assert abs(agg_mean.iloc[0]["aggregated_importance"] - 0.4) < 0.01

    # Median
    agg_median = tracker.get_aggregated_importance(method="median")
    assert abs(agg_median.iloc[0]["aggregated_importance"] - 0.4) < 0.01

    # Max
    agg_max = tracker.get_aggregated_importance(method="max")
    assert abs(agg_max.iloc[0]["aggregated_importance"] - 0.5) < 0.01


def test_feature_stability(storage_path):
    """Test feature stability analysis."""
    tracker = FeatureImportanceTracker(storage_path=storage_path)

    # Feature with stable importance
    for _ in range(5):
        tracker.record_importance(
            feature_importance={"stable_feature": 0.5},
            model_id=f"model_{_}",
        )

    # Feature with volatile importance
    for i, score in enumerate([0.1, 0.9, 0.2, 0.8, 0.3]):
        tracker.record_importance(
            feature_importance={"volatile_feature": score},
            model_id=f"model_vol_{i}",
        )

    stability = tracker.get_feature_stability()

    # Stable feature should have lower coefficient of variation
    stable_cv = stability[stability["feature_name"] == "stable_feature"][
        "cv"
    ].values[0]
    volatile_cv = stability[stability["feature_name"] == "volatile_feature"][
        "cv"
    ].values[0]

    assert stable_cv < volatile_cv


def test_summary_report(storage_path):
    """Test summary report generation."""
    tracker = FeatureImportanceTracker(storage_path=storage_path)

    tracker.record_importance(
        feature_importance={"feature1": 0.5, "feature2": 0.3},
        model_id="model1",
    )

    report = tracker.get_summary_report()

    assert "FEATURE IMPORTANCE SUMMARY" in report
    assert "Total records: 2" in report
    assert "feature1" in report


def test_empty_tracker(storage_path):
    """Test tracker with no data."""
    tracker = FeatureImportanceTracker(storage_path=storage_path)

    agg = tracker.get_aggregated_importance()
    assert len(agg) == 0

    top_features = tracker.get_top_features(n=10)
    assert len(top_features) == 0

    trend = tracker.get_importance_trends("nonexistent")
    assert len(trend) == 0


def test_clear_tracker(storage_path):
    """Test clearing tracker data."""
    tracker = FeatureImportanceTracker(storage_path=storage_path)

    tracker.record_importance(
        feature_importance={"feature1": 0.5}, model_id="model1"
    )

    tracker.clear()

    assert len(tracker.importance_records) == 0
    assert not storage_path.exists()


def test_top_k_filtering(storage_path):
    """Test top-k filtering in aggregation."""
    tracker = FeatureImportanceTracker(storage_path=storage_path)

    # Record many features
    importance = {f"feature_{i}": 1.0 / (i + 1) for i in range(20)}

    tracker.record_importance(
        feature_importance=importance, model_id="model1"
    )

    # Get top 5
    agg = tracker.get_aggregated_importance(top_k=5)

    assert len(agg) == 5

    # Should be sorted by importance
    assert agg.iloc[0]["aggregated_importance"] >= agg.iloc[-1][
        "aggregated_importance"
    ]
