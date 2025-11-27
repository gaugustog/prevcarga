"""Feature importance tracking and aggregation.

This module provides utilities for tracking feature importance across
multiple models and training runs, enabling analysis of feature stability
and impact over time.

Example:
    ```python
    from src.features.performance.importance_tracker import FeatureImportanceTracker

    tracker = FeatureImportanceTracker()

    # Record importance from model
    tracker.record_importance(
        feature_importance={'feature1': 0.5, 'feature2': 0.3},
        model_id='lgbm_v1',
        metadata={'horizon': 1, 'area': 'SE'}
    )

    # Get aggregated importance
    agg_importance = tracker.get_aggregated_importance()
    top_features = tracker.get_top_features(n=10)
    ```
"""

from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class FeatureImportanceTracker:
    """Track and aggregate feature importance across multiple models.

    Maintains historical record of feature importance scores,
    enabling analysis of feature stability and impact over time.

    The tracker stores importance data persistently in Parquet format,
    allowing accumulation across multiple training runs and models.

    Attributes:
        storage_path: Path to persistent storage file.
        importance_records: DataFrame containing all importance records.

    Example:
        >>> tracker = FeatureImportanceTracker()
        >>>
        >>> # Record importance from model
        >>> tracker.record_importance(
        ...     feature_importance={'feature1': 0.5, 'feature2': 0.3},
        ...     model_id='lgbm_v1',
        ...     metadata={'horizon': 1, 'area': 'SE'}
        ... )
        >>>
        >>> # Get aggregated importance
        >>> agg_importance = tracker.get_aggregated_importance()
    """

    def __init__(self, storage_path: Optional[Path] = None):
        """Initialize importance tracker.

        Args:
            storage_path: Path to store importance data.
                If None, uses default path ".importance/feature_importance.parquet"
        """
        self.storage_path = storage_path or Path(
            ".importance/feature_importance.parquet"
        )
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing importance data
        self.importance_records = self._load_records()

        logger.info(
            f"Initialized FeatureImportanceTracker with "
            f"{len(self.importance_records)} records"
        )

    def record_importance(
        self,
        feature_importance: dict[str, float],
        model_id: str,
        importance_type: str = "gain",
        metadata: Optional[dict] = None,
    ) -> None:
        """Record feature importance from a model.

        Args:
            feature_importance: Dictionary mapping features to importance scores.
            model_id: Identifier for the model.
            importance_type: Type of importance (gain, split, permutation, shap).
            metadata: Additional metadata (horizon, area, hyperparameters, etc.).

        Example:
            >>> tracker.record_importance(
            ...     feature_importance={'lag_1': 0.5, 'hour': 0.3},
            ...     model_id='lgbm_d1_SE',
            ...     importance_type='gain',
            ...     metadata={'horizon': 1, 'area': 'SE'}
            ... )
        """
        timestamp = datetime.now()

        records = []
        for feature_name, importance_score in feature_importance.items():
            record = {
                "feature_name": feature_name,
                "importance_score": importance_score,
                "importance_type": importance_type,
                "model_id": model_id,
                "timestamp": timestamp,
                **(metadata or {}),
            }
            records.append(record)

        # Append to existing records
        new_df = pd.DataFrame(records)

        if self.importance_records.empty:
            self.importance_records = new_df
        else:
            self.importance_records = pd.concat(
                [self.importance_records, new_df], ignore_index=True
            )

        logger.info(
            f"Recorded importance for {len(feature_importance)} features "
            f"from {model_id}"
        )

        # Save to disk
        self._save_records()

    def get_aggregated_importance(
        self,
        method: str = "mean",
        importance_type: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> pd.DataFrame:
        """Get aggregated feature importance.

        Args:
            method: Aggregation method ("mean", "median", "max").
            importance_type: Filter by importance type.
            top_k: Return only top-k features.

        Returns:
            DataFrame with aggregated importance scores, sorted by importance.

        Example:
            >>> agg = tracker.get_aggregated_importance(method='mean', top_k=20)
            >>> print(agg.head())
        """
        if self.importance_records.empty:
            logger.warning("No importance records available")
            return pd.DataFrame()

        df = self.importance_records.copy()

        # Filter by importance type
        if importance_type:
            df = df[df["importance_type"] == importance_type]

        if df.empty:
            logger.warning(
                f"No records found for importance_type={importance_type}"
            )
            return pd.DataFrame()

        # Aggregate by feature
        if method == "mean":
            agg_df = df.groupby("feature_name")["importance_score"].mean()
        elif method == "median":
            agg_df = df.groupby("feature_name")["importance_score"].median()
        elif method == "max":
            agg_df = df.groupby("feature_name")["importance_score"].max()
        else:
            raise ValueError(
                f"Unknown aggregation method: {method}. "
                f"Supported: mean, median, max"
            )

        # Convert to DataFrame and sort
        result = agg_df.reset_index()
        result.columns = ["feature_name", "aggregated_importance"]
        result = result.sort_values("aggregated_importance", ascending=False)

        # Top-k filtering
        if top_k:
            result = result.head(top_k)

        logger.info(
            f"Aggregated importance for {len(result)} features using {method}"
        )

        return result

    def get_importance_trends(self, feature_name: str) -> pd.DataFrame:
        """Get importance trend for specific feature.

        Shows how importance of a feature has changed over time
        across different models.

        Args:
            feature_name: Feature to analyze.

        Returns:
            DataFrame with importance over time.

        Example:
            >>> trend = tracker.get_importance_trends('lag_1')
            >>> print(trend)
        """
        if self.importance_records.empty:
            return pd.DataFrame()

        trend = self.importance_records[
            self.importance_records["feature_name"] == feature_name
        ].sort_values("timestamp")

        return trend[["timestamp", "importance_score", "model_id"]]

    def get_top_features(
        self, n: int = 20, importance_type: Optional[str] = None
    ) -> list[str]:
        """Get list of top-n most important features.

        Args:
            n: Number of features to return.
            importance_type: Filter by importance type.

        Returns:
            List of feature names, ordered by importance.

        Example:
            >>> top_features = tracker.get_top_features(n=10)
            >>> print(top_features)
        """
        agg_importance = self.get_aggregated_importance(
            method="mean", importance_type=importance_type, top_k=n
        )

        if agg_importance.empty:
            return []

        return agg_importance["feature_name"].tolist()

    def get_feature_stability(
        self, top_k: Optional[int] = None
    ) -> pd.DataFrame:
        """Analyze feature importance stability.

        Calculates coefficient of variation (std/mean) for each feature's
        importance across models to identify stable vs volatile features.

        Args:
            top_k: Return only top-k features by mean importance.

        Returns:
            DataFrame with stability metrics.

        Example:
            >>> stability = tracker.get_feature_stability(top_k=20)
            >>> print(stability[['feature_name', 'cv', 'mean_importance']])
        """
        if self.importance_records.empty:
            logger.warning("No importance records available")
            return pd.DataFrame()

        df = self.importance_records.copy()

        # Calculate statistics per feature
        stats = df.groupby("feature_name")["importance_score"].agg(
            ["mean", "std", "count", "min", "max"]
        )

        # Coefficient of variation (lower = more stable)
        stats["cv"] = stats["std"] / stats["mean"]

        # Reset index and sort
        stats = stats.reset_index()
        stats = stats.sort_values("mean", ascending=False)

        if top_k:
            stats = stats.head(top_k)

        return stats

    def get_summary_report(self) -> str:
        """Generate human-readable summary report.

        Returns:
            Formatted report string.

        Example:
            >>> print(tracker.get_summary_report())
        """
        if self.importance_records.empty:
            return "No importance records available."

        n_records = len(self.importance_records)
        n_features = self.importance_records["feature_name"].nunique()
        n_models = self.importance_records["model_id"].nunique()

        top_features = self.get_top_features(n=10)

        lines = [
            "=" * 80,
            "FEATURE IMPORTANCE SUMMARY",
            "=" * 80,
            f"Total records: {n_records}",
            f"Unique features: {n_features}",
            f"Unique models: {n_models}",
            "",
            "Top 10 features (by mean importance):",
            "-" * 80,
        ]

        agg = self.get_aggregated_importance(method="mean", top_k=10)

        for idx, row in agg.iterrows():
            lines.append(
                f"  {idx + 1}. {row['feature_name']}: "
                f"{row['aggregated_importance']:.4f}"
            )

        lines.append("=" * 80)

        return "\n".join(lines)

    def _load_records(self) -> pd.DataFrame:
        """Load importance records from disk.

        Returns:
            DataFrame with historical records, or empty DataFrame.
        """
        if self.storage_path.exists():
            try:
                df = pd.read_parquet(self.storage_path)
                logger.info(
                    f"Loaded {len(df)} importance records from "
                    f"{self.storage_path}"
                )
                return df
            except Exception as e:
                logger.error(f"Failed to load importance records: {e}")

        return pd.DataFrame()

    def _save_records(self) -> None:
        """Save importance records to disk."""
        try:
            self.importance_records.to_parquet(
                self.storage_path, compression="gzip", index=False
            )
            logger.debug(f"Saved importance records to {self.storage_path}")
        except Exception as e:
            logger.error(f"Failed to save importance records: {e}")

    def clear(self) -> None:
        """Clear all importance records.

        Warning:
            This permanently deletes all historical importance data.
        """
        self.importance_records = pd.DataFrame()
        if self.storage_path.exists():
            self.storage_path.unlink()
        logger.warning("Cleared all importance records")
