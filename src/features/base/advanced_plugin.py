"""Abstract base class for advanced feature engineering plugins.

This module defines the AdvancedFeaturePlugin abstract class that extends
BaseFeaturePlugin with additional methods for computational complexity
estimation, time estimation, and memory requirements.

Example:
    ```python
    from src.features.base import AdvancedFeaturePlugin
    from typing import Literal

    class WaveletPlugin(AdvancedFeaturePlugin):
        @property
        def name(self) -> str:
            return "wavelet_transform"

        @property
        def version(self) -> str:
            return "1.0.0"

        @property
        def computational_complexity(self) -> Literal["low", "medium", "high"]:
            return "high"

        def estimate_compute_time(self, data_size: int) -> float:
            # Estimate based on empirical measurements
            return data_size * 0.01  # 0.01 seconds per sample

        def generate_features(
            self,
            df: pd.DataFrame,
            config: dict[str, Any]
        ) -> pd.DataFrame:
            # Generate wavelet features
            ...
    ```
"""

from abc import abstractmethod
from typing import Any, Literal

from src.features.base.plugin import BaseFeaturePlugin


class AdvancedFeaturePlugin(BaseFeaturePlugin):
    """Abstract base class for advanced feature engineering plugins.

    This class extends BaseFeaturePlugin with additional capabilities for
    advanced transformations that have higher computational complexity,
    such as wavelet transforms, LOESS smoothing, and feature selection.

    Subclasses must implement:
        - All BaseFeaturePlugin abstract methods
        - computational_complexity: Complexity level indicator
        - estimate_compute_time: Time estimation for planning

    Optional methods to override:
        - supports_incremental_updates: Whether plugin supports incremental updates
        - get_memory_requirements: Memory requirement estimation

    Attributes:
        computational_complexity: Indicates if plugin is "low", "medium", or "high" complexity.
    """

    @property
    @abstractmethod
    def computational_complexity(self) -> Literal["low", "medium", "high"]:
        """Return the computational complexity level of this plugin.

        This helps the orchestrator decide on parallelization strategy
        and resource allocation.

        Returns:
            Complexity level: "low", "medium", or "high".
        """
        ...

    @abstractmethod
    def estimate_compute_time(self, data_size: int) -> float:
        """Estimate computation time for given data size.

        This method provides an estimate of how long the plugin will take
        to process data of the given size. Used for scheduling and
        progress reporting.

        Args:
            data_size: Number of samples/rows in the dataset.

        Returns:
            Estimated computation time in seconds.
        """
        ...

    def supports_incremental_updates(self) -> bool:
        """Check if plugin supports incremental feature updates.

        Incremental updates allow adding new data points without
        recomputing features for the entire dataset. This is useful
        for real-time forecasting scenarios.

        Returns:
            True if plugin supports incremental updates, False otherwise.
        """
        return False

    def get_memory_requirements(self, data_size: int) -> int:
        """Estimate memory requirements for processing.

        This method estimates the peak memory usage in megabytes
        for processing data of the given size. Used for resource
        planning and preventing OOM errors.

        Args:
            data_size: Number of samples/rows in the dataset.

        Returns:
            Estimated memory requirement in megabytes (MB).
        """
        # Default estimate: 0.1 MB per sample
        # This is a conservative estimate for most feature engineering operations
        return int(data_size * 0.1)

    def get_metadata(self) -> dict[str, Any]:
        """Return plugin metadata as a dictionary.

        Extends BaseFeaturePlugin metadata with advanced plugin information.

        Returns:
            Dictionary containing plugin metadata including:
                - All BaseFeaturePlugin metadata
                - computational_complexity: Plugin complexity level
                - supports_incremental: Whether incremental updates are supported
        """
        metadata = super().get_metadata()
        metadata.update(
            {
                "computational_complexity": self.computational_complexity,
                "supports_incremental": self.supports_incremental_updates(),
            }
        )
        return metadata
