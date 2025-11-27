"""Model combination module for ensemble forecasting.

This module provides the foundational infrastructure for combining predictions
from multiple forecasting models in the PrevCarga electric load forecasting
system.

The module includes:
- BaseCombiner: Abstract interface for model combination strategies
- CombinationResult: Result container with combined predictions and metadata
- CombinationMetadata: Tracking information about the combination process
- CombinerConfig: Configuration dataclass for combiners

Key Features:
- Standard combine() and fit() interface for all combiners
- Support for 26 time series (17 areas + 4 subsystems + 4 losses + 1 national)
- Support for 9 horizons (D+0 to D+8)
- Handles missing predictions gracefully
- Weight management and persistence
- Prediction interval support

Example:
    ```python
    from src.models.combination import BaseCombiner, CombinationResult

    class SimpleAverageCombiner(BaseCombiner):
        @property
        def name(self) -> str:
            return "simple_average"

        def combine(self, predictions, metadata=None):
            # Combine predictions with equal weights
            ...

        def fit(self, train_predictions, train_targets):
            # No fitting needed for simple average
            ...

    combiner = SimpleAverageCombiner()
    result = combiner.combine(predictions)
    ```
"""

from src.models.combination.averaging_combiner import SimpleAveragingCombiner
from src.models.combination.base_combiner import BaseCombiner
from src.models.combination.best_model_selector import BestModelSelector
from src.models.combination.context_aware import (
    ContextAnalysis,
    ContextAwareConfig,
    ContextAwareCombiner,
    ContextFeatureExtractor,
)
from src.models.combination.data_structures import (
    CombinationMetadata,
    CombinationResult,
    CombinerConfig,
)
from src.models.combination.diversity_analyzer import (
    DiversityAnalyzer,
    DiversityConfig,
    DiversityResult,
    EnsembleDiversityAnalyzer,
)
from src.models.combination.markov_chain_combiner import (
    MarkovChainCombiner,
    PerformanceState,
    PerformanceTracker,
)
from src.models.combination.performance_selector import (
    PerformanceBasedSelector,
    PerformanceSelectorConfig,
    RollingPerformanceTracker,
    SelectionDiagnostics,
    SelectionRecord,
)
from src.models.combination.stacking_combiner import StackingCombiner
from src.models.combination.weighted_voting import WeightedVotingCombiner

__all__ = [
    "BaseCombiner",
    "BestModelSelector",
    "CombinationMetadata",
    "CombinationResult",
    "CombinerConfig",
    "ContextAnalysis",
    "ContextAwareCombiner",
    "ContextAwareConfig",
    "ContextFeatureExtractor",
    "DiversityAnalyzer",
    "DiversityConfig",
    "DiversityResult",
    "EnsembleDiversityAnalyzer",  # Backwards compatibility
    "MarkovChainCombiner",
    "PerformanceBasedSelector",
    "PerformanceSelectorConfig",
    "PerformanceState",
    "PerformanceTracker",
    "RollingPerformanceTracker",
    "SelectionDiagnostics",
    "SelectionRecord",
    "SimpleAveragingCombiner",
    "StackingCombiner",
    "WeightedVotingCombiner",
]
