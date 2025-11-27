"""Model configuration schemas.

This package provides Pydantic configuration models for different forecasting
model types in the PrevCarga system.

Example:
    ```python
    from src.models.config import LGBMConfig, RandomForestConfig, BLFConfig, SVMProfileConfig

    # Create configuration
    config = LGBMConfig(
        n_estimators=500,
        learning_rate=0.1,
        optimize_hyperparams=True
    )

    rf_config = RandomForestConfig(
        n_estimators=200,
        max_features_per_horizon=50
    )

    blf_config = BLFConfig(
        correction_window_hours=6,
        min_observations=3
    )

    svm_config = SVMProfileConfig(
        kernel="rbf",
        optimize_hyperparams=True,
        cv_folds=5
    )
    ```
"""

from src.models.config.arima_config import ARIMAConfig
from src.models.config.blf_config import BLFConfig
from src.models.config.holtwinters_config import (
    HoltWintersConfig,
    HoltWintersProfileConfig,
)
from src.models.config.lgbm_config import LGBMConfig
from src.models.config.rf_config import RandomForestConfig
from src.models.config.svm_config import SVMProfileConfig

__all__ = [
    "ARIMAConfig",
    "BLFConfig",
    "HoltWintersConfig",
    "HoltWintersProfileConfig",
    "LGBMConfig",
    "RandomForestConfig",
    "SVMProfileConfig",
]
