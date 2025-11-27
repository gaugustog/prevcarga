"""End-to-end forecasting model implementations.

This package provides complete model implementations for load forecasting,
including LightGBM, Random Forest, and BLF predictor for intraday corrections.

Example:
    ```python
    from src.models.end_to_end import LGBMModel, RandomForestModel, BLFPredictor
    from src.models.config import LGBMConfig, RandomForestConfig, BLFConfig

    # Create and train LGBM model
    model = LGBMModel()
    config = LGBMConfig()
    model.fit(X_train, y_train, config.model_dump())

    # Create and train Random Forest model
    rf_model = RandomForestModel()
    rf_config = RandomForestConfig()
    rf_model.fit(X_train, y_train, rf_config.model_dump())

    # Create BLF predictor for intraday corrections
    blf_config = BLFConfig()
    blf = BLFPredictor(model, config=blf_config.model_dump())

    # Make predictions
    predictions = model.predict(X_test, horizons=[0, 1])
    corrected_predictions = blf.predict_corrected(X_test, horizons=[0, 1])
    ```
"""

from src.models.end_to_end.blf_predictor import BLFPredictor
from src.models.end_to_end.lgbm_model import LGBMModel
from src.models.end_to_end.random_forest import RandomForestModel

__all__ = [
    "BLFPredictor",
    "LGBMModel",
    "RandomForestModel",
]
