"""Demand mean forecasting models.

This package provides models for forecasting daily demand mean (average load),
which is used as the first stage in hierarchical forecasting approaches.

Example:
    ```python
    from src.models.demand_mean import ARIMADemandMeanModel, DemandProcessor
    from src.models.config import ARIMAConfig

    # Create and configure model
    model = ARIMADemandMeanModel()
    config = ARIMAConfig(seasonal=True, season_length=7)

    # Train model on daily data
    model.fit(X_daily, y_daily, config={"arima_config": config})

    # Generate forecasts
    predictions = model.predict(X_test, horizons=[0, 1, 2, 3])

    # Process demand data
    processor = DemandProcessor()
    clean_data = processor.process_demand_series(raw_data)
    ```
"""

from src.models.demand_mean.arima_model import ARIMADemandMeanModel
from src.models.demand_mean.demand_processor import DemandProcessor

__all__ = [
    "ARIMADemandMeanModel",
    "DemandProcessor",
]
