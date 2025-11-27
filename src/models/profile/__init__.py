"""Profile forecasting models.

This package provides models for semi-hourly load profile forecasting.
Profile models predict the shape of the load curve by forecasting profile
ratios (load / daily_demand_mean) for each semi-hourly period (0-47).

Example:
    ```python
    from src.models.profile import SVMProfileModel, ProfileNormalizer
    from src.models.config import SVMProfileConfig

    # Create model
    model = SVMProfileModel()
    config = SVMProfileConfig()

    # Train model
    model.fit(X_train, y_train, config={"svm_config": config})

    # Predict profile ratios
    predictions = model.predict(X_test, horizons=[0])

    # Normalize profiles to ensure energy conservation
    normalizer = ProfileNormalizer()
    normalized = normalizer.normalize_daily_profiles(predictions)
    ```
"""

from src.models.profile.profile_normalizer import ProfileNormalizer
from src.models.profile.svm_profile import SVMProfileModel

__all__ = [
    "ProfileNormalizer",
    "SVMProfileModel",
]
