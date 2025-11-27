"""Testing module for PrevCarga.

This package provides testing frameworks and utilities for integration testing,
end-to-end validation, and test data management.

Key Components:
- IntegrationTestFramework: Orchestrates integration tests across all components
- FixtureDataManager: Manages test data fixtures and factories
- IntegrationTestResult: Results from individual integration tests
- IntegrationValidationResult: Aggregated validation results

Example:
    ```python
    from src.testing import (
        IntegrationTestFramework,
        IntegrationTestConfig,
    )

    config = IntegrationTestConfig(
        test_data_path="/path/to/test/data",
        parallel_execution=True,
    )
    framework = IntegrationTestFramework(config)

    result = framework.validate_integration_completeness()
    print(f"Success rate: {result.get_success_rate():.1%}")
    ```
"""

from src.testing.integration_test_framework import (
    FixtureDataManager,
    IntegrationCategory,
    IntegrationTestConfig,
    IntegrationTestFramework,
    IntegrationTestResult,
    IntegrationValidationResult,
    SyntheticDataFactory,
)

# Aliases for backward compatibility
TestCategory = IntegrationCategory
TestDataFactory = SyntheticDataFactory
TestDataManager = FixtureDataManager

__all__ = [
    "IntegrationTestFramework",
    "IntegrationTestConfig",
    "IntegrationTestResult",
    "IntegrationValidationResult",
    "IntegrationCategory",
    "FixtureDataManager",
    "SyntheticDataFactory",
    # Aliases
    "TestCategory",
    "TestDataManager",
    "TestDataFactory",
]
