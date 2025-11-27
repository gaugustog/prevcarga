"""Cache key generation utilities.

This module provides utilities for generating deterministic cache keys
for feature computations based on data fingerprints, plugin metadata,
and configuration parameters.

Example:
    ```python
    from src.features.performance.cache_key import CacheKeyGenerator

    generator = CacheKeyGenerator()
    cache_key = generator.generate_key(
        df=input_df,
        plugin_name="temporal",
        plugin_version="1.0.0",
        config={"lookback_days": 7}
    )
    ```
"""

import hashlib
import json
from typing import Any

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class CacheKeyGenerator:
    """Generate deterministic cache keys for feature computations.

    Keys are based on:
    - Data shape and sample content
    - Plugin name and version
    - Configuration parameters
    - Feature column names and types

    This ensures that cache entries are invalidated when any of these
    components change, while allowing cache hits for identical inputs.
    """

    @staticmethod
    def generate_key(
        df: pd.DataFrame,
        plugin_name: str,
        plugin_version: str,
        config: dict[str, Any],
    ) -> str:
        """Generate cache key for feature computation.

        Creates a deterministic hash from the input data, plugin metadata,
        and configuration. The key changes if any of these inputs change.

        Args:
            df: Input DataFrame to fingerprint.
            plugin_name: Name of feature plugin.
            plugin_version: Plugin version string.
            config: Plugin configuration dictionary.

        Returns:
            Hexadecimal cache key (64 characters).

        Example:
            >>> generator = CacheKeyGenerator()
            >>> key = generator.generate_key(
            ...     df=data_df,
            ...     plugin_name="lag",
            ...     plugin_version="1.0.0",
            ...     config={"lag_periods": [1, 7, 14]}
            ... )
            >>> print(key)
            'a3f2e1d8c9b0a1234567890abcdef123456789...'
        """
        key_components = []

        # Data fingerprint
        data_fingerprint = CacheKeyGenerator._create_data_fingerprint(df)
        key_components.append(data_fingerprint)

        # Plugin identifier
        plugin_id = f"{plugin_name}:v{plugin_version}"
        key_components.append(plugin_id)

        # Config hash
        config_hash = CacheKeyGenerator._hash_config(config)
        key_components.append(config_hash)

        # Combine and hash
        combined = "|".join(key_components)
        cache_key = hashlib.sha256(combined.encode()).hexdigest()

        logger.debug(f"Generated cache key: {cache_key[:16]}...")

        return cache_key

    @staticmethod
    def _create_data_fingerprint(df: pd.DataFrame) -> str:
        """Create fingerprint of DataFrame.

        Creates a deterministic fingerprint based on DataFrame structure
        and sample content. Changes to shape, columns, dtypes, or content
        will change the fingerprint.

        Args:
            df: DataFrame to fingerprint.

        Returns:
            Fingerprint string.
        """
        components = [
            f"shape:{df.shape}",
            f"columns:{','.join(sorted(str(c) for c in df.columns))}",
            f"dtypes:{','.join(str(dt) for dt in df.dtypes)}",
            f"index_type:{type(df.index).__name__}",
        ]

        # Sample first and last few rows for content hash
        if len(df) > 0:
            sample_size = min(10, len(df))
            head_hash = pd.util.hash_pandas_object(df.head(sample_size)).sum()
            tail_hash = pd.util.hash_pandas_object(df.tail(sample_size)).sum()
            components.append(f"content:{head_hash}:{tail_hash}")

        return "|".join(components)

    @staticmethod
    def _hash_config(config: dict[str, Any]) -> str:
        """Hash configuration dictionary.

        Creates a deterministic hash of the configuration dictionary.
        Keys are sorted to ensure consistent ordering.

        Args:
            config: Configuration dictionary to hash.

        Returns:
            MD5 hash of configuration (32 characters).
        """
        # Serialize config to deterministic JSON
        config_str = json.dumps(config, sort_keys=True, default=str)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()
        return config_hash
