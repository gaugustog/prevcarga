"""Sample test file to verify test infrastructure."""

import sys
from pathlib import Path


class TestProjectStructure:
    """Tests for project structure and setup."""

    def test_project_initialized(self) -> None:
        """Verify project structure is initialized."""
        project_root = Path(__file__).parent.parent

        # Check key directories exist
        assert (project_root / "src").exists(), "src/ directory should exist"
        assert (project_root / "tests").exists(), "tests/ directory should exist"
        assert (project_root / "config").exists(), "config/ directory should exist"
        assert (project_root / "scripts").exists(), "scripts/ directory should exist"

    def test_python_version(self) -> None:
        """Verify Python version is 3.11+."""
        assert sys.version_info >= (3, 11), f"Python 3.11+ required, got {sys.version}"

    def test_src_package_structure(self) -> None:
        """Verify src package structure."""
        project_root = Path(__file__).parent.parent
        src_dir = project_root / "src"

        expected_modules = [
            "storage",
            "utils",
            "data",
            "features",
            "models",
            "ensemble",
            "reconciliation",
            "evaluation",
            "orchestration",
            "cli",
        ]

        for module in expected_modules:
            module_dir = src_dir / module
            assert module_dir.exists(), f"src/{module}/ should exist"
            assert (module_dir / "__init__.py").exists(), f"src/{module}/__init__.py should exist"


class TestCoreImports:
    """Tests for core package imports."""

    def test_core_dependencies(self) -> None:
        """Verify core packages can be imported."""
        import boto3  # noqa: F401
        import numpy as np  # noqa: F401
        import pandas as pd  # noqa: F401
        import pydantic  # noqa: F401
        import yaml  # noqa: F401

    def test_dev_dependencies(self) -> None:
        """Verify dev packages can be imported."""
        import black  # noqa: F401
        import mypy  # noqa: F401
        import pytest  # noqa: F401
        import ruff  # noqa: F401


class TestStorageModule:
    """Tests for storage module imports."""

    def test_storage_backend_import(self) -> None:
        """Verify storage backend can be imported."""
        from src.storage import ObjectNotFoundError, StorageBackend, StorageError  # noqa: F401

    def test_storage_factory_import(self) -> None:
        """Verify storage factory can be imported."""
        from src.storage import StorageFactory, get_storage_backend  # noqa: F401

    def test_storage_backends_import(self) -> None:
        """Verify storage backend implementations can be imported."""
        from src.storage import LocalStorageBackend, S3StorageBackend  # noqa: F401


class TestUtilsModule:
    """Tests for utils module imports."""

    def test_logger_import(self) -> None:
        """Verify logger can be imported."""
        from src.utils import LogContext, get_logger, setup_logging  # noqa: F401

    def test_logger_works(self) -> None:
        """Verify logger can log messages."""
        from src.utils import get_logger

        logger = get_logger("test")
        logger.info("Test message from sample test")
