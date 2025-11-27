"""Tests for configuration management CLI commands.

This module provides comprehensive tests for config show, validate, init,
diff, and export commands.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
import yaml
from click.testing import CliRunner

from src.cli.commands.config import (
    ConfigComparator,
    ConfigDiffResult,
    ConfigFormat,
    ConfigTemplate,
    ConfigTemplateGenerator,
    ConfigValidator,
    ValidationIssue,
    ValidationResult,
    ValidationSeverity,
    config,
    config_to_env_vars,
    config_to_json,
    config_to_yaml,
    load_config_file,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def runner() -> CliRunner:
    """Create CLI runner."""
    return CliRunner()


@pytest.fixture
def valid_config() -> dict[str, Any]:
    """Create a valid configuration."""
    return {
        "system": {
            "environment": "development",
            "log_level": "INFO",
            "log_format": "text",
            "timezone": "America/Sao_Paulo",
        },
        "training": {
            "parallel_workers": 4,
            "timeout_seconds": 3600,
            "optimize_hyperparameters": True,
            "optimization_trials": 50,
            "cv_folds": 5,
        },
        "prediction": {
            "output_format": "csv",
            "output_dir": "predictions/",
            "apply_reconciliation": True,
        },
        "backtesting": {
            "start_date": "2024-01-01",
            "end_date": "2024-03-31",
            "step_days": 7,
            "generate_report": True,
        },
    }


@pytest.fixture
def invalid_config() -> dict[str, Any]:
    """Create an invalid configuration."""
    return {
        "system": {
            "environment": "invalid_env",  # Invalid value
            "log_level": "TRACE",  # Invalid value
        },
        "training": {
            "parallel_workers": "four",  # Invalid type
            "timeout_seconds": 10,  # Below minimum
        },
    }


@pytest.fixture
def temp_config_file(valid_config: dict[str, Any]) -> Path:
    """Create a temporary config file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(valid_config, f)
        return Path(f.name)


@pytest.fixture
def temp_json_config(valid_config: dict[str, Any]) -> Path:
    """Create a temporary JSON config file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(valid_config, f)
        return Path(f.name)


# =============================================================================
# ConfigValidator Tests
# =============================================================================


class TestConfigValidator:
    """Tests for ConfigValidator class."""

    def test_validate_valid_config(self, valid_config: dict[str, Any]) -> None:
        """Test validation of valid config."""
        validator = ConfigValidator()
        result = validator.validate(valid_config)

        assert result.is_valid
        assert result.checks_performed > 0
        assert len(result.errors) == 0

    def test_validate_invalid_config(self, invalid_config: dict[str, Any]) -> None:
        """Test validation of invalid config."""
        validator = ConfigValidator()
        result = validator.validate(invalid_config)

        assert not result.is_valid
        assert len(result.errors) > 0

    def test_validate_missing_sections(self) -> None:
        """Test validation with missing sections."""
        config = {"system": {"environment": "development"}}
        validator = ConfigValidator()
        result = validator.validate(config)

        # Should have warnings for missing sections
        assert len(result.warnings) > 0 or len(result.errors) > 0

    def test_validate_invalid_type(self) -> None:
        """Test validation with invalid value types."""
        config = {
            "system": {"environment": "development", "log_level": "INFO"},
            "training": {
                "parallel_workers": "not_a_number",  # Should be int
                "timeout_seconds": 3600,
            },
        }
        validator = ConfigValidator()
        result = validator.validate(config)

        assert not result.is_valid
        assert any("type" in e.lower() for e in result.errors)

    def test_validate_out_of_range(self) -> None:
        """Test validation with values out of range."""
        config = {
            "system": {"environment": "development", "log_level": "INFO"},
            "training": {
                "parallel_workers": 100,  # Above maximum 64
                "timeout_seconds": 3600,
            },
        }
        validator = ConfigValidator()
        result = validator.validate(config)

        assert not result.is_valid
        assert any("range" in e.lower() for e in result.errors)

    def test_validate_strict_mode(self, valid_config: dict[str, Any]) -> None:
        """Test strict validation mode."""
        validator = ConfigValidator(strict=True)
        result = validator.validate(valid_config)

        # Strict mode should do more checks
        assert result.checks_performed > 0

    def test_validate_empty_section_strict(self) -> None:
        """Test empty section detection in strict mode."""
        config = {
            "system": {"environment": "development", "log_level": "INFO"},
            "training": {},  # Empty section
        }
        validator = ConfigValidator(strict=True)
        result = validator.validate(config)

        assert not result.is_valid
        assert any("empty" in e.lower() for e in result.errors)

    def test_validation_result_properties(self) -> None:
        """Test ValidationResult properties."""
        result = ValidationResult(
            is_valid=False,
            checks_performed=10,
            errors=["Error 1", "Error 2"],
            warnings=["Warning 1"],
        )

        assert result.has_errors
        assert result.has_warnings
        assert not result.is_valid

    def test_validation_issue_dataclass(self) -> None:
        """Test ValidationIssue dataclass."""
        issue = ValidationIssue(
            severity=ValidationSeverity.ERROR,
            field="training.parallel_workers",
            message="Invalid value",
            suggestion="Use a number between 1 and 64",
        )

        assert issue.severity == ValidationSeverity.ERROR
        assert "training" in issue.field
        assert issue.suggestion is not None


# =============================================================================
# ConfigComparator Tests
# =============================================================================


class TestConfigComparator:
    """Tests for ConfigComparator class."""

    def test_compare_identical_configs(self, valid_config: dict[str, Any]) -> None:
        """Test comparing identical configs."""
        comparator = ConfigComparator()
        result = comparator.compare(valid_config, valid_config.copy())

        assert not result.has_differences
        assert result.total_changes == 0

    def test_compare_added_fields(self) -> None:
        """Test detecting added fields."""
        config1 = {"system": {"environment": "development"}}
        config2 = {
            "system": {"environment": "development"},
            "training": {"parallel_workers": 4},
        }

        comparator = ConfigComparator()
        result = comparator.compare(config1, config2)

        assert result.has_differences
        assert len(result.added) > 0
        assert "training" in result.added or any("training" in a for a in result.added)

    def test_compare_removed_fields(self) -> None:
        """Test detecting removed fields."""
        config1 = {
            "system": {"environment": "development"},
            "training": {"parallel_workers": 4},
        }
        config2 = {"system": {"environment": "development"}}

        comparator = ConfigComparator()
        result = comparator.compare(config1, config2)

        assert result.has_differences
        assert len(result.removed) > 0

    def test_compare_changed_values(self) -> None:
        """Test detecting changed values."""
        config1 = {"system": {"environment": "development"}}
        config2 = {"system": {"environment": "production"}}

        comparator = ConfigComparator()
        result = comparator.compare(config1, config2)

        assert result.has_differences
        assert len(result.changed) > 0
        assert "system.environment" in result.changed

    def test_compare_nested_changes(self) -> None:
        """Test detecting nested changes."""
        config1 = {
            "training": {
                "parallel_workers": 4,
                "options": {"opt1": "a", "opt2": "b"},
            }
        }
        config2 = {
            "training": {
                "parallel_workers": 8,  # Changed
                "options": {"opt1": "a", "opt2": "c"},  # opt2 changed
            }
        }

        comparator = ConfigComparator()
        result = comparator.compare(config1, config2)

        assert result.has_differences
        assert len(result.changed) >= 2

    def test_diff_result_properties(self) -> None:
        """Test ConfigDiffResult properties."""
        result = ConfigDiffResult(
            added=["field1"],
            removed=["field2"],
            changed={"field3": ("old", "new")},
        )

        assert result.has_differences
        assert result.total_changes == 3


# =============================================================================
# ConfigTemplateGenerator Tests
# =============================================================================


class TestConfigTemplateGenerator:
    """Tests for ConfigTemplateGenerator class."""

    def test_get_development_template(self) -> None:
        """Test getting development template."""
        template = ConfigTemplateGenerator.get_template(ConfigTemplate.DEVELOPMENT)

        assert "system" in template
        assert template["system"]["environment"] == "development"
        assert template["system"]["log_level"] == "DEBUG"

    def test_get_production_template(self) -> None:
        """Test getting production template."""
        template = ConfigTemplateGenerator.get_template(ConfigTemplate.PRODUCTION)

        assert "system" in template
        assert template["system"]["environment"] == "production"
        assert template["system"]["log_level"] == "INFO"

    def test_get_testing_template(self) -> None:
        """Test getting testing template."""
        template = ConfigTemplateGenerator.get_template(ConfigTemplate.TESTING)

        assert "system" in template
        assert template["training"]["parallel_workers"] == 2

    def test_get_template_by_string(self) -> None:
        """Test getting template by string name."""
        template = ConfigTemplateGenerator.get_template("production")

        assert template["system"]["environment"] == "production"

    def test_templates_have_required_sections(self) -> None:
        """Test all templates have required sections."""
        for template_type in ConfigTemplate:
            template = ConfigTemplateGenerator.get_template(template_type)

            assert "system" in template
            assert "training" in template
            assert "prediction" in template
            assert "backtesting" in template


# =============================================================================
# Config Conversion Tests
# =============================================================================


class TestConfigConversion:
    """Tests for config conversion functions."""

    def test_config_to_yaml(self, valid_config: dict[str, Any]) -> None:
        """Test converting config to YAML."""
        yaml_str = config_to_yaml(valid_config)

        assert isinstance(yaml_str, str)
        assert "system:" in yaml_str
        assert "environment:" in yaml_str

        # Should be valid YAML
        parsed = yaml.safe_load(yaml_str)
        assert "system" in parsed

    def test_config_to_json(self, valid_config: dict[str, Any]) -> None:
        """Test converting config to JSON."""
        json_str = config_to_json(valid_config)

        assert isinstance(json_str, str)
        assert "system" in json_str

        # Should be valid JSON
        parsed = json.loads(json_str)
        assert "system" in parsed

    def test_config_to_env_vars(self, valid_config: dict[str, Any]) -> None:
        """Test converting config to env vars."""
        env_str = config_to_env_vars(valid_config)

        assert isinstance(env_str, str)
        assert "PREVCARGA_" in env_str
        assert "SYSTEM" in env_str or "system" in env_str.lower()

    def test_config_to_env_vars_with_list(self) -> None:
        """Test converting config with list to env vars."""
        config = {"prediction": {"horizons": [0, 1, 2, 3]}}
        env_str = config_to_env_vars(config)

        assert "0,1,2,3" in env_str or "HORIZONS" in env_str

    def test_config_to_env_vars_with_bool(self) -> None:
        """Test converting config with bool to env vars."""
        config = {"training": {"optimize_hyperparameters": True}}
        env_str = config_to_env_vars(config)

        assert "true" in env_str or "True" in env_str


# =============================================================================
# Load Config Tests
# =============================================================================


class TestLoadConfig:
    """Tests for load_config_file function."""

    def test_load_yaml_config(self, temp_config_file: Path) -> None:
        """Test loading YAML config."""
        config = load_config_file(temp_config_file)

        assert "system" in config
        assert "training" in config

    def test_load_json_config(self, temp_json_config: Path) -> None:
        """Test loading JSON config."""
        config = load_config_file(temp_json_config)

        assert "system" in config
        assert "training" in config

    def test_load_nonexistent_file(self) -> None:
        """Test loading nonexistent file."""
        from click import ClickException

        with pytest.raises(ClickException) as exc_info:
            load_config_file(Path("/nonexistent/config.yaml"))

        assert "not found" in str(exc_info.value)

    def test_load_invalid_yaml(self) -> None:
        """Test loading invalid YAML."""
        from click import ClickException

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: syntax:")
            temp_path = Path(f.name)

        with pytest.raises(ClickException) as exc_info:
            load_config_file(temp_path)

        assert "Invalid YAML" in str(exc_info.value) or "YAML" in str(exc_info.value)


# =============================================================================
# CLI Command Tests
# =============================================================================


class TestConfigShowCommand:
    """Tests for config show command."""

    def test_show_default_config(self, runner: CliRunner) -> None:
        """Test showing default config when no file exists."""
        result = runner.invoke(config, ["show"])

        # Should show default config or warning
        assert result.exit_code == 0

    def test_show_yaml_format(
        self, runner: CliRunner, temp_config_file: Path
    ) -> None:
        """Test showing config in YAML format."""
        result = runner.invoke(
            config, ["show", "--config-file", str(temp_config_file), "--format", "yaml"]
        )

        assert result.exit_code == 0
        assert "system:" in result.output

    def test_show_json_format(
        self, runner: CliRunner, temp_config_file: Path
    ) -> None:
        """Test showing config in JSON format."""
        result = runner.invoke(
            config, ["show", "--config-file", str(temp_config_file), "--format", "json"]
        )

        assert result.exit_code == 0
        assert '"system"' in result.output

    def test_show_table_format(
        self, runner: CliRunner, temp_config_file: Path
    ) -> None:
        """Test showing config in table format."""
        result = runner.invoke(
            config, ["show", "--config-file", str(temp_config_file), "--format", "table"]
        )

        assert result.exit_code == 0
        assert "SYSTEM" in result.output

    def test_show_section_filter(
        self, runner: CliRunner, temp_config_file: Path
    ) -> None:
        """Test showing specific section."""
        result = runner.invoke(
            config, ["show", "--config-file", str(temp_config_file), "--section", "training"]
        )

        assert result.exit_code == 0
        assert "training" in result.output.lower()

    def test_show_invalid_section(
        self, runner: CliRunner, temp_config_file: Path
    ) -> None:
        """Test showing invalid section."""
        result = runner.invoke(
            config, ["show", "--config-file", str(temp_config_file), "--section", "nonexistent"]
        )

        assert result.exit_code != 0


class TestConfigValidateCommand:
    """Tests for config validate command."""

    def test_validate_valid_config(
        self, runner: CliRunner, temp_config_file: Path
    ) -> None:
        """Test validating valid config."""
        result = runner.invoke(
            config, ["validate", "--config-file", str(temp_config_file)]
        )

        assert result.exit_code == 0
        assert "valid" in result.output.lower()

    def test_validate_strict_mode(
        self, runner: CliRunner, temp_config_file: Path
    ) -> None:
        """Test strict validation mode."""
        result = runner.invoke(
            config, ["validate", "--config-file", str(temp_config_file), "--strict"]
        )

        # May pass or fail depending on config completeness
        assert "Strict mode:" in result.output

    def test_validate_no_file_found(self, runner: CliRunner) -> None:
        """Test validation when no file found."""
        with runner.isolated_filesystem():
            result = runner.invoke(config, ["validate"])

            # Should gracefully handle no file
            assert "No configuration file found" in result.output


class TestConfigInitCommand:
    """Tests for config init command."""

    def test_init_development_template(self, runner: CliRunner) -> None:
        """Test initializing development template."""
        with runner.isolated_filesystem():
            result = runner.invoke(
                config, ["init", "--template", "development"]
            )

            assert result.exit_code == 0
            assert Path("prevcarga.yaml").exists()

            # Verify content
            with open("prevcarga.yaml") as f:
                config_data = yaml.safe_load(f)
            assert config_data["system"]["environment"] == "development"

    def test_init_production_template(self, runner: CliRunner) -> None:
        """Test initializing production template."""
        with runner.isolated_filesystem():
            result = runner.invoke(
                config, ["init", "--template", "production"]
            )

            assert result.exit_code == 0
            assert "production" in result.output

    def test_init_testing_template(self, runner: CliRunner) -> None:
        """Test initializing testing template."""
        with runner.isolated_filesystem():
            result = runner.invoke(
                config, ["init", "--template", "testing"]
            )

            assert result.exit_code == 0

    def test_init_custom_output(self, runner: CliRunner) -> None:
        """Test initializing to custom output path."""
        with runner.isolated_filesystem():
            result = runner.invoke(
                config, ["init", "--output", "custom-config.yaml"]
            )

            assert result.exit_code == 0
            assert Path("custom-config.yaml").exists()

    def test_init_overwrite_confirmation(self, runner: CliRunner) -> None:
        """Test overwrite confirmation."""
        with runner.isolated_filesystem():
            # Create existing file
            Path("prevcarga.yaml").write_text("existing: config")

            # Should ask for confirmation
            result = runner.invoke(config, ["init"], input="n\n")

            assert "cancelled" in result.output.lower()

    def test_init_force_overwrite(self, runner: CliRunner) -> None:
        """Test force overwrite."""
        with runner.isolated_filesystem():
            # Create existing file
            Path("prevcarga.yaml").write_text("existing: config")

            result = runner.invoke(config, ["init", "--force"])

            assert result.exit_code == 0
            assert Path("prevcarga.yaml").exists()

    def test_init_shows_next_steps(self, runner: CliRunner) -> None:
        """Test init shows next steps."""
        with runner.isolated_filesystem():
            result = runner.invoke(config, ["init"])

            assert "Next Steps" in result.output
            assert "validate" in result.output


class TestConfigDiffCommand:
    """Tests for config diff command."""

    def test_diff_identical_configs(
        self, runner: CliRunner, temp_config_file: Path
    ) -> None:
        """Test diffing identical configs."""
        result = runner.invoke(
            config,
            ["diff", "--config1", str(temp_config_file), "--config2", str(temp_config_file)],
        )

        assert result.exit_code == 0
        assert "identical" in result.output.lower()

    def test_diff_different_configs(self, runner: CliRunner) -> None:
        """Test diffing different configs."""
        with runner.isolated_filesystem():
            # Create two different configs
            config1 = {"system": {"environment": "development"}}
            config2 = {"system": {"environment": "production"}}

            with open("config1.yaml", "w") as f:
                yaml.dump(config1, f)
            with open("config2.yaml", "w") as f:
                yaml.dump(config2, f)

            result = runner.invoke(
                config, ["diff", "--config1", "config1.yaml", "--config2", "config2.yaml"]
            )

            assert result.exit_code == 0
            assert "Changed" in result.output

    def test_diff_save_output(self, runner: CliRunner) -> None:
        """Test saving diff to file."""
        with runner.isolated_filesystem():
            config1 = {"system": {"environment": "development"}}
            config2 = {"system": {"environment": "production"}}

            with open("config1.yaml", "w") as f:
                yaml.dump(config1, f)
            with open("config2.yaml", "w") as f:
                yaml.dump(config2, f)

            result = runner.invoke(
                config,
                ["diff", "--config1", "config1.yaml", "--config2", "config2.yaml", "--output", "diff.txt"],
            )

            assert result.exit_code == 0
            assert Path("diff.txt").exists()

    def test_diff_nonexistent_file(self, runner: CliRunner) -> None:
        """Test diffing with nonexistent file."""
        result = runner.invoke(
            config, ["diff", "--config1", "nonexistent.yaml", "--config2", "also-nonexistent.yaml"]
        )

        assert result.exit_code != 0


class TestConfigExportCommand:
    """Tests for config export command."""

    def test_export_yaml(
        self, runner: CliRunner, temp_config_file: Path
    ) -> None:
        """Test exporting to YAML."""
        result = runner.invoke(
            config, ["export", "--config-file", str(temp_config_file), "--format", "yaml"]
        )

        assert result.exit_code == 0
        assert "system:" in result.output

    def test_export_json(
        self, runner: CliRunner, temp_config_file: Path
    ) -> None:
        """Test exporting to JSON."""
        result = runner.invoke(
            config, ["export", "--config-file", str(temp_config_file), "--format", "json"]
        )

        assert result.exit_code == 0
        assert '"system"' in result.output

    def test_export_env(
        self, runner: CliRunner, temp_config_file: Path
    ) -> None:
        """Test exporting to env format."""
        result = runner.invoke(
            config, ["export", "--config-file", str(temp_config_file), "--format", "env"]
        )

        assert result.exit_code == 0
        assert "PREVCARGA_" in result.output

    def test_export_to_file(self, runner: CliRunner) -> None:
        """Test exporting to file."""
        with runner.isolated_filesystem():
            # Create source config
            with open("source.yaml", "w") as f:
                yaml.dump({"system": {"environment": "development"}}, f)

            result = runner.invoke(
                config,
                ["export", "--config-file", "source.yaml", "--format", "json", "--output", "config.json"],
            )

            assert result.exit_code == 0
            assert Path("config.json").exists()


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_config(self) -> None:
        """Test handling empty config."""
        validator = ConfigValidator()
        result = validator.validate({})

        # Should report missing sections
        assert len(result.errors) > 0 or len(result.warnings) > 0

    def test_config_with_none_values(self) -> None:
        """Test handling None values."""
        config = {
            "system": {
                "environment": "development",
                "log_level": None,
            }
        }
        validator = ConfigValidator()
        result = validator.validate(config)

        # Should handle gracefully
        assert result.checks_performed > 0

    def test_nested_dict_comparison(self) -> None:
        """Test deep nested comparison."""
        config1 = {
            "level1": {
                "level2": {
                    "level3": {"value": "a"}
                }
            }
        }
        config2 = {
            "level1": {
                "level2": {
                    "level3": {"value": "b"}
                }
            }
        }

        comparator = ConfigComparator()
        result = comparator.compare(config1, config2)

        assert result.has_differences
        assert any("level3" in k for k in result.changed.keys())

    def test_config_format_enum(self) -> None:
        """Test ConfigFormat enum values."""
        assert ConfigFormat.YAML.value == "yaml"
        assert ConfigFormat.JSON.value == "json"
        assert ConfigFormat.TABLE.value == "table"
        assert ConfigFormat.ENV.value == "env"

    def test_validation_severity_enum(self) -> None:
        """Test ValidationSeverity enum values."""
        assert ValidationSeverity.ERROR.value == "error"
        assert ValidationSeverity.WARNING.value == "warning"
        assert ValidationSeverity.INFO.value == "info"


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for config commands."""

    def test_init_then_validate(self, runner: CliRunner) -> None:
        """Test init followed by validate."""
        with runner.isolated_filesystem():
            # Init
            init_result = runner.invoke(
                config, ["init", "--template", "development"]
            )
            assert init_result.exit_code == 0

            # Validate
            validate_result = runner.invoke(
                config, ["validate", "--config-file", "prevcarga.yaml"]
            )
            assert validate_result.exit_code == 0
            assert "valid" in validate_result.output.lower()

    def test_init_then_export_then_validate(self, runner: CliRunner) -> None:
        """Test init, export, and validate chain."""
        with runner.isolated_filesystem():
            # Init
            runner.invoke(config, ["init", "--template", "production"])

            # Export to JSON
            runner.invoke(
                config,
                ["export", "--config-file", "prevcarga.yaml", "--format", "json", "--output", "config.json"],
            )

            # Validate original
            result = runner.invoke(
                config, ["validate", "--config-file", "prevcarga.yaml"]
            )
            assert "valid" in result.output.lower()

    def test_diff_after_modify(self, runner: CliRunner) -> None:
        """Test diff after modifying config."""
        with runner.isolated_filesystem():
            # Init original
            runner.invoke(config, ["init", "--template", "development", "--output", "original.yaml"])

            # Create modified version
            with open("original.yaml") as f:
                original = yaml.safe_load(f)

            modified = original.copy()
            modified["system"]["environment"] = "staging"
            modified["training"]["parallel_workers"] = 8

            with open("modified.yaml", "w") as f:
                yaml.dump(modified, f)

            # Diff
            result = runner.invoke(
                config, ["diff", "--config1", "original.yaml", "--config2", "modified.yaml"]
            )

            assert result.exit_code == 0
            assert "Changed" in result.output
