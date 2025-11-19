# PC-082-09B: Configuration Management

**Ticket ID:** PC-082-09B  
**Epic:** Epic-09B - Interactive Mode & Advanced Features  
**Parent Epic:** Epic-09 - CLI & User Interface  
**Story Points:** 6  
**Priority:** High  
**Status:** Not Started  

---

## 📋 Description

Implement comprehensive configuration management commands for showing, validating, initializing, comparing, and exporting system configuration.

---

## 🎯 Acceptance Criteria

- [ ] `config show` command with section filtering
- [ ] Output format selection (yaml, json, table)
- [ ] `config validate` command with strict mode
- [ ] Validation checks: required fields, types, ranges, consistency, S3 availability
- [ ] `config init` command with templates
- [ ] Template support: development, production, testing
- [ ] `config diff` command for comparing configurations
- [ ] `config export` command to yaml/json/env formats
- [ ] Overwrite confirmation for init command
- [ ] Detailed error reporting for validation failures
- [ ] Next steps guidance after init

---

## 🔧 Technical Implementation

```python
# prevcarga/cli/commands/config.py
import click
from pathlib import Path
import yaml
import json

from prevcarga.cli.utils import load_config


@click.group()
def config():
    """Configuration management commands."""
    pass


@config.command('show')
@click.option('--section', help='Configuration section to display')
@click.option('--format', type=click.Choice(['yaml', 'json', 'table']), 
              default='yaml', help='Output format')
@click.pass_context
def show_config(ctx, section: str, format: str):
    """Display current system configuration.
    
    Examples:
        prevcarga config show
        prevcarga config show --section models
        prevcarga config show --section data --format json
    """
    cfg = ctx.obj.get('system_config') or load_config(ctx.obj.get('config'))
    
    if section:
        display_config_section(cfg, section, format)
    else:
        display_full_config(cfg, format)


@config.command('validate')
@click.option('--config-file', type=Path, help='Configuration file to validate')
@click.option('--strict', is_flag=True, help='Enable strict validation')
@click.pass_context
def validate_config(ctx, config_file: Path, strict: bool):
    """Validate current configuration for completeness and correctness.
    
    Checks for:
    - Required fields presence
    - Value type correctness
    - Value range validity
    - Cross-field consistency
    - Resource availability (S3, etc.)
    
    Examples:
        prevcarga config validate
        prevcarga config validate --config-file custom-config.yaml --strict
    """
    try:
        if config_file:
            cfg = load_config(config_file)
        else:
            cfg = ctx.obj.get('system_config') or load_config(ctx.obj.get('config'))
        
        validation_result = validate_system_config(cfg, strict=strict)
        
        display_validation_results(validation_result)
        
        if validation_result.is_valid:
            click.secho("\n✓ Configuration is valid", fg='green', bold=True)
        else:
            click.secho("\n✗ Configuration validation failed", fg='red', bold=True)
            raise click.ClickException("Configuration is invalid")
                
    except Exception as e:
        raise click.ClickException(f"Configuration error: {e}")


@config.command('init')
@click.option('--template', type=click.Choice(['development', 'production', 'testing']),
              default='development', help='Configuration template')
@click.option('--output', type=Path, help='Output configuration file path')
@click.pass_context
def init_config(ctx, template: str, output: Path):
    """Initialize configuration from template.
    
    Creates a new configuration file with appropriate defaults
    for the specified environment.
    
    Examples:
        prevcarga config init --template production
        prevcarga config init --template development --output custom-config.yaml
    """
    config_path = output or Path('prevcarga.yaml')
    
    # Check if file exists
    if config_path.exists():
        if not click.confirm(f"Configuration file {config_path} exists. Overwrite?"):
            click.echo("Operation cancelled")
            return
    
    # Create config from template
    click.echo(f"Creating configuration from '{template}' template...")
    create_config_from_template(config_path, template)
    
    click.secho(f"\n✓ Configuration initialized: {config_path}", fg='green', bold=True)
    click.echo(f"  Template: {template}")
    
    click.echo("\n╔═══════════════════════════════════════════════════╗")
    click.echo("║              Next Steps                           ║")
    click.echo("╚═══════════════════════════════════════════════════╝")
    click.echo("  1. Review and customize the configuration")
    click.echo(f"  2. Validate: prevcarga config validate --config-file {config_path}")
    click.echo(f"  3. Use: prevcarga --config {config_path} <command>")
    click.echo()


@config.command('diff')
@click.option('--config1', type=Path, required=True, help='First configuration file')
@click.option('--config2', type=Path, required=True, help='Second configuration file')
@click.option('--output', type=Path, help='Save diff to file')
def diff_config(config1: Path, config2: Path, output: Path):
    """Compare two configuration files.
    
    Shows differences between configuration files,
    useful for comparing environments or tracking changes.
    
    Examples:
        prevcarga config diff --config1 dev.yaml --config2 prod.yaml
        prevcarga config diff --config1 old.yaml --config2 new.yaml --output changes.txt
    """
    # Validate files exist
    if not config1.exists():
        raise click.ClickException(f"File not found: {config1}")
    if not config2.exists():
        raise click.ClickException(f"File not found: {config2}")
    
    cfg1 = load_config(config1)
    cfg2 = load_config(config2)
    
    diff_result = compare_configs(cfg1, cfg2)
    
    display_config_diff(diff_result, config1.name, config2.name)
    
    if output:
        save_config_diff(diff_result, output)
        click.echo(f"\n✓ Diff saved to: {output}")


@config.command('export')
@click.option('--format', type=click.Choice(['yaml', 'json', 'env']),
              default='yaml', help='Export format')
@click.option('--output', type=Path, help='Output file path')
@click.pass_context
def export_config(ctx, format: str, output: Path):
    """Export current configuration to specified format.
    
    Useful for documentation, backups, or conversion between formats.
    
    Examples:
        prevcarga config export --format json
        prevcarga config export --format env --output .env
    """
    cfg = ctx.obj.get('system_config') or load_config(ctx.obj.get('config'))
    
    if format == 'env':
        content = config_to_env_vars(cfg)
    elif format == 'json':
        content = config_to_json(cfg)
    else:
        content = config_to_yaml(cfg)
    
    if output:
        output.write_text(content)
        click.secho(f"✓ Configuration exported to {output}", fg='green')
    else:
        click.echo(content)


# Helper functions

def display_config_section(cfg, section: str, format: str):
    """Display specific configuration section."""
    if not hasattr(cfg, section):
        raise click.ClickException(f"Unknown section: {section}")
    
    section_data = getattr(cfg, section)
    
    if format == 'yaml':
        click.echo(yaml.dump({section: section_data.__dict__}, default_flow_style=False))
    elif format == 'json':
        click.echo(json.dumps({section: section_data.__dict__}, indent=2))
    elif format == 'table':
        display_section_as_table(section, section_data)


def display_full_config(cfg, format: str):
    """Display full configuration."""
    if format == 'yaml':
        click.echo(yaml.dump(cfg.to_dict(), default_flow_style=False))
    elif format == 'json':
        click.echo(json.dumps(cfg.to_dict(), indent=2))
    elif format == 'table':
        display_config_as_table(cfg)


def validate_system_config(cfg, strict: bool):
    """Validate system configuration."""
    from prevcarga.config import ConfigValidator
    
    validator = ConfigValidator(strict=strict)
    return validator.validate(cfg)


def display_validation_results(result):
    """Display validation results."""
    click.echo("\n╔═══════════════════════════════════════════════════╗")
    click.echo("║      Configuration Validation Results             ║")
    click.echo("╚═══════════════════════════════════════════════════╝")
    click.echo(f"\nChecks performed: {result.checks_performed}")
    
    if result.errors:
        click.echo(f"\n✗ Errors ({len(result.errors)}):")
        for error in result.errors:
            click.echo(f"  - {error}")
    
    if result.warnings:
        click.echo(f"\n⚠ Warnings ({len(result.warnings)}):")
        for warning in result.warnings:
            click.echo(f"  - {warning}")
    
    if result.is_valid and not result.warnings:
        click.echo("\n✓ No issues found")


def create_config_from_template(path: Path, template: str):
    """Create configuration from template."""
    from prevcarga.config import ConfigTemplate
    
    template_data = ConfigTemplate.get_template(template)
    
    with open(path, 'w') as f:
        yaml.dump(template_data, f, default_flow_style=False, sort_keys=False)


def compare_configs(cfg1, cfg2):
    """Compare two configurations."""
    from prevcarga.config import ConfigComparator
    
    comparator = ConfigComparator()
    return comparator.compare(cfg1, cfg2)


def display_config_diff(diff_result, name1: str, name2: str):
    """Display configuration diff."""
    click.echo(f"\n╔═══════════════════════════════════════════════════╗")
    click.echo(f"║      Configuration Diff                           ║")
    click.echo(f"╚═══════════════════════════════════════════════════╝")
    click.echo(f"\nComparing: {name1} vs {name2}")
    
    if diff_result.added:
        click.echo(f"\n➕ Added ({len(diff_result.added)}):")
        for item in diff_result.added:
            click.echo(f"  + {item}")
    
    if diff_result.removed:
        click.echo(f"\n➖ Removed ({len(diff_result.removed)}):")
        for item in diff_result.removed:
            click.echo(f"  - {item}")
    
    if diff_result.changed:
        click.echo(f"\n🔄 Changed ({len(diff_result.changed)}):")
        for item, (old, new) in diff_result.changed.items():
            click.echo(f"  ~ {item}: {old} → {new}")
    
    if not (diff_result.added or diff_result.removed or diff_result.changed):
        click.echo("\n✓ Configurations are identical")
```

---

## ✅ Definition of Done

- [ ] All config commands implemented
- [ ] Template system working
- [ ] Validation comprehensive
- [ ] Diff functionality accurate
- [ ] Export formats supported
- [ ] Unit tests pass (>80% coverage)
- [ ] Documentation complete

---

**Assignee:** Backend Team  
**Estimated Hours:** 10-12 hours  
**Target Completion:** Week 25, Day 4
