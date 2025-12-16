# PC-094-11: Default Configuration

**Epic:** [EPIC-11: One-Line Installer](../epics/EPIC-11-installer.md)
**Task Reference:** T-11.9
**Priority:** Medium
**Estimated Effort:** 0.5 days

---

## Summary

Generate a default `config.yaml` configuration file in `~/.prevcarga/config/` with sensible defaults for local development and demo mode.

---

## Acceptance Criteria

- [ ] config.yaml created with default values
- [ ] Uses local storage backend by default
- [ ] Logging configured to console and file
- [ ] All required sections present
- [ ] Example config provided for reference
- [ ] Sensitive values have placeholders

---

## Technical Specification

### Default Configuration Creation

```bash
# ──────────────────────────────────────────────────────────────────────────────
# Default Configuration
# ──────────────────────────────────────────────────────────────────────────────

#' Create default configuration files
create_default_config() {
    info "Criando configuração padrão..."

    local config_dir="$INSTALL_DIR/config"
    local config_file="$config_dir/config.yaml"
    local example_file="$config_dir/config.example.yaml"

    # Create main config
    create_main_config "$config_file"

    # Create example config with full documentation
    create_example_config "$example_file"

    success "Configuração criada: $config_file"
}

#' Create main configuration file
create_main_config() {
    local config_file="$1"

    cat > "$config_file" << 'CONFIG'
# ══════════════════════════════════════════════════════════════════════════════
# PrevCargaONS Configuration
# ══════════════════════════════════════════════════════════════════════════════
#
# This file contains the default configuration for PrevCargaONS.
# See config.example.yaml for full documentation of all options.
#
# ══════════════════════════════════════════════════════════════════════════════

# ──────────────────────────────────────────────────────────────────────────────
# Project Settings
# ──────────────────────────────────────────────────────────────────────────────

project:
  name: "PrevCargaONS"
  version: "1.0.0"
  seed: 42
  environment: "development"

# ──────────────────────────────────────────────────────────────────────────────
# Storage Configuration
# ──────────────────────────────────────────────────────────────────────────────

storage:
  backend: local

  local:
    base_path: ~/.prevcarga/data
    carga_path: carga
    previsao_path: previsao
    models_path: models

  # S3 storage (uncomment and configure for production)
  # s3:
  #   bucket: your-bucket-name
  #   prefix: prevcarga/
  #   region: sa-east-1

# ──────────────────────────────────────────────────────────────────────────────
# Logging Configuration
# ──────────────────────────────────────────────────────────────────────────────

logging:
  level: INFO
  format: json

  handlers:
    - type: console
      level: INFO

    - type: file
      level: DEBUG
      path: ~/.prevcarga/logs/prevcarga.log
      max_size_mb: 10
      backup_count: 5

# ──────────────────────────────────────────────────────────────────────────────
# Model Configuration
# ──────────────────────────────────────────────────────────────────────────────

models:
  default_horizons: [0, 1, 2, 3, 4, 5, 6, 7, 8]
  train_window_days: 365

  enabled:
    - hw

  # Model-specific configurations
  hw:
    seasonal_periods: 24

# ──────────────────────────────────────────────────────────────────────────────
# Feature Configuration
# ──────────────────────────────────────────────────────────────────────────────

features:
  plugins:
    calendar:
      enabled: true

    lag:
      enabled: true
      config:
        lags: [1, 24, 168]

    rolling:
      enabled: true
      config:
        windows: [24, 168]
        functions: [mean, sd]

# ──────────────────────────────────────────────────────────────────────────────
# Areas Configuration
# ──────────────────────────────────────────────────────────────────────────────

areas:
  default: [RJ, SP, MG]

  subsystems:
    SECO: [RJ, SP, MG, ES, GO, MS, DF, MT, AC, RO]
    S: [PR, SC, RS]
    NE: [BA, SE, AL, PE, PB, RN, CE, PI, MA]
    N: [PA, AP, AM, TO, RR]

# ──────────────────────────────────────────────────────────────────────────────
# Parallel Processing
# ──────────────────────────────────────────────────────────────────────────────

parallel:
  enabled: true
  workers: auto  # auto = number of CPU cores - 1
  backend: future

# ──────────────────────────────────────────────────────────────────────────────
# Demo Mode
# ──────────────────────────────────────────────────────────────────────────────

demo:
  enabled: false
  use_sample_data: true
  sample_days: 30
CONFIG
}

#' Create example configuration with full documentation
create_example_config() {
    local example_file="$1"

    cat > "$example_file" << 'EXAMPLE'
# ══════════════════════════════════════════════════════════════════════════════
# PrevCargaONS Configuration - Full Example
# ══════════════════════════════════════════════════════════════════════════════
#
# This file documents all available configuration options.
# Copy sections to config.yaml and modify as needed.
#
# ══════════════════════════════════════════════════════════════════════════════

# ──────────────────────────────────────────────────────────────────────────────
# Project Settings
# ──────────────────────────────────────────────────────────────────────────────

project:
  # Project name (used in reports and logs)
  name: "PrevCargaONS"

  # Version (should match installed version)
  version: "1.0.0"

  # Random seed for reproducibility
  seed: 42

  # Environment: development, staging, production
  environment: "production"

  # Timezone (Brazilian official time)
  timezone: "America/Sao_Paulo"

# ──────────────────────────────────────────────────────────────────────────────
# Storage Configuration
# ──────────────────────────────────────────────────────────────────────────────

storage:
  # Backend: local, s3, azure, gcs
  backend: s3

  # Local filesystem storage
  local:
    base_path: ~/.prevcarga/data
    carga_path: carga
    previsao_path: previsao
    models_path: models

  # AWS S3 storage (production)
  s3:
    bucket: ons-prevcarga-prod
    prefix: data/
    region: sa-east-1
    # Credentials from environment or IAM role
    # access_key: ${AWS_ACCESS_KEY_ID}
    # secret_key: ${AWS_SECRET_ACCESS_KEY}

  # Azure Blob Storage
  azure:
    container: prevcarga
    account_name: onsprevcarga
    # account_key: ${AZURE_STORAGE_KEY}

  # Google Cloud Storage
  gcs:
    bucket: ons-prevcarga
    project: ons-prevcarga
    # credentials_file: ~/.gcloud/credentials.json

# ──────────────────────────────────────────────────────────────────────────────
# Logging Configuration
# ──────────────────────────────────────────────────────────────────────────────

logging:
  # Minimum log level: DEBUG, INFO, WARN, ERROR
  level: INFO

  # Log format: json, text
  format: json

  # Log handlers
  handlers:
    - type: console
      level: INFO
      colored: true

    - type: file
      level: DEBUG
      path: ~/.prevcarga/logs/prevcarga.log
      max_size_mb: 10
      backup_count: 5

    # Optional: send logs to external service
    # - type: cloudwatch
    #   level: WARN
    #   log_group: /prevcarga/production

# ──────────────────────────────────────────────────────────────────────────────
# Model Configuration
# ──────────────────────────────────────────────────────────────────────────────

models:
  # Forecast horizons (D+N days)
  default_horizons: [0, 1, 2, 3, 4, 5, 6, 7, 8]

  # Training window in days
  train_window_days: 365

  # Retraining frequency in days
  retrain_frequency: 7

  # Enabled models (must be registered in ModelRegistry)
  enabled:
    - lgbm
    - rf
    - hw

  # Model-specific configurations
  lgbm:
    num_leaves: 31
    learning_rate: 0.05
    n_estimators: 100
    objective: regression

  rf:
    num_trees: 100
    max_depth: 10
    min_node_size: 5

  hw:
    seasonal_periods: 24
    damped: true

# ──────────────────────────────────────────────────────────────────────────────
# Combination Configuration
# ──────────────────────────────────────────────────────────────────────────────

combination:
  # Default combiner strategy
  default: inverse_mape

  # Available combiners
  enabled:
    - simple_average
    - inverse_mape
    - optimal

  # Bias correction
  bias_correction: true

# ──────────────────────────────────────────────────────────────────────────────
# Reconciliation Configuration
# ──────────────────────────────────────────────────────────────────────────────

reconciliation:
  # Default reconciler
  default: bottom_up

  # Available reconcilers
  enabled:
    - bottom_up
    - ols
    - mint

  # Loss calculation
  include_losses: true
  loss_factors:
    SECO: 0.025
    S: 0.022
    NE: 0.028
    N: 0.030

# ──────────────────────────────────────────────────────────────────────────────
# Parallel Processing
# ──────────────────────────────────────────────────────────────────────────────

parallel:
  # Enable parallel processing
  enabled: true

  # Number of workers (auto = CPU cores - 1)
  workers: auto

  # Backend: future, parallel
  backend: future

  # Memory limit per worker (MB)
  memory_limit: 4096
EXAMPLE
}
```

---

## Test Cases

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| TC-001 | Config file created | File exists |
| TC-002 | Valid YAML syntax | yaml::read_yaml works |
| TC-003 | All sections present | project, storage, logging |
| TC-004 | Local backend default | backend: local |
| TC-005 | Example file created | config.example.yaml exists |
| TC-006 | Paths expanded | ~ expanded to $HOME |
| TC-007 | Logging handlers | console and file |
| TC-008 | Model defaults | hw enabled |
| TC-009 | Area defaults | RJ, SP, MG |
| TC-010 | Parallel enabled | parallel.enabled: true |

---

## Dependencies

- PC-089-11: Directory Structure

---

## Definition of Done

- [ ] config.yaml created
- [ ] config.example.yaml created
- [ ] Valid YAML syntax
- [ ] All required sections present
- [ ] Sensible defaults for development
- [ ] Documented example with all options
- [ ] Paths use ~ for portability
- [ ] Code reviewed and approved
- [ ] Merged to develop branch

---

## Notes

- Default to local storage for simplicity
- S3 config commented but ready
- Logging to both console and file
- Demo mode disabled by default
- Parallel processing enabled by default
