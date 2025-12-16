# PC-092-11: renv Configuration

**Epic:** [EPIC-11: One-Line Installer](../epics/EPIC-11-installer.md)
**Task Reference:** T-11.7
**Priority:** High
**Estimated Effort:** 1 day

---

## Summary

Configure the renv environment with a locked `renv.lock` file containing pinned package versions, and run `renv::restore()` to install all required R packages.

---

## Acceptance Criteria

- [ ] renv.lock file created with pinned versions
- [ ] renv initialized in installation directory
- [ ] All required packages installed
- [ ] Package installation verified
- [ ] Installation completes in reasonable time (<5 min)
- [ ] Cache utilized for faster reinstalls

---

## Technical Specification

### renv Configuration Functions

```bash
# ──────────────────────────────────────────────────────────────────────────────
# renv Configuration
# ──────────────────────────────────────────────────────────────────────────────

#' Configure and restore renv environment
configure_renv() {
    step "6/7 - Configurando ambiente R (renv)"

    local renv_dir="$INSTALL_DIR/renv"

    # Create renv.lock
    create_renv_lock

    # Initialize renv
    initialize_renv

    # Restore packages
    restore_renv_packages

    success "Ambiente R configurado"
}

#' Create renv.lock file with pinned versions
create_renv_lock() {
    info "Criando renv.lock..."

    cat > "$INSTALL_DIR/renv.lock" << 'RENV_LOCK'
{
  "R": {
    "Version": "4.3.3",
    "Repositories": [
      {
        "Name": "CRAN",
        "URL": "https://cloud.r-project.org"
      }
    ]
  },
  "Packages": {
    "renv": {
      "Package": "renv",
      "Version": "1.0.7",
      "Source": "Repository",
      "Repository": "CRAN"
    },
    "R6": {
      "Package": "R6",
      "Version": "2.5.1",
      "Source": "Repository",
      "Repository": "CRAN"
    },
    "data.table": {
      "Package": "data.table",
      "Version": "1.15.4",
      "Source": "Repository",
      "Repository": "CRAN"
    },
    "arrow": {
      "Package": "arrow",
      "Version": "15.0.0",
      "Source": "Repository",
      "Repository": "CRAN"
    },
    "yaml": {
      "Package": "yaml",
      "Version": "2.3.8",
      "Source": "Repository",
      "Repository": "CRAN"
    },
    "paws": {
      "Package": "paws",
      "Version": "0.5.0",
      "Source": "Repository",
      "Repository": "CRAN"
    },
    "future": {
      "Package": "future",
      "Version": "1.33.1",
      "Source": "Repository",
      "Repository": "CRAN"
    },
    "future.apply": {
      "Package": "future.apply",
      "Version": "1.11.1",
      "Source": "Repository",
      "Repository": "CRAN"
    },
    "cli": {
      "Package": "cli",
      "Version": "3.6.2",
      "Source": "Repository",
      "Repository": "CRAN"
    },
    "optparse": {
      "Package": "optparse",
      "Version": "1.7.4",
      "Source": "Repository",
      "Repository": "CRAN"
    },
    "checkmate": {
      "Package": "checkmate",
      "Version": "2.3.1",
      "Source": "Repository",
      "Repository": "CRAN"
    },
    "jsonlite": {
      "Package": "jsonlite",
      "Version": "1.8.8",
      "Source": "Repository",
      "Repository": "CRAN"
    },
    "highcharter": {
      "Package": "highcharter",
      "Version": "0.9.4",
      "Source": "Repository",
      "Repository": "CRAN"
    },
    "rmarkdown": {
      "Package": "rmarkdown",
      "Version": "2.25",
      "Source": "Repository",
      "Repository": "CRAN"
    },
    "htmlwidgets": {
      "Package": "htmlwidgets",
      "Version": "1.6.4",
      "Source": "Repository",
      "Repository": "CRAN"
    },
    "forecast": {
      "Package": "forecast",
      "Version": "8.22.0",
      "Source": "Repository",
      "Repository": "CRAN"
    },
    "lightgbm": {
      "Package": "lightgbm",
      "Version": "4.3.0",
      "Source": "Repository",
      "Repository": "CRAN"
    },
    "ranger": {
      "Package": "ranger",
      "Version": "0.16.0",
      "Source": "Repository",
      "Repository": "CRAN"
    },
    "xgboost": {
      "Package": "xgboost",
      "Version": "1.7.7.1",
      "Source": "Repository",
      "Repository": "CRAN"
    },
    "glue": {
      "Package": "glue",
      "Version": "1.7.0",
      "Source": "Repository",
      "Repository": "CRAN"
    },
    "lubridate": {
      "Package": "lubridate",
      "Version": "1.9.3",
      "Source": "Repository",
      "Repository": "CRAN"
    },
    "testthat": {
      "Package": "testthat",
      "Version": "3.2.1",
      "Source": "Repository",
      "Repository": "CRAN"
    },
    "covr": {
      "Package": "covr",
      "Version": "3.6.4",
      "Source": "Repository",
      "Repository": "CRAN"
    }
  }
}
RENV_LOCK

    success "renv.lock criado"
}

#' Initialize renv in the installation directory
initialize_renv() {
    info "Inicializando renv..."

    # Create renv directory structure
    mkdir -p "$INSTALL_DIR/renv/library"
    mkdir -p "$INSTALL_DIR/renv/cache"
    mkdir -p "$INSTALL_DIR/renv/staging"

    # Create renv settings
    cat > "$INSTALL_DIR/renv/settings.json" << 'SETTINGS'
{
  "bioconductor.version": null,
  "external.libraries": [],
  "ignored.packages": [],
  "package.dependency.fields": ["Imports", "Depends", "LinkingTo"],
  "r.version": null,
  "snapshot.type": "implicit",
  "use.cache": true,
  "vcs.ignore.cellar": true,
  "vcs.ignore.library": true,
  "vcs.ignore.local": true
}
SETTINGS

    # Create activate.R
    R --vanilla -q -e "
        renv_lib <- '$INSTALL_DIR/renv/library'
        dir.create(renv_lib, recursive = TRUE, showWarnings = FALSE)

        # Bootstrap renv
        if (!requireNamespace('renv', quietly = TRUE)) {
            install.packages('renv', lib = renv_lib, repos = 'https://cloud.r-project.org')
        }

        # Write activate script
        library(renv, lib.loc = renv_lib)
        renv::activate(project = '$INSTALL_DIR')
    " 2>&1 | while read -r line; do
        debug "$line"
    done

    success "renv inicializado"
}

#' Restore packages from renv.lock
restore_renv_packages() {
    info "Instalando pacotes R (isso pode demorar alguns minutos)..."

    local start_time=$(date +%s)

    # Run renv::restore()
    R --vanilla -q -e "
        setwd('$INSTALL_DIR')
        source('$INSTALL_DIR/renv/activate.R')

        # Configure options
        options(
            repos = c(CRAN = 'https://cloud.r-project.org'),
            renv.config.cache.enabled = TRUE,
            renv.config.install.verbose = FALSE
        )

        # Restore packages
        cat('Restaurando pacotes do renv.lock...\\n')
        renv::restore(prompt = FALSE)

        cat('Verificando pacotes instalados...\\n')
        installed <- installed.packages()[, 'Package']
        required <- names(renv::dependencies(quiet = TRUE))

        missing <- setdiff(required, installed)
        if (length(missing) > 0) {
            stop('Pacotes faltando: ', paste(missing, collapse = ', '))
        }

        cat('Todos os pacotes instalados com sucesso!\\n')
    " 2>&1 | while read -r line; do
        # Show progress
        if [[ "$line" == *"Installing"* ]]; then
            info "$line"
        else
            debug "$line"
        fi
    done

    local end_time=$(date +%s)
    local elapsed=$((end_time - start_time))

    success "Pacotes instalados em ${elapsed}s"
}

#' Verify package installation
verify_packages() {
    info "Verificando pacotes..."

    local required_packages=(
        "R6"
        "data.table"
        "arrow"
        "yaml"
        "cli"
        "forecast"
    )

    R --vanilla -q -e "
        source('$INSTALL_DIR/renv/activate.R')

        packages <- c('R6', 'data.table', 'arrow', 'yaml', 'cli', 'forecast')
        missing <- packages[!packages %in% installed.packages()[, 'Package']]

        if (length(missing) > 0) {
            cat('Pacotes faltando:', paste(missing, collapse = ', '), '\\n')
            quit(status = 1)
        }

        cat('Todos os pacotes principais verificados\\n')
    " || die "Verificação de pacotes falhou"

    success "Pacotes verificados"
}
```

### Package List

| Package | Version | Purpose |
|---------|---------|---------|
| renv | 1.0.7 | Environment management |
| R6 | 2.5.1 | OOP classes |
| data.table | 1.15.4 | Data manipulation |
| arrow | 15.0.0 | Parquet I/O |
| yaml | 2.3.8 | Configuration |
| paws | 0.5.0 | AWS S3 access |
| future | 1.33.1 | Parallel execution |
| future.apply | 1.11.1 | Parallel apply |
| cli | 3.6.2 | CLI formatting |
| optparse | 1.7.4 | Argument parsing |
| checkmate | 2.3.1 | Input validation |
| jsonlite | 1.8.8 | JSON handling |
| highcharter | 0.9.4 | Interactive charts |
| rmarkdown | 2.25 | Report generation |
| htmlwidgets | 1.6.4 | HTML widgets |
| forecast | 8.22.0 | Holt-Winters model |
| lightgbm | 4.3.0 | LightGBM model |
| ranger | 0.16.0 | Random Forest |
| xgboost | 1.7.7.1 | XGBoost model |
| lubridate | 1.9.3 | Date handling |
| testthat | 3.2.1 | Testing |
| covr | 3.6.4 | Coverage |

---

## Test Cases

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| TC-001 | renv.lock created | Valid JSON file |
| TC-002 | renv initialized | activate.R exists |
| TC-003 | Packages restored | All packages installed |
| TC-004 | Core packages verified | R6, data.table, arrow |
| TC-005 | Install time < 5 min | On good connection |
| TC-006 | Cache utilized | Faster on reinstall |
| TC-007 | Missing package detected | Error reported |
| TC-008 | Arrow loads | Binary works |
| TC-009 | LightGBM loads | Binary works |
| TC-010 | renv activate works | Library path set |

---

## Dependencies

- PC-088-11: R Installation
- PC-089-11: Directory Structure

---

## Definition of Done

- [ ] renv.lock file complete
- [ ] All packages pinned with versions
- [ ] renv initialization working
- [ ] Package restore working
- [ ] Package verification working
- [ ] Cache configured
- [ ] Tested on all platforms
- [ ] Code reviewed and approved
- [ ] Merged to develop branch

---

## Notes

- renv provides reproducible environments
- Cache speeds up reinstalls significantly
- Some packages require compilation (arrow, lightgbm)
- Consider pre-compiled binaries for speed
