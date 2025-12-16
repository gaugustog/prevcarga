# PC-089-11: Directory Structure

**Epic:** [EPIC-11: One-Line Installer](../epics/EPIC-11-installer.md)
**Task Reference:** T-11.4
**Priority:** High
**Estimated Effort:** 0.5 days

---

## Summary

Create the installation directory structure at `~/.prevcarga` with all required subdirectories for binaries, source, renv, data, configuration, and logs.

---

## Acceptance Criteria

- [ ] Installation directory created at `~/.prevcarga`
- [ ] All subdirectories created: bin, src, renv, data, output, config, logs
- [ ] Appropriate permissions set (700 for sensitive, 755 for executables)
- [ ] Backup of existing installation if present
- [ ] .gitignore created for local development

---

## Technical Specification

### Directory Creation Functions

```bash
# ──────────────────────────────────────────────────────────────────────────────
# Directory Structure
# ──────────────────────────────────────────────────────────────────────────────

#' Create installation directory structure
create_directory_structure() {
    step "3/7 - Criando estrutura de diretórios"

    # Check for existing installation
    if [[ -d "$INSTALL_DIR" ]]; then
        handle_existing_installation
    fi

    # Create main directory
    mkdir -p "$INSTALL_DIR"

    # Create subdirectories
    local dirs=(
        "bin"           # Launcher scripts
        "src"           # R source files
        "renv"          # R environment (renv library)
        "data"          # Local data storage
        "data/carga"    # Load data
        "data/previsao" # Forecast output
        "data/models"   # Saved models
        "output"        # Report output
        "config"        # YAML configurations
        "logs"          # Log files
        "cache"         # Temporary cache
    )

    for dir in "${dirs[@]}"; do
        mkdir -p "$INSTALL_DIR/$dir"
        debug "Criado: $INSTALL_DIR/$dir"
    done

    # Set permissions
    set_directory_permissions

    # Create .gitignore
    create_gitignore

    # Create version file
    echo "$VERSION" > "$INSTALL_DIR/.version"

    success "Estrutura de diretórios criada"
    info "Diretório: ${BOLD}$INSTALL_DIR${RESET}"
}

#' Handle existing installation
handle_existing_installation() {
    local existing_version=""

    # Check for version file
    if [[ -f "$INSTALL_DIR/.version" ]]; then
        existing_version=$(cat "$INSTALL_DIR/.version")
    fi

    if [[ -n "$existing_version" ]]; then
        info "Instalação existente encontrada: v$existing_version"

        if [[ "$existing_version" == "$VERSION" ]]; then
            warn "Mesma versão já instalada"
            read -p "  Reinstalar? [s/N] " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Ss]$ ]]; then
                info "Instalação cancelada"
                exit 0
            fi
        fi
    else
        warn "Diretório existente encontrado (versão desconhecida)"
    fi

    # Create backup
    backup_existing_installation
}

#' Backup existing installation
backup_existing_installation() {
    local backup_dir="${INSTALL_DIR}.backup.$(date +%Y%m%d_%H%M%S)"

    info "Criando backup em: $backup_dir"

    # Move existing to backup
    mv "$INSTALL_DIR" "$backup_dir"

    success "Backup criado"

    # Keep data and config from backup
    if [[ -d "$backup_dir/data" ]]; then
        info "Preservando dados existentes..."
        mkdir -p "$INSTALL_DIR"
        cp -r "$backup_dir/data" "$INSTALL_DIR/data"
    fi

    if [[ -d "$backup_dir/config" ]]; then
        info "Preservando configurações existentes..."
        mkdir -p "$INSTALL_DIR"
        cp -r "$backup_dir/config" "$INSTALL_DIR/config"
    fi
}

#' Set appropriate permissions on directories
set_directory_permissions() {
    # Main directory - user only
    chmod 755 "$INSTALL_DIR"

    # Executables
    chmod 755 "$INSTALL_DIR/bin"

    # Sensitive directories - restrict access
    chmod 700 "$INSTALL_DIR/config"
    chmod 700 "$INSTALL_DIR/logs"

    # Data directories
    chmod 755 "$INSTALL_DIR/data"
    chmod 755 "$INSTALL_DIR/output"

    # Cache - can be world-readable
    chmod 755 "$INSTALL_DIR/cache"
}

#' Create .gitignore for local development
create_gitignore() {
    cat > "$INSTALL_DIR/.gitignore" << 'GITIGNORE'
# PrevCargaONS Installation Directory
# This directory is managed by the installer

# Logs
logs/
*.log

# Cache
cache/
*.cache

# Data (may contain sensitive info)
data/

# Output (generated files)
output/

# R environment
renv/library/
renv/staging/
renv/cache/
.Rprofile

# Configuration (may contain credentials)
config/*.yaml
!config/config.example.yaml

# Version file
.version

# Backup directories
*.backup.*
GITIGNORE
}

#' Get installation directory size
get_install_size() {
    du -sh "$INSTALL_DIR" 2>/dev/null | cut -f1
}

#' Verify directory structure
verify_directory_structure() {
    local required_dirs=(
        "bin"
        "src"
        "renv"
        "data"
        "config"
        "logs"
    )

    for dir in "${required_dirs[@]}"; do
        if [[ ! -d "$INSTALL_DIR/$dir" ]]; then
            die "Diretório faltando: $INSTALL_DIR/$dir"
        fi
    done

    success "Estrutura de diretórios verificada"
}
```

### Directory Structure Diagram

```
~/.prevcarga/
├── .version                    # Installation version
├── .gitignore                  # Git ignore rules
├── bin/                        # Executable scripts
│   └── prevcarga               # Main launcher script
├── src/                        # R source code
│   └── prevcarga_shell.R       # Main R shell script
├── renv/                       # R environment
│   ├── library/                # Installed packages
│   ├── staging/                # Package staging
│   └── settings.json           # renv settings
├── data/                       # Local data storage
│   ├── carga/                  # Load data (Parquet)
│   ├── previsao/               # Forecast outputs
│   └── models/                 # Saved model artifacts
├── output/                     # Generated reports
│   └── reports/                # HTML reports
├── config/                     # Configuration files
│   └── config.yaml             # Main configuration
├── logs/                       # Log files
│   └── prevcarga.log           # Application log
└── cache/                      # Temporary cache
    └── features/               # Cached features
```

---

## Test Cases

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| TC-001 | Fresh install | All directories created |
| TC-002 | Existing install | Backup created |
| TC-003 | Same version reinstall | Prompt shown |
| TC-004 | Permissions on config | 700 (user only) |
| TC-005 | Permissions on bin | 755 (executable) |
| TC-006 | .gitignore created | File exists |
| TC-007 | .version created | Contains version |
| TC-008 | Data preserved on upgrade | Data copied from backup |
| TC-009 | Config preserved on upgrade | Config copied from backup |
| TC-010 | Verify structure | All required dirs exist |

---

## Dependencies

- PC-086-11: Installer Script Structure

---

## Definition of Done

- [ ] All directories created correctly
- [ ] Permissions set appropriately
- [ ] Backup mechanism working
- [ ] Data/config preservation working
- [ ] .gitignore created
- [ ] .version file created
- [ ] Directory verification working
- [ ] Code reviewed and approved
- [ ] Merged to develop branch

---

## Notes

- Keep config and data on upgrades
- Log rotation handled by StructuredLogger
- Cache can be cleared safely
- Consider XDG Base Directory spec for future
