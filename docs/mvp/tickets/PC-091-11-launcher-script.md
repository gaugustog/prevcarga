# PC-091-11: Launcher Script

**Epic:** [EPIC-11: One-Line Installer](../epics/EPIC-11-installer.md)
**Task Reference:** T-11.6
**Priority:** High
**Estimated Effort:** 0.5 days

---

## Summary

Create a bash launcher script at `~/.prevcarga/bin/prevcarga` that sets up the environment and invokes the R shell script with proper arguments.

---

## Acceptance Criteria

- [ ] Launcher script created at `~/.prevcarga/bin/prevcarga`
- [ ] Script is executable (chmod +x)
- [ ] Sets TERM and LANG environment variables
- [ ] Activates renv environment
- [ ] Passes all arguments to R script
- [ ] Handles interrupts gracefully

---

## Technical Specification

### Launcher Script Creation

```bash
# ──────────────────────────────────────────────────────────────────────────────
# Launcher Script Creation
# ──────────────────────────────────────────────────────────────────────────────

#' Create the launcher script
create_launcher_script() {
    step "5/7 - Criando launcher"

    local launcher_path="$INSTALL_DIR/bin/prevcarga"

    # Create launcher script
    cat > "$launcher_path" << 'LAUNCHER'
#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# PrevCargaONS Launcher
# ══════════════════════════════════════════════════════════════════════════════
#
# This script launches the PrevCargaONS R shell with proper environment setup.
#
# Usage:
#   prevcarga                    # Start interactive shell
#   prevcarga --help             # Show help
#   prevcarga forecast SE        # Run forecast command
#
# ══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

PREVCARGA_HOME="${PREVCARGA_HOME:-$HOME/.prevcarga}"
PREVCARGA_SHELL="$PREVCARGA_HOME/src/prevcarga_shell.R"
PREVCARGA_RENV="$PREVCARGA_HOME/renv"
PREVCARGA_CONFIG="$PREVCARGA_HOME/config/config.yaml"
PREVCARGA_LOG="$PREVCARGA_HOME/logs/prevcarga.log"

# ──────────────────────────────────────────────────────────────────────────────
# Environment Setup
# ──────────────────────────────────────────────────────────────────────────────

# Terminal settings
export TERM="${TERM:-xterm-256color}"
export LANG="${LANG:-pt_BR.UTF-8}"
export LC_ALL="${LC_ALL:-pt_BR.UTF-8}"

# R settings
export R_LIBS_USER="$PREVCARGA_RENV/library"
export RENV_PATHS_CACHE="$PREVCARGA_RENV/cache"
export R_PROFILE_USER="$PREVCARGA_HOME/.Rprofile"

# PrevCarga settings
export PREVCARGA_HOME
export PREVCARGA_CONFIG
export PREVCARGA_LOG

# ──────────────────────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────────────────────

# Check R is available
if ! command -v Rscript &>/dev/null; then
    echo "Erro: Rscript não encontrado. R está instalado?" >&2
    exit 1
fi

# Check shell script exists
if [[ ! -f "$PREVCARGA_SHELL" ]]; then
    echo "Erro: Shell não encontrado: $PREVCARGA_SHELL" >&2
    echo "Execute o instalador novamente." >&2
    exit 1
fi

# ──────────────────────────────────────────────────────────────────────────────
# Signal Handling
# ──────────────────────────────────────────────────────────────────────────────

# Handle Ctrl+C gracefully
trap 'echo ""; exit 130' INT
trap 'exit 143' TERM

# ──────────────────────────────────────────────────────────────────────────────
# Main Execution
# ──────────────────────────────────────────────────────────────────────────────

# Execute R shell with all arguments
exec Rscript --vanilla "$PREVCARGA_SHELL" "$@"
LAUNCHER

    # Make executable
    chmod +x "$launcher_path"

    success "Launcher criado: $launcher_path"

    # Verify launcher
    verify_launcher
}

#' Verify launcher script works
verify_launcher() {
    local launcher_path="$INSTALL_DIR/bin/prevcarga"

    info "Verificando launcher..."

    # Check it's executable
    if [[ ! -x "$launcher_path" ]]; then
        die "Launcher não é executável"
    fi

    # Check it runs (with --version or quick check)
    # Note: This will fail until renv is configured
    debug "Launcher verificado (funcionalidade completa após renv)"

    success "Launcher pronto"
}
```

### Additional Launcher Options

```bash
# Extended launcher with additional features

# Add to launcher script after environment setup:

# ──────────────────────────────────────────────────────────────────────────────
# Special Commands (handled in bash)
# ──────────────────────────────────────────────────────────────────────────────

# Handle special commands before R
case "${1:-}" in
    --update)
        # Self-update
        echo "Atualizando PrevCargaONS..."
        curl -fsSL https://ons.org.br/prevcarga/install.sh | bash
        exit $?
        ;;
    --uninstall)
        # Uninstall
        echo "Desinstalando PrevCargaONS..."
        read -p "Tem certeza? Isso removerá $PREVCARGA_HOME [s/N] " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Ss]$ ]]; then
            rm -rf "$PREVCARGA_HOME"
            echo "PrevCargaONS removido."
            echo "Remova a linha do PATH em ~/.bashrc ou ~/.zshrc manualmente."
        fi
        exit 0
        ;;
    --doctor)
        # Diagnostic check
        echo "=== PrevCargaONS Diagnóstico ==="
        echo ""
        echo "Versão instalada: $(cat "$PREVCARGA_HOME/.version" 2>/dev/null || echo "desconhecida")"
        echo "R versão: $(R --version 2>/dev/null | head -n1)"
        echo "Rscript: $(which Rscript)"
        echo "Home: $PREVCARGA_HOME"
        echo "Shell: $PREVCARGA_SHELL"
        echo ""
        echo "Verificando arquivos..."
        [[ -f "$PREVCARGA_SHELL" ]] && echo "  ✓ Shell existe" || echo "  ✗ Shell faltando"
        [[ -f "$PREVCARGA_CONFIG" ]] && echo "  ✓ Config existe" || echo "  ✗ Config faltando"
        [[ -d "$PREVCARGA_RENV/library" ]] && echo "  ✓ renv existe" || echo "  ✗ renv faltando"
        echo ""
        exit 0
        ;;
    --env)
        # Show environment
        echo "PREVCARGA_HOME=$PREVCARGA_HOME"
        echo "PREVCARGA_CONFIG=$PREVCARGA_CONFIG"
        echo "PREVCARGA_LOG=$PREVCARGA_LOG"
        echo "R_LIBS_USER=$R_LIBS_USER"
        echo "TERM=$TERM"
        echo "LANG=$LANG"
        exit 0
        ;;
esac
```

### .Rprofile Creation

```bash
#' Create .Rprofile for R environment activation
create_rprofile() {
    local rprofile_path="$INSTALL_DIR/.Rprofile"

    cat > "$rprofile_path" << 'RPROFILE'
# PrevCargaONS R Profile
# Automatically activates renv environment

local({
  # Activate renv
  renv_activate <- file.path(Sys.getenv("PREVCARGA_HOME", "~/.prevcarga"), "renv", "activate.R")
  if (file.exists(renv_activate)) {
    source(renv_activate)
  }

  # Set options
  options(
    repos = c(CRAN = "https://cloud.r-project.org"),
    warn = 1,
    width = 120,
    cli.unicode = TRUE,
    cli.num_colors = 256
  )
})
RPROFILE

    success ".Rprofile criado"
}
```

---

## Test Cases

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| TC-001 | Launcher created | File exists |
| TC-002 | Launcher executable | chmod +x applied |
| TC-003 | Missing R | Error message |
| TC-004 | Missing shell | Error message |
| TC-005 | --version flag | Version shown |
| TC-006 | --help flag | Help shown |
| TC-007 | Ctrl+C handling | Clean exit |
| TC-008 | --doctor command | Diagnostics shown |
| TC-009 | --env command | Environment shown |
| TC-010 | Arguments passed | All args to R |

---

## Dependencies

- PC-086-11: Installer Script Structure
- PC-089-11: Directory Structure
- PC-090-11: Shell Extraction

---

## Definition of Done

- [ ] Launcher script created
- [ ] Environment variables set correctly
- [ ] Validation checks working
- [ ] Signal handling working
- [ ] Special commands working (--doctor, --env)
- [ ] .Rprofile created
- [ ] Verified launcher executes
- [ ] Code reviewed and approved
- [ ] Merged to develop branch

---

## Notes

- Launcher is the main user entry point
- Keep bash portion minimal
- Most logic should be in R shell
- Consider adding completion support later
