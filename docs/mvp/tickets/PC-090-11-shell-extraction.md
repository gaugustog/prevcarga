# PC-090-11: Shell Extraction

**Epic:** [EPIC-11: One-Line Installer](../epics/EPIC-11-installer.md)
**Task Reference:** T-11.5
**Priority:** High
**Estimated Effort:** 0.5 days

---

## Summary

Extract the embedded R shell script from the installer. The R shell is base64-encoded and embedded in the installer script, then decoded and extracted during installation.

---

## Acceptance Criteria

- [ ] Embedded shell variable contains base64-encoded R script
- [ ] Shell extracted to `~/.prevcarga/src/prevcarga_shell.R`
- [ ] Extracted script is executable
- [ ] Extraction verified with checksum
- [ ] Fallback to download if embedded fails

---

## Technical Specification

### Shell Extraction Functions

```bash
# ──────────────────────────────────────────────────────────────────────────────
# Shell Extraction
# ──────────────────────────────────────────────────────────────────────────────

# Embedded R Shell (Base64 Encoded)
# This will be replaced by build_installer.sh with actual content
EMBEDDED_SHELL='__EMBEDDED_SHELL__'

# Expected checksum of the decoded shell (SHA256)
EMBEDDED_SHELL_CHECKSUM='__EMBEDDED_SHELL_CHECKSUM__'

#' Extract and install the R shell script
extract_shell() {
    step "4/7 - Extraindo CLI"

    local shell_path="$INSTALL_DIR/src/prevcarga_shell.R"
    local temp_shell=$(mktemp)

    # Try embedded extraction first
    if extract_embedded_shell "$temp_shell"; then
        success "Shell extraído do instalador"
    else
        warn "Falha ao extrair shell embutido, baixando..."
        download_shell "$temp_shell"
    fi

    # Verify checksum
    verify_shell_checksum "$temp_shell"

    # Move to final location
    mv "$temp_shell" "$shell_path"
    chmod +x "$shell_path"

    success "CLI instalada em: $shell_path"
}

#' Extract embedded shell from installer
#' @param output_path Path to write extracted shell
#' @return 0 on success, 1 on failure
extract_embedded_shell() {
    local output_path="$1"

    # Check if embedded shell is present (not placeholder)
    if [[ "$EMBEDDED_SHELL" == "__EMBEDDED_SHELL__" ]]; then
        debug "Shell não embutido (desenvolvimento)"
        return 1
    fi

    # Decode base64
    if ! echo "$EMBEDDED_SHELL" | base64 -d > "$output_path" 2>/dev/null; then
        error "Falha ao decodificar shell base64"
        return 1
    fi

    # Verify it's a valid R script
    if ! head -n1 "$output_path" | grep -q "#!/usr/bin/env Rscript\|^#'"; then
        error "Shell extraído não parece ser script R válido"
        return 1
    fi

    return 0
}

#' Download shell script from repository
#' @param output_path Path to write downloaded shell
download_shell() {
    local output_path="$1"
    local download_url="${DOWNLOAD_URL}/prevcarga_shell.R"

    info "Baixando de: $download_url"

    if command_exists curl; then
        curl -fsSL "$download_url" -o "$output_path" || \
            die "Falha ao baixar shell via curl"
    elif command_exists wget; then
        wget -q "$download_url" -O "$output_path" || \
            die "Falha ao baixar shell via wget"
    else
        die "curl ou wget necessário para download"
    fi

    success "Shell baixado"
}

#' Verify shell script checksum
#' @param shell_path Path to shell script
verify_shell_checksum() {
    local shell_path="$1"

    # Skip if checksum not embedded (development mode)
    if [[ "$EMBEDDED_SHELL_CHECKSUM" == "__EMBEDDED_SHELL_CHECKSUM__" ]]; then
        warn "Checksum não verificado (modo desenvolvimento)"
        return 0
    fi

    info "Verificando integridade..."

    local actual_checksum
    if command_exists sha256sum; then
        actual_checksum=$(sha256sum "$shell_path" | cut -d' ' -f1)
    elif command_exists shasum; then
        actual_checksum=$(shasum -a 256 "$shell_path" | cut -d' ' -f1)
    else
        warn "sha256sum/shasum não encontrado, pulando verificação"
        return 0
    fi

    if [[ "$actual_checksum" != "$EMBEDDED_SHELL_CHECKSUM" ]]; then
        error "Checksum não confere!"
        error "Esperado: $EMBEDDED_SHELL_CHECKSUM"
        error "Obtido:   $actual_checksum"
        die "Arquivo corrompido ou modificado"
    fi

    success "Checksum verificado"
}

#' Verify shell script functionality
verify_shell_functionality() {
    local shell_path="$INSTALL_DIR/src/prevcarga_shell.R"

    info "Verificando funcionalidade..."

    # Test that R can parse the script
    if ! R --vanilla -q -e "parse('$shell_path')" &>/dev/null; then
        die "Shell R não pode ser parseado"
    fi

    success "Shell R válido"
}
```

### R Shell Script Structure

The R shell script (`prevcarga_shell.R`) should have this structure:

```r
#!/usr/bin/env Rscript
# ══════════════════════════════════════════════════════════════════════════════
# PrevCargaONS - Interactive Shell
# ══════════════════════════════════════════════════════════════════════════════
#
# Usage:
#   prevcarga                    # Start interactive shell
#   prevcarga --help             # Show help
#   prevcarga forecast SE        # Run forecast command
#
# ══════════════════════════════════════════════════════════════════════════════

# Activate renv environment
local({
  renv_dir <- file.path(Sys.getenv("HOME"), ".prevcarga", "renv")
  if (file.exists(file.path(renv_dir, "activate.R"))) {
    source(file.path(renv_dir, "activate.R"))
  }
})

# Load required packages
suppressPackageStartupMessages({
  library(cli)
  library(optparse)
})

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)

# Main entry point
main <- function(args) {
  if (length(args) == 0 || args[1] == "interactive") {
    start_interactive_shell()
  } else if (args[1] == "--help" || args[1] == "-h") {
    print_help()
  } else if (args[1] == "--version" || args[1] == "-v") {
    print_version()
  } else {
    execute_command(args)
  }
}

# Run main
main(args)
```

---

## Test Cases

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| TC-001 | Extract embedded shell | File created |
| TC-002 | Base64 decode | Valid R script |
| TC-003 | Download fallback | Shell downloaded |
| TC-004 | Checksum match | Verification passes |
| TC-005 | Checksum mismatch | Error and exit |
| TC-006 | Shell executable | chmod +x applied |
| TC-007 | R can parse shell | No parse errors |
| TC-008 | Missing curl/wget | Error message |
| TC-009 | Development mode | Skip checksum |
| TC-010 | Invalid base64 | Fallback to download |

---

## Dependencies

- PC-086-11: Installer Script Structure
- PC-089-11: Directory Structure

---

## Definition of Done

- [ ] Embedded shell extraction working
- [ ] Base64 decoding working
- [ ] Download fallback working
- [ ] Checksum verification working
- [ ] Shell marked executable
- [ ] R script validation working
- [ ] Error handling complete
- [ ] Code reviewed and approved
- [ ] Merged to develop branch

---

## Notes

- Base64 encoding done by build script
- Checksum prevents tampering
- Download fallback for development
- Keep shell script minimal for embedding
