# PC-086-11: Installer Script Structure

**Epic:** [EPIC-11: One-Line Installer](../epics/EPIC-11-installer.md)
**Task Reference:** T-11.1
**Priority:** High
**Estimated Effort:** 0.5 days

---

## Summary

Create the foundational installer script with bash best practices, error handling, colored output functions, and ASCII banner for PrevCargaONS branding.

---

## Acceptance Criteria

- [ ] Installer script created at `scripts/install.sh`
- [ ] Uses `set -euo pipefail` for strict error handling
- [ ] Colored output functions implemented (info, success, warn, error, step)
- [ ] ASCII banner displayed on startup
- [ ] Version and installation directory constants defined
- [ ] Script is executable

---

## Technical Specification

### File Location
```
scripts/install.sh
scripts/install_template.sh
```

### Installer Script Foundation

```bash
#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# PrevCargaONS - Instalador One-Line
# ══════════════════════════════════════════════════════════════════════════════
#
# INSTALAÇÃO:
#   curl -fsSL https://ons.org.br/prevcarga/install.sh | bash
#
# REQUISITOS:
#   - Linux (Ubuntu/Debian/Fedora/CentOS) ou macOS
#   - Conexão com a internet
#   - sudo (para instalação do R, se necessário)
#
# ══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

VERSION="1.0.0"
INSTALL_DIR="${HOME}/.prevcarga"
MIN_R_VERSION="4.3.0"
REPO_URL="https://github.com/ons-br/prevcarga-R"
DOWNLOAD_URL="${REPO_URL}/releases/download/v${VERSION}"

# ──────────────────────────────────────────────────────────────────────────────
# Colors and Formatting
# ──────────────────────────────────────────────────────────────────────────────

# Check if terminal supports colors
if [[ -t 1 ]] && command -v tput &>/dev/null && [[ $(tput colors 2>/dev/null) -ge 8 ]]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[0;33m'
    BLUE='\033[0;34m'
    MAGENTA='\033[0;35m'
    CYAN='\033[0;36m'
    WHITE='\033[0;37m'
    BOLD='\033[1m'
    DIM='\033[2m'
    RESET='\033[0m'
else
    RED=''
    GREEN=''
    YELLOW=''
    BLUE=''
    MAGENTA=''
    CYAN=''
    WHITE=''
    BOLD=''
    DIM=''
    RESET=''
fi

# ──────────────────────────────────────────────────────────────────────────────
# Output Functions
# ──────────────────────────────────────────────────────────────────────────────

#' Print informational message
#' @param message Message to print
info() {
    echo -e "  ${CYAN}ℹ${RESET}  $1"
}

#' Print success message
#' @param message Message to print
success() {
    echo -e "  ${GREEN}✓${RESET}  $1"
}

#' Print warning message
#' @param message Message to print
warn() {
    echo -e "  ${YELLOW}!${RESET}  $1"
}

#' Print error message to stderr
#' @param message Message to print
error() {
    echo -e "  ${RED}✗${RESET}  $1" >&2
}

#' Print step header
#' @param step_text Step description
step() {
    echo -e "\n${CYAN}━━━ $1 ━━━${RESET}\n"
}

#' Print debug message (only if DEBUG is set)
#' @param message Message to print
debug() {
    if [[ "${DEBUG:-}" == "1" ]]; then
        echo -e "  ${DIM}[DEBUG] $1${RESET}"
    fi
}

# ──────────────────────────────────────────────────────────────────────────────
# ASCII Banner
# ──────────────────────────────────────────────────────────────────────────────

print_banner() {
    echo -e "${CYAN}"
    cat << 'EOF'
  ╔═══════════════════════════════════════════════════════════════╗
  ║                                                               ║
  ║   ██████╗ ██████╗ ███████╗██╗   ██╗ ██████╗ █████╗ ██████╗   ║
  ║   ██╔══██╗██╔══██╗██╔════╝██║   ██║██╔════╝██╔══██╗██╔══██╗  ║
  ║   ██████╔╝██████╔╝█████╗  ██║   ██║██║     ███████║██████╔╝  ║
  ║   ██╔═══╝ ██╔══██╗██╔══╝  ╚██╗ ██╔╝██║     ██╔══██║██╔══██╗  ║
  ║   ██║     ██║  ██║███████╗ ╚████╔╝ ╚██████╗██║  ██║██║  ██║  ║
  ║   ╚═╝     ╚═╝  ╚═╝╚══════╝  ╚═══╝   ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝  ║
  ║                                                               ║
  ║                     Previsão de Carga ONS                     ║
  ║                                                               ║
  ╚═══════════════════════════════════════════════════════════════╝
EOF
    echo -e "${RESET}"
    echo -e "  ${DIM}Versão ${VERSION}${RESET}"
    echo ""
}

# ──────────────────────────────────────────────────────────────────────────────
# Error Handling
# ──────────────────────────────────────────────────────────────────────────────

#' Cleanup function called on exit
cleanup() {
    local exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        echo ""
        error "Instalação falhou com código de saída: $exit_code"
        error "Para ajuda, visite: ${REPO_URL}/issues"
    fi
}

trap cleanup EXIT

#' Exit with error message
#' @param message Error message
#' @param code Exit code (default: 1)
die() {
    error "$1"
    exit "${2:-1}"
}

#' Check if command exists
#' @param command Command to check
command_exists() {
    command -v "$1" &>/dev/null
}

#' Require a command to be available
#' @param command Command name
#' @param package Package to install (for error message)
require_command() {
    local cmd="$1"
    local package="${2:-$1}"

    if ! command_exists "$cmd"; then
        die "Comando '$cmd' não encontrado. Instale: $package"
    fi
}

# ──────────────────────────────────────────────────────────────────────────────
# Utility Functions
# ──────────────────────────────────────────────────────────────────────────────

#' Compare semantic versions
#' @param version1 First version
#' @param version2 Second version
#' @return 0 if v1 >= v2, 1 otherwise
version_gte() {
    local v1="$1"
    local v2="$2"

    # Use sort -V for version comparison
    if [[ "$(printf '%s\n' "$v1" "$v2" | sort -V | head -n1)" == "$v2" ]]; then
        return 0
    else
        return 1
    fi
}

#' Get R version if installed
#' @return R version string or empty
get_r_version() {
    if command_exists R; then
        R --version 2>/dev/null | head -n1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -n1
    else
        echo ""
    fi
}

#' Check if running as root
is_root() {
    [[ "$(id -u)" -eq 0 ]]
}

#' Run command with sudo if not root
maybe_sudo() {
    if is_root; then
        "$@"
    else
        sudo "$@"
    fi
}

# ──────────────────────────────────────────────────────────────────────────────
# Main Entry Point
# ──────────────────────────────────────────────────────────────────────────────

main() {
    # Print banner
    print_banner

    info "Iniciando instalação do PrevCargaONS..."
    info "Diretório de instalação: ${BOLD}$INSTALL_DIR${RESET}"

    # Installation steps will be called here
    # step "1/7 - Detectando sistema"
    # ...
}

# Run main function
main "$@"
```

### Template Version (for build script)

```bash
# scripts/install_template.sh
# Same as above but with placeholder for embedded shell:

# ...
# After all functions, before main:

# ──────────────────────────────────────────────────────────────────────────────
# Embedded R Shell (Base64 Encoded)
# ──────────────────────────────────────────────────────────────────────────────

EMBEDDED_SHELL='__EMBEDDED_SHELL__'

# This will be replaced by build_installer.sh with actual base64-encoded content
```

---

## Test Cases

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| TC-001 | Run with set -e | Exits on first error |
| TC-002 | Run with set -u | Errors on undefined variable |
| TC-003 | Color output in terminal | Colors displayed |
| TC-004 | Color output in pipe | No color codes |
| TC-005 | info() function | Blue info icon |
| TC-006 | success() function | Green checkmark |
| TC-007 | warn() function | Yellow exclamation |
| TC-008 | error() function | Red X to stderr |
| TC-009 | step() function | Cyan separator line |
| TC-010 | Banner displays | ASCII art shown |

---

## Dependencies

None (foundational ticket)

---

## Definition of Done

- [ ] Installer script created
- [ ] Error handling implemented
- [ ] Output functions working
- [ ] ASCII banner displays correctly
- [ ] Colors work in terminal
- [ ] No colors in non-terminal output
- [ ] Script is shellcheck-clean
- [ ] Code reviewed and approved
- [ ] Merged to develop branch

---

## Notes

- Use `shellcheck` to validate bash syntax
- Test in both bash and zsh environments
- Consider adding `--no-color` flag for CI/CD
- Banner can be customized for different releases
