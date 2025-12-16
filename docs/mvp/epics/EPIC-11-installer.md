# EPIC-11: One-Line Installer

**Duration:** 1 week
**Dependencies:** EPIC-10
**Reference:** [MVP Plan R - Phase 11](../mvp-plan-r.md#phase-11-one-line-installer-1-week)

---

## Objective

Create a one-line curl installer that automatically sets up PrevCargaONS on any supported system (Ubuntu, Debian, Fedora, CentOS, macOS), including R installation, renv setup, and PATH configuration.

---

## Scope

This epic covers:
- One-line curl installer script (`install.sh`)
- Automatic R installation (if not present)
- renv environment setup with locked dependencies
- Embedded R shell extraction and configuration
- Cross-platform support (Linux and macOS)
- PATH configuration for user shell

**Out of Scope:**
- Windows support (future enhancement)
- Uninstaller script (documented manual removal)
- Package repository hosting (separate infrastructure task)

---

## Tasks

### T-11.1: Create Installer Script Structure
- [ ] Create `scripts/install.sh` with bash best practices
- [ ] Implement error handling with `set -euo pipefail`
- [ ] Create colored output functions (info, success, warn, error, step)
- [ ] Add ASCII banner for PrevCargaONS branding
  ```bash
  #!/usr/bin/env bash
  # ══════════════════════════════════════════════════════════════════════════════
  # PrevCargaONS - Instalador One-Line
  # ══════════════════════════════════════════════════════════════════════════════
  #
  # INSTALAÇÃO:
  #   curl -fsSL https://ons.org.br/prevcarga/install.sh | bash
  #
  # ══════════════════════════════════════════════════════════════════════════════

  set -euo pipefail

  VERSION="1.0.0"
  INSTALL_DIR="${HOME}/.prevcarga"

  RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'
  CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

  info()    { echo -e "  ${CYAN}ℹ${RESET}  $1"; }
  success() { echo -e "  ${GREEN}✓${RESET}  $1"; }
  warn()    { echo -e "  ${YELLOW}!${RESET}  $1"; }
  error()   { echo -e "  ${RED}✗${RESET}  $1" >&2; }
  step()    { echo -e "\n${CYAN}━━━ $1 ━━━${RESET}\n"; }
  ```

### T-11.2: Implement OS Detection
- [ ] Detect Linux distributions (Ubuntu, Debian, Fedora, CentOS, RHEL, Rocky)
- [ ] Detect macOS
- [ ] Validate supported systems and fail gracefully for unsupported ones
  ```bash
  step "1/5 - Detectando sistema"

  if [[ "$OSTYPE" == "linux-gnu"* ]]; then
      [[ -f /etc/os-release ]] && . /etc/os-release && OS="$ID"
  elif [[ "$OSTYPE" == "darwin"* ]]; then
      OS="macos"
  else
      error "Sistema não suportado: $OSTYPE"; exit 1
  fi
  info "Sistema: ${BOLD}$OS${RESET}"
  ```

### T-11.3: Implement R Installation
- [ ] Check if R is already installed
- [ ] Install R via package manager per distribution:
  - Ubuntu/Debian: `apt-get` with CRAN repository
  - Fedora/CentOS: `dnf`/`yum` with EPEL
  - macOS: Homebrew
- [ ] Install system dependencies for R packages:
  ```bash
  install_r() {
      warn "R não encontrado. Instalando (requer sudo)..."
      case "$OS" in
          ubuntu|debian|linuxmint|pop)
              sudo apt-get update -qq
              sudo apt-get install -y software-properties-common
              sudo add-apt-repository -y \
                "deb https://cloud.r-project.org/bin/linux/ubuntu $(lsb_release -cs)-cran40/"
              sudo apt-get update -qq
              sudo apt-get install -y r-base r-base-dev \
                libcurl4-openssl-dev libssl-dev libxml2-dev
              ;;
          fedora|centos|rhel|rocky)
              sudo dnf install -y epel-release 2>/dev/null || \
                sudo yum install -y epel-release
              sudo dnf install -y R R-devel 2>/dev/null || \
                sudo yum install -y R R-devel
              ;;
          macos)
              command -v brew &>/dev/null || \
                /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
              brew install r
              ;;
          *)
              error "Instale R manualmente: https://cloud.r-project.org"
              exit 1
              ;;
      esac
  }
  ```

### T-11.4: Create Directory Structure
- [ ] Create installation directory: `~/.prevcarga`
- [ ] Create subdirectories:
  ```bash
  mkdir -p "$INSTALL_DIR"/{bin,src,renv,data,output,config,logs}
  ```
- [ ] Set appropriate permissions

### T-11.5: Embed and Extract R Shell
- [ ] Base64 encode the R shell script
- [ ] Embed in installer as `EMBEDDED_SHELL` variable
- [ ] Extract and decode during installation:
  ```bash
  EMBEDDED_SHELL='<base64-encoded-shell>'

  step "4/5 - Extraindo CLI"

  echo "$EMBEDDED_SHELL" | base64 -d > "$INSTALL_DIR/src/prevcarga_shell.R"
  chmod +x "$INSTALL_DIR/src/prevcarga_shell.R"
  success "Shell R extraído"
  ```

### T-11.6: Create Launcher Script
- [ ] Create bash launcher at `~/.prevcarga/bin/prevcarga`:
  ```bash
  cat > "$INSTALL_DIR/bin/prevcarga" << LAUNCHER
  #!/usr/bin/env bash
  export TERM="\${TERM:-xterm-256color}"
  export LANG="\${LANG:-pt_BR.UTF-8}"
  exec Rscript "$INSTALL_DIR/src/prevcarga_shell.R" "\$@"
  LAUNCHER
  chmod +x "$INSTALL_DIR/bin/prevcarga"
  ```

### T-11.7: Configure renv Environment
- [ ] Create `renv.lock` with pinned package versions:
  ```json
  {
    "R": {"Version": "4.3.3", "Repositories": [
      {"Name": "CRAN", "URL": "https://cloud.r-project.org"}
    ]},
    "Packages": {
      "renv": {"Package": "renv", "Version": "1.0.7"},
      "R6": {"Package": "R6", "Version": "2.5.1"},
      "data.table": {"Package": "data.table", "Version": "1.15.4"},
      "arrow": {"Package": "arrow", "Version": "15.0.0"},
      "yaml": {"Package": "yaml", "Version": "2.3.8"},
      "paws": {"Package": "paws", "Version": "0.5.0"},
      "future": {"Package": "future", "Version": "1.33.1"},
      "future.apply": {"Package": "future.apply", "Version": "1.11.1"},
      "cli": {"Package": "cli", "Version": "3.6.2"},
      "optparse": {"Package": "optparse", "Version": "1.7.4"},
      "checkmate": {"Package": "checkmate", "Version": "2.3.1"},
      "jsonlite": {"Package": "jsonlite", "Version": "1.8.8"},
      "highcharter": {"Package": "highcharter", "Version": "0.9.4"},
      "rmarkdown": {"Package": "rmarkdown", "Version": "2.25"},
      "htmlwidgets": {"Package": "htmlwidgets", "Version": "1.6.4"},
      "forecast": {"Package": "forecast", "Version": "8.22.0"}
    }
  }
  ```
- [ ] Run `renv::init()` and `renv::restore()` during installation

### T-11.8: Configure User PATH
- [ ] Detect user shell (bash/zsh)
- [ ] Add PATH entry to appropriate rc file:
  ```bash
  SHELL_RC="$HOME/.bashrc"
  [[ "$SHELL" == *"zsh"* ]] && SHELL_RC="$HOME/.zshrc"

  if ! grep -q "PrevCargaONS" "$SHELL_RC" 2>/dev/null; then
      echo -e "\n# PrevCargaONS\nexport PATH=\"$INSTALL_DIR/bin:\$PATH\"" >> "$SHELL_RC"
      success "PATH adicionado a $SHELL_RC"
  fi
  ```

### T-11.9: Create Default Configuration
- [ ] Generate default `config.yaml` in `~/.prevcarga/config/`:
  ```yaml
  project:
    name: "PrevCargaONS"
    version: "1.0.0"
    seed: 42

  storage:
    backend: local
    local:
      base_path: ~/.prevcarga/data

  logging:
    level: INFO
    format: json
    handlers:
      - console
      - file: ~/.prevcarga/logs/prevcarga.log
  ```

### T-11.10: Display Success Message
- [ ] Show installation summary with colored output:
  ```bash
  echo -e "${GREEN}══════════════════════════════════════════════════════════════════${RESET}"
  echo -e "${GREEN}  ✓ INSTALAÇÃO CONCLUÍDA!${RESET}"
  echo -e "${GREEN}══════════════════════════════════════════════════════════════════${RESET}"
  echo ""
  echo "  Para usar:"
  echo ""
  echo -e "    ${CYAN}source $SHELL_RC${RESET}    # Carrega PATH"
  echo -e "    ${CYAN}prevcarga${RESET}           # Inicia o shell"
  echo ""
  ```

### T-11.11: Create Build Script for Installer
- [ ] Create `scripts/build_installer.sh` to generate installer:
  ```bash
  #!/usr/bin/env bash
  # Compiles the installer by:
  # 1. Base64-encoding the R shell
  # 2. Embedding it in install.sh template
  # 3. Generating final install.sh

  SHELL_SCRIPT="inst/shell/prevcarga_shell.R"
  TEMPLATE="scripts/install_template.sh"
  OUTPUT="dist/install.sh"

  ENCODED=$(base64 -w0 "$SHELL_SCRIPT")
  sed "s|__EMBEDDED_SHELL__|$ENCODED|g" "$TEMPLATE" > "$OUTPUT"
  chmod +x "$OUTPUT"
  ```

### T-11.12: Write Tests
- [ ] Create `tests/test-installer.sh` with bats or simple bash tests
- [ ] Test OS detection
- [ ] Test R installation check
- [ ] Test directory creation
- [ ] Test PATH configuration
- [ ] Integration test on clean Docker containers

---

## Acceptance Criteria

- [ ] `curl -fsSL <url>/install.sh | bash` completes without errors
- [ ] R is installed automatically if not present
- [ ] All required R packages are installed via renv
- [ ] `prevcarga` command is available after `source ~/.bashrc`
- [ ] `prevcarga --help` displays help message
- [ ] `prevcarga interactive` launches the interactive shell
- [ ] Works on Ubuntu 20.04+, Debian 11+, Fedora 38+, macOS 12+
- [ ] Installation completes in <5 minutes on average

---

## Definition of Done

- [ ] All tasks completed
- [ ] Installer tested on all supported platforms
- [ ] Documentation includes troubleshooting section
- [ ] CI/CD pipeline builds installer on release

---

## Files Created/Modified

| File | Action | Description |
|------|--------|-------------|
| `scripts/install.sh` | Create | Main installer script |
| `scripts/install_template.sh` | Create | Template with placeholder |
| `scripts/build_installer.sh` | Create | Build script for installer |
| `inst/shell/prevcarga_shell.R` | Create | Standalone R shell |
| `tests/test-installer.sh` | Create | Installer tests |
| `docs/installation.md` | Create | Installation guide |

---

## Installation Flow

```
curl -fsSL https://ons.org.br/prevcarga/install.sh | bash
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  1. Detect OS (Ubuntu/Debian/Fedora/CentOS/macOS)           │
├─────────────────────────────────────────────────────────────┤
│  2. Check/Install R                                          │
│     ├── apt-get (Debian/Ubuntu)                             │
│     ├── dnf/yum (Fedora/CentOS)                             │
│     └── brew (macOS)                                        │
├─────────────────────────────────────────────────────────────┤
│  3. Create ~/.prevcarga/ directory structure                │
│     ├── bin/      (launcher scripts)                        │
│     ├── src/      (R source files)                          │
│     ├── renv/     (R environment)                           │
│     ├── data/     (local data storage)                      │
│     ├── config/   (YAML configurations)                     │
│     └── logs/     (log files)                               │
├─────────────────────────────────────────────────────────────┤
│  4. Extract embedded R shell (base64)                       │
├─────────────────────────────────────────────────────────────┤
│  5. Configure renv and install packages                     │
├─────────────────────────────────────────────────────────────┤
│  6. Create launcher script                                  │
├─────────────────────────────────────────────────────────────┤
│  7. Add to PATH (~/.bashrc or ~/.zshrc)                     │
├─────────────────────────────────────────────────────────────┤
│  8. Display success message                                 │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
           source ~/.bashrc && prevcarga
```

---

## Supported Platforms

| Platform | Version | Package Manager | Status |
|----------|---------|-----------------|--------|
| Ubuntu | 20.04+ | apt | Primary |
| Debian | 11+ | apt | Primary |
| Linux Mint | 20+ | apt | Primary |
| Pop!_OS | 20.04+ | apt | Primary |
| Fedora | 38+ | dnf | Supported |
| CentOS | 8+ | dnf/yum | Supported |
| RHEL | 8+ | dnf/yum | Supported |
| Rocky Linux | 8+ | dnf/yum | Supported |
| macOS | 12+ | brew | Supported |
| Windows | - | - | Not Supported |

---

## Command Reference After Installation

```bash
# Start interactive shell
prevcarga

# Run in demo mode
prevcarga --demo

# Run specific command
prevcarga forecast SE
prevcarga status
prevcarga areas

# Show help
prevcarga --help

# Show version
prevcarga --version
```
