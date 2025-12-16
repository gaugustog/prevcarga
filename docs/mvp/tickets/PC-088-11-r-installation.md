# PC-088-11: R Installation

**Epic:** [EPIC-11: One-Line Installer](../epics/EPIC-11-installer.md)
**Task Reference:** T-11.3
**Priority:** High
**Estimated Effort:** 1 day

---

## Summary

Implement automatic R installation via the appropriate package manager for each supported platform, including system dependencies required for R packages.

---

## Acceptance Criteria

- [ ] Checks if R is already installed
- [ ] Validates R version meets minimum (4.3.0)
- [ ] Installs R via apt (Ubuntu/Debian)
- [ ] Installs R via dnf/yum (Fedora/CentOS/RHEL)
- [ ] Installs R via Homebrew (macOS)
- [ ] Installs required system dependencies
- [ ] Installs from CRAN repository for latest R

---

## Technical Specification

### R Installation Functions

```bash
# ──────────────────────────────────────────────────────────────────────────────
# R Installation
# ──────────────────────────────────────────────────────────────────────────────

#' Check and install R if necessary
check_and_install_r() {
    step "2/7 - Verificando R"

    local r_version
    r_version=$(get_r_version)

    if [[ -z "$r_version" ]]; then
        warn "R não encontrado"
        install_r
    elif ! version_gte "$r_version" "$MIN_R_VERSION"; then
        warn "R $r_version encontrado, mas versão $MIN_R_VERSION+ é necessária"
        install_r
    else
        success "R $r_version já instalado"
    fi

    # Verify installation
    r_version=$(get_r_version)
    if [[ -z "$r_version" ]]; then
        die "Falha ao instalar R"
    fi

    info "Usando R versão: ${BOLD}$r_version${RESET}"

    # Install system dependencies
    install_system_deps
}

#' Install R based on detected OS
install_r() {
    info "Instalando R (requer sudo)..."

    case "$OS_FAMILY" in
        debian)
            install_r_debian
            ;;
        fedora|rhel)
            install_r_rhel
            ;;
        darwin)
            install_r_macos
            ;;
        *)
            die "Instalação automática de R não suportada para $OS_FAMILY"
            ;;
    esac

    success "R instalado com sucesso"
}

#' Install R on Debian/Ubuntu
install_r_debian() {
    info "Configurando repositório CRAN para $OS..."

    # Update package list
    maybe_sudo apt-get update -qq

    # Install prerequisites
    maybe_sudo apt-get install -y --no-install-recommends \
        software-properties-common \
        dirmngr \
        gnupg2 \
        apt-transport-https \
        ca-certificates

    # Add CRAN GPG key
    local keyserver="keyserver.ubuntu.com"
    local key="E298A3A825C0D65DFD57CBB651716619E084DAB9"

    maybe_sudo apt-key adv --keyserver "$keyserver" --recv-keys "$key" 2>/dev/null || {
        # Fallback method
        curl -fsSL "https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc" | \
            maybe_sudo tee /etc/apt/trusted.gpg.d/cran_ubuntu_key.asc >/dev/null
    }

    # Determine CRAN repository URL based on distribution
    local cran_repo
    case "$OS" in
        ubuntu)
            cran_repo="deb https://cloud.r-project.org/bin/linux/ubuntu ${OS_CODENAME}-cran40/"
            ;;
        debian)
            cran_repo="deb https://cloud.r-project.org/bin/linux/debian ${OS_CODENAME}-cran40/"
            ;;
        *)
            # For derivatives, try Ubuntu repo
            cran_repo="deb https://cloud.r-project.org/bin/linux/ubuntu focal-cran40/"
            ;;
    esac

    # Add CRAN repository
    echo "$cran_repo" | maybe_sudo tee /etc/apt/sources.list.d/cran.list >/dev/null

    # Update and install R
    maybe_sudo apt-get update -qq
    maybe_sudo apt-get install -y --no-install-recommends \
        r-base \
        r-base-dev
}

#' Install R on Fedora/CentOS/RHEL
install_r_rhel() {
    info "Instalando R via $PACKAGE_MANAGER..."

    # Install EPEL for CentOS/RHEL
    if [[ "$OS" != "fedora" ]]; then
        maybe_sudo $PACKAGE_MANAGER install -y epel-release 2>/dev/null || \
            maybe_sudo yum install -y epel-release
    fi

    # Enable PowerTools/CRB for CentOS 8+/Rocky
    if [[ "$OS" == "centos" || "$OS" == "rocky" || "$OS" == "almalinux" ]]; then
        maybe_sudo $PACKAGE_MANAGER config-manager --set-enabled powertools 2>/dev/null || \
            maybe_sudo $PACKAGE_MANAGER config-manager --set-enabled crb 2>/dev/null || \
            true
    fi

    # Install R
    maybe_sudo $PACKAGE_MANAGER install -y R R-devel
}

#' Install R on macOS
install_r_macos() {
    info "Instalando R via Homebrew..."

    # Check if Homebrew is installed
    if ! command_exists brew; then
        warn "Homebrew não encontrado. Instalando..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

        # Add Homebrew to PATH for this session
        if [[ -f /opt/homebrew/bin/brew ]]; then
            eval "$(/opt/homebrew/bin/brew shellenv)"
        elif [[ -f /usr/local/bin/brew ]]; then
            eval "$(/usr/local/bin/brew shellenv)"
        fi
    fi

    # Install R
    brew install r

    # Verify
    if ! command_exists R; then
        die "Falha ao instalar R via Homebrew"
    fi
}

#' Install system dependencies for R packages
install_system_deps() {
    info "Instalando dependências do sistema..."

    case "$OS_FAMILY" in
        debian)
            install_deps_debian
            ;;
        fedora|rhel)
            install_deps_rhel
            ;;
        darwin)
            install_deps_macos
            ;;
    esac

    success "Dependências instaladas"
}

#' Install system dependencies on Debian/Ubuntu
install_deps_debian() {
    maybe_sudo apt-get install -y --no-install-recommends \
        libcurl4-openssl-dev \
        libssl-dev \
        libxml2-dev \
        libfontconfig1-dev \
        libfreetype6-dev \
        libpng-dev \
        libtiff-dev \
        libjpeg-dev \
        libharfbuzz-dev \
        libfribidi-dev \
        libgit2-dev \
        pandoc \
        git
}

#' Install system dependencies on Fedora/CentOS/RHEL
install_deps_rhel() {
    maybe_sudo $PACKAGE_MANAGER install -y \
        libcurl-devel \
        openssl-devel \
        libxml2-devel \
        fontconfig-devel \
        freetype-devel \
        libpng-devel \
        libtiff-devel \
        libjpeg-turbo-devel \
        harfbuzz-devel \
        fribidi-devel \
        libgit2-devel \
        pandoc \
        git
}

#' Install system dependencies on macOS
install_deps_macos() {
    brew install \
        openssl \
        libxml2 \
        libgit2 \
        pandoc \
        git
}
```

### R Version Check

```bash
#' Get R version if installed
#' @return R version string or empty
get_r_version() {
    if command_exists R; then
        R --version 2>/dev/null | head -n1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -n1
    else
        echo ""
    fi
}

#' Check if R is functional
verify_r_installation() {
    info "Verificando instalação do R..."

    # Test basic R functionality
    if ! R --vanilla -q -e "cat(R.version.string)" &>/dev/null; then
        die "R instalado mas não funcional"
    fi

    # Test package installation capability
    if ! R --vanilla -q -e "install.packages('cli', repos='https://cloud.r-project.org', quiet=TRUE)" &>/dev/null; then
        warn "Possível problema com instalação de pacotes"
    fi

    success "R funcional"
}
```

---

## Test Cases

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| TC-001 | R not installed on Ubuntu | R installed via apt |
| TC-002 | R not installed on Fedora | R installed via dnf |
| TC-003 | R not installed on macOS | R installed via brew |
| TC-004 | R < 4.3.0 installed | New R version installed |
| TC-005 | R >= 4.3.0 installed | Skip installation |
| TC-006 | System deps on Ubuntu | All deps installed |
| TC-007 | System deps on CentOS | All deps installed |
| TC-008 | System deps on macOS | All deps installed |
| TC-009 | CRAN repo configured | Latest R available |
| TC-010 | R verification | R runs successfully |

---

## Dependencies

- PC-086-11: Installer Script Structure
- PC-087-11: OS Detection

---

## Definition of Done

- [ ] R installation working on all platforms
- [ ] System dependencies installed
- [ ] CRAN repository configured (Linux)
- [ ] Homebrew installed if needed (macOS)
- [ ] Version check working
- [ ] Tested on Docker containers
- [ ] Code reviewed and approved
- [ ] Merged to develop branch

---

## Notes

- CRAN provides up-to-date R packages
- System deps required for compiling R packages
- Homebrew is standard for macOS development tools
- Consider caching package downloads
