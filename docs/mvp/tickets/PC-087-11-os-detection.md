# PC-087-11: OS Detection

**Epic:** [EPIC-11: One-Line Installer](../epics/EPIC-11-installer.md)
**Task Reference:** T-11.2
**Priority:** High
**Estimated Effort:** 0.5 days

---

## Summary

Implement operating system detection to identify Linux distributions (Ubuntu, Debian, Fedora, CentOS, RHEL, Rocky) and macOS, with graceful failure for unsupported systems.

---

## Acceptance Criteria

- [ ] Detects Ubuntu, Debian, Linux Mint, Pop!_OS
- [ ] Detects Fedora, CentOS, RHEL, Rocky Linux
- [ ] Detects macOS (12+)
- [ ] Returns appropriate OS identifier
- [ ] Fails gracefully with helpful message for unsupported systems
- [ ] Detects architecture (x86_64, arm64)

---

## Technical Specification

### OS Detection Functions

```bash
# ──────────────────────────────────────────────────────────────────────────────
# OS Detection
# ──────────────────────────────────────────────────────────────────────────────

# Global variables set by detect_os
OS=""
OS_VERSION=""
OS_CODENAME=""
OS_FAMILY=""
ARCH=""
PACKAGE_MANAGER=""

#' Detect operating system and set global variables
detect_os() {
    step "1/7 - Detectando sistema"

    # Detect architecture
    ARCH="$(uname -m)"
    debug "Arquitetura: $ARCH"

    # Detect OS
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        detect_linux
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        detect_macos
    else
        die "Sistema operacional não suportado: $OSTYPE" 2
    fi

    # Display detected system
    info "Sistema: ${BOLD}$OS${RESET} $OS_VERSION ($ARCH)"
    info "Família: ${BOLD}$OS_FAMILY${RESET}"
    info "Gerenciador de pacotes: ${BOLD}$PACKAGE_MANAGER${RESET}"

    # Validate supported
    validate_os_support
}

#' Detect Linux distribution
detect_linux() {
    if [[ -f /etc/os-release ]]; then
        # Source os-release for distribution info
        . /etc/os-release

        OS="${ID:-unknown}"
        OS_VERSION="${VERSION_ID:-unknown}"
        OS_CODENAME="${VERSION_CODENAME:-}"

        # Determine OS family and package manager
        case "$OS" in
            ubuntu|debian|linuxmint|pop|elementary|zorin)
                OS_FAMILY="debian"
                PACKAGE_MANAGER="apt"
                ;;
            fedora)
                OS_FAMILY="fedora"
                PACKAGE_MANAGER="dnf"
                ;;
            centos|rhel|rocky|almalinux|ol)
                OS_FAMILY="rhel"
                # CentOS 8+ and RHEL 8+ use dnf
                if version_gte "${OS_VERSION%%.*}" "8"; then
                    PACKAGE_MANAGER="dnf"
                else
                    PACKAGE_MANAGER="yum"
                fi
                ;;
            opensuse*|sles)
                OS_FAMILY="suse"
                PACKAGE_MANAGER="zypper"
                ;;
            arch|manjaro)
                OS_FAMILY="arch"
                PACKAGE_MANAGER="pacman"
                ;;
            *)
                OS_FAMILY="unknown"
                PACKAGE_MANAGER="unknown"
                ;;
        esac

    elif [[ -f /etc/redhat-release ]]; then
        # Fallback for older RHEL-based systems
        if grep -q "CentOS" /etc/redhat-release; then
            OS="centos"
        elif grep -q "Fedora" /etc/redhat-release; then
            OS="fedora"
        else
            OS="rhel"
        fi
        OS_VERSION=$(grep -oE '[0-9]+\.[0-9]+' /etc/redhat-release | head -n1)
        OS_FAMILY="rhel"
        PACKAGE_MANAGER="yum"

    elif [[ -f /etc/debian_version ]]; then
        # Fallback for older Debian systems
        OS="debian"
        OS_VERSION=$(cat /etc/debian_version)
        OS_FAMILY="debian"
        PACKAGE_MANAGER="apt"

    else
        die "Não foi possível detectar a distribuição Linux" 2
    fi
}

#' Detect macOS version
detect_macos() {
    OS="macos"
    OS_FAMILY="darwin"
    PACKAGE_MANAGER="brew"

    # Get macOS version
    OS_VERSION=$(sw_vers -productVersion 2>/dev/null || echo "unknown")
    OS_CODENAME=$(awk '/SOFTWARE LICENSE AGREEMENT FOR macOS/' \
        '/System/Library/CoreServices/Setup Assistant.app/Contents/Resources/en.lproj/OSXSoftwareLicense.rtf' \
        2>/dev/null | awk -F 'macOS ' '{print $NF}' | awk '{print $1}' || echo "")

    debug "macOS version: $OS_VERSION"
}

#' Validate that the detected OS is supported
validate_os_support() {
    local supported=false

    case "$OS_FAMILY" in
        debian)
            case "$OS" in
                ubuntu)
                    # Ubuntu 20.04+
                    version_gte "${OS_VERSION%%.*}" "20" && supported=true
                    ;;
                debian)
                    # Debian 11+
                    version_gte "${OS_VERSION%%.*}" "11" && supported=true
                    ;;
                linuxmint)
                    # Linux Mint 20+
                    version_gte "${OS_VERSION%%.*}" "20" && supported=true
                    ;;
                pop)
                    # Pop!_OS 20.04+
                    version_gte "${OS_VERSION%%.*}" "20" && supported=true
                    ;;
                *)
                    # Other Debian-based might work
                    warn "Distribuição $OS não testada oficialmente"
                    supported=true
                    ;;
            esac
            ;;
        fedora)
            # Fedora 38+
            version_gte "${OS_VERSION%%.*}" "38" && supported=true
            ;;
        rhel)
            # RHEL/CentOS/Rocky 8+
            version_gte "${OS_VERSION%%.*}" "8" && supported=true
            ;;
        darwin)
            # macOS 12+
            version_gte "${OS_VERSION%%.*}" "12" && supported=true
            ;;
        *)
            supported=false
            ;;
    esac

    if [[ "$supported" != "true" ]]; then
        echo ""
        error "Sistema não suportado: $OS $OS_VERSION"
        echo ""
        echo "  Sistemas suportados:"
        echo "    - Ubuntu 20.04+"
        echo "    - Debian 11+"
        echo "    - Linux Mint 20+"
        echo "    - Pop!_OS 20.04+"
        echo "    - Fedora 38+"
        echo "    - CentOS/RHEL/Rocky 8+"
        echo "    - macOS 12+"
        echo ""
        die "Instale em um sistema suportado ou contribua com suporte para $OS" 2
    fi

    success "Sistema suportado"
}

#' Get package manager update command
get_update_command() {
    case "$PACKAGE_MANAGER" in
        apt)
            echo "apt-get update -qq"
            ;;
        dnf)
            echo "dnf check-update -q || true"
            ;;
        yum)
            echo "yum check-update -q || true"
            ;;
        brew)
            echo "brew update"
            ;;
        zypper)
            echo "zypper refresh -q"
            ;;
        pacman)
            echo "pacman -Sy --noconfirm"
            ;;
        *)
            echo ""
            ;;
    esac
}

#' Get package install command
#' @param packages Space-separated package names
get_install_command() {
    local packages="$1"

    case "$PACKAGE_MANAGER" in
        apt)
            echo "apt-get install -y $packages"
            ;;
        dnf)
            echo "dnf install -y $packages"
            ;;
        yum)
            echo "yum install -y $packages"
            ;;
        brew)
            echo "brew install $packages"
            ;;
        zypper)
            echo "zypper install -y $packages"
            ;;
        pacman)
            echo "pacman -S --noconfirm $packages"
            ;;
        *)
            die "Gerenciador de pacotes não suportado: $PACKAGE_MANAGER"
            ;;
    esac
}
```

### Architecture Validation

```bash
#' Validate architecture is supported
validate_architecture() {
    case "$ARCH" in
        x86_64|amd64)
            debug "Arquitetura x86_64 suportada"
            ;;
        aarch64|arm64)
            debug "Arquitetura ARM64 suportada"
            ;;
        *)
            warn "Arquitetura $ARCH pode não ser totalmente suportada"
            ;;
    esac
}
```

---

## Test Cases

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| TC-001 | Detect Ubuntu 22.04 | OS=ubuntu, VERSION=22.04 |
| TC-002 | Detect Debian 12 | OS=debian, VERSION=12 |
| TC-003 | Detect Fedora 39 | OS=fedora, VERSION=39 |
| TC-004 | Detect CentOS 8 | OS=centos, PACKAGE_MANAGER=dnf |
| TC-005 | Detect CentOS 7 | OS=centos, PACKAGE_MANAGER=yum |
| TC-006 | Detect macOS Sonoma | OS=macos, PACKAGE_MANAGER=brew |
| TC-007 | Reject Windows/WSL | Error message shown |
| TC-008 | Detect x86_64 arch | ARCH=x86_64 |
| TC-009 | Detect arm64 arch | ARCH=aarch64 or arm64 |
| TC-010 | Old Ubuntu (18.04) | Rejected with message |

---

## Dependencies

- PC-086-11: Installer Script Structure

---

## Definition of Done

- [ ] All supported distros detected correctly
- [ ] Correct package manager identified
- [ ] Version comparison working
- [ ] Unsupported systems rejected gracefully
- [ ] Architecture detected
- [ ] Tested on Docker containers for each distro
- [ ] Code reviewed and approved
- [ ] Merged to develop branch

---

## Notes

- Use Docker for testing different distributions
- Consider adding WSL detection with warning
- os-release is the standard for modern Linux
- Keep fallbacks for older systems
