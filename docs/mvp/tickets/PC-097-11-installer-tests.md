# PC-097-11: Installer Tests

**Epic:** [EPIC-11: One-Line Installer](../epics/EPIC-11-installer.md)
**Task Reference:** T-11.12
**Priority:** High
**Estimated Effort:** 1 day

---

## Summary

Create comprehensive tests for the installer using bash test frameworks (bats) and Docker containers to verify installation works on all supported platforms.

---

## Acceptance Criteria

- [ ] Test suite created at `tests/test-installer.sh`
- [ ] OS detection tests implemented
- [ ] R installation check tests implemented
- [ ] Directory creation tests implemented
- [ ] PATH configuration tests implemented
- [ ] Docker integration tests for all platforms
- [ ] CI/CD pipeline integration

---

## Technical Specification

### Test File Location
```
tests/installer/
├── test-installer.bats     # Main test file (bats format)
├── test-functions.bats     # Function unit tests
├── test-integration.sh     # Integration tests
├── helpers.bash            # Test helpers
├── docker/
│   ├── Dockerfile.ubuntu   # Ubuntu test image
│   ├── Dockerfile.debian   # Debian test image
│   ├── Dockerfile.fedora   # Fedora test image
│   ├── Dockerfile.centos   # CentOS test image
│   └── docker-compose.yml  # Multi-platform testing
└── fixtures/
    └── test-data/          # Test fixtures
```

### Bats Test Suite

```bash
#!/usr/bin/env bats
# tests/installer/test-installer.bats
#
# PrevCargaONS Installer Tests
#
# Run with: bats tests/installer/test-installer.bats

load 'helpers'

# ──────────────────────────────────────────────────────────────────────────────
# Setup and Teardown
# ──────────────────────────────────────────────────────────────────────────────

setup() {
    # Create temporary directory for tests
    TEST_DIR=$(mktemp -d)
    export HOME="$TEST_DIR"
    export INSTALL_DIR="$HOME/.prevcarga"

    # Source installer functions
    source scripts/install_template.sh --dry-run
}

teardown() {
    # Cleanup
    rm -rf "$TEST_DIR"
}

# ──────────────────────────────────────────────────────────────────────────────
# OS Detection Tests
# ──────────────────────────────────────────────────────────────────────────────

@test "detect_os identifies Ubuntu" {
    # Mock /etc/os-release
    mkdir -p "$TEST_DIR/etc"
    cat > "$TEST_DIR/etc/os-release" << 'EOF'
ID=ubuntu
VERSION_ID="22.04"
VERSION_CODENAME=jammy
EOF

    # Override to use test directory
    detect_linux

    [ "$OS" = "ubuntu" ]
    [ "$OS_VERSION" = "22.04" ]
    [ "$OS_FAMILY" = "debian" ]
    [ "$PACKAGE_MANAGER" = "apt" ]
}

@test "detect_os identifies Fedora" {
    mkdir -p "$TEST_DIR/etc"
    cat > "$TEST_DIR/etc/os-release" << 'EOF'
ID=fedora
VERSION_ID="39"
EOF

    detect_linux

    [ "$OS" = "fedora" ]
    [ "$OS_FAMILY" = "fedora" ]
    [ "$PACKAGE_MANAGER" = "dnf" ]
}

@test "detect_os identifies CentOS 8" {
    mkdir -p "$TEST_DIR/etc"
    cat > "$TEST_DIR/etc/os-release" << 'EOF'
ID=centos
VERSION_ID="8"
EOF

    detect_linux

    [ "$OS" = "centos" ]
    [ "$OS_FAMILY" = "rhel" ]
    [ "$PACKAGE_MANAGER" = "dnf" ]
}

@test "detect_os identifies CentOS 7" {
    mkdir -p "$TEST_DIR/etc"
    cat > "$TEST_DIR/etc/os-release" << 'EOF'
ID=centos
VERSION_ID="7"
EOF

    detect_linux

    [ "$OS" = "centos" ]
    [ "$PACKAGE_MANAGER" = "yum" ]
}

@test "detect_os rejects unsupported OS" {
    mkdir -p "$TEST_DIR/etc"
    cat > "$TEST_DIR/etc/os-release" << 'EOF'
ID=arch
VERSION_ID="rolling"
EOF

    detect_linux

    run validate_os_support
    [ "$status" -ne 0 ]
}

# ──────────────────────────────────────────────────────────────────────────────
# Version Comparison Tests
# ──────────────────────────────────────────────────────────────────────────────

@test "version_gte compares correctly" {
    run version_gte "4.3.3" "4.3.0"
    [ "$status" -eq 0 ]

    run version_gte "4.2.0" "4.3.0"
    [ "$status" -ne 0 ]

    run version_gte "4.3.0" "4.3.0"
    [ "$status" -eq 0 ]
}

# ──────────────────────────────────────────────────────────────────────────────
# Directory Structure Tests
# ──────────────────────────────────────────────────────────────────────────────

@test "create_directory_structure creates all directories" {
    create_directory_structure

    [ -d "$INSTALL_DIR/bin" ]
    [ -d "$INSTALL_DIR/src" ]
    [ -d "$INSTALL_DIR/renv" ]
    [ -d "$INSTALL_DIR/data" ]
    [ -d "$INSTALL_DIR/config" ]
    [ -d "$INSTALL_DIR/logs" ]
}

@test "directory permissions are correct" {
    create_directory_structure

    # Config should be 700
    local config_perms=$(stat -c %a "$INSTALL_DIR/config" 2>/dev/null || \
                         stat -f %Lp "$INSTALL_DIR/config")
    [ "$config_perms" = "700" ]

    # Bin should be 755
    local bin_perms=$(stat -c %a "$INSTALL_DIR/bin" 2>/dev/null || \
                      stat -f %Lp "$INSTALL_DIR/bin")
    [ "$bin_perms" = "755" ]
}

@test "backup created for existing installation" {
    # Create existing installation
    mkdir -p "$INSTALL_DIR/data"
    echo "test" > "$INSTALL_DIR/data/test.txt"
    echo "1.0.0" > "$INSTALL_DIR/.version"

    # Run installation
    create_directory_structure

    # Backup should exist
    [ -d "$INSTALL_DIR.backup."* ]
}

# ──────────────────────────────────────────────────────────────────────────────
# Shell Extraction Tests
# ──────────────────────────────────────────────────────────────────────────────

@test "extract_embedded_shell decodes base64" {
    # Create test shell
    local test_shell="#!/usr/bin/env Rscript\ncat('Hello')\n"
    EMBEDDED_SHELL=$(echo -e "$test_shell" | base64 -w0 2>/dev/null || \
                     echo -e "$test_shell" | base64 | tr -d '\n')

    local output=$(mktemp)
    run extract_embedded_shell "$output"
    [ "$status" -eq 0 ]

    # Verify content
    grep -q "Rscript" "$output"
    rm "$output"
}

@test "extract fails for placeholder" {
    EMBEDDED_SHELL="__EMBEDDED_SHELL__"

    local output=$(mktemp)
    run extract_embedded_shell "$output"
    [ "$status" -ne 0 ]
    rm "$output"
}

# ──────────────────────────────────────────────────────────────────────────────
# Launcher Tests
# ──────────────────────────────────────────────────────────────────────────────

@test "launcher script is created" {
    create_directory_structure
    create_launcher_script

    [ -f "$INSTALL_DIR/bin/prevcarga" ]
    [ -x "$INSTALL_DIR/bin/prevcarga" ]
}

@test "launcher contains correct paths" {
    create_directory_structure
    create_launcher_script

    grep -q "PREVCARGA_HOME" "$INSTALL_DIR/bin/prevcarga"
    grep -q "prevcarga_shell.R" "$INSTALL_DIR/bin/prevcarga"
}

# ──────────────────────────────────────────────────────────────────────────────
# PATH Configuration Tests
# ──────────────────────────────────────────────────────────────────────────────

@test "PATH added to .bashrc" {
    touch "$HOME/.bashrc"
    SHELL="/bin/bash"

    configure_path

    grep -q "PrevCargaONS" "$HOME/.bashrc"
    grep -q "$INSTALL_DIR/bin" "$HOME/.bashrc"
}

@test "PATH added to .zshrc for zsh" {
    touch "$HOME/.zshrc"
    SHELL="/bin/zsh"

    configure_path

    grep -q "PrevCargaONS" "$HOME/.zshrc"
}

@test "duplicate PATH entries avoided" {
    touch "$HOME/.bashrc"
    echo "# PrevCargaONS" >> "$HOME/.bashrc"
    echo "export PATH=\"$INSTALL_DIR/bin:\$PATH\"" >> "$HOME/.bashrc"
    SHELL="/bin/bash"

    configure_path

    # Should only have one entry
    local count=$(grep -c "PrevCargaONS" "$HOME/.bashrc")
    [ "$count" -eq 1 ]
}

# ──────────────────────────────────────────────────────────────────────────────
# Configuration Tests
# ──────────────────────────────────────────────────────────────────────────────

@test "default config is valid YAML" {
    create_directory_structure
    create_default_config

    [ -f "$INSTALL_DIR/config/config.yaml" ]

    # Verify YAML syntax (requires yq or python)
    if command -v yq &>/dev/null; then
        yq '.' "$INSTALL_DIR/config/config.yaml" > /dev/null
    elif command -v python3 &>/dev/null; then
        python3 -c "import yaml; yaml.safe_load(open('$INSTALL_DIR/config/config.yaml'))"
    fi
}

@test "example config created" {
    create_directory_structure
    create_default_config

    [ -f "$INSTALL_DIR/config/config.example.yaml" ]
}
```

### Docker Integration Tests

```yaml
# tests/installer/docker/docker-compose.yml
version: '3.8'

services:
  ubuntu-22.04:
    build:
      context: .
      dockerfile: Dockerfile.ubuntu
      args:
        VERSION: "22.04"
    volumes:
      - ../../../dist:/installer:ro

  ubuntu-20.04:
    build:
      context: .
      dockerfile: Dockerfile.ubuntu
      args:
        VERSION: "20.04"
    volumes:
      - ../../../dist:/installer:ro

  debian-12:
    build:
      context: .
      dockerfile: Dockerfile.debian
      args:
        VERSION: "12"
    volumes:
      - ../../../dist:/installer:ro

  fedora-39:
    build:
      context: .
      dockerfile: Dockerfile.fedora
      args:
        VERSION: "39"
    volumes:
      - ../../../dist:/installer:ro

  centos-8:
    build:
      context: .
      dockerfile: Dockerfile.centos
      args:
        VERSION: "8"
    volumes:
      - ../../../dist:/installer:ro
```

```dockerfile
# tests/installer/docker/Dockerfile.ubuntu
ARG VERSION=22.04
FROM ubuntu:${VERSION}

# Install minimal dependencies
RUN apt-get update && apt-get install -y curl ca-certificates

# Copy and run installer
COPY --from=installer /dist/install.sh /tmp/install.sh

# Test installation
RUN bash /tmp/install.sh

# Verify installation
RUN /root/.prevcarga/bin/prevcarga --version

CMD ["bash"]
```

### CI/CD Integration

```yaml
# .github/workflows/test-installer.yml
name: Test Installer

on:
  push:
    paths:
      - 'scripts/install*.sh'
      - 'tests/installer/**'
  pull_request:
    paths:
      - 'scripts/install*.sh'
      - 'tests/installer/**'

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install bats
        run: |
          sudo apt-get update
          sudo apt-get install -y bats

      - name: Run unit tests
        run: bats tests/installer/test-installer.bats

  integration-tests:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        platform:
          - ubuntu-22.04
          - ubuntu-20.04
          - debian-12
          - fedora-39

    steps:
      - uses: actions/checkout@v4

      - name: Build installer
        run: ./scripts/build_installer.sh

      - name: Test on ${{ matrix.platform }}
        run: |
          docker build \
            -f tests/installer/docker/Dockerfile.${{ matrix.platform }} \
            -t test-installer:${{ matrix.platform }} \
            .
          docker run --rm test-installer:${{ matrix.platform }}

  macos-test:
    runs-on: macos-latest
    steps:
      - uses: actions/checkout@v4

      - name: Build installer
        run: ./scripts/build_installer.sh

      - name: Test installation
        run: bash dist/install.sh --dry-run
```

---

## Test Cases

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| TC-001 | Detect Ubuntu 22.04 | OS=ubuntu, VERSION=22.04 |
| TC-002 | Detect Fedora 39 | OS=fedora |
| TC-003 | Detect CentOS 8 | PACKAGE_MANAGER=dnf |
| TC-004 | Reject unsupported OS | Exit with error |
| TC-005 | Directory structure | All dirs created |
| TC-006 | Directory permissions | Correct modes |
| TC-007 | Backup on upgrade | Backup created |
| TC-008 | Shell extraction | Base64 decoded |
| TC-009 | Launcher created | Executable file |
| TC-010 | PATH configured | Entry in rc file |
| TC-011 | Docker Ubuntu | Full install works |
| TC-012 | Docker Fedora | Full install works |
| TC-013 | Docker CentOS | Full install works |
| TC-014 | macOS local | Dry run works |
| TC-015 | CI pipeline | All tests pass |

---

## Dependencies

- All previous EPIC-11 tickets
- bats-core for testing
- Docker for integration tests

---

## Definition of Done

- [ ] Bats test suite created
- [ ] All function unit tests passing
- [ ] Docker integration tests passing
- [ ] CI/CD pipeline configured
- [ ] Tests run on all platforms
- [ ] 90%+ function coverage
- [ ] Documentation for running tests
- [ ] Code reviewed and approved
- [ ] Merged to develop branch

---

## Notes

- Use bats-core for bash testing
- Docker provides clean environments
- CI/CD ensures cross-platform compatibility
- Keep tests fast (<5 min total)
- Add new tests for each bug fix
