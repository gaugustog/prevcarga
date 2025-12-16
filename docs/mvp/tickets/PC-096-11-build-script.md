# PC-096-11: Build Script

**Epic:** [EPIC-11: One-Line Installer](../epics/EPIC-11-installer.md)
**Task Reference:** T-11.11
**Priority:** High
**Estimated Effort:** 0.5 days

---

## Summary

Create a build script that compiles the final `install.sh` by base64-encoding the R shell script, calculating checksums, and embedding them in the installer template.

---

## Acceptance Criteria

- [ ] Build script created at `scripts/build_installer.sh`
- [ ] R shell script base64-encoded
- [ ] SHA256 checksum calculated
- [ ] Placeholders replaced in template
- [ ] Output written to `dist/install.sh`
- [ ] Script is executable and self-contained

---

## Technical Specification

### Build Script

```bash
#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# PrevCargaONS Installer Build Script
# ══════════════════════════════════════════════════════════════════════════════
#
# This script builds the one-line installer by:
# 1. Base64-encoding the R shell script
# 2. Calculating the SHA256 checksum
# 3. Embedding both in the installer template
# 4. Generating the final install.sh
#
# Usage:
#   ./scripts/build_installer.sh
#   ./scripts/build_installer.sh --version 1.0.1
#
# ══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Input files
SHELL_SCRIPT="$PROJECT_ROOT/inst/shell/prevcarga_shell.R"
TEMPLATE="$PROJECT_ROOT/scripts/install_template.sh"
RENV_LOCK="$PROJECT_ROOT/renv.lock"

# Output
OUTPUT_DIR="$PROJECT_ROOT/dist"
OUTPUT_FILE="$OUTPUT_DIR/install.sh"

# Default version (can be overridden)
VERSION="${VERSION:-1.0.0}"

# ──────────────────────────────────────────────────────────────────────────────
# Functions
# ──────────────────────────────────────────────────────────────────────────────

print_header() {
    echo "══════════════════════════════════════════════════════════════════"
    echo "  PrevCargaONS Installer Builder"
    echo "══════════════════════════════════════════════════════════════════"
    echo ""
}

info() {
    echo "[INFO] $1"
}

error() {
    echo "[ERROR] $1" >&2
}

die() {
    error "$1"
    exit 1
}

#' Check required files exist
check_prerequisites() {
    info "Checking prerequisites..."

    if [[ ! -f "$SHELL_SCRIPT" ]]; then
        die "Shell script not found: $SHELL_SCRIPT"
    fi

    if [[ ! -f "$TEMPLATE" ]]; then
        die "Template not found: $TEMPLATE"
    fi

    info "  ✓ Shell script: $SHELL_SCRIPT"
    info "  ✓ Template: $TEMPLATE"
}

#' Base64 encode the shell script
encode_shell() {
    info "Encoding shell script..."

    local encoded
    if command -v base64 &>/dev/null; then
        # Linux base64 (with -w0 for no line wrapping)
        if base64 --help 2>&1 | grep -q "\-w"; then
            encoded=$(base64 -w0 "$SHELL_SCRIPT")
        else
            # macOS base64 (no -w option needed)
            encoded=$(base64 "$SHELL_SCRIPT" | tr -d '\n')
        fi
    else
        die "base64 command not found"
    fi

    echo "$encoded"
}

#' Calculate SHA256 checksum
calculate_checksum() {
    local file="$1"

    info "Calculating checksum..."

    if command -v sha256sum &>/dev/null; then
        sha256sum "$file" | cut -d' ' -f1
    elif command -v shasum &>/dev/null; then
        shasum -a 256 "$file" | cut -d' ' -f1
    else
        die "sha256sum or shasum not found"
    fi
}

#' Embed renv.lock in installer
encode_renv_lock() {
    if [[ -f "$RENV_LOCK" ]]; then
        info "Encoding renv.lock..."
        base64 -w0 "$RENV_LOCK" 2>/dev/null || base64 "$RENV_LOCK" | tr -d '\n'
    else
        echo ""
    fi
}

#' Build the installer
build_installer() {
    info "Building installer..."

    # Create output directory
    mkdir -p "$OUTPUT_DIR"

    # Encode shell script
    local encoded_shell
    encoded_shell=$(encode_shell)
    info "  Shell encoded: ${#encoded_shell} bytes"

    # Calculate checksum
    local checksum
    checksum=$(calculate_checksum "$SHELL_SCRIPT")
    info "  Checksum: $checksum"

    # Encode renv.lock (optional)
    local encoded_renv_lock
    encoded_renv_lock=$(encode_renv_lock)
    if [[ -n "$encoded_renv_lock" ]]; then
        info "  renv.lock encoded: ${#encoded_renv_lock} bytes"
    fi

    # Read template
    local template_content
    template_content=$(cat "$TEMPLATE")

    # Replace placeholders
    local output_content="$template_content"
    output_content="${output_content//__EMBEDDED_SHELL__/$encoded_shell}"
    output_content="${output_content//__EMBEDDED_SHELL_CHECKSUM__/$checksum}"
    output_content="${output_content//__EMBEDDED_RENV_LOCK__/$encoded_renv_lock}"
    output_content="${output_content//__VERSION__/$VERSION}"

    # Write output
    echo "$output_content" > "$OUTPUT_FILE"
    chmod +x "$OUTPUT_FILE"

    info "  Output: $OUTPUT_FILE"
}

#' Verify built installer
verify_installer() {
    info "Verifying installer..."

    # Check file exists
    if [[ ! -f "$OUTPUT_FILE" ]]; then
        die "Output file not created"
    fi

    # Check file size (should be > 10KB with embedded content)
    local size
    size=$(stat -f%z "$OUTPUT_FILE" 2>/dev/null || stat -c%s "$OUTPUT_FILE")
    info "  File size: $size bytes"

    if [[ $size -lt 10000 ]]; then
        die "Output file too small - embedding may have failed"
    fi

    # Check placeholders are replaced
    if grep -q "__EMBEDDED_SHELL__" "$OUTPUT_FILE"; then
        die "Placeholder __EMBEDDED_SHELL__ not replaced"
    fi

    # Run shellcheck if available
    if command -v shellcheck &>/dev/null; then
        info "  Running shellcheck..."
        if shellcheck "$OUTPUT_FILE"; then
            info "  ✓ Shellcheck passed"
        else
            error "  Shellcheck found issues (non-fatal)"
        fi
    fi

    info "  ✓ Installer verified"
}

#' Calculate installer hash for releases
generate_release_hash() {
    info "Generating release hash..."

    local hash
    hash=$(calculate_checksum "$OUTPUT_FILE")

    echo "$hash" > "$OUTPUT_DIR/install.sh.sha256"
    info "  Hash: $hash"
    info "  Saved: $OUTPUT_DIR/install.sh.sha256"
}

#' Print summary
print_summary() {
    echo ""
    echo "══════════════════════════════════════════════════════════════════"
    echo "  Build Complete!"
    echo "══════════════════════════════════════════════════════════════════"
    echo ""
    echo "  Version:  $VERSION"
    echo "  Output:   $OUTPUT_FILE"
    echo "  Size:     $(du -h "$OUTPUT_FILE" | cut -f1)"
    echo ""
    echo "  To test locally:"
    echo "    bash $OUTPUT_FILE"
    echo ""
    echo "  To publish:"
    echo "    Upload $OUTPUT_FILE to release assets"
    echo ""
}

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

main() {
    print_header

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --version)
                VERSION="$2"
                shift 2
                ;;
            --help)
                echo "Usage: $0 [--version X.Y.Z]"
                exit 0
                ;;
            *)
                die "Unknown argument: $1"
                ;;
        esac
    done

    info "Building version: $VERSION"
    echo ""

    check_prerequisites
    build_installer
    verify_installer
    generate_release_hash
    print_summary
}

main "$@"
```

### Installer Template Structure

```bash
# scripts/install_template.sh
#
# This is the template that gets filled in by build_installer.sh
# Placeholders:
#   __EMBEDDED_SHELL__          - Base64-encoded R shell
#   __EMBEDDED_SHELL_CHECKSUM__ - SHA256 of R shell
#   __EMBEDDED_RENV_LOCK__      - Base64-encoded renv.lock
#   __VERSION__                 - Version string

#!/usr/bin/env bash
# ... (all the installer functions from previous tickets)

VERSION="__VERSION__"

EMBEDDED_SHELL='__EMBEDDED_SHELL__'
EMBEDDED_SHELL_CHECKSUM='__EMBEDDED_SHELL_CHECKSUM__'
EMBEDDED_RENV_LOCK='__EMBEDDED_RENV_LOCK__'

# ... (rest of installer)
```

### Makefile Integration

```makefile
# Makefile

VERSION ?= 1.0.0

.PHONY: build-installer test-installer clean

build-installer:
	@echo "Building installer v$(VERSION)..."
	./scripts/build_installer.sh --version $(VERSION)

test-installer: build-installer
	@echo "Testing installer in Docker..."
	docker run --rm -v $(PWD)/dist:/dist ubuntu:22.04 \
		bash /dist/install.sh

clean:
	rm -rf dist/

release: build-installer
	@echo "Creating release..."
	gh release create v$(VERSION) dist/install.sh dist/install.sh.sha256
```

---

## Test Cases

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| TC-001 | Shell encoded | Base64 string generated |
| TC-002 | Checksum calculated | SHA256 hash |
| TC-003 | Placeholders replaced | No __PLACEHOLDER__ |
| TC-004 | Output created | dist/install.sh exists |
| TC-005 | Output executable | chmod +x applied |
| TC-006 | Size reasonable | > 10KB |
| TC-007 | Shellcheck passes | No errors |
| TC-008 | Hash file created | .sha256 exists |
| TC-009 | Version embedded | Correct version |
| TC-010 | Verify decode | Can extract shell |

---

## Dependencies

- PC-086-11: Installer Script Structure
- PC-090-11: Shell Extraction

---

## Definition of Done

- [ ] Build script created
- [ ] Base64 encoding working
- [ ] Checksum calculation working
- [ ] Placeholder replacement working
- [ ] Output verification working
- [ ] Shellcheck integration (optional)
- [ ] Release hash generation
- [ ] Makefile targets added
- [ ] Code reviewed and approved
- [ ] Merged to develop branch

---

## Notes

- Build before each release
- Verify on all platforms
- Keep installer < 1MB if possible
- Consider gzip compression for large shells
