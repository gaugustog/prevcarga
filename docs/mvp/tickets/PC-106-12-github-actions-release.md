# PC-106-12: GitHub Actions Release

**Epic:** [EPIC-12: Documentation & Deploy](../epics/EPIC-12-documentation-deploy.md)
**Task Reference:** T-12.9
**Priority:** High
**Estimated Effort:** 1 day

---

## Summary

Create GitHub Actions release workflow for automated releases including semantic versioning, changelog generation, Docker image publishing, GitHub releases, and installer script deployment.

---

## Acceptance Criteria

- [ ] Trigger on version tags (v*)
- [ ] Build and publish Docker image to registry
- [ ] Create GitHub release with changelog
- [ ] Deploy installer script to hosting
- [ ] Update documentation version
- [ ] Notify on release completion
- [ ] Support pre-release versions

---

## Technical Specification

### Release Workflow

```yaml
# .github/workflows/release.yml
# PrevCargaONS Release Workflow

name: Release

on:
  push:
    tags:
      - 'v*.*.*'  # Matches v1.0.0, v1.2.3, etc.
  workflow_dispatch:
    inputs:
      version:
        description: 'Version to release (without v prefix)'
        required: true
        type: string
      prerelease:
        description: 'Mark as pre-release'
        required: false
        type: boolean
        default: false

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  # ────────────────────────────────────────────────────────────────────────────
  # Validate Release
  # ────────────────────────────────────────────────────────────────────────────
  validate:
    name: Validate Release
    runs-on: ubuntu-latest
    outputs:
      version: ${{ steps.version.outputs.version }}
      prerelease: ${{ steps.version.outputs.prerelease }}

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Extract version
        id: version
        run: |
          if [[ "${{ github.event_name }}" == "workflow_dispatch" ]]; then
            VERSION="${{ github.event.inputs.version }}"
            PRERELEASE="${{ github.event.inputs.prerelease }}"
          else
            VERSION="${GITHUB_REF#refs/tags/v}"
            # Check if pre-release (contains alpha, beta, rc)
            if [[ "$VERSION" =~ (alpha|beta|rc) ]]; then
              PRERELEASE="true"
            else
              PRERELEASE="false"
            fi
          fi
          echo "version=${VERSION}" >> $GITHUB_OUTPUT
          echo "prerelease=${PRERELEASE}" >> $GITHUB_OUTPUT
          echo "Releasing version: ${VERSION} (prerelease: ${PRERELEASE})"

      - name: Validate version format
        run: |
          VERSION="${{ steps.version.outputs.version }}"
          if [[ ! "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+(-[a-zA-Z0-9.]+)?$ ]]; then
            echo "Invalid version format: ${VERSION}"
            echo "Expected format: X.Y.Z or X.Y.Z-suffix"
            exit 1
          fi

      - name: Check DESCRIPTION version
        run: |
          VERSION="${{ steps.version.outputs.version }}"
          DESC_VERSION=$(grep "^Version:" DESCRIPTION | sed 's/Version: //')
          if [[ "$DESC_VERSION" != "$VERSION" ]]; then
            echo "Version mismatch!"
            echo "Tag version: ${VERSION}"
            echo "DESCRIPTION version: ${DESC_VERSION}"
            exit 1
          fi

  # ────────────────────────────────────────────────────────────────────────────
  # Run Tests
  # ────────────────────────────────────────────────────────────────────────────
  test:
    name: Run Tests
    needs: validate
    runs-on: ubuntu-latest

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Setup R
        uses: r-lib/actions/setup-r@v2
        with:
          r-version: '4.4'
          use-public-rspm: true

      - name: Setup R dependencies
        uses: r-lib/actions/setup-r-dependencies@v2
        with:
          extra-packages: |
            any::rcmdcheck
            any::testthat

      - name: R CMD check
        uses: r-lib/actions/check-r-package@v2
        with:
          args: 'c("--no-manual", "--as-cran")'
          error-on: '"error"'

  # ────────────────────────────────────────────────────────────────────────────
  # Build and Push Docker Image
  # ────────────────────────────────────────────────────────────────────────────
  docker:
    name: Build and Push Docker Image
    needs: [validate, test]
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Set up QEMU
        uses: docker/setup-qemu-action@v3

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Login to GitHub Container Registry
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Login to Docker Hub
        uses: docker/login-action@v3
        with:
          username: ${{ secrets.DOCKERHUB_USERNAME }}
          password: ${{ secrets.DOCKERHUB_TOKEN }}
        if: ${{ secrets.DOCKERHUB_USERNAME != '' }}

      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: |
            ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
            ons/prevcarga
          tags: |
            type=semver,pattern={{version}},value=${{ needs.validate.outputs.version }}
            type=semver,pattern={{major}}.{{minor}},value=${{ needs.validate.outputs.version }}
            type=semver,pattern={{major}},value=${{ needs.validate.outputs.version }}
            type=raw,value=latest,enable=${{ needs.validate.outputs.prerelease == 'false' }}

      - name: Build and push
        uses: docker/build-push-action@v5
        with:
          context: .
          platforms: linux/amd64,linux/arm64
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
          build-args: |
            VERSION=${{ needs.validate.outputs.version }}
            BUILD_DATE=${{ github.event.head_commit.timestamp }}
            VCS_REF=${{ github.sha }}

  # ────────────────────────────────────────────────────────────────────────────
  # Build Installer
  # ────────────────────────────────────────────────────────────────────────────
  installer:
    name: Build Installer
    needs: [validate, test]
    runs-on: ubuntu-latest

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Build installer script
        run: |
          VERSION="${{ needs.validate.outputs.version }}"
          ./scripts/build_installer.sh "${VERSION}"

      - name: Upload installer artifact
        uses: actions/upload-artifact@v4
        with:
          name: installer
          path: dist/install.sh

      - name: Upload to release hosting
        run: |
          # Upload to S3 or other hosting
          # aws s3 cp dist/install.sh s3://ons-releases/prevcarga/install.sh
          echo "Installer built: dist/install.sh"
        # env:
        #   AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
        #   AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}

  # ────────────────────────────────────────────────────────────────────────────
  # Generate Changelog
  # ────────────────────────────────────────────────────────────────────────────
  changelog:
    name: Generate Changelog
    needs: validate
    runs-on: ubuntu-latest
    outputs:
      changelog: ${{ steps.changelog.outputs.changelog }}

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Generate changelog
        id: changelog
        uses: orhun/git-cliff-action@v3
        with:
          config: cliff.toml
          args: --latest --strip header
        env:
          OUTPUT: CHANGELOG.md

      - name: Read changelog
        id: read_changelog
        run: |
          CHANGELOG=$(cat CHANGELOG.md)
          echo "changelog<<EOF" >> $GITHUB_OUTPUT
          echo "$CHANGELOG" >> $GITHUB_OUTPUT
          echo "EOF" >> $GITHUB_OUTPUT

  # ────────────────────────────────────────────────────────────────────────────
  # Create GitHub Release
  # ────────────────────────────────────────────────────────────────────────────
  release:
    name: Create GitHub Release
    needs: [validate, test, docker, installer, changelog]
    runs-on: ubuntu-latest
    permissions:
      contents: write

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Download installer
        uses: actions/download-artifact@v4
        with:
          name: installer
          path: dist/

      - name: Create Release
        uses: softprops/action-gh-release@v1
        with:
          tag_name: v${{ needs.validate.outputs.version }}
          name: PrevCargaONS v${{ needs.validate.outputs.version }}
          body: |
            ## What's Changed

            ${{ needs.changelog.outputs.changelog }}

            ## Installation

            ### One-Line Installer (Linux/macOS)
            ```bash
            curl -fsSL https://ons.org.br/prevcarga/install.sh | bash
            ```

            ### Docker
            ```bash
            docker pull ons/prevcarga:${{ needs.validate.outputs.version }}
            docker run -it --rm ons/prevcarga:${{ needs.validate.outputs.version }}
            ```

            ### WSL (Windows)
            ```powershell
            wsl --install -d Ubuntu-22.04
            # Then run the one-line installer inside WSL
            ```

            ## Docker Images

            - `ons/prevcarga:${{ needs.validate.outputs.version }}`
            - `ons/prevcarga:latest`
            - `ghcr.io/ons-br/prevcarga-r:${{ needs.validate.outputs.version }}`

            ## Full Changelog

            See [CHANGELOG.md](https://github.com/ons-br/prevcarga-R/blob/main/CHANGELOG.md)
          draft: false
          prerelease: ${{ needs.validate.outputs.prerelease == 'true' }}
          files: |
            dist/install.sh

  # ────────────────────────────────────────────────────────────────────────────
  # Deploy Documentation
  # ────────────────────────────────────────────────────────────────────────────
  docs:
    name: Deploy Documentation
    needs: [validate, release]
    runs-on: ubuntu-latest
    if: needs.validate.outputs.prerelease == 'false'

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install Sphinx
        run: pip install -r docs/requirements.txt

      - name: Build documentation
        run: sphinx-build -b html docs/ docs/_build/html

      - name: Deploy to GitHub Pages
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: docs/_build/html

  # ────────────────────────────────────────────────────────────────────────────
  # Notify
  # ────────────────────────────────────────────────────────────────────────────
  notify:
    name: Notify Release
    needs: [validate, release, docker, docs]
    runs-on: ubuntu-latest
    if: always()

    steps:
      - name: Send notification
        run: |
          VERSION="${{ needs.validate.outputs.version }}"
          if [[ "${{ needs.release.result }}" == "success" ]]; then
            echo "Release v${VERSION} completed successfully!"
            # Notify via Slack, email, etc.
            # curl -X POST -H 'Content-type: application/json' \
            #   --data '{"text":"PrevCargaONS v${{ needs.validate.outputs.version }} released!"}' \
            #   ${{ secrets.SLACK_WEBHOOK_URL }}
          else
            echo "Release v${VERSION} failed!"
          fi
```

### Git-Cliff Configuration

```toml
# cliff.toml
# Changelog generator configuration

[changelog]
header = """
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

"""
body = """
{% if version %}\
    ## [{{ version | trim_start_matches(pat="v") }}] - {{ timestamp | date(format="%Y-%m-%d") }}
{% else %}\
    ## [Unreleased]
{% endif %}\
{% for group, commits in commits | group_by(attribute="group") %}
    ### {{ group | upper_first }}
    {% for commit in commits %}
        - {% if commit.scope %}**{{ commit.scope }}:** {% endif %}{{ commit.message | upper_first }}\
    {% endfor %}
{% endfor %}\n
"""
footer = ""
trim = true

[git]
conventional_commits = true
filter_unconventional = true
split_commits = false

commit_preprocessors = [
    { pattern = '\((\w+\s)?#([0-9]+)\)', replace = "([#${2}](https://github.com/ons-br/prevcarga-R/issues/${2}))" },
]

commit_parsers = [
    { message = "^feat", group = "Features" },
    { message = "^fix", group = "Bug Fixes" },
    { message = "^doc", group = "Documentation" },
    { message = "^perf", group = "Performance" },
    { message = "^refactor", group = "Refactoring" },
    { message = "^style", group = "Styling" },
    { message = "^test", group = "Testing" },
    { message = "^chore\\(release\\)", skip = true },
    { message = "^chore", group = "Miscellaneous" },
]

filter_commits = false
tag_pattern = "v[0-9]*"
skip_tags = ""
ignore_tags = ""
date_order = false
sort_commits = "oldest"
```

### Version Bump Script

```bash
#!/bin/bash
# scripts/bump-version.sh
# Bump version in all files

set -euo pipefail

NEW_VERSION="${1:-}"

if [[ -z "$NEW_VERSION" ]]; then
    echo "Usage: $0 <new-version>"
    echo "Example: $0 1.2.0"
    exit 1
fi

# Validate version format
if [[ ! "$NEW_VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+(-[a-zA-Z0-9.]+)?$ ]]; then
    echo "Invalid version format: $NEW_VERSION"
    exit 1
fi

echo "Bumping version to ${NEW_VERSION}..."

# Update DESCRIPTION
sed -i "s/^Version: .*/Version: ${NEW_VERSION}/" DESCRIPTION

# Update VERSION file
echo "${NEW_VERSION}" > VERSION

# Update inst/config/config.default.yaml
sed -i "s/version: \".*\"/version: \"${NEW_VERSION}\"/" inst/config/config.default.yaml

# Update Dockerfile labels
sed -i "s/LABEL version=\".*\"/LABEL version=\"${NEW_VERSION}\"/" Dockerfile

# Update documentation conf.py
sed -i "s/^release = .*/release = '${NEW_VERSION}'/" docs/conf.py
sed -i "s/^version = .*/version = '${NEW_VERSION%.*}'/" docs/conf.py

echo "Version bumped to ${NEW_VERSION}"
echo ""
echo "Next steps:"
echo "  1. Review changes: git diff"
echo "  2. Commit: git commit -am 'chore(release): bump version to ${NEW_VERSION}'"
echo "  3. Tag: git tag -a v${NEW_VERSION} -m 'Release v${NEW_VERSION}'"
echo "  4. Push: git push origin main --tags"
```

### Release Checklist

```markdown
# Release Checklist

## Pre-Release

- [ ] All tests passing on main branch
- [ ] Code coverage meets threshold (≥70%)
- [ ] No critical security vulnerabilities
- [ ] Documentation up to date
- [ ] CHANGELOG.md updated (or will be auto-generated)
- [ ] Version bumped in all files

## Release Process

1. **Prepare release**
   ```bash
   # Ensure main is up to date
   git checkout main
   git pull origin main

   # Bump version
   ./scripts/bump-version.sh 1.2.0

   # Commit
   git commit -am "chore(release): bump version to 1.2.0"
   ```

2. **Create and push tag**
   ```bash
   git tag -a v1.2.0 -m "Release v1.2.0"
   git push origin main --tags
   ```

3. **Monitor release workflow**
   - Check GitHub Actions for release progress
   - Verify Docker images published
   - Verify GitHub release created

## Post-Release

- [ ] Docker images available on registries
- [ ] GitHub release published
- [ ] Documentation deployed
- [ ] Installer script updated
- [ ] Announce release (if major)
```

---

## Test Cases

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| TC-001 | Tag triggers workflow | Release starts |
| TC-002 | Version validation | Invalid rejected |
| TC-003 | Tests run | Must pass |
| TC-004 | Docker builds | Multi-arch images |
| TC-005 | Docker pushes | Available on registry |
| TC-006 | Changelog generates | Markdown output |
| TC-007 | GitHub release created | With assets |
| TC-008 | Installer uploaded | Accessible |
| TC-009 | Docs deployed | Pages updated |
| TC-010 | Pre-release handling | Marked correctly |

---

## Dependencies

- PC-103-12: Dockerfile
- PC-105-12: GitHub Actions CI
- PC-096-11: Build Script (for installer)

---

## Definition of Done

- [ ] Release workflow created
- [ ] Docker publishing working
- [ ] GitHub releases created
- [ ] Changelog generation working
- [ ] Version bump script created
- [ ] Documentation deployment working
- [ ] Release checklist documented
- [ ] Code reviewed and approved
- [ ] Merged to develop branch

---

## Notes

- Use semantic versioning (MAJOR.MINOR.PATCH)
- Pre-release versions use suffixes (1.0.0-alpha.1)
- Multi-architecture Docker builds (amd64, arm64)
- Git-cliff generates changelog from conventional commits
- GitHub Pages hosts documentation
