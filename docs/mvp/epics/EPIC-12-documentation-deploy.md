# EPIC-12: Documentation & Deploy

**Duration:** 2 weeks
**Dependencies:** EPIC-11
**Reference:** [MVP Plan R - Phase 12](../mvp-plan-r.md#phase-12-documentation--deploy-2-weeks)

---

## Objective

Complete documentation, create production-ready Docker image, and configure CI/CD pipeline.

---

## Scope

This epic covers:
- Complete README and user documentation
- Sphinx documentation site with ReadTheDocs theme
- Documentation pages for each component
- Dockerfile and docker-compose
- CI/CD with GitHub Actions

**Out of Scope:**
- Plugin implementations
- Feature additions
- AWS/Cloud deployment (separate infrastructure task)
- Operational runbooks

---

## Tasks

### T-12.1: Complete README
- [ ] Create comprehensive README.md with:
  - Project overview
  - Installation instructions
  - Quick start guide
  - Configuration examples
  - CLI command reference
  - Plugin development overview
  - Contributing guidelines

### T-12.2: Create Sphinx Documentation Site
- [ ] Initialize Sphinx in `docs/`:
  ```bash
  pip install sphinx sphinx-rtd-theme myst-parser sphinx-copybutton
  sphinx-quickstart docs/
  ```
- [ ] Configure `docs/conf.py`:
  ```python
  project = 'PrevCargaONS'
  copyright = '2025, ONS'
  author = 'ONS'
  release = '1.0.0'

  extensions = [
      'myst_parser',
      'sphinx.ext.autodoc',
      'sphinx.ext.viewcode',
      'sphinx.ext.intersphinx',
      'sphinx_copybutton',
  ]

  # MyST-Parser settings (Markdown support)
  myst_enable_extensions = [
      'colon_fence',
      'deflist',
      'fieldlist',
      'tasklist',
  ]

  templates_path = ['_templates']
  exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

  html_theme = 'sphinx_rtd_theme'
  html_static_path = ['_static']
  html_logo = '_static/logo.png'

  # Intersphinx for linking to R documentation
  intersphinx_mapping = {
      'r': ('https://rdrr.io/r/', None),
  }
  ```
- [ ] Create `docs/index.md` (main landing page)
- [ ] Create `docs/_static/` for custom CSS and logo
- [ ] Configure ReadTheDocs in `.readthedocs.yaml`:
  ```yaml
  version: 2
  build:
    os: ubuntu-22.04
    tools:
      python: "3.11"
  sphinx:
    configuration: docs/conf.py
  python:
    install:
      - requirements: docs/requirements.txt
  ```
- [ ] Create `docs/requirements.txt`:
  ```
  sphinx>=7.0
  sphinx-rtd-theme>=2.0
  myst-parser>=2.0
  sphinx-copybutton>=0.5
  ```
- [ ] Build site: `sphinx-build -b html docs/ docs/_build/html`

### T-12.3: Create Documentation Pages
- [ ] `docs/getting-started/installation.md` - Installation overview
- [ ] `docs/getting-started/installation-docker.md` - Docker installation guide
- [ ] `docs/getting-started/installation-wsl.md` - WSL installation guide (Windows users)
- [ ] `docs/getting-started/installation-curl.md` - One-line installer guide (Linux/macOS)
- [ ] `docs/getting-started/quickstart.md` - Quick start tutorial
- [ ] `docs/user-guide/configuration.md` - Configuration reference
- [ ] `docs/user-guide/cli-reference.md` - CLI command reference
- [ ] `docs/user-guide/storage-backends.md` - S3 and local storage
- [ ] `docs/plugin-guide/overview.md` - Plugin architecture
- [ ] `docs/plugin-guide/feature-plugins.md` - Feature plugin development
- [ ] `docs/plugin-guide/model-plugins.md` - Model plugin development
- [ ] `docs/plugin-guide/combiner-plugins.md` - Combiner plugin development
- [ ] `docs/plugin-guide/reconciler-plugins.md` - Reconciler plugin development
- [ ] `docs/api/index.md` - R6 class API reference

### T-12.4: Docker Installation Documentation
- [ ] Document Docker Desktop installation (Windows/macOS)
- [ ] Document docker-compose usage:
  ```bash
  # Clone repository
  git clone https://github.com/ons/prevcargaons.git
  cd prevcargaons

  # Start with docker-compose
  docker-compose up -d

  # Run commands
  docker-compose exec prevcarga prevcarga --help
  docker-compose exec prevcarga prevcarga interactive
  ```
- [ ] Document volume mounts for data persistence
- [ ] Document environment variables configuration

### T-12.5: WSL Installation Documentation
- [ ] Document WSL2 installation on Windows:
  ```powershell
  # Enable WSL
  wsl --install

  # Install Ubuntu
  wsl --install -d Ubuntu-22.04
  ```
- [ ] Document R installation in WSL:
  ```bash
  # Inside WSL Ubuntu
  sudo apt-get update
  sudo apt-get install -y r-base r-base-dev

  # Install PrevCargaONS
  curl -fsSL https://ons.org.br/prevcarga/install.sh | bash
  ```
- [ ] Document Windows Terminal integration
- [ ] Document file system access between Windows and WSL
- [ ] Document VS Code + WSL development setup

### T-12.6: Create Dockerfile
- [ ] Create `docker/Dockerfile`:
  ```dockerfile
  FROM rocker/r-ver:4.3.0

  # System dependencies
  RUN apt-get update && apt-get install -y \
      libcurl4-openssl-dev \
      libssl-dev \
      libxml2-dev \
      libpq-dev \
      && rm -rf /var/lib/apt/lists/*

  # Install R packages
  RUN R -e "install.packages(c( \
      'R6', \
      'data.table', \
      'arrow', \
      'yaml', \
      'paws', \
      'future', \
      'future.apply', \
      'cli', \
      'optparse', \
      'checkmate', \
      'jsonlite', \
      'forecast' \
  ), repos='https://cloud.r-project.org/')"

  # Working directory
  WORKDIR /app

  # Copy package
  COPY . /app/

  # Install package
  RUN R CMD INSTALL .

  # Entry point
  ENTRYPOINT ["Rscript", "-e", "prevcargaons::main()"]
  ```

### T-12.7: Create docker-compose
- [ ] Create `docker/docker-compose.yml`:
  ```yaml
  version: '3.8'
  services:
    prevcarga:
      build:
        context: ..
        dockerfile: docker/Dockerfile
      volumes:
        - ./data:/app/data
        - ./config:/app/config
      command: ["interactive"]
  ```

### T-12.8: Configure GitHub Actions CI
- [ ] Create `.github/workflows/ci.yml`:
  ```yaml
  name: CI

  on:
    push:
      branches: [main, develop]
    pull_request:
      branches: [main]

  jobs:
    check:
      runs-on: ubuntu-latest
      steps:
        - uses: actions/checkout@v3

        - uses: r-lib/actions/setup-r@v2
          with:
            r-version: '4.3.0'

        - uses: r-lib/actions/setup-r-dependencies@v2
          with:
            extra-packages: any::rcmdcheck

        - name: Check
          run: rcmdcheck::rcmdcheck(error_on = "error")
          shell: Rscript {0}

    test:
      runs-on: ubuntu-latest
      needs: check
      steps:
        - uses: actions/checkout@v3

        - uses: r-lib/actions/setup-r@v2

        - uses: r-lib/actions/setup-r-dependencies@v2

        - name: Test
          run: testthat::test_local()
          shell: Rscript {0}

        - name: Coverage
          run: |
            covr::codecov()
          shell: Rscript {0}

    lint:
      runs-on: ubuntu-latest
      steps:
        - uses: actions/checkout@v3

        - uses: r-lib/actions/setup-r@v2

        - name: Lint
          run: lintr::lint_package()
          shell: Rscript {0}

    docs:
      runs-on: ubuntu-latest
      steps:
        - uses: actions/checkout@v3

        - uses: actions/setup-python@v4
          with:
            python-version: '3.11'

        - name: Install Sphinx
          run: pip install -r docs/requirements.txt

        - name: Build docs
          run: sphinx-build -b html docs/ docs/_build/html
  ```

### T-12.9: Configure GitHub Actions Release
- [ ] Create `.github/workflows/release.yml`:
  ```yaml
  name: Release

  on:
    push:
      tags:
        - 'v*'

  jobs:
    release:
      runs-on: ubuntu-latest
      steps:
        - uses: actions/checkout@v3

        - name: Build Docker image
          run: |
            docker build -t prevcargaons:${{ github.ref_name }} \
              -f docker/Dockerfile .

        - name: Create GitHub Release
          uses: softprops/action-gh-release@v1
          with:
            generate_release_notes: true
  ```

---

## Acceptance Criteria

- [ ] README is comprehensive and accurate
- [ ] Sphinx documentation builds without errors (`sphinx-build`)
- [ ] Documentation renders correctly with ReadTheDocs theme
- [ ] All documentation pages are complete and linked
- [ ] Docker installation guide is complete and tested
- [ ] WSL installation guide is complete and tested
- [ ] Docker image builds and runs locally
- [ ] CI pipeline passes all checks
- [ ] Release workflow creates tagged releases

---

## Definition of Done

- [ ] All tasks completed
- [ ] Documentation reviewed and approved
- [ ] Docker image tested locally
- [ ] CI/CD pipelines verified
- [ ] Code merged to main branch

---

## Files Created/Modified

| File | Action | Description |
|------|--------|-------------|
| `README.md` | Modify | Comprehensive readme |
| `docs/conf.py` | Create | Sphinx configuration |
| `docs/index.md` | Create | Documentation landing page |
| `docs/requirements.txt` | Create | Sphinx dependencies |
| `docs/getting-started/installation.md` | Create | Installation overview |
| `docs/getting-started/installation-docker.md` | Create | Docker installation guide |
| `docs/getting-started/installation-wsl.md` | Create | WSL installation guide |
| `docs/getting-started/installation-curl.md` | Create | One-line installer guide |
| `docs/getting-started/quickstart.md` | Create | Quick start tutorial |
| `docs/user-guide/*.md` | Create | User guide pages |
| `docs/plugin-guide/*.md` | Create | Plugin development guides |
| `docs/api/*.md` | Create | API reference |
| `.readthedocs.yaml` | Create | ReadTheDocs configuration |
| `docker/Dockerfile` | Create | Production Dockerfile |
| `docker/docker-compose.yml` | Create | Local development compose |
| `.github/workflows/ci.yml` | Create | CI pipeline |
| `.github/workflows/release.yml` | Create | Release pipeline |
