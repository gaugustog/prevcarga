# PC-105-12: GitHub Actions CI

**Epic:** [EPIC-12: Documentation & Deploy](../epics/EPIC-12-documentation-deploy.md)
**Task Reference:** T-12.8
**Priority:** High
**Estimated Effort:** 1 day

---

## Summary

Create GitHub Actions CI workflow for automated testing, linting, documentation build, and code coverage on pull requests and pushes to main/develop branches.

---

## Acceptance Criteria

- [ ] Trigger on push to main/develop and PRs
- [ ] R CMD check passes
- [ ] Unit tests run with testthat
- [ ] Code coverage reported to Codecov
- [ ] Linting with lintr
- [ ] Documentation builds with Sphinx
- [ ] Matrix testing (R versions, OS)
- [ ] Caching for dependencies

---

## Technical Specification

### Main CI Workflow

```yaml
# .github/workflows/ci.yml
# PrevCargaONS Continuous Integration

name: CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]
  workflow_dispatch:  # Manual trigger

concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

env:
  R_LIBS_USER: ${{ github.workspace }}/renv/library

jobs:
  # ────────────────────────────────────────────────────────────────────────────
  # R CMD Check
  # ────────────────────────────────────────────────────────────────────────────
  check:
    name: R CMD check (${{ matrix.os }}, R ${{ matrix.r }})
    runs-on: ${{ matrix.os }}

    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, macos-latest]
        r: ['4.3', '4.4']
        exclude:
          # Reduce matrix for faster CI
          - os: macos-latest
            r: '4.3'

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Setup R
        uses: r-lib/actions/setup-r@v2
        with:
          r-version: ${{ matrix.r }}
          use-public-rspm: true

      - name: Setup R dependencies
        uses: r-lib/actions/setup-r-dependencies@v2
        with:
          cache-version: 2
          extra-packages: |
            any::rcmdcheck
            any::covr
            any::lintr

      - name: Restore renv cache
        uses: actions/cache@v4
        with:
          path: |
            ~/.local/share/renv
            renv/library
          key: renv-${{ runner.os }}-${{ matrix.r }}-${{ hashFiles('renv.lock') }}
          restore-keys: |
            renv-${{ runner.os }}-${{ matrix.r }}-

      - name: Restore renv
        run: |
          renv::restore()
        shell: Rscript {0}

      - name: R CMD check
        uses: r-lib/actions/check-r-package@v2
        with:
          args: 'c("--no-manual", "--as-cran")'
          error-on: '"error"'
          check-dir: '"check"'

      - name: Upload check results
        if: failure()
        uses: actions/upload-artifact@v4
        with:
          name: check-results-${{ matrix.os }}-r${{ matrix.r }}
          path: check

  # ────────────────────────────────────────────────────────────────────────────
  # Unit Tests and Coverage
  # ────────────────────────────────────────────────────────────────────────────
  test:
    name: Tests and Coverage
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
          cache-version: 2
          extra-packages: |
            any::testthat
            any::covr

      - name: Restore renv
        run: |
          renv::restore()
        shell: Rscript {0}

      - name: Run tests
        run: |
          testthat::test_local(reporter = testthat::JunitReporter$new(file = "test-results.xml"))
        shell: Rscript {0}

      - name: Upload test results
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: test-results
          path: test-results.xml

      - name: Calculate coverage
        run: |
          cov <- covr::package_coverage(
            type = "tests",
            quiet = FALSE,
            clean = FALSE
          )
          covr::to_cobertura(cov, "coverage.xml")
          print(cov)
        shell: Rscript {0}

      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v4
        with:
          token: ${{ secrets.CODECOV_TOKEN }}
          files: coverage.xml
          flags: unittests
          name: codecov-umbrella
          fail_ci_if_error: false
          verbose: true

  # ────────────────────────────────────────────────────────────────────────────
  # Linting
  # ────────────────────────────────────────────────────────────────────────────
  lint:
    name: Lint
    runs-on: ubuntu-latest

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Setup R
        uses: r-lib/actions/setup-r@v2
        with:
          r-version: '4.4'
          use-public-rspm: true

      - name: Install lintr
        run: |
          install.packages("lintr")
        shell: Rscript {0}

      - name: Lint package
        run: |
          lints <- lintr::lint_package()
          if (length(lints) > 0) {
            print(lints)
            stop("Linting errors found")
          }
        shell: Rscript {0}

  # ────────────────────────────────────────────────────────────────────────────
  # Documentation Build
  # ────────────────────────────────────────────────────────────────────────────
  docs:
    name: Build Documentation
    runs-on: ubuntu-latest

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: 'pip'
          cache-dependency-path: docs/requirements.txt

      - name: Install Sphinx dependencies
        run: |
          pip install -r docs/requirements.txt

      - name: Build documentation
        run: |
          sphinx-build -b html docs/ docs/_build/html -W --keep-going

      - name: Check links
        run: |
          sphinx-build -b linkcheck docs/ docs/_build/linkcheck || true

      - name: Upload documentation
        uses: actions/upload-artifact@v4
        with:
          name: documentation
          path: docs/_build/html

  # ────────────────────────────────────────────────────────────────────────────
  # Docker Build Test
  # ────────────────────────────────────────────────────────────────────────────
  docker:
    name: Docker Build
    runs-on: ubuntu-latest

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Build Docker image
        uses: docker/build-push-action@v5
        with:
          context: .
          push: false
          tags: ons/prevcarga:test
          cache-from: type=gha
          cache-to: type=gha,mode=max

      - name: Test Docker image
        run: |
          docker run --rm ons/prevcarga:test --version
          docker run --rm ons/prevcarga:test --help

  # ────────────────────────────────────────────────────────────────────────────
  # Security Scan
  # ────────────────────────────────────────────────────────────────────────────
  security:
    name: Security Scan
    runs-on: ubuntu-latest

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Run Trivy vulnerability scanner
        uses: aquasecurity/trivy-action@master
        with:
          scan-type: 'fs'
          scan-ref: '.'
          format: 'sarif'
          output: 'trivy-results.sarif'
          severity: 'CRITICAL,HIGH'

      - name: Upload Trivy scan results
        uses: github/codeql-action/upload-sarif@v3
        if: always()
        with:
          sarif_file: 'trivy-results.sarif'

  # ────────────────────────────────────────────────────────────────────────────
  # CI Summary
  # ────────────────────────────────────────────────────────────────────────────
  ci-summary:
    name: CI Summary
    needs: [check, test, lint, docs, docker, security]
    runs-on: ubuntu-latest
    if: always()

    steps:
      - name: Check status
        run: |
          if [[ "${{ needs.check.result }}" == "failure" ]] || \
             [[ "${{ needs.test.result }}" == "failure" ]] || \
             [[ "${{ needs.lint.result }}" == "failure" ]] || \
             [[ "${{ needs.docs.result }}" == "failure" ]] || \
             [[ "${{ needs.docker.result }}" == "failure" ]]; then
            echo "CI failed"
            exit 1
          fi
          echo "CI passed!"
```

### Lintr Configuration

```yaml
# .lintr
# PrevCargaONS Lintr Configuration

linters: linters_with_defaults(
    line_length_linter(120),
    object_name_linter(styles = c("snake_case", "CamelCase")),
    cyclocomp_linter(complexity_limit = 25),
    commented_code_linter = NULL,  # Allow commented code examples
    object_usage_linter = NULL     # Disable for R6 classes
  )

exclusions: list(
    "renv" = list(
      linters = list()
    ),
    "inst" = list(
      linters = list()
    )
  )

encoding: "UTF-8"
```

### Codecov Configuration

```yaml
# codecov.yml
# PrevCargaONS Codecov Configuration

coverage:
  precision: 2
  round: down
  range: "60...100"

  status:
    project:
      default:
        target: 70%
        threshold: 5%

    patch:
      default:
        target: 80%
        threshold: 5%

parsers:
  cobertura:
    branch_detection:
      conditional: yes
      loop: yes
      method: no
      macro: no

comment:
  layout: "reach,diff,flags,files"
  behavior: default
  require_changes: false
  require_base: false
  require_head: true

flags:
  unittests:
    paths:
      - R/
    carryforward: true
```

### PR Template

```markdown
<!-- .github/PULL_REQUEST_TEMPLATE.md -->

## Description

<!-- Describe your changes -->

## Type of Change

- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Checklist

- [ ] My code follows the project's style guidelines
- [ ] I have performed a self-review of my code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes

## Related Issues

<!-- Link related issues: Fixes #123, Closes #456 -->
```

### Branch Protection Rules

```yaml
# Recommended GitHub settings (configure via UI or API)

# Branch: main
protection_rules:
  required_status_checks:
    strict: true
    contexts:
      - "R CMD check (ubuntu-latest, R 4.4)"
      - "Tests and Coverage"
      - "Lint"
      - "Build Documentation"
      - "Docker Build"

  required_pull_request_reviews:
    required_approving_review_count: 1
    dismiss_stale_reviews: true
    require_code_owner_reviews: false

  enforce_admins: false
  required_linear_history: false
  allow_force_pushes: false
  allow_deletions: false

# Branch: develop
protection_rules:
  required_status_checks:
    strict: false
    contexts:
      - "R CMD check (ubuntu-latest, R 4.4)"
      - "Tests and Coverage"
```

---

## Test Cases

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| TC-001 | Workflow triggers on PR | Jobs start |
| TC-002 | R CMD check passes | No errors |
| TC-003 | Tests execute | All pass |
| TC-004 | Coverage reported | Codecov updated |
| TC-005 | Lint runs | No errors |
| TC-006 | Docs build | HTML generated |
| TC-007 | Docker builds | Image created |
| TC-008 | Security scan | Results uploaded |
| TC-009 | Cache works | Faster builds |
| TC-010 | Matrix works | Multiple R versions |

---

## Dependencies

- PC-099-12: Sphinx Documentation Site (for docs build)
- PC-103-12: Dockerfile (for Docker build)

---

## Definition of Done

- [ ] CI workflow created
- [ ] All jobs pass on test PR
- [ ] Coverage integration working
- [ ] Lint configuration set
- [ ] Documentation builds
- [ ] Docker build succeeds
- [ ] Branch protection configured
- [ ] PR template created
- [ ] Code reviewed and approved
- [ ] Merged to develop branch

---

## Notes

- Use r-lib actions for R-specific setup
- Matrix testing ensures broad compatibility
- Caching speeds up subsequent runs
- Security scanning catches vulnerabilities
- Branch protection enforces quality gates
