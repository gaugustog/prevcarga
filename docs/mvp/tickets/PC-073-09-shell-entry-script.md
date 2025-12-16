# PC-073-09: Shell Entry Script

**Epic:** [EPIC-09: CLI](../epics/EPIC-09-cli.md)
**Task Reference:** T-09.10
**Priority:** High
**Estimated Effort:** 0.5 days

---

## Summary

Create the shell entry script that allows users to run PrevCarga from the command line. This includes the main executable script, shell completion files, and environment setup.

---

## Acceptance Criteria

- [ ] Shell script created in `inst/shell/prevcarga`
- [ ] Executable on Linux/macOS
- [ ] Windows batch file alternative
- [ ] Shell completion for bash/zsh
- [ ] Environment variable support
- [ ] Package installation validation

---

## Technical Specification

### File Location
```
inst/shell/prevcarga       # Unix shell script
inst/shell/prevcarga.bat   # Windows batch file
inst/shell/prevcarga.ps1   # PowerShell script
inst/shell/_prevcarga      # Zsh completion
inst/shell/prevcarga.bash  # Bash completion
```

### Unix Shell Script

```bash
#!/usr/bin/env Rscript
# PrevCarga CLI Entry Point
# =====================================================
# Electric Load Forecasting System for Brazilian Power Grid
#
# Usage: prevcarga <command> [options]
#
# Run 'prevcarga help' for available commands.
# =====================================================

# Set default library path if not set
if (Sys.getenv("R_LIBS_USER") == "") {
  default_lib <- file.path(Sys.getenv("HOME"), "R", "library")
  .libPaths(c(default_lib, .libPaths()))
}

# Check package is installed
if (!requireNamespace("prevcargaons", quietly = TRUE)) {
  cat("Error: prevcargaons package not installed.\n", file = stderr())
  cat("Install with: install.packages('prevcargaons')\n", file = stderr())
  quit(status = 1)
}

# Run main function
prevcargaons::main()
```

### Alternative Rscript Wrapper (for PATH installation)

```bash
#!/bin/bash
# prevcarga - PrevCarga CLI
# =====================================================
# Wrapper script for calling the R CLI
#
# Environment Variables:
#   PREVCARGA_CONFIG  - Default config file path
#   PREVCARGA_ENV     - Environment (development/production)
#   PREVCARGA_CACHE   - Cache directory
#   R_LIBS_USER       - R library path
# =====================================================

set -e

# Find Rscript
RSCRIPT=$(which Rscript 2>/dev/null)

if [ -z "$RSCRIPT" ]; then
    echo "Error: Rscript not found in PATH" >&2
    echo "Please install R: https://cran.r-project.org/" >&2
    exit 1
fi

# Check R version
R_VERSION=$($RSCRIPT --version 2>&1 | grep -oP '\d+\.\d+' | head -1)
R_MAJOR=$(echo $R_VERSION | cut -d. -f1)

if [ "$R_MAJOR" -lt 4 ]; then
    echo "Warning: R version $R_VERSION detected. PrevCarga requires R >= 4.0" >&2
fi

# Build R command
R_CMD="library(prevcargaons); prevcargaons::main()"

# Pass all arguments
exec $RSCRIPT -e "$R_CMD" "$@"
```

### Windows Batch File

```batch
@echo off
REM PrevCarga CLI for Windows
REM =====================================================

setlocal EnableDelayedExpansion

REM Find Rscript
where Rscript >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Error: Rscript not found in PATH
    echo Please install R from https://cran.r-project.org/
    exit /b 1
)

REM Run PrevCarga
Rscript -e "library(prevcargaons); prevcargaons::main()" %*
```

### PowerShell Script

```powershell
#!/usr/bin/env pwsh
# PrevCarga CLI for PowerShell
# =====================================================

param(
    [Parameter(Position=0, ValueFromRemainingArguments=$true)]
    [string[]]$Arguments
)

# Check Rscript
$rscript = Get-Command Rscript -ErrorAction SilentlyContinue

if (-not $rscript) {
    Write-Error "Rscript not found. Please install R from https://cran.r-project.org/"
    exit 1
}

# Build command
$cmd = "library(prevcargaons); prevcargaons::main()"

# Run with arguments
& Rscript -e $cmd $Arguments
```

### Bash Completion

```bash
# prevcarga bash completion
# Install: cp prevcarga.bash /etc/bash_completion.d/prevcarga
#      or: source prevcarga.bash

_prevcarga_completions() {
    local cur prev commands options

    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"

    # Main commands
    commands="train predict backtest gen-features eval-features eval-model eval-combination combine report interactive help version"

    # Common options
    common_options="--config --verbose --quiet --help"

    # Command-specific options
    case "${COMP_WORDS[1]}" in
        train)
            options="--areas --models --start-date --end-date --version --parallel --storage-backend --output-dir --dry-run $common_options"
            ;;
        predict)
            options="--date --mode --areas --models --model-version --horizons --reconcile --no-reconcile --combine --no-combine --output --format $common_options"
            ;;
        backtest)
            options="--start-date --end-date --retrain-intervals --walk-forward --no-walk-forward --areas --models --report --output-dir --parallel $common_options"
            ;;
        gen-features)
            options="--model --areas --start-date --end-date --plugins --output --format --include-target --drop-na $common_options"
            ;;
        eval-features)
            options="--input --model --area --method --shap-samples --correlation-method --output --format --top $common_options"
            ;;
        eval-model)
            options="--model --model-version --model-path --test-data --start-date --end-date --areas --metrics --by-horizon --by-period --by-area --output --format --report $common_options"
            ;;
        eval-combination)
            options="--models --strategies --start-date --end-date --areas --metrics --output --report $common_options"
            ;;
        combine)
            options="--models --strategy --date --areas --weights --output --format $common_options"
            ;;
        report)
            options="--output --areas --date --period --models --backtest-id --area --model --baseline-date --open $common_options"
            ;;
        *)
            if [[ ${cur} == -* ]]; then
                options="$common_options"
            else
                options="$commands"
            fi
            ;;
    esac

    # Generate completions
    if [[ ${cur} == -* ]]; then
        COMPREPLY=( $(compgen -W "${options}" -- ${cur}) )
    elif [[ ${COMP_CWORD} -eq 1 ]]; then
        COMPREPLY=( $(compgen -W "${commands}" -- ${cur}) )
    else
        # Context-aware completions
        case "${prev}" in
            --config)
                COMPREPLY=( $(compgen -f -X '!*.yaml' -- ${cur}) )
                ;;
            --output|--output-dir)
                COMPREPLY=( $(compgen -d -- ${cur}) )
                ;;
            --format)
                COMPREPLY=( $(compgen -W "parquet csv rds json" -- ${cur}) )
                ;;
            --mode)
                COMPREPLY=( $(compgen -W "batch intraday" -- ${cur}) )
                ;;
            --method)
                COMPREPLY=( $(compgen -W "all shap correlation stability permutation" -- ${cur}) )
                ;;
            --strategy|--strategies)
                COMPREPLY=( $(compgen -W "simple_average inverse_mape optimal" -- ${cur}) )
                ;;
            *)
                COMPREPLY=()
                ;;
        esac
    fi

    return 0
}

complete -F _prevcarga_completions prevcarga
```

### Zsh Completion

```zsh
#compdef prevcarga
# Zsh completion for prevcarga
# Install: cp _prevcarga ~/.zsh/completions/

local -a commands
commands=(
    'train:Train forecasting models'
    'predict:Generate predictions'
    'backtest:Run backtesting simulation'
    'gen-features:Generate features'
    'eval-features:Evaluate feature importance'
    'eval-model:Evaluate model performance'
    'eval-combination:Evaluate combination strategies'
    'combine:Combine forecasts'
    'report:Generate HTML reports'
    'interactive:Launch interactive shell'
    'help:Show help'
    'version:Show version'
)

local -a common_options
common_options=(
    '--config[Path to config file]:config:_files -g "*.yaml"'
    '--verbose[Enable verbose output]'
    '--quiet[Suppress output]'
    '--help[Show help]'
)

_prevcarga() {
    local curcontext="$curcontext" state line
    typeset -A opt_args

    _arguments -C \
        '1: :->command' \
        '*: :->args'

    case $state in
        command)
            _describe -t commands 'prevcarga command' commands
            ;;
        args)
            case $line[1] in
                train)
                    _arguments \
                        $common_options \
                        '--areas[Area codes]:areas:' \
                        '--models[Model names]:models:' \
                        '--start-date[Start date]:date:' \
                        '--end-date[End date]:date:' \
                        '--version[Version string]:version:' \
                        '--parallel[Workers]:workers:' \
                        '--dry-run[Validate without running]'
                    ;;
                predict)
                    _arguments \
                        $common_options \
                        '--date[Target date]:date:' \
                        '--mode[Mode]:mode:(batch intraday)' \
                        '--areas[Areas]:areas:' \
                        '--models[Models]:models:' \
                        '--reconcile[Enable reconciliation]' \
                        '--combine[Enable combination]' \
                        '--output[Output path]:output:_files' \
                        '--format[Output format]:format:(parquet csv rds)'
                    ;;
                report)
                    _arguments \
                        $common_options \
                        '1:report type:(forecast backtest compare dashboard drift)' \
                        '--output[Output file]:output:_files -g "*.html"' \
                        '--open[Open in browser]'
                    ;;
                *)
                    _arguments $common_options
                    ;;
            esac
            ;;
    esac
}

_prevcarga "$@"
```

### Installation Script

```r
#' Install CLI scripts
#'
#' Installs prevcarga command-line scripts to system path.
#'
#' @param path Installation directory (default: ~/bin)
#' @param completion Install shell completions
#' @export
install_cli <- function(path = "~/bin", completion = TRUE) {
  path <- path.expand(path)

  # Create directory
  if (!dir.exists(path)) {
    dir.create(path, recursive = TRUE)
  }

  # Find package scripts
  pkg_path <- system.file("shell", package = "prevcargaons")

  # Copy main script
  script_src <- file.path(pkg_path, "prevcarga")
  script_dst <- file.path(path, "prevcarga")

  file.copy(script_src, script_dst, overwrite = TRUE)

  # Make executable (Unix)
  if (.Platform$OS.type == "unix") {
    Sys.chmod(script_dst, "755")
  }

  message(sprintf("Installed: %s", script_dst))

  # Install completions
  if (completion) {
    install_completions(pkg_path)
  }

  # Check PATH
  if (!path %in% strsplit(Sys.getenv("PATH"), .Platform$path.sep)[[1]]) {
    message(sprintf("\nAdd %s to your PATH:", path))
    message(sprintf("  export PATH=\"%s:$PATH\"", path))
  }

  invisible(TRUE)
}


#' Install shell completions
#' @noRd
install_completions <- function(pkg_path) {
  # Bash completion
  bash_src <- file.path(pkg_path, "prevcarga.bash")
  bash_dst <- "/etc/bash_completion.d/prevcarga"

  if (file.exists("/etc/bash_completion.d") && file.access("/etc/bash_completion.d", 2) == 0) {
    file.copy(bash_src, bash_dst, overwrite = TRUE)
    message(sprintf("Installed bash completion: %s", bash_dst))
  }

  # Zsh completion
  zsh_dirs <- c(
    "~/.zsh/completions",
    "/usr/local/share/zsh/site-functions"
  )

  for (zsh_dir in zsh_dirs) {
    zsh_dir <- path.expand(zsh_dir)
    if (dir.exists(zsh_dir)) {
      zsh_src <- file.path(pkg_path, "_prevcarga")
      zsh_dst <- file.path(zsh_dir, "_prevcarga")
      file.copy(zsh_src, zsh_dst, overwrite = TRUE)
      message(sprintf("Installed zsh completion: %s", zsh_dst))
      break
    }
  }
}
```

### Usage Examples

```bash
# After installation
prevcarga help
prevcarga train --help
prevcarga predict --date 2025-01-17

# With tab completion (bash/zsh)
prevcarga tr<TAB>        # completes to: prevcarga train
prevcarga train --<TAB>  # shows: --areas --models --start-date ...
prevcarga train --config <TAB>  # completes: .yaml files
```

---

## Test Cases

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| TC-001 | Unix script executable | Runs correctly |
| TC-002 | Windows batch file | Runs correctly |
| TC-003 | PowerShell script | Runs correctly |
| TC-004 | Missing R | Error message |
| TC-005 | Missing package | Install instructions |
| TC-006 | Bash completion | Commands complete |
| TC-007 | Zsh completion | Commands complete |
| TC-008 | install_cli() | Script installed |
| TC-009 | Environment variables | Respected |
| TC-010 | Arguments passed | Correct parsing |

---

## Dependencies

- PC-064-09: CLI Entry Point

---

## Definition of Done

- [ ] Unix shell script working
- [ ] Windows batch file working
- [ ] PowerShell script working
- [ ] Bash completion installed
- [ ] Zsh completion installed
- [ ] install_cli() function working
- [ ] Error handling for missing R
- [ ] Documentation complete
- [ ] Tested on Linux, macOS, Windows
- [ ] Code reviewed and approved
- [ ] Merged to develop branch

---

## Notes

- Shell scripts go in inst/shell for package distribution
- Users may need to add to PATH manually
- Completions enhance developer experience significantly
- Consider adding Fish shell completion in future
