# PC-075-09: CLI Tests

**Epic:** [EPIC-09: CLI](../epics/EPIC-09-cli.md)
**Task Reference:** T-09.11
**Priority:** High
**Estimated Effort:** 2 days

---

## Summary

Create comprehensive test suite for the CLI module covering command parsing, validators, shell commands, and end-to-end workflow tests.

---

## Acceptance Criteria

- [ ] Test file created at `tests/testthat/test-cli.R`
- [ ] Command parsing tests
- [ ] Validator tests for all types
- [ ] Shell command tests
- [ ] Display utility tests
- [ ] E2E tests for main workflows
- [ ] Test coverage ≥80% for CLI module

---

## Technical Specification

### File Location
```
tests/testthat/test-cli.R
tests/testthat/test-cli-validators.R
tests/testthat/test-cli-display.R
tests/testthat/test-cli-commands.R
tests/testthat/test-cli-e2e.R
```

### Test Fixtures

```r
# Helper: Create test config
create_test_cli_config <- function(tmp_dir) {
  config_content <- '
project:
  name: "CLI-Test"
  version: "1.0.0"

storage:
  backend: "local"
  local_path: "test_data"

regions:
  areas:
    - RJ
    - SP
    - MG

models:
  default: "lgbm"
  plugins:
    lgbm:
      enabled: true
      config:
        num_leaves: 31
    rf:
      enabled: true
      config:
        num.trees: 100
    hw:
      enabled: false
'
  config_path <- file.path(tmp_dir, "test_config.yaml")
  writeLines(config_content, config_path)
  config_path
}


# Helper: Capture CLI output
capture_cli_output <- function(expr) {
  output <- capture.output({
    result <- tryCatch(
      expr,
      error = function(e) e
    )
  })

  list(
    output = output,
    result = result
  )
}


# Helper: Mock storage backend
create_mock_cli_storage <- function() {
  list(
    save = function(data, path) invisible(TRUE),
    load = function(path) NULL,
    exists = function(path) FALSE,
    list = function(path) character(0)
  )
}
```

### Main Entry Point Tests

```r
describe("CLI Entry Point", {
  it("shows help with no arguments", {
    result <- capture_cli_output(main(c()))
    expect_true(any(grepl("Usage:", result$output)))
    expect_true(any(grepl("Commands:", result$output)))
  })

  it("shows help with --help", {
    result <- capture_cli_output(main(c("--help")))
    expect_true(any(grepl("Usage:", result$output)))
  })

  it("shows version with --version", {
    result <- capture_cli_output(main(c("--version")))
    expect_true(any(grepl("PrevCarga version", result$output)))
  })

  it("errors on unknown command", {
    result <- capture_cli_output(main(c("unknown_cmd")))
    expect_true(inherits(result$result, "error"))
    expect_match(conditionMessage(result$result), "Unknown command")
  })

  it("passes arguments to command handler", {
    # Mock train command
    with_mock(
      cmd_train = function(args) {
        list(args = args)
      },
      {
        result <- main(c("train", "--areas", "RJ"))
        expect_equal(result$args, c("--areas", "RJ"))
      }
    )
  })
})
```

### Command Registry Tests

```r
describe("Command Registry", {
  it("registers commands", {
    register_command("test_cmd", function(args) "test", "Test command")
    expect_true(has_command("test_cmd"))
  })

  it("retrieves registered handlers", {
    handler <- get_command_handler("test_cmd")
    expect_true(is.function(handler))
  })

  it("returns NULL for unregistered command", {
    handler <- get_command_handler("nonexistent")
    expect_null(handler)
  })

  it("lists all commands", {
    commands <- get_all_commands()
    expect_true(is.list(commands))
    expect_true(length(commands) > 0)
  })
})
```

### Validator Tests

```r
describe("Date Validators", {
  it("validates YYYY-MM-DD format", {
    result <- validate_date("2024-01-15")
    expect_s3_class(result, "Date")
    expect_equal(result, as.Date("2024-01-15"))
  })

  it("validates YYYY/MM/DD format", {
    result <- validate_date("2024/01/15")
    expect_equal(result, as.Date("2024-01-15"))
  })

  it("validates YYYYMMDD format", {
    result <- validate_date("20240115")
    expect_equal(result, as.Date("2024-01-15"))
  })

  it("rejects invalid date", {
    expect_error(validate_date("not-a-date"), "invalid date format")
  })

  it("rejects NULL when not allowed", {
    expect_error(validate_date(NULL), "is required")
  })

  it("allows NULL when specified", {
    result <- validate_date(NULL, allow_null = TRUE)
    expect_null(result)
  })
})


describe("Date Range Validators", {
  it("validates valid range", {
    result <- validate_date_range("2024-01-01", "2024-12-31")
    expect_equal(result$start_date, as.Date("2024-01-01"))
    expect_equal(result$end_date, as.Date("2024-12-31"))
  })

  it("rejects start >= end", {
    expect_error(
      validate_date_range("2024-12-31", "2024-01-01"),
      "must be before"
    )
  })

  it("warns on very large range", {
    expect_warning(
      validate_date_range("2010-01-01", "2024-12-31"),
      "very large range"
    )
  })
})


describe("Area Validators", {
  tmp_dir <- NULL
  config <- NULL

  setup({
    tmp_dir <<- tempdir()
    config_path <- create_test_cli_config(tmp_dir)
    config <<- ConfigManager$new()
    config$load(config_path)
  })

  it("validates single area", {
    result <- validate_areas("RJ", config)
    expect_equal(result, "RJ")
  })

  it("validates multiple areas", {
    result <- validate_areas("RJ,SP,MG", config)
    expect_equal(result, c("RJ", "SP", "MG"))
  })

  it("expands 'all' to config areas", {
    result <- validate_areas("all", config)
    expect_equal(result, c("RJ", "SP", "MG"))
  })

  it("rejects invalid area", {
    expect_error(
      validate_areas("INVALID", config),
      "invalid area codes"
    )
  })

  it("removes duplicates", {
    expect_warning(
      result <- validate_areas("RJ,RJ,SP", config),
      "duplicate"
    )
    expect_equal(result, c("RJ", "SP"))
  })
})


describe("Model Validators", {
  it("validates registered model", {
    # Assume lgbm is registered
    with_mock(
      has_model = function(name) name %in% c("lgbm", "rf"),
      list_models = function() c("lgbm", "rf"),
      {
        result <- validate_models("lgbm")
        expect_equal(result, "lgbm")
      }
    )
  })

  it("validates multiple models", {
    with_mock(
      has_model = function(name) name %in% c("lgbm", "rf"),
      {
        result <- validate_models("lgbm,rf")
        expect_equal(result, c("lgbm", "rf"))
      }
    )
  })

  it("rejects unknown model", {
    with_mock(
      has_model = function(name) FALSE,
      list_models = function() c("lgbm", "rf"),
      {
        expect_error(validate_models("unknown"), "unknown model")
      }
    )
  })
})


describe("File Path Validators", {
  it("validates existing file", {
    tmp_file <- tempfile(fileext = ".yaml")
    writeLines("test", tmp_file)
    on.exit(unlink(tmp_file))

    result <- validate_file_path(tmp_file)
    expect_equal(result, path.expand(tmp_file))
  })

  it("rejects non-existent file", {
    expect_error(
      validate_file_path("/nonexistent/file.txt"),
      "not found"
    )
  })

  it("creates directory when requested", {
    tmp_dir <- file.path(tempdir(), "new_dir")
    on.exit(unlink(tmp_dir, recursive = TRUE))

    result <- validate_dir_path(tmp_dir, create = TRUE)
    expect_true(dir.exists(tmp_dir))
  })
})


describe("Numeric Validators", {
  it("validates integer", {
    result <- validate_integer(5)
    expect_equal(result, 5L)
  })

  it("validates integer range", {
    result <- validate_integer(5, min = 1, max = 10)
    expect_equal(result, 5L)
  })

  it("rejects out of range", {
    expect_error(validate_integer(15, max = 10), "at most")
    expect_error(validate_integer(0, min = 1), "at least")
  })

  it("validates integer list", {
    result <- validate_integer_list("1,2,3,4")
    expect_equal(result, 1:4)
  })
})


describe("Choice Validators", {
  it("validates valid choice", {
    result <- validate_choice("batch", c("batch", "intraday"))
    expect_equal(result, "batch")
  })

  it("is case insensitive", {
    result <- validate_choice("BATCH", c("batch", "intraday"))
    expect_equal(result, "batch")
  })

  it("rejects invalid choice", {
    expect_error(
      validate_choice("invalid", c("batch", "intraday")),
      "invalid value"
    )
  })
})
```

### Display Utility Tests

```r
describe("Progress Bar", {
  it("creates progress bar", {
    pb <- progress_bar(100)
    expect_s3_class(pb, "ProgressBar")
  })

  it("updates on tick", {
    pb <- progress_bar(10)
    output <- capture.output(pb$tick())
    expect_true(length(output) > 0)
  })

  it("calculates ETA", {
    pb <- progress_bar(100)
    pb$tick(50)
    # ETA should be calculated
  })

  it("finishes at 100%", {
    pb <- progress_bar(10)
    output <- capture.output({
      for (i in 1:10) pb$tick()
    })
    expect_true(any(grepl("100%", output)))
  })
})


describe("Sparkline", {
  it("creates sparkline from values", {
    result <- sparkline(c(1, 4, 2, 8, 5))
    expect_true(is.character(result))
    expect_true(nchar(result) == 5)
  })

  it("handles empty input", {
    result <- sparkline(c())
    expect_equal(result, "")
  })

  it("handles NA values", {
    result <- sparkline(c(1, NA, 3))
    expect_true(nchar(result) == 2)  # NA removed
  })

  it("downsamples when width specified", {
    result <- sparkline(1:100, width = 10)
    expect_true(nchar(result) == 10)
  })
})


describe("Table Formatter", {
  it("formats data frame", {
    df <- data.frame(a = 1:3, b = c("x", "y", "z"))
    lines <- format_table(df)

    expect_true(length(lines) == 5)  # header + separator + 3 rows
    expect_true(any(grepl("a", lines)))
  })

  it("truncates long values", {
    df <- data.frame(long = rep("very_long_string_here", 3))
    lines <- format_table(df, max_width = 10)

    expect_true(all(nchar(lines) < 20))
  })
})


describe("Color Utilities", {
  it("applies color codes", {
    result <- color("test", "red")
    expect_match(result, "\033\\[31m")
    expect_match(result, "\033\\[0m")
  })

  it("applies multiple styles", {
    result <- color("test", "bold", "green")
    expect_match(result, "\033\\[1m")
    expect_match(result, "\033\\[32m")
  })

  it("detects color support", {
    result <- has_color_support()
    expect_true(is.logical(result))
  })
})
```

### Command Tests

```r
describe("Train Command", {
  tmp_dir <- NULL
  config_path <- NULL

  setup({
    tmp_dir <<- tempdir()
    config_path <<- create_test_cli_config(tmp_dir)
  })

  it("parses basic options", {
    with_mock(
      execute_train = function(opts) opts,
      {
        result <- cmd_train(c(
          "--config", config_path,
          "--areas", "RJ",
          "--start-date", "2024-01-01",
          "--end-date", "2024-12-31"
        ))

        expect_equal(result$areas, "RJ")
        expect_equal(result$`start-date`, "2024-01-01")
      }
    )
  })

  it("requires start-date", {
    expect_error(
      cmd_train(c("--config", config_path, "--end-date", "2024-12-31")),
      "start-date"
    )
  })

  it("requires end-date", {
    expect_error(
      cmd_train(c("--config", config_path, "--start-date", "2024-01-01")),
      "end-date"
    )
  })

  it("validates date order", {
    expect_error(
      cmd_train(c(
        "--config", config_path,
        "--start-date", "2024-12-31",
        "--end-date", "2024-01-01"
      )),
      "before"
    )
  })
})


describe("Predict Command", {
  it("parses mode option", {
    with_mock(
      execute_predict = function(opts) opts,
      {
        result <- cmd_predict(c("--mode", "intraday", "--date", "2024-01-15"))
        expect_equal(result$mode, "intraday")
      }
    )
  })

  it("validates mode", {
    expect_error(
      cmd_predict(c("--mode", "invalid")),
      "batch.*intraday"
    )
  })
})


describe("Report Command", {
  it("requires output", {
    expect_error(
      cmd_report(c("forecast", "--date", "2024-01-15")),
      "output.*required"
    )
  })

  it("validates report type", {
    expect_error(
      cmd_report(c("invalid_type", "--output", "out.html")),
      "Invalid report type"
    )
  })

  it("requires date for forecast", {
    expect_error(
      cmd_report(c("forecast", "--output", "out.html")),
      "date.*required"
    )
  })
})
```

### E2E Tests

```r
describe("E2E: Train Workflow", {
  skip_if_not_installed("prevcargaons")

  tmp_dir <- NULL

  setup({
    tmp_dir <<- tempdir()
    # Set up test data
  })

  it("runs complete training", {
    # Mock all dependencies
    # Run full workflow
    # Verify outputs
  })
})


describe("E2E: Predict Workflow", {
  skip_if_not_installed("prevcargaons")

  it("generates predictions", {
    # Mock dependencies
    # Run prediction
    # Verify output file
  })
})
```

---

## Test Cases Summary

| Component | Test Count | Coverage Target |
|-----------|------------|-----------------|
| Entry Point | 8 | 90% |
| Command Registry | 5 | 95% |
| Date Validators | 10 | 95% |
| Area Validators | 6 | 90% |
| Model Validators | 4 | 90% |
| File Validators | 5 | 90% |
| Numeric Validators | 6 | 95% |
| Progress Bar | 5 | 85% |
| Sparkline | 5 | 90% |
| Table Formatter | 3 | 85% |
| Color Utilities | 4 | 80% |
| Train Command | 8 | 85% |
| Predict Command | 6 | 85% |
| Report Command | 6 | 85% |
| E2E Tests | 5 | N/A |
| **Total** | **~86** | **≥80%** |

---

## Dependencies

- PC-064-09: CLI Entry Point
- PC-065-09 through PC-074-09: All CLI commands
- PC-072-09: Input Validators
- PC-071-09: Display Utilities

---

## Definition of Done

- [ ] All test files created
- [ ] Entry point tests passing
- [ ] Validator tests passing
- [ ] Display utility tests passing
- [ ] Command tests passing
- [ ] E2E tests passing
- [ ] Coverage ≥80% for CLI module
- [ ] CI pipeline passing
- [ ] Code reviewed and approved
- [ ] Merged to develop branch

---

## Notes

- Use `testthat` describe/it style for BDD
- Mock external dependencies with `testthat::with_mock`
- E2E tests may be skipped in CI for speed
- Consider adding integration tests with real data samples
