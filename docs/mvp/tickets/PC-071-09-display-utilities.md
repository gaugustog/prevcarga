# PC-071-09: Display Utilities

**Epic:** [EPIC-09: CLI](../epics/EPIC-09-cli.md)
**Task Reference:** T-09.8
**Priority:** Medium
**Estimated Effort:** 1 day

---

## Summary

Implement display utilities for the CLI including progress bars, sparklines, tables, and colored output helpers. These utilities enhance user experience during long-running operations.

---

## Acceptance Criteria

- [ ] Display module created in `R/cli/display.R`
- [ ] Progress bar with percentage and ETA
- [ ] Sparkline for inline data visualization
- [ ] Table formatter for structured output
- [ ] Color and styling utilities
- [ ] Spinner for indeterminate progress

---

## Technical Specification

### File Location
```
R/cli/display.R
```

### Progress Bar

```r
#' Create progress bar
#'
#' @param total Total number of items
#' @param width Bar width in characters
#' @param format Format string
#' @return ProgressBar object
#' @export
progress_bar <- function(total, width = 40, format = NULL) {
  ProgressBar$new(total = total, width = width, format = format)
}


#' @title ProgressBar
#' @description Terminal progress bar with ETA
#' @export
ProgressBar <- R6::R6Class(
"ProgressBar",
  private = list(
    total = NULL,
    current = 0,
    width = 40,
    start_time = NULL,
    format = NULL,
    last_render = NULL,
    render_interval = 0.1,  # seconds

    # Unicode blocks for smooth progress
    blocks = c(
      " ",           # 0/8
      "\u258F",      # 1/8
      "\u258E",      # 2/8
      "\u258D",      # 3/8
      "\u258C",      # 4/8
      "\u258B",      # 5/8
      "\u258A",      # 6/8
      "\u2589",      # 7/8
      "\u2588"       # 8/8
    )
  ),

  public = list(
    #' @description Initialize progress bar
    #' @param total Total number of items
    #' @param width Bar width
    #' @param format Custom format string
    initialize = function(total, width = 40, format = NULL) {
      private$total <- total
      private$width <- width
      private$format <- format %||% "[{bar}] {percent}% | {current}/{total} | ETA: {eta}"
      private$start_time <- Sys.time()
    },

    #' @description Update progress
    #' @param n Increment (default 1)
    #' @param message Optional status message
    tick = function(n = 1, message = NULL) {
      private$current <- min(private$current + n, private$total)

      # Throttle rendering
      now <- Sys.time()
      if (!is.null(private$last_render) &&
          as.numeric(now - private$last_render) < private$render_interval) {
        return(invisible(self))
      }

      private$last_render <- now
      self$render(message)

      invisible(self)
    },

    #' @description Set absolute progress
    #' @param value Current value
    set = function(value) {
      private$current <- min(max(0, value), private$total)
      self$render()
      invisible(self)
    },

    #' @description Render progress bar
    #' @param message Optional status message
    render = function(message = NULL) {
      # Calculate percentage
      pct <- private$current / private$total

      # Build bar
      bar <- self$build_bar(pct)

      # Calculate ETA
      eta <- self$calculate_eta()

      # Format output
      output <- private$format
      output <- gsub("\\{bar\\}", bar, output)
      output <- gsub("\\{percent\\}", sprintf("%3.0f", pct * 100), output)
      output <- gsub("\\{current\\}", private$current, output)
      output <- gsub("\\{total\\}", private$total, output)
      output <- gsub("\\{eta\\}", eta, output)

      if (!is.null(message)) {
        output <- sprintf("%s | %s", output, message)
      }

      # Clear line and print
      cat("\r", output, sep = "")

      # Newline on completion
      if (private$current >= private$total) {
        cat("\n")
      }

      invisible(self)
    },

    #' @description Build bar string
    #' @param pct Percentage complete
    build_bar = function(pct) {
      filled_width <- pct * private$width
      full_blocks <- floor(filled_width)
      partial <- filled_width - full_blocks

      # Full blocks
      bar <- strrep(private$blocks[9], full_blocks)

      # Partial block
      if (full_blocks < private$width) {
        partial_idx <- round(partial * 8) + 1
        bar <- paste0(bar, private$blocks[partial_idx])
        bar <- paste0(bar, strrep(private$blocks[1], private$width - full_blocks - 1))
      }

      bar
    },

    #' @description Calculate estimated time remaining
    calculate_eta = function() {
      if (private$current == 0) {
        return("--:--")
      }

      elapsed <- as.numeric(Sys.time() - private$start_time, units = "secs")
      rate <- private$current / elapsed
      remaining <- (private$total - private$current) / rate

      if (remaining < 60) {
        sprintf("%02.0fs", remaining)
      } else if (remaining < 3600) {
        sprintf("%02.0f:%02.0f", remaining %/% 60, remaining %% 60)
      } else {
        sprintf("%dh %02.0fm", remaining %/% 3600, (remaining %% 3600) %/% 60)
      }
    },

    #' @description Finish progress bar
    finish = function() {
      private$current <- private$total
      self$render()
      invisible(self)
    }
  )
)
```

### Sparkline

```r
#' Create sparkline from values
#'
#' @param values Numeric vector
#' @param width Optional width (default uses all values)
#' @return Character string sparkline
#' @export
sparkline <- function(values, width = NULL) {
  chars <- c(
    "\u2581",  # ▁
    "\u2582",  # ▂
    "\u2583",  # ▃
    "\u2584",  # ▄
    "\u2585",  # ▅
    "\u2586",  # ▆
    "\u2587",  # ▇
    "\u2588"   # █
  )

  # Handle NA
  values <- values[!is.na(values)]

  if (length(values) == 0) {
    return("")
  }

  # Downsample if needed
  if (!is.null(width) && length(values) > width) {
    idx <- seq(1, length(values), length.out = width)
    values <- values[round(idx)]
  }

  # Normalize to 0-7 range
  min_val <- min(values)
  max_val <- max(values)

  if (max_val == min_val) {
    indices <- rep(4, length(values))  # Middle
  } else {
    normalized <- (values - min_val) / (max_val - min_val)
    indices <- round(normalized * 7) + 1
  }

  paste(chars[indices], collapse = "")
}


#' Create inline bar chart
#'
#' @param value Value (0-100)
#' @param width Bar width
#' @param show_pct Show percentage
#' @return Character string
#' @export
inline_bar <- function(value, width = 10, show_pct = TRUE) {
  value <- max(0, min(100, value))
  filled <- round(value / 100 * width)

  bar <- paste0(
    strrep("\u2588", filled),
    strrep("\u2591", width - filled)
  )

  if (show_pct) {
    sprintf("%s %3.0f%%", bar, value)
  } else {
    bar
  }
}
```

### Spinner

```r
#' Create spinner for indeterminate progress
#'
#' @param frames Spinner frame characters
#' @param interval Frame interval in seconds
#' @return Spinner object
#' @export
spinner <- function(frames = NULL, interval = 0.1) {
  Spinner$new(frames = frames, interval = interval)
}


#' @title Spinner
#' @description Animated spinner for indeterminate progress
#' @export
Spinner <- R6::R6Class(
  "Spinner",
  private = list(
    frames = NULL,
    interval = 0.1,
    current_frame = 1,
    running = FALSE,
    message = ""
  ),

  public = list(
    #' @description Initialize spinner
    initialize = function(frames = NULL, interval = 0.1) {
      private$frames <- frames %||% c(
        "\u280B", "\u2819", "\u2839", "\u2838",
        "\u283C", "\u2834", "\u2826", "\u2827",
        "\u2807", "\u280F"
      )
      private$interval <- interval
    },

    #' @description Start spinner with message
    #' @param message Status message
    start = function(message = "Processing...") {
      private$message <- message
      private$running <- TRUE
      self$render()
    },

    #' @description Update spinner message
    #' @param message New message
    update = function(message) {
      private$message <- message
      self$render()
    },

    #' @description Render current frame
    render = function() {
      frame <- private$frames[private$current_frame]
      cat(sprintf("\r%s %s", frame, private$message))

      private$current_frame <- (private$current_frame %% length(private$frames)) + 1
    },

    #' @description Stop spinner with success
    #' @param message Final message
    succeed = function(message = NULL) {
      private$running <- FALSE
      final_message <- message %||% private$message
      cat(sprintf("\r\u2713 %s\n", final_message))
    },

    #' @description Stop spinner with failure
    #' @param message Error message
    fail = function(message = NULL) {
      private$running <- FALSE
      final_message <- message %||% private$message
      cat(sprintf("\r\u2717 %s\n", final_message))
    }
  )
)
```

### Table Formatter

```r
#' Format data as table for terminal
#'
#' @param data Data frame or matrix
#' @param max_width Maximum column width
#' @param header Show header row
#' @return Character vector of lines
#' @export
format_table <- function(data, max_width = 20, header = TRUE) {
  data <- as.data.frame(data)

  # Truncate columns
  for (col in names(data)) {
    data[[col]] <- sapply(data[[col]], function(x) {
      s <- as.character(x)
      if (nchar(s) > max_width) {
        paste0(substr(s, 1, max_width - 3), "...")
      } else {
        s
      }
    })
  }

  # Calculate column widths
  widths <- sapply(names(data), function(col) {
    max(nchar(col), max(nchar(data[[col]])))
  })

  # Format header
  lines <- character()

  if (header) {
    header_line <- paste(mapply(function(name, width) {
      sprintf(paste0("%-", width, "s"), name)
    }, names(data), widths), collapse = " | ")

    separator <- paste(sapply(widths, function(w) strrep("-", w)),
                       collapse = "-+-")

    lines <- c(lines, header_line, separator)
  }

  # Format rows
  for (i in seq_len(nrow(data))) {
    row_line <- paste(mapply(function(val, width) {
      sprintf(paste0("%-", width, "s"), val)
    }, data[i, ], widths), collapse = " | ")

    lines <- c(lines, row_line)
  }

  lines
}


#' Print formatted table
#'
#' @param data Data frame
#' @param ... Arguments passed to format_table
#' @export
print_table <- function(data, ...) {
  lines <- format_table(data, ...)
  cat(paste(lines, collapse = "\n"), "\n")
}
```

### Color Utilities

```r
#' ANSI color codes
#' @noRd
.colors <- list(
  reset = "\033[0m",
  bold = "\033[1m",
  dim = "\033[2m",
  italic = "\033[3m",
  underline = "\033[4m",

  black = "\033[30m",
  red = "\033[31m",
  green = "\033[32m",
  yellow = "\033[33m",
  blue = "\033[34m",
  magenta = "\033[35m",
  cyan = "\033[36m",
  white = "\033[37m",

  bg_black = "\033[40m",
  bg_red = "\033[41m",
  bg_green = "\033[42m",
  bg_yellow = "\033[43m",
  bg_blue = "\033[44m",
  bg_magenta = "\033[45m",
  bg_cyan = "\033[46m",
  bg_white = "\033[47m"
)


#' Apply color to text
#'
#' @param text Text to color
#' @param ... Color names
#' @return Colored text string
#' @export
color <- function(text, ...) {
  colors <- list(...)
  codes <- sapply(colors, function(c) .colors[[c]] %||% "")
  sprintf("%s%s%s", paste(codes, collapse = ""), text, .colors$reset)
}


#' Check if terminal supports colors
#'
#' @return Logical
#' @export
has_color_support <- function() {
  # Check common indicators
  if (Sys.getenv("NO_COLOR") != "") return(FALSE)
  if (Sys.getenv("TERM") == "dumb") return(FALSE)
  if (!isatty(stdout())) return(FALSE)

  TRUE
}


#' Conditional color application
#'
#' @param text Text
#' @param ... Colors to apply
#' @return Text with or without colors
#' @export
cli_color <- function(text, ...) {
  if (has_color_support()) {
    color(text, ...)
  } else {
    text
  }
}
```

### Usage Examples

```r
# Progress bar
pb <- progress_bar(100)
for (i in 1:100) {
  Sys.sleep(0.05)
  pb$tick(message = sprintf("Processing item %d", i))
}
# [████████████████████████████████████████] 100% | 100/100 | ETA: 00s

# Sparkline
values <- c(1, 4, 2, 8, 5, 3, 9, 2, 6)
cat("Trend:", sparkline(values), "\n")
# Trend: ▂▄▂█▅▃█▂▆

# Inline bar
cat("Progress:", inline_bar(75), "\n")
# Progress: ████████░░ 75%

# Spinner
spin <- spinner()
spin$start("Loading data...")
Sys.sleep(2)
spin$succeed("Data loaded")
# ✓ Data loaded

# Table
df <- data.frame(
  Model = c("lgbm", "rf", "hw"),
  MAPE = c(3.2, 3.8, 4.5),
  MAE = c(150, 175, 210)
)
print_table(df)
# Model | MAPE | MAE
# ------+------+----
# lgbm  | 3.2  | 150
# rf    | 3.8  | 175
# hw    | 4.5  | 210

# Colors
cat(color("Success!", "green", "bold"), "\n")
cat(color("Warning!", "yellow"), "\n")
cat(color("Error!", "red", "bold"), "\n")
```

---

## Test Cases

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| TC-001 | progress_bar() creation | Bar created |
| TC-002 | ProgressBar$tick() | Progress updates |
| TC-003 | ProgressBar ETA | ETA calculated |
| TC-004 | sparkline() | Sparkline string |
| TC-005 | sparkline() empty | Empty string |
| TC-006 | inline_bar() | Bar with percentage |
| TC-007 | Spinner start/stop | Frames animate |
| TC-008 | format_table() | Lines formatted |
| TC-009 | color() | ANSI codes added |
| TC-010 | has_color_support() | Boolean returned |

---

## Dependencies

None (standalone utility module)

---

## Definition of Done

- [ ] Progress bar with ETA implemented
- [ ] Sparkline function working
- [ ] Inline bar function working
- [ ] Spinner animation working
- [ ] Table formatter implemented
- [ ] Color utilities with terminal check
- [ ] roxygen2 documentation complete
- [ ] Unit tests (≥80% coverage)
- [ ] Code reviewed and approved
- [ ] Merged to develop branch

---

## Notes

- Unicode characters require UTF-8 terminal support
- Colors disabled if NO_COLOR env var is set
- Progress bar throttles rendering for performance
- Consider adding fallback ASCII characters for limited terminals
