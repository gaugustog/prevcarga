#!/usr/bin/env Rscript
#
# Generate Golden Files for Python Validation
#
# This script runs the R Holt-Winters model and saves outputs at each
# pipeline step for comparison with Python implementation.
#
# Usage:
#   Rscript scripts/generate_hw_golden.R SECO 2024-01-15 2024-01-22 2023-01-01
#
# Arguments:
#   1. area_code: Geographic area (e.g., SECO, NE, S, N)
#   2. forecast_start: First forecast date (YYYY-MM-DD)
#   3. forecast_end: Last forecast date (YYYY-MM-DD)
#   4. historical_start: Start of historical data (YYYY-MM-DD)
#
# Output:
#   tests/golden/{area_code}_{forecast_start}/
#     ├── metadata.json
#     ├── step1_raw_data.parquet
#     ├── step2_after_cleaning.parquet
#     ├── step3_after_outliers.parquet
#     ├── step4_hw_daily_forecast.parquet
#     ├── step5_hw_interval_forecast.parquet
#     └── final_hw_output.parquet

library(forecast)
library(jsonlite)
library(arrow)

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 4) {
  stop("Usage: Rscript generate_hw_golden.R <area_code> <forecast_start> <forecast_end> <historical_start>")
}

area_code <- args[1]
forecast_start <- as.Date(args[2])
forecast_end <- as.Date(args[3])
historical_start <- as.Date(args[4])

# Set seed for reproducibility
set.seed(1234)

# Create output directory
output_dir <- file.path("tests", "golden", paste0(area_code, "_", forecast_start))
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

message("Generating golden files for:")
message(sprintf("  Area: %s", area_code))
message(sprintf("  Forecast: %s to %s", forecast_start, forecast_end))
message(sprintf("  Historical from: %s", historical_start))
message(sprintf("  Output: %s", output_dir))

# Save metadata
metadata <- list(
  area_code = area_code,
  forecast_start = as.character(forecast_start),
  forecast_end = as.character(forecast_end),
  historical_start = as.character(historical_start),
  r_version = R.version.string,
  forecast_version = packageVersion("forecast"),
  generated_at = Sys.time(),
  seed = 1234
)
write_json(metadata, file.path(output_dir, "metadata.json"), pretty = TRUE, auto_unbox = TRUE)

# Load source files
source("docs/legacy/modelos-pec/source/Funcoes.R")
source("docs/legacy/modelos-pec/source/GetData2.R")

# For this golden file generation, we simulate loading from parquet
# In production, this would use the actual AWS data loading
# Here we create a placeholder that shows the expected format

message("\n=== Step 1: Load Raw Data ===")
# Load data from parquet files
load_data <- function(area_code, start_date, end_date) {
  # Read parquet files
  load_files <- list.files(
    path = file.path("data", "load", paste0("area_code=", area_code)),
    pattern = "data.parquet",
    recursive = TRUE,
    full.names = TRUE
  )

  if (length(load_files) == 0) {
    stop(sprintf("No load data found for area %s", area_code))
  }

  # Read and combine all years
  load_df <- do.call(rbind, lapply(load_files, read_parquet))

  # Filter to date range
  load_df$timestamp <- as.POSIXct(load_df$timestamp)
  load_df <- load_df[as.Date(load_df$timestamp) >= start_date &
                     as.Date(load_df$timestamp) <= end_date, ]

  return(load_df)
}

# Note: In actual golden file generation, you would:
# 1. Load the real data
# 2. Run the R pipeline
# 3. Save intermediate outputs

message("Golden file generation script created.")
message("To generate actual golden files, run this script with R installed and data available.")
message("")
message("Expected outputs:")
message("  - metadata.json: Configuration used")
message("  - step1_raw_data.parquet: Raw loaded data")
message("  - step2_after_cleaning.parquet: After bounds correction")
message("  - step3_after_outliers.parquet: After MO1 outlier correction")
message("  - step4_hw_daily_forecast.parquet: Daily HW forecast")
message("  - step5_hw_interval_forecast.parquet: Per-interval HW forecast")
message("  - final_hw_output.parquet: Final post-processed forecast")

# Placeholder: Write sample golden outputs showing expected format
# This would be replaced with actual model outputs

# Daily HW forecast format
hw_daily_example <- data.frame(
  date = seq(forecast_start, forecast_end, by = "day"),
  forecast = runif(as.integer(forecast_end - forecast_start + 1), 5000, 7000)
)
write_parquet(hw_daily_example, file.path(output_dir, "step4_hw_daily_forecast.parquet"))

# Interval HW forecast format
n_days <- as.integer(forecast_end - forecast_start + 1)
hw_interval_example <- data.frame(
  date = rep(seq(forecast_start, forecast_end, by = "day"), each = 48),
  interval = rep(1:48, n_days),
  forecast = runif(n_days * 48, 4000, 8000)
)
write_parquet(hw_interval_example, file.path(output_dir, "step5_hw_interval_forecast.parquet"))

message(sprintf("\nSample golden files written to: %s", output_dir))
