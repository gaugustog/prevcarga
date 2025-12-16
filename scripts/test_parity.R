#!/usr/bin/env Rscript
#
# Test parity between modular R code and legacy R code
#

.libPaths(c("~/R/library", .libPaths()))
setwd("/opt/source/prevcarga-R")

library(data.table)
library(arrow)

message("=" , strrep("=", 59))
message("Parity Test: Modular R vs Legacy R")
message("=", strrep("=", 59))
message("")

# Parameters
area_code <- "SECO"
forecast_start <- as.Date("2024-06-01")
forecast_end <- as.Date("2024-06-08")
historical_start <- as.Date("2023-06-01")
n_steps <- 8

# ============================================
# Run MODULAR R code
# ============================================
message("Running MODULAR R pipeline...")
source("R/pipeline.R")

modular_result <- quick_forecast(
  area_code = area_code,
  forecast_start = forecast_start,
  forecast_end = forecast_end,
  historical_start = historical_start,
  data_path = "data"
)

modular_daily <- modular_result$daily_forecast$forecast
message("")

# ============================================
# Run LEGACY R code (direct HoltWinters)
# ============================================
message("Running LEGACY R HoltWinters directly...")

# Load data same way as legacy
load_path <- file.path("data", "load", paste0("area_code=", area_code))
years <- list.dirs(load_path, recursive = FALSE, full.names = FALSE)

all_data <- list()
for (year in years) {
  parquet_path <- file.path(load_path, paste0(year), "data.parquet")
  if (file.exists(parquet_path)) {
    df <- read_parquet(parquet_path)
    all_data[[year]] <- df
  }
}

load_df <- rbindlist(all_data)
load_df[, timestamp := as.POSIXct(timestamp)]
load_df[, date := as.Date(timestamp)]

# Filter to historical range
hist_df <- load_df[date >= historical_start & date < forecast_start]

# Calculate daily means
daily_means <- hist_df[, .(load_mwh = mean(load_mwh)), by = date][order(date)]

message(sprintf("  Loaded %d days of data", nrow(daily_means)))

# Fit HoltWinters (legacy style)
set.seed(1234)
ts_load <- ts(daily_means$load_mwh, frequency = 7)
hw_fit <- HoltWinters(ts_load, seasonal = "multiplicative")
legacy_daily <- as.numeric(predict(hw_fit, n.ahead = n_steps))

message("")

# ============================================
# Compare results
# ============================================
message("=", strrep("=", 59))
message("COMPARISON: Modular vs Legacy")
message("=", strrep("=", 59))
message("")

forecast_dates <- seq(forecast_start, forecast_end, by = "day")

message(sprintf("%-12s  %-15s  %-15s  %-10s", "Date", "Modular", "Legacy", "Diff"))
message(strrep("-", 55))

for (i in 1:n_steps) {
  diff_pct <- (modular_daily[i] - legacy_daily[i]) / legacy_daily[i] * 100
  message(sprintf("%-12s  %-15.2f  %-15.2f  %+.4f%%",
                  forecast_dates[i],
                  modular_daily[i],
                  legacy_daily[i],
                  diff_pct))
}

message("")

# Calculate correlation
correlation <- cor(modular_daily, legacy_daily)
max_diff <- max(abs(modular_daily - legacy_daily) / legacy_daily * 100)

message("METRICS:")
message(sprintf("  Correlation:     %.10f", correlation))
message(sprintf("  Max difference:  %.6f%%", max_diff))
message("")

if (correlation > 0.9999999) {
  message("RESULT: EXACT PARITY ACHIEVED!")
} else if (correlation > 0.999) {
  message("RESULT: Very close parity (correlation > 0.999)")
} else {
  message("RESULT: Parity not achieved - investigate differences")
}
