##----------------------------------------------------------------------------##
##-------------------------AVALIAÇÃO DOS RESULTADOS DO MODELO-----------------##
################################################################################

############################################################################
#FUNÇÃO QUE AVALIA A PERFORMANCE DOS MODELOS NO TESTE
############################################################################
evaluate_multioutput_model <- function(y_true, y_pred, target_cols) {
  
  
  metrics <- data.table(
    lead_hour = as.numeric(gsub("y_h", "", target_cols)),
    day = ceiling(as.numeric(gsub("y_h", "", target_cols)) / 24),
    hour_of_day = ((as.numeric(gsub("y_h", "", target_cols)) - 1) %% 24) + 1,
    mae = NA_real_,
    rmse = NA_real_,
    mape = NA_real_,
    r2 = NA_real_
  )
  
  for (i in 1:length(target_cols)) {
    true_vals <- y_true[, i]
    pred_vals <- y_pred[, i]
    
    # Remove nas
    valid_idx <- !is.na(true_vals) & !is.na(pred_vals)
    true_vals <- true_vals[valid_idx]
    pred_vals <- pred_vals[valid_idx]
    
    if (length(true_vals) > 0) {
      metrics$mae[i] <- mean(abs(true_vals - pred_vals))
      metrics$rmse[i] <- sqrt(mean((true_vals - pred_vals)^2))
      metrics$mape[i] <- mean(abs((true_vals - pred_vals) / true_vals)) * 100
      ss_res <- sum((true_vals - pred_vals)^2)
      ss_tot <- sum((true_vals - mean(true_vals))^2)
      metrics$r2[i] <- 1 - (ss_res / ss_tot)
    }
  }
  
  return(metrics)
}


############################################################################
#FUNÇÃO QUE PLOTA A CARGA PREVISTA E A ESPERADA
############################################################################
plot_load_comparison <- function(y_true, y_pred, test_data, target_cols, 
                                 forecast_origins = NULL, max_days = 3) {
  
  # formato lonh de plot
  n_origins <- nrow(y_true)
  
  # Select which forecast origins to plot (default: first few)
  if (is.null(forecast_origins)) {
    origins_to_plot <- 1:min(max_days, n_origins)
  } else {
    origins_to_plot <- forecast_origins
  }
  
  # formato long para plot
  plot_data_list <- list()
  
  for (i in origins_to_plot) {
    origin_date <- test_data$dataOrigem[i]
    
    # Create hourly timestamps for this forecast
    forecast_hours <- 1:length(target_cols)
    timestamps <- origin_date + hours(forecast_hours)
    
    origin_data <- data.table(
      forecast_origin = origin_date,
      hour = forecast_hours,
      timestamp = timestamps,
      day = ceiling(forecast_hours / 24),
      hour_of_day = ((forecast_hours - 1) %% 24) + 1,
      actual = as.numeric(y_true[i, ]),
      predicted = as.numeric(y_pred[i, ])
    )
    
    plot_data_list[[i]] <- origin_data
  }
  
  plot_data <- rbindlist(plot_data_list)
  
  # Create time series plot
  p1 <- ggplot(plot_data, aes(x = timestamp)) +
    geom_line(aes(y = actual, color = "Actual"), size = 0.8) +
    geom_line(aes(y = predicted, color = "Predicted"), size = 0.8, alpha = 0.8) +
    facet_wrap(~ paste("Origin:", forecast_origin), scales = "free_x", ncol = 1) +
    labs(title = "Actual vs Predicted Load - Time Series View",
         x = "Forecast Timestamp", y = "Load (MW)",
         color = "Type") +
    scale_color_manual(values = c("Actual" = "black", "Predicted" = "red")) +
    theme_minimal() +
    theme(legend.position = "bottom")
  
  # Create scatter plot by forecast day
  p2 <- ggplot(plot_data, aes(x = actual, y = predicted)) +
    geom_point(alpha = 0.6, size = 0.8) +
    geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
    facet_wrap(~ paste("Day", day), ncol = 3) +
    labs(title = "Actual vs Predicted by Forecast Day",
         x = "Actual Load (MW)", y = "Predicted Load (MW)") +
    theme_minimal()
  
  # Create daily profile comparison
  daily_profiles <- plot_data[, .(
    actual = mean(actual, na.rm = TRUE),
    predicted = mean(predicted, na.rm = TRUE)
  ), by = .(day, hour_of_day)]
  
  daily_profiles_long <- melt(daily_profiles, 
                              id.vars = c("day", "hour_of_day"),
                              variable.name = "type", 
                              value.name = "load")
  
  p3 <- ggplot(daily_profiles_long, aes(x = hour_of_day, y = load, color = type)) +
    geom_line(size = 1) +
    facet_wrap(~ paste("Day", day), ncol = 3) +
    labs(title = "Average Daily Load Profiles by Forecast Day",
         x = "Hour of Day", y = "Average Load (MW)",
         color = "Type") +
    scale_color_manual(values = c("actual" = "black", "predicted" = "red")) +
    scale_x_continuous(breaks = seq(0, 24, 6)) +
    theme_minimal() +
    theme(legend.position = "bottom")

  return(list(timeseries = p1, scatter = p2, profiles = p3))
}