##----------------------------------------------------------------------------##
##--------------FUNÇÕES PARA O TREINAMENTO E TESTE DO MODELO------------------##
################################################################################

############################################################################
#FUNÇÃO QUE TREINA MODELOS RANDOM FOREST, UM PARA CADA LEAD HOUR
############################################################################
train_multioutput_rf <- function(train_data, feature_cols, target_cols, ntree = 100) {
  
  cat("Treinando modelo Random Forest com", length(target_cols), "targets...\n")
  
  # Treina um modelo para cada hora de saída - lead hour
  models <- list()
  feature_usage_summary <- data.frame(
    lead_hour = integer(),
    n_features = integer(),
    stringsAsFactors = FALSE
  )
  
  for (i in 1:length(target_cols)) {
    target_name <- target_cols[i]
    lead_hour <- as.numeric(gsub("y_h", "", target_name))
    day_ahead <- ceiling(lead_hour / 24)
    hour_of_day <- ((lead_hour - 1) %% 24)
    
    if (i %% 24 == 1) {  # Printa o progresso
      cat(sprintf("Treinando os modelos para o dia %d of 9 (lead hours %d-%d)...\n", 
                  day_ahead, i, min(i+23, length(target_cols))))
    }
    
    # ====== SELECT FEATURES SPECIFIC TO THIS LEAD HOUR ======
    
    # 1. Origin calendar features 
    origin_features <- grep("^origin_", feature_cols, value = TRUE)
    
    # 2. Temperature features for the TARGET DAY
    #temp_features <- grep(paste0("^(tmax|tmin)_", day_ahead, "$"), feature_cols, value = TRUE)
    # 2. Temperature features for the TARGET DAY and ALL PRECEDING DAYS
    temp_features <- grep(paste0("^(tmax|tmin)_[1-", day_ahead, "]$"), 
                          feature_cols, value = TRUE)
    
    # 3. Day-of-week features for the TARGET DAY only
    dow_features <- grep(paste0("^day", day_ahead, "_"), feature_cols, value = TRUE)
    
    # 4. Hour-specific features for THIS LEAD HOUR only
    hour_features <- grep(paste0("^h", sprintf("%03d", lead_hour), "_"), feature_cols, value = TRUE)
    
    # 5. Seasonal features for the TARGET DAY only
    seasonal_features <- grep(paste0("^day", day_ahead, "_(month|quarter|is_summer|is_winter)"), 
                              feature_cols, value = TRUE)
    # 6. Previous day load features 
    prev_day_features <- grep("^prev_day_", feature_cols, value = TRUE)
    
    # 7. Previous 2-day load features 
    prev_2day_features <- grep("^prev_2day_", feature_cols, value = TRUE)
    
    # 8. Recent load level features 
    recent_level_features <- grep("^(prev_7day_mean|prev_3day_mean|recent_vs_historical)$", 
                                  feature_cols, value = TRUE)
    
    # 9. Holiday proximity features
    holiday_features <- grep("^(has_holiday_in_horizon|days_to_next_holiday|days_since_last_holiday)$", 
                             feature_cols, value = TRUE)
    
    # 10. Christmas/New Year features for the TARGET DAY and origin
    christmas_newyear_features <- grep(paste0("^(origin_is_christmas|origin_is_newyear|day", day_ahead, "_(is_christmas|is_newyear))$"), 
                                       feature_cols, value = TRUE)
    

    # Combine all relevant features for this lead_hour
    selected_features <- unique(c(
      origin_features,
      temp_features,
      dow_features,
      hour_features,
      seasonal_features,
      prev_day_features,
      prev_2day_features,
      recent_level_features,
      holiday_features
    ))

    # Verifica se as features selcionadas existem na tabela anterior
    selected_features <- intersect(selected_features, feature_cols)
    
    # Track feature usage
    feature_usage_summary <- rbind(feature_usage_summary, 
                                   data.frame(lead_hour = lead_hour, 
                                              n_features = length(selected_features)))
    
    # Monta o dataset de treino
    X_train <- as.matrix(train_data[, selected_features, with = FALSE])
    Y_train <- train_data[[target_name]]
    
    # Pega só casos completos
    complete_rows <- complete.cases(X_train) & !is.na(Y_train)
    X_train_clean <- X_train[complete_rows, , drop = FALSE]
    Y_train_clean <- Y_train[complete_rows]
    
    set.seed(123)
    
    
    # Treina o RF para o leadhour
    rf_model <- randomForest(
      x = X_train_clean,
      y = Y_train_clean,
      ntree = ntree,
      importance = TRUE,
      do.trace = FALSE
    )
    
    # Guarda o modelo
    models[[target_name]] <- list(
      model = rf_model,
      features = selected_features,
      lead_hour = lead_hour,
      day_ahead = day_ahead,
      hour_of_day = hour_of_day,
      n_obs = nrow(X_train_clean)
    )
  }
  
  cat("\n=== TRAINING COMPLETE ===\n")
  cat(length(models), "modelos individuais treinados\n")
  cat(sprintf("Media de features por modelo: %.1f (vs %d total)\n", 
              mean(feature_usage_summary$n_features), length(feature_cols)))
  
  return(models)
}

############################################################################
#FUNÇÃO QUE TESTA MODELOS RANDOM FOREST, UM PARA CADA LEAD HOUR
############################################################################
predict_multioutput_rf <- function(models, test_data, feature_cols, target_cols) {
  # Faz a previsão para cada saída 
  predictions <- matrix(NA, nrow = nrow(test_data), ncol = length(target_cols))
  colnames(predictions) <- target_cols
  
  for (i in 1:length(target_cols)) {
    target_name <- target_cols[i]
    model_info <- models[[target_name]]
    
    # Seleciona as features utilizadas para esse modelo
    selected_features <- model_info$features
    
    # Monta a matriz de teste
    X_test <- as.matrix(test_data[, selected_features, with = FALSE])
    
    # Realiza as previsões
    predictions[, i] <- predict(model_info$model, X_test)
  }
  
  return(predictions)
}
