##----------------------------------------------------------------------------##
##---------------------------EXECUÇÃO DO PROGRAMA-----------------------------##
################################################################################
setwd('D:/OneDrive - Operador Nacional do Sistema Eletrico/_PEC_Energetico/Modelos/PrevCarga_Dmais/R')
source('./0_config.R')
source('./1_funcoes_obtencao_dados.R')
source('./2_funcoes_dataset_modelo.R')
source('./3_treinamento_teste_modelo.R')
source('./4_avalia_performance_modelo.R')

##----------------------------------------------------------------------------##
##---------------------------EXECUÇÃO POR MÊS DE TESTE-----------------------##
################################################################################

# Lista para armazenar todos os resultados
resultados_todos_meses <- list()

# Loop através de cada mês de teste
for (m in names(periodos_treino)) {
  
  cat("\n\n########################################\n")
  cat("PROCESSANDO MÊS DE TESTE:", m, "\n")
  cat("########################################\n")
  
  # Configurar datas para este mês
  DataIni <- periodos_treino[[m]]$DataIni
  DataFin <- periodos_treino[[m]]$DataFin
  
  cat("Período de dados:", DataIni, "a", DataFin, "\n")
  
  # Criar pasta para resultados deste mês
  pasta_resultados <- file.path("D:/OneDrive - Operador Nacional do Sistema Eletrico/_PEC_Energetico/Modelos/PrevCarga_Dmais/resultados", m)
  if (!dir.exists(pasta_resultados)) {
    dir.create(pasta_resultados, recursive = TRUE)
  }
  
  caminho_entrada <- file.path("D:/OneDrive - Operador Nacional do Sistema Eletrico/_PEC_Energetico/Modelos/PrevCarga_Dmais/dados_entrada")

  ##----------------------------------------------------------------------------##
  ##---------------------------IMPORTA DADOS------------------------------------##
  ##----------------------------------------------------------------------------##
  
  cat("\n=== IMPORTANDO DADOS ===\n")
  
  # Importa dados de carga
  for (s in names(subsistemas)) {
    dt_load <- fread(file.path(caminho_entrada, paste0("dt_load_", s, ".csv")),
                     sep = ";",
                     dec = ",")
    name_dt <- paste0("dt_load_", s)
    assign(name_dt, dt_load[dia >= DataIni & dia <= DataFin])
  }
  
  # Importa dados de feriados
  for (s in names(subsistemas)){
    assign(paste0("dt_feriados_", s), get_feriado(s, DataIni, feriados_unificados))
  }
  
  # Importa dados de temperatura
  for (s in names(subsistemas)){
    dt_temp <- fread(file.path(caminho_entrada, paste0("dt_temp_", s, ".csv")),
                     sep = ";",
                     dec = ",")
    name_dt_temp <- paste0("dt_temp_", s)
    assign(name_dt_temp, dt_temp[dataOrigem >= DataIni & dataOrigem <= DataFin])
  }
  
  ##----------------------------------------------------------------------------##
  ##---------------------------ORGANIZAÇÃO DOS DTs------------------------------##
  ##----------------------------------------------------------------------------##
  
  cat("\n=== ORGANIZANDO DATASETS ===\n")
  
  for (s in names(subsistemas)){
    assign(paste0("dt_temp_ajustado_", s), ajusta_dt_temp(s))
    assign(paste0("dt_", s), uniao_dts(s))
  }
  
  ##----------------------------------------------------------------------------##
  ##------------------CONSTRUÇÃO DO DATASET PARA MULTIOUTPUT MODEL--------------##
  ##----------------------------------------------------------------------------##
  
  cat("\n=== CONSTRUINDO FEATURES ===\n")
  
  for (s in names(subsistemas)){
    assign(paste0("dt_features_", s), multioutput_dt(s))
    assign(paste0("dt_final_", s), split_train_test(s))
  }

  ##----------------------------------------------------------------------------##
  ##------------------TREINAMENTO E TESTE DO MODELO ----------------------------##
  ##----------------------------------------------------------------------------##
  
  cat("\n=== TREINANDO MODELOS ===\n")
  
  # Lista para armazenar resultados de todos os subsistemas
  resultados_modelos <- list()
  
  for (s in names(subsistemas)) {
    cat("\n==============================\n")
    cat("Treinando modelo para subsistema:", s, "\n")
    cat("==============================\n")
    
    dt_final <- get(paste0("dt_final_", s))
    train_data <- dt_final$train
    test_data  <- dt_final$test
    feature_cols <- dt_final$feature_cols
    target_cols  <- dt_final$target_cols
    
    # Treina o modelo
    rf_models <- train_multioutput_rf(
      train_data = train_data,
      feature_cols = feature_cols,
      target_cols = target_cols,
      ntree = 100
    )
    
    # Previsões
    test_predictions <- predict_multioutput_rf(
      models = rf_models,
      test_data = test_data,
      feature_cols = feature_cols,
      target_cols = target_cols
    )
    
    # Avaliação
    test_actual <- as.matrix(test_data[, target_cols, with = FALSE])
    performance_metrics <- evaluate_multioutput_model(
      y_true = test_actual,
      y_pred = test_predictions,
      target_cols = target_cols
    )
    
    # Resumo
    summary_metrics <- data.table(
      subsistema = s,
      Mean_MAE  = round(mean(performance_metrics$mae, na.rm = TRUE), 2),
      Mean_RMSE = round(mean(performance_metrics$rmse, na.rm = TRUE), 2),
      Mean_MAPE = round(mean(performance_metrics$mape, na.rm = TRUE), 2),
      Mean_R2   = round(mean(performance_metrics$r2, na.rm = TRUE), 3)
    )
    
    cat("\nPerformance Summary -", s, ":\n")
    print(summary_metrics)
    
    # Performance por horizonte
    daily_performance <- performance_metrics[, .(
      mae  = round(mean(mae, na.rm = TRUE), 2),
      rmse = round(mean(rmse, na.rm = TRUE), 2),
      mape = round(mean(mape, na.rm = TRUE), 2),
      r2   = round(mean(r2, na.rm = TRUE), 3)
    ), by = day]
    print(daily_performance)
    
    # Armazena resultados
    resultados_modelos[[s]] <- list(
      modelo = rf_models,
      metrics = performance_metrics,
      resumo = summary_metrics,
      daily = daily_performance,
      pred = test_predictions,
      test_data = test_data
    )
    
    # Plotar previsões para cada subsistema
    try({
      plot_load_comparison(
        y_true = test_actual,
        y_pred = test_predictions,
        test_data = test_data,
        target_cols = target_cols,
        max_days = 3
      )
    }, silent = TRUE)
  }
  
  ##----------------------------------------------------------------------------##
  ##------------------CONSOLIDAÇÃO DE RESULTADOS--------------------------------##
  ##----------------------------------------------------------------------------##
  
  # Consolidar métricas de todos os subsistemas
  resumo_geral <- rbindlist(lapply(resultados_modelos, function(x) x$resumo))
  cat("\n==== RESUMO FINAL DE PERFORMANCE ====\n")
  print(resumo_geral)

  ##----------------------------------------------------------------------------##
  ##------------------COMPARAÇÃO COM O PREVCARGADESSEM--------------------------##
  ##----------------------------------------------------------------------------##
  
  cat("\n=== COMPARAÇÃO COM PREVCARGA ===\n")
  
  setwd('D:/OneDrive - Operador Nacional do Sistema Eletrico/_PEC_Energetico/Modelos/PrevCarga_Dmais')
  
  # Carrega dados do PrevCarga 
  dt_SECO_prevcarga <- fread(
    "./dados_prevcarga/dt_SECO_prevcarga.csv",
    sep = ";",
    dec = ",",
    header = TRUE
  )
  
  # Processar dados do PrevCarga
  dt_SECO_prevcarga[, hora := lubridate::hour(Data_target)]
  dt_SECO_prevcarga[, dia_target := lubridate::date(Data_target)]
  dt_SECO_prevcarga <- dt_SECO_prevcarga[, .(previsao_prevcarga = mean(previsao_prevcarga, na.rm = TRUE)), by = .(hora, dia_target, cod_areacarga, DataOrigem)]
  
  dt_SECO_prevcarga[, day_of_horizon := as.integer(date(dia_target) - date(DataOrigem))]
  dt_SECO_prevcarga[, lead_hour := (day_of_horizon - 1L) * 24L + (hora + 1L)]
  
  dt_SECO_prevcarga_multioutput <- dt_SECO_prevcarga[lead_hour >= 1 & lead_hour <= 168]
  
  # Criar matriz para comparação
  dt_SECO_prevcarga_target_wide <- dcast(
    dt_SECO_prevcarga_multioutput,
    DataOrigem ~ lead_hour,
    value.var = "previsao_prevcarga"
  )
  

  target_cols <- paste0("y_h", sprintf("%03d", 1:168))
  setnames(dt_SECO_prevcarga_target_wide, as.character(1:168), target_cols)
  
  # Obter dados de teste do modelo atual
  test_data_seco <- resultados_modelos[["SECO"]]$test_data
  test_actual_seco <- as.matrix(test_data_seco[, target_cols, with = FALSE])
  rf_predictions_seco <- resultados_modelos[["SECO"]]$pred[, 1:168]  # Take only first 168 hours
  
  # Filtrar dados do PrevCarga para as mesmas datas de teste
  days_test <- test_data_seco$dataOrigem
  prevcarga_test <- dt_SECO_prevcarga_target_wide[DataOrigem %in% days_test]
  prevcarga_test_matrix <- as.matrix(prevcarga_test[, target_cols, with = FALSE])
  
  # Avaliar performance do PrevCarga
  performance_metrics_prevcarga <- evaluate_multioutput_model(
    y_true = test_actual_seco,
    y_pred = prevcarga_test_matrix,
    target_cols = target_cols
  )
  
  # Obter performance do modelo Random Forest
  performance_metrics_rf <- resultados_modelos[["SECO"]]$metrics
  performance_metrics_rf <- performance_metrics_rf[day <= 7]  # Focar apenas nos primeiros 7 dias (168 horas)
  ##----------------------------------------------------------------------------##
  ##------------------COMPARAÇÃO DE MÉTRICAS LADO A LADO-----------------------##
  ##----------------------------------------------------------------------------##
  
  # Resumo geral - Random Forest
  summary_rf <- data.table(
    Modelo = "Random Forest",
    Mean_MAE  = round(mean(performance_metrics_rf$mae, na.rm = TRUE), 2),
    Mean_RMSE = round(mean(performance_metrics_rf$rmse, na.rm = TRUE), 2),
    Mean_MAPE = round(mean(performance_metrics_rf$mape, na.rm = TRUE), 2),
    Mean_R2   = round(mean(performance_metrics_rf$r2, na.rm = TRUE), 3)
  )
  
  # Resumo geral - PrevCarga
  summary_prevcarga <- data.table(
    Modelo = "PrevCarga",
    Mean_MAE  = round(mean(performance_metrics_prevcarga$mae, na.rm = TRUE), 2),
    Mean_RMSE = round(mean(performance_metrics_prevcarga$rmse, na.rm = TRUE), 2),
    Mean_MAPE = round(mean(performance_metrics_prevcarga$mape, na.rm = TRUE), 2),
    Mean_R2   = round(mean(performance_metrics_prevcarga$r2, na.rm = TRUE), 3)
  )
  
  # Tabela comparativa geral
  comparacao_geral <- rbind(summary_rf, summary_prevcarga)
  cat("\n==== COMPARAÇÃO GERAL DE PERFORMANCE - SUBSISTEMA SECO ====\n")
  print(comparacao_geral)
  
  # Performance por horizonte de previsão
  daily_performance_rf <- performance_metrics_rf[, .(
    MAE_RF = round(mean(mae, na.rm = TRUE), 2),
    RMSE_RF = round(mean(rmse, na.rm = TRUE), 2),
    MAPE_RF = round(mean(mape, na.rm = TRUE), 2),
    R2_RF = round(mean(r2, na.rm = TRUE), 3)
  ), by = day]
  
  daily_performance_prevcarga <- performance_metrics_prevcarga[, .(
    MAE_PrevCarga = round(mean(mae, na.rm = TRUE), 2),
    RMSE_PrevCarga = round(mean(rmse, na.rm = TRUE), 2),
    MAPE_PrevCarga = round(mean(mape, na.rm = TRUE), 2),
    R2_PrevCarga = round(mean(r2, na.rm = TRUE), 3)
  ), by = day]
  
  # Combinar métricas por dia
  comparacao_diaria <- merge(daily_performance_rf, daily_performance_prevcarga, by = "day")
  
  cat("\n==== COMPARAÇÃO POR HORIZONTE DE PREVISÃO (DIAS) ====\n")
  print(comparacao_diaria)
  
  ##----------------------------------------------------------------------------##
  ##------------------SALVAR RESULTADOS DO MÊS----------------------------------##
  ##----------------------------------------------------------------------------##
  
  cat("\n=== SALVANDO RESULTADOS PARA MÊS", m, "===\n")
  
  comparacao_geral[, mes_teste := m]
  comparacao_diaria[, mes_teste := m]
  
  # Salvar tabelas CSV
  fwrite(comparacao_geral, file.path(pasta_resultados, "comparacao_geral.csv"), sep = ";", dec = ",")
  fwrite(comparacao_diaria, file.path(pasta_resultados, "comparacao_diaria.csv"), sep = ";", dec = ",")
  
  cat("✅ Tabelas salvas em:", pasta_resultados, "\n")
  
  ##----------------------------------------------------------------------------##
  ##------------------VISUALIZAÇÕES COMPARATIVAS-------------------------------##
  ##----------------------------------------------------------------------------##

  create_comparison_plots <- function(y_true, y_pred_rf, y_pred_prevcarga, test_data, 
                                      target_cols, days_per_plot = 3, output_dir = pasta_resultados) {
    
    require(ggplot2)
    require(data.table)
    require(lubridate)
    
    if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)
    
    n_origins <- nrow(y_true)
    plot_data_list <- vector("list", n_origins)
    
    for (i in seq_len(n_origins)) {
      origin_date <- test_data$dataOrigem[i]
      forecast_hours <- seq_along(target_cols)
      timestamps <- origin_date + hours(forecast_hours)
      
      plot_data_list[[i]] <- data.table(
        forecast_origin = origin_date,
        hour = forecast_hours,
        timestamp = timestamps,
        day = ceiling(forecast_hours / 24),
        hour_of_day = ((forecast_hours - 1) %% 24) + 1,
        actual = as.numeric(y_true[i, ]),
        rf_predicted = as.numeric(y_pred_rf[i, ]),
        prevcarga_predicted = as.numeric(y_pred_prevcarga[i, ])
      )
    }
    
    plot_data <- rbindlist(plot_data_list)
    
    plot_data_long <- melt(
      plot_data,
      id.vars = c("forecast_origin", "hour", "timestamp", "day", "hour_of_day"),
      measure.vars = c("actual", "rf_predicted", "prevcarga_predicted"),
      variable.name = "type", value.name = "load"
    )
    
    n_days <- length(unique(plot_data_long$forecast_origin))
    n_plots <- ceiling(n_days / days_per_plot)
    
    plot_list <- list()
    
    for (k in seq_len(n_plots)) {
      start_idx <- (k - 1) * days_per_plot + 1
      end_idx <- min(k * days_per_plot, n_days)
      origins_subset <- unique(plot_data_long$forecast_origin)[start_idx:end_idx]
      subset_data <- plot_data_long[forecast_origin %in% origins_subset]
      
      p <- ggplot(subset_data, aes(x = timestamp, y = load, color = type)) +
        geom_line(size = 0.8, alpha = 0.9) +
        facet_wrap(~ paste("Origem:", forecast_origin), scales = "free_x", ncol = 1) +
        labs(
          title = sprintf("Comparação: Carga Real vs Random Forest vs PrevCarga\nMês de Teste: %s (Dias %d–%d)", m, start_idx, end_idx),
          x = "Data/Hora", y = "Carga (MW)", color = "Modelo"
        ) +
        scale_color_manual(
          values = c("actual" = "black", "rf_predicted" = "blue", "prevcarga_predicted" = "red"),
          labels = c("Real", "Random Forest", "PrevCarga")
        ) +
        theme_minimal() +
        theme(
          legend.position = "bottom",
          axis.text.x = element_text(angle = 45, hjust = 1)
        )
      
      filename <- sprintf("comparacao_dias_%d_%d_mes_%s.png", start_idx, end_idx, m)
      filepath <- file.path(output_dir, filename)
      
      ggsave(filepath, plot = p, width = 12, height = 8, dpi = 300)
      cat("✅ Gráfico salvo:", filename, "\n")
      
      plot_list[[k]] <- p
    }
    
    return(plot_list)
  }
  

  cat("\n=== GERANDO GRÁFICOS COMPARATIVOS ===\n")
  try({
    comparison_plots <- create_comparison_plots(
      y_true = test_actual_seco,
      y_pred_rf = rf_predictions_seco,
      y_pred_prevcarga = prevcarga_test_matrix,
      test_data = test_data_seco,
      target_cols = target_cols,
      days_per_plot = 3
    )
  }, silent = FALSE)
  
  # Armazenar resultados deste mês
  resultados_todos_meses[[m]] <- list(
    comparacao_geral = comparacao_geral,
    comparacao_diaria = comparacao_diaria,
    pasta_resultados = pasta_resultados
  )
  
  cat("\n✅ MÊS", m, "PROCESSADO COM SUCESSO!\n")
  cat("Resultados salvos em:", pasta_resultados, "\n")
  
  setwd('D:/OneDrive - Operador Nacional do Sistema Eletrico/_PEC_Energetico/Modelos/PrevCarga_Dmais/R')
}

##----------------------------------------------------------------------------##
##------------------CONSOLIDAÇÃO FINAL DE TODOS OS MESES----------------------##
##----------------------------------------------------------------------------##

cat("\n\n########################################\n")
cat("CONSOLIDANDO RESULTADOS DE TODOS OS MESES\n")
cat("########################################\n")

# Combinar todas as comparações gerais
todas_comparacoes_gerais <- rbindlist(lapply(resultados_todos_meses, function(x) x$comparacao_geral))

# Combinar todas as comparações diárias  
todas_comparacoes_diarias <- rbindlist(lapply(resultados_todos_meses, function(x) x$comparacao_diaria))

setwd('D:/OneDrive - Operador Nacional do Sistema Eletrico/_PEC_Energetico/Modelos/PrevCarga_Dmais')
fwrite(todas_comparacoes_gerais, "./resultados/consolidacao_geral_todos_meses.csv", sep = ";", dec = ",")
fwrite(todas_comparacoes_diarias, "./resultados/consolidacao_diaria_todos_meses.csv", sep = ";", dec = ",")

cat("\n==== RESUMO FINAL - TODOS OS MESES ====\n")
print(todas_comparacoes_gerais)

cat("\n########################################\n")
cat("EXECUÇÃO FINALIZADA COM SUCESSO!\n")
cat("Resultados processados para", length(resultados_todos_meses), "meses de teste\n")
cat("########################################\n")
