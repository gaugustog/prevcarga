##----------------------------------------------------------------------------##
##-----------------FUNÇÕES PARA CONSTRUIR O DATASET PARA O MODELO-------------##
################################################################################

############################################################################
#FUNÇÃO QUE CRIA AS FEATURES PARA OS DIAS DE PREVISÃO E O DIA DE ORIGEM
############################################################################
multioutput_dt <- function(subsistema){
  # Recupera a tabela de carga
  nm_load <- paste0("dt_load_", subsistema)
  if (!exists(nm_load, envir = .GlobalEnv)) {
    stop(sprintf("Objeto '%s' não encontrado no ambiente global.", nm_load))
  }
  dt_load <- copy(get(nm_load, envir = .GlobalEnv))
  
  nm <- paste0("dt_", subsistema)
  if (!exists(nm, envir = .GlobalEnv)) {
    stop(sprintf("Objeto '%s' não encontrado no ambiente global.", nm))
  }
  dt <- copy(get(nm, envir = .GlobalEnv))
  
  nm_feriados <- paste0("dt_feriados_", subsistema)
  if (!exists(nm_feriados, envir = .GlobalEnv)) {
    stop(sprintf("Objeto '%s' não encontrado no ambiente global.", nm_feriados))
  }
  dt_feriados_obj <- copy(get(nm_feriados, envir = .GlobalEnv))
  
  multioutput_data <- dt[lead_hour >= 1 & lead_hour <= 216]
  
  # Cria a matriz de saídas desejadas para o treinamento, 216 saídas
  target_wide <- dcast(
    multioutput_data,
    dataOrigem ~ lead_hour,
    value.var = "mean_val_cargaglobalcons"
  )
  
  # Renomeia as colunas para y_h001, y_h002, etc.
  target_cols <- paste0("y_h", sprintf("%03d", 1:216))
  setnames(target_wide, as.character(1:216), target_cols)
  
  # Criação das matrizes de features (atributos)
  # Temperaturas
  temp_features <- dcast(
    multioutput_data[hora == 0], 
    dataOrigem ~ day_of_horizon,
    value.var = c("tmax", "tmin")
  )
  
  # Calendario
  calendar_features <- multioutput_data[lead_hour == 1, .(
    dataOrigem,
    origin_dow = wday(dataOrigem, week_start = 1),
    origin_month = month(dataOrigem),
    origin_is_weekend = wday(dataOrigem, week_start = 1) %in% c(6, 7),
    origin_is_feriado = fifelse(dataOrigem %in% dt_feriados_obj, TRUE, FALSE)
  )]
  
  
  # Features de dia da semana para as saídas (Vê cada dia do horizonte)
  dow_features <- multioutput_data[lead_hour == 1, .(dataOrigem)]  
  for (day_ahead in 1:9) {
    target_dates <- dow_features$dataOrigem + day_ahead
    target_dow <- wday(target_dates, week_start = 1)
    dow_features[, paste0("day", day_ahead, "_dow") := target_dow]
    dow_features[, paste0("day", day_ahead, "_is_weekend") := target_dow %in% c(6,7)]
    # dias one-hot
    dow_features[, paste0("day", day_ahead, "_is_monday") := target_dow == 1]
    dow_features[, paste0("day", day_ahead, "_is_tuesday") := target_dow == 2]
    dow_features[, paste0("day", day_ahead, "_is_wednesday") := target_dow == 3]
    dow_features[, paste0("day", day_ahead, "_is_thursday") := target_dow == 4]
    dow_features[, paste0("day", day_ahead, "_is_friday") := target_dow == 5]
    dow_features[, paste0("day", day_ahead, "_is_saturday") := target_dow == 6]
    dow_features[, paste0("day", day_ahead, "_is_sunday") := target_dow == 7]
    # feriado do target day
    dow_features[, paste0("day", day_ahead, "_is_feriado") := (target_dates %in% dt_feriados_obj)]
  }
  
  # Features de hora
  
  hour_features <- multioutput_data[lead_hour == 1, .(dataOrigem)]
  
  # Define os períodos criticos
  daily_peak_hours <- 9:18     
  evening_peak_hours <- 18:23   
  minimum_hours <- 0:8     
  
  for (day_ahead in 1:9) {
    for (hour in 0:23) {
      lead_hour_num <- (day_ahead - 1) * 24 + hour + 1
      col_name <- paste0("h", sprintf("%03d", lead_hour_num))
      
      # Calcula o mês do target day para interação
      target_date <- hour_features$dataOrigem + day_ahead
      target_month <- month(target_date)
      
      # Features binarias para os tipos de hora
      hour_features[, paste0(col_name, "_is_daily_peak") := hour %in% daily_peak_hours]
      hour_features[, paste0(col_name, "_is_evening_peak") := hour %in% evening_peak_hours]
      hour_features[, paste0(col_name, "_is_minimum") := hour %in% minimum_hours]
      hour_features[, paste0(col_name, "_hour_sin") := sin(2 * pi * hour / 24)]
      hour_features[, paste0(col_name, "_hour_cos") := cos(2 * pi * hour / 24)]
      
      # NOVA FEATURE: Interação Hour×Month para sazonalidade
      hour_features[, paste0(col_name, "_hour_month_interaction") := hour * target_month]
    }
  }
  
  
  # Features sasonais
  seasonal_features <- multioutput_data[lead_hour == 1, .(dataOrigem)]
  
  for (day_ahead in 1:9) {
    target_dates <- calendar_features$dataOrigem + day_ahead
    target_month <- month(target_dates)
    target_quarter <- ceiling(target_month / 3)  
    target_season <- ifelse(
      target_month %in% c(12, 1, 2), "verao",  
      ifelse(target_month %in% c(3, 4, 5), "outono",
             ifelse(target_month %in% c(6, 7, 8), "inverno", "primavera"))
    )
    
    seasonal_features[, paste0("day", day_ahead, "_month") := target_month]
    seasonal_features[, paste0("day", day_ahead, "_quarter") := target_quarter]
    seasonal_features[, paste0("day", day_ahead, "_is_summer") := target_season == "verao"]
    seasonal_features[, paste0("day", day_ahead, "_is_winter") := target_season == "inverno"]
  }
  
  
  # Features de Natal e Ano Novo
  cat("Criando features de Natal e Ano Novo...\n")
  christmas_newyear_features <- multioutput_data[lead_hour == 1, .(dataOrigem)]
  
  # Função auxiliar para verificar se uma data está no período de Natal ou Ano Novo
  is_christmas_period <- function(date) {
    month_day <- format(date, "%m-%d")
    year <- year(date)
    
    # Período de Natal: 23/12 a 29/12
    christmas_start <- sprintf("%d-12-23", year)
    christmas_end <- sprintf("%d-12-29", year)
    
    # Verifica se está no período de Natal do mesmo ano
    (date >= as.Date(christmas_start) & date <= as.Date(christmas_end))
  }
  
  is_newyear_period <- function(date) {
    month_day <- format(date, "%m-%d")
    year <- year(date)
    
    # Período de Ano Novo: 30/12 a 04/01
    # Precisa considerar a virada do ano
    dec_start <- sprintf("%d-12-30", year)
    dec_end <- sprintf("%d-12-31", year)
    jan_start <- sprintf("%d-01-01", year + 1)
    jan_end <- sprintf("%d-01-04", year + 1)
    
    # Também verifica período do ano anterior (30/12 do ano anterior a 04/01 do ano atual)
    prev_dec_start <- sprintf("%d-12-30", year - 1)
    prev_dec_end <- sprintf("%d-12-31", year - 1)
    curr_jan_start <- sprintf("%d-01-01", year)
    curr_jan_end <- sprintf("%d-01-04", year)
    
    # Verifica se está em qualquer um dos períodos de Ano Novo
    (date >= as.Date(dec_start) & date <= as.Date(dec_end)) |
    (date >= as.Date(jan_start) & date <= as.Date(jan_end)) |
    (date >= as.Date(prev_dec_start) & date <= as.Date(prev_dec_end)) |
    (date >= as.Date(curr_jan_start) & date <= as.Date(curr_jan_end))
  }
  
  # Features para data de origem
  christmas_newyear_features[, origin_is_christmas := is_christmas_period(dataOrigem)]
  christmas_newyear_features[, origin_is_newyear := is_newyear_period(dataOrigem)]
  
  # Features para cada dia do horizonte de previsão
  for (day_ahead in 1:9) {
    target_dates <- christmas_newyear_features$dataOrigem + day_ahead
    
    christmas_newyear_features[, paste0("day", day_ahead, "_is_christmas") := is_christmas_period(target_dates)]
    christmas_newyear_features[, paste0("day", day_ahead, "_is_newyear") := is_newyear_period(target_dates)]
  }
  
  
  # Features de feriados no horizonte de previsão
  cat("Criando features de proximidade de feriados...\n")
  holiday_features <- multioutput_data[lead_hour == 1, .(dataOrigem)]
  
  # Converte feriados para Date se necessário
  feriados_dates <- as.Date(dt_feriados_obj)
  
  for (i in 1:nrow(holiday_features)) {
    data_origem <- holiday_features$dataOrigem[i]
    
    forecast_dates <- data_origem + 1:9
    has_holiday_in_horizon <- any(forecast_dates %in% feriados_dates)
    
    future_holidays <- feriados_dates[feriados_dates > data_origem]
    days_to_next_holiday <- if (length(future_holidays) > 0) {
      as.numeric(min(future_holidays) - data_origem)
    } else 365
    
    past_holidays <- feriados_dates[feriados_dates < data_origem]
    days_since_last_holiday <- if (length(past_holidays) > 0) {
      as.numeric(data_origem - max(past_holidays))
    } else 365
    
    # Atualiza diretamente as colunas
    set(holiday_features, i, "has_holiday_in_horizon", has_holiday_in_horizon)
    set(holiday_features, i, "days_to_next_holiday", days_to_next_holiday)
    set(holiday_features, i, "days_since_last_holiday", days_since_last_holiday)
  }
  
  

  # Features do dia anterior
  cat("Criando features do dia anterior...\n")
  
  # Calcula estatísticas agregadas por dia
  daily_load_stats <- dt_load[, .(
    prev_day_mean = mean(mean_val_cargaglobalcons, na.rm = TRUE),
    prev_day_peak = max(mean_val_cargaglobalcons, na.rm = TRUE),
    prev_day_min = min(mean_val_cargaglobalcons, na.rm = TRUE),
    prev_day_daily_peak = fifelse(
      sum(hora %in% 9:17) > 0,
      max(mean_val_cargaglobalcons[hora %in% 9:17], na.rm = TRUE),
      NA_real_
    ),
    prev_day_evening_peak = fifelse(
      sum(hora %in% 18:23) > 0,
      max(mean_val_cargaglobalcons[hora %in% 18:23], na.rm = TRUE),
      NA_real_
    )
  ), by = dia]
  
  # Corrige Inf/-Inf
  cols_to_fix <- c("prev_day_peak", "prev_day_min", "prev_day_daily_peak", "prev_day_evening_peak")
  for (col in cols_to_fix) {
    daily_load_stats[is.infinite(get(col)), (col) := NA_real_]
  }
  
  # Adiciona informações de calendário do dia anterior
  daily_load_stats[, prev_day_dow := wday(dia, week_start = 1)]
  daily_load_stats[, prev_day_is_weekend := prev_day_dow %in% c(6, 7)]
  daily_load_stats[, prev_day_is_feriado := dia %in% dt_feriados_obj]
  
  # Cria tabela de features do dia anterior
  prev_day_features <- data.table(dataOrigem = unique(multioutput_data$dataOrigem))
  
  # Faz join para pegar dados do dia anterior (dataOrigem - 1)
  daily_load_stats[, join_date := dia + 1]  # dia + 1 = dataOrigem
  prev_day_features <- daily_load_stats[prev_day_features, on = .(join_date = dataOrigem)]
  
  # Remove colunas desnecessárias
  prev_day_features[, c("dia") := NULL]
  
  # Renomeia dataOrigem
  setnames(prev_day_features, "join_date", "dataOrigem", skip_absent = TRUE)
  
  # Features de dois dias antes (dataOrigem - 2)
  cat("Criando features de dois dias anteriores...\n")
  
  # Calcula estatísticas agregadas para dois dias antes
  daily_load_stats_2days <- dt_load[, .(
    prev_2day_mean = mean(mean_val_cargaglobalcons, na.rm = TRUE),
    prev_2day_peak = max(mean_val_cargaglobalcons, na.rm = TRUE),
    prev_2day_min = min(mean_val_cargaglobalcons, na.rm = TRUE),
    prev_2day_daily_peak = fifelse(
      sum(hora %in% 9:17) > 0,
      max(mean_val_cargaglobalcons[hora %in% 9:17], na.rm = TRUE),
      NA_real_
    ),
    prev_2day_evening_peak = fifelse(
      sum(hora %in% 18:23) > 0,
      max(mean_val_cargaglobalcons[hora %in% 18:23], na.rm = TRUE),
      NA_real_
    )
  ), by = dia]
  
  # Corrige Inf/-Inf para features de dois dias antes
  cols_to_fix_2days <- c("prev_2day_peak", "prev_2day_min", "prev_2day_daily_peak", "prev_2day_evening_peak")
  for (col in cols_to_fix_2days) {
    daily_load_stats_2days[is.infinite(get(col)), (col) := NA_real_]
  }
  
  # Adiciona informações de calendário de dois dias antes
  daily_load_stats_2days[, prev_2day_dow := wday(dia, week_start = 1)]
  daily_load_stats_2days[, prev_2day_is_weekend := prev_2day_dow %in% c(6, 7)]
  daily_load_stats_2days[, prev_2day_is_feriado := dia %in% dt_feriados_obj]
  
  # Cria tabela de features de dois dias antes
  prev_2day_features <- data.table(dataOrigem = unique(multioutput_data$dataOrigem))
  
  # Faz join para pegar dados de dois dias antes (dataOrigem - 2)
  daily_load_stats_2days[, join_date := dia + 2]  # dia + 2 = dataOrigem
  prev_2day_features <- daily_load_stats_2days[prev_2day_features, on = .(join_date = dataOrigem)]
  
  # Remove colunas desnecessárias
  prev_2day_features[, c("dia") := NULL]
  
  # Renomeia dataOrigem  
  setnames(prev_2day_features, "join_date", "dataOrigem", skip_absent = TRUE)

  
  # Features de nível de carga recente (7 dias, 3 dias e desvio)
  cat("Criando features de nível de carga recente...\n")
  
  recent_level_features <- data.table(dataOrigem = unique(multioutput_data$dataOrigem))
  
  # Calcular estatísticas de carga de 7 dias anteriores (t-7 a t-1)
  daily_load_stats_7days <- dt_load[, .(
    daily_mean = mean(mean_val_cargaglobalcons, na.rm = TRUE)
  ), by = dia]
  
  for (i in seq_len(nrow(recent_level_features))) {
    data_origem <- recent_level_features$dataOrigem[i]
    
    # 7-day average load level (business cycle context)
    # Período: t-7 a t-1
    period_7day_start <- data_origem - 7
    period_7day_end <- data_origem - 1
    
    load_7days <- daily_load_stats_7days[dia >= period_7day_start & dia <= period_7day_end, daily_mean]
    
    if (length(load_7days) > 0) {
      prev_7day_mean <- mean(load_7days, na.rm = TRUE)
    } else {
      prev_7day_mean <- NA_real_
    }
    
    # 3-day average load level (recent trend)  
    # Período: t-3 a t-1
    period_3day_start <- data_origem - 3
    period_3day_end <- data_origem - 1
    
    load_3days <- daily_load_stats_7days[dia >= period_3day_start & dia <= period_3day_end, daily_mean]
    
    if (length(load_3days) > 0) {
      prev_3day_mean <- mean(load_3days, na.rm = TRUE)
    } else {
      prev_3day_mean <- NA_real_
    }
    
    # Recent vs historical deviation (adaptive baseline)
    if (!is.na(prev_3day_mean) && !is.na(prev_7day_mean) && prev_7day_mean != 0) {
      recent_vs_historical <- prev_3day_mean / prev_7day_mean
    } else {
      recent_vs_historical <- NA_real_
    }
    
    # Atualizar features usando set()
    set(recent_level_features, i, "prev_7day_mean", prev_7day_mean)
    set(recent_level_features, i, "prev_3day_mean", prev_3day_mean)
    set(recent_level_features, i, "recent_vs_historical", recent_vs_historical)
  }
  
  cat("Features de nível de carga recente criadas para", nrow(recent_level_features), "origens de previsão\n")

  
  # Combina as features
  features_wide <- Reduce(function(x,y) merge(x,y, by = "dataOrigem", all = TRUE),
                          list(temp_features, calendar_features, dow_features, hour_features, 
                               seasonal_features, christmas_newyear_features, 
                               holiday_features, prev_day_features, prev_2day_features, 
                               recent_level_features))
  
  
  # Junta com os targets e coloca o data set final
  multioutput_final <- merge(features_wide, target_wide, by = "dataOrigem")
  
  
  cat("Multi-output dataset dimensions:", dim(multioutput_final), "\n")
  cat("Number of forecast origins:", nrow(multioutput_final), "\n")
  cat("Number of features:", max(0, ncol(features_wide)-1), "\n")
  cat("Number of targets:", length(target_cols), "\n")
  
  return(multioutput_final)
  
}


############################################################################
#FUNÇÃO QUE DIVIDE O DATASET EM TREINO E TESTE PARA O MODELO
############################################################################
split_train_test <- function(subsistema){
  nm <- paste0("dt_features_", subsistema)
  if (!exists(nm, envir = .GlobalEnv)) {
    stop(sprintf("Objeto '%s' não encontrado no ambiente global.", nm))
  }
  dt_features <- copy(get(nm, envir = .GlobalEnv))
  
  setorder(dt_features, dataOrigem)
  
  dt_features <- na.omit(dt_features)
  
  # Identifica targets
  target_cols <- grep("^y_h", names(dt_features), value = TRUE)
  
  # Identifica features
  feature_cols <- setdiff(names(dt_features), c("dataOrigem", target_cols))
  
  # Separação
  n_total <- nrow(dt_features)
  n_train <- floor(n_total - 30)
  
  train_data <- dt_features[1:n_train]
  test_data  <- dt_features[(n_train + 1):n_total]
  

  cat("Período de treino:", as.character(range(train_data$dataOrigem)), "\n")
  cat("Período de teste: ", as.character(range(test_data$dataOrigem)), "\n")
  cat("Nº de features:", length(feature_cols), "\n")
  cat("Nº de targets:", length(target_cols), "\n")
  
  return(list(
    train = train_data,
    test  = test_data,
    feature_cols = feature_cols,
    target_cols  = target_cols
  ))
}