##----------------------------------------------------------------------------##
##---------------------------FUNÇÕES------------------------------------------##
################################################################################


############################################################################
#FUNÇÃO QUE OBTÉM A CARGA VERIFICADA DO SUBSISTEMA - COLOCA EM VALOR HORÁRIO
############################################################################

get_carga <- function(subsistema, DataIni, DataFin){
  # Obtém carga do subsistema
  dt_carga <- data.table(CGPeriodo(codArea = subsistema, DataIni, DataFin, flatted = T))
  dt_carga[, DataSemihora := with_tz(fastPOSIXct(din_referenciautc, "UTC"), "America/Sao_Paulo")]
  dt_carga$DataSemihora <- dt_carga$DataSemihora %m-% minutes(30)
  dt_carga[, c("cod_areacarga", "dat_referencia", "din_referenciautc", "val_cargammgd", 
               "val_consistencia", "val_cargasup", "val_cargansup", "val_cargaglobal", "val_cargarvd") := NULL]
  
  dt_carga[, hora := lubridate::hour(DataSemihora)]
  dt_carga[, dia := lubridate::date(DataSemihora)]
  
  # Calculate mean of val_cargaglobalcons by hora and dia
  dt_carga <- dt_carga[, .(mean_val_cargaglobalcons = mean(val_cargaglobalcons, na.rm = TRUE)), by = .(hora, dia)]
  return(dt_carga)
}

############################################################################
#FUNÇÃO QUE OBTÉM OS FERIADOS PARA O SUBSISTEMA - NÃO FUNCIONA APÓS 2023, ENTÃO USO O VETOR_FERIADOS
############################################################################

get_feriado <- function(subsistema, DataIni, vetor_feriados) {
  # Tenta obter os dias especiais
  Feriados_raw <- GetDiasEspeciaisAssociados(subsistema, anoInicial = year(DataIni))$results
  
  # Converte em data.table apenas se não for nulo/vazio
  if (is.null(Feriados_raw) || length(Feriados_raw) == 0) {
    return(unique(vetor_feriados))
  }
  
  Feriados <- data.table(Feriados_raw)
  
  # Se a tabela estiver vazia (sem linhas)
  if (nrow(Feriados) == 0) {
    return(unique(vetor_feriados))
  }
  
  # Processa normalmente
  Feriados[, Data := as.Date(dat_diaespecial)]
  Feriados <- unique(Feriados, by = "dat_diaespecial")
  
  vetor_feriados <- as.Date(vetor_feriados)
  Feriados$Data <- as.Date(Feriados$Data)
  
  aux_feriados <- c(vetor_feriados, Feriados$Data)
  
  return(unique(aux_feriados))
}

###################################################################################
#FUNÇÃO QUE OBTÉM A TEMPERATURA PREVISTA PARA OS PROXIMOS 10 DIAS PARA O SUBSISTEMA 
###################################################################################
get_temp_data <- function(cod_serie, n = 10, DataIni, DataFin) {
  
  datasorigem <- seq(as.Date(DataIni), as.Date(DataFin), by = "days")
  resultados <- list()
  
  for (d in seq_along(datasorigem)) {
    
    data_ref <- datasorigem[d]
    
    d_init <- data_ref + 1
    hora_ini <- paste0(d_init, " 00:00:00")
    d_fim <- data_ref + n - 1
    hora_fim <- paste0(d_fim, " 23:30:00")
    
    # Obtém os dados de temperatura previstos no dia dataOrigem para o horizonte 
    # entre hora_ini e hora_fim
    temp_raw <- TemperaturaPonderada(
      idSerieSPCA = cod_serie,
      dataHoraInicial = hora_ini,
      dataHoraFinal = hora_fim,
      prevista = TRUE,
      dataOrigem = data_ref
    )
    
    # loop para lidar com temperaturas faltando, copia o valor do dia anterior
    # Verificação: resultado vazio ou NULL
    if (is.null(temp_raw) || length(temp_raw) == 0) {
      message(sprintf("Temperatura vazia em %s, usando dia anterior.", data_ref))
      
      # se for o primeiro dia, cria linha com NA
      if (d == 1) {
        linha <- data.table(dataOrigem = data_ref)
      } else {
        linha <- copy(resultados[[d - 1]])
        linha[, dataOrigem := data_ref]
      }
      
      resultados[[d]] <- linha
      next
    }
    
    # Continua processamento normal
    temp_dt <- data.table(temp_raw)
    
    # Caso também esteja vazio depois de converter
    if (nrow(temp_dt) == 0) {
      message(sprintf("Sem linhas em Temperatura para %s, usando dia anterior.", data_ref))
      
      if (d == 1) {
        linha <- data.table(dataOrigem = data_ref)
      } else {
        linha <- copy(resultados[[d - 1]])
        linha[, dataOrigem := data_ref]
      }
      
      resultados[[d]] <- linha
      next
    }
    
    # Processamento normal
    temp_dt[, dia_previsto := as.Date(DinOcorrencia)]
    
    # Obtém os valores máximos e mínimos que serão os presentes para os dias futuros
    resumo <- temp_dt[, .(
      temp_max = max(ValItemserieoriginal, na.rm = TRUE),
      temp_min = min(ValItemserieoriginal, na.rm = TRUE)
    ), by = dia_previsto]
    
    resumo[, horizonte := as.integer(dia_previsto - data_ref)]
    resumo <- resumo[horizonte >= 1 & horizonte <= n - 1]
    
    linha <- dcast(
      resumo,
      formula = . ~ horizonte,
      value.var = c("temp_max", "temp_min")
    )
    
    linha[, . := NULL]  # remove a coluna "." de dcast
    linha[, dataOrigem := data_ref]
    resultados[[d]] <- linha
  }
  
  dt_final <- rbindlist(resultados, fill = TRUE)
  setcolorder(dt_final, 'dataOrigem')
  return(dt_final)
}

###################################################################################
#AJUSTA O DATA TABLE DE TEMPERATURA
###################################################################################
ajusta_dt_temp <- function(subsistema){
  # Recupera a tabela de temperatura
  nm <- paste0("dt_temp_", subsistema)
  if (!exists(nm, envir = .GlobalEnv)) {
    stop(sprintf("Objeto '%s' não encontrado no ambiente global.", nm))
  }
  
  dt <- copy(get(nm, envir = .GlobalEnv))
  # Verifica se há colunas esperadas
  col_max <- grep("^temp_max_\\d+$", names(dt), value = TRUE)
  col_min <- grep("^temp_min_\\d+$", names(dt), value = TRUE)
  if (length(col_max) == 0 || length(col_min) == 0) {
    stop("Colunas de temperatura máxima/mínima não encontradas no data.table.")
  }
  
  # Converte para formato longo, com uma coluna para a data de horigem, uma para o 
  # horizonte de previsão e duas para a temperatura maxima e minima prevista para aquele dia do horizonte
  wx_long <- melt(
    dt,
    id.vars = "dataOrigem",
    measure.vars = patterns(tmax = "^temp_max_\\d+$", tmin = "^temp_min_\\d+$"),
    variable.name = "h"
  )
  # Converte o horizonte de previsão para inteiro e cria data alvo
  wx_long[, day_of_horizon := as.integer(h)]
  wx_long[, target_date := dataOrigem + day_of_horizon]
  
  # Expande para 24 horas por dia de horizonte
  wx_hourly <- wx_long[, .(hora = 0:23, tmax = tmax, tmin = tmin),
                       by = .(dataOrigem, day_of_horizon, target_date)]
  
  # Cria timestamp alvo
  wx_hourly[, ts := as.POSIXct(sprintf("%s %02d:00:00", target_date, hora), tz = "UTC")]
  
  # Calcula lead_hour (1..n*24)
  wx_hourly[, lead_hour := (day_of_horizon - 1L) * 24L + (hora + 1L)]
  
  setkey(wx_hourly, ts)
  
  message(sprintf("Ajuste concluído para %s: %d linhas, %d horizontes.",
                  subsistema, nrow(wx_hourly), max(wx_hourly$day_of_horizon)))
  return(wx_hourly)
}
###################################################################################
# UNE OS DATA TABLES DE TEMPERATURA, CARGA E FERIADOS
###################################################################################
uniao_dts <- function(subsistema){
  # CRIA UM DATA TABLE COMPLETO PARA PREVISÃO, MAS AINDA NÃO ESTÁ NO FORMATO DE MULTIOUTPUT
  # Recupera a tabela de carga
  nm_load <- paste0("dt_load_", subsistema)
  if (!exists(nm_load, envir = .GlobalEnv)) {
    stop(sprintf("Objeto '%s' não encontrado no ambiente global.", nm_load))
  }
  dt_load <- copy(get(nm_load, envir = .GlobalEnv))
  # recupera a tabela de temperatura
  nm_temp <- paste0("dt_temp_ajustado_", subsistema)
  if (!exists(nm_temp, envir = .GlobalEnv)) {
    stop(sprintf("Objeto '%s' não encontrado no ambiente global.", nm_temp))
  }
  dt_temp <- copy(get(nm_temp, envir = .GlobalEnv))
  
  # Ajusta horário da carga
  dt_load[, ts := as.POSIXct(sprintf("%s %02d:00:00", dia, hora), tz = "UTC")]
  setkey(dt_load, ts)
  
  # União dos datatables
  train_x_y <- dt_temp[dt_load, on = .(ts), nomatch = 0L]
  
  setDT(train_x_y)
  
  train_x_y[, c("i.hora", "dia") := NULL]
  
  # # Adiciona features
  # train_x_y[, tmean := (tmax + tmin) / 2]
  # 
  # train_x_y[, `:=`(
  #   cdd20 = pmax(tmean - 20, 0),
  #   hdd18 = pmax(18 - tmean, 0)
  # )]
  # 
  # train_x_y[, `:=`(
  #   dow = lubridate::wday(ts, week_start = 1),  # 1=segunda..7=domingo
  #   hod = hour(ts),                              # hora do dia
  #   month = month(ts)
  # )]
  # 
  # train_x_y[, is_weekend := dow %in% c(6, 7)]
  
  cat("──────────────────────────────────────────────\n")
  cat(sprintf("Subsistema: %s\n", subsistema))
  cat("Dimensões da base final:", dim(train_x_y), "\n")
  cat("Datas de origem únicas:", length(unique(train_x_y$dataOrigem)), "\n")
  cat("Faixa de lead_hour:", paste(range(train_x_y$lead_hour, na.rm = TRUE), collapse = " – "), "\n")
  cat("Horizontes disponíveis:", paste(sort(unique(train_x_y$day_of_horizon)), collapse = ", "), "\n")
  cat("──────────────────────────────────────────────\n")
  
  return(train_x_y)
}

