library(data.table)
library(LoadServices)
library(lubridate)
library(fasttime)
library(caret)
library(Metrics)
library(foreach)
library(doParallel)
library(nnet)
library(progressr)
library(wavelets)
library(zoo)
library(lightgbm)
library(randomForest)

AmbienteAtivo("PRD")
set.seed(123) # Para reprodutibilidade
#### INFO ####
InicioBase <- "2022-10-01"
Hoje <- with_tz(Sys.time(), tzone = "America/Sao_Paulo")
FimBase <- as.Date(Hoje - days(1))
DataHoraIni <- paste0(InicioBase," ","00:30:00")
DataHoraFin <- paste0(FimBase," ","00:00:00")
# Codigos
listaarea <- GetAreasCarga()
  #Area <- listaarea$results$cod_areacarga[23]
#Area <- "SP"
listaspca <- SerieSPCA()
Indice <- data.table(
  AREA = c("MS","RS","PR","MG","S","SP","SECO","RJ","ES","SC","GO","MT","DF","AC","RO","N","NE","AM","PA","MA","TON","AP","BASE","ALPE","PBRN","BAOE","CE","PI","RR"),
  SPCA = c(4,1,3,26,10,5,11,6,27,2,28,29,12,14,15,8,9,13,36,37,40,42,48,49,50,51,52,53,55),
  Heatindex = c("MS","RS","PR","MG","RS","SP","SP","RJ","ES","SC","GO","MT","DF","AC","RO","AM","CE","AM","PA","MA","TO","AP","BASE","ALPE","PBRN","BAOE","CE","PI","RR")
)
spca <- Indice[AREA == Area, SPCA]
heat <- Indice[AREA == Area, Heatindex]

#####  Funcao
obtemDadosSeTiver <- function(cod_areacarga,datIni,datFim){
  try(CGPeriodo(cod_areacarga,datIni,datFim,T))
}
#####

# ------------------------------------------------------------------------
# Carga
Carga <- lapply(Area, obtemDadosSeTiver, InicioBase, FimBase)
Carga <- rbindlist(Carga)
Carga$DataHora <- with_tz(fastPOSIXct(Carga$din_referenciautc, "UTC"), "America/Sao_Paulo")
Carga$DataHora <- (Carga$DataHora %m-% minutes(30))
Carga[, Data := as.Date(DataHora, tz = "America/Sao_Paulo")]
Carga[, DiaSemana := weekdays(DataHora)]

# ------------------------------------------------------------------------
# Temperatura e Heat Index
heatindex <- fread(paste0("D:/OneDrive - Operador Nacional do Sistema Eletrico/Arquivos Importantes/PrevCargaDESSEM_Aprimoramento/heatindex/heatindex_", heat, ".csv"))
setnames(heatindex, c("Area", "DataHora", "Temperatura_ECMWF", "parametro", "Heatindex"))
heatindex[, DataHora := with_tz(DataHora, tzone = "America/Sao_Paulo")]

TemperaturaD1 <- data.table(TemperaturaPrevistaDelay(spca, DataHoraIni, DataHoraFin, delay = 0))
TemperaturaD1[, DataHora := DinOcorrencia]
setorder(TemperaturaD1, DataHora)

TemperaturaD2 <- data.table(TemperaturaPrevistaDelay(spca, DataHoraIni, DataHoraFin, delay = 1))
TemperaturaD2[, DataHora := DinOcorrencia]
setorder(TemperaturaD2, DataHora)

# Preencher séries faltantes da temperatura
frequencia <- "1 hour"
datas_completas <- data.table(DataHora = seq(min(TemperaturaD1$DataHora), max(TemperaturaD1$DataHora), by = frequencia))
TemperaturaD1 <- merge(datas_completas, TemperaturaD1, by = "DataHora", all.x = TRUE)
lag_n <- if (frequencia == "30 mins") 48 else 24
TemperaturaD1[, ValItemserieoriginal := fifelse(is.na(ValItemserieoriginal),
                                                shift(ValItemserieoriginal, n = lag_n, type = "lag"),
                                                ValItemserieoriginal)]
TemperaturaD1[, DinOrigem := fifelse(is.na(DinOrigem), DataHora - days(1), DinOrigem)]
TemperaturaD1[, DinOcorrencia := fifelse(is.na(DinOcorrencia), DataHora, DinOcorrencia)]
TemperaturaD1[, IdSeriehistorica := fifelse(is.na(IdSeriehistorica), 5L, IdSeriehistorica)]
TemperaturaD1 <- TemperaturaD1[, .(IdSeriehistorica, DinOrigem, DinOcorrencia, ValItemserieoriginal, DataHora)]

datas_completas <- data.table(DataHora = seq(min(TemperaturaD2$DataHora), max(TemperaturaD2$DataHora), by = frequencia))
TemperaturaD2 <- merge(datas_completas, TemperaturaD2, by = "DataHora", all.x = TRUE)
lag_n <- if (frequencia == "30 mins") 48 else 24
TemperaturaD2[, ValItemserieoriginal := fifelse(is.na(ValItemserieoriginal),
                                                shift(ValItemserieoriginal, n = lag_n, type = "lag"),
                                                ValItemserieoriginal)]
TemperaturaD2[, DinOrigem := fifelse(is.na(DinOrigem), DataHora - days(1), DinOrigem)]
TemperaturaD2[, DinOcorrencia := fifelse(is.na(DinOcorrencia), DataHora, DinOcorrencia)]
TemperaturaD2[, IdSeriehistorica := fifelse(is.na(IdSeriehistorica), 5L, IdSeriehistorica)]
TemperaturaD2 <- TemperaturaD2[, .(IdSeriehistorica, DinOrigem, DinOcorrencia, ValItemserieoriginal, DataHora)]
# ------------------------------------------------------------------------
# Feriados
Feriados <- data.table(GetDiasEspeciaisAssociados(Area, anoInicial = year(DataHoraIni))$results)
if (nrow(Feriados) == 0) {
  Feriados <- data.table(GetDiasEspeciaisAssociados("AM", anoInicial = year(DataHoraIni))$results)
}
Feriados[, Data := as.Date(dat_diaespecial)]
Feriados <- unique(Feriados, by = "dat_diaespecial")

# ------------------------------------------------------------------------
# Integrar Carga, Temperatura e Feriado
Total <- Feriados[Carga, on = .(Data = Data)]
Total[is.na(id_tipodiaespecial), id_tipodiaespecial := 0]
Total[, DataHora_chr := as.character(DataHora)]

TemperaturaD1[, DataHora_chr := as.character(DataHora)]
TemperaturaD2[, DataHora_chr := as.character(DataHora)]
heatindex[, DataHora_chr := as.character(DataHora)]

TemperaturaD1[, DataHora := NULL]
TemperaturaD2[, DataHora := NULL]
heatindex[, DataHora := NULL]

Total <- merge(Total, TemperaturaD1, by = "DataHora_chr", all.x = TRUE, all.y = TRUE)
Total <- merge(Total, TemperaturaD2, by = "DataHora_chr", all.x = TRUE, all.y = TRUE)
Total <- merge(Total, heatindex,      by = "DataHora_chr", all.x = TRUE, all.y = TRUE)


# Corrigir tipos e preenchimentos
Total[, DataHora_chr := ifelse(nchar(as.character(DataHora_chr)) == 10,
                               paste0(as.character(DataHora_chr), " 00:00:00"),
                               as.character(DataHora_chr))]
Total[, DataHora := as.POSIXct(DataHora_chr, format = "%Y-%m-%d %H:%M:%S", tz = "America/Sao_Paulo")]
Total[, DataHora_chr := NULL]
# ------------------------------------------------------------------------
# Preencher valores faltantes para Temperatura, ECMWF e HeatIndex
# TemperaturaD1
Total[, ValItemserieoriginal_prev := nafill(ValItemserieoriginal.x, type = "locf"), by = Data]
Total[, ValItemserieoriginal_next := nafill(ValItemserieoriginal.x, type = "nocb"), by = Data]
Total[is.na(ValItemserieoriginal.x) & 
        !is.na(ValItemserieoriginal_prev) & 
        !is.na(ValItemserieoriginal_next), 
      ValItemserieoriginal.x := (ValItemserieoriginal_prev + ValItemserieoriginal_next) / 2, by = Data]
Total[, c("ValItemserieoriginal_prev", "ValItemserieoriginal_next") := NULL, by = Data]
Total[, ValItemserieoriginal.x := nafill(ValItemserieoriginal.x, type = "locf"), by = Data]
# TemperaturaD2
Total[, ValItemserieoriginal_prev := nafill(ValItemserieoriginal.y, type = "locf"), by = Data]
Total[, ValItemserieoriginal_next := nafill(ValItemserieoriginal.y, type = "nocb"), by = Data]
Total[is.na(ValItemserieoriginal.y) & 
        !is.na(ValItemserieoriginal_prev) & 
        !is.na(ValItemserieoriginal_next), 
      ValItemserieoriginal.y := (ValItemserieoriginal_prev + ValItemserieoriginal_next) / 2, by = Data]
Total[, c("ValItemserieoriginal_prev", "ValItemserieoriginal_next") := NULL, by = Data]
Total[, ValItemserieoriginal.y := nafill(ValItemserieoriginal.y, type = "locf"), by = Data]
# ECMWF
Total[, Temperatura_ECMWF_prev := nafill(Temperatura_ECMWF, type = "locf"), by = Data]
Total[, Temperatura_ECMWF_next := nafill(Temperatura_ECMWF, type = "nocb"), by = Data]
Total[is.na(Temperatura_ECMWF) & 
        !is.na(Temperatura_ECMWF_prev) & 
        !is.na(Temperatura_ECMWF_next), 
      Temperatura_ECMW := (Temperatura_ECMWF_prev + Temperatura_ECMWF_next) / 2, by = Data]
Total[, c("Temperatura_ECMWF_prev", "Temperatura_ECMWF_next") := NULL, by = Data]
Total[, Temperatura_ECMWF := nafill(Temperatura_ECMWF, type = "locf"), by = Data]
# Heatindex
Total[, Heatindex_prev := nafill(Heatindex, type = "locf"), by = Data]
Total[, Heatindex_next := nafill(Heatindex, type = "nocb"), by = Data]
Total[is.na(Heatindex) & 
        !is.na(Heatindex_prev) & 
        !is.na(Heatindex_next), 
      Heatindex := (Heatindex_prev + Heatindex_next) / 2, by = Data]
Total[, c("Heatindex_prev", "Heatindex_next") := NULL, by = Data]
Total[, Heatindex := nafill(Heatindex, type = "locf"), by = Data]

# Selecionar variáveis
Total <- Total[, .(DataHora, Data, val_cargaglobalcons, val_cargammgd, ValItemserieoriginal.x,ValItemserieoriginal.y, id_tipodiaespecial, Temperatura_ECMWF, Heatindex)]
setnames(Total, c("DataHora", "Data", "CargaGlobal", "GerMMGD", "TemperaturaD1","TemperaturaD2", "Feriado", "Temperatura_ECMWF", "Heatindex"))

# ------------------------------------------------------------------------
# Criar variáveis adicionais
Total[, `:=`(
  DiaSemana = weekdays(DataHora),
  Hora = hour(DataHora) + minute(DataHora) / 60,
  Minuto = minute(DataHora),
  DiaAno = yday(DataHora),
  SemanaAno = week(DataHora),
  Mes = month(DataHora),
  Ano = year(DataHora),
  FimDeSemana = as.integer(weekdays(DataHora) %in% c("sábado", "domingo"))
)]
Total[, `:=`(
  Hora_seno = sin(2 * pi * Hora / 24),
  Hora_cosseno = cos(2 * pi * Hora / 24)
)]

# Codificar Dia da Semana (cíclico)
mapeamento_dias <- list("domingo" = 1, "segunda-feira" = 3, "terça-feira" = 4, "quarta-feira" = 5, "quinta-feira" = 6, "sexta-feira" = 7, "sábado" = 2)
Total[, DiaSemanaNum := as.numeric(unlist(mapeamento_dias[DiaSemana]))]
Total[, MesNum := month(DataHora)]
Total[, DiaAno := yday(DataHora)]

# Marcar feriados nacionais
feriados_nacionais <- as.Date(c(
  "2021-01-01", "2021-04-02", "2021-04-21", "2021-05-01", "2021-09-07", "2021-10-12",
  "2021-11-02", "2021-11-15", "2021-12-25", "2022-01-01", "2022-04-15", "2022-04-21",
  "2022-05-01", "2022-09-07", "2022-10-12", "2022-11-02", "2022-11-15", "2022-12-25",
  "2023-01-01", "2023-04-07", "2023-04-21", "2023-05-01", "2023-09-07", "2023-10-12",
  "2023-11-02", "2023-11-15", "2023-12-25", "2024-01-01", "2024-03-29", "2024-04-21",
  "2024-05-01", "2024-09-07", "2024-10-12", "2024-11-02", "2024-11-15", "2024-12-25",
  "2025-01-01", "2025-04-18", "2025-04-21", "2025-05-01", "2025-09-07", "2025-10-12",
  "2025-11-02", "2025-11-15", "2025-12-25", "2024-11-20", "2025-11-20"
))
Total[Data %in% feriados_nacionais, Feriado := 10]
Total[Feriado >= 1, Feriado := 1]

# Criar dummies de dia da semana
dias_unicos <- unique(Total$DiaSemana)
dias_unicos <- dias_unicos[-2]
for (dia in dias_unicos) {
  nome_coluna <- paste0("DiaSemana_", dia)
  Total[, (nome_coluna) := as.integer(DiaSemana == dia)]
}
dias_dummy <- grep("^DiaSemana_", names(Total), value = TRUE)

# Criar dummies de mês
mes_unicos <- unique(Total$MesNum)
mes_unicos <- mes_unicos[-2]
for (mes in mes_unicos) {
  nome_coluna <- paste0("Mes_", mes)
  Total[, (nome_coluna) := as.integer(MesNum == mes)]
}
mes_dummy <- grep("^Mes_", names(Total), value = TRUE)

# Criar dummies de feriados adjacentes
Data_feriado <- unique(Total[Feriado == 1, Data])
Total[, PreFeriado := as.integer(Data %in% (Data_feriado - 1))]
Total[, PosFeriado := as.integer(Data %in% (Data_feriado + 1))]

# Converter para fatores
Total[, Feriado := as.factor(Feriado)]
Total[, PreFeriado := as.factor(PreFeriado)]
Total[, PosFeriado := as.factor(PosFeriado)]
Total[, Mes := as.factor(Mes)]
Total[, DiaSemanaNum := as.factor(DiaSemanaNum)]
Total[, (dias_dummy) := lapply(.SD, as.factor), .SDcols = dias_dummy]
Total[, (mes_dummy) := lapply(.SD, as.factor), .SDcols = mes_dummy]

# Ordenar dados
setorder(Total, Data, Hora, Minuto)
Total <- na.omit(Total)
gc()

# Criar variável auxiliar para hora contínua
Total[, HoraDecimal := Hora + Minuto / 60]

# Aplicar LOESS por dia para cada variável
Total[, TemperaturaD1amort := predict(
  loess(TemperaturaD1 ~ HoraDecimal, span = 0.3, degree = 2)
), by = Data]

Total[, TemperaturaD2amort := predict(
  loess(TemperaturaD2 ~ HoraDecimal, span = 0.3, degree = 2)
), by = Data]

Total[, CargaGlobalamort := predict(
  loess(CargaGlobal ~ HoraDecimal, span = 0.3, degree = 2)
), by = Data]

Total[, Temperatura_ECMWFamort := predict(
  loess(Temperatura_ECMWF ~ HoraDecimal, span = 0.3, degree = 2)
), by = Data]

# Ajustando heatindex

# Calcular a diferença individual
# Total[, Heatindexdif := Heatindex - Temperatura_ECMWF]
# 
# # Calcular a média da diferença por mês, hora e minuto (30 min)
# medias_mensais_30min <- Total[, .(
#   Heatindexdif_Mensal30min = mean(Heatindexdif, na.rm = TRUE)
# ), by = .(Ano, Mes, Hora, Minuto)]
# 
# # Juntar essa média de volta à tabela Total
# Total <- merge(
#   Total,
#   medias_mensais_30min,
#   by = c("Ano", "Mes", "Hora", "Minuto"),
#   all.x = TRUE
# )

# ==========================================================================================
# 1. Identificar datas especiais de Natal e Ano Novo
# ==========================================================================================

# Criar coluna 'DiaEspecial' vazia
Total[, DiaEspecial := NA_character_]

# Marcar datas específicas
Total[format(Data, "%m-%d") %in% c("12-25", "01-01"), DiaEspecial := "Natal_AnoNovo"]              # Natal e Ano Novo
Total[format(Data, "%m-%d") %in% c("12-24", "12-31"), DiaEspecial := "Vespera"]                    # Vésperas de Natal e Ano Novo
Total[format(Data, "%m-%d") %in% c("12-23", "12-30"), DiaEspecial := "PreVespera"]                 # Pré-vésperas
Total[format(Data, "%m-%d") %in% c("12-26", "01-02"), DiaEspecial := "PosNatalAnoNovo"]            # Pós Natal e Pós Ano Novo
Total[format(Data, "%m-%d") %in% sprintf("12-%02d", 27:30), DiaEspecial := "SemanaNatalAnoNovo"]   # Semana Natal e Ano Novo
Total[format(Data, "%m-%d") %in% sprintf("01-%02d", 03:07), DiaEspecial := "PrimeiraSemanaAno"]    # Primeira Semana do Ano

# ==========================================================================================
# 2. Criar variáveis dummy para cada DiaEspecial
# ==========================================================================================

# Pegar valores únicos de DiaEspecial, ignorando NAs
dias_especiais_unicos <- unique(na.omit(Total$DiaEspecial))

# Criar uma coluna dummy para cada valor único de DiaEspecial
for (dia in dias_especiais_unicos) {
  nome_coluna <- paste0("DiaEspecial_", dia)
  Total[, (nome_coluna) := as.integer(DiaEspecial == dia)]
}

# Garantir que todos os NAs nas dummies sejam zeros
dias_especiais_dummy <- grep("^DiaEspecial_", names(Total), value = TRUE)
for (col in dias_especiais_dummy) {
  Total[is.na(get(col)), (col) := 0L]
}

# Ajustar coluna DiaEspecial (transformar NAs em 0)
Total[is.na(DiaEspecial), DiaEspecial := 0]

# ==========================================================================================
# 3. Criar variável de Tipo de Dia (útil, sábado, domingo)
# ==========================================================================================

Total[, TipoDia := fcase(
  DiaSemana == "segunda-feira", "segunda-feira",
  DiaSemana == "sexta-feira",   "sexta-feira",
  DiaSemana == "sábado",        "sábado",
  DiaSemana == "domingo",       "domingo",
  default = "Dia útil intermediário"
)]

Total[, TipoDia := factor(TipoDia, levels = c(
  "Dia útil intermediário", "segunda-feira", "sexta-feira", "sábado", "domingo"
))]

# ==========================================================================================
# 4. Sazonalidade Diária: Suavização da Carga por Tipo de Dia, Feriado e Dia Especial
# ==========================================================================================

# 4.1 Calcular a média horária da carga, considerando TipoDia, Hora, Minuto, Feriado e DiaEspecial
sazonalidade_diaria_mes <- Total[, .(
  CargaGlobal_media = mean(CargaGlobal, na.rm = TRUE)
), by = .(TipoDia, Hora, Minuto, Feriado, DiaEspecial)]

# 4.2 Criar variável contínua de horário (ex: 18h30 = 18.5)
sazonalidade_diaria_mes[, Hora_Minuto := Hora + Minuto / 60]

# 4.3 Suavizar a curva com LOESS por TipoDia, Feriado e DiaEspecial
sazonalidade_diaria_mes[, Sazonalidade := predict(
  loess(CargaGlobal_media ~ Hora, span = 0.2, degree = 2)
), by = .(TipoDia, Feriado, DiaEspecial)]

# ==========================================================================================
# 5. Merge da sazonalidade suavizada com a base Total
# ==========================================================================================

Total <- merge(
  Total,
  sazonalidade_diaria_mes[, .(TipoDia, Hora, Minuto, Feriado, DiaEspecial, Sazonalidade)],
  by = c("TipoDia", "Hora", "Minuto", "Feriado", "DiaEspecial"),
  all.x = TRUE
)

# ==========================================================================================
# 6. Limpeza Final
# ==========================================================================================

# Remover coluna 'DiaEspecial' que não é mais necessária
#Total[, DiaEspecial := NULL]

# Ordenar base por Data, Hora e Minuto
setorder(Total, Data, Hora, Minuto)

