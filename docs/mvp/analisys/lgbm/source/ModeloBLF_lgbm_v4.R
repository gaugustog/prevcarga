set.seed(123) # Para reprodutibilidade dos resultados

# 1. Criar as colunas de lag (valores do mesmo horário do dia seguinte) para CargaGlobal e Temperatura
lag_BLF <- Total[, .(
  Data_lag_BLF = Data + days(+1),              # Data ajustada para o dia seguinte
  Hora,                                         # Hora original
  CargaGlobal_lag_BLF = CargaGlobalamort,            # CargaGlobal original como lag
  TemperaturaECMWF_lag_BLF = Temperatura_ECMWFamort, # Temperatura ECMWF como lag
  Heatindex_lag_BLF = Heatindex                 # Heatindex como lag
)]

# 2. Realizar o merge para alinhar cada registro com os valores do dia anterior na mesma hora
Total_BLF <- merge(
  Total,                          # Tabela original
  lag_BLF,                         # Tabela com os valores lag
  by.x = c("Data", "Hora"),         # Chaves de junção na tabela original
  by.y = c("Data_lag_BLF", "Hora"), # Chaves de junção na tabela de lag
  all.x = TRUE,                     # Mantém todos os registros da tabela original (LEFT JOIN)
  suffixes = c("", "_BLF")          # Sufixos para evitar conflitos de nome
)

# 3. Organizar base com renomeação e limpeza
Total_BLF[, CargaGlobal_anterior_BLF := CargaGlobal_lag_BLF]                # Renomeia a coluna de lag da carga
Total_BLF[, TemperaturaECMWF_anterior_BLF := TemperaturaECMWF_lag_BLF]      # Renomeia a coluna de lag da temperatura ECMWF
Total_BLF[, Heatindex_anterior_BLF := Heatindex_lag_BLF]                    # Renomeia a coluna de lag do Heatindex
Total_BLF[, c("CargaGlobal_lag_BLF", "TemperaturaECMWF_lag_BLF", "Heatindex_lag_BLF") := NULL] # Remove colunas intermediárias de lag

# 4. Eliminar registros com NA nos lags
Total_BLF <- Total_BLF[!is.na(CargaGlobal_anterior_BLF)]

# 5. Deixar somente dias completos para verificação de MAPE
contagem_dias_BLF <- Total_BLF[, .N, by = Data] # Contar registros por dia
dias_completos_BLF <- contagem_dias_BLF[N == 48, Data] # Selecionar dias com 48 registros
Total_completos_BLF <- Total_BLF[Data %in% dias_completos_BLF] # Filtrar dias completos

# 6. Organizar valores previstos
Total_completos_BLF[, Temperatura_prevista_BLF := TemperaturaD1amort]      # Criar 'Temperatura_prevista_BLF'
Total_completos_BLF[, CargaGlobal_real_BLF := CargaGlobal]      # Criar 'CargaGlobal_real_BLF'
Total_completos_BLF[, `:=`(TemperaturaD1 = NULL, CargaGlobal = NULL)]  # Remover colunas originais
Total_completos_BLF <- Total_completos_BLF[!is.na(Heatindex_anterior_BLF)]

# 7. Calcula wavelet
setwd("D:/OneDrive - Operador Nacional do Sistema Eletrico/Arquivos Importantes/PrevCargaDESSEM_Aprimoramento/")
source("funcoes_aux.R")
#carga_wavelets_BLF <- build_wavelets(Total_completos_BLF$CargaGlobal_real_BLF) # Wavelet para carga
carga_wavelets_BLF <- build_wavelets(Total_completos_BLF$CargaGlobal_anterior_BLF) # Wavelet para carga
setnames(carga_wavelets_BLF, old = names(carga_wavelets_BLF), new = paste0("CarWav_", seq_along(carga_wavelets_BLF)))
carga_wavelets_BLF[, DataHora := Total_completos_BLF$DataHora + 60 * 60 * 48]
Total_completos_BLF <- merge(Total_completos_BLF, carga_wavelets_BLF, by = "DataHora", all.x = TRUE)
carga_wav <- setdiff(colnames(carga_wavelets_BLF),"DataHora")
gc()
temperatura_wavelets_BLF <- build_wavelets(Total_completos_BLF$Temperatura_prevista_BLF) # Wavelet para carga
setnames(temperatura_wavelets_BLF, old = names(temperatura_wavelets_BLF), new = paste0("TempWav_", seq_along(temperatura_wavelets_BLF)))
temperatura_wavelets_BLF[, DataHora := Total_completos_BLF$DataHora]
Total_completos_BLF <- merge(Total_completos_BLF, temperatura_wavelets_BLF, by = "DataHora", all.x = TRUE)
temp_wav <- setdiff(colnames(temperatura_wavelets_BLF),"DataHora")
gc()

# 8. Definir as colunas a serem usadas como entradas, incluindo dummy variables de 'Feriado'
colunas_entrada_BLF <- c(
  "CargaGlobal_anterior_BLF",    # Carga do dia anterior no mesmo horário
  "TemperaturaECMWF_anterior_BLF",    # Temperatura do dia anterior no mesmo horário
  #"Heatindex_anterior_BLF",
  "Temperatura_prevista_BLF",    # Temperatura prevista (auxiliar)
  "GerMMGD",
  "Feriado",
  "PreFeriado",
  "PosFeriado",
  "Sazonalidade",
  "Hora_seno",
  carga_wav,
  temp_wav,
  dias_dummy,
  dias_especiais_dummy,
  mes_dummy
)

# 9. Normalização dos dados
colunas_nao_normalizar <- c("Feriado","PreFeriado","PosFeriado","Mes",dias_dummy,mes_dummy,dias_especiais_dummy)
colunas_normalizar_BLF <- setdiff(
  c(colunas_entrada_BLF, "CargaGlobal_real_BLF"), 
  colunas_nao_normalizar
)

# Verificar se alguma coluna está armazenada como lista e converter para numérico
cols_as_list_BLF <- colunas_normalizar_BLF[
  sapply(Total_completos_BLF[, ..colunas_normalizar_BLF], is.list) # Verificar se há colunas armazenadas como lista
]
if (length(cols_as_list_BLF) > 0) {  # Converter listas para numérico
  for (col in cols_as_list_BLF) {
    Total_completos_BLF[, (col) := as.numeric(unlist(get(col)))]
  }
} else {print("OK")}

# Definir funções de normalização e desnormalização
normalizar_BLF <- function(x, min_val, max_val) {
  return((x - min_val) / (max_val - min_val))
}

denormalizar_BLF <- function(x, min_val, max_val) {
  return(x * (max_val - min_val) + min_val)
}

# Armazenar min e max para cada coluna
min_max_BLF <- lapply(
  Total_completos_BLF[, ..colunas_normalizar_BLF],
  function(x) {
    c(min = min(x, na.rm = TRUE), max = max(x, na.rm = TRUE))
  }
)
names(min_max_BLF) <- colunas_normalizar_BLF  # Nomear os elementos da lista

# Aplicar normalização
for (i in 1:length(colunas_normalizar_BLF)) {
  col <- colunas_normalizar_BLF[i]
  min_val <- min_max_BLF[[col]]["min"]
  max_val <- min_max_BLF[[col]]["max"]
  Total_completos_BLF[, (col) := normalizar_BLF(get(col), min_val, max_val)]
}

# 8. Dividir os dados de treino e teste até 30/09/2024
primeiro_dia_mes <- as.Date(paste(ANO, MES, "01", sep = "-"))
#data_corte_BLFtreino <- as.Date("2024-08-31") # Data de corte para treinamento
#data_corte_BLFvalidacao <- as.Date("2024-09-30") # Data de corte para treinamento
data_corte_BLFvalidacao <- primeiro_dia_mes - 1 # Data de corte para treinamento
data_corte_BLFtreino <- as.Date(paste(year(data_corte_BLFvalidacao), month(data_corte_BLFvalidacao), "01", sep = "-")) - 1 # Data de corte para treinamento
data_corte_BLFteste <- data_corte_BLFvalidacao # Data de corte para treinamento
dados_treino_BLF <- na.omit(Total_completos_BLF[Data <= data_corte_BLFtreino])       # Dados até 30/09/2024 para treinamento
dados_validacao_BLF <- na.omit(Total_completos_BLF[Data > data_corte_BLFtreino & Data <= data_corte_BLFvalidacao])
dados_teste_BLF <- na.omit(Total_completos_BLF[Data >= data_corte_BLFvalidacao & Data < Sys.Date()])

# 7. Separar as entradas e saídas para cada conjunto
entradas_treino_BLF <- dados_treino_BLF[, ..colunas_entrada_BLF]       # Entradas para treinamento
saidas_treino_BLF <- dados_treino_BLF$CargaGlobal_real_BLF      # Saídas para treinamento
entradas_validacao_BLF <- dados_validacao_BLF[, ..colunas_entrada_BLF]       # Entradas para treinamento
saidas_validacao_BLF <- dados_validacao_BLF$CargaGlobal_real_BLF      # Saídas para treinamento
entradas_teste_BLF <- dados_teste_BLF[, ..colunas_entrada_BLF]       # Entradas para treinamento
saidas_teste_BLF <- dados_teste_BLF$CargaGlobal_real_BLF      # Saídas para treinamento

# 1. Definir as variáveis numéricas e categóricas (mantendo os nomes existentes)
coluna_numerica_BLF <- setdiff(names(entradas_treino_BLF), colunas_nao_normalizar)
coluna_categorica_BLF <- setdiff(names(entradas_treino_BLF), colunas_normalizar_BLF)
parametros_BLF <- list(
  categorical_features = match(coluna_categorica_BLF, colnames(entradas_treino_BLF)),
  feature_pre_filter = FALSE
)

# 2. Calcular o vetor de penalidade com base na coluna 'Hora'
#    Para observações com Hora > 18, aplica um fator de penalidade (por exemplo, 2); caso contrário, 1.
fator_penalidade <- 1
vetor_penalidade_treino <- ifelse(hour(dados_treino_BLF$DataHora) > 18, fator_penalidade, 1)
vetor_penalidade_validacao <- ifelse(hour(dados_validacao_BLF$DataHora) > 18, fator_penalidade, 1)

# 3. Criar os objetos lgb.Dataset, passando o vetor de penalidade como weight no conjunto de treino
treina_dadoslgbm_BLF <- lgb.Dataset(
  data = data.matrix(entradas_treino_BLF[, .SD, .SDcols = c(coluna_numerica_BLF, coluna_categorica_BLF)]),
  label = saidas_treino_BLF,
  weight = vetor_penalidade_treino,  # Utiliza o vetor de penalidade calculado para treino
  params = parametros_BLF
)

validacao_dadoslgbm_BLF <- lgb.Dataset(
  data = data.matrix(entradas_validacao_BLF[, .SD, .SDcols = c(coluna_numerica_BLF, coluna_categorica_BLF)]),
  label = saidas_validacao_BLF,
  params = parametros_BLF
)

# 4. Definir a função objetivo customizada que penaliza subestimação (quando previsão < valor real)
rotulos_treino_global <- saidas_treino_BLF

objetivo_customizado <- function(preds, dtrain) {
  
  labels <- rotulos_treino_global
  pesos <- attr(dtrain, "weight")
  if (is.null(pesos)) {
    pesos <- rep(1, length(labels))
  }
    res <- preds - labels
  grad <- ifelse((res < 0) & (pesos > 1), pesos * res, res)
  hess <- ifelse((res < 0) & (pesos > 1), pesos, 1)

  return(list(grad = grad, hess = hess))
}

# 4. Definir uma função de avaliação customizada que calcula o weighted MAPE
avaliacao_customizada <- function(preds, dtrain) {
  # Tenta recuperar os rótulos a partir do objeto dtrain; se não houver, usa o conjunto global de validação
  labels <- attr(dtrain, "label")
  if (is.null(labels)) {
    labels <- saidas_validacao_BLF
  }
  # Utiliza o vetor de penalidade para validação
  pesos_validacao <- vetor_penalidade_validacao
  # Calcula o erro percentual absoluto para cada observação
  errors <- abs(preds - labels) / pmax(abs(labels), 1e-6)
  # Nova fórmula: soma dos erros ponderados dividida pelo número total de observações
  weighted_mape <- (sum(pesos_validacao * errors) / length(errors)) * 100
  return(list(name = "weighted_mape_new", value = weighted_mape, higher_better = FALSE))
}


# 5. Definir a grade de hiperparâmetros
grid_params <- expand.grid(
  max_leaves = c(16),
  min_data_in_leaf = c(50),
  learning_rate = c(0.1),
  num_iterations = c(5000),
  stringsAsFactors = FALSE
)

resultados <- list()
melhor_mape <- Inf
melhor_modelo <- NULL

#Aplica modelo

for (i in seq_len(nrow(grid_params))) {
  parametros_iteracao <- list(
    objective = objetivo_customizado,    # Utiliza funcao customizada
    #alpha = 0.5,               # Define o quantil; com 0.9, penaliza mais a subestimação
    metric = "mape",           # Métrica padrão para referência (o early stopping usa esta métrica)
    max_leaves = grid_params$max_leaves[i],
    min_data_in_leaf = grid_params$min_data_in_leaf[i],
    learning_rate = grid_params$learning_rate[i],
    num_iterations = grid_params$num_iterations[i],
    early_stopping_rounds = 50
  )
  
  modelo <- lgb.train(
    parametros_iteracao,
    data = treina_dadoslgbm_BLF,
    valids = list(valid = validacao_dadoslgbm_BLF),
    eval = avaliacao_customizada,  # Continua utilizando a avaliação customizada (weighted MAPE)
    
    verbose = -1
  )
  
  # Obtém o weighted MAPE mínimo registrado no conjunto de validação
  mape <- min(unlist(modelo$record_evals$valid$weighted_mape_new$eval))
  
  # Calcula o weighted MAPE manualmente para o conjunto de validação
  previsoes <- predict(modelo, data.matrix(entradas_validacao_BLF))
  reais <- saidas_validacao_BLF
  pesos_validacao <- vetor_penalidade_validacao
  mape_ajustado <- sum(pesos_validacao * abs(previsoes - reais) / pmax(abs(reais), 1e-6)) / sum(pesos_validacao) * 100
  
  resultados[[i]] <- list(parametros = parametros_iteracao, mape = mape, mape_ajustado = mape_ajustado)
  
  if (mape_ajustado < melhor_mape) {
    melhor_mape <- mape_ajustado
    melhor_modelo <- modelo
    melhor_parametros <- parametros_iteracao  # Salva a configuração que gerou o melhor resultado
  }
}

cat("\n===== Melhor Configuração Encontrada =====\n")
print(melhor_parametros)
cat("Weighted MAPE Ajustado do Melhor Modelo:", melhor_mape, "\n")
iteracoes <- unlist(melhor_modelo$record_evals$valid$weighted_mape_new$eval)
plot(iteracoes, type = "o")


##################################################

primeiro_dia_mes <- as.Date(paste(ANO, MES, "01", sep = "-"))
ultimo_dia_mes_anterior <- primeiro_dia_mes - 1

ANO_ANT <- year(ultimo_dia_mes_anterior)
MES_ANT <- month(ultimo_dia_mes_anterior)

teste <- dados_teste_BLF[
  (Ano == ANO & Mes == MES) |
    (Ano == ANO_ANT & Mes == MES_ANT & Data == ultimo_dia_mes_anterior)]

Base <- Total_BLF[
  (Ano == ANO & Mes == MES) |
    (Ano == ANO_ANT & Mes == MES_ANT & Data == ultimo_dia_mes_anterior)]

resultados_teste <- list()
  
  # Garantir que datas_teste está definida corretamente
  datas_teste <- unique(teste$Data)
  
  for (i in seq_along(datas_teste)) {
    # 1. Previsão inicial do dia atual completo usando o dia anterior
    entradas_dia_atual <- copy(teste[Data == datas_teste[i], ..colunas_entrada_BLF])
    
    previsao_dia_atual <- predict(
      melhor_modelo,
      data.matrix(entradas_dia_atual[, c(coluna_numerica_BLF, coluna_categorica_BLF), with = FALSE])
    )
    
    # 2. Calcular as diferenças horárias percentuais (h2 / h1)
    diferencas_horarias <- c(1, round(previsao_dia_atual[-1] / previsao_dia_atual[-length(previsao_dia_atual)], 7))
    
    # 3. Aplicar as diferenças a partir dos valores verificados até 8h
    valores_verificados <- teste[hour(DataHora) <= 8 & Data == datas_teste[i], .(DataHora, CargaGlobal_real_BLF)]
    
    valores_completos <- copy(valores_verificados) # Usar uma cópia para evitar modificar o original
    
    for (j in seq_len(length(previsao_dia_atual))[-seq_len(nrow(valores_verificados))]) {
      proximo_valor <- valores_completos$CargaGlobal_real_BLF[nrow(valores_completos)] * diferencas_horarias[j]
      valores_completos <- rbind(
        valores_completos,
        data.table(
          DataHora = teste[Data == datas_teste[i] & Hora > 8]$DataHora[j - nrow(valores_verificados)],
          CargaGlobal_real_BLF = proximo_valor
        )
      )
    }
    
    # 4. Preparar as entradas para prever o dia seguinte
    entradas_dia_seguinte <- copy(teste[Data == datas_teste[i] + 1, ..colunas_entrada_BLF])
    TemperaturaD2 <- Base[Data == datas_teste[i] + 1,]
    
    # Aplicar as diferenças temperatura a partir dos valores verificados até 8h
    diferencas_horarias_temp <- c(1, round(TemperaturaD2$TemperaturaD1[-1] / TemperaturaD2$TemperaturaD1[-length(TemperaturaD2$TemperaturaD1)], 7))
    valores_verificados_temp <- teste[hour(DataHora) <= 8 & Data == datas_teste[i], .(DataHora, TemperaturaECMWF_anterior_BLF)]
    valores_completos_temp <- copy(valores_verificados_temp) # Usar uma cópia para evitar modificar o original
    for (j in seq_len(length(TemperaturaD2$TemperaturaD1))[-seq_len(nrow(valores_verificados_temp))]) {
      proximo_valor <- valores_completos_temp$TemperaturaECMWF_anterior_BLF[nrow(valores_completos_temp)] * diferencas_horarias_temp[j]
      valores_completos_temp <- rbind(
        valores_completos_temp,
        data.table(
          DataHora = teste[Data == datas_teste[i] & Hora > 8]$DataHora[j - nrow(valores_verificados_temp)],
          TemperaturaECMWF_anterior_BLF = proximo_valor
        )
      )
    }

    if (nrow(entradas_dia_seguinte) > 0) {
      entradas_dia_seguinte[, `:=`(
        CargaGlobal_anterior_BLF = valores_completos$CargaGlobal_real_BLF,
        TemperaturaECMWF_anterior_BLF = valores_completos_temp$TemperaturaECMWF_anterior_BLF,
        Temperatura_prevista_BLF = TemperaturaD2$TemperaturaD2amort
      )]
      
      # 5. Prever o dia completo de amanhã
      previsao_dia_seguinte <- predict(
        melhor_modelo,
        data.matrix(entradas_dia_seguinte[, c(coluna_numerica_BLF, coluna_categorica_BLF), with = FALSE])
      )
      
      # 6. Salvar o resultado final
      resultado_previsao <- data.table(
        DataHora = teste[Data == datas_teste[i] + 1]$DataHora,
        CargaGlobal_prevista = previsao_dia_seguinte
      )
      resultado_previsao[, CargaGlobalSuavizada := loess(CargaGlobal_prevista ~ as.numeric(DataHora),data = .SD, span = 0.1)$fitted]
      
      # Adicionar aos resultados
      resultados_teste[[as.character(datas_teste[i])]] <- resultado_previsao
    } else {
      warning(paste("Entradas para o dia seguinte estão vazias na data:", datas_teste[i]))
    }
  }
  
  # Consolidar resultados em um único data.table
  resultado_final_teste <- rbindlist(resultados_teste, fill = TRUE)

#########################################################################
# 
# dados_filtrados <- Total_BLF[Data == "2024-10-28", .(Temperatura)]
# 
# # Plota um gráfico de linhas da temperatura nesse dia
# plot(
#   x = seq_along(dados_filtrados$Temperatura),
#   y = dados_filtrados$Temperatura,
#   type = "l",                        # "l" para plotar linhas
#   col = "blue",                      # cor da linha
#   main = "Temperatura em 2024-10-28",
#   xlab = "Índice de Observação", 
#   ylab = "Temperatura"
# )
# ggplot(resultado_previsao, aes(x = DataHora)) +
#   geom_line(aes(y = CargaGlobal_prevista, color = "CargaGlobal Prevista")) +
#   geom_line(aes(y = CargaGlobalSuavizada, color = "CargaGlobal Suavizada")) +
#   labs(
#     title = "Comparação entre CargaGlobal Prevista e Suavizada",
#     x = "Data e Hora",
#     y = "Carga Global",
#     color = "Legenda"
#   ) +
#   theme_minimal() +
#   theme(axis.text.x = element_text(angle = 45, hjust = 1)) # Rotacionar rótulos do eixo X para melhor visualização
#########################################################################


AvaliacaoFinal <- merge(
  resultado_final_teste,
  Total[, .(DataHora, CargaGlobal)],
  by = "DataHora",
  all.x = TRUE
)
AvaliacaoFinal[, Data := as.Date(DataHora, tz = "America/Sao_Paulo")] # Cria Data

# Desnormalizar os valores reais
#AvaliacaoFinal[, CargaGlobal_real_BLF_denorm := denormalizar_BLF(
#  CargaGlobal_real_BLF, 
#  min_val = min_max_BLF$CargaGlobal_real_BLF["min"], 
#  max_val = min_max_BLF$CargaGlobal_real_BLF["max"]
#)]

# Desnormalizar os valores previstos
AvaliacaoFinal[, Previsto_denorm := denormalizar_BLF(
  CargaGlobalSuavizada, 
  min_val = min_max_BLF$CargaGlobal_real_BLF["min"], 
  max_val = min_max_BLF$CargaGlobal_real_BLF["max"]
)]

AvaliacaoFinal[, CargaGlobalSuavizada := {
  ajuste <- loess(Previsto_denorm ~ as.numeric(DataHora), data = .SD, span = 0.15)
  ajuste$fitted
}, by = Data]

#AvaliacaoFinal[, PrevisaoCargaGlobal := Previsto_denorm]
AvaliacaoFinal[, Previsto_denorm := CargaGlobalSuavizada]

avaliacao <- AvaliacaoFinal[(year(DataHora) == ANO) & (month(DataHora) == MES)]
avaliacao[, desvio := CargaGlobal - Previsto_denorm]
#avaliacao <- avaliacao[!weekdays(Data) %in% c("sábado", "domingo")]
#dplot <- dplot[Feriado == 0 & !(DiaSemana %in% c("sábado","domingo"))]
PontaNoturna <- avaliacao[hour(DataHora) > 18 & hour(DataHora) < 22]


mape <- function(Real, Previsto) {
  # Calcular o MAPE apenas para esses casos
  return(mean(abs((Real - Previsto) / Real)) * 100)
}

# Calcular o MAPE utilizando os valores desnormalizados
mape_diatodo <- mape(
  Real = avaliacao$CargaGlobal, 
  Previsto = avaliacao$Previsto_denorm
)

mape_noite <- mape(
  Real = PontaNoturna$CargaGlobal, 
  Previsto = PontaNoturna$Previsto_denorm
)

# Viés da noite
viés_medio_ponta <- mean(PontaNoturna$Previsto_denorm - PontaNoturna$CargaGlobal, na.rm = TRUE)

desvio_medio_por_dia <- PontaNoturna[, .(
  DesvioMedio_MW = mean(CargaGlobal - Previsto_denorm, na.rm = TRUE)
), by = as.Date(DataHora)]

dias_subestimativa <- desvio_medio_por_dia[DesvioMedio_MW < 0, .N]
subestimativa_maxima <- max(PontaNoturna[Previsto_denorm < CargaGlobal, abs(CargaGlobal - Previsto_denorm)], na.rm = TRUE)

#dplot <- dplot[Feriado == 0 & !(DiaSemana %in% c("sábado","domingo"))]

# Exibir o resultado
print(paste("MAPE:", round(mape_diatodo, 2), "%"))
print(paste("MAPE Noite:", round(mape_noite, 2), "%"))
print(paste("Maior Desvio Ponta:", max(round(PontaNoturna$desvio, 2))))
print(paste("Viés médio no horário crítico (MW):", round(viés_medio_ponta, 2)))
print(paste("Número de dias com subestimativa:", dias_subestimativa))
print(paste("Subestimativa máxima no horário crítico (MW):", round(subestimativa_maxima, 2)))

titulo_mape <- paste0(
  ANO, "/", MES,
  " | MAPE: ", round(mape_diatodo, 2), "%",
  " | MAPE Noite: ", round(mape_noite, 2), "%"
)

grafico <- ggplot() +
  geom_line(data = avaliacao, aes(DataHora, CargaGlobal, color = "Verificado")) +
  geom_line(data = avaliacao, aes(DataHora, Previsto_denorm, color = "Previsto")) +
  scale_x_datetime(date_labels = "%H:%M") +
  facet_wrap(~ Data, scales = "free_x") +
  labs(
    title = paste0(ANO, "/", MES),
    subtitle = paste0("MAPE: ", round(mape_diatodo, 2), "% | MAPE Noite: ", round(mape_noite, 2), "%")
  ) +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 20, hjust = 1, vjust = 1))
grafico

ggsave(filename = paste0("Resultados/",Area,"_",ANO, MES, "_v3.png"), plot = grafico, width = 10, height = 8, dpi = 600)
fwrite(avaliacao,paste0("D:/OneDrive - Operador Nacional do Sistema Eletrico/Arquivos Importantes/PrevCargaDESSEM_Aprimoramento/Resultados/",Area,"_",ANO,MES,"_v3.csv"), dateTimeAs = "write.csv")


#rm(list = setdiff(ls(), c("ANO","MES","lista_areas","Area")))