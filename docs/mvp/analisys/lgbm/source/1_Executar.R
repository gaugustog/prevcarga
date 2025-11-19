library(data.table)
library(lubridate)

# Avaliacao
ANO <- 2024
MES <- 05

# Define o diretório de trabalho
setwd("D:/OneDrive - Operador Nacional do Sistema Eletrico/Arquivos Importantes/PrevCargaDESSEM_Aprimoramento/Modelo Consolidado/")

# Lista de áreas a serem processadas
#lista_areas <- c("MS","RS","PR","MG","S","SP","SECO","RJ","ES","SC","GO","MT","DF","AC","RO","N","NE","AM","PA","MA","TON","AP","BASE","ALPE","PBRN","BAOE","CE","PI","RR")
lista_areas <- c("SECO")

# Loop sobre cada área
for (Area in lista_areas) {
  cat("Iniciando processamento para a área:", Area, "\n")
  
  tryCatch({
    # === Limpa tudo do ambiente global, exceto objetos fixos do loop ===
    #rm(list = setdiff(ls(envir = .GlobalEnv), c("Area", "ANO", "MES", "lista_areas")), envir = .GlobalEnv)
    
    # Guardar valores essenciais
    Area_atual <- Area
    ANO_atual <- ANO
    MES_atual <- MES
    
    # Recriar variáveis no ambiente global antes de cada source
    assign("Area", Area_atual, envir = .GlobalEnv)
    assign("ANO", ANO_atual, envir = .GlobalEnv)
    assign("MES", MES_atual, envir = .GlobalEnv)
    
    # Calcular datas dinamicamente
    primeiro_dia_mes <- as.Date(paste(ANO_atual, MES_atual, "01", sep = "-"))
    data_corte_BLFvalidacao <- primeiro_dia_mes - 1
    data_corte_BLFtreino <- as.Date(paste(year(data_corte_BLFvalidacao), month(data_corte_BLFvalidacao), "01", sep = "-")) - 1
    
    assign("data_corte_BLFvalidacao", data_corte_BLFvalidacao, envir = .GlobalEnv)
    assign("data_corte_BLFtreino", data_corte_BLFtreino, envir = .GlobalEnv)
    
    # Executar os scripts
    setwd("D:/OneDrive - Operador Nacional do Sistema Eletrico/Arquivos Importantes/PrevCargaDESSEM_Aprimoramento/Modelo Consolidado/")
    source("ANN_PVDS.R")
    
    # Reatribuir após possível limpeza dentro dos scripts
    assign("Area", Area_atual, envir = .GlobalEnv)
    assign("ANO", ANO_atual, envir = .GlobalEnv)
    assign("MES", MES_atual, envir = .GlobalEnv)
    assign("data_corte_BLFvalidacao", data_corte_BLFvalidacao, envir = .GlobalEnv)
    assign("data_corte_BLFtreino", data_corte_BLFtreino, envir = .GlobalEnv)
    
    setwd("D:/OneDrive - Operador Nacional do Sistema Eletrico/Arquivos Importantes/PrevCargaDESSEM_Aprimoramento/Modelo Consolidado/")
    source("ModeloBLF_lgbm_v4.R")
    
    cat("Finalizado com sucesso para a área:", Area, "\n\n")
    
  }, error = function(e) {
    cat("Erro ao processar a área:", Area, "\n")
    cat("Mensagem de erro:", e$message, "\n\n")
  })
}
