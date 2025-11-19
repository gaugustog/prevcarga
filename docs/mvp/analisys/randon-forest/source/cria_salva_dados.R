##----------------------------------------------------------------------------##
##---------------------------EXECUÇÃO DO PROGRAMA-----------------------------##
################################################################################
setwd('D:/OneDrive - Operador Nacional do Sistema Eletrico/_PEC_Energetico/Modelos/PrevCarga_Dmais/R')
source('./0_config.R')
source('./1_funcoes_obtencao_dados.R')

DataIni <- '2022-05-01'
#DataIni <- '2025-10-01'
DataFin <- '2025-10-10'


# Criar pasta para resultados deste mês
caminho_entrada <- file.path("D:/OneDrive - Operador Nacional do Sistema Eletrico/_PEC_Energetico/Modelos/PrevCarga_Dmais/dados_entrada")
if (!dir.exists(caminho_entrada)) {
  dir.create(caminho_entrada, recursive = TRUE)
}



##----------------------------------------------------------------------------##
##---------------------------IMPORTA DADOS------------------------------------##
##----------------------------------------------------------------------------##

cat("\n=== IMPORTANDO DADOS ===\n")

# Importa dados de carga
for (s in names(subsistemas)){
  assign(paste0("dt_load_", s), get_carga(s, DataIni, DataFin))
  fwrite(get(paste0("dt_load_", s)), file.path(caminho_entrada, paste0("dt_load_", s, '.csv')), sep = ";", dec = ",")
}

# Importa dados de temperatura
for (s in names(subsistemas)){
  assign(paste0("dt_temp_", s), get_temp_data(subsistemas[[s]],n = 10, DataIni, DataFin))
  fwrite(get(paste0("dt_temp_", s)), file.path(caminho_entrada, paste0("dt_temp_", s, '.csv')), sep = ";", dec = ",")
  
}
