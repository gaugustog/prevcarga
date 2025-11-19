library(data.table)
library(wavelets)
library(zoo)

# --------------------------------------------------------------------------------------------------

w2dt <- function(wt) {
  ws <- wt@W
  names <- lapply(names(ws), function(s) paste0(s, "_", seq_len(nrow(ws[[s]]))))
  ws <- lapply(ws, t)
  #ws <- lapply(ws, as.data.table)
  ws <- do.call(cbind, ws)
  colnames(ws) <- unlist(names)
  return(ws)
}

v2dt <- function(wt) {
  vs <- wt@V
  names <- lapply(names(vs), function(s) paste0(s, "_", seq_len(nrow(vs[[s]]))))
  vs <- lapply(vs, t)
  #vs <- lapply(vs, as.data.table)
  vs <- do.call(cbind, vs)
  colnames(vs) <- unlist(names)
  return(vs)
}

build_wavelets_single_window <- function(v, filter = "haar") {
  wt <- dwt(v, filter = filter)
  ws <- w2dt(wt)
  vs <- v2dt(wt)
  wavs <- cbind(ws, vs)
  return(wavs)
}

build_wavelets <- function(ts, window = 2^5, filter = "haar") {
  out <- zoo::rollapply(ts, width = window, build_wavelets_single_window,
                        na.pad = TRUE, align = "right")
  out <- as.data.table(out)
  return(out)
}

mape <- function(Real, Previsto) {
  # Calcular o MAPE apenas para esses casos
  return(mean(abs((Real - Previsto) / Real)) * 100)
}

# Definir funções de normalização e desnormalização
normalizar_BLF <- function(x, min_val, max_val) {
  return((x - min_val) / (max_val - min_val))
}
denormalizar_BLF <- function(x, min_val, max_val) {
  return(x * (max_val - min_val) + min_val)
}
