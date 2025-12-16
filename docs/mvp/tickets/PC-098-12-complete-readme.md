# PC-098-12: Complete README

**Epic:** [EPIC-12: Documentation & Deploy](../epics/EPIC-12-documentation-deploy.md)
**Task Reference:** T-12.1
**Priority:** High
**Estimated Effort:** 1 day

---

## Summary

Create a comprehensive README.md with project overview, installation instructions, quick start guide, configuration examples, CLI command reference, plugin development overview, and contributing guidelines.

---

## Acceptance Criteria

- [ ] Project overview with description and badges
- [ ] Installation instructions for all methods
- [ ] Quick start guide with examples
- [ ] Configuration reference
- [ ] CLI command reference
- [ ] Plugin development overview
- [ ] Contributing guidelines
- [ ] License and credits

---

## Technical Specification

### README Structure

```markdown
# PrevCargaONS

[![CI](https://github.com/ons-br/prevcarga-R/actions/workflows/ci.yml/badge.svg)](https://github.com/ons-br/prevcarga-R/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/ons-br/prevcarga-R/branch/main/graph/badge.svg)](https://codecov.io/gh/ons-br/prevcarga-R)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Sistema de Previsão de Carga Elétrica para o Operador Nacional do Sistema Elétrico (ONS).

## 📋 Índice

- [Sobre](#sobre)
- [Instalação](#instalação)
- [Quick Start](#quick-start)
- [Configuração](#configuração)
- [Comandos CLI](#comandos-cli)
- [Desenvolvimento de Plugins](#desenvolvimento-de-plugins)
- [Contribuindo](#contribuindo)
- [Licença](#licença)

## Sobre

PrevCargaONS é um sistema modular de previsão de carga elétrica desenvolvido em R,
projetado para:

- Previsão de demanda para múltiplos horizontes (D+0 a D+8)
- Suporte a múltiplas áreas e subsistemas (SECO, S, NE, N)
- Arquitetura de plugins para modelos, combinadores e reconciliadores
- Reconciliação hierárquica para consistência entre níveis
- Interface CLI completa para operação e automação

### Arquitetura

```
┌─────────────────────────────────────────────────────────────┐
│                        CLI Layer                            │
├─────────────────────────────────────────────────────────────┤
│                     Orchestrator                            │
├──────────┬──────────┬──────────┬──────────┬────────────────┤
│  Data    │ Feature  │  Model   │ Combine  │  Reconcile     │
│  Layer   │  Layer   │  Layer   │  Layer   │  Layer         │
└──────────┴──────────┴──────────┴──────────┴────────────────┘
```

## Instalação

### Método 1: One-Line Installer (Linux/macOS)

```bash
curl -fsSL https://ons.org.br/prevcarga/install.sh | bash
```

### Método 2: Docker

```bash
docker pull ons/prevcarga:latest
docker run -it ons/prevcarga interactive
```

### Método 3: WSL (Windows)

```powershell
# Instalar WSL
wsl --install -d Ubuntu-22.04

# Dentro do WSL
curl -fsSL https://ons.org.br/prevcarga/install.sh | bash
```

### Método 4: Manual

```bash
# Clonar repositório
git clone https://github.com/ons-br/prevcarga-R.git
cd prevcarga-R

# Instalar dependências
R -e "renv::restore()"

# Instalar pacote
R CMD INSTALL .
```

### Requisitos

- R >= 4.3.0
- Sistema operacional: Linux (Ubuntu 20.04+, Debian 11+, Fedora 38+), macOS 12+, ou Windows com WSL2

## Quick Start

### Modo Interativo

```bash
prevcarga
# ou
prevcarga interactive
```

### Previsão Rápida

```bash
# Previsão para área RJ, horizonte D+1
prevcarga predict --area RJ --horizon 1

# Previsão para múltiplas áreas
prevcarga predict --areas RJ,SP,MG --horizons 0,1,2

# Previsão para subsistema
prevcarga predict --subsystem SECO --horizons 0:8
```

### Treinamento de Modelo

```bash
# Treinar modelo LightGBM para área RJ
prevcarga train --model lgbm --area RJ --output models/

# Treinar todos os modelos habilitados
prevcarga train --all-models --areas RJ,SP,MG
```

### Backtest

```bash
# Executar backtest de 1 ano
prevcarga backtest \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --areas RJ,SP,MG \
  --output reports/backtest_2024.html
```

## Configuração

A configuração é feita via arquivo YAML em `~/.prevcarga/config/config.yaml`:

```yaml
# Configuração de projeto
project:
  name: "PrevCargaONS"
  environment: "production"
  seed: 42

# Armazenamento
storage:
  backend: s3  # ou 'local'
  s3:
    bucket: ons-prevcarga-prod
    prefix: data/
    region: sa-east-1

# Modelos
models:
  enabled:
    - lgbm
    - rf
    - hw
  default_horizons: [0, 1, 2, 3, 4, 5, 6, 7, 8]

# Reconciliação
reconciliation:
  default: bottom_up
  include_losses: true

# Logging
logging:
  level: INFO
  format: json
```

Para mais detalhes, consulte a [documentação de configuração](docs/user-guide/configuration.md).

## Comandos CLI

| Comando | Descrição |
|---------|-----------|
| `prevcarga train` | Treinar modelos |
| `prevcarga predict` | Gerar previsões |
| `prevcarga backtest` | Executar backtest |
| `prevcarga eval-model` | Avaliar modelo |
| `prevcarga gen-features` | Gerar features |
| `prevcarga combine` | Combinar previsões |
| `prevcarga reconcile` | Reconciliar hierarquia |
| `prevcarga report` | Gerar relatórios |
| `prevcarga interactive` | Modo interativo |
| `prevcarga status` | Status do sistema |
| `prevcarga --help` | Ajuda |

### Exemplos

```bash
# Treinar com configuração customizada
prevcarga train \
  --model lgbm \
  --config custom_config.yaml \
  --areas RJ,SP \
  --train-start 2023-01-01 \
  --train-end 2023-12-31

# Previsão em modo batch
prevcarga predict \
  --mode batch \
  --date 2025-01-15 \
  --areas all \
  --output predictions/

# Gerar relatório interativo
prevcarga report forecast \
  --date 2025-01-15 \
  --areas RJ,SP \
  --output reports/forecast.html \
  --open
```

## Desenvolvimento de Plugins

PrevCargaONS suporta extensão via plugins para:

- **Modelos** (`BaseModel`)
- **Features** (`BaseFeaturePlugin`)
- **Combinadores** (`BaseCombiner`)
- **Reconciliadores** (`BaseReconciler`)

### Exemplo: Plugin de Modelo

```r
#' @title MyModel
#' @description Custom model implementation
MyModel <- R6::R6Class(
  "MyModel",
  inherit = BaseModel,

  public = list(
    initialize = function(config = list()) {
      super$initialize(
        model_id = "my_model",
        model_type = "custom",
        config = config
      )
    },

    fit = function(X, y, ...) {
      # Implementar treinamento
      private$.fitted <- TRUE
      invisible(self)
    },

    predict = function(X, ...) {
      # Implementar predição
      predictions
    }
  )
)

# Registrar no registry
ModelRegistry$register("my_model", MyModel)
```

Para guias completos, consulte a [documentação de plugins](docs/plugin-guide/overview.md).

## Contribuindo

Contribuições são bem-vindas! Por favor, leia o [guia de contribuição](CONTRIBUTING.md).

### Desenvolvimento Local

```bash
# Clonar repositório
git clone https://github.com/ons-br/prevcarga-R.git
cd prevcarga-R

# Restaurar ambiente
R -e "renv::restore()"

# Executar testes
R -e "testthat::test_local()"

# Verificar estilo
R -e "lintr::lint_package()"

# Build e check
R CMD build .
R CMD check prevcargaons_*.tar.gz
```

## Licença

Este projeto está licenciado sob a [Licença MIT](LICENSE).

## Créditos

Desenvolvido pelo [Operador Nacional do Sistema Elétrico (ONS)](https://www.ons.org.br/).
```

### Badge Configuration

```markdown
<!-- GitHub Actions CI -->
[![CI](https://github.com/ons-br/prevcarga-R/actions/workflows/ci.yml/badge.svg)](...)

<!-- Code Coverage -->
[![codecov](https://codecov.io/gh/ons-br/prevcarga-R/branch/main/graph/badge.svg)](...)

<!-- License -->
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](...)

<!-- R Version -->
[![R](https://img.shields.io/badge/R-%3E%3D4.3.0-blue.svg)](...)

<!-- Documentation -->
[![Documentation](https://readthedocs.org/projects/prevcargaons/badge/?version=latest)](...)
```

---

## Test Cases

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| TC-001 | README exists | File at root |
| TC-002 | All sections present | TOC matches content |
| TC-003 | Links work | No broken links |
| TC-004 | Code blocks valid | Syntax highlighted |
| TC-005 | Badges render | Images load |
| TC-006 | Examples accurate | Match CLI |
| TC-007 | Installation works | All methods documented |
| TC-008 | Configuration valid | YAML parses |
| TC-009 | Plugin example runs | R code works |
| TC-010 | Portuguese accurate | Proper grammar |

---

## Dependencies

- None (foundational documentation)

---

## Definition of Done

- [ ] README.md created at project root
- [ ] All sections complete
- [ ] Links verified
- [ ] Examples tested
- [ ] Badges configured
- [ ] Portuguese grammar reviewed
- [ ] Code reviewed and approved
- [ ] Merged to develop branch

---

## Notes

- Keep README concise but comprehensive
- Use Portuguese for user-facing content
- Link to detailed docs for deep-dives
- Include visual architecture diagram
- Ensure all examples are tested
