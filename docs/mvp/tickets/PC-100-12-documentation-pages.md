# PC-100-12: Documentation Pages

**Epic:** [EPIC-12: Documentation & Deploy](../epics/EPIC-12-documentation-deploy.md)
**Task Reference:** T-12.3
**Priority:** High
**Estimated Effort:** 3 days

---

## Summary

Create all documentation pages for the Sphinx site including getting started guides, user guide, plugin development guides, and API reference.

---

## Acceptance Criteria

- [ ] Getting Started section complete (4 pages)
- [ ] User Guide section complete (3 pages)
- [ ] Plugin Guide section complete (5 pages)
- [ ] API Reference section complete (1+ pages)
- [ ] All pages linked in TOC
- [ ] All code examples tested

---

## Technical Specification

### Getting Started Section

#### Installation Overview

```markdown
# docs/getting-started/installation.md

# Instalação

Este guia apresenta as diferentes formas de instalar o PrevCargaONS.

## Métodos de Instalação

| Método | Sistema | Recomendado Para |
|--------|---------|------------------|
| [One-Line Installer](installation-curl.md) | Linux, macOS | Usuários finais |
| [Docker](installation-docker.md) | Qualquer | Ambientes isolados |
| [WSL](installation-wsl.md) | Windows | Usuários Windows |
| Manual | Qualquer | Desenvolvedores |

## Requisitos

### Requisitos Mínimos

- **R**: >= 4.3.0
- **RAM**: 4 GB (8 GB recomendado)
- **Disco**: 2 GB de espaço livre
- **CPU**: 2 cores (4+ recomendado para paralelismo)

### Sistemas Operacionais Suportados

- **Linux**: Ubuntu 20.04+, Debian 11+, Fedora 38+, CentOS 8+
- **macOS**: 12 (Monterey) ou superior
- **Windows**: Windows 10/11 com WSL2

## Instalação Manual

Para desenvolvedores que desejam contribuir com o projeto:

:::bash
# Clonar repositório
git clone https://github.com/ons-br/prevcarga-R.git
cd prevcarga-R

# Restaurar ambiente renv
R -e "renv::restore()"

# Instalar pacote em modo desenvolvimento
R CMD INSTALL .

# Verificar instalação
prevcarga --version
:::

## Próximos Passos

Após a instalação, siga o [Quick Start](quickstart.md) para começar a usar o sistema.
```

#### Quick Start

```markdown
# docs/getting-started/quickstart.md

# Quick Start

Este guia mostra como começar a usar o PrevCargaONS em 5 minutos.

## Passo 1: Verificar Instalação

:::bash
prevcarga --version
# PrevCargaONS v1.0.0
:::

## Passo 2: Modo Interativo

Inicie o shell interativo:

:::bash
prevcarga
:::

Você verá o prompt:

:::
╔══════════════════════════════════════════════════════════════════╗
║                     PrevCargaONS v1.0.0                          ║
║        Sistema de Previsão de Carga Elétrica - ONS               ║
╚══════════════════════════════════════════════════════════════════╝

prevcarga>
:::

## Passo 3: Comandos Básicos

### Listar Áreas

:::bash
prevcarga> areas
# Áreas disponíveis: RJ, SP, MG, ES, GO, ...
:::

### Verificar Status

:::bash
prevcarga> status
# Status do sistema e modelos disponíveis
:::

### Gerar Previsão

:::bash
prevcarga> forecast RJ
# Previsão para área RJ, horizonte D+1
:::

## Passo 4: Uso via CLI

Para automação, use comandos diretamente:

:::bash
# Previsão simples
prevcarga predict --area RJ --horizon 1

# Previsão para múltiplas áreas
prevcarga predict --areas RJ,SP,MG --horizons 0,1,2

# Treinar modelo
prevcarga train --model lgbm --area RJ

# Executar backtest
prevcarga backtest --start-date 2024-01-01 --end-date 2024-12-31
:::

## Passo 5: Configuração

A configuração padrão está em `~/.prevcarga/config/config.yaml`.

Para personalizar:

:::bash
# Editar configuração
nano ~/.prevcarga/config/config.yaml

# Ou usar configuração customizada
prevcarga predict --config ~/my_config.yaml --area RJ
:::

## Próximos Passos

- [Configuração detalhada](../user-guide/configuration.md)
- [Referência de comandos CLI](../user-guide/cli-reference.md)
- [Desenvolvimento de plugins](../plugin-guide/overview.md)
```

### User Guide Section

#### Configuration Reference

```markdown
# docs/user-guide/configuration.md

# Configuração

Referência completa para configuração do PrevCargaONS.

## Localização do Arquivo

O arquivo de configuração principal está em:

- **Linux/macOS**: `~/.prevcarga/config/config.yaml`
- **Docker**: `/app/config/config.yaml`

## Estrutura do Arquivo

### Configuração de Projeto

:::yaml
project:
  name: "PrevCargaONS"
  version: "1.0.0"
  seed: 42
  environment: "production"  # development, staging, production
  timezone: "America/Sao_Paulo"
:::

### Configuração de Armazenamento

:::yaml
storage:
  backend: local  # local, s3, azure, gcs

  local:
    base_path: ~/.prevcarga/data
    carga_path: carga
    previsao_path: previsao
    models_path: models

  s3:
    bucket: ons-prevcarga-prod
    prefix: data/
    region: sa-east-1
    # Credenciais via variáveis de ambiente:
    # AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
:::

### Configuração de Logging

:::yaml
logging:
  level: INFO  # DEBUG, INFO, WARN, ERROR
  format: json  # json, text

  handlers:
    - type: console
      level: INFO
      colored: true

    - type: file
      level: DEBUG
      path: ~/.prevcarga/logs/prevcarga.log
      max_size_mb: 10
      backup_count: 5
:::

### Configuração de Modelos

:::yaml
models:
  enabled:
    - lgbm
    - rf
    - hw

  default_horizons: [0, 1, 2, 3, 4, 5, 6, 7, 8]
  train_window_days: 365
  retrain_frequency: 7

  lgbm:
    num_leaves: 31
    learning_rate: 0.05
    n_estimators: 100

  rf:
    num_trees: 100
    max_depth: 10

  hw:
    seasonal_periods: 24
    damped: true
:::

### Configuração de Features

:::yaml
features:
  plugins:
    calendar:
      enabled: true

    lag:
      enabled: true
      config:
        lags: [1, 24, 168]

    rolling:
      enabled: true
      config:
        windows: [24, 168]
        functions: [mean, sd]
:::

### Configuração de Combinação

:::yaml
combination:
  default: inverse_mape
  enabled:
    - simple_average
    - inverse_mape
    - optimal
  bias_correction: true
:::

### Configuração de Reconciliação

:::yaml
reconciliation:
  default: bottom_up
  enabled:
    - bottom_up
    - ols
    - mint
  include_losses: true
  loss_factors:
    SECO: 0.025
    S: 0.022
    NE: 0.028
    N: 0.030
:::

### Configuração de Paralelismo

:::yaml
parallel:
  enabled: true
  workers: auto  # auto = CPU cores - 1
  backend: future
  memory_limit: 4096  # MB por worker
:::

## Variáveis de Ambiente

| Variável | Descrição |
|----------|-----------|
| `PREVCARGA_HOME` | Diretório de instalação |
| `PREVCARGA_CONFIG` | Caminho para config.yaml |
| `PREVCARGA_LOG_LEVEL` | Override do nível de log |
| `AWS_ACCESS_KEY_ID` | Credencial AWS (para S3) |
| `AWS_SECRET_ACCESS_KEY` | Credencial AWS (para S3) |
| `AWS_REGION` | Região AWS |

## Exemplo Completo

Veja o arquivo de exemplo em `~/.prevcarga/config/config.example.yaml` para
todas as opções disponíveis.
```

#### CLI Reference

```markdown
# docs/user-guide/cli-reference.md

# Referência CLI

Documentação completa de todos os comandos disponíveis.

## Sintaxe Geral

:::bash
prevcarga <command> [options]
:::

## Comandos Principais

### train

Treina modelos de previsão.

:::bash
prevcarga train [options]
:::

**Opções:**

| Opção | Descrição | Padrão |
|-------|-----------|--------|
| `--model` | Modelo a treinar (lgbm, rf, hw) | Todos habilitados |
| `--area` | Área específica | Todas |
| `--areas` | Lista de áreas (separadas por vírgula) | - |
| `--train-start` | Data início do treino | 1 ano atrás |
| `--train-end` | Data fim do treino | Ontem |
| `--config` | Arquivo de configuração | Padrão |
| `--output` | Diretório de saída | models/ |
| `--parallel` | Usar processamento paralelo | true |

**Exemplos:**

:::bash
# Treinar LightGBM para área RJ
prevcarga train --model lgbm --area RJ

# Treinar todos os modelos para múltiplas áreas
prevcarga train --areas RJ,SP,MG --parallel

# Treinar com período específico
prevcarga train --model lgbm --train-start 2023-01-01 --train-end 2023-12-31
:::

### predict

Gera previsões de carga.

:::bash
prevcarga predict [options]
:::

**Opções:**

| Opção | Descrição | Padrão |
|-------|-----------|--------|
| `--area` | Área para previsão | - |
| `--areas` | Lista de áreas | - |
| `--subsystem` | Subsistema (SECO, S, NE, N) | - |
| `--horizon` | Horizonte único (0-8) | 1 |
| `--horizons` | Lista de horizontes | 0:8 |
| `--date` | Data base da previsão | Hoje |
| `--mode` | Modo (batch, intraday) | batch |
| `--output` | Diretório de saída | predictions/ |
| `--format` | Formato (csv, parquet, json) | parquet |

**Exemplos:**

:::bash
# Previsão D+1 para RJ
prevcarga predict --area RJ --horizon 1

# Previsão todos os horizontes para subsistema
prevcarga predict --subsystem SECO --horizons 0:8

# Previsão em modo intraday
prevcarga predict --area RJ --mode intraday --horizon 0
:::

### backtest

Executa validação retrospectiva.

:::bash
prevcarga backtest [options]
:::

**Opções:**

| Opção | Descrição | Padrão |
|-------|-----------|--------|
| `--start-date` | Data início do backtest | - |
| `--end-date` | Data fim do backtest | - |
| `--areas` | Áreas para backtest | Todas |
| `--models` | Modelos para backtest | Todos |
| `--step` | Passo de retreino (dias) | 7 |
| `--output` | Arquivo de saída | backtest_results.html |
| `--parallel` | Usar paralelismo | true |

**Exemplos:**

:::bash
# Backtest de 1 ano
prevcarga backtest --start-date 2024-01-01 --end-date 2024-12-31

# Backtest com retreino semanal
prevcarga backtest --start-date 2024-01-01 --end-date 2024-06-30 --step 7
:::

### report

Gera relatórios interativos.

:::bash
prevcarga report <type> [options]
:::

**Tipos de relatório:**

| Tipo | Descrição |
|------|-----------|
| `forecast` | Análise de previsão |
| `backtest` | Resultados de backtest |
| `compare` | Comparação de modelos |
| `dashboard` | Dashboard de área |
| `drift` | Análise de drift |

**Exemplos:**

:::bash
# Relatório de previsão
prevcarga report forecast --date 2025-01-15 --areas RJ,SP -o forecast.html

# Comparação de modelos
prevcarga report compare --models lgbm,rf --period 2024-10-01:2024-12-31
:::

### interactive

Inicia o shell interativo.

:::bash
prevcarga interactive
# ou simplesmente
prevcarga
:::

## Opções Globais

| Opção | Descrição |
|-------|-----------|
| `--config` | Arquivo de configuração |
| `--verbose` | Saída detalhada |
| `--quiet` | Saída mínima |
| `--version` | Mostrar versão |
| `--help` | Mostrar ajuda |
```

### Plugin Guide Section

#### Plugin Overview

```markdown
# docs/plugin-guide/overview.md

# Arquitetura de Plugins

O PrevCargaONS utiliza uma arquitetura de plugins extensível que permite
adicionar novos modelos, features, combinadores e reconciliadores sem
modificar o código principal.

## Tipos de Plugins

| Tipo | Classe Base | Registry | Descrição |
|------|-------------|----------|-----------|
| Modelo | `BaseModel` | `ModelRegistry` | Algoritmos de previsão |
| Feature | `BaseFeaturePlugin` | `FeaturePluginRegistry` | Geradores de features |
| Combiner | `BaseCombiner` | `CombinerRegistry` | Combinadores de previsões |
| Reconciler | `BaseReconciler` | `ReconcilerRegistry` | Reconciliadores hierárquicos |

## Padrão de Implementação

Todos os plugins seguem o mesmo padrão:

1. **Herdar da classe base** (`inherit = BaseXxx`)
2. **Implementar métodos abstratos** (definidos na classe base)
3. **Registrar no registry** (`XxxRegistry$register(...)`)

## Exemplo Básico

:::r
# 1. Definir a classe
MyPlugin <- R6::R6Class(
  "MyPlugin",
  inherit = BaseModel,

  public = list(
    initialize = function(config = list()) {
      super$initialize(
        model_id = "my_plugin",
        model_type = "custom",
        config = config
      )
    },

    fit = function(X, y, ...) {
      # Implementação
      private$.fitted <- TRUE
      invisible(self)
    },

    predict = function(X, ...) {
      # Implementação
      predictions
    }
  )
)

# 2. Registrar
ModelRegistry$register("my_plugin", MyPlugin)

# 3. Usar
model <- ModelRegistry$create("my_plugin")
model$fit(X, y)
predictions <- model$predict(X_new)
:::

## Ciclo de Vida

:::
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Register   │ ──► │   Create    │ ──► │    Use      │
└─────────────┘     └─────────────┘     └─────────────┘
       │                   │                   │
       ▼                   ▼                   ▼
  XxxRegistry        XxxRegistry          instance
  $register()        $create()            $method()
:::

## Guias Detalhados

- [Feature Plugins](feature-plugins.md)
- [Model Plugins](model-plugins.md)
- [Combiner Plugins](combiner-plugins.md)
- [Reconciler Plugins](reconciler-plugins.md)
```

---

## Page List Summary

### Getting Started (4 pages)
1. `installation.md` - Installation overview
2. `installation-docker.md` - Docker installation
3. `installation-wsl.md` - WSL installation
4. `installation-curl.md` - One-line installer
5. `quickstart.md` - Quick start tutorial

### User Guide (3 pages)
1. `configuration.md` - Configuration reference
2. `cli-reference.md` - CLI command reference
3. `storage-backends.md` - Storage backends

### Plugin Guide (5 pages)
1. `overview.md` - Plugin architecture
2. `feature-plugins.md` - Feature plugin development
3. `model-plugins.md` - Model plugin development
4. `combiner-plugins.md` - Combiner plugin development
5. `reconciler-plugins.md` - Reconciler plugin development

### API Reference (1+ pages)
1. `index.md` - R6 class reference overview

---

## Test Cases

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| TC-001 | All pages linked | No orphan pages |
| TC-002 | TOC navigation | Works correctly |
| TC-003 | Code examples | Syntax correct |
| TC-004 | Internal links | No 404s |
| TC-005 | YAML examples | Valid syntax |
| TC-006 | CLI examples | Match actual CLI |
| TC-007 | Portuguese grammar | Correct |
| TC-008 | Formatting | Consistent |
| TC-009 | Images render | If any |
| TC-010 | Mobile readable | Responsive |

---

## Dependencies

- PC-099-12: Sphinx Documentation Site

---

## Definition of Done

- [ ] All 14 documentation pages created
- [ ] All pages linked in TOC
- [ ] Code examples tested
- [ ] Internal links verified
- [ ] Sphinx build passes
- [ ] Portuguese reviewed
- [ ] Code reviewed and approved
- [ ] Merged to develop branch

---

## Notes

- Keep content in Portuguese
- Use consistent formatting
- Include practical examples
- Link between related pages
- Add admonitions for important notes
