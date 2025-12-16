# PC-101-12: Docker Installation Documentation

**Epic:** [EPIC-12: Documentation & Deploy](../epics/EPIC-12-documentation-deploy.md)
**Task Reference:** T-12.4
**Priority:** Medium
**Estimated Effort:** 0.5 days

---

## Summary

Create comprehensive Docker installation documentation including prerequisites, pull/run commands, volume mounts, environment variables, and common use cases.

---

## Acceptance Criteria

- [ ] Prerequisites section (Docker installation)
- [ ] Quick start with docker run
- [ ] Volume mount guide for data persistence
- [ ] Environment variable reference
- [ ] Docker Compose usage
- [ ] Common use cases documented
- [ ] Troubleshooting section

---

## Technical Specification

### Documentation File

```markdown
# docs/getting-started/installation-docker.md

# Instalação via Docker

Esta guia descreve como executar o PrevCargaONS usando Docker.

## Pré-requisitos

### Docker Engine

Instale o Docker seguindo as instruções oficiais:

- **Linux**: [Docker Engine](https://docs.docker.com/engine/install/)
- **Windows/macOS**: [Docker Desktop](https://docs.docker.com/desktop/)

Verifique a instalação:

:::bash
docker --version
# Docker version 24.0.0 ou superior

docker compose version
# Docker Compose version v2.20.0 ou superior
:::

### Requisitos de Sistema

| Recurso | Mínimo | Recomendado |
|---------|--------|-------------|
| RAM | 4 GB | 8 GB |
| CPU | 2 cores | 4+ cores |
| Disco | 5 GB | 10 GB |

## Quick Start

### Executar Interativamente

:::bash
docker run -it --rm ons/prevcarga:latest
:::

Isso inicia o shell interativo do PrevCargaONS.

### Executar Comando Específico

:::bash
# Previsão para área RJ
docker run --rm ons/prevcarga:latest predict --area RJ --horizon 1

# Verificar versão
docker run --rm ons/prevcarga:latest --version
:::

## Persistência de Dados

Para persistir dados entre execuções, monte volumes:

### Estrutura de Diretórios

:::bash
# Criar estrutura local
mkdir -p ~/prevcarga/{data,models,config,logs,reports}
:::

### Docker Run com Volumes

:::bash
docker run -it --rm \
  -v ~/prevcarga/data:/app/data \
  -v ~/prevcarga/models:/app/models \
  -v ~/prevcarga/config:/app/config \
  -v ~/prevcarga/logs:/app/logs \
  -v ~/prevcarga/reports:/app/reports \
  ons/prevcarga:latest
:::

### Explicação dos Volumes

| Volume Host | Container Path | Descrição |
|-------------|----------------|-----------|
| `~/prevcarga/data` | `/app/data` | Dados de entrada (carga histórica) |
| `~/prevcarga/models` | `/app/models` | Modelos treinados |
| `~/prevcarga/config` | `/app/config` | Arquivos de configuração |
| `~/prevcarga/logs` | `/app/logs` | Logs de execução |
| `~/prevcarga/reports` | `/app/reports` | Relatórios gerados |

## Variáveis de Ambiente

Configure o container via variáveis de ambiente:

:::bash
docker run -it --rm \
  -e PREVCARGA_LOG_LEVEL=DEBUG \
  -e AWS_ACCESS_KEY_ID=AKIA... \
  -e AWS_SECRET_ACCESS_KEY=... \
  -e AWS_REGION=sa-east-1 \
  ons/prevcarga:latest
:::

### Variáveis Disponíveis

| Variável | Descrição | Padrão |
|----------|-----------|--------|
| `PREVCARGA_LOG_LEVEL` | Nível de log (DEBUG, INFO, WARN, ERROR) | INFO |
| `PREVCARGA_CONFIG` | Caminho do arquivo de configuração | /app/config/config.yaml |
| `AWS_ACCESS_KEY_ID` | Credencial AWS (para S3) | - |
| `AWS_SECRET_ACCESS_KEY` | Credencial AWS (para S3) | - |
| `AWS_REGION` | Região AWS | sa-east-1 |
| `TZ` | Timezone | America/Sao_Paulo |

## Docker Compose

Para uso regular, recomendamos Docker Compose:

### docker-compose.yml

:::yaml
version: '3.8'

services:
  prevcarga:
    image: ons/prevcarga:latest
    container_name: prevcarga
    stdin_open: true
    tty: true
    environment:
      - PREVCARGA_LOG_LEVEL=INFO
      - TZ=America/Sao_Paulo
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./config:/app/config
      - ./logs:/app/logs
      - ./reports:/app/reports
    # Para acesso S3, adicione credenciais
    # env_file:
    #   - .env
:::

### Comandos Docker Compose

:::bash
# Iniciar interativamente
docker compose run --rm prevcarga

# Executar comando
docker compose run --rm prevcarga predict --area RJ

# Treinar modelo
docker compose run --rm prevcarga train --model lgbm --area RJ

# Backtest
docker compose run --rm prevcarga backtest \
  --start-date 2024-01-01 \
  --end-date 2024-12-31
:::

## Casos de Uso

### 1. Previsão Diária Automatizada

:::bash
# Criar script de previsão diária
cat > run_daily_forecast.sh << 'EOF'
#!/bin/bash
DATE=$(date +%Y-%m-%d)
docker compose run --rm prevcarga predict \
  --areas all \
  --horizons 0:8 \
  --date $DATE \
  --output /app/reports/forecast_${DATE}.csv
EOF

chmod +x run_daily_forecast.sh

# Adicionar ao cron (executa às 6:00)
# 0 6 * * * /path/to/run_daily_forecast.sh
:::

### 2. Treinamento Semanal

:::bash
# Script de retreino semanal
cat > run_weekly_training.sh << 'EOF'
#!/bin/bash
docker compose run --rm prevcarga train \
  --all-models \
  --areas all \
  --output /app/models
EOF
:::

### 3. Acesso a Dados S3

:::bash
# Criar arquivo .env
cat > .env << 'EOF'
AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE
AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
AWS_REGION=sa-east-1
EOF

# Usar com docker-compose
docker compose --env-file .env run --rm prevcarga predict --area RJ
:::

### 4. Desenvolvimento e Debug

:::bash
# Iniciar com acesso ao código fonte (desenvolvimento)
docker run -it --rm \
  -v $(pwd):/app/src \
  -v ~/prevcarga/data:/app/data \
  -w /app/src \
  ons/prevcarga:latest bash

# Dentro do container
R -e "devtools::load_all(); prevcarga_cli()"
:::

## Tags de Imagem

| Tag | Descrição |
|-----|-----------|
| `latest` | Versão estável mais recente |
| `1.0.0` | Versão específica |
| `1.0` | Última patch da versão 1.0.x |
| `develop` | Build de desenvolvimento (instável) |

:::bash
# Usar versão específica
docker pull ons/prevcarga:1.0.0
docker run -it --rm ons/prevcarga:1.0.0
:::

## Troubleshooting

### Container não inicia

:::bash
# Verificar logs
docker logs prevcarga

# Verificar recursos disponíveis
docker info | grep -E "Memory|CPUs"
:::

### Permissão negada nos volumes

:::bash
# Linux: ajustar permissões
sudo chown -R $(id -u):$(id -g) ~/prevcarga/

# Ou usar usuário específico
docker run -it --rm \
  --user $(id -u):$(id -g) \
  -v ~/prevcarga/data:/app/data \
  ons/prevcarga:latest
:::

### Memória insuficiente

:::bash
# Aumentar limite de memória
docker run -it --rm \
  --memory=8g \
  --memory-swap=8g \
  ons/prevcarga:latest
:::

### Conexão S3 falha

:::bash
# Testar credenciais
docker run --rm \
  -e AWS_ACCESS_KEY_ID=... \
  -e AWS_SECRET_ACCESS_KEY=... \
  amazon/aws-cli s3 ls s3://seu-bucket/
:::

## Próximos Passos

- [Quick Start](quickstart.md) - Tutorial inicial
- [Configuração](../user-guide/configuration.md) - Personalizar configurações
- [CLI Reference](../user-guide/cli-reference.md) - Todos os comandos
```

---

## Test Cases

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| TC-001 | Docker run works | Container starts |
| TC-002 | Volumes mount correctly | Data persists |
| TC-003 | Environment vars work | Config applied |
| TC-004 | Docker Compose works | Service starts |
| TC-005 | Commands execute | Output correct |
| TC-006 | S3 access works | Data loads |
| TC-007 | Troubleshooting accurate | Fixes work |
| TC-008 | Links valid | No 404s |

---

## Dependencies

- PC-099-12: Sphinx Documentation Site
- PC-103-12: Dockerfile (for image reference)

---

## Definition of Done

- [ ] Documentation page created
- [ ] All commands tested
- [ ] Screenshots added (if needed)
- [ ] Links verified
- [ ] Added to Sphinx TOC
- [ ] Code reviewed and approved
- [ ] Merged to develop branch

---

## Notes

- Keep examples practical and tested
- Include common error scenarios
- Reference actual Docker image tags
- Ensure Portuguese is correct
