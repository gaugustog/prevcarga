# PC-102-12: WSL Installation Documentation

**Epic:** [EPIC-12: Documentation & Deploy](../epics/EPIC-12-documentation-deploy.md)
**Task Reference:** T-12.5
**Priority:** Medium
**Estimated Effort:** 0.5 days

---

## Summary

Create comprehensive WSL (Windows Subsystem for Linux) installation documentation for Windows users, covering WSL2 setup, Ubuntu installation, and running the one-line installer.

---

## Acceptance Criteria

- [ ] WSL2 installation prerequisites
- [ ] Step-by-step WSL setup guide
- [ ] Ubuntu distribution installation
- [ ] PrevCarga installation via curl
- [ ] VS Code integration guide
- [ ] Windows Terminal configuration
- [ ] File system access documentation
- [ ] Troubleshooting section

---

## Technical Specification

### Documentation File

```markdown
# docs/getting-started/installation-wsl.md

# Instalação no Windows via WSL

Este guia descreve como instalar e executar o PrevCargaONS no Windows usando
o Windows Subsystem for Linux (WSL2).

## O que é WSL?

O WSL (Windows Subsystem for Linux) permite executar um ambiente Linux
diretamente no Windows, sem necessidade de máquina virtual ou dual boot.

**Vantagens:**
- Desempenho nativo de Linux
- Acesso ao sistema de arquivos Windows
- Integração com VS Code
- Suporte a Docker

## Pré-requisitos

### Requisitos de Sistema

- **Windows**: Windows 10 versão 2004+ ou Windows 11
- **Arquitetura**: 64-bit (x64 ou ARM64)
- **RAM**: 8 GB (mínimo 4 GB)
- **Virtualização**: Habilitada na BIOS

### Verificar Versão do Windows

Pressione `Win + R`, digite `winver` e verifique:
- Build 19041 ou superior (Windows 10)
- Qualquer build (Windows 11)

## Passo 1: Instalar WSL2

### Método Simplificado (Windows 10 build 19041+)

Abra o **PowerShell como Administrador** e execute:

:::powershell
wsl --install
:::

Este comando:
1. Habilita os componentes do WSL
2. Habilita a virtualização
3. Instala o kernel Linux
4. Define WSL2 como padrão
5. Instala Ubuntu

**Reinicie o computador** após a instalação.

### Método Manual (Windows mais antigo)

Se o comando acima não funcionar:

:::powershell
# 1. Habilitar WSL
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart

# 2. Habilitar Virtual Machine Platform
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart

# 3. Reiniciar o computador
Restart-Computer

# 4. Após reiniciar, baixar e instalar o kernel WSL2
# https://aka.ms/wsl2kernel

# 5. Definir WSL2 como padrão
wsl --set-default-version 2
:::

## Passo 2: Instalar Ubuntu

### Via Microsoft Store (Recomendado)

1. Abra a **Microsoft Store**
2. Pesquise "Ubuntu 22.04 LTS"
3. Clique em **Obter** / **Instalar**
4. Após instalação, clique em **Abrir**

### Via Linha de Comando

:::powershell
# Listar distribuições disponíveis
wsl --list --online

# Instalar Ubuntu 22.04
wsl --install -d Ubuntu-22.04
:::

### Configurar Ubuntu

Na primeira execução, configure:

1. **Username**: Seu nome de usuário Linux (ex: `gabriel`)
2. **Password**: Senha para o usuário (será pedida para `sudo`)

:::bash
# Após configurar, você verá o prompt:
gabriel@DESKTOP-ABC123:~$
:::

## Passo 3: Atualizar Ubuntu

Antes de instalar o PrevCargaONS, atualize o sistema:

:::bash
sudo apt update && sudo apt upgrade -y
:::

## Passo 4: Instalar PrevCargaONS

Agora execute o instalador one-line:

:::bash
curl -fsSL https://ons.org.br/prevcarga/install.sh | bash
:::

O instalador irá:
1. Detectar Ubuntu/Debian
2. Instalar R via apt
3. Criar estrutura de diretórios
4. Instalar dependências via renv
5. Configurar PATH

### Verificar Instalação

:::bash
# Fechar e reabrir terminal, ou:
source ~/.bashrc

# Verificar instalação
prevcarga --version
# PrevCargaONS v1.0.0

# Iniciar modo interativo
prevcarga
:::

## Integração com VS Code

O VS Code tem excelente integração com WSL.

### Instalar Extensão WSL

1. Abra o VS Code no Windows
2. Instale a extensão **WSL** (ms-vscode-remote.remote-wsl)
3. Clique no ícone verde no canto inferior esquerdo
4. Selecione **Connect to WSL**

### Abrir Projeto no WSL

:::bash
# No terminal WSL, navegue até seu projeto
cd ~/projects/prevcarga

# Abra no VS Code
code .
:::

O VS Code abrirá com acesso completo ao ambiente Linux.

### Configurar R no VS Code

Instale a extensão **R** (REditorSupport.r) para:
- Syntax highlighting
- IntelliSense
- Execução de código
- Debug

## Windows Terminal (Recomendado)

O Windows Terminal oferece melhor experiência que o terminal padrão.

### Instalar

1. Abra a **Microsoft Store**
2. Pesquise "Windows Terminal"
3. Instale

### Configurar Ubuntu como Padrão

1. Abra Windows Terminal
2. Clique na seta ao lado da aba
3. **Configurações** > **Inicialização**
4. Defina Ubuntu como perfil padrão

### Configurações Recomendadas

Adicione ao `settings.json`:

:::json
{
    "profiles": {
        "list": [
            {
                "name": "Ubuntu-22.04",
                "source": "Windows.Terminal.Wsl",
                "startingDirectory": "//wsl$/Ubuntu-22.04/home/SEU_USUARIO",
                "colorScheme": "One Half Dark",
                "font": {
                    "face": "Cascadia Code",
                    "size": 11
                }
            }
        ]
    }
}
:::

## Acesso a Arquivos

### Do Windows para WSL

Os arquivos do WSL estão acessíveis em:
```
\\wsl$\Ubuntu-22.04\home\seu_usuario
```

Você pode mapear como unidade de rede:

:::powershell
# Mapear como unidade W:
net use W: \\wsl$\Ubuntu-22.04\home\seu_usuario
:::

### Do WSL para Windows

Os discos do Windows estão montados em `/mnt/`:

:::bash
# Acessar C:
cd /mnt/c/Users/SeuUsuario

# Acessar D:
cd /mnt/d/Dados
:::

### Recomendação de Performance

Para melhor desempenho, mantenha arquivos de projeto **dentro do WSL**:

:::bash
# BOM (arquivos no WSL)
~/projects/prevcarga/

# EVITAR (arquivos no Windows, acessados via WSL)
/mnt/c/Users/Usuario/projects/prevcarga/
:::

## Docker no WSL

O Docker Desktop integra-se com WSL2.

### Configurar Docker Desktop

1. Instale o [Docker Desktop](https://www.docker.com/products/docker-desktop)
2. Abra **Settings** > **Resources** > **WSL Integration**
3. Habilite para sua distribuição Ubuntu

### Usar Docker no WSL

:::bash
# No terminal Ubuntu
docker --version
docker run hello-world

# Executar PrevCarga via Docker
docker run -it --rm ons/prevcarga:latest
:::

## Troubleshooting

### WSL não inicia

:::powershell
# Verificar status
wsl --status

# Reiniciar WSL
wsl --shutdown
wsl

# Verificar virtualização na BIOS
# Reinicie e acesse BIOS (geralmente F2, F10, ou Del)
# Procure por "Virtualization Technology" ou "VT-x"
:::

### Erro de memória insuficiente

Crie ou edite `%USERPROFILE%\.wslconfig`:

:::ini
[wsl2]
memory=4GB
processors=2
swap=8GB
:::

Reinicie WSL:
:::powershell
wsl --shutdown
:::

### Instalação do R falha

:::bash
# Verificar erros
sudo apt update
sudo apt install -y r-base

# Se houver erro de chave GPG
wget -qO- https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc | sudo tee -a /etc/apt/trusted.gpg.d/cran_ubuntu_key.asc
:::

### PATH não configurado

:::bash
# Adicionar manualmente ao ~/.bashrc
echo 'export PATH="$HOME/.prevcarga/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
:::

### Locale warnings

:::bash
sudo locale-gen en_US.UTF-8
sudo update-locale LANG=en_US.UTF-8
:::

## Desinstalar

### Remover PrevCargaONS

:::bash
rm -rf ~/.prevcarga
# Remover linha do PATH em ~/.bashrc
:::

### Remover Ubuntu do WSL

:::powershell
wsl --unregister Ubuntu-22.04
:::

### Desabilitar WSL completamente

:::powershell
dism.exe /online /disable-feature /featurename:Microsoft-Windows-Subsystem-Linux
dism.exe /online /disable-feature /featurename:VirtualMachinePlatform
:::

## Próximos Passos

- [Quick Start](quickstart.md) - Tutorial inicial
- [Configuração](../user-guide/configuration.md) - Personalizar configurações
- [Docker](installation-docker.md) - Usar com Docker
```

---

## Test Cases

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| TC-001 | WSL installs | wsl --status works |
| TC-002 | Ubuntu installs | Can login |
| TC-003 | PrevCarga installs | prevcarga --version works |
| TC-004 | VS Code integrates | Opens in WSL |
| TC-005 | File access works | Both directions |
| TC-006 | Docker works | Container runs |
| TC-007 | Troubleshooting accurate | Fixes work |
| TC-008 | Links valid | No 404s |

---

## Dependencies

- PC-099-12: Sphinx Documentation Site

---

## Definition of Done

- [ ] Documentation page created
- [ ] All steps tested on Windows 10/11
- [ ] Screenshots added for key steps
- [ ] Links verified
- [ ] Added to Sphinx TOC
- [ ] Code reviewed and approved
- [ ] Merged to develop branch

---

## Notes

- Test on both Windows 10 and Windows 11
- Include screenshots for BIOS/Windows settings
- Keep PowerShell and Bash commands clearly separated
- Address common Windows-specific issues
