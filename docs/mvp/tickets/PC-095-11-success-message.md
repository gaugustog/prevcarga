# PC-095-11: Success Message

**Epic:** [EPIC-11: One-Line Installer](../epics/EPIC-11-installer.md)
**Task Reference:** T-11.10
**Priority:** Medium
**Estimated Effort:** 0.25 days

---

## Summary

Display a clear success message after installation with instructions for getting started, including how to source the shell configuration and run the first command.

---

## Acceptance Criteria

- [ ] Success banner displayed with colored output
- [ ] Shows `source` command for current shell
- [ ] Shows `prevcarga` command example
- [ ] Displays installation directory
- [ ] Shows version installed
- [ ] Provides next steps for users

---

## Technical Specification

### Success Message Function

```bash
# ──────────────────────────────────────────────────────────────────────────────
# Success Message
# ──────────────────────────────────────────────────────────────────────────────

#' Display installation success message
show_success_message() {
    local elapsed_seconds="$1"
    local shell_rc

    # Determine shell rc file
    case "$(detect_user_shell)" in
        bash)
            shell_rc="~/.bashrc"
            [[ "$OS_FAMILY" == "darwin" ]] && shell_rc="~/.bash_profile"
            ;;
        zsh)
            shell_rc="~/.zshrc"
            ;;
        fish)
            shell_rc="~/.config/fish/config.fish"
            ;;
        *)
            shell_rc="seu arquivo de shell"
            ;;
    esac

    echo ""
    echo -e "${GREEN}══════════════════════════════════════════════════════════════════${RESET}"
    echo -e "${GREEN}${RESET}"
    echo -e "${GREEN}  ✓ INSTALAÇÃO CONCLUÍDA!${RESET}"
    echo -e "${GREEN}${RESET}"
    echo -e "${GREEN}══════════════════════════════════════════════════════════════════${RESET}"
    echo ""
    echo -e "  ${DIM}Versão:${RESET}     ${BOLD}$VERSION${RESET}"
    echo -e "  ${DIM}Diretório:${RESET}  ${BOLD}$INSTALL_DIR${RESET}"
    echo -e "  ${DIM}Tempo:${RESET}      ${elapsed_seconds}s"
    echo ""
    echo -e "  ${CYAN}━━━ Próximos Passos ━━━${RESET}"
    echo ""
    echo -e "  ${BOLD}1.${RESET} Recarregue seu shell:"
    echo ""
    echo -e "     ${CYAN}source $shell_rc${RESET}"
    echo ""
    echo -e "  ${BOLD}2.${RESET} Inicie o PrevCargaONS:"
    echo ""
    echo -e "     ${CYAN}prevcarga${RESET}               # Modo interativo"
    echo -e "     ${CYAN}prevcarga --demo${RESET}        # Modo demonstração"
    echo -e "     ${CYAN}prevcarga --help${RESET}        # Ajuda"
    echo ""
    echo -e "  ${BOLD}3.${RESET} Comandos úteis:"
    echo ""
    echo -e "     ${CYAN}prevcarga forecast SE${RESET}   # Previsão para SE"
    echo -e "     ${CYAN}prevcarga status${RESET}        # Status do sistema"
    echo -e "     ${CYAN}prevcarga areas${RESET}         # Listar áreas"
    echo ""
    echo -e "  ${CYAN}━━━ Documentação ━━━${RESET}"
    echo ""
    echo -e "  Guia completo: ${CYAN}https://github.com/ons-br/prevcarga-R#readme${RESET}"
    echo -e "  Reportar bugs: ${CYAN}https://github.com/ons-br/prevcarga-R/issues${RESET}"
    echo ""
    echo -e "${GREEN}══════════════════════════════════════════════════════════════════${RESET}"
    echo ""
}

#' Display quick start hint (alternative compact version)
show_quick_start() {
    echo ""
    echo -e "${GREEN}✓${RESET} Instalação concluída!"
    echo ""
    echo -e "  Para começar:"
    echo -e "    ${CYAN}source ~/.bashrc && prevcarga${RESET}"
    echo ""
}

#' Display error summary if installation failed
show_error_summary() {
    local error_msg="$1"

    echo ""
    echo -e "${RED}══════════════════════════════════════════════════════════════════${RESET}"
    echo -e "${RED}${RESET}"
    echo -e "${RED}  ✗ INSTALAÇÃO FALHOU${RESET}"
    echo -e "${RED}${RESET}"
    echo -e "${RED}══════════════════════════════════════════════════════════════════${RESET}"
    echo ""
    echo -e "  ${RED}Erro:${RESET} $error_msg"
    echo ""
    echo -e "  ${CYAN}━━━ Soluções ━━━${RESET}"
    echo ""
    echo -e "  ${BOLD}1.${RESET} Verifique os logs:"
    echo ""
    echo -e "     ${CYAN}cat /tmp/prevcarga_install.log${RESET}"
    echo ""
    echo -e "  ${BOLD}2.${RESET} Tente novamente com modo debug:"
    echo ""
    echo -e "     ${CYAN}curl -fsSL .../install.sh | DEBUG=1 bash${RESET}"
    echo ""
    echo -e "  ${BOLD}3.${RESET} Reporte o problema:"
    echo ""
    echo -e "     ${CYAN}https://github.com/ons-br/prevcarga-R/issues${RESET}"
    echo ""
    echo -e "${RED}══════════════════════════════════════════════════════════════════${RESET}"
    echo ""
}

#' Display update available message
show_update_available() {
    local current_version="$1"
    local latest_version="$2"

    echo ""
    echo -e "${YELLOW}━━━ Atualização Disponível ━━━${RESET}"
    echo ""
    echo -e "  Versão atual:  ${BOLD}$current_version${RESET}"
    echo -e "  Nova versão:   ${BOLD}$latest_version${RESET}"
    echo ""
    echo -e "  Para atualizar:"
    echo -e "    ${CYAN}prevcarga --update${RESET}"
    echo ""
}

#' Display demo mode instructions
show_demo_instructions() {
    echo ""
    echo -e "${CYAN}━━━ Modo Demonstração ━━━${RESET}"
    echo ""
    echo -e "  O PrevCargaONS está em modo demonstração com dados simulados."
    echo ""
    echo -e "  Experimente:"
    echo -e "    ${CYAN}forecast SE${RESET}         # Previsão para Sudeste"
    echo -e "    ${CYAN}status${RESET}              # Status do sistema"
    echo -e "    ${CYAN}areas${RESET}               # Listar áreas"
    echo -e "    ${CYAN}models${RESET}              # Listar modelos"
    echo ""
    echo -e "  Para sair: ${CYAN}quit${RESET} ou ${CYAN}Ctrl+D${RESET}"
    echo ""
}
```

### Main Installation Completion

```bash
#' Complete installation and show success
complete_installation() {
    local start_time="$1"
    local end_time=$(date +%s)
    local elapsed=$((end_time - start_time))

    # Verify installation
    verify_installation || {
        show_error_summary "Verificação de instalação falhou"
        exit 1
    }

    # Show success message
    show_success_message "$elapsed"

    # Log success
    info "Instalação concluída com sucesso em ${elapsed}s"

    exit 0
}

#' Verify installation is complete
verify_installation() {
    local checks_passed=true

    # Check directories
    for dir in bin src renv config logs; do
        if [[ ! -d "$INSTALL_DIR/$dir" ]]; then
            error "Diretório faltando: $dir"
            checks_passed=false
        fi
    done

    # Check launcher
    if [[ ! -x "$INSTALL_DIR/bin/prevcarga" ]]; then
        error "Launcher não encontrado ou não executável"
        checks_passed=false
    fi

    # Check shell
    if [[ ! -f "$INSTALL_DIR/src/prevcarga_shell.R" ]]; then
        error "Shell R não encontrado"
        checks_passed=false
    fi

    # Check config
    if [[ ! -f "$INSTALL_DIR/config/config.yaml" ]]; then
        error "Configuração não encontrada"
        checks_passed=false
    fi

    # Check renv
    if [[ ! -d "$INSTALL_DIR/renv/library" ]]; then
        error "Biblioteca renv não encontrada"
        checks_passed=false
    fi

    [[ "$checks_passed" == "true" ]]
}
```

---

## Test Cases

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| TC-001 | Success banner displayed | Green box shown |
| TC-002 | Version shown | Correct version |
| TC-003 | Directory shown | $INSTALL_DIR |
| TC-004 | Time shown | Elapsed seconds |
| TC-005 | source command correct | Matches shell |
| TC-006 | Examples shown | 3 commands |
| TC-007 | Links shown | GitHub URLs |
| TC-008 | Error summary works | Red box on failure |
| TC-009 | Colors disabled in pipe | No escape codes |
| TC-010 | Quick start works | Compact version |

---

## Dependencies

- PC-086-11: Installer Script Structure
- PC-093-11: PATH Configuration

---

## Definition of Done

- [ ] Success banner implemented
- [ ] Correct source command for each shell
- [ ] Example commands shown
- [ ] Documentation links included
- [ ] Error summary implemented
- [ ] Installation verification working
- [ ] Colors work in terminal
- [ ] Code reviewed and approved
- [ ] Merged to develop branch

---

## Notes

- Keep messages in Portuguese
- Use consistent colors throughout
- Provide actionable next steps
- Include links for support
