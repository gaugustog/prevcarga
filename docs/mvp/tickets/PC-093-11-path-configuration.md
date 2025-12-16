# PC-093-11: PATH Configuration

**Epic:** [EPIC-11: One-Line Installer](../epics/EPIC-11-installer.md)
**Task Reference:** T-11.8
**Priority:** High
**Estimated Effort:** 0.5 days

---

## Summary

Configure the user's shell PATH to include `~/.prevcarga/bin` so that the `prevcarga` command is available from any directory after installation.

---

## Acceptance Criteria

- [ ] Detects user's default shell (bash/zsh)
- [ ] Adds PATH entry to appropriate rc file
- [ ] Avoids duplicate PATH entries
- [ ] Works with fish shell (optional)
- [ ] Provides manual instructions for other shells
- [ ] Creates shell completion script

---

## Technical Specification

### PATH Configuration Functions

```bash
# ──────────────────────────────────────────────────────────────────────────────
# PATH Configuration
# ──────────────────────────────────────────────────────────────────────────────

#' Configure PATH in user's shell
configure_path() {
    step "7/7 - Configurando PATH"

    local bin_path="$INSTALL_DIR/bin"

    # Detect shell
    local user_shell
    user_shell=$(detect_user_shell)

    info "Shell detectado: ${BOLD}$user_shell${RESET}"

    case "$user_shell" in
        bash)
            configure_bash_path "$bin_path"
            ;;
        zsh)
            configure_zsh_path "$bin_path"
            ;;
        fish)
            configure_fish_path "$bin_path"
            ;;
        *)
            show_manual_path_instructions "$bin_path"
            ;;
    esac

    # Create shell completion
    create_shell_completion "$user_shell"

    success "PATH configurado"
}

#' Detect user's default shell
detect_user_shell() {
    # Try to get from SHELL variable
    local shell_name
    shell_name=$(basename "${SHELL:-/bin/bash}")

    # Validate it's a known shell
    case "$shell_name" in
        bash|zsh|fish|tcsh|csh|sh)
            echo "$shell_name"
            ;;
        *)
            # Fallback to bash
            echo "bash"
            ;;
    esac
}

#' Configure PATH for bash
configure_bash_path() {
    local bin_path="$1"
    local rc_file="$HOME/.bashrc"

    # Also check .bash_profile for macOS
    if [[ "$OS_FAMILY" == "darwin" ]] && [[ -f "$HOME/.bash_profile" ]]; then
        rc_file="$HOME/.bash_profile"
    fi

    add_path_to_rc "$bin_path" "$rc_file"
}

#' Configure PATH for zsh
configure_zsh_path() {
    local bin_path="$1"
    local rc_file="$HOME/.zshrc"

    add_path_to_rc "$bin_path" "$rc_file"
}

#' Configure PATH for fish
configure_fish_path() {
    local bin_path="$1"
    local fish_config="$HOME/.config/fish/config.fish"

    # Create directory if needed
    mkdir -p "$(dirname "$fish_config")"

    # Check if already configured
    if grep -q "PrevCargaONS" "$fish_config" 2>/dev/null; then
        info "PATH já configurado em $fish_config"
        return 0
    fi

    # Add fish-style PATH
    cat >> "$fish_config" << EOF

# PrevCargaONS
set -gx PATH $bin_path \$PATH
EOF

    success "PATH adicionado a $fish_config"
}

#' Add PATH entry to rc file
add_path_to_rc() {
    local bin_path="$1"
    local rc_file="$2"

    # Create file if it doesn't exist
    touch "$rc_file"

    # Check if already configured
    if grep -q "PrevCargaONS" "$rc_file" 2>/dev/null; then
        info "PATH já configurado em $rc_file"
        return 0
    fi

    # Check if PATH already contains bin_path
    if echo "$PATH" | tr ':' '\n' | grep -q "^${bin_path}$"; then
        info "PATH já contém $bin_path"
        # Still add to rc file for future sessions
    fi

    # Add PATH entry with marker
    cat >> "$rc_file" << EOF

# PrevCargaONS
export PATH="$bin_path:\$PATH"
EOF

    success "PATH adicionado a $rc_file"
}

#' Show manual instructions for unsupported shells
show_manual_path_instructions() {
    local bin_path="$1"

    warn "Shell não suportado para configuração automática"
    echo ""
    echo "  Adicione manualmente ao seu arquivo de configuração:"
    echo ""
    echo -e "    ${CYAN}export PATH=\"$bin_path:\$PATH\"${RESET}"
    echo ""
}

#' Create shell completion script
create_shell_completion() {
    local shell="$1"
    local completion_dir="$INSTALL_DIR/share/completion"

    mkdir -p "$completion_dir"

    case "$shell" in
        bash)
            create_bash_completion "$completion_dir"
            ;;
        zsh)
            create_zsh_completion "$completion_dir"
            ;;
        fish)
            create_fish_completion "$completion_dir"
            ;;
    esac
}

#' Create bash completion script
create_bash_completion() {
    local completion_dir="$1"

    cat > "$completion_dir/prevcarga.bash" << 'COMPLETION'
# PrevCargaONS bash completion

_prevcarga_completions() {
    local cur="${COMP_WORDS[COMP_CWORD]}"
    local prev="${COMP_WORDS[COMP_CWORD-1]}"

    # Main commands
    local commands="forecast backtest train predict status areas models help interactive"

    # Subsystem codes
    local subsystems="SECO S NE N SIN"

    case "$prev" in
        prevcarga)
            COMPREPLY=($(compgen -W "$commands --help --version --demo" -- "$cur"))
            ;;
        forecast|predict)
            COMPREPLY=($(compgen -W "$subsystems" -- "$cur"))
            ;;
        --config)
            COMPREPLY=($(compgen -f -X '!*.yaml' -- "$cur"))
            ;;
        *)
            COMPREPLY=()
            ;;
    esac
}

complete -F _prevcarga_completions prevcarga
COMPLETION

    # Install completion
    if [[ -d /etc/bash_completion.d ]]; then
        maybe_sudo cp "$completion_dir/prevcarga.bash" /etc/bash_completion.d/prevcarga
    elif [[ -d "$HOME/.local/share/bash-completion/completions" ]]; then
        cp "$completion_dir/prevcarga.bash" "$HOME/.local/share/bash-completion/completions/prevcarga"
    fi

    success "Bash completion criado"
}

#' Create zsh completion script
create_zsh_completion() {
    local completion_dir="$1"

    cat > "$completion_dir/_prevcarga" << 'COMPLETION'
#compdef prevcarga

_prevcarga() {
    local -a commands
    commands=(
        'forecast:Run load forecast'
        'backtest:Run backtesting'
        'train:Train models'
        'predict:Generate predictions'
        'status:Show system status'
        'areas:List area codes'
        'models:List available models'
        'help:Show help'
        'interactive:Start interactive shell'
    )

    local -a subsystems
    subsystems=('SECO' 'S' 'NE' 'N' 'SIN')

    _arguments \
        '1: :->command' \
        '2: :->subcommand' \
        '--help[Show help]' \
        '--version[Show version]' \
        '--demo[Run in demo mode]' \
        '--config[Config file]:config file:_files -g "*.yaml"'

    case $state in
        command)
            _describe 'command' commands
            ;;
        subcommand)
            case $words[2] in
                forecast|predict)
                    _describe 'subsystem' subsystems
                    ;;
            esac
            ;;
    esac
}

_prevcarga "$@"
COMPLETION

    # Install completion
    local zsh_completion_dir="$HOME/.zsh/completion"
    mkdir -p "$zsh_completion_dir"
    cp "$completion_dir/_prevcarga" "$zsh_completion_dir/_prevcarga"

    # Add to fpath if not already
    if ! grep -q "fpath.*\.zsh/completion" "$HOME/.zshrc" 2>/dev/null; then
        echo 'fpath=(~/.zsh/completion $fpath)' >> "$HOME/.zshrc"
        echo 'autoload -Uz compinit && compinit' >> "$HOME/.zshrc"
    fi

    success "Zsh completion criado"
}

#' Create fish completion script
create_fish_completion() {
    local completion_dir="$1"
    local fish_completion_dir="$HOME/.config/fish/completions"

    mkdir -p "$fish_completion_dir"

    cat > "$fish_completion_dir/prevcarga.fish" << 'COMPLETION'
# PrevCargaONS fish completion

complete -c prevcarga -f

# Commands
complete -c prevcarga -n __fish_use_subcommand -a forecast -d 'Run load forecast'
complete -c prevcarga -n __fish_use_subcommand -a backtest -d 'Run backtesting'
complete -c prevcarga -n __fish_use_subcommand -a train -d 'Train models'
complete -c prevcarga -n __fish_use_subcommand -a predict -d 'Generate predictions'
complete -c prevcarga -n __fish_use_subcommand -a status -d 'Show system status'
complete -c prevcarga -n __fish_use_subcommand -a areas -d 'List area codes'
complete -c prevcarga -n __fish_use_subcommand -a models -d 'List available models'
complete -c prevcarga -n __fish_use_subcommand -a help -d 'Show help'
complete -c prevcarga -n __fish_use_subcommand -a interactive -d 'Start interactive shell'

# Subsystems for forecast/predict
complete -c prevcarga -n '__fish_seen_subcommand_from forecast predict' -a 'SECO S NE N SIN'

# Options
complete -c prevcarga -l help -d 'Show help'
complete -c prevcarga -l version -d 'Show version'
complete -c prevcarga -l demo -d 'Run in demo mode'
complete -c prevcarga -l config -rF -d 'Config file'
COMPLETION

    success "Fish completion criado"
}
```

---

## Test Cases

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| TC-001 | Detect bash shell | shell=bash |
| TC-002 | Detect zsh shell | shell=zsh |
| TC-003 | Add to .bashrc | Entry added |
| TC-004 | Add to .zshrc | Entry added |
| TC-005 | Avoid duplicate | No duplicate entry |
| TC-006 | Fish shell support | config.fish updated |
| TC-007 | Bash completion | Tab completion works |
| TC-008 | Zsh completion | Tab completion works |
| TC-009 | Unknown shell | Manual instructions |
| TC-010 | PATH works | prevcarga found |

---

## Dependencies

- PC-086-11: Installer Script Structure
- PC-091-11: Launcher Script

---

## Definition of Done

- [ ] Shell detection working
- [ ] PATH added to bash
- [ ] PATH added to zsh
- [ ] PATH added to fish
- [ ] No duplicate entries
- [ ] Shell completions created
- [ ] Completions installed
- [ ] Manual instructions provided
- [ ] Code reviewed and approved
- [ ] Merged to develop branch

---

## Notes

- macOS defaults to zsh since Catalina
- Ubuntu defaults to bash
- Fish is optional but nice to have
- Completion improves UX significantly
