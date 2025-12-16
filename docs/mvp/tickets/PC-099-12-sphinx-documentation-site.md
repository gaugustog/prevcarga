# PC-099-12: Sphinx Documentation Site

**Epic:** [EPIC-12: Documentation & Deploy](../epics/EPIC-12-documentation-deploy.md)
**Task Reference:** T-12.2
**Priority:** High
**Estimated Effort:** 1 day

---

## Summary

Initialize Sphinx documentation site with ReadTheDocs theme, MyST-Parser for Markdown support, and configure ReadTheDocs hosting.

---

## Acceptance Criteria

- [ ] Sphinx initialized in `docs/` directory
- [ ] `conf.py` configured with RTD theme
- [ ] MyST-Parser enabled for Markdown
- [ ] Landing page (`index.md`) created
- [ ] Custom CSS and logo in `_static/`
- [ ] ReadTheDocs configuration (`.readthedocs.yaml`)
- [ ] Documentation builds without errors

---

## Technical Specification

### Directory Structure

```
docs/
├── conf.py                    # Sphinx configuration
├── index.md                   # Landing page
├── requirements.txt           # Sphinx dependencies
├── _static/
│   ├── css/
│   │   └── custom.css        # Custom styling
│   ├── logo.png              # Project logo
│   └── favicon.ico           # Favicon
├── _templates/
│   └── layout.html           # Custom layout (optional)
├── getting-started/
│   └── .gitkeep
├── user-guide/
│   └── .gitkeep
├── plugin-guide/
│   └── .gitkeep
└── api/
    └── .gitkeep
```

### Sphinx Configuration

```python
# docs/conf.py
# ══════════════════════════════════════════════════════════════════════════════
# PrevCargaONS Documentation Configuration
# ══════════════════════════════════════════════════════════════════════════════

import os
import sys
from datetime import datetime

# ──────────────────────────────────────────────────────────────────────────────
# Project Information
# ──────────────────────────────────────────────────────────────────────────────

project = 'PrevCargaONS'
copyright = f'{datetime.now().year}, ONS - Operador Nacional do Sistema Elétrico'
author = 'ONS'
release = '1.0.0'
version = '1.0'

# ──────────────────────────────────────────────────────────────────────────────
# General Configuration
# ──────────────────────────────────────────────────────────────────────────────

extensions = [
    'myst_parser',              # Markdown support
    'sphinx.ext.autodoc',       # Auto-generate docs from docstrings
    'sphinx.ext.viewcode',      # Add source code links
    'sphinx.ext.intersphinx',   # Link to other documentation
    'sphinx.ext.napoleon',      # Google/NumPy docstring support
    'sphinx_copybutton',        # Copy button for code blocks
    'sphinx.ext.todo',          # Todo directives
]

# MyST-Parser settings (Markdown support)
myst_enable_extensions = [
    'colon_fence',      # ::: directive syntax
    'deflist',          # Definition lists
    'fieldlist',        # Field lists
    'tasklist',         # Task lists with checkboxes
    'substitution',     # Substitution definitions
    'attrs_inline',     # Inline attributes
]

myst_heading_anchors = 3  # Auto-generate anchors for h1-h3

# Source file suffixes
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# Master document
master_doc = 'index'

# Language
language = 'pt_BR'

# Exclude patterns
exclude_patterns = [
    '_build',
    'Thumbs.db',
    '.DS_Store',
    '**.ipynb_checkpoints',
    'mvp/**',  # Exclude MVP planning docs from build
]

# Templates path
templates_path = ['_templates']

# ──────────────────────────────────────────────────────────────────────────────
# HTML Output Options
# ──────────────────────────────────────────────────────────────────────────────

html_theme = 'sphinx_rtd_theme'

html_theme_options = {
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'both',
    'style_external_links': True,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False,
}

html_static_path = ['_static']
html_css_files = ['css/custom.css']
html_logo = '_static/logo.png'
html_favicon = '_static/favicon.ico'

html_context = {
    'display_github': True,
    'github_user': 'ons-br',
    'github_repo': 'prevcarga-R',
    'github_version': 'main',
    'conf_py_path': '/docs/',
}

# ──────────────────────────────────────────────────────────────────────────────
# Intersphinx Configuration
# ──────────────────────────────────────────────────────────────────────────────

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
}

# ──────────────────────────────────────────────────────────────────────────────
# Copy Button Configuration
# ──────────────────────────────────────────────────────────────────────────────

copybutton_prompt_text = r'>>> |\.\.\. |\$ |> '
copybutton_prompt_is_regexp = True
copybutton_remove_prompts = True

# ──────────────────────────────────────────────────────────────────────────────
# Todo Extension Configuration
# ──────────────────────────────────────────────────────────────────────────────

todo_include_todos = True
```

### Landing Page

```markdown
# docs/index.md

# PrevCargaONS

Sistema de Previsão de Carga Elétrica para o Operador Nacional do Sistema Elétrico.

```{toctree}
:maxdepth: 2
:caption: Início

getting-started/installation
getting-started/quickstart
```

```{toctree}
:maxdepth: 2
:caption: Guia do Usuário

user-guide/configuration
user-guide/cli-reference
user-guide/storage-backends
```

```{toctree}
:maxdepth: 2
:caption: Desenvolvimento de Plugins

plugin-guide/overview
plugin-guide/feature-plugins
plugin-guide/model-plugins
plugin-guide/combiner-plugins
plugin-guide/reconciler-plugins
```

```{toctree}
:maxdepth: 2
:caption: Referência API

api/index
```

## Visão Geral

PrevCargaONS é um sistema modular de previsão de carga elétrica desenvolvido em R,
projetado para o Operador Nacional do Sistema Elétrico (ONS).

### Características

- **Previsão Multi-Horizonte**: D+0 a D+8
- **Múltiplas Áreas**: Suporte a todas as áreas e subsistemas do SIN
- **Arquitetura Modular**: Plugins para modelos, features, combinação e reconciliação
- **Reconciliação Hierárquica**: Consistência entre níveis da hierarquia
- **CLI Completa**: Interface de linha de comando para operação e automação

### Quick Start

:::bash
# Instalação
curl -fsSL https://ons.org.br/prevcarga/install.sh | bash

# Previsão
prevcarga predict --area RJ --horizon 1

# Modo interativo
prevcarga interactive
:::

## Índices e Tabelas

- {ref}`genindex`
- {ref}`search`
```

### Requirements File

```text
# docs/requirements.txt
sphinx>=7.0
sphinx-rtd-theme>=2.0
myst-parser>=2.0
sphinx-copybutton>=0.5
```

### Custom CSS

```css
/* docs/_static/css/custom.css */

/* ══════════════════════════════════════════════════════════════════════════════
   PrevCargaONS Documentation Custom Styles
   ══════════════════════════════════════════════════════════════════════════════ */

/* Brand Colors */
:root {
    --prevcarga-primary: #1a5276;
    --prevcarga-secondary: #2e86ab;
    --prevcarga-accent: #f39c12;
    --prevcarga-success: #27ae60;
    --prevcarga-warning: #f39c12;
    --prevcarga-error: #e74c3c;
}

/* Sidebar styling */
.wy-side-nav-search {
    background-color: var(--prevcarga-primary);
}

.wy-side-nav-search a {
    color: white;
}

/* Code block styling */
.highlight {
    background: #f8f9fa !important;
}

div.highlight pre {
    padding: 12px;
    border-radius: 4px;
}

/* Copy button positioning */
.copybtn {
    top: 0.5em !important;
    right: 0.5em !important;
}

/* Task list styling */
.contains-task-list {
    list-style-type: none;
    padding-left: 0;
}

.task-list-item input[type="checkbox"] {
    margin-right: 0.5em;
}

/* Admonition styling */
.admonition {
    border-radius: 4px;
}

.admonition.note {
    border-left-color: var(--prevcarga-secondary);
}

.admonition.warning {
    border-left-color: var(--prevcarga-warning);
}

.admonition.danger {
    border-left-color: var(--prevcarga-error);
}

/* Table styling */
table.docutils {
    width: 100%;
    border-collapse: collapse;
}

table.docutils th {
    background-color: var(--prevcarga-primary);
    color: white;
    padding: 8px 12px;
}

table.docutils td {
    padding: 8px 12px;
    border-bottom: 1px solid #ddd;
}

/* Version badge */
.rst-current-version {
    background-color: var(--prevcarga-primary) !important;
}

/* Responsive adjustments */
@media screen and (max-width: 768px) {
    .wy-nav-content {
        padding: 1em;
    }
}
```

### ReadTheDocs Configuration

```yaml
# .readthedocs.yaml
version: 2

build:
  os: ubuntu-22.04
  tools:
    python: "3.11"

sphinx:
  configuration: docs/conf.py
  builder: html
  fail_on_warning: false

python:
  install:
    - requirements: docs/requirements.txt

formats:
  - pdf
  - epub
```

### Build Commands

```bash
# Install dependencies
pip install -r docs/requirements.txt

# Build HTML
sphinx-build -b html docs/ docs/_build/html

# Build with auto-reload (development)
sphinx-autobuild docs/ docs/_build/html

# Check for broken links
sphinx-build -b linkcheck docs/ docs/_build/linkcheck

# Build PDF (requires LaTeX)
sphinx-build -b latex docs/ docs/_build/latex
cd docs/_build/latex && make
```

---

## Test Cases

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| TC-001 | Sphinx builds | No errors |
| TC-002 | All pages render | HTML generated |
| TC-003 | TOC correct | Navigation works |
| TC-004 | Code blocks | Syntax highlighted |
| TC-005 | Copy buttons work | Copy to clipboard |
| TC-006 | Search works | Results returned |
| TC-007 | RTD theme applied | Styling correct |
| TC-008 | Custom CSS loads | Styles applied |
| TC-009 | Logo displays | Image renders |
| TC-010 | Mobile responsive | Readable on phone |

---

## Dependencies

- PC-098-12: Complete README (for consistent content)

---

## Definition of Done

- [ ] Sphinx initialized
- [ ] conf.py configured
- [ ] index.md created
- [ ] Custom CSS added
- [ ] ReadTheDocs config created
- [ ] Build passes without errors
- [ ] Code reviewed and approved
- [ ] Merged to develop branch

---

## Notes

- Use MyST-Parser for Markdown (more familiar than RST)
- RTD theme is standard and well-supported
- Enable copy buttons for code blocks
- Support both HTML and PDF output
- Configure for Portuguese language
