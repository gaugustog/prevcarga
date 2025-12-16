#!/usr/bin/env python3
"""
Gerador de diagramas PNG para a apresentação PrevCarga Unificado.
Usa matplotlib para criar diagramas visuais profissionais.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np
import os

# Cores consistentes com a apresentação
COLORS = {
    'primary': '#025159',      # Verde escuro (títulos)
    'secondary': '#0e5d5a',    # Verde médio
    'accent': '#27AE60',       # Verde claro (destaque)
    'warning': '#F39C12',      # Laranja (alertas)
    'danger': '#E74C3C',       # Vermelho (crítico)
    'background': '#F8F9FA',   # Cinza claro
    'text': '#2C3E50',         # Texto escuro
    'white': '#FFFFFF',
    'light_blue': '#3498DB',
    'purple': '#9B59B6',
}

OUTPUT_DIR = 'assets'

def setup_figure(figsize=(12, 6)):
    """Configura figura com fundo branco."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    return fig, ax

def add_box(ax, x, y, width, height, text, color=None, text_color='white', fontsize=11):
    """Adiciona uma caixa com texto."""
    if color is None:
        color = COLORS['primary']
    box = FancyBboxPatch((x, y), width, height,
                         boxstyle="round,pad=0.05,rounding_size=0.2",
                         facecolor=color, edgecolor='none')
    ax.add_patch(box)
    ax.text(x + width/2, y + height/2, text,
            ha='center', va='center', fontsize=fontsize,
            color=text_color, fontweight='bold', wrap=True)
    return box

def add_arrow(ax, start, end, color=None):
    """Adiciona uma seta entre dois pontos."""
    if color is None:
        color = COLORS['text']
    arrow = FancyArrowPatch(start, end,
                            arrowstyle='-|>',
                            mutation_scale=15,
                            color=color, linewidth=2)
    ax.add_patch(arrow)

def add_diamond(ax, x, y, size, text, color=None):
    """Adiciona um losango (decisão)."""
    if color is None:
        color = COLORS['warning']
    diamond = plt.Polygon([(x, y-size), (x+size, y), (x, y+size), (x-size, y)],
                         facecolor=color, edgecolor='none')
    ax.add_patch(diamond)
    ax.text(x, y, text, ha='center', va='center', fontsize=10,
            color='white', fontweight='bold')

# =============================================================================
# DIAGRAMA 1: Sistema de Plugins
# =============================================================================
def diagrama_plugins():
    """Fluxo @register -> Registry -> Factory -> Plugin Ativo"""
    fig, ax = setup_figure((14, 7))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)

    # Boxes do fluxo principal
    add_box(ax, 0.5, 3.5, 2.5, 1.2, '@register_*\ndecorator', COLORS['secondary'])
    add_box(ax, 4, 3.5, 2.5, 1.2, 'Plugin\nRegistry', COLORS['primary'])
    add_box(ax, 7.5, 3.5, 2.5, 1.2, 'config.yaml', COLORS['warning'])
    add_box(ax, 11, 3.5, 2.5, 1.2, 'Factory\ncreate()', COLORS['accent'])

    # Setas
    add_arrow(ax, (3, 4.1), (4, 4.1))
    add_arrow(ax, (6.5, 4.1), (7.5, 4.1))
    add_arrow(ax, (10, 4.1), (11, 4.1))

    # Tipos de plugins (abaixo)
    ax.text(7, 6.5, 'Tipos de Plugins', ha='center', fontsize=14,
            fontweight='bold', color=COLORS['text'])

    plugin_types = [
        ('Feature', COLORS['light_blue'], 1.5),
        ('Model', COLORS['purple'], 5),
        ('Combiner', COLORS['accent'], 8.5),
        ('Reconciler', COLORS['danger'], 12),
    ]

    for name, color, x in plugin_types:
        add_box(ax, x, 5.5, 2.3, 0.8, name, color, fontsize=10)

    # Resultado final
    add_box(ax, 5.5, 1, 3, 1.2, 'Plugin Ativo', COLORS['accent'], fontsize=12)
    add_arrow(ax, (12.25, 3.5), (12.25, 2.5))
    add_arrow(ax, (12.25, 2.5), (8.5, 1.6))

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/diagrama_plugins.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("  Gerado: diagrama_plugins.png")

# =============================================================================
# DIAGRAMA 2: Storage Híbrido
# =============================================================================
def diagrama_storage():
    """Bifurcação S3/Local com mesma interface."""
    fig, ax = setup_figure((12, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)

    # StorageFactory no topo
    add_box(ax, 4.5, 5.5, 3, 1, 'StorageFactory', COLORS['primary'])

    # Decisão
    add_diamond(ax, 6, 4, 0.6, '?', COLORS['warning'])
    add_arrow(ax, (6, 5.5), (6, 4.6))

    # Labels
    ax.text(3.5, 4, 'Produção', ha='center', fontsize=10, color=COLORS['text'])
    ax.text(8.5, 4, 'Dev', ha='center', fontsize=10, color=COLORS['text'])

    # Backends
    add_box(ax, 1, 2, 3, 1.2, 'S3 Backend', COLORS['light_blue'])
    add_box(ax, 8, 2, 3, 1.2, 'Local Backend', COLORS['accent'])

    # Setas da decisão
    add_arrow(ax, (5.4, 4), (4, 3.2))
    add_arrow(ax, (6.6, 4), (8, 3.2))

    # Interface comum
    add_box(ax, 3.5, 0.3, 5, 1, 'Mesma Interface\n(IStorageBackend)', COLORS['secondary'])
    add_arrow(ax, (2.5, 2), (4.5, 1.3))
    add_arrow(ax, (9.5, 2), (7.5, 1.3))

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/diagrama_storage.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("  Gerado: diagrama_storage.png")

# =============================================================================
# DIAGRAMA 3: Estrutura de Dados
# =============================================================================
def diagrama_estrutura_dados():
    """Árvore de pastas visual."""
    fig, ax = setup_figure((10, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)

    # Pasta raiz
    add_box(ax, 3.5, 5.8, 3, 0.8, 'storage/', COLORS['primary'], fontsize=12)

    # Subpastas
    folders = [
        ('raw_data/', 'Histórico completo', COLORS['light_blue'], 0.5, 4.2),
        ('features/', 'Cache 30 dias', COLORS['accent'], 0.5, 3.0),
        ('models/', 'v1.0.0 (versionado)', COLORS['purple'], 0.5, 1.8),
        ('results/', 'Histórico completo', COLORS['secondary'], 5.5, 4.2),
        ('cache/', '7 dias', COLORS['warning'], 5.5, 3.0),
    ]

    for name, desc, color, x, y in folders:
        add_box(ax, x, y, 2, 0.8, name, color, fontsize=10)
        ax.text(x + 2.2, y + 0.4, desc, ha='left', va='center',
                fontsize=9, color=COLORS['text'])

    # Linhas de conexão
    for _, _, _, x, y in folders:
        ax.plot([5, x + 1], [5.8, y + 0.8], color=COLORS['text'], linewidth=1.5)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/diagrama_estrutura_dados.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("  Gerado: diagrama_estrutura_dados.png")

# =============================================================================
# DIAGRAMA 4: Combinação de Modelos
# =============================================================================
def diagrama_combinacao():
    """Modelos -> Combiner -> Previsão Final."""
    fig, ax = setup_figure((14, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)

    # Modelos
    models = [
        ('LGBM', COLORS['accent'], 4.5),
        ('RF', COLORS['light_blue'], 3),
        ('PrevCarga', COLORS['purple'], 1.5),
    ]

    ax.text(1.5, 5.5, 'Modelos', ha='center', fontsize=12,
            fontweight='bold', color=COLORS['text'])

    for name, color, y in models:
        add_box(ax, 0.5, y, 2, 0.8, name, color, fontsize=11)
        add_arrow(ax, (2.5, y + 0.4), (4.5, 3))

    # Combiner (diamante grande)
    add_diamond(ax, 5.5, 3, 1, 'Combiner', COLORS['warning'])

    # Estratégias
    ax.text(5.5, 1, 'Voting | Stacking | Markov', ha='center',
            fontsize=10, color=COLORS['text'], style='italic')

    # Seta para resultado
    add_arrow(ax, (6.5, 3), (8, 3))

    # Resultado
    add_box(ax, 8, 2.5, 2.5, 1, 'Previsão\nCombinada', COLORS['accent'], fontsize=11)

    # Reconciliação
    add_arrow(ax, (10.5, 3), (11.5, 3))
    add_box(ax, 11.5, 2.5, 2, 1, 'Reconciliação', COLORS['secondary'], fontsize=10)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/diagrama_combinacao.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("  Gerado: diagrama_combinacao.png")

# =============================================================================
# DIAGRAMA 5: Markov Chain Combiner
# =============================================================================
def diagrama_markov():
    """Estados Manhã/Tarde/Noite com transições."""
    fig, ax = setup_figure((12, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)

    # Título
    ax.text(6, 6.5, 'Markov Chain Combiner', ha='center', fontsize=14,
            fontweight='bold', color=COLORS['text'])

    # Estados como círculos grandes
    states = [
        ('Manhã\nLGBM 70%', COLORS['warning'], 2, 4),
        ('Tarde\nRF 60%', COLORS['light_blue'], 6, 4),
        ('Noite\nLGBM 65%', COLORS['purple'], 10, 4),
    ]

    for label, color, x, y in states:
        circle = Circle((x, y), 1.3, facecolor=color, edgecolor='none')
        ax.add_patch(circle)
        ax.text(x, y, label, ha='center', va='center', fontsize=10,
                color='white', fontweight='bold')

    # Transições (setas curvas)
    # Manhã -> Tarde
    ax.annotate('', xy=(4.7, 4.3), xytext=(3.3, 4.3),
                arrowprops=dict(arrowstyle='-|>', color=COLORS['text'], lw=2))
    # Tarde -> Noite
    ax.annotate('', xy=(8.7, 4.3), xytext=(7.3, 4.3),
                arrowprops=dict(arrowstyle='-|>', color=COLORS['text'], lw=2))
    # Noite -> Manhã (curva por baixo)
    ax.annotate('', xy=(2, 2.7), xytext=(10, 2.7),
                arrowprops=dict(arrowstyle='-|>', color=COLORS['text'], lw=2,
                               connectionstyle='arc3,rad=0.3'))

    # Legenda
    ax.text(6, 1.2, 'Pesos dinâmicos por contexto:\nhora do dia, dia da semana, feriados',
            ha='center', fontsize=10, color=COLORS['text'], style='italic')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/diagrama_markov.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("  Gerado: diagrama_markov.png")

# =============================================================================
# DIAGRAMA 6: Reconciliação Hierárquica
# =============================================================================
def diagrama_reconciliacao():
    """Hierarquia SIN -> Subsistemas -> Áreas."""
    fig, ax = setup_figure((12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)

    # SIN no topo
    add_box(ax, 4.5, 6.8, 3, 0.8, 'SIN Nacional', COLORS['primary'], fontsize=12)

    # Subsistemas
    subsistemas = [
        ('SECO', 1.5, 5),
        ('S', 4, 5),
        ('NE', 6.5, 5),
        ('N', 9, 5),
    ]

    for name, x, y in subsistemas:
        add_box(ax, x, y, 1.8, 0.7, name, COLORS['secondary'], fontsize=11)
        ax.plot([6, x + 0.9], [6.8, y + 0.7], color=COLORS['text'], linewidth=1.5)

    # Áreas
    add_box(ax, 4, 3.2, 4, 0.8, '17 Áreas de Carga', COLORS['light_blue'], fontsize=11)
    for name, x, y in subsistemas:
        ax.plot([x + 0.9, 6], [5, 4], color=COLORS['text'], linewidth=1.5)

    # Perdas
    add_box(ax, 4.5, 1.5, 3, 0.8, 'Perdas (PE*)', COLORS['warning'], fontsize=11)
    ax.plot([6, 6], [3.2, 2.3], color=COLORS['text'], linewidth=1.5)

    # MinT
    ax.text(10.5, 3.5, 'MinT\nReconciliation', ha='center', fontsize=10,
            color=COLORS['accent'], fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor=COLORS['accent']))

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/diagrama_reconciliacao.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("  Gerado: diagrama_reconciliacao.png")

# =============================================================================
# DIAGRAMA 7: Auto-Avaliação
# =============================================================================
def diagrama_auto_avaliacao():
    """Ciclo com semáforos de drift."""
    fig, ax = setup_figure((14, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)

    # Fluxo principal
    add_box(ax, 0.5, 2.5, 2, 1, 'Previsão', COLORS['primary'])
    add_arrow(ax, (2.5, 3), (3.5, 3))

    add_box(ax, 3.5, 2.5, 2, 1, 'Verificado', COLORS['secondary'])
    add_arrow(ax, (5.5, 3), (6.5, 3))

    add_box(ax, 6.5, 2.5, 2, 1, 'Calcular\nMAPE', COLORS['light_blue'])
    add_arrow(ax, (8.5, 3), (9.5, 3))

    # Decisão drift
    add_diamond(ax, 10.5, 3, 0.7, 'Drift?', COLORS['warning'])

    # Semáforos
    # Verde - OK
    circle_ok = Circle((12.5, 5), 0.4, facecolor='#27AE60', edgecolor='none')
    ax.add_patch(circle_ok)
    ax.text(12.5, 5, 'OK', ha='center', va='center', fontsize=9, color='white', fontweight='bold')
    ax.text(12.5, 4.3, '<5%', ha='center', fontsize=9, color=COLORS['text'])
    ax.annotate('', xy=(12.5, 4.6), xytext=(11, 3.5),
                arrowprops=dict(arrowstyle='-|>', color='#27AE60', lw=2))

    # Amarelo - Alerta
    circle_warn = Circle((12.5, 3), 0.4, facecolor='#F39C12', edgecolor='none')
    ax.add_patch(circle_warn)
    ax.text(12.5, 3, '!', ha='center', va='center', fontsize=12, color='white', fontweight='bold')
    ax.text(13.2, 3, '+10%\nAlerta', ha='left', fontsize=9, color=COLORS['text'])
    ax.annotate('', xy=(12.1, 3), xytext=(11.2, 3),
                arrowprops=dict(arrowstyle='-|>', color='#F39C12', lw=2))

    # Vermelho - Retrain
    circle_danger = Circle((12.5, 1), 0.4, facecolor='#E74C3C', edgecolor='none')
    ax.add_patch(circle_danger)
    ax.text(12.5, 1, 'R', ha='center', va='center', fontsize=10, color='white', fontweight='bold')
    ax.text(13.2, 1, '+20%\nRetrain', ha='left', fontsize=9, color=COLORS['text'])
    ax.annotate('', xy=(12.5, 1.4), xytext=(11, 2.5),
                arrowprops=dict(arrowstyle='-|>', color='#E74C3C', lw=2))

    # Loop de volta
    ax.annotate('', xy=(1.5, 2.5), xytext=(12, 5.3),
                arrowprops=dict(arrowstyle='-|>', color='#27AE60', lw=2,
                               connectionstyle='arc3,rad=-0.2'))

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/diagrama_auto_avaliacao.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("  Gerado: diagrama_auto_avaliacao.png")

# =============================================================================
# DIAGRAMA 8: Ciclo de Retraining
# =============================================================================
def diagrama_retraining():
    """Ciclo semanal de retraining."""
    fig, ax = setup_figure((12, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)

    # Ciclo em círculo
    center_x, center_y = 5, 3.5
    radius = 2.5

    # Etapas do ciclo
    angles = [90, 0, -90, -180]
    labels = [
        ('Domingo\nAvaliar', COLORS['primary']),
        ('MAPE > 5%?', COLORS['warning']),
        ('Retrain\n(se necessário)', COLORS['danger']),
        ('Validar\nDeploy', COLORS['accent']),
    ]

    positions = []
    for angle, (label, color) in zip(angles, labels):
        rad = np.radians(angle)
        x = center_x + radius * np.cos(rad)
        y = center_y + radius * np.sin(rad)
        positions.append((x, y))

        if 'MAPE' in label:
            add_diamond(ax, x, y, 0.8, label, color)
        else:
            add_box(ax, x - 1, y - 0.4, 2, 0.8, label, color, fontsize=9)

    # Setas entre etapas
    for i in range(len(positions)):
        start = positions[i]
        end = positions[(i + 1) % len(positions)]
        # Ajustar pontos para não sobrepor boxes
        sx = start[0] + (0.8 if start[0] < end[0] else -0.8 if start[0] > end[0] else 0)
        sy = start[1] + (0.5 if start[1] < end[1] else -0.5 if start[1] > end[1] else 0)
        ex = end[0] + (-0.8 if start[0] < end[0] else 0.8 if start[0] > end[0] else 0)
        ey = end[1] + (-0.5 if start[1] < end[1] else 0.5 if start[1] > end[1] else 0)
        add_arrow(ax, (sx, sy), (ex, ey))

    # Notas
    ax.text(10, 5.5, 'Regras:', ha='left', fontsize=10, fontweight='bold', color=COLORS['text'])
    ax.text(10, 4.8, '- Carnaval/Natal:\n  adiamento', ha='left', fontsize=9, color=COLORS['text'])
    ax.text(10, 3.5, '- Cooldown:\n  2 semanas', ha='left', fontsize=9, color=COLORS['text'])

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/diagrama_retraining.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("  Gerado: diagrama_retraining.png")

# =============================================================================
# DIAGRAMA 9: Pipeline RF
# =============================================================================
def diagrama_pipeline_rf():
    """Pipeline horizontal de 7 estágios."""
    fig, ax = setup_figure((16, 5))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 5)

    # Título
    ax.text(8, 4.5, 'Pipeline Random Forest (432 modelos)', ha='center',
            fontsize=14, fontweight='bold', color=COLORS['text'])

    # Estágios
    stages = [
        ('Import', COLORS['primary']),
        ('Features', COLORS['secondary']),
        ('Train', COLORS['accent']),
        ('Inference', COLORS['light_blue']),
        ('Predict', COLORS['purple']),
        ('Eval', COLORS['warning']),
        ('Retrain', COLORS['danger']),
    ]

    x_start = 0.5
    width = 1.8
    gap = 0.3

    for i, (name, color) in enumerate(stages):
        x = x_start + i * (width + gap)
        add_box(ax, x, 2, width, 1, name, color, fontsize=10)
        if i < len(stages) - 1:
            add_arrow(ax, (x + width, 2.5), (x + width + gap, 2.5))

    # Loop de retrain
    ax.annotate('', xy=(x_start + 2 * (width + gap) + width/2, 2),
                xytext=(x_start + 6 * (width + gap) + width/2, 2),
                arrowprops=dict(arrowstyle='-|>', color=COLORS['danger'], lw=2,
                               connectionstyle='arc3,rad=0.4'))
    ax.text(8, 0.8, 'se drift detectado', ha='center', fontsize=9,
            color=COLORS['danger'], style='italic')

    # Info adicional
    ax.text(8, 3.5, '9 dias x 48 períodos = 432 modelos por subsistema',
            ha='center', fontsize=10, color=COLORS['text'])

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/diagrama_pipeline_rf.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("  Gerado: diagrama_pipeline_rf.png")

# =============================================================================
# DIAGRAMA 10: Pipeline LGBM
# =============================================================================
def diagrama_pipeline_lgbm():
    """Pipeline com destaque para BLF."""
    fig, ax = setup_figure((16, 5))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 5)

    # Título
    ax.text(8, 4.5, 'Pipeline LGBM (BLF)', ha='center',
            fontsize=14, fontweight='bold', color=COLORS['text'])

    # Estágios
    stages = [
        ('Import', COLORS['primary'], False),
        ('LOESS +\nWavelet', COLORS['secondary'], False),
        ('Train', COLORS['accent'], False),
        ('BLF\n08:00', COLORS['warning'], True),  # Destaque
        ('Decay', COLORS['light_blue'], False),
        ('Predict', COLORS['purple'], False),
    ]

    x_start = 1
    width = 2
    gap = 0.4

    for i, (name, color, highlight) in enumerate(stages):
        x = x_start + i * (width + gap)
        height = 1.3 if highlight else 1
        y = 2 - (0.15 if highlight else 0)

        if highlight:
            # Box com borda
            box = FancyBboxPatch((x, y), width, height,
                                boxstyle="round,pad=0.05,rounding_size=0.2",
                                facecolor=color, edgecolor=COLORS['danger'], linewidth=3)
            ax.add_patch(box)
            ax.text(x + width/2, y + height/2, name,
                    ha='center', va='center', fontsize=11,
                    color='white', fontweight='bold')
        else:
            add_box(ax, x, y, width, height, name, color, fontsize=10)

        if i < len(stages) - 1:
            add_arrow(ax, (x + width, 2.5), (x + width + gap, 2.5))

    # Legenda BLF
    ax.text(8, 0.7, 'BLF = Baseline Load Forecast (ancora a previsão no início do dia)',
            ha='center', fontsize=10, color=COLORS['text'], style='italic')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/diagrama_pipeline_lgbm.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("  Gerado: diagrama_pipeline_lgbm.png")

# =============================================================================
# DIAGRAMA 11: Por que LGBM é estável
# =============================================================================
def diagrama_lgbm_estavel():
    """Diagrama causa-efeito da estabilidade."""
    fig, ax = setup_figure((14, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)

    # Título
    ax.text(7, 5.5, 'Por que o LGBM é mais estável?', ha='center',
            fontsize=14, fontweight='bold', color=COLORS['text'])

    # Causas (esquerda)
    causes = [
        ('LOESS\nTemperatura', COLORS['light_blue'], 4.2),
        ('Wavelets\nHaar', COLORS['purple'], 2.8),
        ('BLF\n08:00', COLORS['warning'], 1.4),
        ('Decaimento\nExponencial', COLORS['secondary'], 0),
    ]

    ax.text(1.5, 4.8, 'Features', ha='center', fontsize=11,
            fontweight='bold', color=COLORS['text'])
    ax.text(1.5, 2.2, 'Correção', ha='center', fontsize=11,
            fontweight='bold', color=COLORS['text'])

    for name, color, y in causes:
        add_box(ax, 0.5, y, 2, 0.9, name, color, fontsize=9)
        add_arrow(ax, (2.5, y + 0.45), (4.5, 2.5))

    # Resultado central
    circle = Circle((5.5, 2.5), 1.2, facecolor=COLORS['accent'], edgecolor='none')
    ax.add_patch(circle)
    ax.text(5.5, 2.5, 'Previsão\nEstável', ha='center', va='center',
            fontsize=11, color='white', fontweight='bold')

    # Resultado final (direita)
    add_arrow(ax, (6.7, 2.5), (8, 2.5))
    add_box(ax, 8, 2, 5, 1, 'Adapta rapidamente às\nvariações de temperatura',
            COLORS['primary'], fontsize=10)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/diagrama_lgbm_estavel.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("  Gerado: diagrama_lgbm_estavel.png")

# =============================================================================
# DIAGRAMA 12: Arquitetura do Sistema
# =============================================================================
def diagrama_arquitetura():
    """Stack vertical de camadas."""
    fig, ax = setup_figure((10, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)

    # Camadas de cima para baixo
    layers = [
        ('CLI (Click)', COLORS['primary'], 6.5),
        ('Orchestrator', COLORS['secondary'], 5),
        ('Data | Feature | Model', COLORS['light_blue'], 3.5),
        ('Combiner | Reconciler | Evaluator', COLORS['purple'], 2),
        ('Storage Backend', COLORS['accent'], 0.5),
    ]

    for name, color, y in layers:
        add_box(ax, 2, y, 6, 1, name, color, fontsize=11)

    # Setas entre camadas
    for i in range(len(layers) - 1):
        y_top = layers[i][2]
        y_bottom = layers[i + 1][2] + 1
        ax.annotate('', xy=(5, y_bottom), xytext=(5, y_top),
                    arrowprops=dict(arrowstyle='-|>', color=COLORS['text'], lw=2))

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/diagrama_arquitetura.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("  Gerado: diagrama_arquitetura.png")

# =============================================================================
# DIAGRAMA 13: Hierarquia SIN
# =============================================================================
def diagrama_hierarquia_sin():
    """Mapa visual dos subsistemas do Brasil."""
    fig, ax = setup_figure((14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)

    # Título
    ax.text(7, 7.5, 'Sistema Interligado Nacional (SIN)', ha='center',
            fontsize=14, fontweight='bold', color=COLORS['text'])

    # Representação simplificada do Brasil como retângulos
    # Norte
    add_box(ax, 3, 5.5, 4, 1.2, 'N (Norte)\nAM, PA, MA, TO, RR, AP',
            COLORS['accent'], fontsize=9)

    # Nordeste
    add_box(ax, 8, 5.5, 4, 1.2, 'NE (Nordeste)\nALPE, PBRN, BASE,\nCE, PI, BAOE',
            COLORS['warning'], fontsize=9)

    # Sudeste/Centro-Oeste
    add_box(ax, 3, 3, 6, 1.8, 'SECO (Sudeste/CO)\nRJ, SP, MG, ES, MT, MS,\nAC, RO, DF, GO',
            COLORS['primary'], fontsize=10)

    # Sul
    add_box(ax, 10, 3, 3, 1.8, 'S (Sul)\nPR, SC, RS',
            COLORS['light_blue'], fontsize=10)

    # Conexões
    ax.plot([5, 5], [5.5, 4.8], color=COLORS['text'], linewidth=2)
    ax.plot([10, 9], [5.5, 4.8], color=COLORS['text'], linewidth=2)
    ax.plot([9, 10], [3.9, 3.9], color=COLORS['text'], linewidth=2)

    # Legenda
    ax.text(7, 1, 'Perdas: PESE (SECO), PES (S), PENE (NE), PEN (N)',
            ha='center', fontsize=10, color=COLORS['text'], style='italic')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/diagrama_hierarquia_sin.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("  Gerado: diagrama_hierarquia_sin.png")


# =============================================================================
# MAIN
# =============================================================================
def main():
    """Gera todos os diagramas."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Gerando diagramas PNG...")

    diagrama_plugins()
    diagrama_storage()
    diagrama_estrutura_dados()
    diagrama_combinacao()
    diagrama_markov()
    diagrama_reconciliacao()
    diagrama_auto_avaliacao()
    diagrama_retraining()
    diagrama_pipeline_rf()
    diagrama_pipeline_lgbm()
    diagrama_lgbm_estavel()
    diagrama_arquitetura()
    diagrama_hierarquia_sin()

    print(f"\nTodos os 13 diagramas foram gerados em '{OUTPUT_DIR}/'")


if __name__ == '__main__':
    main()
