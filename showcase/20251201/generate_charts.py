#!/usr/bin/env python3
"""
Script para gerar gráficos da apresentação PrevCarga Unificado.

Gera 8 gráficos PNG a partir dos dados CSV de validação dos modelos.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Configurações globais
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['figure.facecolor'] = 'white'

# Paleta de cores ONS
COLORS = {
    'teal': '#025159',
    'dark_teal': '#0e5d5a',
    'orange': '#F39C12',
    'red': '#E74C3C',
    'green': '#27AE60',
    'blue': '#3498DB',
    'purple': '#9B59B6',
}

# Diretórios
DATA_DIR = Path(__file__).parent / 'data'
ASSETS_DIR = Path(__file__).parent / 'assets'


def load_csv(filename: str) -> pd.DataFrame:
    """Carrega CSV com separador ; e decimal ,"""
    df = pd.read_csv(
        DATA_DIR / filename,
        sep=';',
        decimal=','
    )
    return df


def save_chart(fig, filename: str, dpi: int = 150):
    """Salva gráfico como PNG"""
    filepath = ASSETS_DIR / filename
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Salvo: {filepath}")


def chart_1_mape_rf_vs_prevcarga():
    """Gráfico 1: MAPE Mensal RF vs PrevCarga (SECO) - Linhas"""
    print("Gerando gráfico 1: MAPE RF vs PrevCarga...")

    df = load_csv('consolidacao_geral_todos_meses_SECO.csv')

    # Separar por modelo
    rf = df[df['Modelo'] == 'Random Forest'].copy()
    pc = df[df['Modelo'] == 'PrevCarga'].copy()

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(rf['mes_teste'], rf['Mean_MAPE'],
            marker='o', linewidth=2, markersize=6,
            color=COLORS['teal'], label='Random Forest')
    ax.plot(pc['mes_teste'], pc['Mean_MAPE'],
            marker='s', linewidth=2, markersize=6,
            color=COLORS['orange'], label='PrevCarga')

    ax.set_xlabel('Mês de Teste')
    ax.set_ylabel('MAPE (%)')
    ax.set_title('Comparativo MAPE Mensal: Random Forest vs PrevCarga (SECO)')
    ax.legend(loc='upper right')
    ax.tick_params(axis='x', rotation=45)

    # Linha de referência (meta 5%)
    ax.axhline(y=5, color='gray', linestyle='--', alpha=0.5, label='Meta 5%')

    plt.tight_layout()
    save_chart(fig, 'grafico_mape_rf_vs_prevcarga_seco.png')


def chart_2_r2_comparativo():
    """Gráfico 2: R² Comparativo RF vs PrevCarga (SECO) - Linhas"""
    print("Gerando gráfico 2: R² Comparativo...")

    df = load_csv('consolidacao_geral_todos_meses_SECO.csv')

    rf = df[df['Modelo'] == 'Random Forest'].copy()
    pc = df[df['Modelo'] == 'PrevCarga'].copy()

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(rf['mes_teste'], rf['Mean_R2'],
            marker='o', linewidth=2, markersize=6,
            color=COLORS['teal'], label='Random Forest')
    ax.plot(pc['mes_teste'], pc['Mean_R2'],
            marker='s', linewidth=2, markersize=6,
            color=COLORS['orange'], label='PrevCarga')

    ax.set_xlabel('Mês de Teste')
    ax.set_ylabel('R²')
    ax.set_title('Coeficiente de Determinação (R²): Random Forest vs PrevCarga (SECO)')
    ax.legend(loc='lower right')
    ax.tick_params(axis='x', rotation=45)

    # Linha de referência (R²=0.8)
    ax.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

    plt.tight_layout()
    save_chart(fig, 'grafico_r2_comparativo.png')


def chart_3_mape_horizonte_seco():
    """Gráfico 3: MAPE por Horizonte - SECO - Barras"""
    print("Gerando gráfico 3: MAPE por Horizonte SECO...")

    df = load_csv('consolidacao_diaria_todos_meses_SECO.csv')

    # Média por horizonte (day)
    horizonte = df.groupby('day').agg({
        'MAPE_RF': 'mean',
        'MAPE_PrevCarga': 'mean'
    }).reset_index()

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(horizonte))
    width = 0.35

    bars1 = ax.bar(x - width/2, horizonte['MAPE_RF'], width,
                   label='Random Forest', color=COLORS['teal'])
    bars2 = ax.bar(x + width/2, horizonte['MAPE_PrevCarga'], width,
                   label='PrevCarga', color=COLORS['orange'])

    ax.set_xlabel('Horizonte de Previsão')
    ax.set_ylabel('MAPE Médio (%)')
    ax.set_title('MAPE por Horizonte de Previsão (SECO)')
    ax.set_xticks(x)
    ax.set_xticklabels([f'D+{d}' for d in horizonte['day']])
    ax.legend()

    # Valores nas barras
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    save_chart(fig, 'grafico_mape_horizonte_seco.png')


def chart_4_mape_horizonte_sul():
    """Gráfico 4: MAPE por Horizonte - SUL - Barras"""
    print("Gerando gráfico 4: MAPE por Horizonte SUL...")

    df = load_csv('consolidacao_diaria_todos_meses_SUL.csv')

    horizonte = df.groupby('day').agg({
        'MAPE_RF': 'mean',
        'MAPE_PrevCarga': 'mean'
    }).reset_index()

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(horizonte))
    width = 0.35

    bars1 = ax.bar(x - width/2, horizonte['MAPE_RF'], width,
                   label='Random Forest', color=COLORS['teal'])
    bars2 = ax.bar(x + width/2, horizonte['MAPE_PrevCarga'], width,
                   label='PrevCarga', color=COLORS['orange'])

    ax.set_xlabel('Horizonte de Previsão')
    ax.set_ylabel('MAPE Médio (%)')
    ax.set_title('MAPE por Horizonte de Previsão (SUL)')
    ax.set_xticks(x)
    ax.set_xticklabels([f'D+{d}' for d in horizonte['day']])
    ax.legend()

    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    save_chart(fig, 'grafico_mape_horizonte_sul.png')


def chart_5_heatmap_seco():
    """Gráfico 5: Heatmap MAPE por Mês × Horizonte (SECO)"""
    print("Gerando gráfico 5: Heatmap MAPE SECO...")

    df = load_csv('consolidacao_diaria_todos_meses_SECO.csv')

    # Pivot para criar matriz mês × horizonte
    pivot = df.pivot_table(
        values='MAPE_RF',
        index='day',
        columns='mes_teste',
        aggfunc='mean'
    )

    fig, ax = plt.subplots(figsize=(12, 5))

    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn_r',
                ax=ax, cbar_kws={'label': 'MAPE (%)'},
                linewidths=0.5, annot_kws={'size': 9})

    ax.set_xlabel('Mês de Teste')
    ax.set_ylabel('Horizonte')
    ax.set_yticklabels([f'D+{int(d)}' for d in pivot.index])
    ax.set_title('Heatmap: MAPE Random Forest por Mês e Horizonte (SECO)')

    plt.tight_layout()
    save_chart(fig, 'heatmap_mape_seco.png')


def chart_6_lgbm_seco():
    """Gráfico 6: LGBM SECO - MAPE Mensal - Barras"""
    print("Gerando gráfico 6: LGBM SECO...")

    df = load_csv('desvio_LGBM_mensal.csv')

    # Filtrar apenas LGBM padrão para SECO
    lgbm_seco = df[(df['cod_modeloprevisaocarga'] == 'LGBM') &
                   (df['cod_areacarga'] == 'SECO')].copy()
    lgbm_seco = lgbm_seco.sort_values('month')

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(lgbm_seco))
    width = 0.35

    bars1 = ax.bar(x - width/2, lgbm_seco['mape'], width,
                   label='MAPE Geral', color=COLORS['teal'])
    bars2 = ax.bar(x + width/2, lgbm_seco['mape_ponta'], width,
                   label='MAPE Ponta', color=COLORS['dark_teal'])

    ax.set_xlabel('Mês')
    ax.set_ylabel('MAPE (%)')
    ax.set_title('LGBM - MAPE Mensal (SECO): Geral vs Ponta')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{int(m):02d}' for m in lgbm_seco['month']])
    ax.legend()

    # Linha de meta
    ax.axhline(y=3, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    save_chart(fig, 'grafico_lgbm_seco.png')


def chart_7_lgbm_variantes_sul():
    """Gráfico 7: LGBM Variantes - SUL - Barras"""
    print("Gerando gráfico 7: LGBM Variantes SUL...")

    df = load_csv('desvio_LGBM_mensal.csv')

    # Filtrar SUL
    sul = df[df['cod_areacarga'] == 'S'].copy()

    # Pivot por modelo e mês
    pivot = sul.pivot_table(
        values='mape',
        index='month',
        columns='cod_modeloprevisaocarga',
        aggfunc='mean'
    ).reset_index()

    # Garantir que temos as colunas esperadas
    modelos = ['LGBM', 'LGBM_BU_verif', 'LGBM_BU_naive']
    modelos_existentes = [m for m in modelos if m in pivot.columns]

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(pivot))
    width = 0.25
    colors = [COLORS['teal'], COLORS['green'], COLORS['orange']]

    for i, modelo in enumerate(modelos_existentes):
        offset = (i - len(modelos_existentes)/2 + 0.5) * width
        bars = ax.bar(x + offset, pivot[modelo], width,
                      label=modelo, color=colors[i])

    ax.set_xlabel('Mês')
    ax.set_ylabel('MAPE (%)')
    ax.set_title('Comparativo LGBM Variantes (SUL)')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{int(m):02d}' for m in pivot['month']])
    ax.legend()

    plt.tight_layout()
    save_chart(fig, 'grafico_lgbm_variantes_sul.png')


def chart_8_mape_subsistemas():
    """Gráfico 8: MAPE PrevCarga por Subsistema - Linhas"""
    print("Gerando gráfico 8: MAPE por Subsistema...")

    df = load_csv('desvio_prevcarga_mensal.csv')

    fig, ax = plt.subplots(figsize=(12, 5))

    colors = {
        'SECO': COLORS['teal'],
        'S': COLORS['orange'],
        'N': COLORS['green'],
        'NE': COLORS['purple']
    }

    for subsistema in ['SECO', 'S', 'N', 'NE']:
        data = df[df['subsistema'] == subsistema].copy()
        data = data.sort_values('mes')
        ax.plot(data['mes'], data['MAPE'],
                marker='o', linewidth=2, markersize=4,
                color=colors[subsistema], label=subsistema)

    ax.set_xlabel('Mês')
    ax.set_ylabel('MAPE (%)')
    ax.set_title('MAPE PrevCarga por Subsistema (2024-2025)')
    ax.legend(loc='upper right')
    ax.tick_params(axis='x', rotation=45)

    # Linhas de referência
    ax.axhline(y=5, color='gray', linestyle='--', alpha=0.3)
    ax.axhline(y=3, color='gray', linestyle=':', alpha=0.3)

    plt.tight_layout()
    save_chart(fig, 'grafico_mape_subsistemas.png')


def main():
    """Gera todos os gráficos"""
    print("=" * 50)
    print("Gerando gráficos para apresentação PrevCarga")
    print("=" * 50)

    # Garantir que diretório assets existe
    ASSETS_DIR.mkdir(exist_ok=True)

    # Gerar cada gráfico
    chart_1_mape_rf_vs_prevcarga()
    chart_2_r2_comparativo()
    chart_3_mape_horizonte_seco()
    chart_4_mape_horizonte_sul()
    chart_5_heatmap_seco()
    chart_6_lgbm_seco()
    chart_7_lgbm_variantes_sul()
    chart_8_mape_subsistemas()

    print("=" * 50)
    print("Todos os gráficos gerados com sucesso!")
    print("=" * 50)


if __name__ == '__main__':
    main()
