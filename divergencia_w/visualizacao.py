# -*- coding: utf-8 -*-
"""
Divergência W - Módulo de Visualização
Autor: Luiz Tiago Wilcke
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import os

# Configuração de estilo
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

DIRETORIO_GRAFICOS = "graficos"


def criar_diretorio_graficos():
    """Cria o diretório para salvar gráficos se não existir."""
    if not os.path.exists(DIRETORIO_GRAFICOS):
        os.makedirs(DIRETORIO_GRAFICOS)


def plotar_distribuicoes(x: np.ndarray, p: np.ndarray, q: np.ndarray,
                         titulo: str = "Comparação de Distribuições",
                         salvar: bool = True) -> plt.Figure:
    """Plota duas distribuições sobrepostas."""
    criar_diretorio_graficos()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.fill_between(x, p, alpha=0.5, label='P(x)', color='#3498db')
    ax.fill_between(x, q, alpha=0.5, label='Q(x)', color='#e74c3c')
    ax.plot(x, p, color='#2980b9', linewidth=2)
    ax.plot(x, q, color='#c0392b', linewidth=2)
    
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('Probabilidade', fontsize=12)
    ax.set_title(titulo, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    if salvar:
        plt.savefig(f'{DIRETORIO_GRAFICOS}/distribuicoes.png', dpi=150)
    return fig


def plotar_comparacao_divergencias(resultados: Dict[str, Dict],
                                   salvar: bool = True) -> plt.Figure:
    """Plota comparação de divergências em diferentes cenários."""
    criar_diretorio_graficos()
    
    cenarios = list(resultados.keys())
    metodos = ['Divergência W', 'KL', 'Jensen-Shannon', 'Hellinger']
    
    n_cenarios = len(cenarios)
    x = np.arange(n_cenarios)
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(14, 7))
    cores = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']
    
    for i, metodo in enumerate(metodos):
        valores = []
        for cenario in cenarios:
            v = resultados[cenario].get(metodo, np.nan)
            valores.append(v if np.isfinite(v) else 0)
        
        bars = ax.bar(x + i * width, valores, width, label=metodo, color=cores[i], alpha=0.8)
    
    ax.set_xlabel('Cenário', fontsize=12)
    ax.set_ylabel('Valor da Divergência', fontsize=12)
    ax.set_title('Comparação de Divergências por Cenário', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([c[:20] + '...' if len(c) > 20 else c for c in cenarios], rotation=45, ha='right')
    ax.legend()
    
    plt.tight_layout()
    if salvar:
        plt.savefig(f'{DIRETORIO_GRAFICOS}/comparacao_divergencias.png', dpi=150)
    return fig


def plotar_benchmark_tempo(resultados: Dict[str, Dict], salvar: bool = True) -> plt.Figure:
    """Plota benchmark de tempo de execução."""
    criar_diretorio_graficos()
    
    metodos = list(resultados.keys())
    tempos = [resultados[m]['tempo_medio'] * 1000 for m in metodos]
    erros = [resultados[m]['tempo_std'] * 1000 for m in metodos]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    cores = sns.color_palette("husl", len(metodos))
    
    bars = ax.barh(metodos, tempos, xerr=erros, color=cores, alpha=0.8, capsize=5)
    ax.set_xlabel('Tempo (ms)', fontsize=12)
    ax.set_title('Benchmark de Tempo de Execução', fontsize=14, fontweight='bold')
    
    for bar, tempo in zip(bars, tempos):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{tempo:.3f} ms', va='center', fontsize=10)
    
    plt.tight_layout()
    if salvar:
        plt.savefig(f'{DIRETORIO_GRAFICOS}/benchmark_tempo.png', dpi=150)
    return fig


def plotar_estabilidade(resultados: Dict, salvar: bool = True) -> plt.Figure:
    """Plota análise de estabilidade com zeros."""
    criar_diretorio_graficos()
    
    proporcoes = [p * 100 for p in resultados['proporcoes']]
    
    # Calcular médias dos valores válidos
    w_medias = []
    kl_medias = []
    w_estavel = []
    kl_estavel = []
    
    for i in range(len(proporcoes)):
        w_vals = resultados['w_valores'][i]
        kl_vals = resultados['kl_valores'][i]
        
        w_medias.append(np.mean(w_vals) if w_vals else 0)
        kl_medias.append(np.mean(kl_vals) if kl_vals else 0)
        w_estavel.append(resultados['w_infinitos'][i] == 0)
        kl_estavel.append(resultados['kl_infinitos'][i] == 0)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    x = np.arange(len(proporcoes))
    width = 0.35
    
    # Gráfico 1: Valores médios das divergências
    bars1 = axes[0].bar(x - width/2, w_medias, width, label='Divergência W', color='#3498db', alpha=0.8)
    bars2 = axes[0].bar(x + width/2, kl_medias, width, label='KL', color='#e74c3c', alpha=0.8)
    
    axes[0].set_xlabel('Proporção de Zeros (%)', fontsize=12)
    axes[0].set_ylabel('Valor Médio da Divergência', fontsize=12)
    axes[0].set_title('Valores Médios por Esparsidade', fontsize=13, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f'{int(p)}%' for p in proporcoes])
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Gráfico 2: Status de estabilidade
    cores_w = ['#27ae60' if e else '#e74c3c' for e in w_estavel]
    cores_kl = ['#27ae60' if e else '#e74c3c' for e in kl_estavel]
    
    axes[1].barh([f'{int(p)}% - W' for p in proporcoes], [1]*len(proporcoes), color=cores_w, alpha=0.8)
    axes[1].barh([f'{int(p)}% - KL' for p in proporcoes], [1]*len(proporcoes), color=cores_kl, alpha=0.8, left=1.2)
    
    # Legenda manual
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#27ae60', label='Estável (0 infinitos)'),
                       Patch(facecolor='#e74c3c', label='Instável (com infinitos)')]
    axes[1].legend(handles=legend_elements, loc='lower right')
    axes[1].set_xlabel('Status', fontsize=12)
    axes[1].set_title('Estabilidade Numérica', fontsize=13, fontweight='bold')
    axes[1].set_xlim(0, 2.5)
    axes[1].set_xticks([0.5, 1.7])
    axes[1].set_xticklabels(['W', 'KL'])
    
    plt.suptitle('Análise de Estabilidade com Distribuições Esparsas', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if salvar:
        plt.savefig(f'{DIRETORIO_GRAFICOS}/estabilidade_zeros.png', dpi=150)
    return fig


def plotar_simetria(resultados: Dict, salvar: bool = True) -> plt.Figure:
    """Plota teste de simetria."""
    criar_diretorio_graficos()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # W (simétrica)
    axes[0].hist(resultados['diferencas_w'], bins=30, color='#3498db', alpha=0.7, edgecolor='white')
    axes[0].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[0].set_xlabel('|W(P,Q) - W(Q,P)|', fontsize=11)
    axes[0].set_ylabel('Frequência', fontsize=11)
    axes[0].set_title('Divergência W (Simétrica)', fontsize=12, fontweight='bold')
    axes[0].set_xlim(-1e-15, max(resultados['diferencas_w']) * 1.1 + 1e-15)
    
    # KL (assimétrica)
    axes[1].hist(resultados['diferencas_kl'], bins=30, color='#e74c3c', alpha=0.7, edgecolor='white')
    axes[1].set_xlabel('|KL(P||Q) - KL(Q||P)|', fontsize=11)
    axes[1].set_ylabel('Frequência', fontsize=11)
    axes[1].set_title('KL (Assimétrica)', fontsize=12, fontweight='bold')
    
    plt.suptitle('Teste de Simetria das Divergências', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if salvar:
        plt.savefig(f'{DIRETORIO_GRAFICOS}/teste_simetria.png', dpi=150)
    return fig


def plotar_sensibilidade_lambda(resultados: Dict, salvar: bool = True) -> plt.Figure:
    """Plota sensibilidade ao parâmetro lambda."""
    criar_diretorio_graficos()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(resultados['lambdas'], resultados['pequena_diferenca'],
            label='Pequena diferença (μ=0 vs μ=0.5)', linewidth=2, color='#3498db')
    ax.plot(resultados['lambdas'], resultados['grande_diferenca'],
            label='Grande diferença (μ=0 vs μ=3)', linewidth=2, color='#e74c3c')
    
    ax.set_xlabel('λ (lambda)', fontsize=12)
    ax.set_ylabel('Divergência W', fontsize=12)
    ax.set_title('Sensibilidade ao Parâmetro λ', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if salvar:
        plt.savefig(f'{DIRETORIO_GRAFICOS}/sensibilidade_lambda.png', dpi=150)
    return fig


def gerar_todos_graficos(salvar: bool = True):
    """Gera todos os gráficos de análise."""
    from .analise import (benchmark_divergencias, testar_simetria,
                          testar_estabilidade_zeros, comparar_cenarios,
                          analise_sensibilidade_lambda)
    from .gerador_dados import gerar_par_gaussianas
    
    print("Gerando gráficos...")
    criar_diretorio_graficos()
    
    # 1. Distribuições
    x, p, q = gerar_par_gaussianas(100, separacao_medias=2.0)
    plotar_distribuicoes(x, p, q, "Gaussianas com Médias Diferentes", salvar)
    print("  ✓ distribuicoes.png")
    
    # 2. Comparação
    cenarios = comparar_cenarios()
    plotar_comparacao_divergencias(cenarios, salvar)
    print("  ✓ comparacao_divergencias.png")
    
    # 3. Benchmark
    benchmark = benchmark_divergencias(n_repeticoes=50)
    plotar_benchmark_tempo(benchmark, salvar)
    print("  ✓ benchmark_tempo.png")
    
    # 4. Estabilidade
    estab = testar_estabilidade_zeros()
    plotar_estabilidade(estab, salvar)
    print("  ✓ estabilidade_zeros.png")
    
    # 5. Simetria
    simetria = testar_simetria(n_testes=100)
    plotar_simetria(simetria, salvar)
    print("  ✓ teste_simetria.png")
    
    # 6. Lambda
    sensib = analise_sensibilidade_lambda()
    plotar_sensibilidade_lambda(sensib, salvar)
    print("  ✓ sensibilidade_lambda.png")
    
    print(f"\nTodos os gráficos salvos em '{DIRETORIO_GRAFICOS}/'")


if __name__ == "__main__":
    gerar_todos_graficos()
