# -*- coding: utf-8 -*-
"""
Divergência W - Módulo de Análise
Autor: Luiz Tiago Wilcke
"""

import numpy as np
import time
from typing import Dict, List, Optional
from scipy import stats as scipy_stats
import warnings

from .matematica import (
    calcular_kl, calcular_w, calcular_jensen_shannon, calcular_hellinger
)
from .gerador_dados import gerar_gaussiana, gerar_esparsa, gerar_cenarios_teste


def benchmark_divergencias(n_repeticoes: int = 100, tamanho_distribuicao: int = 1000,
                           seed: Optional[int] = 42) -> Dict[str, Dict]:
    """Benchmark de tempo de execução das divergências."""
    if seed:
        np.random.seed(seed)
    
    _, p = gerar_gaussiana(tamanho_distribuicao, media=0, desvio=1)
    _, q = gerar_gaussiana(tamanho_distribuicao, media=0.5, desvio=1.2)
    
    resultados = {}
    
    for nome, func in [('Divergência W', lambda: calcular_w(p, q)),
                       ('Divergência KL', lambda: calcular_kl(p, q)),
                       ('scipy.stats.entropy', lambda: scipy_stats.entropy(p, q)),
                       ('Jensen-Shannon', lambda: calcular_jensen_shannon(p, q)),
                       ('Hellinger', lambda: calcular_hellinger(p, q))]:
        tempos = []
        for _ in range(n_repeticoes):
            inicio = time.perf_counter()
            func()
            tempos.append(time.perf_counter() - inicio)
        
        resultados[nome] = {
            'tempo_medio': np.mean(tempos),
            'tempo_std': np.std(tempos),
            'tempo_min': np.min(tempos),
            'tempo_max': np.max(tempos)
        }
    
    return resultados


def testar_simetria(n_testes: int = 100, tamanho: int = 100,
                    tolerancia: float = 1e-10, seed: Optional[int] = 42) -> Dict:
    """Testa a propriedade de simetria: W(P, Q) == W(Q, P)."""
    if seed:
        np.random.seed(seed)
    
    diferencas_w, diferencas_kl = [], []
    
    for _ in range(n_testes):
        p = np.random.exponential(1, tamanho)
        q = np.random.exponential(1.5, tamanho)
        p, q = p / np.sum(p), q / np.sum(q)
        
        diferencas_w.append(abs(calcular_w(p, q) - calcular_w(q, p)))
        diferencas_kl.append(abs(calcular_kl(p, q) - calcular_kl(q, p)))
    
    return {
        'todos_simetricos_w': np.all(np.array(diferencas_w) < tolerancia),
        'diferenca_maxima_w': np.max(diferencas_w),
        'diferenca_media_w': np.mean(diferencas_w),
        'diferencas_w': np.array(diferencas_w),
        'diferenca_maxima_kl': np.max(diferencas_kl),
        'diferenca_media_kl': np.mean(diferencas_kl),
        'diferencas_kl': np.array(diferencas_kl),
        'n_testes': n_testes
    }


def testar_estabilidade_zeros(proporcoes_zeros: List[float] = None,
                              n_repeticoes: int = 50, tamanho: int = 100,
                              seed: Optional[int] = 42) -> Dict:
    """Testa estabilidade numérica com zeros."""
    if proporcoes_zeros is None:
        proporcoes_zeros = [0.0, 0.3, 0.5, 0.7, 0.9, 0.95]
    if seed:
        np.random.seed(seed)
    
    resultados = {'proporcoes': proporcoes_zeros, 'w_estavel': [], 'kl_estavel': [],
                  'w_valores': [], 'kl_valores': [], 'w_infinitos': [], 'kl_infinitos': []}
    
    for prop in proporcoes_zeros:
        valores_w, valores_kl, infinitos_w, infinitos_kl = [], [], 0, 0
        
        for _ in range(n_repeticoes):
            _, p = gerar_esparsa(tamanho, proporcao_zeros=prop)
            _, q = gerar_esparsa(tamanho, proporcao_zeros=prop)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    w = calcular_w(p, q)
                    if np.isfinite(w): valores_w.append(w)
                    else: infinitos_w += 1
                except: infinitos_w += 1
                
                try:
                    kl = calcular_kl(p, q)
                    if np.isfinite(kl): valores_kl.append(kl)
                    else: infinitos_kl += 1
                except: infinitos_kl += 1
        
        resultados['w_estavel'].append(infinitos_w == 0)
        resultados['kl_estavel'].append(infinitos_kl == 0)
        resultados['w_valores'].append(valores_w)
        resultados['kl_valores'].append(valores_kl)
        resultados['w_infinitos'].append(infinitos_w)
        resultados['kl_infinitos'].append(infinitos_kl)
    
    return resultados


def comparar_cenarios() -> Dict[str, Dict]:
    """Compara divergências em cenários pré-definidos."""
    cenarios = gerar_cenarios_teste()
    resultados = {}
    
    for cenario in cenarios:
        nome, p, q = cenario['nome'], cenario['p'], cenario['q']
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            resultados[nome] = {
                'Divergência W': calcular_w(p, q) if np.all(np.isfinite(p)) else np.nan,
                'KL': calcular_kl(p, q) if np.all(np.isfinite(p)) else np.nan,
                'Jensen-Shannon': calcular_jensen_shannon(p, q) if np.all(np.isfinite(p)) else np.nan,
                'Hellinger': calcular_hellinger(p, q) if np.all(np.isfinite(p)) else np.nan
            }
    
    return resultados


def analise_sensibilidade_lambda(lambdas: np.ndarray = None, seed: Optional[int] = 42) -> Dict:
    """Analisa como λ afeta a Divergência W."""
    if lambdas is None:
        lambdas = np.linspace(0.01, 3.0, 50)
    
    _, p1 = gerar_gaussiana(100, media=0, desvio=1, seed=seed)
    _, q1 = gerar_gaussiana(100, media=0.5, desvio=1, seed=seed)
    _, p2 = gerar_gaussiana(100, media=0, desvio=1, seed=seed)
    _, q2 = gerar_gaussiana(100, media=3, desvio=1, seed=seed)
    
    return {
        'lambdas': lambdas,
        'pequena_diferenca': [calcular_w(p1, q1, lambda_suavizacao=l) for l in lambdas],
        'grande_diferenca': [calcular_w(p2, q2, lambda_suavizacao=l) for l in lambdas]
    }
