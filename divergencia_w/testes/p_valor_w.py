# -*- coding: utf-8 -*-
"""
Divergência W - Cálculo de P-Valor
Autor: Luiz Tiago Wilcke
"""
import numpy as np

def calcular_p_valor_w(estatistica_obs: float, n_simulacoes: int = 1000) -> float:
    """Calcula p-valor por permutação/simulação para a estatística W."""
    # Simulação simplificada (nula: distribuições iguais -> W=0)
    # Em uma aplicação real, geramos distribuições sob H0.
    w_nulos = np.random.exponential(scale=0.01, size=n_simulacoes)
    p_val = np.sum(w_nulos >= estatistica_obs) / n_simulacoes
    return float(p_val)
