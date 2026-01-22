# -*- coding: utf-8 -*-
"""
Divergência W - Simulação Monte Carlo
Autor: Luiz Tiago Wilcke
"""
import numpy as np
from ..core.matematica_base import calcular_w

def simular_monte_carlo_w(n_sim: int = 1000, dim: int = 10) -> np.ndarray:
    """Gera distribuição de W sob hipótese nula via Monte Carlo."""
    w_stats = np.zeros(n_sim)
    for i in range(n_sim):
        p = np.random.dirichlet([1]*dim)
        q = np.random.dirichlet([1]*dim)
        w_stats[i] = calcular_w(p, q)
    return w_stats
