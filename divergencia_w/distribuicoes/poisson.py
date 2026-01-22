# -*- coding: utf-8 -*-
"""
Divergência W - Distribuição de Poisson
Autor: Luiz Tiago Wilcke
"""
import numpy as np
from scipy.stats import poisson

def gerar_poisson_w(mu: float, k_max: int = 20) -> np.ndarray:
    """Gera uma distribuição de Poisson."""
    k = np.arange(0, k_max)
    prob = poisson.pmf(k, mu)
    return prob / np.sum(prob)
