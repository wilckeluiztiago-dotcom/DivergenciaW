# -*- coding: utf-8 -*-
"""
Divergência W - Distribuição Beta
Autor: Luiz Tiago Wilcke
"""
import numpy as np
from scipy.stats import beta

def gerar_beta_w(a: float, b: float, n: int = 100) -> np.ndarray:
    """Gera uma distribuição Beta."""
    x = np.linspace(0.01, 0.99, n)
    prob = beta.pdf(x, a, b)
    return prob / np.sum(prob)
