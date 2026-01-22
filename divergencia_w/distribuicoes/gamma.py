# -*- coding: utf-8 -*-
"""
Divergência W - Distribuição Gamma
Autor: Luiz Tiago Wilcke
"""
import numpy as np
from scipy.stats import gamma

def gerar_gamma_w(a: float, n: int = 100) -> np.ndarray:
    """Gera uma distribuição Gamma."""
    x = np.linspace(0, 20, n)
    prob = gamma.pdf(x, a)
    return prob / np.sum(prob)
