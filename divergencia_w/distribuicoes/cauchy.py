# -*- coding: utf-8 -*-
"""
Divergência W - Distribuição de Cauchy
Autor: Luiz Tiago Wilcke
"""
import numpy as np
from scipy.stats import cauchy

def gerar_cauchy_w(loc: float, scale: float, n: int = 100) -> np.ndarray:
    """Gera uma distribuição de Cauchy."""
    x = np.linspace(loc - 5*scale, loc + 5*scale, n)
    prob = cauchy.pdf(x, loc, scale)
    return prob / np.sum(prob)
