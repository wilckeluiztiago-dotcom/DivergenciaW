# -*- coding: utf-8 -*-
"""
Divergência W - Distribuição Gaussiana
Autor: Luiz Tiago Wilcke
"""
import numpy as np
from scipy.stats import norm

def gerar_gaussiana_w(mu: float, sigma: float, n: int = 100) -> np.ndarray:
    """Gera uma densidade Gaussiana normalizada."""
    x = np.linspace(mu - 4*sigma, mu + 4*sigma, n)
    prob = norm.pdf(x, mu, sigma)
    return prob / np.sum(prob)
