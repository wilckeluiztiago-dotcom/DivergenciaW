# -*- coding: utf-8 -*-
"""
Divergência W - Distribuição Exponencial
Autor: Luiz Tiago Wilcke
"""
import numpy as np
from scipy.stats import expon

def gerar_exponencial_w(escala: float, n: int = 100) -> np.ndarray:
    """Gera uma distribuição exponencial."""
    x = np.linspace(0, 10, n)
    prob = expon.pdf(x, scale=escala)
    return prob / np.sum(prob)
