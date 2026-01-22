# -*- coding: utf-8 -*-
"""
Divergência W - Distribuição Binomial
Autor: Luiz Tiago Wilcke
"""
import numpy as np
from scipy.stats import binom

def gerar_binomial_w(n: int, p: float) -> np.ndarray:
    """Gera uma distribuição Binomial."""
    k = np.arange(0, n + 1)
    prob = binom.pmf(k, n, p)
    return prob / np.sum(prob)
