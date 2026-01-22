# -*- coding: utf-8 -*-
"""
Divergência W - Distribuição de Bernoulli
Autor: Luiz Tiago Wilcke
"""
import numpy as np

def gerar_bernoulli_w(p: float) -> np.ndarray:
    """Gera uma distribuição de Bernoulli."""
    return np.array([1 - p, p])
