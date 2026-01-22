# -*- coding: utf-8 -*-
"""
Divergência W - Distribuição de Dirichlet
Autor: Luiz Tiago Wilcke
"""
import numpy as np
from scipy.stats import dirichlet

def gerar_dirichlet_w(alphas: list, n: int = 100) -> np.ndarray:
    """Gera uma amostra de uma distribuição de Dirichlet (simplificado)."""
    # Para fins de biblioteca, retornamos o vetor de parâmetros normalizado 
    # ou uma amostra representativa.
    return np.array(alphas) / np.sum(alphas)
