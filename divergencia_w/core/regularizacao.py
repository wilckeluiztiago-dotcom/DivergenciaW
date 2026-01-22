# -*- coding: utf-8 -*-
"""
Divergência W - Regularização e Suavização
Autor: Luiz Tiago Wilcke
"""
import numpy as np

def aplicar_suavizacao(p: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    """Aplica suavização de Laplace (Additive Smoothing) à distribuição."""
    n = len(p)
    return (p + alpha) / (np.sum(p) + alpha * n)

def tunar_lambda(p: np.ndarray, q: np.ndarray) -> float:
    """Sugere um valor de λ baseado na variância das distribuições."""
    var_media = (np.var(p) + np.var(q)) / 2
    return float(1.0 / (np.sqrt(var_media) + 1e-5))
