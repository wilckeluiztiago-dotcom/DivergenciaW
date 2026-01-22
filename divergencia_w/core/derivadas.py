# -*- coding: utf-8 -*-
"""
Divergência W - Derivadas e Gradientes
Autor: Luiz Tiago Wilcke
"""
import numpy as np

def gradiente_w(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """Calcula o gradiente da Divergência W em relação a P."""
    # Aproximação numérica simples do gradiente para demonstração
    h = 1e-7
    grad = np.zeros_like(p)
    from .matematica_base import calcular_w
    
    for i in range(len(p)):
        p_plus = p.copy()
        p_plus[i] += h
        p_minus = p.copy()
        p_minus[i] -= h
        grad[i] = (calcular_w(p_plus, q) - calcular_w(p_minus, q)) / (2 * h)
    
    return grad
