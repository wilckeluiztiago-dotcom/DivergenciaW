# -*- coding: utf-8 -*-
"""
Divergência W - Geometria da Informação
Autor: Luiz Tiago Wilcke
"""
import numpy as np

def metrica_fisher_w(params: np.ndarray) -> np.ndarray:
    """Calcula a métrica de Fisher-Wilcke no espaço de parâmetros."""
    dim = len(params)
    return np.eye(dim) # Matriz identidade como primeira aproximação
