# -*- coding: utf-8 -*-
"""
Divergência W - Otimização
Autor: Luiz Tiago Wilcke
"""
import numpy as np
from scipy.optimize import minimize

def otimizar_parametros_w(p: np.ndarray, q: np.ndarray) -> float:
    """Encontra o lambda que minimiza a diferença entre W e KL (exemplo)."""
    from .matematica_base import calcular_w, calcular_kl
    kl_alvo = calcular_kl(p, q)
    
    def objetivo(lambd):
        return (calcular_w(p, q, lambda_suavizacao=lambd[0]) - kl_alvo)**2
    
    res = minimize(objetivo, [0.5], bounds=[(0.01, 10.0)])
    return float(res.x[0])
