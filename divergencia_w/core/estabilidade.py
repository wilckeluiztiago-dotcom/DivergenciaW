# -*- coding: utf-8 -*-
"""
Divergência W - Estabilidade Numérica
Autor: Luiz Tiago Wilcke
"""
import numpy as np

def verificar_estabilidade_numerica(p: np.ndarray, q: np.ndarray) -> bool:
    """Verifica se há riscos de instabilidade (NaN, Inf) nos cálculos de W."""
    if np.any(np.isnan(p)) or np.any(np.isnan(q)):
        return False
    if np.any(np.isinf(p)) or np.any(np.isinf(q)):
        return False
    return True

def tratar_divisao_zero(numerador, denominador, epsilon=1e-15):
    """Auxiliar para evitar divisões por zero de forma segura."""
    return numerador / (denominador + epsilon)
