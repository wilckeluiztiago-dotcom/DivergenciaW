# -*- coding: utf-8 -*-
"""
Divergência W - Verificação de Robustez
Autor: Luiz Tiago Wilcke
"""
import numpy as np
from ..core.matematica_base import calcular_w

def verificar_robustez_zeros(n_zeros: int = 5) -> bool:
    """Verifica se W permanece estável com distribuições contendo zeros."""
    p = np.array([0.5, 0.5] + [0.0] * n_zeros)
    q = np.array([0.0] * n_zeros + [0.5, 0.5])
    try:
        w = calcular_w(p, q)
        return not np.isnan(w) and not np.isinf(w)
    except:
        return False
