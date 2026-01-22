# -*- coding: utf-8 -*-
"""
Divergência W - Entropia de Wilcke
Autor: Luiz Tiago Wilcke
"""
import numpy as np
from ..core.matematica_base import calcular_w

def calcular_entropia_w(p: np.ndarray) -> float:
    """Calcula a entropia baseada na Divergência W em relação à uniforme."""
    n = len(p)
    u = np.ones(n) / n
    return 1.0 - calcular_w(p, u)
