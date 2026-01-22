# -*- coding: utf-8 -*-
"""
Divergência W - Complexidade
Autor: Luiz Tiago Wilcke
"""
import numpy as np
from ..core.matematica_base import calcular_w

def medir_complexidade_w(p: np.ndarray) -> float:
    """Mede complexidade estatística via Divergência W."""
    # Baseado na distância entre a distribuição e o equilíbrio (uniforme)
    n = len(p)
    u = np.ones(n) / n
    return calcular_w(p, u) * (1.0 - calcular_w(p, u))
