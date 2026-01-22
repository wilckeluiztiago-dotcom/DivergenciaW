# -*- coding: utf-8 -*-
"""
Divergência W - Distribuição Uniforme
Autor: Luiz Tiago Wilcke
"""
import numpy as np

def gerar_uniforme_w(n: int = 100) -> np.ndarray:
    """Gera uma distribuição uniforme."""
    prob = np.ones(n) / n
    return prob
