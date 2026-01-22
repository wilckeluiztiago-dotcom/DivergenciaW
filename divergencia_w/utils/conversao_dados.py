# -*- coding: utf-8 -*-
"""
Divergência W - Conversão de Dados
Autor: Luiz Tiago Wilcke
"""
import numpy as np

def para_probabilidade(x: np.ndarray) -> np.ndarray:
    """Converte um vetor arbitrário em uma distribuição de probabilidade."""
    x = np.abs(x)
    return x / np.sum(x)
