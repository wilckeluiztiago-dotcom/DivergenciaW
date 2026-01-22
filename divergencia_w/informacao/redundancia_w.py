# -*- coding: utf-8 -*-
"""
Divergência W - Redundância
Autor: Luiz Tiago Wilcke
"""
import numpy as np
from .entropia_w import calcular_entropia_w

def calcular_redundancia_w(p: np.ndarray) -> float:
    """Calcula a redundância de uma distribuição."""
    h_max = 1.0 # Para uniforme normalizada
    h = calcular_entropia_w(p)
    return h_max - h
