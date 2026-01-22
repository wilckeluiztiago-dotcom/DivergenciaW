# -*- coding: utf-8 -*-
"""
Divergência W - Informação Mútua
Autor: Luiz Tiago Wilcke
"""
import numpy as np
from ..core.matematica_base import calcular_w

def calcular_informacao_mutua_w(p_xy: np.ndarray) -> float:
    """Calcula a informação mútua baseada na Divergência W."""
    # p_xy é a distribuição conjunta
    p_x = np.sum(p_xy, axis=1)
    p_y = np.sum(p_xy, axis=0)
    p_x_p_y = np.outer(p_x, p_y)
    return calcular_w(p_xy.flatten(), p_x_p_y.flatten())
