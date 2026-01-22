# -*- coding: utf-8 -*-
"""
Divergência W - Teste de Homogeneidade
Autor: Luiz Tiago Wilcke
"""
import numpy as np
from ..core.matematica_base import calcular_w

def teste_homogeneidade_w(dist_a: np.ndarray, dist_b: np.ndarray) -> float:
    """Calcula se duas amostras provêm da mesma distribuição usando W."""
    return calcular_w(dist_a, dist_b)
