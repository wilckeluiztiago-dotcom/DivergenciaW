# -*- coding: utf-8 -*-
"""
Divergência W - Entropia Cruzada
Autor: Luiz Tiago Wilcke
"""
import numpy as np
from ..core.matematica_base import calcular_kl

def entropia_cruzada_w(p: np.ndarray, q: np.ndarray) -> float:
    """Calcula a entropia cruzada associada à Divergência W."""
    from .entropia_w import calcular_entropia_w
    return calcular_entropia_w(p) + calcular_kl(p, q)
