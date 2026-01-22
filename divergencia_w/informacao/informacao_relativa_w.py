# -*- coding: utf-8 -*-
"""
Divergência W - Informação Relativa
Autor: Luiz Tiago Wilcke
"""
import numpy as np
from ..core.matematica_base import calcular_w

def informacao_relativa_w(p: np.ndarray, q: np.ndarray) -> float:
    """Calcula a informação relativa (ganho de informação) via W."""
    return calcular_w(p, q)
