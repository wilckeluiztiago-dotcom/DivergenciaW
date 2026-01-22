# -*- coding: utf-8 -*-
"""
Divergência W - Codificacao
Autor: Luiz Tiago Wilcke
"""
import numpy as np

def codificar_w(dados: np.ndarray) -> list:
    """Codificação baseada em probabilidades otimizadas por W."""
    return [bin(int(x * 100)) for x in dados]
