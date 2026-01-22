# -*- coding: utf-8 -*-
"""
Divergência W - Capacidade de Canal
Autor: Luiz Tiago Wilcke
"""
import numpy as np

def calcular_capacidade_w(matriz_transicao: np.ndarray) -> float:
    """Estima a capacidade de um canal de comunicação usando W."""
    # Maximização da informação mútua baseada em W (simplificado)
    return np.max(np.sum(matriz_transicao * np.log(matriz_transicao + 1e-10), axis=1))
