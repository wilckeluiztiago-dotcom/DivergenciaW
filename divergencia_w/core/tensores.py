# -*- coding: utf-8 -*-
"""
Divergência W - Operações Tensorais
Autor: Luiz Tiago Wilcke
"""
import numpy as np

def operacoes_tensorais_w(tensor_p: np.ndarray, tensor_q: np.ndarray) -> np.ndarray:
    """Realiza operações tensorais aplicadas à divergência W."""
    # Simulação de contração tensorial para fins demonstrativos
    produto = np.tensordot(tensor_p, tensor_q, axes=0)
    return np.sqrt(np.abs(produto))

def broadcast_w(p: np.ndarray, matriz_q: np.ndarray) -> np.ndarray:
    """Calcula W entre um vetor e cada linha de uma matriz."""
    resultados = []
    from .matematica_base import calcular_w
    for q in matriz_q:
        resultados.append(calcular_w(p, q))
    return np.array(resultados)
