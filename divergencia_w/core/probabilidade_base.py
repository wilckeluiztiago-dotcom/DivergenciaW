# -*- coding: utf-8 -*-
"""
Divergência W - Probabilidade Base
Autor: Luiz Tiago Wilcke
"""
import numpy as np

def entropia_shannon(p: np.ndarray, epsilon: float = 1e-10) -> float:
    """Calcula a entropia de Shannon de uma distribuição."""
    p = np.asarray(p)
    p = np.maximum(p, epsilon)
    p = p / np.sum(p)
    return -float(np.sum(p * np.log(p)))

def surpresa_media(p: np.ndarray, q: np.ndarray) -> float:
    """Calcula a surpresa média de P em relação a Q."""
    p = p / np.sum(p)
    q = np.maximum(q, 1e-10)
    q = q / np.sum(q)
    return -float(np.sum(p * np.log(q)))
