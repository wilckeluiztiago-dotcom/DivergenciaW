# -*- coding: utf-8 -*-
"""
Divergência W - Bootstrap
Autor: Luiz Tiago Wilcke
"""
import numpy as np
from ..core.matematica_base import calcular_w

def bootstrap_divergencia_w(data_p: np.ndarray, data_q: np.ndarray, n_boot: int = 100) -> np.ndarray:
    """Estima a distribuição de W via bootstrap."""
    w_values = []
    n_p = len(data_p)
    n_q = len(data_q)
    for _ in range(n_boot):
        sample_p = np.random.choice(data_p, size=n_p, replace=True)
        sample_q = np.random.choice(data_q, size=n_q, replace=True)
        w_values.append(calcular_w(sample_p, sample_q))
    return np.array(w_values)
