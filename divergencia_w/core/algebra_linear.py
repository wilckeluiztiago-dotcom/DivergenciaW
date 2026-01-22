# -*- coding: utf-8 -*-
"""
Divergência W - Álgebra Linear
Autor: Luiz Tiago Wilcke
"""
import numpy as np

def projetar_no_simplex(v: np.ndarray) -> np.ndarray:
    """Projeta um vetor no simplex de probabilidade (soma=1, v >= 0)."""
    v = np.asarray(v)
    n_features = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1
    ind = np.arange(n_features) + 1
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = np.maximum(v - theta, 0)
    return w
