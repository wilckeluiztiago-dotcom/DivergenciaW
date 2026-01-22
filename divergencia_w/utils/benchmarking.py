# -*- coding: utf-8 -*-
"""
Divergência W - Benchmarking
Autor: Luiz Tiago Wilcke
"""
import time
from ..core.matematica_base import calcular_w

def benchmark_w(p: np.ndarray, q: np.ndarray, n_iter: int = 1000):
    """Mede o tempo médio de execução de W."""
    start = time.time()
    for _ in range(n_iter):
        calcular_w(p, q)
    end = time.time()
    return (end - start) / n_iter
