# -*- coding: utf-8 -*-
"""
Divergência W - Verificação de Simetria
Autor: Luiz Tiago Wilcke
"""
import numpy as np
from ..core.matematica_base import calcular_w

def verificar_simetria_global(n_testes: int = 100, dim: int = 10) -> bool:
    """Verifica estatisticamente se W é simétrico para várias distribuições."""
    porcentagem_sucesso = 0
    for _ in range(n_testes):
        p = np.random.rand(dim)
        q = np.random.rand(dim)
        w_pq = calcular_w(p, q)
        w_qp = calcular_w(q, p)
        if abs(w_pq - w_qp) < 1e-12:
            porcentagem_sucesso += 1
    return porcentagem_sucesso == n_testes
