# -*- coding: utf-8 -*-
"""
Divergência W - Propriedades de Espaços Métricos
Autor: Luiz Tiago Wilcke
"""
import numpy as np

def verificar_axiomas_metrica(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> dict:
    """Verifica se a Divergência W satisfaz propriedades de métrica."""
    from .matematica_base import calcular_w
    
    w_pq = calcular_w(p, q)
    w_qp = calcular_w(q, p)
    w_pp = calcular_w(p, p)
    
    # Desigualdade triangular (pode não satisfazer estritamente como divergência)
    w_pr = calcular_w(p, r)
    w_qr = calcular_w(q, r)
    
    return {
        "identidade": w_pp < 1e-12,
        "simetria": abs(w_pq - w_qp) < 1e-12,
        "triangular": w_pr <= (w_pq + w_qr) + 1e-10,
        "nao_negatividade": w_pq >= -1e-12
    }
