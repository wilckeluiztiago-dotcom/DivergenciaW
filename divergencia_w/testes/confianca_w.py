# -*- coding: utf-8 -*-
"""
Divergência W - Intervalo de Confiança
Autor: Luiz Tiago Wilcke
"""
import numpy as np

def intervalo_confianca_w(w_boot: np.ndarray, alpha: float = 0.05) -> tuple:
    """Calcula intervalo de confiança para W usando amostras bootstrap."""
    lower = np.percentile(w_boot, 100 * alpha / 2)
    upper = np.percentile(w_boot, 100 * (1 - alpha / 2))
    return float(lower), float(upper)
