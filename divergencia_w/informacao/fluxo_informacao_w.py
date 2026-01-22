# -*- coding: utf-8 -*-
"""
Divergência W - Fluxo de Informação
Autor: Luiz Tiago Wilcke
"""
import numpy as np

def estimar_fluxo_w(serie_temporal: np.ndarray) -> float:
    """Estima o fluxo de informação em uma série temporal usando W."""
    # Diferença entre P(t) e P(t+1)
    p_t = serie_temporal[:-1] / np.sum(serie_temporal[:-1])
    p_next = serie_temporal[1:] / np.sum(serie_temporal[1:])
    from ..core.matematica_base import calcular_w
    return calcular_w(p_t, p_next)
