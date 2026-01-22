# -*- coding: utf-8 -*-
"""
Divergência W - Teste de Aderência
Autor: Luiz Tiago Wilcke
"""
import numpy as np
from ..core.matematica_base import calcular_w

def teste_aderencia_w(observado: np.ndarray, esperado: np.ndarray) -> float:
    """Calcula a estatística W de aderência entre observado e esperado."""
    return calcular_w(observado, esperado)
