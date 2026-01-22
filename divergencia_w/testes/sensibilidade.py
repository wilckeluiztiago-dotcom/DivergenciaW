# -*- coding: utf-8 -*-
"""
Divergência W - Análise de Sensibilidade
Autor: Luiz Tiago Wilcke
"""
import numpy as np
from ..core.matematica_base import calcular_w

def analise_sensibilidade_lambda(p: np.ndarray, q: np.ndarray, lambdas: list) -> dict:
    """Analisa como o valor de W varia com o parâmetro lambda."""
    resultados = {}
    for l in lambdas:
        resultados[l] = calcular_w(p, q, lambda_suavizacao=l)
    return resultados
