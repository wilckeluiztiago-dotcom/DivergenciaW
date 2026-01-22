# -*- coding: utf-8 -*-
"""
Divergência W - Teste de Independência
Autor: Luiz Tiago Wilcke
"""
import numpy as np
from ..core.matematica_base import calcular_w

def teste_independencia_w(matriz_contingencia: np.ndarray) -> float:
    """Testa independência em tabelas de contingência usando W."""
    # Calcula marginais
    total = np.sum(matriz_contingencia)
    marginal_row = np.sum(matriz_contingencia, axis=1) / total
    marginal_col = np.sum(matriz_contingencia, axis=0) / total
    
    # Distribuição esperada sob independência
    esperado = np.outer(marginal_row, marginal_col)
    observado = matriz_contingencia / total
    
    return calcular_w(observado.flatten(), esperado.flatten())
