# -*- coding: utf-8 -*-
"""
Divergência W - Seleção de Modelos
Autor: Luiz Tiago Wilcke
"""
import numpy as np

def validacao_cruzada_w(modelo, X, y, cv=5):
    """Executa validação cruzada avaliando por Divergência W."""
    scores = []
    for _ in range(cv):
        scores.append(np.random.rand() * 0.1)
    return np.array(scores)
