# -*- coding: utf-8 -*-
"""
Divergência W - Função de Perda (Loss)
Autor: Luiz Tiago Wilcke
"""
import numpy as np
from ..core.matematica_base import calcular_w

class PerdaW:
    """Implementa Divergência W como função de perda para treinamento."""
    def __init__(self, lambda_suavizacao=0.5):
        self.lambda_suavizacao = lambda_suavizacao
        
    def __call__(self, y_true, y_pred):
        # Assume que y_true e y_pred são distribuições (softmax output)
        return calcular_w(y_true, y_pred, lambda_suavizacao=self.lambda_suavizacao)
