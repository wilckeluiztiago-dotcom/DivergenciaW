# -*- coding: utf-8 -*-
"""
Divergência W - Otimizador Customizado
Autor: Luiz Tiago Wilcke
"""
import numpy as np

class OtimizadorW:
    """Otimizador simples que usa gradiente da Divergência W."""
    def __init__(self, lr=0.01):
        self.lr = lr
        
    def step(self, params, grads):
        for i in range(len(params)):
            params[i] -= self.lr * grads[i]
        return params
