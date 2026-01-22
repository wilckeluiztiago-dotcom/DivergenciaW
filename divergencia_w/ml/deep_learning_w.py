# -*- coding: utf-8 -*-
"""
Divergência W - Deep Learning (Custom Layer)
Autor: Luiz Tiago Wilcke
"""
import numpy as np

class CamadaW:
    """Camada de rede neural customizada baseada em Divergência W."""
    def __init__(self, units):
        self.units = units
        self.weights = np.random.randn(units)
        
    def forward(self, input_data):
        return input_data * self.weights
