# -*- coding: utf-8 -*-
"""
Divergência W - Classificação (KNN)
Autor: Luiz Tiago Wilcke
"""
import numpy as np
from ..core.matematica_base import calcular_w

class KNN_W:
    """K-Nearest Neighbors usando Divergência W."""
    def __init__(self, k=3):
        self.k = k
        
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        
    def predict(self, X):
        preds = []
        for x in X:
            distancias = [calcular_w(x, xt) for xt in self.X_train]
            vizinhos = np.argsort(distancias)[:self.k]
            rotulos = self.y_train[vizinhos]
            preds.append(np.bincount(rotulos).argmax())
        return np.array(preds)
