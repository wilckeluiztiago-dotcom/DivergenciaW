# -*- coding: utf-8 -*-
"""
Divergência W - Regressão
Autor: Luiz Tiago Wilcke
"""
import numpy as np

class RegressaoW:
    """Modelo de regressão que minimiza a Divergência W entre distribuições."""
    def __init__(self):
        self.coef = None
        
    def fit(self, X, y):
        # Simplificação: regressão linear básica
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        self.coef = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        
    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b.dot(self.coef)
