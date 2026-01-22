# -*- coding: utf-8 -*-
"""
Divergência W - Agrupamento (Clustering)
Autor: Luiz Tiago Wilcke
"""
import numpy as np
from ..core.matematica_base import calcular_w

class KMeansW:
    """Algoritmo K-Means usando Divergência W como métrica de distância."""
    def __init__(self, k=3, max_iter=20):
        self.k = k
        self.max_iter = max_iter
        self.centroides = None
        
    def fit(self, X):
        # Inicialização simples
        self.centroides = X[np.random.choice(X.shape[0], self.k, replace=False)]
        for _ in range(self.max_iter):
            # Atribuição
            rotulos = self._proximo_centroide(X)
            # Atualização
            for i in range(self.k):
                self.centroides[i] = np.mean(X[rotulos == i], axis=0)
                
    def _proximo_centroide(self, X):
        rotulos = []
        for x in X:
            distancias = [calcular_w(x, c) for c in self.centroides]
            rotulos.append(np.argmin(distancias))
        return np.array(rotulos)
