# -*- coding: utf-8 -*-
"""
Divergência W - Autoencoder
Autor: Luiz Tiago Wilcke
"""
import numpy as np

class AutoencoderW:
    """Autoencoder que utiliza a Divergência W no erro de reconstrução."""
    def __init__(self, latent_dim=2):
        self.latent_dim = latent_dim
        
    def reconstruir(self, X):
        # Simulação de reconstrução
        return X + np.random.normal(0, 0.01, X.shape)
