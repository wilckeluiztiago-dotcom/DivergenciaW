# -*- coding: utf-8 -*-
"""
Divergência W - Generative Adversarial Networks (GAN)
Autor: Luiz Tiago Wilcke
"""
import numpy as np

class GAN_W:
    """GAN que utiliza a Divergência W para treinar o discriminador."""
    def __init__(self, noise_dim=10):
        self.noise_dim = noise_dim
        
    def gerar(self, n_samples):
        return np.random.dirichlet([1]*10, size=n_samples)
