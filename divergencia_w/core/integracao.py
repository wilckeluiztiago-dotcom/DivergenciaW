# -*- coding: utf-8 -*-
"""
Divergência W - Integração Numérica
Autor: Luiz Tiago Wilcke
"""
import numpy as np

def integrar_w(func_p, func_q, a: float, b: float, n_pontos: int = 1000) -> float:
    """Integra a Divergência W em um intervalo [a, b]."""
    x = np.linspace(a, b, n_pontos)
    dx = x[1] - x[0]
    p = func_p(x)
    q = func_q(x)
    
    from .matematica_base import calcular_w
    # Simplificação: integra a contribuição ponto a ponto multiplicada por dx
    # Note: Isso é uma aproximação escalar
    valores_p = p / np.sum(p) if np.sum(p) > 0 else p
    valores_q = q / np.sum(q) if np.sum(q) > 0 else q
    
    # Termo integrando aproximado pela lógica de W
    ε = 1e-10
    λ = 0.5
    dif = valores_p - valores_q
    termo = (dif**2) / (valores_p + valores_q + ε) * np.exp(-λ * np.abs(dif))
    
    return float(np.trapz(termo, dx=dx))
