# -*- coding: utf-8 -*-
"""
Divergência W - Matemática Base
Autor: Luiz Tiago Wilcke
"""
import numpy as np
from typing import Union, Tuple

EPSILON_PADRAO = 1e-10
LAMBDA_PADRAO = 0.5

def normalizar_distribuicao(p: np.ndarray, epsilon: float = EPSILON_PADRAO) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    p = np.maximum(p, epsilon)
    return p / np.sum(p)

def validar_distribuicoes(p: np.ndarray, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    if p.shape != q.shape:
        raise ValueError(f"Dimensões incompatíveis: {p.shape} vs {q.shape}")
    return p, q

def calcular_kl(p: np.ndarray, q: np.ndarray, epsilon: float = EPSILON_PADRAO, normalizar: bool = True) -> float:
    p, q = validar_distribuicoes(p, q)
    if normalizar:
        p = normalizar_distribuicao(p, epsilon)
        q = normalizar_distribuicao(q, epsilon)
    else:
        p = np.maximum(p, epsilon)
        q = np.maximum(q, epsilon)
    return float(np.sum(p * np.log(p / q)))

def calcular_w(p: np.ndarray, q: np.ndarray, epsilon: float = EPSILON_PADRAO, 
               lambda_suavizacao: float = LAMBDA_PADRAO, normalizar: bool = True) -> float:
    p, q = validar_distribuicoes(p, q)
    if normalizar:
        p = normalizar_distribuicao(p, epsilon)
        q = normalizar_distribuicao(q, epsilon)
    dif = p - q
    denominador = p + q + epsilon
    termo_chi = (dif ** 2) / denominador
    fator_exp = np.exp(-lambda_suavizacao * np.abs(dif))
    return float(np.sum(termo_chi * fator_exp))

def calcular_jensen_shannon(p: np.ndarray, q: np.ndarray, epsilon: float = EPSILON_PADRAO) -> float:
    p, q = validar_distribuicoes(p, q)
    p = normalizar_distribuicao(p, epsilon)
    q = normalizar_distribuicao(q, epsilon)
    m = 0.5 * (p + q)
    return 0.5 * calcular_kl(p, m, epsilon, False) + 0.5 * calcular_kl(q, m, epsilon, False)

def calcular_hellinger(p: np.ndarray, q: np.ndarray, epsilon: float = EPSILON_PADRAO) -> float:
    p, q = validar_distribuicoes(p, q)
    p = normalizar_distribuicao(p, epsilon)
    q = normalizar_distribuicao(q, epsilon)
    return float(np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)))
