# -*- coding: utf-8 -*-
"""
Divergência W - Módulo Matemático Central
Autor: Luiz Tiago Wilcke

Este módulo implementa a Divergência W, uma nova medida de divergência
estatística simétrica e robusta, juntamente com a Divergência KL para comparação.
"""

import numpy as np
from typing import Union, Tuple

# Constantes de Regularização
EPSILON_PADRAO = 1e-10
LAMBDA_PADRAO = 0.5


def normalizar_distribuicao(p: np.ndarray, epsilon: float = EPSILON_PADRAO) -> np.ndarray:
    """
    Normaliza um array para que a soma seja 1 (distribuição de probabilidade válida).
    
    Parâmetros:
    -----------
    p : np.ndarray
        Array de valores não-negativos
    epsilon : float
        Valor mínimo para evitar zeros
        
    Retorna:
    --------
    np.ndarray
        Distribuição normalizada
    """
    p = np.asarray(p, dtype=np.float64)
    p = np.maximum(p, epsilon)
    return p / np.sum(p)


def validar_distribuicoes(p: np.ndarray, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Valida e prepara duas distribuições para cálculo de divergência.
    
    Parâmetros:
    -----------
    p, q : np.ndarray
        Distribuições a serem validadas
        
    Retorna:
    --------
    Tuple[np.ndarray, np.ndarray]
        Distribuições validadas e normalizadas
        
    Raises:
    -------
    ValueError
        Se as distribuições tiverem tamanhos diferentes
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    
    if p.shape != q.shape:
        raise ValueError(f"Distribuições devem ter o mesmo tamanho: {p.shape} vs {q.shape}")
    
    if len(p.shape) > 1:
        raise ValueError("Apenas distribuições unidimensionais são suportadas")
    
    return p, q


def calcular_kl(p: np.ndarray, q: np.ndarray, 
                epsilon: float = EPSILON_PADRAO,
                normalizar: bool = True) -> float:
    """
    Calcula a Divergência de Kullback-Leibler D_KL(P || Q).
    
    A divergência KL mede a diferença entre duas distribuições de probabilidade.
    Fórmula: D_KL(P || Q) = Σ P(x) * log(P(x) / Q(x))
    
    Parâmetros:
    -----------
    p : np.ndarray
        Distribuição P (verdadeira/referência)
    q : np.ndarray
        Distribuição Q (aproximação)
    epsilon : float
        Valor mínimo para evitar log(0) e divisão por zero
    normalizar : bool
        Se True, normaliza as distribuições antes do cálculo
        
    Retorna:
    --------
    float
        Valor da divergência KL (sempre >= 0)
        
    Notas:
    ------
    - KL não é simétrica: D_KL(P||Q) != D_KL(Q||P) em geral
    - KL pode tender a infinito quando Q(x) → 0 e P(x) > 0
    """
    p, q = validar_distribuicoes(p, q)
    
    if normalizar:
        p = normalizar_distribuicao(p, epsilon)
        q = normalizar_distribuicao(q, epsilon)
    else:
        p = np.maximum(p, epsilon)
        q = np.maximum(q, epsilon)
    
    # KL = Σ p(x) * log(p(x) / q(x))
    razao = p / q
    kl = np.sum(p * np.log(razao))
    
    return float(kl)


def calcular_w(p: np.ndarray, q: np.ndarray,
               epsilon: float = EPSILON_PADRAO,
               lambda_suavizacao: float = LAMBDA_PADRAO,
               normalizar: bool = True) -> float:
    """
    Calcula a Divergência W (Wilcke).
    
    A Divergência W é uma nova medida de divergência estatística que oferece:
    - Simetria: W(P, Q) = W(Q, P)
    - Robustez: Estabilidade numérica mesmo com zeros
    - Eficiência: Implementação vetorizada com NumPy
    
    Fórmula:
    --------
    W(P, Q) = Σ [(P(x) - Q(x))² / (P(x) + Q(x) + ε)] × exp(-λ|P(x) - Q(x)|)
    
    A fórmula combina:
    1. Distância Chi-quadrado modificada no numerador
    2. Termo de estabilização no denominador
    3. Fator de suavização exponencial para penalizar discrepâncias grandes
    
    Parâmetros:
    -----------
    p : np.ndarray
        Primeira distribuição de probabilidade
    q : np.ndarray
        Segunda distribuição de probabilidade
    epsilon : float
        Parâmetro de regularização para estabilidade numérica (padrão: 1e-10)
    lambda_suavizacao : float
        Parâmetro de suavização exponencial (padrão: 0.5)
        - λ maior: penaliza menos discrepâncias grandes
        - λ menor: mais sensível a discrepâncias
    normalizar : bool
        Se True, normaliza as distribuições antes do cálculo
        
    Retorna:
    --------
    float
        Valor da Divergência W (sempre >= 0)
        
    Propriedades:
    -------------
    1. W(P, P) = 0 (identidade dos indiscerníveis)
    2. W(P, Q) = W(Q, P) (simetria)
    3. W(P, Q) >= 0 (não-negatividade)
    4. Numericamente estável mesmo quando P(x) ou Q(x) = 0
    
    Exemplo:
    --------
    >>> import numpy as np
    >>> p = np.array([0.4, 0.3, 0.2, 0.1])
    >>> q = np.array([0.25, 0.25, 0.25, 0.25])
    >>> w = calcular_w(p, q)
    >>> print(f"Divergência W: {w:.6f}")
    """
    p, q = validar_distribuicoes(p, q)
    
    if normalizar:
        p = normalizar_distribuicao(p, epsilon)
        q = normalizar_distribuicao(q, epsilon)
    
    # Diferença entre distribuições
    diferenca = p - q
    diferenca_abs = np.abs(diferenca)
    diferenca_quadrado = diferenca ** 2
    
    # Termo do denominador (estabilização)
    denominador = p + q + epsilon
    
    # Termo Chi-quadrado modificado
    termo_chi = diferenca_quadrado / denominador
    
    # Fator de suavização exponencial
    fator_suavizacao = np.exp(-lambda_suavizacao * diferenca_abs)
    
    # Divergência W final
    w = np.sum(termo_chi * fator_suavizacao)
    
    return float(w)


def calcular_w_continua(p: np.ndarray, q: np.ndarray,
                        dx: float = 1.0,
                        epsilon: float = EPSILON_PADRAO,
                        lambda_suavizacao: float = LAMBDA_PADRAO) -> float:
    """
    Calcula a Divergência W para distribuições contínuas discretizadas.
    
    Esta versão considera o espaçamento entre pontos (dx) para
    aproximar a integral da divergência.
    
    Fórmula:
    --------
    W_c(P, Q) ≈ ∫ [(p(x) - q(x))² / (p(x) + q(x) + ε)] × exp(-λ|p(x) - q(x)|) dx
    
    Parâmetros:
    -----------
    p : np.ndarray
        Densidades avaliadas para a primeira distribuição
    q : np.ndarray
        Densidades avaliadas para a segunda distribuição
    dx : float
        Espaçamento entre pontos de avaliação
    epsilon : float
        Parâmetro de regularização
    lambda_suavizacao : float
        Parâmetro de suavização
        
    Retorna:
    --------
    float
        Valor aproximado da integral da Divergência W
    """
    p, q = validar_distribuicoes(p, q)
    
    # Para densidades contínuas, não normalizamos da mesma forma
    p = np.maximum(p, 0)
    q = np.maximum(q, 0)
    
    diferenca = p - q
    diferenca_abs = np.abs(diferenca)
    diferenca_quadrado = diferenca ** 2
    
    denominador = p + q + epsilon
    termo_chi = diferenca_quadrado / denominador
    fator_suavizacao = np.exp(-lambda_suavizacao * diferenca_abs)
    
    # Integração por regra do trapézio
    integrando = termo_chi * fator_suavizacao
    w_integral = np.trapz(integrando, dx=dx)
    
    return float(w_integral)


def calcular_jensen_shannon(p: np.ndarray, q: np.ndarray,
                            epsilon: float = EPSILON_PADRAO) -> float:
    """
    Calcula a Divergência de Jensen-Shannon (para comparação).
    
    JSD é a versão simétrica da KL:
    JSD(P, Q) = 0.5 * D_KL(P || M) + 0.5 * D_KL(Q || M)
    onde M = 0.5 * (P + Q)
    
    Parâmetros:
    -----------
    p, q : np.ndarray
        Distribuições de probabilidade
    epsilon : float
        Parâmetro de regularização
        
    Retorna:
    --------
    float
        Valor da divergência JSD (0 <= JSD <= log(2))
    """
    p, q = validar_distribuicoes(p, q)
    p = normalizar_distribuicao(p, epsilon)
    q = normalizar_distribuicao(q, epsilon)
    
    # Mistura M
    m = 0.5 * (p + q)
    
    # JSD
    jsd = 0.5 * calcular_kl(p, m, epsilon, normalizar=False) + \
          0.5 * calcular_kl(q, m, epsilon, normalizar=False)
    
    return float(jsd)


def calcular_hellinger(p: np.ndarray, q: np.ndarray,
                       epsilon: float = EPSILON_PADRAO) -> float:
    """
    Calcula a Distância de Hellinger (para comparação).
    
    H(P, Q) = (1/√2) * √(Σ(√P(x) - √Q(x))²)
    
    Parâmetros:
    -----------
    p, q : np.ndarray
        Distribuições de probabilidade
    epsilon : float
        Parâmetro de regularização
        
    Retorna:
    --------
    float
        Distância de Hellinger (0 <= H <= 1)
    """
    p, q = validar_distribuicoes(p, q)
    p = normalizar_distribuicao(p, epsilon)
    q = normalizar_distribuicao(q, epsilon)
    
    raiz_p = np.sqrt(p)
    raiz_q = np.sqrt(q)
    
    hellinger = np.sqrt(0.5 * np.sum((raiz_p - raiz_q) ** 2))
    
    return float(hellinger)


def decomposicao_w(p: np.ndarray, q: np.ndarray,
                   epsilon: float = EPSILON_PADRAO,
                   lambda_suavizacao: float = LAMBDA_PADRAO) -> dict:
    """
    Retorna a decomposição termo a termo da Divergência W.
    
    Útil para análise e interpretação da contribuição de cada
    ponto para a divergência total.
    
    Parâmetros:
    -----------
    p, q : np.ndarray
        Distribuições de probabilidade
    epsilon, lambda_suavizacao : float
        Parâmetros da Divergência W
        
    Retorna:
    --------
    dict
        Dicionário com:
        - 'contribuicoes': contribuição de cada ponto
        - 'termo_chi': termo chi-quadrado para cada ponto
        - 'fator_suavizacao': fator exponencial para cada ponto
        - 'divergencia_total': soma das contribuições
    """
    p, q = validar_distribuicoes(p, q)
    p = normalizar_distribuicao(p, epsilon)
    q = normalizar_distribuicao(q, epsilon)
    
    diferenca = p - q
    diferenca_abs = np.abs(diferenca)
    diferenca_quadrado = diferenca ** 2
    
    denominador = p + q + epsilon
    termo_chi = diferenca_quadrado / denominador
    fator_suavizacao = np.exp(-lambda_suavizacao * diferenca_abs)
    contribuicoes = termo_chi * fator_suavizacao
    
    return {
        'contribuicoes': contribuicoes,
        'termo_chi': termo_chi,
        'fator_suavizacao': fator_suavizacao,
        'divergencia_total': np.sum(contribuicoes),
        'p_normalizado': p,
        'q_normalizado': q
    }


if __name__ == "__main__":
    # Teste básico
    print("=" * 60)
    print("TESTE DO MÓDULO MATEMÁTICO - DIVERGÊNCIA W")
    print("=" * 60)
    
    # Distribuições de exemplo
    p = np.array([0.4, 0.3, 0.2, 0.1])
    q = np.array([0.25, 0.25, 0.25, 0.25])
    
    print(f"\nDistribuição P: {p}")
    print(f"Distribuição Q: {q}")
    
    # Cálculos
    kl_pq = calcular_kl(p, q)
    kl_qp = calcular_kl(q, p)
    w_pq = calcular_w(p, q)
    w_qp = calcular_w(q, p)
    jsd = calcular_jensen_shannon(p, q)
    hellinger = calcular_hellinger(p, q)
    
    print(f"\n--- Resultados ---")
    print(f"KL(P || Q): {kl_pq:.6f}")
    print(f"KL(Q || P): {kl_qp:.6f}")
    print(f"Assimetria KL: |KL(P||Q) - KL(Q||P)| = {abs(kl_pq - kl_qp):.6f}")
    
    print(f"\nW(P, Q): {w_pq:.6f}")
    print(f"W(Q, P): {w_qp:.6f}")
    print(f"Simetria W: |W(P,Q) - W(Q,P)| = {abs(w_pq - w_qp):.2e}")
    
    print(f"\nJensen-Shannon: {jsd:.6f}")
    print(f"Hellinger: {hellinger:.6f}")
    
    # Teste com zeros
    print("\n--- Teste de Robustez (com zeros) ---")
    p_zeros = np.array([0.5, 0.5, 0.0, 0.0])
    q_zeros = np.array([0.0, 0.0, 0.5, 0.5])
    
    w_zeros = calcular_w(p_zeros, q_zeros)
    print(f"P com zeros: {p_zeros}")
    print(f"Q com zeros: {q_zeros}")
    print(f"W(P, Q) com zeros: {w_zeros:.6f} (estável!)")
    
    print("\n✓ Todos os testes básicos concluídos com sucesso!")
