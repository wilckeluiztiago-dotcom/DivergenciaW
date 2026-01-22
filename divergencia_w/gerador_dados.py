# -*- coding: utf-8 -*-
"""
Divergência W - Módulo de Geração de Dados
Autor: Luiz Tiago Wilcke

Este módulo gera distribuições de probabilidade sintéticas para
testes e demonstrações da Divergência W.
"""

import numpy as np
from typing import Tuple, Optional, List
from scipy import stats


def gerar_gaussiana(n_pontos: int = 100,
                    media: float = 0.0,
                    desvio: float = 1.0,
                    limites: Tuple[float, float] = (-5, 5),
                    seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gera uma distribuição Gaussiana (Normal) discretizada.
    
    Parâmetros:
    -----------
    n_pontos : int
        Número de pontos na discretização
    media : float
        Média da distribuição
    desvio : float
        Desvio padrão
    limites : Tuple[float, float]
        Intervalo de discretização
    seed : Optional[int]
        Semente para reprodutibilidade
        
    Retorna:
    --------
    Tuple[np.ndarray, np.ndarray]
        (pontos_x, probabilidades)
    """
    if seed is not None:
        np.random.seed(seed)
    
    x = np.linspace(limites[0], limites[1], n_pontos)
    pdf = stats.norm.pdf(x, loc=media, scale=desvio)
    probabilidades = pdf / np.sum(pdf)  # Normalização
    
    return x, probabilidades


def gerar_uniforme(n_pontos: int = 100,
                   limites: Tuple[float, float] = (0, 1)) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gera uma distribuição Uniforme discretizada.
    
    Parâmetros:
    -----------
    n_pontos : int
        Número de pontos
    limites : Tuple[float, float]
        Intervalo da distribuição uniforme
        
    Retorna:
    --------
    Tuple[np.ndarray, np.ndarray]
        (pontos_x, probabilidades)
    """
    x = np.linspace(limites[0], limites[1], n_pontos)
    probabilidades = np.ones(n_pontos) / n_pontos
    
    return x, probabilidades


def gerar_esparsa(n_pontos: int = 100,
                  proporcao_zeros: float = 0.7,
                  seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gera uma distribuição esparsa (muitos zeros).
    
    Este tipo de distribuição é útil para testar a robustez
    de medidas de divergência quando há muitos zeros.
    
    Parâmetros:
    -----------
    n_pontos : int
        Número de pontos
    proporcao_zeros : float
        Proporção de zeros (0 a 1)
    seed : Optional[int]
        Semente para reprodutibilidade
        
    Retorna:
    --------
    Tuple[np.ndarray, np.ndarray]
        (pontos_x, probabilidades)
    """
    if seed is not None:
        np.random.seed(seed)
    
    x = np.arange(n_pontos)
    
    # Gerar valores aleatórios
    valores = np.random.exponential(1, n_pontos)
    
    # Zerar uma proporção dos valores
    n_zeros = int(n_pontos * proporcao_zeros)
    indices_zeros = np.random.choice(n_pontos, n_zeros, replace=False)
    valores[indices_zeros] = 0
    
    # Normalizar (se houver valores não-zero)
    if np.sum(valores) > 0:
        probabilidades = valores / np.sum(valores)
    else:
        probabilidades = np.ones(n_pontos) / n_pontos
    
    return x, probabilidades


def gerar_bimodal(n_pontos: int = 100,
                  media1: float = -2.0,
                  media2: float = 2.0,
                  desvio: float = 0.8,
                  peso1: float = 0.5,
                  limites: Tuple[float, float] = (-6, 6),
                  seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gera uma distribuição bimodal (mistura de duas Gaussianas).
    
    Parâmetros:
    -----------
    n_pontos : int
        Número de pontos
    media1, media2 : float
        Médias das duas componentes
    desvio : float
        Desvio padrão (comum às duas componentes)
    peso1 : float
        Peso da primeira componente (0 a 1)
    limites : Tuple[float, float]
        Intervalo de discretização
    seed : Optional[int]
        Semente para reprodutibilidade
        
    Retorna:
    --------
    Tuple[np.ndarray, np.ndarray]
        (pontos_x, probabilidades)
    """
    if seed is not None:
        np.random.seed(seed)
    
    x = np.linspace(limites[0], limites[1], n_pontos)
    
    # Mistura de duas Gaussianas
    pdf1 = stats.norm.pdf(x, loc=media1, scale=desvio)
    pdf2 = stats.norm.pdf(x, loc=media2, scale=desvio)
    
    pdf_mistura = peso1 * pdf1 + (1 - peso1) * pdf2
    probabilidades = pdf_mistura / np.sum(pdf_mistura)
    
    return x, probabilidades


def gerar_exponencial(n_pontos: int = 100,
                      escala: float = 1.0,
                      limites: Tuple[float, float] = (0, 10)) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gera uma distribuição Exponencial discretizada.
    
    Parâmetros:
    -----------
    n_pontos : int
        Número de pontos
    escala : float
        Parâmetro de escala (1/lambda)
    limites : Tuple[float, float]
        Intervalo de discretização
        
    Retorna:
    --------
    Tuple[np.ndarray, np.ndarray]
        (pontos_x, probabilidades)
    """
    x = np.linspace(limites[0], limites[1], n_pontos)
    x = np.maximum(x, 0)  # Exponencial só para x >= 0
    
    pdf = stats.expon.pdf(x, scale=escala)
    probabilidades = pdf / np.sum(pdf)
    
    return x, probabilidades


def gerar_poisson(n_pontos: int = 20,
                  lambda_param: float = 5.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gera uma distribuição de Poisson discreta.
    
    Parâmetros:
    -----------
    n_pontos : int
        Número de pontos (k = 0, 1, 2, ..., n_pontos-1)
    lambda_param : float
        Parâmetro λ da Poisson
        
    Retorna:
    --------
    Tuple[np.ndarray, np.ndarray]
        (valores_k, probabilidades)
    """
    k = np.arange(n_pontos)
    probabilidades = stats.poisson.pmf(k, lambda_param)
    
    # Renormalizar para garantir soma = 1
    probabilidades = probabilidades / np.sum(probabilidades)
    
    return k, probabilidades


def gerar_beta(n_pontos: int = 100,
               alpha: float = 2.0,
               beta: float = 5.0,
               limites: Tuple[float, float] = (0, 1)) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gera uma distribuição Beta discretizada.
    
    Parâmetros:
    -----------
    n_pontos : int
        Número de pontos
    alpha, beta : float
        Parâmetros de forma da distribuição Beta
    limites : Tuple[float, float]
        Intervalo (deve estar em [0, 1])
        
    Retorna:
    --------
    Tuple[np.ndarray, np.ndarray]
        (pontos_x, probabilidades)
    """
    x = np.linspace(limites[0] + 1e-6, limites[1] - 1e-6, n_pontos)
    
    pdf = stats.beta.pdf(x, alpha, beta)
    probabilidades = pdf / np.sum(pdf)
    
    return x, probabilidades


def gerar_par_gaussianas(n_pontos: int = 100,
                         separacao_medias: float = 2.0,
                         desvio: float = 1.0,
                         limites: Tuple[float, float] = (-5, 5),
                         seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Gera um par de distribuições Gaussianas com médias separadas.
    
    Útil para testar como as divergências capturam diferenças entre distribuições.
    
    Parâmetros:
    -----------
    n_pontos : int
        Número de pontos
    separacao_medias : float
        Distância entre as médias das duas Gaussianas
    desvio : float
        Desvio padrão comum
    limites : Tuple[float, float]
        Intervalo de discretização
    seed : Optional[int]
        Semente para reprodutibilidade
        
    Retorna:
    --------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        (pontos_x, probabilidades_p, probabilidades_q)
    """
    if seed is not None:
        np.random.seed(seed)
    
    x = np.linspace(limites[0], limites[1], n_pontos)
    
    media_p = -separacao_medias / 2
    media_q = separacao_medias / 2
    
    pdf_p = stats.norm.pdf(x, loc=media_p, scale=desvio)
    pdf_q = stats.norm.pdf(x, loc=media_q, scale=desvio)
    
    prob_p = pdf_p / np.sum(pdf_p)
    prob_q = pdf_q / np.sum(pdf_q)
    
    return x, prob_p, prob_q


def gerar_cenarios_teste() -> List[dict]:
    """
    Gera uma lista de cenários de teste padrão.
    
    Retorna:
    --------
    List[dict]
        Lista de dicionários com cenários de teste contendo:
        - 'nome': nome do cenário
        - 'p': primeira distribuição
        - 'q': segunda distribuição
        - 'x': pontos x (se aplicável)
    """
    cenarios = []
    n = 100
    
    # Cenário 1: Gaussianas com médias diferentes
    x1, p1 = gerar_gaussiana(n, media=0, desvio=1, seed=42)
    _, q1 = gerar_gaussiana(n, media=1, desvio=1, seed=42)
    cenarios.append({
        'nome': 'Gaussianas (μ=0 vs μ=1)',
        'p': p1, 'q': q1, 'x': x1
    })
    
    # Cenário 2: Gaussiana vs Uniforme
    x2, p2 = gerar_gaussiana(n, media=0, desvio=1, seed=42)
    _, q2 = gerar_uniforme(n, limites=(-5, 5))
    cenarios.append({
        'nome': 'Gaussiana vs Uniforme',
        'p': p2, 'q': q2, 'x': x2
    })
    
    # Cenário 3: Distribuições idênticas
    x3, p3 = gerar_gaussiana(n, media=0, desvio=1, seed=42)
    cenarios.append({
        'nome': 'Idênticas (P == Q)',
        'p': p3, 'q': p3.copy(), 'x': x3
    })
    
    # Cenário 4: Distribuições esparsas
    x4, p4 = gerar_esparsa(n, proporcao_zeros=0.7, seed=42)
    _, q4 = gerar_esparsa(n, proporcao_zeros=0.7, seed=123)
    cenarios.append({
        'nome': 'Esparsas (70% zeros)',
        'p': p4, 'q': q4, 'x': x4
    })
    
    # Cenário 5: Bimodal vs Unimodal
    x5, p5 = gerar_bimodal(n, media1=-2, media2=2, seed=42)
    _, q5 = gerar_gaussiana(n, media=0, desvio=2, seed=42)
    cenarios.append({
        'nome': 'Bimodal vs Unimodal',
        'p': p5, 'q': q5, 'x': x5
    })
    
    # Cenário 6: Exponenciais com escalas diferentes
    x6, p6 = gerar_exponencial(n, escala=1.0)
    _, q6 = gerar_exponencial(n, escala=2.0)
    cenarios.append({
        'nome': 'Exponenciais (escala=1 vs 2)',
        'p': p6, 'q': q6, 'x': x6
    })
    
    # Cenário 7: Quase-zeros (caso extremo)
    p7 = np.array([0.5, 0.5, 0.0, 0.0, 0.0])
    q7 = np.array([0.0, 0.0, 0.0, 0.5, 0.5])
    cenarios.append({
        'nome': 'Suporte Disjunto',
        'p': p7, 'q': q7, 'x': np.arange(5)
    })
    
    return cenarios


if __name__ == "__main__":
    print("=" * 60)
    print("TESTE DO MÓDULO GERADOR DE DADOS")
    print("=" * 60)
    
    # Testar cada gerador
    print("\n1. Gaussiana:")
    x, p = gerar_gaussiana(n_pontos=10, media=0, desvio=1)
    print(f"   x: {x[:5]}...")
    print(f"   p: {p[:5]}... (soma = {np.sum(p):.6f})")
    
    print("\n2. Uniforme:")
    x, p = gerar_uniforme(n_pontos=10)
    print(f"   p: {p[:5]}... (soma = {np.sum(p):.6f})")
    
    print("\n3. Esparsa (70% zeros):")
    x, p = gerar_esparsa(n_pontos=10, proporcao_zeros=0.7, seed=42)
    print(f"   p: {p} (zeros: {np.sum(p == 0)})")
    
    print("\n4. Bimodal:")
    x, p = gerar_bimodal(n_pontos=10)
    print(f"   p: {p[:5]}... (soma = {np.sum(p):.6f})")
    
    print("\n5. Cenários de teste:")
    cenarios = gerar_cenarios_teste()
    for i, c in enumerate(cenarios):
        print(f"   {i+1}. {c['nome']}")
    
    print("\n✓ Geração de dados concluída com sucesso!")
