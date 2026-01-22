# -*- coding: utf-8 -*-
"""
Divergência W - Detector de Anomalias em Séries Temporais
Autor: Luiz Tiago Wilcke
"""

import numpy as np
from ..core.matematica_base import calcular_w, calcular_kl

class DetectorAnomalias:
    """
    Detector de anomalias em séries temporais usando divergência estatística.
    Compara a distribuição em uma janela de referência com uma janela atual.
    """
    
    def __init__(self, tamanho_janela=50, threshold=0.1, usar_w=True):
        self.tamanho_janela = tamanho_janela
        self.threshold = threshold
        self.usar_w = usar_w
        self.historico_divergencias = []
        
    def _calcular_distribuicao(self, dados):
        """Converte dados numéricos em uma distribuição de probabilidade suave."""
        # Histograma simples
        hist, _ = np.histogram(dados, bins=10, density=True)
        # Suavização para evitar zeros
        hist = np.maximum(hist, 1e-10)
        return hist / np.sum(hist)
        
    def detectar(self, serie_temporal):
        """
        Processa uma série temporal e retorna índices de anomalias.
        
        Args:
            serie_temporal (array-like): Dados da série temporal
            
        Returns:
            list: Índices onde anomalias foram detectadas
            list: Valores de divergência calculados
        """
        dados = np.array(serie_temporal)
        n = len(dados)
        anomalias = []
        divergencias = [0.0] * self.tamanho_janela # Padding inicial
        
        # Janela deslizante
        for i in range(self.tamanho_janela, n):
            # Janela de Referência (anterior)
            janela_ref = dados[i-self.tamanho_janela:i]
            
            # Janela Atual (pequena, recente) - aqui usamos uma sub-janela recente
            # para comparar com o histórico imediato.
            # Uma abordagem comum é comparar janela anterior vs janela atual 
            # mas aqui vamos comparar "janela de referência longa" vs "janela recente curta"
            # ou "metade anterior" vs "metade atual" da janela deslizante.
            
            # Abordagem: Window Splitting
            # Compara a distribuição da primeira metade da janela com a segunda metade?
            # Ou compara janela [i-W : i] com [i : i+W]?
            
            # Vamos usar: Referência Fixa Deslizante vs Ponto Atual (contextual)
            # Mas divergência precisa de distribuição.
            # Vamos comparar: Janela [i-W : i] vs Janela [i : i+W/2] (lookahead) 
            # ou mais simples: Janela [i-W : i] vs Janela [i-W/2 : i+W/2] (centrada)
            
            # Simplificação robusta para detecção ONLINE:
            # Referência: [i - N : i]
            # Teste: [i : i + M] (precisa de dados futuros) ou
            # Referência: [i - 2N : i - N]
            # Teste: [i - N : i]
            
            # Vamos usar Referência [i-2w : i-w] vs Teste [i-w : i]
            # Onde w = tamanho_janela // 2
            
            w_size = self.tamanho_janela
            if i < 2 * w_size:
                divergencias.append(0.0)
                continue
                
            ref_data = dados[i - 2*w_size : i - w_size]
            teste_data = dados[i - w_size : i]
            
            p = self._calcular_distribuicao(ref_data)
            q = self._calcular_distribuicao(teste_data)
            
            if self.usar_w:
                div = calcular_w(p, q)
            else:
                div = calcular_kl(p, q)
                
            divergencias.append(div)
            
            if div > self.threshold:
                anomalias.append(i)
                
        self.historico_divergencias = divergencias
        return anomalias, divergencias
