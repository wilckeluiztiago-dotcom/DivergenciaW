# -*- coding: utf-8 -*-
"""
Divergência W - Análise Financeira e de Mercado
Autor: Luiz Tiago Wilcke
"""

import numpy as np
from ..core.matematica_base import calcular_w

class AnaliseFinanceira:
    """
    Ferramentas para análise de séries financeiras usando Divergência W.
    Foca na detecção de mudanças de regime (regime switching) e eventos de cauda.
    """
    
    def __init__(self, window_size=252):
        self.window_size = window_size
        
    def calcular_retornos(self, precos):
        """Calcula retornos logarítmicos."""
        precos = np.array(precos)
        retornos = np.diff(np.log(precos))
        # Remove NaNs e inits
        return retornos[~np.isnan(retornos)]
        
    def detectar_mudanca_regime(self, retornos):
        """
        Detecta mudanças de regime de volatilidade comparando
        a distribuição do último mês com o último ano.
        """
        # Janela Curta (ex: 21 dias - 1 mês) vs Janela Longa (ex: 252 dias - 1 ano)
        janela_curta = 21
        janela_longa = 252
        
        scores_regime = []
        indices = []
        
        n = len(retornos)
        if n < janela_longa:
            return [], []
            
        for i in range(janela_longa, n):
            # Baseline: Último ano até o mês atual
            # Teste: Último mês
            
            # Para evitar overlap excessivo que mascara mudanças, comparamos:
            # Baseline: [i-longa : i-curta]
            # Teste: [i-curta : i]
            
            base_data = retornos[i - janela_longa : i - janela_curta]
            teste_data = retornos[i - janela_curta : i]
            
            p = self._kde_simples(base_data)
            q = self._kde_simples(teste_data)
            
            w_score = calcular_w(p, q)
            scores_regime.append(w_score)
            indices.append(i)
            
        return indices, scores_regime

    def _kde_simples(self, dados, bins=50):
        """Estimativa de densidade simples via histograma suavizado."""
        # Range fixo para comparações financeiras padronizadas (ex: -10% a +10%)
        # Mas adaptativo é melhor se normalizarmos os dados.
        # Vamos usar range fixo de -0.15 a 0.15 que cobre a maioria dos movimentos diários
        range_min, range_max = -0.15, 0.15
        
        if len(dados) > 0:
             # Ajuste fino se dados excederem
            range_min = min(range_min, np.min(dados))
            range_max = max(range_max, np.max(dados))
            
        hist, _ = np.histogram(dados, bins=bins, range=(range_min, range_max), density=True)
        return np.maximum(hist, 1e-10) / np.sum(np.maximum(hist, 1e-10))
