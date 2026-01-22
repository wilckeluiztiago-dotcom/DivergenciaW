# -*- coding: utf-8 -*-
"""
Divergência W - Monitor de Data Drift
Autor: Luiz Tiago Wilcke
"""

import numpy as np
from ..core.matematica_base import calcular_w, calcular_kl

class MonitorDataDrift:
    """
    Monitora a qualidade dos dados (Data Drift) comparando dados de produção
    com um baseline (referência).
    """
    
    def __init__(self, baseline_data):
        self.baseline_data = np.array(baseline_data)
        self.baseline_dist = self._calcular_distribuicao(self.baseline_data)
        
    def _calcular_distribuicao(self, dados, bins=20):
        """Gera distribuição normalizada a partir dos dados."""
        # Usa range fixo baseado no baseline para garantir comparabilidade
        range_min = np.min(self.baseline_data)
        range_max = np.max(self.baseline_data)
        
        # Margem de segurança
        margem = (range_max - range_min) * 0.1
        bins_edges = np.linspace(range_min - margem, range_max + margem, bins + 1)
        
        hist, _ = np.histogram(dados, bins=bins_edges, density=True)
        return np.maximum(hist, 1e-10) / np.sum(np.maximum(hist, 1e-10))
        
    def verificar_drift(self, novos_dados):
        """
        Calcula o score de drift para novos dados.
        
        Args:
            novos_dados (array-like): Lote de novos dados de produção.
            
        Returns:
            dict: Resultados contendo score W, score KL e status.
        """
        novos_dados = np.array(novos_dados)
        prod_dist = self._calcular_distribuicao(novos_dados)
        
        score_w = calcular_w(self.baseline_dist, prod_dist)
        try:
            score_kl = calcular_kl(self.baseline_dist, prod_dist)
        except:
            score_kl = float('inf')
            
        # Classificação de severidade baseada em heurísticas
        nivel_drift = "Normal"
        if score_w > 0.5:
            nivel_drift = "CRÍTICO"
        elif score_w > 0.2:
            nivel_drift = "Moderado"
        elif score_w > 0.05:
            nivel_drift = "Leve"
            
        return {
            "drift_score_w": score_w,
            "drift_score_kl": score_kl,
            "nivel": nivel_drift,
            "tamanho_amostra": len(novos_dados)
        }
