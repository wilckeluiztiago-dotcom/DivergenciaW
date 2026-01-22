# -*- coding: utf-8 -*-
"""
Divergência W - Gerador de Cenários para Testes
Autor: Luiz Tiago Wilcke
"""

import numpy as np

class GeradorCenarios:
    """Gera dados sintéticos com anomalias e mudanças de padrão controladas."""
    
    @staticmethod
    def gerar_serie_com_anomalia(n_pontos=1000, ponto_anomalia=500, duracao=50):
        """Série temporal normal que sofre uma perturbação temporária."""
        np.random.seed(42)
        # Regime normal: Ruído gaussiano
        dados = np.random.normal(0, 1, n_pontos)
        
        # Inserir anomalia: Aumento de variância e média
        fim_anomalia = min(n_pontos, ponto_anomalia + duracao)
        dados[ponto_anomalia:fim_anomalia] = np.random.normal(2, 3, fim_anomalia - ponto_anomalia)
        
        return dados
        
    @staticmethod
    def gerar_data_drift(n_amostras=1000, intensidade_drift=0.5):
        """Gera dataset de treinamento (baseline) e produção (com drift)."""
        np.random.seed(100)
        
        # Baseline: Mistura de duas gaussianas
        baseline = np.concatenate([
            np.random.normal(-2, 1, n_amostras // 2),
            np.random.normal(2, 1, n_amostras // 2)
        ])
        
        # Produção com Drift: As médias se deslocam
        producao = np.concatenate([
            np.random.normal(-2 + intensidade_drift, 1, n_amostras // 2),
            np.random.normal(2 + intensidade_drift, 1.5, n_amostras // 2) # Aumenta variância também
        ])
        
        return baseline, producao
        
    @staticmethod
    def gerar_dados_financeiros_sinteticos(n_dias=1000):
        """Simula preços de ativos com mudanças de regime (calmo -> crise -> calmo)."""
        np.random.seed(2026)
        
        # Regime 1: Calmo (Bull Market constante)
        retornos_calmo = np.random.normal(0.0005, 0.01, 400) # Média positiva leve, baixa vol
        
        # Regime 2: Crise (Alta volatilidade, média negativa)
        retornos_crise = np.random.normal(-0.002, 0.04, 200) # Crash
        
        # Regime 3: Recuperação (Volatilidade média)
        retornos_recup = np.random.normal(0.001, 0.015, 400)
        
        retornos = np.concatenate([retornos_calmo, retornos_crise, retornos_recup])
        
        # Reconstrói preços
        preco_inicial = 100
        precos = [preco_inicial]
        for r in retornos:
            precos.append(precos[-1] * np.exp(r))
            
        return np.array(precos), np.array(retornos)
