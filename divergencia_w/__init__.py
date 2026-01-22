# -*- coding: utf-8 -*-
"""
Divergência W - Uma Nova Medida de Divergência Estatística
Autor: Luiz Tiago Wilcke
"""

from .matematica import calcular_kl, calcular_w, calcular_w_continua
from .gerador_dados import (
    gerar_gaussiana,
    gerar_uniforme,
    gerar_esparsa,
    gerar_bimodal,
    gerar_exponencial
)
from .analise import (
    benchmark_divergencias,
    testar_simetria,
    testar_estabilidade_zeros,
    comparar_cenarios
)
from .visualizacao import (
    plotar_comparacao_divergencias,
    plotar_benchmark_tempo,
    plotar_estabilidade,
    plotar_distribuicoes
)

__version__ = "1.0.0"
__author__ = "Luiz Tiago Wilcke"
