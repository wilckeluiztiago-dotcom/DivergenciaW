# -*- coding: utf-8 -*-
from .matematica_base import (
    calcular_kl, calcular_w, calcular_jensen_shannon, 
    calcular_hellinger, normalizar_distribuicao, validar_distribuicoes
)
from .tensores import operacoes_tensorais_w
from .integracao import integrar_w
from .derivadas import gradiente_w
from .espacos_metricos import verificar_axiomas_metrica
from .otimizacao import otimizar_parametros_w
from .algebra_linear import projetar_no_simplex
from .probabilidade_base import entropia_shannon
from .regularizacao import aplicar_suavizacao
from .estabilidade import verificar_estabilidade_numerica
