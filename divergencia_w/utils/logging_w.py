# -*- coding: utf-8 -*-
"""
Divergência W - Logging
Autor: Luiz Tiago Wilcke
"""
import logging

def configurar_logger():
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger("DivergenciaW")
