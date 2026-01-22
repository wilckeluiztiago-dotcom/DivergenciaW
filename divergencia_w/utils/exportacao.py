# -*- coding: utf-8 -*-
"""
Divergência W - Exportação
Autor: Luiz Tiago Wilcke
"""
import pandas as pd

def exportar_resultados_w(resultados: dict, path: str):
    """Exporta resultados para CSV."""
    df = pd.DataFrame([resultados])
    df.to_csv(path, index=False)
