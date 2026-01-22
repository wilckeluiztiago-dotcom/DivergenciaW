# Divergência W

**Uma Nova Medida de Divergência Estatística**

**Autor:** Luiz Tiago Wilcke

---

## Descrição

A **Divergência W** é uma nova medida de divergência estatística proposta para ser mais eficiente e robusta que a Divergência de Kullback-Leibler (KL). Este projeto implementa a Divergência W em Python, juntamente com comparações de performance e documentação detalhada.

## Motivação

A Divergência de Kullback-Leibler (KL) é amplamente utilizada em estatística e aprendizado de máquina, mas possui limitações importantes:

1. **Assimetria:** KL(P || Q) ≠ KL(Q || P)
2. **Instabilidade:** Diverge para infinito quando Q(x) → 0 e P(x) > 0
3. **Sensibilidade a zeros:** Problemas numéricos com distribuições esparsas

A Divergência W foi projetada para superar essas limitações.

## Definição Matemática

A Divergência W é definida como:

![Divergência W](https://latex.codecogs.com/png.latex?%5Cdpi%7B150%7D%20W%28P%2C%20Q%29%20%3D%20%5Csum_%7Bx%7D%20%5Cfrac%7B%28P%28x%29%20-%20Q%28x%29%29%5E2%7D%7BP%28x%29%20%2B%20Q%28x%29%20%2B%20%5Cepsilon%7D%20%5Ctimes%20%5Cexp%28-%5Clambda%20%7CP%28x%29%20-%20Q%28x%29%7C%29)

Onde:
- **P(x), Q(x):** Distribuições de probabilidade
- **ε:** Parâmetro de regularização para estabilidade numérica (padrão: 10⁻¹⁰)
- **λ:** Parâmetro de suavização exponencial (padrão: 0.5)

### Propriedades

| Propriedade | Divergência W | KL |
|-------------|---------------|-----|
| Simetria | ✓ W(P,Q) = W(Q,P) | ✗ |
| Robustez a zeros | ✓ Estável | ✗ Diverge |
| Não-negatividade | ✓ W ≥ 0 | ✓ |
| Identidade | ✓ W(P,P) = 0 | ✓ |

## Estrutura do Projeto

```
DivergenciaW/
├── divergencia_w/
│   ├── __init__.py          # Inicialização do pacote
│   ├── matematica.py        # Funções matemáticas centrais
│   ├── aplicacoes/          # [NOVO] Módulos de aplicação prática
│   │   ├── detector_anomalias.py
│   │   ├── monitor_data_drift.py
│   │   └── analise_financeira.py
│   ├── gerador_dados.py     # Geração de distribuições sintéticas
│   ├── analise.py           # Benchmarks e testes
│   └── visualizacao.py      # Gráficos comparativos
├── main.py                   # Script principal
├── demo_aplicacoes.py        # [NOVO] Demo de casos de uso reais
├── README.md                 # Esta documentação
└── graficos/                 # Gráficos gerados
```

## Instalação

```bash
# Dependências
pip install numpy scipy matplotlib seaborn
```

## Uso

### Exemplo Básico

```python
import numpy as np
from divergencia_w import calcular_w, calcular_kl

# Distribuições de probabilidade
p = np.array([0.4, 0.3, 0.2, 0.1])
q = np.array([0.25, 0.25, 0.25, 0.25])

# Calcular divergências
w = calcular_w(p, q)
kl = calcular_kl(p, q)

print(f"Divergência W: {w:.6f}")
print(f"Divergência KL: {kl:.6f}")
```

### Verificar Simetria

```python
from divergencia_w import calcular_w, calcular_kl

# W é simétrica
print(f"W(P,Q) = {calcular_w(p, q):.6f}")
print(f"W(Q,P) = {calcular_w(q, p):.6f}")  # Mesmo valor!

# KL é assimétrica
print(f"KL(P||Q) = {calcular_kl(p, q):.6f}")
print(f"KL(Q||P) = {calcular_kl(q, p):.6f}")  # Valores diferentes!
```

### Executar Demonstração Completa

```bash
python main.py
```

## Resultados

### Benchmark de Performance

A Divergência W apresenta performance competitiva com as demais divergências:

| Método | Tempo Médio |
|--------|-------------|
| Divergência W | ~0.05 ms |
| KL | ~0.04 ms |
| Jensen-Shannon | ~0.08 ms |
| Hellinger | ~0.03 ms |

### Teste de Simetria

Em 100 testes com distribuições aleatórias:
- **Divergência W:** |W(P,Q) - W(Q,P)| < 10⁻¹⁵ ✓
- **KL:** |KL(P||Q) - KL(Q||P)| ≈ 0.1 a 1.0 (assimétrica)

### Estabilidade com Zeros

| % de Zeros | W Estável | KL Estável |
|------------|-----------|------------|
| 0% | ✓ | ✓ |
| 50% | ✓ | ✓ |
| 90% | ✓ | ⚠ |
| 95% | ✓ | ✗ |

## Parâmetros

### Epsilon (ε)

Controla a estabilidade numérica no denominador:
- **Valores típicos:** 10⁻¹⁵ a 10⁻⁵
- **Padrão:** 10⁻¹⁰

### Lambda (λ)

Controla a suavização exponencial:
- **λ pequeno (0.1):** Mais sensível a discrepâncias
- **λ grande (2.0):** Menos penalização para diferenças grandes
- **Padrão:** 0.5

- **Padrão:** 0.5

## Aplicações Práticas

O projeto inclui módulos para aplicação em cenários reais (`divergencia_w/aplicacoes/`):

### 1. Detecção de Anomalias
Detecta perturbações em séries temporais usando janelas deslizantes.
```python
from divergencia_w.aplicacoes import DetectorAnomalias
detector = DetectorAnomalias()
anomalias, scores = detector.detectar(serie_temporal)
```

### 2. Monitoramento de Data Drift
Monitora a qualidade de dados em produção comparando com um baseline. A Divergência W oferece scores mais estáveis e interpretáveis que KL para este fim.

### 3. Análise Financeira
Detecta mudanças de regime (ex: baixa volatilidade para crise) em ativos financeiros.

Para rodar a demonstração completa dessas aplicações:
```bash
python demo_aplicacoes.py
```

## Comparação com Outras Divergências

### Kullback-Leibler (KL)

![KL](https://latex.codecogs.com/png.latex?%5Cdpi%7B120%7D%20D_%7BKL%7D%28P%20%7C%7C%20Q%29%20%3D%20%5Csum_x%20P%28x%29%20%5Clog%20%5Cfrac%7BP%28x%29%7D%7BQ%28x%29%7D)

### Jensen-Shannon (JSD)

![JSD](https://latex.codecogs.com/png.latex?%5Cdpi%7B120%7D%20JSD%28P%2C%20Q%29%20%3D%20%5Cfrac%7B1%7D%7B2%7D%20D_%7BKL%7D%28P%20%7C%7C%20M%29%20%2B%20%5Cfrac%7B1%7D%7B2%7D%20D_%7BKL%7D%28Q%20%7C%7C%20M%29)

onde M = (P + Q) / 2

### Hellinger

![Hellinger](https://latex.codecogs.com/png.latex?%5Cdpi%7B120%7D%20H%28P%2C%20Q%29%20%3D%20%5Cfrac%7B1%7D%7B%5Csqrt%7B2%7D%7D%20%5Csqrt%7B%5Csum_x%20%28%5Csqrt%7BP%28x%29%7D%20-%20%5Csqrt%7BQ%28x%29%7D%29%5E2%7D)

## Licença

Este projeto está licenciado sob a Licença MIT.

---

**© 2026 Luiz Tiago Wilcke**
