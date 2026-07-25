# Evolução do pipeline de classificação — Task 05 (v3)

**Branch:** `task05_melhorias_pipeline_modelos`
**Data:** 24/07/2026
**Dataset:** `resultado_laudo_afericao.xlsx` (35.318 linhas, 50 classes originais)
**Público-alvo:** equipe técnica e de negócio

---

## Resumo executivo

A v3 substitui o limiar único e arbitrário de expurgo (15%, usado na v2) por um **sistema de camadas de qualificação de classes (A/B/C/D)**, mais escalável e ordenado, e roda o **pipeline `sklearn` completo** (pré-processamento + CatBoost + XGBoost + SMOTE/ADASYN) com dados reais — algo que não tinha sido possível nas versões v1/v2 por bloqueio de DLL do `scipy` nesta máquina.

Com o novo limiar padrão (camada C, 0,1%), o problema deixa de ser "50 classes → 2 classes" (v2) e passa a ser **"50 classes → 14 classes"**, retendo **99,25%** das linhas. O F1 macro sai de ~25% (baseline histórico de 41 classes) para **56%–61%** (14 classes, camadas A-C) — um salto de cobertura e de sinal útil por classe, mesmo sem chegar aos 100% "fáceis" da v2 (que só tinha 2 classes linearmente separáveis).

**Conclusões principais:**
- A camada B (1%-15%) é o principal "vencedor" do novo limiar: 5 classes com volume de centenas/milhares de amostras, com métricas por classe consistentemente boas (F1 ~0,64 a ~0,88).
- **SMOTE é neutro/marginal** — ganho pequeno no macro, efeito misto por classe.
- **ADASYN prejudica o desempenho geral** neste dataset — não é recomendado.
- A camada C (0,1%-1%) continua sendo uma **zona cinzenta de baixa confiabilidade** — poucas amostras de teste (algumas com menos de 15), métricas instáveis entre execuções e técnicas.

---

## O sistema de camadas A/B/C/D

A v2 usava um único limiar (`MIN_CLASS_PERCENTAGE_THRESHOLD = 15.0`) para decidir "fica ou descarta". Isso tem dois problemas: (1) é uma decisão binária que esconde a heterogeneidade das classes retidas/descartadas; (2) qualquer ajuste fino do limiar (ex.: baixar para 5%, para 1%) exige reinterpretar tudo do zero, sem que o código expresse *por que* aquele número foi escolhido.

A v3 introduz `CLASS_TIER_THRESHOLDS`, em `src/machine_learning/feature_engineering.py`, como fonte única de verdade das camadas:

```python
CLASS_TIER_THRESHOLDS = {"A": 15.0, "B": 1.0, "C": 0.1}
```

| Camada | Limiar (% do total de linhas) | Significado |
|---|---|---|
| **A** | ≥ 15% | Classes dominantes, sem necessidade de tratamento especial |
| **B** | 1% – 15% | Volume relevante (centenas a milhares de amostras); candidata legítima a SMOTE/ADASYN, pois há sinal real suficiente para interpolar com robustez |
| **C** | 0,1% – 1% | Volume marginal (dezenas a poucas centenas de amostras); zona cinzenta — augmentation só com validação cruzada rigorosa e ciência de que o ganho pode não generalizar |
| **D** (`CLASS_TIER_DISCARD_LABEL`) | < 0,1% | Cauda estatística — sempre descartada do treino; não há dados suficientes para nenhuma técnica produzir resultado confiável |

Novas funções que operacionalizam o sistema:
- `classify_class_tier(percentage)`: classifica um percentual isolado na camada correspondente.
- `qualify_class_distribution(distribution)`: decora uma distribuição de classes com a coluna `tier`.
- `compute_balanced_class_weights(distribution)`: calcula o peso "balanced" (`total / (n_classes * count)`) de cada classe.
- `build_class_weight_registry(df, target_column)`: monta o registro completo (camada + peso + resumo do descarte) para persistência.

Por que isso é mais escalável do que um limiar único: adicionar, remover ou reajustar uma camada é uma mudança de **uma linha** em `CLASS_TIER_THRESHOLDS`, e ela se propaga automaticamente para o expurgo (`MIN_CLASS_PERCENTAGE_THRESHOLD = CLASS_TIER_THRESHOLDS["C"]`), para os relatórios de distribuição (`data_exploration`), para as métricas por classe (`classification/workflow.py`) e para o registro de pesos — sem duplicar lógica em vários lugares. O sistema também documenta *o racional de negócio* de cada faixa (ver docstring de `CLASS_TIER_THRESHOLDS`), em vez de um número solto sem contexto.

`MIN_CLASS_PERCENTAGE_THRESHOLD` (usado em `data_preparation.py` e `classification/workflow.py` como limiar padrão de expurgo) foi alterado de `15.0` para `CLASS_TIER_THRESHOLDS["C"]` (`0.1`) — o padrão agora retém as camadas A+B+C e descarta apenas a camada D.

### `model/class_weight_registry.json` — insumo para uma futura narrativa via LLM

A cada execução de `data_preparation.prepare_training_dataset`, o registro de pesos por classe é persistido automaticamente em `model/class_weight_registry.json` (via `data_preparation.save_class_weight_registry`). Para cada classe retida, ele guarda `count`, `percentage`, `tier` e `weight` (peso "balanced").

> ⚠️ **Isso é insumo para uma feature futura, ainda não implementada**: a ideia é que uma camada de inferência componha, via LLM, um texto de retorno da API mencionando a confiabilidade/acurácia esperada para a classe prevista, com base na sua camada e peso (ex.: "esta classe pertence à camada C, com histórico de poucas amostras — recomenda-se revisão manual"). Ver `docs/task006_proximos_passos.md` para o plano dessa próxima task.

---

## Distribuição de classes e camadas (dados reais)

Dataset: `resultado_laudo_afericao.xlsx` — 35.318 linhas, 50 classes originais em `CODRSTAFER`.

Expurgo a 0,1% (limiar padrão, camada C): **50 → 14 classes**, retendo **99,25%** das linhas (35.053 de 35.318). As 36 classes descartadas (camada D) somam apenas ~0,75% das linhas (265 registros).

### Classes retidas por camada

| Camada | Classe | % do total | Contagem |
|---|---|---|---|
| A | `10` | 46,10% | 16.280 |
| A | `1` | 33,84% | 11.950 |
| B | `165` | 5,92% | 2.090 |
| B | `106` | 4,50% | 1.591 |
| B | `111` | 2,98% | 1.053 |
| B | `121` | 1,91% | 673 |
| B | `104` | 1,53% | 540 |
| C | `102` | 0,86% | 302 |
| C | `109` | 0,51% | 181 |
| C | `115` | 0,39% | 137 |
| C | `146` | 0,27% | 94 |
| C | `145` | 0,16% | 57 |
| C | `101` | 0,16% | 55 |
| C | `107` | 0,14% | 50 |

Comparado à v2 (limiar 15%, só camada A → **2 classes**), a v3 cobre **7x mais classes** (14 vs. 2), trazendo as camadas B e C para dentro do escopo do modelo.

Artefatos gerados:
- `model/exploration/class_distribution_before_purge_codrstafer.json` / `.csv` / `.html` / `.png`
- `model/exploration/class_distribution_after_purge_codrstafer.json` / `.csv` / `.html` / `.png`
- `model/class_weight_registry.json`

Os gráficos (PNG via seaborn, HTML via plotly) agora colorem as barras por camada, facilitando a leitura visual de onde cada classe se encaixa.

---

## Métricas de classificação (pipeline completo, dados reais)

Fonte: `model/classification/classification_details_v3_real.json` (com `per_class` detalhado) e `model/classification/task05_v3_run_summary.json`.

Preparação: 35.053 linhas → pré-processamento completo (heurística de colunas + correlação + `TruncatedSVD`, 128 componentes, variância explicada acumulada 94,81%). Split treino/teste padrão do workflow. **Sem** busca de hiperparâmetros nem cross-validation nesta rodada (ver seção de CV abaixo para um dado complementar).

### 🟢 Comparativo macro — 6 combos (algoritmo × resampling)

| Algoritmo | Resampling | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|---|
| CatBoost | nenhum | 90,74% | 54,89% | 60,23% | **56,51%** | 98,48% | 60,13% |
| CatBoost | SMOTE | 91,04% | 55,26% | 60,40% | **56,81%** | 98,51% | 59,95% |
| CatBoost | ADASYN | 88,12% | 51,89% | 59,01% | **53,87%** | 98,59% | 56,04% |
| XGBoost | nenhum | 93,27% | 62,22% | 59,10% | **59,52%** | 98,65% | 60,95% |
| XGBoost | SMOTE | 93,44% | 62,23% | 61,17% | **61,14%** | 98,12% | 59,80% |
| XGBoost | ADASYN | 92,33% | 58,82% | 58,21% | **57,84%** | 97,93% | 58,42% |

XGBoost supera CatBoost em accuracy/precision/F1 macro nos três cenários de resampling; CatBoost tem ROC-AUC levemente superior. Em ambos os algoritmos, o padrão é o mesmo: **SMOTE ≈ nenhum** (ganho marginal), **ADASYN < nenhum** (piora).

### 🟢 StratifiedKFold (k=5) — dado complementar

Rodada anterior (interrompida por lentidão antes da correção de `n_jobs=1`, portanto **não repetida** com os cenários SMOTE/ADASYN nem com a correção de paralelismo — apenas CatBoost sem resampling):

| Algoritmo | Resampling | F1 macro | Recall macro | ROC-AUC macro |
|---|---|---|---|---|
| CatBoost | nenhum | 57,99% ± 1,75pp | 62,13% ± 1,82pp | 98,17% ± 0,14pp |

O desvio-padrão baixo entre folds (< 2pp) é um indício de estabilidade do modelo CatBoost sem resampling, consistente com o resultado de holdout (F1 56,51%) reportado na tabela principal. Recomenda-se refazer essa validação cruzada para os 6 combos em uma próxima rodada, agora que a suíte de testes caiu de ~106s para ~20s e o `n_jobs=1` elimina o gargalo de paralelismo aninhado.

### 🟡 Métricas por classe (camadas B/C) — CatBoost

| Classe | Camada | n (teste) | Recall (nenhum) | Recall (SMOTE) | Recall (ADASYN) | F1 (nenhum) | F1 (SMOTE) | F1 (ADASYN) |
|---|---|---|---|---|---|---|---|---|
| `104` | B | 108 | 0,787 | 0,731 | 0,759 | 0,783 | 0,767 | 0,749 |
| `106` | B | 318 | 0,774 | 0,786 | 0,494 | 0,799 | 0,808 | 0,543 |
| `111` | B | 211 | 0,678 | 0,664 | 0,682 | 0,745 | 0,743 | 0,699 |
| `121` | B | 135 | 0,719 | 0,770 | 0,681 | 0,638 | 0,673 | 0,667 |
| `165` | B | 418 | 0,957 | 0,935 | 0,969 | 0,789 | 0,796 | 0,782 |
| `101` | C | 11 | 0,364 | 0,364 | 0,364 | 0,286 | 0,276 | 0,333 |
| `102` | C | 61 | 0,607 | 0,590 | 0,590 | 0,372 | 0,367 | 0,396 |
| `107` | C | 10 | 0,200 | 0,200 | 0,200 | 0,182 | 0,182 | 0,103 |
| `109` | C | 36 | 0,222 | 0,278 | 0,389 | 0,216 | 0,260 | 0,162 |
| `115` | C | 27 | 0,481 | 0,481 | 0,519 | 0,520 | 0,500 | 0,528 |
| `145` | C | 11 | 0,091 | 0,091 | 0,091 | 0,100 | 0,095 | 0,105 |
| `146` | C | 19 | 0,684 | 0,684 | 0,684 | 0,565 | 0,565 | 0,578 |

### 🟡 Métricas por classe (camadas B/C) — XGBoost

| Classe | Camada | n (teste) | Recall (nenhum) | Recall (SMOTE) | Recall (ADASYN) | F1 (nenhum) | F1 (SMOTE) | F1 (ADASYN) |
|---|---|---|---|---|---|---|---|---|
| `104` | B | 108 | 0,787 | 0,787 | 0,769 | 0,773 | 0,769 | 0,748 |
| `106` | B | 318 | 0,874 | 0,836 | 0,777 | 0,859 | 0,838 | 0,783 |
| `111` | B | 211 | 0,791 | 0,758 | 0,730 | 0,786 | 0,792 | 0,746 |
| `121` | B | 135 | 0,793 | 0,778 | 0,726 | 0,723 | 0,727 | 0,690 |
| `165` | B | 418 | 0,895 | 0,892 | 0,892 | 0,864 | 0,876 | 0,865 |
| `101` | C | 11 | 0,182 | 0,273 | 0,273 | 0,250 | 0,333 | 0,353 |
| `102` | C | 61 | 0,426 | 0,475 | 0,459 | 0,377 | 0,403 | 0,403 |
| `107` | C | 10 | 0,200 | 0,200 | 0,100 | 0,250 | 0,211 | 0,095 |
| `109` | C | 36 | 0,250 | 0,278 | 0,278 | 0,340 | 0,303 | 0,222 |
| `115` | C | 27 | 0,370 | 0,519 | 0,444 | 0,455 | 0,609 | 0,533 |
| `145` | C | 11 | 0,091 | 0,091 | 0,091 | 0,111 | 0,118 | 0,118 |
| `146` | C | 19 | 0,684 | 0,737 | 0,684 | 0,605 | 0,636 | 0,605 |

XGBoost tem precision/F1 sistematicamente melhores que CatBoost na camada B (ex.: `106` F1 0,86 vs. 0,80; `165` F1 0,86-0,88 vs. 0,79-0,80). Na camada C, ambos os algoritmos oscilam bastante entre execuções — sinal de que o tamanho de amostra (10-61 no teste) é o fator limitante, não o algoritmo.

---

## Interpretação dos resultados

1. **A camada B é o principal "vencedor" do aumento do limiar.** As 5 classes da camada B (`104`, `106`, `111`, `121`, `165`) têm F1 consistentemente entre 0,64 e 0,88 nos dois algoritmos — sinal real e robusto, não ruído. Isso confirma a hipótese de que a v2 (limiar 15%, só camada A) estava descartando classes com sinal aproveitável. Subir a cobertura de 2 para 14 classes elevou o F1 macro de ~25% (baseline histórico de 41 classes, critério antigo) para **56%-58% no CatBoost e 58%-61% no XGBoost**.

2. **SMOTE é neutro/marginal.** No CatBoost, F1 macro sobe de 0,5651 (nenhum) para 0,5681 (SMOTE) — uma diferença de 0,3pp, dentro do que se poderia atribuir a ruído de amostragem. No XGBoost, o ganho é um pouco maior (0,5952 → 0,6114, ~1,6pp), mas o efeito por classe é misto: algumas classes da camada B melhoram (`102`, `115`), outras da camada B pioram levemente (`104`, `165` no CatBoost). Não há evidência forte de benefício sistemático — SMOTE pode ser mantido como opção, mas não deveria ser vendido como solução do desbalanceamento.

3. **ADASYN prejudica o desempenho geral neste dataset.** Em ambos os algoritmos, ADASYN é a pior opção: CatBoost cai de 0,5651 para 0,5387 de F1 macro (accuracy cai de 90,7% para 88,1%), XGBoost cai de 0,5952 para 0,5784. O efeito por classe também é inconsistente — em algumas classes da camada B há queda acentuada (ex.: `106` no CatBoost: recall 0,774 → 0,494). **Recomendação: não usar ADASYN neste cenário.**

4. **A camada C continua sendo uma zona cinzenta de baixa confiabilidade**, independentemente da técnica de resampling usada. Classes como `107` (n=10 no teste) e `145` (n=11) têm F1 na faixa de 0,10-0,25 em praticamente todas as combinações — não é uma limitação do algoritmo ou do resampling, é uma limitação de volume de dados: com 10-20 amostras de teste, qualquer métrica tem variância enorme e não é confiável para decisões de negócio automatizadas. `101`, `102`, `109` também oscilam bastante entre técnicas sem uma tendência clara de melhora.

5. **Este é um resultado honesto, não uma "vitória" como a v2.** A v2 tinha 100% em tudo, mas cobria só 2 classes fáceis (linearmente separáveis). A v3 cobre 14 classes com métricas realistas (56-61% de F1 macro) — é um trade-off proposital entre cobertura e "beleza" do número, e reflete melhor o problema real de negócio.

---

## Correções de performance e ambiente

Durante esta fase, três problemas de ambiente/performance bloqueavam a execução do pipeline `sklearn` completo com dados reais nesta máquina Windows:

1. **`n_jobs=-1` causava paralelismo aninhado.** `hyperparameter_search.py` (`RandomizedSearchCV`) e `cross_validation.py` (`cross_validate`) usavam `n_jobs=-1`, o que disparava ~16 processos `joblib`/`loky`. Nesta máquina, a política de Application Control/antivírus escaneia cada processo novo, tornando essa paralelização brutalmente lenta (uma busca de hiperparâmetros que deveria levar minutos ficou rodando por mais de 1h sem terminar). **Correção:** `n_jobs=1` em ambos os arquivos — CatBoost/XGBoost já paralelizam internamente via threads, então não há necessidade de paralelismo por processo do lado do `sklearn`. Após a correção, a suíte de testes caiu de ~106s para ~20s em `tests/test_classification_workflow.py`.

2. **CatBoost gravava arquivos de log em disco a cada treino.** Sem `allow_writing_files=False`, o `CatBoostClassifier` grava a pasta `catboost_info/` a cada execução, o que (a) causava `UnicodeDecodeError` intermitente no Windows quando várias instâncias de treino ocorriam em sequência, e (b) suja o repositório a cada execução. **Correção:** `allow_writing_files=False` em `classification/models.py`.

3. **Bloqueio de DLL do `scipy` (v1/v2) não se repetiu na v3.** Nas versões anteriores, a política de Controle de Aplicativo do Windows bloqueava DLLs do `scipy` (`ImportError: DLL load failed while importing _group_columns`), impedindo o pipeline `sklearn` completo (e obrigando a v2 a usar um "pipeline lite" sem pré-processamento completo nem SMOTE/ADASYN real). Esse bloqueio se resolveu no ambiente entre as sessões (não foi uma correção de código desta fase) — e é o que viabilizou rodar o pipeline completo, com pré-processamento de produção (`TruncatedSVD`, correlação, etc.) e SMOTE/ADASYN reais, nesta v3.

---

## Como reproduzir

```powershell
# Preparação real + classificação CatBoost/XGBoost × resampling none/smote/adasyn
python scripts/run_task05_v3.py
```

O script roda a preparação (leitura do Excel, limpeza, expurgo por camada, `TruncatedSVD`) e, em seguida, os 6 combos de classificação (sem busca de hiperparâmetros nem cross-validation, para ser mais rápido). Saídas:
- `model/prepared_training_dataset.csv`, `model/preprocessing_pipeline.pkl`, `model/preparation_metadata.json`
- `model/class_weight_registry.json`
- `model/classification/classification_details_v3_real.json` (detalhado, com `per_class`)
- `model/classification/task05_v3_run_summary.json` (resumo consolidado)

> Combos com SMOTE/ADASYN sobre o dataset real (~35k linhas, 14 classes) podem levar vários minutos cada, pois o oversampling expande a base de treino antes do ajuste do modelo — isso é esperado.

---

## Recomendações

1. **Manter `MIN_CLASS_PERCENTAGE_THRESHOLD` no valor padrão** (`CLASS_TIER_THRESHOLDS["C"]` = 0,1%), retendo as camadas A-C (14 classes, 99,25% das linhas). Não há motivo para voltar ao limiar de 15% da v2, que descartava 20% do volume de negócio.
2. **Não usar ADASYN** como estratégia de resampling padrão — piora o desempenho geral e é inconsistente por classe neste dataset.
3. **SMOTE é opcional/neutro** — pode ser mantido disponível como configuração, mas não deve ser apresentado como uma melhoria garantida; o ganho observado (0,3-1,6pp de F1 macro) está na margem do ruído.
4. **Tratar a camada D (36 classes, ~0,75% das linhas) como fora do escopo automatizado** — considerar uma categoria de "revisão manual" na resposta da API para esses casos, já que não há volume suficiente para nenhuma técnica de modelagem confiável.
5. **Considerar o mesmo tratamento de "revisão manual" para os casos mais fracos da camada C** (ex.: `101`, `107`, `109`, `145`, com F1 < 0,35 e n < 40 no teste) — usar o `tier` e o `weight` do `model/class_weight_registry.json` como critério objetivo para sinalizar baixa confiabilidade na predição, alimentando a futura narrativa via LLM descrita em `docs/task006_proximos_passos.md`.
6. **Refazer a validação cruzada (StratifiedKFold k=5) para os 6 combos** em uma próxima rodada, já que a correção de `n_jobs=1` tornou isso viável em tempo hábil (o dado atual de CV é só para CatBoost sem resampling, de uma rodada anterior à correção).

---

## Status final por item

| Item | Status v3 |
|---|---|
| Sistema de camadas A/B/C/D | 🟢 Implementado e testado (`CLASS_TIER_THRESHOLDS`) |
| `model/class_weight_registry.json` | 🟢 Persistido automaticamente na preparação |
| Métricas por classe (`per_class`) | 🟢 Implementado, presente no JSON e no console |
| Pipeline `sklearn` completo com dados reais | 🟢 Executado (bloqueio de `scipy` resolvido no ambiente) |
| Expurgo padrão (camada C, 0,1%) | 🟢 14 classes retidas, 99,25% das linhas |
| SMOTE (CatBoost + XGBoost) | 🟢 Executado — efeito neutro/marginal |
| ADASYN (CatBoost + XGBoost) | 🟢 Executado — efeito negativo, não recomendado |
| Correção `n_jobs=1` (sklearn) | 🟢 Aplicada em `hyperparameter_search.py` e `cross_validation.py` |
| Correção `allow_writing_files=False` (CatBoost) | 🟢 Aplicada em `classification/models.py` |
| StratifiedKFold k=5 (todos os 6 combos) | 🟡 Pendente — só há dado de CatBoost sem resampling, de rodada anterior à correção |
| Busca de hiperparâmetros (`RandomizedSearchCV`) | 🔴 Não executada nesta rodada (config `enable_hyperparameter_search=False`) |
| pytest (17 testes) | 🟢 Todos passando |
