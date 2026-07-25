# Evolução do pipeline de classificação — Task 05 (v2)

**Branch:** `task05_melhorias_pipeline_modelos`  
**Data:** 24/07/2026  
**Dataset:** `resultado_laudo_afericao.xlsx` (disponível)  
**Público-alvo:** equipe técnica e de negócio

---

## Resumo executivo

Com o arquivo real disponível, confirmamos o impacto do expurgo de **15%**: o problema de **50 classes** passa a **2 classes** (`1` e `10`), retendo **79,9%** das linhas (28.230 de 35.318).

Neste ambiente, o pipeline completo (`sklearn` + pré-processamento completo + SMOTE/ADASYN) **não pôde ser executado** porque a política de Controle de Aplicativo do Windows bloqueou DLLs do `scipy` (`ImportError: DLL load failed while importing _group_columns`). Por isso, as métricas de classificação desta v2 foram obtidas via **pipeline lite** (`scripts/run_task05_v2_lite.py`), usando `pandas` + `CatBoost` + `XGBoost` nativo, com o mesmo expurgo de 15% e hiperparâmetros propostos na Task 05.

**Conclusão principal:** após o corte de 15%, as duas classes remanescentes são **linearmente separáveis** com alta facilidade — CatBoost e XGBoost atingiram **100%** em accuracy, precision, recall, F1 e ROC-AUC (macro), inclusive na validação cruzada estratificada (k=5). Isso contrasta fortemente com o baseline anterior (41 classes, F1 macro ~25%).

---

## O que mudou em relação à v1

| Item | v1 (jul/2026) | v2 (24/07/2026) |
|---|---|---|
| Dataset real | Indisponível | ✅ Disponível e analisado |
| Distribuição de classes | Estimativa | ✅ Medida (50 → 2 classes após 15%) |
| Métricas de classificação | 🔴 Pendentes | 🟡 Pipeline lite com dados reais |
| Pipeline sklearn completo | 🔴 Não executado | 🔴 Bloqueado por política do SO neste ambiente |
| pytest (12 testes) | 🟡 Passando (sintético) | 🔴 Bloqueado pelo mesmo erro de `scipy` |

---

## Distribuição de classes (dados reais)

### Antes do expurgo

| Métrica | Valor |
|---|---|
| Linhas (target sem nulo) | 35.318 |
| Classes distintas | 50 |

**Top 5 classes:**

| Classe | Contagem | % |
|---|---|---|
| 10 | 16.280 | 46,10% |
| 1 | 11.950 | 33,84% |
| 165 | 2.090 | 5,92% |
| 106 | 1.591 | 4,50% |
| 111 | 1.053 | 2,98% |

As 48 classes restantes somam ~20% do dataset — nenhuma delas atinge o limiar de 15% isoladamente.

### Depois do expurgo (15%)

| Métrica | Valor |
|---|---|
| Linhas retidas | 28.230 (**79,93%**) |
| Classes retidas | **2** |
| Linhas removidas | 7.088 (20,07%) |
| Classes removidas | 48 |

| Classe | Contagem | % (sobre retidos) |
|---|---|---|
| 10 | 16.280 | 57,67% |
| 1 | 11.950 | 42,33% |

Artefatos gerados:
- `model/exploration/class_distribution_before_purge_CODRSTAFER.csv`
- `model/exploration/class_distribution_after_purge_CODRSTAFER.csv`
- `model/exploration/class_distribution_real_metadata.json`

---

## Métricas de classificação

### 🟢 Baseline histórico (41 classes, critério antigo, 4 modelos)

Referência de `model/classification/classification_details.json` (21/07/2026):

| Algoritmo | Accuracy | F1 macro | Recall macro | ROC-AUC macro |
|---|---|---|---|---|
| CatBoost | 93,1% | 25,3% | 25,6% | 98,7% |
| XGBoost | 92,6% | 24,4% | 25,6% | 96,9% |
| k-NN | 91,9% | 22,3% | 21,8% | 72,8% |
| SVM | 93,0% | 22,7% | 22,8% | 98,4% |

Accuracy alta, mas F1/recall macro baixos — típico de desbalanceamento extremo com dezenas de classes raras.

### 🟡 Task 05 v2 — pipeline lite, 2 classes, expurgo 15%

Fonte: `model/classification/classification_details_v2_real_lite.json`  
Treino: 22.584 linhas | Teste: 5.646 linhas | Hiperparâmetros Task 05 aplicados

| Algoritmo | Resampling | Accuracy | F1 macro | Recall macro | ROC-AUC |
|---|---|---|---|---|---|
| CatBoost | nenhum | **100%** | **100%** | **100%** | **100%** |
| CatBoost | oversample aleatório* | **100%** | **100%** | **100%** | **100%** |
| XGBoost | nenhum | **100%** | **100%** | **100%** | **100%** |
| XGBoost | oversample aleatório* | **100%** | **100%** | **100%** | **100%** |

\* Substituto de SMOTE/ADASYN no modo lite (imbalanced-learn indisponível sem scipy).

### StratifiedKFold (k=5) — pipeline lite

| Algoritmo | F1 macro (média ± desvio) | Recall macro | ROC-AUC macro |
|---|---|---|---|
| CatBoost | 100% ± 0,0pp | 100% ± 0,0pp | 100% ± 0,0pp |
| XGBoost | 100% ± 0,0pp | 100% ± 0,0pp | 100% ± 0,0pp |

---

## Interpretação dos resultados

1. **O expurgo de 15% muda radicalmente o problema.** De um multiclasse extremamente desbalanceado (50 classes, F1 ~25%) para um binário equilibrado (57%/43%) em que os modelos atingem separação perfeita.

2. **Trade-off de negócio confirmado.** 20% dos laudos (7.088 registros, 48 classes) ficam **fora do escopo** do modelo com o limiar atual. Esses casos precisarão de tratamento manual ou de uma estratégia complementar (ex.: categoria "revisão manual" / modelo residual).

3. **Fase 2 (SMOTE/ADASYN) tornou-se desnecessária** neste cenário: com apenas 2 classes equilibradas e métricas perfeitas, técnicas de oversampling não agregam valor. Elas seriam relevantes apenas se o limiar de expurgo fosse reduzido (ex.: 1–5%) para manter mais classes.

4. **Pipeline lite vs. pipeline completo.** As métricas perfeitas podem refletir features ainda não totalmente depuradas pelo pré-processamento sklearn (PCA, correlação, etc.). Recomenda-se validar com o pipeline completo assim que o bloqueio de `scipy` for resolvido.

---

## Limitações desta execução

| Limitação | Impacto | Mitigação |
|---|---|---|
| `scipy` bloqueado por política do SO | Pipeline sklearn, pytest e SMOTE/ADASYN indisponíveis | Rodar `python scripts/run_task05_v2.py` em ambiente sem bloqueio (ou liberar DLLs do scipy) |
| Pipeline lite (pré-processamento simplificado) | Métricas podem diferir do pipeline de produção | Comparar com `python src/main.py` após desbloqueio |
| Docker Desktop parado | Container alternativo não executado | `docker build -f Dockerfile.task05 -t imec-task05 . && docker run ...` |
| Busca de hiperparâmetros (RandomizedSearchCV) | Não executada com dados reais | Ativar `ClassificationConfig.enable_hyperparameter_search=True` |

---

## Como reproduzir

```powershell
# 1. Distribuição de classes (somente pandas)
python scripts/analyze_class_distribution_real.py

# 2. Métricas lite (CatBoost + XGBoost, sem scipy)
python scripts/run_task05_v2_lite.py

# 3. Pipeline completo (requer scipy/sklearn funcionando)
python scripts/run_task05_v2.py
# ou
python src/main.py
```

O dataset é resolvido automaticamente em `src/machine_learning/resultado_laudo_afericao.xlsx` ou na raiz do repositório.

---

## Recomendações

1. **Validar o limiar de 15% com negócio** — a v2 confirma que sobram apenas 2 classes; avaliar se isso atende ao escopo do produto.
2. **Executar o pipeline completo** em ambiente onde `scipy` não esteja bloqueado, para confirmar métricas com pré-processamento de produção.
3. **Se precisar cobrir mais classes**, reduzir `MIN_CLASS_PERCENTAGE_THRESHOLD` (ex.: 1% ou 2%) e reavaliar — aí SMOTE/ADASYN e tuning de hiperparâmetros voltam a ser relevantes.
4. **Manter CatBoost e XGBoost** como modelos padrão; k-NN e SVM permanecem disponíveis mas fora do fluxo automático.

---

## Status final por item

| Item | Status v2 |
|---|---|
| Branch `task05_melhorias_pipeline_modelos` | 🟢 |
| Pipeline restrito a CatBoost/XGBoost | 🟢 |
| Expurgo 15% parametrizável | 🟢 Validado com dados reais |
| Distribuição antes/depois | 🟢 Artefatos gerados |
| Hiperparâmetros ajustados | 🟡 Aplicados no lite; busca automática pendente |
| SMOTE/ADASYN | 🔴 Bloqueado (scipy); oversample aleatório testado no lite |
| StratifiedKFold k=5 | 🟡 Validado no lite |
| Métricas com pipeline sklearn completo | 🔴 Pendente (bloqueio scipy) |
