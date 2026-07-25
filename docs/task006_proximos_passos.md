# Próximos passos — Task 006: Compilação do modelo e contratos de API

**Branch de origem:** `task05_melhorias_pipeline_modelos`
**Depende de:** Task 05 (v1–v3) — pipeline de preparação + classificação (CatBoost/XGBoost), sistema de camadas A–D e `class_weight_registry`
**Escopo desta task:** compilar os modelos treinados em artefatos `*.pkl` consumíveis, definir os contratos de entrada (Pydantic) da API e implementar **apenas os endpoints** (documentados via OpenAPI/Swagger). Os *services* efetivamente disparados pelos endpoints (carregar `.pkl`, pré-processar, inferir, compor a resposta) ficam para uma task futura.

> Este documento é um **plano**, não uma implementação. Nenhum código de produção foi alterado a partir daqui — é o roteiro para a próxima iteração.

---

## Contexto (o que já existe e serve de base)

| Artefato / módulo | Onde | Papel na Task 006 |
|---|---|---|
| `preprocessing_pipeline.pkl` | `model/preprocessing_pipeline.pkl` | Pipeline `sklearn` (imputação, encoding, scaling, PCA) já treinado — será reaproveitado na inferência |
| `target_encoder.pkl` | `model/target_encoder.pkl` | Decodifica a predição numérica de volta para o rótulo original de `CODRSTAFER` |
| `class_weight_registry.json` | `model/class_weight_registry.json` | Camada (A/B/C) e peso de cada classe retida — insumo para o texto de retorno da API (§5) |
| `preparation_metadata.json` | `model/preparation_metadata.json` | Colunas de entrada esperadas, colunas descartadas, tipos, etc. |
| `MANUALLY_REMOVED_FEATURES` | `src/machine_learning/feature_engineering.py` | Lista de colunas descartadas manualmente (usada no contrato §2.2) |
| Modelos treinados (CatBoost/XGBoost) | resultado de `run_classification_workflow` | Hoje não são persistidos em `.pkl` — é o item 1 desta task |
| `src/api/main.py`, `request_model.py`, `response_model.py` | `src/api/` | Esqueleto FastAPI existente com 1 endpoint de exemplo (`/analise_inspecao`) — será estendido |

---

## 1. Compilação dos modelos em `*.pkl`

**Objetivo:** persistir os modelos treinados (CatBoost e XGBoost) como artefatos binários reutilizáveis pela API, sem precisar re-treinar a cada chamada.

### 1.1. Onde plugar no pipeline

Adicionar uma etapa de "compilação" ao final de `run_classification_workflow` (`src/machine_learning/classification/workflow.py`), análoga a `save_preprocessing_artifacts`/`save_class_weight_registry`:

```python
def compile_model_artifact(
    trained_model: object,
    algorithm: str,
    resampling: str | None,
    output_dir: str | Path | None = None,
) -> Path:
    """Persiste um modelo treinado em model/compiled/{algorithm}_{resampling}.pkl."""
```

- Formato sugerido: `pickle` (já usado em `save_preprocessing_artifacts`) — avaliar `joblib` como alternativa (mais eficiente para objetos com arrays grandes; CatBoost/XGBoost também têm `save_model` nativo, a considerar como opção mais portátil/estável entre versões de lib).
- Convenção de nome: `model/compiled/{algorithm}_{resampling_or_none}.pkl` (ex.: `catboost_none.pkl`, `catboost_smote.pkl`), permitindo múltiplas variantes lado a lado.
- Critério de "modelo campeão": persistir automaticamente a combinação algoritmo+resampling com melhor `f1_macro` (ou métrica a definir com negócio) como `model/compiled/champion.pkl`, além das variantes individuais — a API de inferência (task futura) consome o `champion.pkl` por padrão.

### 1.2. Metadados obrigatórios junto ao `.pkl`

Cada modelo compilado deve vir acompanhado de um `.json` de mesmo nome com, no mínimo:
- `algorithm`, `resampling`, `trained_at`, `feature_columns` (ordem exata esperada pelo `preprocessing_pipeline.pkl`), `target_classes` (rótulos originais via `target_encoder.pkl`), `metrics` (accuracy/precision/recall/f1/roc_auc macro), `tier_thresholds` e referência ao `class_weight_registry.json` vigente no momento do treino.
- Isso evita o erro clássico de servir um `.pkl` sem saber qual pipeline/feature-set ele espera.

### 1.3. Entregável

- Novo módulo `src/machine_learning/classification/model_compilation.py` (ou função em `workflow.py`) + script `scripts/compile_models.py` para rodar isoladamente (sem re-treinar tudo via CLI).
- Testes: `tests/test_model_compilation.py` garantindo que o `.pkl` gerado é carregável e reproduz a mesma predição do modelo em memória.

---

## 2. Contratos de entrada (Pydantic)

Três contratos, cada um mapeado a um endpoint próprio (ver §4). Todos devem herdar de uma base comum para reuso de validação:

```python
class InspecaoMedidorBase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # validações comuns (tipos, ranges) aqui
```

### 2.1. Contrato "todas as features" (`LaudoCompletoRequest`)

- Espelha **todas** as colunas de `resultado_laudo_afericao.xlsx` (exceto o target `CODRSTAFER`, que é o que se quer prever).
- **Geração do contrato não deve ser manual**: criar um utilitário (`scripts/generate_pydantic_schema.py`) que leia o dataset (ou `preparation_metadata.json`) via `feature_engineering.build_dataset_profile` e gere os campos Pydantic automaticamente (nome, tipo inferido, nullable conforme `null_pct`), evitando divergência entre o schema da API e o dataset real ao longo do tempo.
- Uso: cenário em que o consumidor da API já possui o laudo completo e não quer se preocupar em saber quais colunas o modelo de fato usa.

### 2.2. Contrato "features do modelo" (`LaudoSinteticoRequest`)

- Mesma base do item 2.1, mas **excluindo** as colunas listadas em `MANUALLY_REMOVED_FEATURES` (`src/machine_learning/feature_engineering.py`) — ou seja, exatamente o conjunto de colunas que o `preprocessing_pipeline.pkl` espera antes do drop heurístico/constante/correlação (esses três últimos são aplicados automaticamente pelo pipeline, não pelo contrato).
- Gerado pelo mesmo utilitário do item 2.1, com um parâmetro `excluded_columns=MANUALLY_REMOVED_FEATURES` — garante que os dois contratos nunca fiquem dessincronizados manualmente.
- Uso: cenário de integração mais "magra", em que o consumidor já sabe filtrar o que é irrelevante para o modelo.

### 2.3. Contrato de upload CSV (`InspecaoCsvUploadRequest`)

- Não é um `BaseModel` de campos individuais, e sim um endpoint que recebe `UploadFile` (multipart/form-data) com um CSV no mesmo layout de `resultado_laudo_afericao.xlsx` (mesmas colunas de 2.1, uma ou mais linhas).
- Contrato de validação pós-upload: reaproveitar o mesmo schema Pydantic de 2.1 para validar **cada linha** do CSV (via `pd.read_csv` + `FeaturesCompletasRequest.model_validate(row.to_dict())` por linha, ou validação vetorizada), retornando erros por linha/coluna em caso de schema inválido.
- Uso: cenário de análise em lote (vários medidores de uma vez).

### 2.4. Observações de design comuns aos 3 contratos

- Todos devem ser tolerantes a nulos onde o dataset real também é (`null_pct` do `DatasetProfile`), mas **obrigatórios** para as colunas usadas como feature relevante pelo modelo — a divisão exata deve ser revisada com negócio antes da implementação.
- Adicionar exemplos (`json_schema_extra`) reais (uma linha anonimizada do dataset), no mesmo estilo já usado em `request_model.py`.
- Reaproveitar `resultado_detalhado` do `response_model.py` atual como o campo onde a narrativa da camada/confiabilidade (baseada no `class_weight_registry.json`, camadas A/B/C) será futuramente injetada pelo LLM (ver §5) — o contrato de **resposta** pode já ser desenhado nesta task, mesmo que o *service* que o popula fique para depois.

---

## 3. Endpoints (apenas a camada HTTP, sem o service)

Todos em `src/api/`, seguindo o padrão já estabelecido em `main.py` (FastAPI + tags + `response_model`).

| Método | Rota | Request | Response | Descrição |
|---|---|---|---|---|
| `POST` | `/inspecao/laudo_completo` | `LaudoCompletoRequest` | `InspecaoMedidorResponse` (a definir) | Recebe todas as features do laudo (§2.1) |
| `POST` | `/inspecao/laudo_sintetico` | `LaudoSinteticoRequest` | `InspecaoMedidorResponse` (a definir) | Recebe apenas as features usadas pelo modelo (§2.2) |
| `POST` | `/inspecao/csv` | `UploadFile` (CSV) | `list[InspecaoMedidorResponse]` | Upload em lote (§2.3) |
| `GET` | `/inspecao/modelos` | — | `ModeloInfoResponse` (novo) | Metadados do(s) modelo(s) compilado(s) ativos: algoritmo, versão, métricas, camadas suportadas — útil para consumidores validarem compatibilidade antes de chamar os endpoints acima |

Para esta task, cada handler deve:
1. Validar o payload via o contrato Pydantic correspondente (a validação de schema é o único "trabalho" feito aqui).
2. Chamar um placeholder de service (`NotImplementedError` ou stub com `TODO` explícito referenciando a task de implementação do service), deixando claro no código e na documentação OpenAPI (`description=...`) que a lógica de inferência ainda não está implementada.
3. Manter o padrão de logging já usado em `main.py` (`log.info`/`log.exception`).

### Documentação obrigatória por endpoint (Swagger/OpenAPI)

- `summary` e `description` claros (como já feito no endpoint existente).
- `responses={422: {...}, 500: {...}}` documentando principais erros de validação e falha de inferência.
- Exemplos de request/response via `json_schema_extra` em cada modelo Pydantic (não apenas no endpoint).
- Agrupar as 4 rotas na mesma `tag` (`"Inspeção de Medidor de Consumo"`) para aparecerem juntas no Swagger UI.

---

## 4. Fora do escopo desta task (fica para a task de "services")

- Carregamento efetivo do `.pkl` compilado (§1) em runtime (cache/singleton do modelo carregado).
- Pré-processamento da requisição com `preprocessing_pipeline.pkl` antes de chamar `model.predict`.
- Lógica de decodificação do resultado (`target_encoder.pkl`) e composição do texto de `resultado_detalhado`.
- Integração com LLM para narrativa de confiabilidade/acurácia usando `class_weight_registry.json` (mencionada como visão futura na Task 05; aqui só o **campo** no contrato de resposta é reservado, não a lógica).
- Tratamento de classes da camada "D" (descartadas do treino) quando aparecerem em produção — precisa de uma decisão de negócio (ex.: retornar "fora do escopo do modelo — revisão manual").

---

## 5. Critérios de aceite sugeridos

1. `scripts/compile_models.py` gera `model/compiled/*.pkl` + `.json` de metadados, carregáveis e testados.
2. Os 3 contratos Pydantic (§2.1–2.3) existem, com testes de validação (payload válido/invalido) e exemplos no schema.
3. Os 4 endpoints (§3) respondem no Swagger UI (`/docs`) com documentação completa, mesmo que o corpo do handler seja um stub.
4. Nenhuma lógica de inferência real é implementada nesta task (apenas contratos + endpoints + compilação de modelo).
5. `pytest` cobrindo os novos módulos, mantendo a suíte atual 100% verde.
