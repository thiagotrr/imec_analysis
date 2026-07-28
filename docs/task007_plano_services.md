# Task 007 — Plano de implantação: Services de Inspeção

**Branch de origem sugerida:** `task006_correcoes` (após fechamento da Task 006)  
**Depende de:** Task 006 — contratos Pydantic, endpoints HTTP (stubs), compilação `model/compiled/*`  
**Escopo:** implementar a camada de *services* (carga de artefatos, inferência, composição da resposta). Contratos e rotas **não** são reabertos.

> Artefato visual principal: Canvas Cursor `task007-plano-services.canvas.tsx` (painel ao lado do chat). Este markdown é o espelho versionado no repositório.

---

## 1. Objetivo

Substituir os `NotImplementedError` em:

- `analisar_laudo_completo`
- `analisar_laudo_sintetico`
- `analisar_csv_upload`

por inferência real com `champion.pkl` + `preprocessing_pipeline.pkl` + `target_encoder.pkl`, populando `InspecaoLaudoResponse` (`classe_prevista`, `camada`, `resultado`, `resultado_detalhado`).

`GET /inspecao/modelos` (`obter_info_modelos`) já é funcional — manter sem regressão.

---

## 2. Endpoints (tag "Inspeção de Medidor de Consumo")

| Método | Rota | Service | Estado atual |
|---|---|---|---|
| POST | `/inspecao/laudo_completo` | `analisar_laudo_completo` | Stub 500 |
| POST | `/inspecao/laudo_sintetico` | `analisar_laudo_sintetico` | Stub 500 |
| POST | `/inspecao/csv` | `analisar_csv_upload` | Stub 500 (validação CSV já no router) |
| GET | `/inspecao/modelos` | `obter_info_modelos` | OK |

---

## 3. Runtime

1. No `lifespan` de `src/api/main.py`, carregar artefatos uma vez.
2. Guardar em `app.state.runtime` (ou singleton tipado `ModelRuntime`).
3. Services de análise consomem o runtime; não reler `.pkl` a cada request.

Artefatos obrigatórios:

- `model/compiled/champion.pkl` (+ `.json`)
- `model/preprocessing_pipeline.pkl`
- `model/target_encoder.pkl`
- `model/class_weight_registry.json` (narrativa A–C)

Pré-requisito: `python scripts/compile_models.py` (ou workflow que compile) se `champion.pkl` ainda não existir.

---

## 4. Pipeline de inferência (único)

1. Payload já validado (Pydantic / CSV no router).
2. `model_dump()` → DataFrame; laudo completo filtra `RETAINED_FEATURE_COLUMNS`.
3. `preprocessing_pipeline.transform`.
4. `champion.predict` (opcional `predict_proba` só no texto detalhado).
5. `target_encoder.inverse_transform` → rótulo `CODRSTAFER`.
6. Lookup no `class_weight_registry` → `camada` / peso → compor `resultado` + `resultado_detalhado` (template, sem LLM).

CSV: `infer_one` por linha + `numero_linha`.

---

## 5. Erros HTTP

| HTTP | Causa |
|---|---|
| 422 | Já coberto (Pydantic / linhas CSV inválidas) |
| 500 | Artefato ausente, falha de carga ou erro de inferência |
| 200 | Inferência OK; política sugerida para “fora de escopo” também em 200 com revisão manual |

---

## 6. Módulos sugeridos

| Arquivo | Papel |
|---|---|
| `src/api/model_runtime.py` (novo) | Load/cache |
| `src/api/inference_pipeline.py` (novo) | Pipeline compartilhado |
| `src/api/narrative.py` (novo, leve) | Template A–C |
| `src/api/inspecao_services.py` | Remover stubs |
| `src/api/main.py` | Lifespan |
| `src/api/inspecao_router.py` | Mapear exceções de domínio |
| `tests/test_inspecao_services.py` + ajuste em `test_inspecao_endpoints.py` | Aceite |

---

## 7. Critérios de aceite (resumo)

1. Três POSTs de análise retornam 200 com resposta completa (sem `NotImplementedError`).
2. Artefatos carregados no startup (não por request).
3. Ausência de artefato → 500 claro (ou fail-fast no boot — ver decisões).
4. `resultado_detalhado` cita camada/peso do registry (template).
5. `GET /inspecao/modelos` sem regressão; `pytest` verde.

---

## 8. Fora de escopo

- Narrativa via LLM
- Retreino / alteração de thresholds A–C
- Mudança dos contratos Pydantic fixos
- Paralelismo/vetorização avançada no CSV (opcional pós-v1)

---

## 9. Decisões abertas

1. **Classe D / fora do registry:** preferência — HTTP 200 + `resultado="Revisão manual"`.
2. **Fail-fast no lifespan** vs API sobe e devolve 500 só nas rotas de análise.
3. **Texto curto de `resultado`:** glossário negócio vs string genérica com código+camada.
4. **Score/`predict_proba`:** só em `resultado_detalhado` (sem mudar schema) vs estender response model.
