# Task 010 — Firestore + Autenticação JWT + Histórico de Inferências

**Depende de:** Task 007 (services de inspeção já implementados e em produção experimental no Cloud Run).
**Escopo:** (A) arcabouço reutilizável de acesso ao Google Firestore; (B) autenticação JWT simples; (C) persistência das inferências e novo grupo de endpoints `historico`.

> Numeração: "Task 008" já está em uso no código para a revisão LLM/LangChain (`src/llm/config.py`, `src/api/inference_pipeline.py`) e "Task 009" já foi usado duas vezes em commits anteriores (glossário CODRSTAFER, `dsc_classe_prevista`/top-3 `predict_proba`). `task010` é o próximo número livre.

---

## 1. Decisões confirmadas

- Criação de usuário: **só via scripts administrativos** (`scripts/db/firestore/gerenciar_usuarios.py`) — não há endpoint HTTP de cadastro. O único endpoint de auth é `POST /auth/login`.
- Autenticação **obrigatória em todas as rotas do recurso `/inspecao`** (`laudo_completo`, `laudo_sintetico`, `csv`, `modelos`) e em `GET /historico/{numero_laudo}`. Isso muda a postura do experimento público documentado em `docs/proposta_publicacao_experimental.md` (`--allow-unauthenticated` deixa de significar "sem login" nessas rotas).
- Persistência do CSV em lote **fora de escopo** (mesma decisão já aplicada à revisão LLM em lote): a rota `/inspecao/csv` passa a exigir login, mas não grava histórico.
- Login é obrigatoriamente um e-mail corporativo `@energisa.com.br`.
- Nome do banco Firestore: **`imec-analysis`** (não o banco `(default)` de um projeto GCP) — criado explicitamente com esse nome. (Nota: o Firestore não aceita underscore em `database_id` — só `[a-z][0-9]-`; por isso hífen, não `imec_analysis`.)

---

## 2. Firestore — arcabouço isolado e reutilizável

Pacote `src/db/firestore/` — namespace `src/db/` fica reservado para eventuais outros bancos no futuro; o submódulo `firestore` isola tudo o que é específico do Firestore e não conhece "inspeção" nem "auth":

- `src/db/firestore/settings.py` — `FirestoreSettings` (`enabled`, `project_id`, `database_id="imec-analysis"`) + `load_firestore_settings()`. Env vars: `FIRESTORE_ENABLED`, `FIRESTORE_PROJECT_ID` (opcional — ADC resolve o projeto), `FIRESTORE_DATABASE_ID`.
- `src/db/firestore/client.py` — `FirestoreRuntimeError`, `FirestoreRuntime` (client + settings), `load_firestore_runtime()` (carregado uma vez no `lifespan` de `src/api/main.py`, guardado em `app.state.firestore`), `get_collection()` (único helper genérico exposto).

**Coleções** (Firestore não tem DDL — uma coleção "existe" a partir do primeiro documento gravado nela, não há passo explícito de criação):
- `users` — usuários de autenticação.
- `inferencias` — histórico de inferências persistidas.

**Biblioteca:** `google-cloud-firestore` (client síncrono — os endpoints da API são `def` síncronos rodando em threadpool do FastAPI).

**Fail-soft**, por consistência com `src/api/model_runtime.py`: se `load_firestore_runtime()` falhar no startup, `app.state.firestore = None` e a API sobe; endpoints que dependem de Firestore devolvem 503 (auth/histórico) ou pulam a gravação com log (persistência de inferência, fail-soft duplo).

**Credenciais (ADC) — apenas Cloud Run:** a service account de runtime já em uso (`PROJECT_NUMBER-compute@developer.gserviceaccount.com`) fornece Application Default Credentials automaticamente via metadata server — nenhuma variável `GOOGLE_APPLICATION_CREDENTIALS`, chave JSON ou setup adicional. Testes automatizados nunca tocam Firestore real (`FIRESTORE_ENABLED=false` forçado em `tests/conftest.py`, mesmo padrão já usado para `LLM_ENABLED=false`).

### Infra GCP — script automatizado

`scripts/db/firestore/setup_gcp.ps1` (idempotente — pode ser reexecutado sem erro se API/banco/binding/índice já existirem). Resolve `PROJECT_NUMBER` dinamicamente via `gcloud projects describe` em vez de exigir que o operador copie o valor manualmente.

Pré-requisitos: `gcloud auth login` e billing habilitado no projeto (`docs/proposta_publicacao_experimental.md §3.1`).

```powershell
./scripts/db/firestore/setup_gcp.ps1
# ou, explicitando parâmetros:
./scripts/db/firestore/setup_gcp.ps1 -ProjectId imec-analysis -Region us-central1
```

Equivalente em bash (`scripts/db/firestore/setup_gcp.sh`) para rodar no Cloud Shell ou qualquer shell POSIX, sem PowerShell/instalação local:

```bash
./scripts/db/firestore/setup_gcp.sh
./scripts/db/firestore/setup_gcp.sh imec-analysis us-central1
```

O script executa:
1. `gcloud services enable firestore.googleapis.com`
2. `gcloud firestore databases create --database=imec-analysis --location=us-central1 --type=firestore-native`
3. `gcloud projects add-iam-policy-binding` concedendo `roles/datastore.user` à service account do Cloud Run
4. `gcloud firestore indexes composite create` — índice `numero_laudo ASC, criado_em DESC` na coleção `inferencias`, necessário para a query do histórico.

### Variáveis de ambiente

| Variável | Onde | Observação |
|---|---|---|
| `FIRESTORE_ENABLED` | `--set-env-vars` (Cloud Run) / `.env` (local) | `true` por padrão |
| `FIRESTORE_PROJECT_ID` | `--set-env-vars` | Opcional — ADC resolve o projeto no Cloud Run |
| `FIRESTORE_DATABASE_ID` | `--set-env-vars` | `imec-analysis` |

---

## 3. Autenticação JWT simples

- **Único endpoint:** `POST /auth/login` (`src/api/routers/auth.py`, tag "Autenticação"). Recebe e-mail (`@energisa.com.br`, validação de domínio via `field_validator`) + senha; retorna `{access_token, token_type="bearer", expires_in}`.
- **Criação de usuário:** só via `scripts/db/firestore/gerenciar_usuarios.py`:
  - `create-user` — cadastro individual interativo (prompt de senha sem eco).
  - `load-initial` — carga inicial em lote, lendo `scripts/db/firestore/usuarios.json` (mesmo diretório do script; nunca commitado — está no `.gitignore`, contém senhas em texto claro). O próprio script traz, em docstring, o formato esperado do arquivo.
- **Armazenamento:** coleção `users`, doc ID = e-mail normalizado (lowercase). `src/auth/users_repository.py`.
- **Hash de senha:** `bcrypt` — biblioteca ativa e leve (`hashpw`/`checkpw`); dispensa `passlib` (manutenção parada) e dispensa reimplementar KDF de senha em cima de `cryptography`.
- **JWT:** `PyJWT` + `HS256` — lib enxuta, único serviço emite e valida o token hoje (RS256 só se justificaria com múltiplos serviços consumidores). Algoritmo fixado em código, nunca lido do header do token (`algorithms=["HS256"]` sempre explícito no `jwt.decode` — evita algorithm confusion). Segredo via Secret Manager (`JWT_SECRET`), mesmo padrão de `OPENAI_API_KEY` (`docs/proposta_publicacao_experimental.md §3.4`). Claims: `sub` (e-mail), `iat`, `exp`, `iss="imec-analysis-api"`. Sem refresh token — expiração curta configurável (`JWT_EXPIRE_MINUTES`, default 480 = 8h); relogin cobre o caso de uso.

### Onde o JWT "mora" — modelo stateless

O token **não é armazenado em nenhum lugar do servidor** (nem Firestore, nem memória, nem cache) — é o próprio ponto forte do JWT, elimina a necessidade de um "session store" central:

1. No login, o servidor apenas **gera** o token e o devolve — depois disso, "esquece".
2. O **cliente** é responsável por guardar o token e enviá-lo em `Authorization: Bearer <token>`.
3. A cada requisição a uma rota protegida, `get_current_user` (`src/api/dependencies/auth.py`) **recalcula** a validade na hora — verifica assinatura HMAC + `exp` — sem consultar o Firestore. A identidade (`sub`) vem do próprio token, não de uma leitura em `users/{email}`.
4. Por isso, `ativo=false` só é checado no login, não a cada request: checar exigiria voltar a consultar o Firestore em toda chamada, contradizendo o modelo stateless.
5. Expiração é aritmética pura no `decode` (`exp = iat + JWT_EXPIRE_MINUTES`, embutido no payload assinado na emissão) — não existe lista de "tokens ativos" nem relógio server-side.

### Rotas protegidas e Swagger

`Depends(get_current_user)` (primeira `Depends()` do projeto) em: `POST /inspecao/laudo_completo`, `POST /inspecao/laudo_sintetico`, `POST /inspecao/csv`, `GET /inspecao/modelos`, `GET /historico/{numero_laudo}`.

Todas essas rotas documentam `401` no OpenAPI (`responses={401: RESPONSE_401_UNAUTHORIZED, ...}`, constante compartilhada em `src/api/routers/_common_responses.py`) — o Swagger UI exibe o cadeado e o código de erro em cada endpoint (o FastAPI preenche `securitySchemes` automaticamente a partir do uso de `HTTPBearer` como dependency).

### Decisões arquiteturais

- **HS256 vs RS256:** HS256 por simplicidade — único serviço emite e valida o token hoje; RS256 adicionaria gestão de par de chaves sem benefício real (YAGNI).
- **Sem refresh token:** expiração curta + relogin simples é suficiente para o volume de uso.
- **Firestore para usuários (não Secret Manager com lista fixa):** permite ativar/desativar usuário sem novo deploy, e reaproveita o mesmo arcabouço da seção 2.
- **Sem blacklist/revogação:** `ativo=false` só é checado no login (ver "modelo stateless" acima) — trade-off aceito: usuário desativado com token válido ainda acessa até o `exp`.
- **Sem cadastro público, sem rate limiting no login:** menor superfície de ataque; rate limiting fica como risco conhecido/fora de escopo.
- **Mudança de postura em `/inspecao/*`:** o Swagger deixa de ser utilizável anonimamente — o runbook de smoke test de `docs/proposta_publicacao_experimental.md` passa a precisar de um usuário de teste.

### Segredo JWT — Secret Manager

Mesmo padrão de `docs/proposta_publicacao_experimental.md §3.4`:

```powershell
$secure = Read-Host "Cole o JWT_SECRET (não ecoa)" -AsSecureString
$bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
$plain = [Runtime.InteropServices.Marshal]::PtrToStringAuto($bstr)
[Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
$plain | gcloud secrets create jwt-secret --data-file=- --project=imec-analysis
Remove-Variable plain, secure, bstr -ErrorAction SilentlyContinue

gcloud secrets add-iam-policy-binding jwt-secret `
  --member="serviceAccount:${SA}" --role="roles/secretmanager.secretAccessor" --project=imec-analysis

gcloud run services update imec-analysis-api `
  --project imec-analysis --region us-central1 `
  --update-secrets "JWT_SECRET=jwt-secret:latest"
```

Gere o valor de `JWT_SECRET` com alta entropia (ex.: `python -c "import secrets; print(secrets.token_urlsafe(48))"`).

---

## 4. Persistência de inferências + histórico

### Modelo de persistência — documento "achatado"

`InspecaoLaudoHistoricoEntry` (`src/api/models/inspecao_historico.py`) **herda** de `InspecaoLaudoResponse` (mesmo padrão de `InspecaoLaudoCsvItemResponse`) — todos os campos da resposta de análise viram campos de primeiro nível do documento Firestore, lado a lado com `usuario_id`/`criado_em`. Não há aninhamento: 1 inferência = 1 documento plano.

```python
class InspecaoLaudoHistoricoEntry(InspecaoLaudoResponse):
    id: str            # não gravado como campo — é o Document ID do Firestore
    usuario_id: str    # email de quem gerou a inferência
    criado_em: datetime
```

Exemplo do documento gravado em `inferencias/{auto_id}`:
```json
{
  "numero_laudo": "2025006988",
  "classe_prevista": "10",
  "dsc_classe_prevista": "O medidor está funcionando de acordo com o Regulamento Técnico Metrológico...",
  "camada": "A",
  "situacao_afericao": "Reprovado",
  "resultado": "Classe 10 (camada A)",
  "resultado_detalhado": "Classe prevista: 10 (camada A). ...",
  "predict_proba": {"10": 0.9123},
  "dsc_predict_proba": {"10": "O medidor está funcionando..."},
  "revisao_llm": null,
  "usuario_id": "fulano@energisa.com.br",
  "criado_em": "2026-09-10T14:32:00Z"
}
```

- **Coleção:** `inferencias`, doc ID auto-gerado (não `numero_laudo` — o mesmo laudo pode ser reanalisado várias vezes; o histórico não sobrescreve entradas anteriores).
- **Ponto de gravação:** no router (`src/api/routers/inspecao.py`), não no service — mantém `services/inspecao.py` puro/testável com fakes. Gravação **fail-soft**: falha ao persistir não derruba a resposta de análise já montada ao cliente (mesmo raciocínio de `revisao_llm` fail-soft).
- **Service** `src/api/services/historico.py`: `persistir_inferencia(client, response, *, usuario_id)`, `listar_historico(client, numero_laudo, *, data_inicio=None, data_fim=None, limit=50)`.

### `GET /historico/{numero_laudo}`

**Request:**
- `numero_laudo` (path, obrigatório).
- `data_inicio` / `data_fim` (query, opcionais, ISO-8601) — filtram por `criado_em`.

**Response — 200, array de `InspecaoLaudoHistoricoEntry`**, mais recente primeiro; lista vazia (não é erro) quando o laudo não tem inferências persistidas.

Exemplo: `GET /historico/2025006988?data_inicio=2026-08-01T00:00:00Z&data_fim=2026-09-01T00:00:00Z`

---

## 5. Arquivos novos/alterados

| Arquivo | Responsabilidade |
|---|---|
| `src/db/firestore/settings.py`, `client.py` | Arcabouço Firestore |
| `src/auth/settings.py`, `security.py`, `users_repository.py` | JWT + hash + repositório de usuários |
| `src/api/dependencies/auth.py` | `get_current_user` |
| `src/api/models/auth.py` | `LoginRequest`, `TokenResponse`, `AuthenticatedUser` |
| `src/api/routers/auth.py` | `POST /auth/login` |
| `src/api/models/inspecao_historico.py` | `InspecaoLaudoHistoricoEntry` |
| `src/api/services/historico.py` | `persistir_inferencia`, `listar_historico` |
| `src/api/routers/historico.py` | `GET /historico/{numero_laudo}` |
| `src/api/routers/_common_responses.py` | `RESPONSE_401_UNAUTHORIZED` compartilhado |
| `scripts/db/firestore/setup_gcp.ps1` / `.sh` | Automação de infra GCP (PowerShell e bash/Cloud Shell) |
| `scripts/db/firestore/gerenciar_usuarios.py` | CLI de usuários (`create-user`, `load-initial`) |
| `src/api/main.py` | lifespan carrega `app.state.firestore`/`jwt_settings`; registra `auth_router`/`historico_router` |
| `src/api/routers/inspecao.py` | `Depends(get_current_user)` em todas as rotas + persistência fail-soft nas 2 unitárias |
| `requirements.txt` | `google-cloud-firestore`, `PyJWT`, `bcrypt` |
| `tests/conftest.py` | `FIRESTORE_ENABLED=false`/`JWT_SECRET` de teste + `FakeFirestoreClient` |

## 6. Testes

`tests/test_firestore_client.py`, `tests/test_auth_security.py`, `tests/test_auth_endpoints.py`, `tests/test_historico_services.py`, `tests/test_historico_endpoints.py` — todos usam `FakeFirestoreClient` (stub in-memory em `tests/conftest.py`), nunca tocam um Firestore real. `tests/test_inspecao_endpoints.py` foi ajustado para enviar token JWT nas rotas agora protegidas e cobrir 401 sem token/com token inválido.

## 7. Riscos / fora de escopo

- Revogação de token / blacklist — não implementado (expiração curta mitiga).
- Rate limiting / lockout no login — não implementado; risco de força bruta aceito dado volume interno esperado baixo.
- RBAC/papéis — não implementado (usuário único "tipo").
- Persistência/auth granular do CSV em lote — fora de escopo (login exigido, mas sem gravação de histórico).
- CORS — não configurado.
- Auditoria de login (sucesso/falha) — só no log de aplicação, não no Firestore.
