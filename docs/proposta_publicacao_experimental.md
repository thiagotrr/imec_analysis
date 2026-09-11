# Proposta de publicação experimental — API + modelo compilado (PKL)

**Escopo:** expor a API FastAPI de inferência já existente, carregando os artefatos compilados (`champion.pkl` + `preprocessing_pipeline.pkl`).  
**Fora de escopo:** reexecutar pipelines de preparação, exploração, treino ou `scripts/compile_models.py`. O modelo já está compilado e versionado.  
**Objetivo:** avaliar o comportamento online com **consumo mínimo de requisições**, em ambiente gratuito, sem pretensão de produção.

> **Atualizado pela Task 010** (Firestore + autenticação JWT + histórico de inferências — ver `docs/task010_firestore_auth_historico.md` para as decisões arquiteturais completas). A partir desta task, `/inspecao/*` e `/historico/*` exigem login (`POST /auth/login`) — `--allow-unauthenticated` no Cloud Run continua controlando apenas quem pode **invocar** o serviço na rede (nível de infraestrutura); a autenticação por JWT é uma camada adicional, na aplicação. Os passos de infraestrutura do Firestore (§3.1b) e do segredo JWT (§3.4b) são novos nesta revisão.

---

## 1. Recomendação

**Plataforma primária: Google Cloud Run (serviço HTTP, billing por requisição, scale-to-zero).**

Justificativa para *este* projeto (FastAPI + XGBoost/sklearn + PKLs ~44 MB em disco):

| Critério | Cloud Run |
|---|---|
| Custo | **US$ 0** no free tier para o volume experimental (dezenas/centenas de chamadas) |
| Requisições | 2 milhões/mês gratuitas — a cota não é o gargalo |
| Idle | Scale-to-zero: **não cobra** CPU/RAM quando ninguém chama a API |
| RAM | Configurável (recomendado **2 GiB**) — necessário para carregar pandas/sklearn/xgboost e os PKLs |
| Stack | Container Docker com Uvicorn; HTTPS gerenciado (`*.run.app`) |
| Deploy | `gcloud run deploy` a partir do `Dockerfile` da raiz |

O sleep / cold start é aceitável: o experimento é pontual. **Não** configurar ping periódico em `/health` — isso anularia o scale-to-zero e aumentaria o consumo.

### Alternativas

1. **Render (Web Service Free)** — mais simples (GitHub, sem cartão). Sleep após 15 min de idle (desejável). Limitação crítica: **512 MB RAM / 0,1 CPU**. Os PKLs somam ~44 MB em disco, mas o processo Python (pandas + sklearn + XGBoost + cadeia de import que puxa matplotlib via `data_preparation`) tende a estourar 512 MB. Usar só se o Cloud Run não for viável e após validar memória localmente.
2. **Koyeb (instância Free)** — 512 MB, scale-to-zero após 1 h de idle, um serviço gratuito por organização. Mesmo risco de RAM que o Render.

Descartados para este recorte:

- GitHub Pages / Cloudflare Pages / Vercel (estático ou serverless curto; não servem FastAPI + pickle sklearn/XGBoost).
- Streamlit Community Cloud (stack errada: isto é API, não app Streamlit).
- Hugging Face Spaces Docker (em 2026 o SDK Docker/Gradio em CPU passou a exigir plano pago).
- Railway / Fly.io para contas novas (trial ou pay-as-you-go; não são free contínuo).

---

## 2. Arquitetura do deploy

```text
Cliente (Swagger / curl / SIMEC experimental)
        │  HTTPS
        ▼
Cloud Run  (min-instances=0, request-based billing, 2 GiB)
        │  uvicorn → api.main:app
        │  lifespan: load_model_runtime()  (uma vez por instância)
        ▼
Artefatos no container (já compilados, sem treino):
  model/compiled/champion.pkl + champion.json
  model/preprocessing_pipeline.pkl
  model/class_weight_registry.json
  model/codrstafer_glossary.json
```

Fluxo de uma requisição de análise:

1. **Login** (`POST /auth/login`, e-mail `@energisa.com.br` + senha) → token JWT (HS256, stateless, 8h por default).
2. Validação Pydantic (`LaudoCompletoRequest` / `LaudoSinteticoRequest` / CSV), com `Authorization: Bearer <token>` obrigatório.
3. `preprocessing_pipeline.transform` → `champion.predict` / `predict_proba`.
4. Decodificação da classe + camada A–D + texto de resultado.
5. Revisão LLM **desligada por default** no experimento (`LLM_ENABLED=false`) para não gastar cota de OpenAI/Gemini nem gerar tráfego de saída extra. Para ligar com segurança, use Secret Manager (seção 3.4) — nunca Git, Dockerfile ou `--set-env-vars`.
6. Inferência persistida no Firestore (`inferencias`, fail-soft — se o Firestore estiver indisponível, a resposta ao cliente não é afetada) para consulta posterior em `GET /historico/{numero_laudo}` (também autenticado).

O `Dockerfile` da raiz **não** copia dataset `.xlsx`, **não** roda `--ml` e **não** recompila modelo. O `Dockerfile.task05` permanece só para o pipeline histórico de treino.

### Endpoints a exercitar

| Método | Rota | Auth | Papel no experimento |
|---|---|---|---|
| GET | `/health` | Não | Health check / cold start / `runtime_loaded` |
| GET | `/docs` | Não | Swagger (avaliação manual) |
| POST | `/auth/login` | Não (é o próprio login) | Emite o token usado nas demais chamadas |
| GET | `/inspecao/modelos` | **Sim** | Metadados do champion (sem inferência) |
| POST | `/inspecao/laudo_sintetico` | **Sim** | Inferência magra (features retidas) |
| POST | `/inspecao/laudo_completo` | **Sim** | Inferência com layout completo do laudo |
| POST | `/inspecao/csv` | **Sim** | Lote (usar poucas linhas); não persiste histórico |
| GET | `/historico/{numero_laudo}` | **Sim** | Consulta inferências já persistidas para o laudo |

---

## 3. Plano de implementação (passo a passo)

Pré-requisitos no repositório (já feitos): `Dockerfile` da API, `.dockerignore`, `.gcloudignore`, `GET /health`, respeito à variável `PORT` das PaaS.

**Projeto GCP (já criado — não criar outro):** `imec-analysis`  
**Serviço Cloud Run:** `imec-analysis-api`  
**Região (free tier):** `us-central1`

O `gcloud` **não** deve ser autenticado com credenciais inventadas. O usuário faz login na **própria** conta Google (passo 3.1). Sem esse login, o deploy não avança.

### 3.1. Instalar o gcloud (Windows) e autenticar

Se `gcloud --version` falhar no PowerShell, instale o Google Cloud SDK. Duas opções:

**Opção A — winget (já disponível nesta máquina):**

```powershell
winget install -e --id Google.CloudSDK --accept-package-agreements --accept-source-agreements
```

Feche e reabra o PowerShell para o `PATH` passar a incluir `gcloud`.

**Opção B — instalador oficial:**

```powershell
(New-Object Net.WebClient).DownloadFile(
  "https://dl.google.com/dl/cloudsdk/channels/rapid/GoogleCloudSDKInstaller.exe",
  "$env:Temp\GoogleCloudSDKInstaller.exe"
)
& $env:Temp\GoogleCloudSDKInstaller.exe
```

Siga o assistente (Python bundled). Depois, **nova** janela do PowerShell.

Login (abre o browser — o usuário escolhe a conta Google dele; não pular, não colar senha no chat):

```powershell
gcloud auth login
gcloud config set project imec-analysis
gcloud auth list
gcloud config get-value project
gcloud projects describe imec-analysis
```

Esperado: a conta aparece como `ACTIVE` e `projects describe` retorna o projeto `imec-analysis`. Se o describe falhar com 403, a conta logada não tem acesso a esse projeto — pedir permissão IAM ou logar na conta dona do projeto.

**Faturamento:** o Cloud Run exige billing habilitado no projeto (cartão cadastrado). O free tier cobre o volume experimental; sem billing o deploy é recusado. Conferir em https://console.cloud.google.com/billing?project=imec-analysis

Habilitar APIs (depois do login):

```powershell
gcloud services enable `
  run.googleapis.com `
  artifactregistry.googleapis.com `
  cloudbuild.googleapis.com `
  secretmanager.googleapis.com `
  firestore.googleapis.com `
  --project=imec-analysis
```

### 3.1b. Firestore — infraestrutura (Task 010)

Além de habilitar a API acima, é preciso criar o banco Firestore (`imec-analysis`, modo Native — nome com hífen: o Firestore não aceita underscore em `database_id`), conceder `roles/datastore.user` à service account do Cloud Run e criar o índice composto usado pela consulta de histórico. Tudo isso é feito por um script idempotente (pode ser reexecutado sem erro):

```powershell
./scripts/db/firestore/setup_gcp.ps1
```

**Sem PowerShell/`gcloud` local?** Use o equivalente em bash direto no **Cloud Shell** (console.cloud.google.com, projeto `imec-analysis`, ícone de terminal no canto superior direito — já vem com `gcloud` autenticado, nada para instalar):

```bash
./scripts/db/firestore/setup_gcp.sh
```

Mesma lógica, mesma idempotência dos dois scripts — use o que for mais conveniente no ambiente em que você estiver (Windows local → `.ps1`; Cloud Shell/Linux/macOS → `.sh`).

Detalhes de cada passo e as decisões por trás (nome do banco, região, doc ID) em `docs/task010_firestore_auth_historico.md §2`. Depois de rodar o script, crie ao menos um usuário de teste (necessário para o smoke test em §3.5):

```powershell
python scripts/db/firestore/gerenciar_usuarios.py create-user
```

### 3.2. Build e deploy (LLM desligada)

Na raiz do repositório (`D:\Repos\Energisa\imec_analysis`). **Não** reexecutar treino nem `scripts/compile_models.py` — os PKLs já estão em `model/`.

Primeiro deploy: LLM **off**. Sem secret LLM, sem chave no ambiente. O `JWT_SECRET`, esse sim, é obrigatório desde o primeiro deploy — sem ele, `POST /auth/login` e todas as rotas protegidas respondem com erro (fail-soft: a API sobe, só essas rotas ficam indisponíveis). Configure-o primeiro (§3.4b) e só então faça o deploy:

```powershell
gcloud config set project imec-analysis

gcloud run deploy imec-analysis-api `
  --source . `
  --project imec-analysis `
  --region us-central1 `
  --allow-unauthenticated `
  --memory 2Gi `
  --cpu 1 `
  --min-instances 0 `
  --max-instances 1 `
  --timeout 300 `
  --set-env-vars "LLM_ENABLED=false,IMEC_API_RELOAD=false,LLM_PROVIDER=openai,FIRESTORE_PROJECT_ID=imec-analysis,FIRESTORE_DATABASE_ID=imec-analysis" `
  --set-secrets "JWT_SECRET=jwt-secret:latest"
```

`--source .` usa o `Dockerfile`. O `.gcloudignore` impede upload de `.env` e `certs/` ao Cloud Build; o `.dockerignore` impede que esses arquivos entrem na imagem. `--min-instances 0` garante scale-to-zero. `--max-instances 1` evita fan-out acidental. `--timeout 300` cobre o cold start (carga dos PKLs).

`--allow-unauthenticated` continua simplificando o teste, mas agora só no nível de rede: qualquer um alcança `/docs`, `/health` e consegue *tentar* chamar `/inspecao/*`/`/historico/*` — só não terá um `access_token` válido sem um usuário cadastrado (§3.1b) e sem passar por `/auth/login`. Não é controle de acesso robusto (sem rate limiting, sem MFA — ver §4), mas já reduz a exposição em relação à versão anterior desta proposta, onde qualquer chamada anônima executava inferência. Para um recorte mais fechado ainda, é possível omitir `--allow-unauthenticated` e somar identidade IAM por cima do JWT — fora de escopo aqui. **Nunca** use `--allow-unauthenticated` como desculpa para colocar API keys ou `JWT_SECRET` em `--set-env-vars`.

### 3.3. Variáveis de ambiente (não-segredos)

| Variável | Valor experimental | Onde |
|---|---|---|
| `PORT` | injetada pelo Cloud Run (em geral 8080) | Já honrada em `src/main.py` e no `CMD` do Docker (`${PORT:-8000}`) |
| `LLM_ENABLED` | `false` até o secret existir | `--set-env-vars` (e default no Dockerfile) |
| `LLM_PROVIDER` | `openai` (default do código) ou `gemini` | `--set-env-vars`; a app lê em `src/llm/config.py` |
| `IMEC_API_RELOAD` | `false` | `--set-env-vars` / Dockerfile |
| `FIRESTORE_ENABLED` | `true` (default do código) | `--set-env-vars`, só se precisar sobrescrever o default |
| `FIRESTORE_PROJECT_ID` | `imec-analysis` | `--set-env-vars`; a app lê em `src/db/firestore/settings.py` |
| `FIRESTORE_DATABASE_ID` | `imec-analysis` | `--set-env-vars` |
| `JWT_EXPIRE_MINUTES` | `480` (default do código) | `--set-env-vars`, só se precisar sobrescrever o default |
| HTTPS da API | gerenciado pelo Cloud Run | **Não** usar `certs/` (TLS local) |

A app lê as chaves assim (`src/llm/config.py` → `load_llm_settings()`):

- `OPENAI_API_KEY` se `LLM_PROVIDER=openai` (default)
- `GEMINI_API_KEY` se `LLM_PROVIDER=gemini`
- `is_configured()` só é verdadeiro com `LLM_ENABLED=true` **e** a chave do provedor ativo preenchida
- Sem isso, `build_default_reviewer` usa `NullLlmReviewer` (fail-soft; `revisao_llm` vem `null`)

**Local:** `.env` na raiz (gitignored). `load_dotenv()` em `src/api/main.py` e `src/llm/config.py` não sobrescreve variáveis já definidas no processo.

**Cloud Run:** não há `.env` no container. Só o que o Cloud Run injeta (`PORT`, `--set-env-vars`, `--set-secrets`).

### 3.4. Token LLM — Secret Manager (obrigatório se for ligar a revisão)

A chave **não** vai para: Git, `.env` commitado, `Dockerfile ENV`, `--set-env-vars` (visível no console / `gcloud run services describe` / logs de revisão), chat, prints de debug.

A chave **vai** para: Secret Manager, montada como a **mesma** variável de ambiente que o código já lê.

#### Nomes recomendados

| Secret no GCP | Env var no processo | Quando criar |
|---|---|---|
| `openai-api-key` | `OPENAI_API_KEY` | Se `LLM_PROVIDER=openai` (default deste repo) |
| `gemini-api-key` | `GEMINI_API_KEY` | Só se for usar Gemini |

Não precisa criar os dois. Crie só o do provedor ativo.

#### Criar o secret (PowerShell — a chave não ecoa)

Faça **depois** do `gcloud auth login` e com billing/APIs ok. Cole a chave no prompt; não cole no histórico do comando.

```powershell
gcloud config set project imec-analysis

$secure = Read-Host "Cole a OPENAI_API_KEY (não ecoa)" -AsSecureString
$bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
$plain = [Runtime.InteropServices.Marshal]::PtrToStringAuto($bstr)
[Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)

# Primeira vez:
$plain | gcloud secrets create openai-api-key --data-file=- --project=imec-analysis

# Se o secret já existir, adicione uma versão:
# $plain | gcloud secrets versions add openai-api-key --data-file=- --project=imec-analysis

Remove-Variable plain, secure, bstr -ErrorAction SilentlyContinue
```

Gemini (só se `LLM_PROVIDER=gemini`):

```powershell
$secure = Read-Host "Cole a GEMINI_API_KEY (não ecoa)" -AsSecureString
$bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
$plain = [Runtime.InteropServices.Marshal]::PtrToStringAuto($bstr)
[Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
$plain | gcloud secrets create gemini-api-key --data-file=- --project=imec-analysis
Remove-Variable plain, secure, bstr -ErrorAction SilentlyContinue
```

#### Conceder leitura à service account do Cloud Run

O serviço usa a Compute Engine default SA: `PROJECT_NUMBER-compute@developer.gserviceaccount.com`.

```powershell
$PROJECT_NUMBER = gcloud projects describe imec-analysis --format="value(projectNumber)"
$SA = "${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"

gcloud secrets add-iam-policy-binding openai-api-key `
  --member="serviceAccount:${SA}" `
  --role="roles/secretmanager.secretAccessor" `
  --project=imec-analysis
```

(Repetir o binding para `gemini-api-key` se esse secret existir.)

#### Ligar a LLM no serviço (só depois do secret + IAM)

O `--set-secrets` injeta o valor no processo como `OPENAI_API_KEY`. A app não muda: continua `os.getenv("OPENAI_API_KEY")`.

```powershell
gcloud run services update imec-analysis-api `
  --project imec-analysis `
  --region us-central1 `
  --update-env-vars "LLM_ENABLED=true,LLM_PROVIDER=openai,IMEC_API_RELOAD=false" `
  --update-secrets "OPENAI_API_KEY=openai-api-key:latest"
```

Gemini:

```powershell
gcloud run services update imec-analysis-api `
  --project imec-analysis `
  --region us-central1 `
  --update-env-vars "LLM_ENABLED=true,LLM_PROVIDER=gemini,IMEC_API_RELOAD=false" `
  --update-secrets "GEMINI_API_KEY=gemini-api-key:latest"
```

Deploy inicial já com LLM (alternativa a dois passos — só se o secret já existir):

```powershell
gcloud run deploy imec-analysis-api `
  --source . `
  --project imec-analysis `
  --region us-central1 `
  --allow-unauthenticated `
  --memory 2Gi `
  --cpu 1 `
  --min-instances 0 `
  --max-instances 1 `
  --timeout 300 `
  --set-env-vars "LLM_ENABLED=true,IMEC_API_RELOAD=false,LLM_PROVIDER=openai" `
  --set-secrets "OPENAI_API_KEY=openai-api-key:latest"
```

Manter a LLM **off** até o secret existir: deixe `LLM_ENABLED=false` e **não** passe `--set-secrets`. Ligar `LLM_ENABLED=true` sem chave é fail-soft (não quebra a API), mas não revisa nada.

Desligar de novo sem apagar o secret:

```powershell
gcloud run services update imec-analysis-api `
  --project imec-analysis `
  --region us-central1 `
  --update-env-vars "LLM_ENABLED=false" `
  --remove-secrets OPENAI_API_KEY
```

### 3.4b. Segredo JWT — Secret Manager (obrigatório desde o primeiro deploy)

Mesmo padrão do secret LLM acima: a chave **não** vai para Git, `.env` commitado, `Dockerfile ENV`, `--set-env-vars`, chat ou prints de debug — só para o Secret Manager, montada como a variável `JWT_SECRET` que o código já lê (`src/auth/settings.py`).

Gere um valor com alta entropia (não reaproveite senha nenhuma):

```powershell
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

Crie o secret (PowerShell — não ecoa; cole o valor gerado acima):

```powershell
gcloud config set project imec-analysis

$secure = Read-Host "Cole o JWT_SECRET (não ecoa)" -AsSecureString
$bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
$plain = [Runtime.InteropServices.Marshal]::PtrToStringAuto($bstr)
[Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)

$plain | gcloud secrets create jwt-secret --data-file=- --project=imec-analysis
# Se já existir e for trocar (ex.: rotação): $plain | gcloud secrets versions add jwt-secret --data-file=- --project=imec-analysis

Remove-Variable plain, secure, bstr -ErrorAction SilentlyContinue
```

Conceder leitura à service account do Cloud Run (mesma SA de §3.4):

```powershell
$PROJECT_NUMBER = gcloud projects describe imec-analysis --format="value(projectNumber)"
$SA = "${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"

gcloud secrets add-iam-policy-binding jwt-secret `
  --member="serviceAccount:${SA}" `
  --role="roles/secretmanager.secretAccessor" `
  --project=imec-analysis
```

O comando de deploy em §3.2 já inclui `--set-secrets "JWT_SECRET=jwt-secret:latest"`. Para atualizar um serviço já implantado sem refazer o deploy completo:

```powershell
gcloud run services update imec-analysis-api `
  --project imec-analysis `
  --region us-central1 `
  --update-secrets "JWT_SECRET=jwt-secret:latest"
```

Rotação: crie uma nova versão do secret (`gcloud secrets versions add`) e rode o `update` acima de novo — como o JWT é stateless (§B de `docs/task010_firestore_auth_historico.md`), trocar o segredo invalida instantaneamente todos os tokens já emitidos (usuários precisam logar de novo), sem exigir nenhuma limpeza adicional no Firestore.

### 3.5. Smoke test (poucas requisições)

Substituir `BASE` pela URL `https://….run.app`. A maioria das rotas agora exige login — faça o login primeiro e capture o token:

```powershell
curl "$BASE/health"

$login = curl -s -X POST "$BASE/auth/login" -H "Content-Type: application/json" `
  -d '{"email": "fulano@energisa.com.br", "senha": "..."}' | ConvertFrom-Json
$token = $login.access_token

curl "$BASE/inspecao/modelos" -H "Authorization: Bearer $token"
curl "$BASE/historico/2025006988" -H "Authorization: Bearer $token"
# Abrir $BASE/docs, usar "Authorize" (cadeado) com o token acima e disparar o example de /inspecao/laudo_sintetico
```

URL do serviço após o deploy:

```powershell
gcloud run services describe imec-analysis-api `
  --project imec-analysis `
  --region us-central1 `
  --format="value(status.url)"
```

Esperado: `/health` com `"status":"ok"` e `"runtime_loaded":true`; primeira chamada após idle pode levar dezenas de segundos (cold start).

---

### 3.6. CD via GitHub Actions (deploy manual, sem precisar de `gcloud` local)

Alternativa a rodar os comandos `gcloud` manualmente na sua máquina (útil se você não tem o Google Cloud SDK instalado localmente): `.github/workflows/deploy-cloud-run.yml` empacota exatamente o comando de deploy de §3.2 num workflow do GitHub Actions, disparado **só manualmente** (`workflow_dispatch` — nenhum deploy automático em push/merge, decisão deliberada para não publicar código não revisado no serviço público sem intervenção humana).

Para disparar, depois do setup único abaixo: aba **Actions** do repositório no GitHub → "Deploy Cloud Run (manual)" → **Run workflow** (escolher `llm_enabled`/`llm_provider` se quiser ligar a LLM) — ou via CLI do GitHub: `gh workflow run deploy-cloud-run.yml`.

#### Setup único (antes do primeiro uso)

O workflow autentica no GCP com uma chave de service account guardada como secret do repositório (`GCP_SA_KEY`) — **não** reutiliza a service account de runtime do Cloud Run (`PROJECT_NUMBER-compute@developer.gserviceaccount.com`, usada pela aplicação em produção); é uma identidade separada, só para "quem pode disparar deploy", com papéis diferentes de "o que o container em execução pode fazer".

Como isso precisa do `gcloud` (ou pode ser feito pelo Console do GCP), e nem sempre você terá o SDK instalado localmente, duas opções:

- **Cloud Shell** (recomendado se não tiver `gcloud` local): abra [console.cloud.google.com](https://console.cloud.google.com), projeto `imec-analysis`, clique no ícone de terminal (Cloud Shell) no canto superior direito — vem com `gcloud` pronto, sem instalar nada na sua máquina. Rode os comandos abaixo lá.
- **Console (sem terminal nenhum):** IAM e administrador → Contas de serviço → Criar conta de serviço → conceder os papéis da tabela abaixo → aba "Chaves" → Adicionar chave → JSON (baixa o arquivo automaticamente).

Comandos (Cloud Shell ou `gcloud` local):

```bash
gcloud iam service-accounts create github-deployer \
  --project=imec-analysis \
  --display-name="GitHub Actions - Deploy Cloud Run"

for ROLE in roles/run.admin roles/iam.serviceAccountUser roles/cloudbuild.builds.editor roles/artifactregistry.writer; do
  gcloud projects add-iam-policy-binding imec-analysis \
    --member="serviceAccount:github-deployer@imec-analysis.iam.gserviceaccount.com" \
    --role="$ROLE"
done

# Gera a chave JSON num arquivo temporário — não commitar, apagar depois de colar no GitHub.
gcloud iam service-accounts keys create github-deployer-key.json \
  --iam-account=github-deployer@imec-analysis.iam.gserviceaccount.com
```

| Papel concedido | Por quê |
|---|---|
| `roles/run.admin` | Criar/atualizar o serviço Cloud Run |
| `roles/iam.serviceAccountUser` | Permitir que o deploy "aja como" a service account de runtime do Cloud Run |
| `roles/cloudbuild.builds.editor` | `--source .` builda a imagem via Cloud Build |
| `roles/artifactregistry.writer` | Cloud Build publica a imagem gerada no Artifact Registry |

Se o deploy falhar por permissão no bucket de staging do Cloud Build, adicione também `roles/storage.admin` à mesma service account (variação observada entre projetos, conforme a configuração do bucket padrão do Cloud Build).

Depois de gerar `github-deployer-key.json`:

1. GitHub → repositório → **Settings → Secrets and variables → Actions → New repository secret**.
2. Nome: `GCP_SA_KEY`. Valor: conteúdo completo do arquivo JSON (abra e cole o texto).
3. **Apague o arquivo `github-deployer-key.json` local** assim que colar (`rm github-deployer-key.json` / `del github-deployer-key.json`) — não faz sentido deixá-lo no disco depois de estar no GitHub Secrets.

O secret `GCP_SA_KEY` nunca aparece em log do workflow (o GitHub mascara automaticamente qualquer secret registrado). Rotação: gere uma nova chave (`gcloud iam service-accounts keys create`), atualize o secret no GitHub, depois revogue a chave antiga (`gcloud iam service-accounts keys delete <KEY_ID> --iam-account=github-deployer@imec-analysis.iam.gserviceaccount.com`) para não acumular chaves órfãs.

Este workflow é **um caminho a mais**, não substitui o runbook manual de §3.1–§3.5 — ambos chegam ao mesmo `gcloud run deploy`; use o que for mais conveniente no momento.

---

## 4. Limitações esperadas

- **Cold start:** a primeira request após scale-to-zero carrega Python + sklearn/XGBoost + PKLs. Pode levar ~30–90 s. Timeout do cliente precisa ser alto.
- **Memória:** 2 GiB recomendado. 512 MB (Render/Koyeb free) é arriscado.
- **Free tier Cloud Run (billing por request, valores oficiais vigentes em 2026):** 180 000 vCPU-segundos/mês, 360 000 GiB-segundos/mês, 2 milhões de requests/mês. Uso experimental fica muito abaixo.
- **Cartão no GCP:** necessário para ativar o projeto; não implica cobrança se o uso permanecer no free tier. Acompanhar o billing dashboard.
- **Filesystem efêmero:** uploads CSV não persistem no disco da instância (o histórico de inferências unitárias, esse sim, persiste no Firestore — ver Task 010); cada instância é stateless (adequado — o modelo vai no image).
- **LLM desligada por default:** `revisao_llm` virá `null` enquanto `LLM_ENABLED=false` ou a chave do provedor não estiver no Secret Manager. Ligar a LLM gera custo/cota no provedor (OpenAI/Gemini), fora do free tier do Cloud Run.
- **Dados:** não publicar dataset `.xlsx` (já está no `.gitignore` / `.dockerignore` / `.gcloudignore`).
- **Firestore fail-soft:** se o banco/API estiver indisponível, a API sobe normalmente, mas `/auth/login` e `/historico/*` respondem 503 e a persistência de inferências é pulada silenciosamente (log de erro, resposta ao cliente não é afetada).
- **Auth simples, não é produção:** login por JWT (Task 010) reduz a exposição em relação à versão puramente pública anterior, mas continua sem rate limiting/lockout no login, sem MFA, sem revogação de token antes do `exp` (8h por default) e sem RBAC — ver `docs/task010_firestore_auth_historico.md §7` para a lista completa de riscos aceitos. Sem SLA, instância única, possível restart a qualquer momento. Segredos (LLM e JWT) continuam só no Secret Manager, nunca no HTTP.

---

## 5. Como medir o comportamento online

Registrar, para cada chamada (planilha simples ou anotações):

1. Tempo até `/health` após idle longo (cold start) vs. chamada quente em seguida.
2. Latência de `POST /inspecao/laudo_sintetico` e `POST /inspecao/laudo_completo` (exemplo do Swagger).
3. Status HTTP e se `runtime_loaded` permanece `true` após o cold start.
4. Uso no console Cloud Run: request count, latência p50/p95, memória, instâncias (deve voltar a 0 após idle).
5. Billing: confirmar US$ 0 no período do teste.
6. Opcional: um CSV de 2–3 linhas em `/inspecao/csv` — não usar lotes grandes.

Critério de sucesso do experimento: a API sobe, carrega o PKL, responde 200 nas rotas de inspeção, escala a zero quando ociosa e não gera custo.

---

## 6. Rollback / teardown

Para encerrar o experimento e zerar consumo:

```powershell
gcloud run services delete imec-analysis-api --region us-central1 --project imec-analysis
```

Opcional: apagar a imagem no Artifact Registry, os secrets (`gcloud secrets delete openai-api-key --project=imec-analysis`, `gcloud secrets delete jwt-secret --project=imec-analysis`), o banco Firestore (`gcloud firestore databases delete --database=imec-analysis --project=imec-analysis` — **irreversível**, apaga todo o histórico de inferências e os usuários cadastrados) e desabilitar `run.googleapis.com`/`firestore.googleapis.com` no projeto.

No Render/Koyeb (alternativa): suspender ou deletar o Web Service no dashboard.

Nada disto altera os PKLs locais nem exige re-treino.

---

## 7. Caminho Render (alternativa, se Cloud Run não for opção)

1. Conta Render → New Web Service → repositório GitHub `imec_analysis`.
2. Runtime: Docker (`Dockerfile` na raiz).
3. Instance type: **Free**.
4. Health check: `/health`.
5. Env: `LLM_ENABLED=false`, `IMEC_API_RELOAD=false`.
6. Se o deploy morrer por OOM, o experimento confirma a limitação de 512 MB — voltar ao Cloud Run com 2 GiB.

---

## Referências

- Cloud Run (visão geral e scale-to-zero): https://cloud.google.com/run/docs/overview/what-is-cloud-run
- Preços / free tier: https://cloud.google.com/run/pricing
- Secret Manager + Cloud Run (`--set-secrets`): https://cloud.google.com/run/docs/configuring/services/secrets
- Instalação do gcloud (Windows): https://cloud.google.com/sdk/docs/install-sdk
- Render Free (sleep 15 min, 512 MB): https://render.com/docs/free
- Carga do modelo na API: `src/api/model_runtime.py` (`load_model_runtime` no `lifespan` de `src/api/main.py`)
- Leitura das chaves LLM: `src/llm/config.py` (`OPENAI_API_KEY` / `GEMINI_API_KEY`, `LLM_ENABLED`, `LLM_PROVIDER`)
- Firestore + JWT + histórico (Task 010): `docs/task010_firestore_auth_historico.md` (decisões arquiteturais completas), `src/db/firestore/`, `src/auth/`, `scripts/db/firestore/`
- CD via GitHub Actions (manual, §3.6): `.github/workflows/deploy-cloud-run.yml`
- GitHub Actions: autenticação no GCP (`google-github-actions/auth`): https://github.com/google-github-actions/auth
