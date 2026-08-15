# Proposta de publicação experimental — API + modelo compilado (PKL)

**Escopo:** expor a API FastAPI de inferência já existente, carregando os artefatos compilados (`champion.pkl` + `preprocessing_pipeline.pkl`).  
**Fora de escopo:** reexecutar pipelines de preparação, exploração, treino ou `scripts/compile_models.py`. O modelo já está compilado e versionado.  
**Objetivo:** avaliar o comportamento online com **consumo mínimo de requisições**, em ambiente gratuito, sem pretensão de produção.

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

1. Validação Pydantic (`LaudoCompletoRequest` / `LaudoSinteticoRequest` / CSV).
2. `preprocessing_pipeline.transform` → `champion.predict` / `predict_proba`.
3. Decodificação da classe + camada A–D + texto de resultado.
4. Revisão LLM **desligada por default** no experimento (`LLM_ENABLED=false`) para não gastar cota de OpenAI/Gemini nem gerar tráfego de saída extra. Para ligar com segurança, use Secret Manager (seção 3.4) — nunca Git, Dockerfile ou `--set-env-vars`.

O `Dockerfile` da raiz **não** copia dataset `.xlsx`, **não** roda `--ml` e **não** recompila modelo. O `Dockerfile.task05` permanece só para o pipeline histórico de treino.

### Endpoints a exercitar

| Método | Rota | Papel no experimento |
|---|---|---|
| GET | `/health` | Health check / cold start / `runtime_loaded` |
| GET | `/docs` | Swagger (avaliação manual) |
| GET | `/inspecao/modelos` | Metadados do champion (sem inferência) |
| POST | `/inspecao/laudo_sintetico` | Inferência magra (features retidas) |
| POST | `/inspecao/laudo_completo` | Inferência com layout completo do laudo |
| POST | `/inspecao/csv` | Lote (usar poucas linhas) |

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
  --project=imec-analysis
```

### 3.2. Build e deploy (LLM desligada)

Na raiz do repositório (`D:\Repos\Energisa\imec_analysis`). **Não** reexecutar treino nem `scripts/compile_models.py` — os PKLs já estão em `model/`.

Primeiro deploy: LLM **off**. Sem secret, sem chave no ambiente. O Swagger fica público (`--allow-unauthenticated`) só para o experimento; a chave LLM, se existir depois, continua só no Secret Manager (não vai para o HTTP).

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
  --set-env-vars "LLM_ENABLED=false,IMEC_API_RELOAD=false,LLM_PROVIDER=openai"
```

`--source .` usa o `Dockerfile`. O `.gcloudignore` impede upload de `.env` e `certs/` ao Cloud Build; o `.dockerignore` impede que esses arquivos entrem na imagem. `--min-instances 0` garante scale-to-zero. `--max-instances 1` evita fan-out acidental. `--timeout 300` cobre o cold start (carga dos PKLs).

`--allow-unauthenticated` simplifica o teste (Swagger público). Para um recorte um pouco mais fechado, omitir essa flag e chamar com identidade IAM. **Nunca** use `--allow-unauthenticated` como desculpa para colocar a API key em `--set-env-vars`.

### 3.3. Variáveis de ambiente (não-segredos)

| Variável | Valor experimental | Onde |
|---|---|---|
| `PORT` | injetada pelo Cloud Run (em geral 8080) | Já honrada em `src/main.py` e no `CMD` do Docker (`${PORT:-8000}`) |
| `LLM_ENABLED` | `false` até o secret existir | `--set-env-vars` (e default no Dockerfile) |
| `LLM_PROVIDER` | `openai` (default do código) ou `gemini` | `--set-env-vars`; a app lê em `src/llm/config.py` |
| `IMEC_API_RELOAD` | `false` | `--set-env-vars` / Dockerfile |
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

### 3.5. Smoke test (poucas requisições)

Substituir `BASE` pela URL `https://….run.app`:

```powershell
curl "$BASE/health"
curl "$BASE/inspecao/modelos"
# Abrir $BASE/docs e disparar o example de /inspecao/laudo_sintetico
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

## 4. Limitações esperadas

- **Cold start:** a primeira request após scale-to-zero carrega Python + sklearn/XGBoost + PKLs. Pode levar ~30–90 s. Timeout do cliente precisa ser alto.
- **Memória:** 2 GiB recomendado. 512 MB (Render/Koyeb free) é arriscado.
- **Free tier Cloud Run (billing por request, valores oficiais vigentes em 2026):** 180 000 vCPU-segundos/mês, 360 000 GiB-segundos/mês, 2 milhões de requests/mês. Uso experimental fica muito abaixo.
- **Cartão no GCP:** necessário para ativar o projeto; não implica cobrança se o uso permanecer no free tier. Acompanhar o billing dashboard.
- **Filesystem efêmero:** uploads CSV não persistem; cada instância é stateless (adequado — o modelo vai no image).
- **LLM desligada por default:** `revisao_llm` virá `null` enquanto `LLM_ENABLED=false` ou a chave do provedor não estiver no Secret Manager. Ligar a LLM gera custo/cota no provedor (OpenAI/Gemini), fora do free tier do Cloud Run.
- **Dados:** não publicar dataset `.xlsx` (já está no `.gitignore` / `.dockerignore` / `.gcloudignore`).
- **Não é produção:** Swagger público se `--allow-unauthenticated`; sem autenticação forte, sem SLA, instância única, possível restart a qualquer momento. Segredos continuam no Secret Manager, não no HTTP.

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

Opcional: apagar a imagem no Artifact Registry, o secret (`gcloud secrets delete openai-api-key --project=imec-analysis`) e desabilitar `run.googleapis.com` no projeto.

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

## 8. Runbook operacional

Passo a passo do zero, deploys seguintes, build local vs Cloud Build e **deploy automático em todo pull request**: [deploy_cloud_run.md](deploy_cloud_run.md).

---

## Referências

- Cloud Run (visão geral e scale-to-zero): https://cloud.google.com/run/docs/overview/what-is-cloud-run
- Preços / free tier: https://cloud.google.com/run/pricing
- Secret Manager + Cloud Run (`--set-secrets`): https://cloud.google.com/run/docs/configuring/services/secrets
- Instalação do gcloud (Windows): https://cloud.google.com/sdk/docs/install-sdk
- Render Free (sleep 15 min, 512 MB): https://render.com/docs/free
- Carga do modelo na API: `src/api/model_runtime.py` (`load_model_runtime` no `lifespan` de `src/api/main.py`)
- Leitura das chaves LLM: `src/llm/config.py` (`OPENAI_API_KEY` / `GEMINI_API_KEY`, `LLM_ENABLED`, `LLM_PROVIDER`)
