# Runbook de deploy — Cloud Run (`imec-analysis`)

Documentação operacional para publicar a **API de inferência** (PKLs já compilados).  
Não reexecuta treino, preparação nem `scripts/compile_models.py`.

Runbook complementar à proposta experimental: [proposta_publicacao_experimental.md](proposta_publicacao_experimental.md).

| Item | Valor |
|---|---|
| Projeto GCP | `imec-analysis` |
| Número do projeto | `800201627986` |
| Serviço Cloud Run | `imec-analysis-api` |
| Região | `us-central1` |
| URL | https://imec-analysis-api-800201627986.us-central1.run.app |
| Swagger | https://imec-analysis-api-800201627986.us-central1.run.app/docs |
| SA de runtime | `800201627986-compute@developer.gserviceaccount.com` |
| SA de CI (GitHub) | `github-deploy@imec-analysis.iam.gserviceaccount.com` |
| Repositório GitHub | `thiagotrr/imec_analysis` |

Cada pull request para `main` **substitui** o que está no ar neste mesmo serviço. Não há URL de preview. PRs concorrentes brigam pelo mesmo endpoint; código ainda não mergeado fica público.

Não configurar ping periódico em `/health`: isso anula o scale-to-zero.

---

## 1. Build in cloud vs build locally

O Cloud Run **sempre executa** no GCP. A diferença é **onde a imagem Docker é gerada**.

| | **Padrão — build in cloud** | **Alternativa — build locally** |
|---|---|---|
| Comando | `gcloud run deploy --source .` | `docker build` + `docker push` + `gcloud run deploy --image` |
| Quem builda | Cloud Build (GCP) | Docker nesta máquina |
| Quando usar | Do zero, deploys seguintes, **GitHub Actions** | Iteração local com Docker já instalado |
| Pré-requisito extra | APIs Cloud Build + Artifact Registry | Docker Desktop; imagem `linux/amd64` |
| CI | **Sim** (único caminho do workflow) | **Não** |

Runtime (2 GiB, `min-instances=0`, secrets) é o mesmo nos dois casos.

---

## 2. Subir do zero (padrão: Cloud Build)

Projeto `imec-analysis` já existe — **não criar outro**.

### 2.1. SDK e login

```powershell
winget install -e --id Google.CloudSDK --accept-package-agreements --accept-source-agreements
```

Feche e reabra o PowerShell.

```powershell
gcloud --version
gcloud auth login
gcloud config set project imec-analysis
gcloud auth list
gcloud config get-value project
gcloud projects describe imec-analysis
```

Billing obrigatório: https://console.cloud.google.com/billing?project=imec-analysis

### 2.2. APIs

```powershell
gcloud services enable `
  run.googleapis.com `
  artifactregistry.googleapis.com `
  cloudbuild.googleapis.com `
  secretmanager.googleapis.com `
  iam.googleapis.com `
  iamcredentials.googleapis.com `
  sts.googleapis.com `
  --project=imec-analysis
```

(`iamcredentials` e `sts` entram por causa do Workload Identity Federation do GitHub.)

### 2.3. IAM da service account de runtime

O `--source .` faz upload para o GCS e o Cloud Build precisa ler o zip. Na SA padrão de Compute:

```powershell
$SA = "800201627986-compute@developer.gserviceaccount.com"

gcloud projects add-iam-policy-binding imec-analysis --member="serviceAccount:${SA}" --role="roles/cloudbuild.builds.builder"
gcloud projects add-iam-policy-binding imec-analysis --member="serviceAccount:${SA}" --role="roles/storage.objectAdmin"
gcloud projects add-iam-policy-binding imec-analysis --member="serviceAccount:${SA}" --role="roles/artifactregistry.writer"
gcloud projects add-iam-policy-binding imec-analysis --member="serviceAccount:${SA}" --role="roles/logging.logWriter"
gcloud iam service-accounts add-iam-policy-binding $SA `
  --member="serviceAccount:${SA}" `
  --role="roles/iam.serviceAccountUser" `
  --project=imec-analysis
```

### 2.4. Primeiro deploy (LLM off)

Na raiz do repositório. `--source .` usa o `Dockerfile`; `.gcloudignore` impede upload de `.env` e `certs/`; os PKLs **entram** na imagem.

```powershell
cd D:\Repos\Energisa\imec_analysis

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

`--set-env-vars` só neste primeiro deploy (substitui a lista inteira de env). Depois use a seção 3.

### 2.5. Smoke test

```powershell
$BASE = "https://imec-analysis-api-800201627986.us-central1.run.app"
curl.exe "$BASE/health"
curl.exe "$BASE/inspecao/modelos"
```

Esperado: `/health` com `"status":"ok"` e `"runtime_loaded":true`. A primeira chamada após idle pode levar dezenas de segundos (cold start).

### 2.6. LLM (opcional) — Secret Manager

A chave **não** vai para Git, Dockerfile, `--set-env-vars`, chat ou `.env` commitado.

| Secret Manager | Variável na API | Quando |
|---|---|---|
| `openai-api-key` | `OPENAI_API_KEY` | `LLM_PROVIDER=openai` |
| `gemini-api-key` | `GEMINI_API_KEY` | `LLM_PROVIDER=gemini` |

**Pitfall:** `gcloud secrets create` sem dados cria o secret **sem versões**. O Cloud Run falha com `.../versions/latest was not found`. Sempre confira `gcloud secrets versions list`. No PowerShell, não use pipe para `--data-file=-` (costuma chegar vazio).

Criar (primeira vez) **com arquivo temporário**:

```powershell
gcloud config set project imec-analysis
$keyFile = Join-Path $env:TEMP "openai-api-key.txt"
$secure = Read-Host "Cole a OPENAI_API_KEY (não ecoa)" -AsSecureString
$bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
$plain = [Runtime.InteropServices.Marshal]::PtrToStringAuto($bstr)
[Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
[System.IO.File]::WriteAllText($keyFile, $plain.Trim())
Remove-Variable plain -ErrorAction SilentlyContinue

gcloud secrets create openai-api-key --data-file=$keyFile --project=imec-analysis
Remove-Item $keyFile -Force -ErrorAction SilentlyContinue
gcloud secrets versions list openai-api-key --project=imec-analysis
```

Se o secret **já existe** sem versão:

```powershell
# mesma preparação de $keyFile acima
gcloud secrets versions add openai-api-key --data-file=$keyFile --project=imec-analysis
```

IAM para a SA de runtime:

```powershell
$SA = "800201627986-compute@developer.gserviceaccount.com"
gcloud secrets add-iam-policy-binding openai-api-key `
  --member="serviceAccount:${SA}" `
  --role="roles/secretmanager.secretAccessor" `
  --project=imec-analysis
```

Ligar a LLM **sem** reenviar a chave em env visível:

```powershell
gcloud run services update imec-analysis-api `
  --project imec-analysis `
  --region us-central1 `
  --update-env-vars "LLM_ENABLED=true,LLM_PROVIDER=openai,IMEC_API_RELOAD=false" `
  --update-secrets "OPENAI_API_KEY=openai-api-key:latest"
```

Desligar:

```powershell
gcloud run services update imec-analysis-api `
  --project imec-analysis `
  --region us-central1 `
  --update-env-vars "LLM_ENABLED=false" `
  --remove-secrets OPENAI_API_KEY
```

---

## 3. Deploys seguintes (manual, padrão Cloud Build)

**Não** passe `--set-env-vars` nem `--set-secrets`: isso substitui a lista e pode apagar `OPENAI_API_KEY` / `LLM_ENABLED`.

```powershell
cd D:\Repos\Energisa\imec_analysis

gcloud run deploy imec-analysis-api `
  --source . `
  --project imec-analysis `
  --region us-central1 `
  --memory 2Gi `
  --cpu 1 `
  --min-instances 0 `
  --max-instances 1 `
  --timeout 300
```

O Cloud Run mantém env e secrets já configurados.

### Rollback

```powershell
gcloud run revisions list --service imec-analysis-api --region us-central1 --project imec-analysis

gcloud run services update-traffic imec-analysis-api `
  --region us-central1 `
  --project imec-analysis `
  --to-revisions REVISAO_ANTERIOR=100
```

### Teardown

```powershell
gcloud run services delete imec-analysis-api --region us-central1 --project imec-analysis
```

Opcional: apagar imagens no Artifact Registry e o secret `openai-api-key`.

---

## 4. Alternativa — build locally

Não usado no GitHub Actions. Cloud Run exige imagem **Linux amd64** (não Windows).

### 4.1. Docker + Artifact Registry

```powershell
gcloud auth configure-docker us-central1-docker.pkg.dev --quiet

gcloud artifacts repositories create imec-analysis-api `
  --repository-format=docker `
  --location=us-central1 `
  --project=imec-analysis
```

Se o repositório já existir, o `create` falha com “already exists” — siga em frente.

### 4.2. Build, push e deploy

```powershell
cd D:\Repos\Energisa\imec_analysis

$TAG = Get-Date -Format "yyyyMMdd-HHmmss"
$IMAGE = "us-central1-docker.pkg.dev/imec-analysis/imec-analysis-api/api:$TAG"

docker build --platform linux/amd64 -t $IMAGE .
docker push $IMAGE

gcloud run deploy imec-analysis-api `
  --image $IMAGE `
  --project imec-analysis `
  --region us-central1 `
  --memory 2Gi `
  --cpu 1 `
  --min-instances 0 `
  --max-instances 1 `
  --timeout 300
```

No **primeiro** deploy por imagem (serviço ainda inexistente), acrescente as mesmas flags da seção 2.4 (`--allow-unauthenticated`, `--set-env-vars` com LLM off). Nos seguintes, **não** use `--set-env-vars` / `--set-secrets`.

---

## 5. Deploy automático em todo pull request

Arquivo: [`.github/workflows/deploy-cloud-run.yml`](../.github/workflows/deploy-cloud-run.yml).

Fluxo: `pytest` → se passar, `gcloud run deploy --source .` (Cloud Build) no serviço `imec-analysis-api`.

Triggers:

- `pull_request` para `main` (`opened`, `synchronize`, `reopened`)
- `workflow_dispatch` (redeploy manual da branch)

O runner **não** executa `docker build`.

### 5.1. Workload Identity Federation (uma vez)

Sem JSON key no GitHub. Nomes fixos (não são segredos) — o workflow já os referencia.

```powershell
gcloud config set project imec-analysis

gcloud iam service-accounts create github-deploy `
  --display-name="GitHub Actions deploy Cloud Run" `
  --project=imec-analysis

$DEPLOY_SA = "github-deploy@imec-analysis.iam.gserviceaccount.com"
$RUNTIME_SA = "800201627986-compute@developer.gserviceaccount.com"

gcloud projects add-iam-policy-binding imec-analysis --member="serviceAccount:${DEPLOY_SA}" --role="roles/run.admin"
gcloud projects add-iam-policy-binding imec-analysis --member="serviceAccount:${DEPLOY_SA}" --role="roles/cloudbuild.builds.editor"
gcloud projects add-iam-policy-binding imec-analysis --member="serviceAccount:${DEPLOY_SA}" --role="roles/artifactregistry.writer"
gcloud projects add-iam-policy-binding imec-analysis --member="serviceAccount:${DEPLOY_SA}" --role="roles/storage.admin"
gcloud iam service-accounts add-iam-policy-binding $RUNTIME_SA `
  --member="serviceAccount:${DEPLOY_SA}" `
  --role="roles/iam.serviceAccountUser" `
  --project=imec-analysis

gcloud iam workload-identity-pools create github `
  --location=global `
  --display-name="GitHub Actions" `
  --project=imec-analysis

gcloud iam workload-identity-pools providers create-oidc github `
  --location=global `
  --workload-identity-pool=github `
  --display-name="GitHub" `
  --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository,attribute.repository_owner=assertion.repository_owner" `
  --attribute-condition="assertion.repository=='thiagotrr/imec_analysis'" `
  --issuer-uri="https://token.actions.githubusercontent.com" `
  --project=imec-analysis

$WIF_SA_MEMBER = "principalSet://iam.googleapis.com/projects/800201627986/locations/global/workloadIdentityPools/github/attribute.repository/thiagotrr/imec_analysis"

gcloud iam service-accounts add-iam-policy-binding $DEPLOY_SA `
  --project=imec-analysis `
  --role="roles/iam.workloadIdentityUser" `
  --member="$WIF_SA_MEMBER"
```

Provider usado pelo Actions:

`projects/800201627986/locations/global/workloadIdentityPools/github/providers/github`

Se `create` falhar porque pool/provider/SA já existem, use o recurso atual e só ajuste IAM.

### 5.2. O que o workflow faz

1. Checkout do código do PR.
2. `pytest` (LLM forçada off em `tests/conftest.py`; não usa chaves reais).
3. Autentica no GCP via WIF (`id-token: write`).
4. `gcloud run deploy imec-analysis-api --source .` **sem** `--set-env-vars` / `--set-secrets`.
5. Comenta no PR a URL do serviço experimental.

Logs: GitHub → Actions. Build da imagem: Cloud Build no console GCP.

Até o WIF da seção 5.1 estar criado, o job de deploy falha na autenticação; o job de teste ainda roda.

---

## 6. Checklist rápido

**Do zero:** login → APIs → IAM runtime → `deploy --source .` (LLM off) → smoke `/health`.

**Deploy seguinte:** `deploy --source .` sem mexer em env/secrets.

**Build local:** `--platform linux/amd64` → push Artifact Registry → `deploy --image`.

**PR:** WIF (uma vez) → abrir/atualizar PR para `main` → Actions publica no mesmo Cloud Run.

**LLM:** secret **com versão** + `--update-secrets`; nunca `--set-env-vars` com a chave.

---

## Referências

- Cloud Run: https://cloud.google.com/run/docs/overview/what-is-cloud-run
- Deploy from source: https://cloud.google.com/run/docs/deploying-source-code
- Secrets no Cloud Run: https://cloud.google.com/run/docs/configuring/services/secrets
- Workload Identity Federation + GitHub: https://github.com/google-github-actions/auth
- Carga do modelo: `src/api/model_runtime.py`
- Gate LLM: `src/llm/gate.py` / `src/llm/config.py`
