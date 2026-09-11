<#
.SYNOPSIS
    Automação da infraestrutura GCP necessária ao Firestore (Task 010).

.DESCRIPTION
    Idempotente: pode ser reexecutado sem erro se a API, o banco, o binding
    IAM ou o índice composto já existirem. Resolve PROJECT_NUMBER
    dinamicamente via `gcloud projects describe` (não precisa ser colado
    manualmente). Pré-requisitos: `gcloud auth login` já feito e billing
    habilitado no projeto (ver docs/proposta_publicacao_experimental.md §3.1).

.PARAMETER ProjectId
    ID do projeto GCP (default: imec-analysis).

.PARAMETER Region
    Região do banco Firestore (default: us-central1, mesma do Cloud Run).

.EXAMPLE
    ./scripts/db/firestore/setup_gcp.ps1
    ./scripts/db/firestore/setup_gcp.ps1 -ProjectId imec-analysis -Region us-central1
#>
param(
    [string]$ProjectId = "imec-analysis",
    [string]$Region = "us-central1"
)

$ErrorActionPreference = "Stop"
$DatabaseId = "imec-analysis"

Write-Host "Habilitando firestore.googleapis.com em $ProjectId..."
gcloud services enable firestore.googleapis.com --project=$ProjectId

Write-Host "Criando banco Firestore '$DatabaseId' (Native, $Region) — idempotente, ignora erro se já existir..."
try {
    gcloud firestore databases create `
        --database=$DatabaseId --location=$Region --type=firestore-native `
        --project=$ProjectId
} catch {
    Write-Host "Banco '$DatabaseId' provavelmente já existe — prosseguindo."
}

$ProjectNumber = gcloud projects describe $ProjectId --format="value(projectNumber)"
$Sa = "${ProjectNumber}-compute@developer.gserviceaccount.com"
Write-Host "Concedendo roles/datastore.user à service account $Sa..."
gcloud projects add-iam-policy-binding $ProjectId `
    --member="serviceAccount:${Sa}" --role="roles/datastore.user" | Out-Null

Write-Host "Criando índice composto (numero_laudo ASC, criado_em DESC) na coleção 'inferencias'..."
try {
    gcloud firestore indexes composite create `
        --collection-group=inferencias `
        --database=$DatabaseId `
        --field-config field-path=numero_laudo,order=ascending `
        --field-config field-path=criado_em,order=descending `
        --project=$ProjectId
} catch {
    Write-Host "Índice provavelmente já existe — prosseguindo."
}

Write-Host "`nConcluído. Variáveis de ambiente da aplicação:"
Write-Host "  FIRESTORE_PROJECT_ID=$ProjectId"
Write-Host "  FIRESTORE_DATABASE_ID=$DatabaseId"
