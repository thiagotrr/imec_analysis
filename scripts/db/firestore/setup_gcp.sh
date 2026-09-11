#!/usr/bin/env bash
# Automação da infraestrutura GCP necessária ao Firestore (Task 010).
#
# Equivalente em bash de setup_gcp.ps1 — para rodar direto no Cloud Shell
# (console.cloud.google.com, ícone de terminal) ou qualquer shell POSIX,
# sem precisar de PowerShell nem instalar nada localmente. Mesma lógica,
# mesma idempotência: pode ser reexecutado sem erro se a API, o banco, o
# binding IAM ou o índice composto já existirem.
#
# Uso:
#   ./scripts/db/firestore/setup_gcp.sh
#   ./scripts/db/firestore/setup_gcp.sh imec-analysis us-central1
#
# Pré-requisitos: gcloud autenticado (gcloud auth login) e billing
# habilitado no projeto (ver docs/proposta_publicacao_experimental.md §3.1).
# No Cloud Shell, o gcloud já vem pronto e autenticado com a conta logada
# no navegador — basta colar e rodar.

set -euo pipefail

PROJECT_ID="${1:-imec-analysis}"
REGION="${2:-us-central1}"
DATABASE_ID="imec-analysis"

echo "Habilitando firestore.googleapis.com em ${PROJECT_ID}..."
gcloud services enable firestore.googleapis.com --project="${PROJECT_ID}"

echo "Criando banco Firestore '${DATABASE_ID}' (Native, ${REGION}) — idempotente, ignora erro se já existir..."
if ! gcloud firestore databases create \
    --database="${DATABASE_ID}" --location="${REGION}" --type=firestore-native \
    --project="${PROJECT_ID}"; then
  echo "Banco '${DATABASE_ID}' provavelmente já existe — prosseguindo."
fi

PROJECT_NUMBER="$(gcloud projects describe "${PROJECT_ID}" --format='value(projectNumber)')"
SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"
echo "Concedendo roles/datastore.user à service account ${SA}..."
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:${SA}" --role="roles/datastore.user" >/dev/null

echo "Criando índice composto (numero_laudo ASC, criado_em DESC) na coleção 'inferencias'..."
if ! gcloud firestore indexes composite create \
    --collection-group=inferencias \
    --database="${DATABASE_ID}" \
    --field-config field-path=numero_laudo,order=ascending \
    --field-config field-path=criado_em,order=descending \
    --project="${PROJECT_ID}"; then
  echo "Índice provavelmente já existe — prosseguindo."
fi

echo ""
echo "Concluído. Variáveis de ambiente da aplicação:"
echo "  FIRESTORE_PROJECT_ID=${PROJECT_ID}"
echo "  FIRESTORE_DATABASE_ID=${DATABASE_ID}"
