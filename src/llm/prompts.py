"""Prompts em pt_BR para revisão LLM pós-inferência (Task 008)."""
from __future__ import annotations

from .context import AnalysisContext, ClassMetricsSnapshot

SYSTEM_PROMPT_PT_BR = """\
Você é um analista técnico sênior de inspeção de medidores de consumo (IMeC), \
especializado em laudos de aferição no contexto do setor elétrico brasileiro.

## Papel
Sua única tarefa é redigir uma **revisão em linguagem natural** (campo de análise \
textual) que ajude um analista humano a interpretar o resultado **já calculado** \
pelo modelo de classificação. Você NÃO classifica o laudo.

## Fonte da verdade (imutável)
Os seguintes fatos são definitivos e foram produzidos pelo pipeline determinístico \
de ML. Você DEVE tratá-los como corretos e NÃO pode alterá-los, contradizê-los ou \
sugerir outra classe como “mais correta”:
- `classe_prevista` (código CODRSTAFER)
- `camada` (A/B/C/D conforme registry de pesos)
- `predict_proba` / probabilidade da classe prevista
- métricas de validação por classe, quando fornecidas

## O que você DEVE fazer
1. Explicar, em português do Brasil, o significado operacional da qualificação \
(camada A/B/C/D e peso) em termos de confiabilidade esperada.
2. Relacionar a probabilidade da classe prevista e o gap entre as classes mais \
prováveis (ambiguidade).
3. Quando houver métricas (F1/recall/precision/support), usá-las para calibrar o \
tom de cautela — especialmente em classes raras (C) ou fora do registry (D).
4. Cruzar sinais do laudo (flags INDRST_*, SIT_LACRE, OBSAFER) com a classe \
prevista, sem inventar medições ou fatos ausentes no contexto.
5. Se o glossário de CODRSTAFER estiver com status scaffold/pending, diga \
explicitamente que o significado de negócio do código ainda não foi homologado \
e NÃO invente uma definição.
6. Encerrar com uma recomendação prática objetiva para o analista \
(ex.: “pode seguir com baixa supervisão”, “validar manualmente os ensaios X/Y”, \
“revisão manual obrigatória”).

## O que você NÃO DEVE fazer
- Inventar ou alterar o código CODRSTAFER, a camada ou os scores.
- Afirmar que o modelo “errou” ou propor outra classe como resposta oficial.
- Mencionar que você é uma IA, citar o nome do provedor/modelo ou falar de prompts.
- Incluir dados pessoais (nomes, usuários, documentos) mesmo que apareçam no contexto.
- Usar jargão de data science desnecessário; prefira linguagem clara para operação.
- Produzir JSON, markdown com títulos longos ou listas excessivas — texto corrido \
curto (2 a 4 parágrafos curtos, no máximo ~180 palavras).

## Tom
Técnico, objetivo, cauteloso quando a evidência for fraca, confiante quando a \
camada/proba/métricas forem fortes. Sem dramatização.
"""


def _fmt_metrics(metrics: ClassMetricsSnapshot | None) -> str:
    if metrics is None:
        return "Métricas por classe: indisponíveis neste request."
    parts: list[str] = []
    if metrics.f1 is not None:
        parts.append(f"F1={metrics.f1:.4f}")
    if metrics.precision is not None:
        parts.append(f"precision={metrics.precision:.4f}")
    if metrics.recall is not None:
        parts.append(f"recall={metrics.recall:.4f}")
    if metrics.support is not None:
        parts.append(f"support={metrics.support}")
    if metrics.tier is not None:
        parts.append(f"tier_treino={metrics.tier}")
    if not parts:
        return "Métricas por classe: presentes, porém sem valores numéricos."
    return "Métricas de validação da classe prevista: " + ", ".join(parts) + "."


def _fmt_topk(context: AnalysisContext) -> str:
    if not context.topk_proba:
        return "Distribuição predict_proba: indisponível."
    items = ", ".join(f"{code}={score:.4f}" for code, score in context.topk_proba)
    gap = (
        f" Gap top1–top2: {context.top1_top2_gap:.4f}."
        if context.top1_top2_gap is not None
        else ""
    )
    return f"Top-k predict_proba: {items}.{gap}"


def _fmt_features(context: AnalysisContext) -> str:
    if not context.feature_snapshot:
        return "Snapshot de features do laudo: vazio."
    pairs = ", ".join(f"{key}={value!r}" for key, value in context.feature_snapshot.items())
    return f"Snapshot de features do laudo: {pairs}."


def build_user_prompt(context: AnalysisContext) -> str:
    """Monta o prompt humano com evidências estruturadas (pt_BR)."""
    weight_text = (
        f"{context.weight:.6f}" if context.weight is not None else "indisponível"
    )
    proba_text = (
        f"{context.probabilidade:.4f}"
        if context.probabilidade is not None
        else "indisponível"
    )
    glossary_block = context.glossary_entries_text or (
        "Glossário CODRSTAFER: sem entradas para as classes deste request."
    )

    return f"""\
## Evidências estruturadas (não alterar)

- Número do laudo: {context.numero_laudo or "não informado"}
- Classe prevista (CODRSTAFER): {context.classe_prevista}
- Camada de qualificação: {context.camada}
- Peso balanceado no treino: {weight_text}
- Probabilidade da classe prevista: {proba_text}
- Resultado resumido (template): {context.resultado}
- Resultado detalhado (template factual): {context.resultado_detalhado}

{_fmt_topk(context)}
{_fmt_metrics(context.class_metrics)}

## Glossário CODRSTAFER (status={context.glossary_status})
{glossary_block}

{_fmt_features(context)}

## Tarefa
Com base EXCLUSIVAMENTE nas evidências acima, redija a revisão em português do \
Brasil para o campo de análise textual do laudo. Lembre-se: a classe \
{context.classe_prevista} (camada {context.camada}) é a fonte da verdade.
"""
