"""Gate condicional para disparo da revisão LLM (Task 008)."""
from __future__ import annotations

from dataclasses import dataclass

from .config import LlmSettings
from .context import AnalysisContext


@dataclass(frozen=True)
class LlmGateDecision:
    should_call: bool
    reason: str


def should_request_llm(
    context: AnalysisContext,
    settings: LlmSettings,
    *,
    force: bool | None = None,
) -> LlmGateDecision:
    """Decide se a revisão LLM deve ser solicitada.

    Regras (default):
    - ``force=True`` → sempre chama (quando o provedor estiver configurado).
    - ``force=False`` → nunca chama.
    - Camada A (classes majoritárias): **não** chama por default (`skip_tier_a`).
    - Camadas B/C/D: chama.
    - Mesmo fora de A, baixa proba ou gap top1–top2 pequeno reforçam o disparo
      (já coberto por B/C/D; útil se ``skip_tier_a=False``).
    """
    if force is True:
        return LlmGateDecision(True, "forçado via parâmetro revisao_llm=true")
    if force is False:
        return LlmGateDecision(False, "desabilitado via parâmetro revisao_llm=false")

    if not settings.enabled:
        return LlmGateDecision(False, "LLM_ENABLED=false")

    if not settings.is_configured():
        return LlmGateDecision(False, f"provedor {settings.provider} sem API key configurada")

    camada = (context.camada or "").upper()
    if settings.skip_tier_a and camada == "A":
        return LlmGateDecision(
            False,
            "camada A (classe majoritária): LLM omitido por default (LLM_SKIP_TIER_A)",
        )

    if camada in {"B", "C"} or camada == context.discard_tier_label:
        return LlmGateDecision(True, f"camada {camada}: revisão LLM recomendada")

    probabilidade = context.probabilidade
    if probabilidade is not None and probabilidade < settings.min_proba:
        return LlmGateDecision(
            True,
            f"probabilidade {probabilidade:.4f} < LLM_MIN_PROBA={settings.min_proba}",
        )

    gap = context.top1_top2_gap
    if gap is not None and gap < settings.min_top_gap:
        return LlmGateDecision(
            True,
            f"gap top1–top2 {gap:.4f} < LLM_MIN_TOP_GAP={settings.min_top_gap}",
        )

    return LlmGateDecision(False, "critérios de gate não atendidos")
