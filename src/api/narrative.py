"""Composição textual do resultado da inspeção (template A–C, sem LLM)."""
from __future__ import annotations

_TIER_BLURBS: dict[str, str] = {
    "A": (
        "Camada A: classe dominante no histórico de treino (≥15% das amostras) — "
        "sinal de maior confiabilidade estatística."
    ),
    "B": (
        "Camada B: classe com representação intermediária (1–15%) — "
        "sinal sólido, porém menos frequente que as classes da camada A."
    ),
    "C": (
        "Camada C: classe rara (0,1–1%) — zona cinza; recomenda-se atenção "
        "adicional na validação do laudo."
    ),
}


def compose_resultado(classe_prevista: str, camada: str, discard_tier_label: str = "D") -> str:
    """Texto curto: código + camada; fora do registry / D → revisão manual."""
    if camada == discard_tier_label:
        return "Revisão manual"
    return f"Classe {classe_prevista} (camada {camada})"


def compose_resultado_detalhado(
    *,
    classe_prevista: str,
    camada: str,
    weight: float | None,
    probabilidade: float | None,
    discard_tier_label: str = "D",
) -> str:
    """Narrativa template citando camada/peso (e score quando disponível)."""
    if camada == discard_tier_label:
        parts = [
            f"Classe prevista {classe_prevista} está fora do escopo do registry "
            f"(camada {discard_tier_label}) ou não consta em class_weight_registry.json.",
            "Recomenda-se revisão manual do laudo.",
        ]
    else:
        blurb = _TIER_BLURBS.get(
            camada,
            f"Camada {camada}: sem descrição padrão cadastrada.",
        )
        weight_text = (
            f"Peso balanceado da classe no treino: {weight:.6f}."
            if weight is not None
            else "Peso da classe não disponível no registry."
        )
        parts = [
            f"Classe prevista: {classe_prevista} (camada {camada}).",
            blurb,
            weight_text,
        ]

    if probabilidade is not None:
        parts.append(f"Probabilidade da classe prevista (predict_proba): {probabilidade:.4f}.")

    return " ".join(parts)
