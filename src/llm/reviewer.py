"""Revisor LLM via LangChain (OpenAI / Gemini) — pós-processamento fail-soft."""
from __future__ import annotations

from typing import Protocol

from log import get_log

from .config import LlmSettings, load_llm_settings
from .context import AnalysisContext
from .gate import LlmGateDecision, should_request_llm
from .prompts import SYSTEM_PROMPT_PT_BR, build_user_prompt

log = get_log()


class SupportsReview(Protocol):
    def review(self, context: AnalysisContext) -> str | None: ...


class NullLlmReviewer:
    """No-op: usado quando LLM está desabilitado ou sem chave."""

    def review(self, context: AnalysisContext) -> str | None:
        return None


class FakeLlmReviewer:
    """Reviewer determinístico para testes (não chama rede)."""

    def __init__(self, text: str = "Revisão LLM fake para testes.") -> None:
        self.text = text
        self.calls: list[AnalysisContext] = []

    def review(self, context: AnalysisContext) -> str | None:
        self.calls.append(context)
        return self.text


class LangChainLlmReviewer:
    """Abstração LangChain sobre ChatOpenAI ou ChatGoogleGenerativeAI."""

    def __init__(self, settings: LlmSettings | None = None) -> None:
        self.settings = settings or load_llm_settings()
        self._chat_model = None

    def _build_chat_model(self):
        if self._chat_model is not None:
            return self._chat_model

        settings = self.settings
        if not settings.is_configured():
            raise RuntimeError(
                f"LLM não configurado para provedor={settings.provider!r} "
                "(API key ausente ou LLM_ENABLED=false)."
            )

        if settings.provider == "openai":
            from langchain_openai import ChatOpenAI

            self._chat_model = ChatOpenAI(
                model=settings.openai_model,
                api_key=settings.openai_api_key,
                temperature=settings.temperature,
                timeout=settings.timeout_seconds,
            )
        elif settings.provider == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI

            # langchain-google-genai >=4 aceita api_key (GEMINI_API_KEY / GOOGLE_API_KEY).
            self._chat_model = ChatGoogleGenerativeAI(
                model=settings.gemini_model,
                api_key=settings.gemini_api_key,
                temperature=settings.temperature,
                timeout=settings.timeout_seconds,
            )
        else:  # pragma: no cover - validado em load_llm_settings
            raise RuntimeError(f"Provedor LLM não suportado: {settings.provider!r}")

        return self._chat_model

    def review(self, context: AnalysisContext) -> str | None:
        from langchain_core.messages import HumanMessage, SystemMessage

        try:
            chat = self._build_chat_model()
            messages = [
                SystemMessage(content=SYSTEM_PROMPT_PT_BR),
                HumanMessage(content=build_user_prompt(context)),
            ]
            response = chat.invoke(messages)
            content = getattr(response, "content", None)
            if content is None:
                return None
            if isinstance(content, list):
                # Alguns provedores retornam blocos multimodais.
                text_parts = [
                    block.get("text", "") if isinstance(block, dict) else str(block)
                    for block in content
                ]
                text = " ".join(part for part in text_parts if part).strip()
            else:
                text = str(content).strip()
            return text or None
        except Exception:
            log.exception(
                "Falha fail-soft na revisão LLM (provider=%s model=%s)",
                self.settings.provider,
                self.settings.active_model,
            )
            return None


def build_default_reviewer(settings: LlmSettings | None = None) -> SupportsReview:
    resolved = settings or load_llm_settings()
    if not resolved.is_configured():
        return NullLlmReviewer()
    return LangChainLlmReviewer(resolved)


def generate_revisao_llm(
    context: AnalysisContext,
    reviewer: SupportsReview | None,
    settings: LlmSettings | None = None,
    *,
    force: bool | None = None,
) -> tuple[str | None, LlmGateDecision]:
    """Aplica gate + reviewer; nunca propaga exceção ao caller HTTP."""
    resolved_settings = settings or load_llm_settings()
    decision = should_request_llm(context, resolved_settings, force=force)
    if not decision.should_call:
        log.info("LLM gate: omitido (%s)", decision.reason)
        return None, decision

    active = reviewer or build_default_reviewer(resolved_settings)
    try:
        text = active.review(context)
    except Exception:
        log.exception("Reviewer LLM levantou exceção inesperada; fail-soft → None")
        return None, decision

    if text:
        log.info("LLM gate: revisão gerada (%s)", decision.reason)
    else:
        log.warning("LLM gate: chamado mas sem texto (%s)", decision.reason)
    return text, decision
