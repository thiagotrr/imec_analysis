"""Pacote de análise textual e integração LLM (Task 008).

- ``analysis`` — template factual A–C
- ``context`` — evidências para o prompt
- ``glossary`` — glossário CODRSTAFER
- ``config`` — settings via ``.env`` (OpenAI / Gemini)
- ``gate`` — disparo condicional
- ``prompts`` — prompts pt_BR
- ``reviewer`` — LangChain + fail-soft

Importe dos submódulos (ex.: ``from llm.analysis import compose_resultado``)
para evitar carregamento circular com ``api.inference_pipeline``.
"""
