"""Configuração global do LlamaIndex (LLM e modelo de embedding da OpenAI)."""

from __future__ import annotations

from llama_index.core import Settings as LlamaSettings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

from rag.config import Settings


def configure_llama_index(settings: Settings, temperature: float | None = None) -> None:
    """Configura os `Settings` globais do LlamaIndex com os modelos da OpenAI.

    Define o LLM de chat, o modelo de embedding e os parâmetros padrão de
    chunking usados por todos os componentes do LlamaIndex no processo.
    Deve ser chamada antes de criar índices ou chat engines.

    Args:
        settings: Configurações do projeto (modelos, chave de API, chunking).
        temperature: Temperatura do LLM. Se `None`, usa `settings.temperature`.
    """
    LlamaSettings.llm = OpenAI(
        model=settings.llm_model,
        api_key=settings.openai_api_key,
        temperature=settings.temperature if temperature is None else temperature,
    )
    LlamaSettings.embed_model = OpenAIEmbedding(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
    )
    LlamaSettings.chunk_size = settings.chunk_size
    LlamaSettings.chunk_overlap = settings.chunk_overlap
