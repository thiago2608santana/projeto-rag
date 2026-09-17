"""Configuração central do sistema RAG.

Lê variáveis do arquivo .env e expõe um objeto `Settings` imutável
com todos os parâmetros usados pela ingestão e pela consulta.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def _export_streamlit_secrets() -> None:
    """Exporta os secrets do Streamlit para variáveis de ambiente.

    Em produção (Streamlit Community Cloud) não existe `.env`: as chaves vêm do
    gerenciador de secrets do painel. O Streamlit publica os secrets de nível
    raiz em `os.environ`, mas de forma preguiçosa — isso só acontece no primeiro
    acesso real a `st.secrets`. Como `Settings` lê o ambiente direto via
    `os.getenv`, sem este toque nada teria sido exportado ainda e a inicialização
    falharia no deploy.

    O `len()` abaixo é apenas o gatilho desse carregamento: um único acesso faz o
    parse do arquivo inteiro e exporta *todas* as chaves de nível raiz de uma vez
    (OpenAI, Pinecone e as opcionais). Não é preciso citar chave por chave.

    Fora de um app Streamlit (scripts, testes), não há o que carregar e a função
    simplesmente não faz nada.
    """
    try:
        import streamlit as st

        len(st.secrets)
    except Exception:  # noqa: BLE001 - ausência de secrets é o caso normal local
        pass


_export_streamlit_secrets()

# Dimensões dos modelos de embedding da OpenAI (usadas ao criar o index no Pinecone).
EMBEDDING_DIMENSIONS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".csv", ".json", ".html", ".pptx"}


def _env_flag(name: str, default: bool = False) -> bool:
    """Lê uma variável de ambiente booleana.

    Aceita `1`, `true`, `yes` e `on` (sem diferenciar maiúsculas) como verdadeiro;
    qualquer outro valor preenchido é falso.

    Args:
        name: Nome da variável de ambiente.
        default: Valor usado quando a variável não está definida.

    Returns:
        O booleano correspondente.
    """
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _require(name: str) -> str:
    """Lê uma variável de ambiente obrigatória.

    Args:
        name: Nome da variável de ambiente.

    Returns:
        O valor da variável.

    Raises:
        RuntimeError: se a variável não estiver definida ou estiver vazia.
    """
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"Variável de ambiente '{name}' não encontrada. "
            "Localmente, verifique o arquivo .env na raiz do projeto; "
            "no Streamlit Community Cloud, verifique os secrets do app."
        )
    return value


@dataclass(frozen=True)
class Settings:
    """Parâmetros de configuração do sistema, carregados do ambiente.

    Todos os campos são preenchidos a partir de variáveis de ambiente no momento
    da instanciação. As credenciais e o nome do modelo de LLM são obrigatórios
    (`OPENAI_API_KEY`, `PINECONE_API_KEY`, `OPENAI_MODEL_NAME`); os demais campos
    têm valores padrão. A instância é imutável (`frozen=True`).

    Attributes:
        openai_api_key: Chave da API da OpenAI.
        pinecone_api_key: Chave da API do Pinecone.
        llm_model: Nome do modelo de chat da OpenAI usado para gerar respostas.
        embedding_model: Nome do modelo de embedding da OpenAI.
        index_name: Nome do index no Pinecone.
        namespace: Namespace dentro do index onde os vetores são armazenados.
        cloud: Provedor de nuvem do index serverless (ex.: "aws").
        region: Região do index serverless (ex.: "us-east-1").
        chunk_size: Tamanho máximo de cada chunk, em tokens.
        chunk_overlap: Sobreposição entre chunks consecutivos, em tokens.
        top_k: Número padrão de chunks recuperados por consulta.
        temperature: Temperatura padrão do LLM.
        documents_dir: Pasta local onde os documentos-fonte ficam armazenados.
        manifest_path: Caminho do arquivo JSON que registra o que já foi indexado.
    """

    # Credenciais / modelos
    openai_api_key: str = field(default_factory=lambda: _require("OPENAI_API_KEY"))
    pinecone_api_key: str = field(default_factory=lambda: _require("PINECONE_API_KEY"))
    llm_model: str = field(default_factory=lambda: _require("OPENAI_MODEL_NAME"))
    embedding_model: str = field(
        default_factory=lambda: os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    )

    # Pinecone
    index_name: str = field(default_factory=lambda: os.getenv("PINECONE_INDEX_NAME", "rag-local"))
    namespace: str = field(default_factory=lambda: os.getenv("PINECONE_NAMESPACE", "default"))
    cloud: str = field(default_factory=lambda: os.getenv("PINECONE_CLOUD", "aws"))
    region: str = field(default_factory=lambda: os.getenv("PINECONE_REGION", "us-east-1"))

    # Chunking / recuperação (valores padrão; podem ser alterados pela interface)
    chunk_size: int = field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "1024")))
    chunk_overlap: int = field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "128")))
    top_k: int = field(default_factory=lambda: int(os.getenv("TOP_K", "5")))
    temperature: float = field(default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0.1")))

    # Operações destrutivas: desligadas por padrão (ver ALLOW_INDEX_RESET no README).
    allow_index_reset: bool = field(default_factory=lambda: _env_flag("ALLOW_INDEX_RESET"))

    # Pastas locais
    documents_dir: Path = field(default_factory=lambda: BASE_DIR / "data" / "documents")
    manifest_path: Path = field(default_factory=lambda: BASE_DIR / "data" / "manifest.json")

    @property
    def embedding_dimension(self) -> int:
        """Dimensão dos vetores gerados pelo modelo de embedding configurado.

        Usada para criar o index no Pinecone com o tamanho correto.

        Returns:
            Número de dimensões do embedding.

        Raises:
            ValueError: se o modelo não constar em `EMBEDDING_DIMENSIONS`.
        """
        try:
            return EMBEDDING_DIMENSIONS[self.embedding_model]
        except KeyError as exc:
            raise ValueError(
                f"Dimensão desconhecida para o embedding '{self.embedding_model}'. "
                f"Modelos suportados: {', '.join(EMBEDDING_DIMENSIONS)}"
            ) from exc


def get_settings() -> Settings:
    """Cria o objeto `Settings` e garante que as pastas locais existam.

    Cria `documents_dir` e o diretório pai de `manifest_path` caso ainda não
    existam, para que ingestão e manifesto possam ser usados imediatamente.

    Returns:
        Instância de `Settings` pronta para uso.

    Raises:
        RuntimeError: se alguma variável de ambiente obrigatória estiver ausente.
    """
    settings = Settings()
    settings.documents_dir.mkdir(parents=True, exist_ok=True)
    settings.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    return settings
