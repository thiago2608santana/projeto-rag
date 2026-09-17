"""Motor de consulta: recupera contexto do Pinecone e gera respostas com a OpenAI.

Usa o modo `condense_plus_context` do LlamaIndex: a pergunta é reescrita
levando em conta o histórico da conversa e, em seguida, o contexto
recuperado é injetado no prompt do LLM.
"""

from __future__ import annotations

from dataclasses import dataclass

from llama_index.core import VectorStoreIndex
from llama_index.core.chat_engine.types import BaseChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.postprocessor import SimilarityPostprocessor

from rag.config import Settings
from rag.llm import configure_llama_index
from rag.pinecone_manager import PineconeManager

SYSTEM_PROMPT = """Você é um assistente que responde perguntas com base exclusivamente
nos documentos fornecidos como contexto.

Regras:
- Responda em português do Brasil, de forma objetiva e bem estruturada.
- Use apenas as informações presentes no contexto. Se a resposta não estiver
  nos documentos, diga claramente que não encontrou a informação.
- Quando citar dados específicos, mencione o nome do arquivo de origem entre colchetes,
  por exemplo: [relatorio.pdf].
- Não invente números, datas ou nomes.
"""


@dataclass
class Source:
    """Trecho de documento recuperado e usado como contexto para uma resposta.

    Attributes:
        file_name: Nome do arquivo de origem.
        page: Rótulo da página (quando o leitor fornece, ex.: PDFs), ou `None`.
        score: Similaridade com a pergunta, arredondada a 3 casas, ou `None`.
        text: Conteúdo do chunk.
    """

    file_name: str
    page: str | None
    score: float | None
    text: str


@dataclass
class Answer:
    """Resposta completa do engine (modo não-streaming).

    Attributes:
        text: Texto gerado pelo LLM.
        sources: Trechos recuperados que embasaram a resposta.
    """

    text: str
    sources: list[Source]


class RagEngine:
    """Chat engine com memória que responde perguntas a partir dos documentos indexados.

    Monta um `VectorStoreIndex` sobre o Pinecone e um chat engine no modo
    `condense_plus_context`: a pergunta é condensada com o histórico da conversa,
    os chunks mais similares são recuperados e injetados no prompt, e o LLM gera
    a resposta seguindo `SYSTEM_PROMPT`.

    Os parâmetros de recuperação e geração são fixados na construção; para
    alterá-los, crie um novo engine (o que também zera a memória).

    Attributes:
        settings: Configurações do projeto.
        memory: Buffer de memória conversacional compartilhado com o chat engine.
        chat_engine: Chat engine do LlamaIndex.
    """

    def __init__(
        self,
        settings: Settings,
        pinecone: PineconeManager,
        top_k: int | None = None,
        similarity_cutoff: float = 0.0,
        temperature: float | None = None,
    ) -> None:
        """Configura o LlamaIndex e constrói o chat engine.

        Args:
            settings: Configurações do projeto.
            pinecone: Gerenciador do index no Pinecone.
            top_k: Quantidade de chunks recuperados. Se `None`, usa `settings.top_k`.
            similarity_cutoff: Score mínimo para manter um chunk; 0 desativa o filtro.
            temperature: Temperatura do LLM. Se `None`, usa `settings.temperature`.
        """
        self.settings = settings
        configure_llama_index(settings, temperature=temperature)

        index = VectorStoreIndex.from_vector_store(pinecone.vector_store())
        postprocessors = []
        if similarity_cutoff > 0:
            postprocessors.append(SimilarityPostprocessor(similarity_cutoff=similarity_cutoff))

        self.memory = ChatMemoryBuffer.from_defaults(token_limit=4000)
        self.chat_engine: BaseChatEngine = index.as_chat_engine(
            chat_mode="condense_plus_context",
            memory=self.memory,
            system_prompt=SYSTEM_PROMPT,
            similarity_top_k=top_k or settings.top_k,
            node_postprocessors=postprocessors,
            verbose=False,
        )

    def ask(self, question: str) -> Answer:
        """Responde a uma pergunta de forma síncrona (resposta completa de uma vez).

        A pergunta e a resposta são adicionadas à memória da conversa.

        Args:
            question: Pergunta do usuário.

        Returns:
            Texto da resposta e os trechos usados como contexto.
        """
        response = self.chat_engine.chat(question)
        sources = [
            Source(
                file_name=node.metadata.get("file_name", "desconhecido"),
                page=node.metadata.get("page_label"),
                score=round(node.score, 3) if node.score is not None else None,
                text=node.get_content().strip(),
            )
            for node in response.source_nodes
        ]
        return Answer(text=str(response), sources=sources)

    def stream(self, question: str):
        """Gera a resposta token a token.

        Os trechos recuperados ficam disponíveis em `last_sources` assim que o
        gerador começa a ser consumido, antes mesmo do término da resposta.
        A pergunta e a resposta são adicionadas à memória da conversa.

        Args:
            question: Pergunta do usuário.

        Yields:
            Fragmentos de texto da resposta, na ordem em que o LLM os produz.
        """
        response = self.chat_engine.stream_chat(question)
        self._last_sources = [
            Source(
                file_name=node.metadata.get("file_name", "desconhecido"),
                page=node.metadata.get("page_label"),
                score=round(node.score, 3) if node.score is not None else None,
                text=node.get_content().strip(),
            )
            for node in response.source_nodes
        ]
        yield from response.response_gen

    @property
    def last_sources(self) -> list[Source]:
        """Trechos recuperados na última chamada a `stream()`.

        Returns:
            Lista de fontes, ou lista vazia se `stream()` ainda não foi chamado.
        """
        return getattr(self, "_last_sources", [])

    def reset(self) -> None:
        """Limpa a memória da conversa, mantendo o índice e os parâmetros."""
        self.chat_engine.reset()
