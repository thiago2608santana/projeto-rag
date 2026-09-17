"""Pipeline de ingestão: pasta local -> chunks -> embeddings -> Pinecone.

Um manifesto (`data/manifest.json`) registra o hash de cada arquivo já
indexado, para que apenas arquivos novos ou modificados sejam processados.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter

from rag.config import SUPPORTED_EXTENSIONS, Settings
from rag.llm import configure_llama_index
from rag.pinecone_manager import PineconeManager


@dataclass
class IngestionResult:
    """Resumo de uma execução de `Ingestor.sync`.

    Attributes:
        indexed: Nomes dos arquivos indexados pela primeira vez.
        updated: Nomes dos arquivos re-indexados por terem sido alterados.
        skipped: Nomes dos arquivos ignorados por já estarem indexados sem alteração.
        failed: Mapeamento `nome do arquivo -> mensagem de erro` para os que falharam.
        total_chunks: Soma dos chunks gerados para os arquivos indexados ou atualizados.
    """

    indexed: list[str]
    updated: list[str]
    skipped: list[str]
    failed: dict[str, str]
    total_chunks: int


def _sha256(path: Path) -> str:
    """Calcula o hash SHA-256 do conteúdo de um arquivo.

    Lê o arquivo em blocos de 1 MiB para não carregar arquivos grandes na memória.

    Args:
        path: Caminho do arquivo.

    Returns:
        Hash em hexadecimal.
    """
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class Manifest:
    """Registro local dos arquivos já indexados no Pinecone.

    Persistido como JSON em disco. Cada entrada é indexada pelo nome do arquivo
    e guarda o hash SHA-256, o número de chunks gerados e a data de ingestão.
    É a fonte de verdade local sobre o que está no vector store; se o namespace
    for limpo por fora, o manifesto precisa ser zerado também.

    Attributes:
        path: Caminho do arquivo JSON.
        data: Conteúdo do manifesto em memória (`nome do arquivo -> entrada`).
    """

    def __init__(self, path: Path) -> None:
        """Carrega o manifesto do disco, ou inicia vazio se o arquivo não existir.

        Args:
            path: Caminho do arquivo JSON do manifesto.
        """
        self.path = path
        self.data: dict[str, dict] = {}
        if path.exists():
            self.data = json.loads(path.read_text(encoding="utf-8"))

    def save(self) -> None:
        """Grava o conteúdo atual do manifesto em disco, em JSON indentado."""
        self.path.write_text(json.dumps(self.data, indent=2, ensure_ascii=False), encoding="utf-8")

    def get(self, file_name: str) -> dict | None:
        """Retorna a entrada de um arquivo, ou `None` se ele não estiver indexado.

        Args:
            file_name: Nome do arquivo (sem diretório).

        Returns:
            Dicionário com `sha256`, `chunks` e `ingested_at`, ou `None`.
        """
        return self.data.get(file_name)

    def set(self, file_name: str, sha: str, chunks: int) -> None:
        """Cria ou substitui a entrada de um arquivo, marcando a data atual em UTC.

        Não grava em disco; chame `save()` em seguida.

        Args:
            file_name: Nome do arquivo (sem diretório).
            sha: Hash SHA-256 do conteúdo indexado.
            chunks: Quantidade de chunks gerados para o arquivo.
        """
        self.data[file_name] = {
            "sha256": sha,
            "chunks": chunks,
            "ingested_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }

    def remove(self, file_name: str) -> None:
        """Remove a entrada de um arquivo, se existir. Não grava em disco.

        Args:
            file_name: Nome do arquivo (sem diretório).
        """
        self.data.pop(file_name, None)

    def clear(self) -> None:
        """Descarta todas as entradas em memória. Não grava em disco."""
        self.data = {}


class Ingestor:
    """Orquestra a ingestão de documentos da pasta local para o Pinecone.

    Gerencia os arquivos em `settings.documents_dir`, mantém o manifesto
    sincronizado e executa o pipeline leitura -> chunking -> embedding -> upsert.

    Attributes:
        settings: Configurações do projeto.
        pinecone: Gerenciador do index no Pinecone.
        manifest: Registro dos arquivos já indexados.
    """

    def __init__(self, settings: Settings, pinecone: PineconeManager) -> None:
        """Carrega o manifesto e configura os modelos globais do LlamaIndex.

        Args:
            settings: Configurações do projeto.
            pinecone: Gerenciador do index no Pinecone.
        """
        self.settings = settings
        self.pinecone = pinecone
        self.manifest = Manifest(settings.manifest_path)
        configure_llama_index(settings)

    # ------------------------------------------------------------------ #
    def list_local_documents(self) -> list[Path]:
        """Lista os arquivos suportados presentes na pasta de documentos.

        Considera apenas arquivos (não subpastas) cuja extensão esteja em
        `SUPPORTED_EXTENSIONS`, ignorando maiúsculas/minúsculas.

        Returns:
            Caminhos ordenados alfabeticamente.
        """
        return sorted(
            p
            for p in self.settings.documents_dir.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        )

    def save_uploaded(self, file_name: str, content: bytes) -> Path:
        """Grava um arquivo enviado pela interface na pasta de documentos.

        Usa apenas o nome base do arquivo, descartando qualquer diretório
        informado. Arquivos com o mesmo nome são sobrescritos. Não indexa;
        chame `sync()` em seguida.

        Args:
            file_name: Nome do arquivo enviado.
            content: Conteúdo binário do arquivo.

        Returns:
            Caminho do arquivo gravado.
        """
        target = self.settings.documents_dir / Path(file_name).name
        target.write_bytes(content)
        return target

    def remove_document(self, file_name: str) -> None:
        """Remove um documento por completo: arquivo local, vetores e entrada no manifesto.

        Args:
            file_name: Nome do arquivo (sem diretório).
        """
        path = self.settings.documents_dir / file_name
        if path.exists():
            path.unlink()
        self.pinecone.delete_by_filename(file_name)
        self.manifest.remove(file_name)
        self.manifest.save()

    def reset(self) -> None:
        """Apaga todos os vetores do namespace e zera o manifesto.

        Os arquivos locais são mantidos; uma chamada subsequente a `sync()`
        re-indexa tudo do zero.
        """
        self.pinecone.delete_namespace()
        self.manifest.clear()
        self.manifest.save()

    # ------------------------------------------------------------------ #
    def sync(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        progress: Callable[[str], None] | None = None,
    ) -> IngestionResult:
        """Indexa arquivos novos ou alterados presentes na pasta local.

        Para cada arquivo suportado, calcula o SHA-256 e compara com o manifesto:
        - hash igual: o arquivo é ignorado;
        - arquivo já conhecido com hash diferente: os vetores antigos são apagados
          (filtro por `file_name`) e o arquivo é re-indexado;
        - arquivo novo: é indexado.

        Cada documento recebe o metadado `file_name`, usado para deleção seletiva
        e para as citações na interface. Metadados irrelevantes (caminho, tamanho,
        datas) são excluídos do texto enviado ao LLM e ao embedding. Falhas em um
        arquivo são registradas em `IngestionResult.failed` sem abortar o lote.

        Args:
            chunk_size: Tamanho dos chunks em tokens. Se `None`, usa `settings.chunk_size`.
            chunk_overlap: Sobreposição entre chunks. Se `None`, usa `settings.chunk_overlap`.
            progress: Callback opcional que recebe mensagens de andamento.

        Returns:
            Resumo do que foi indexado, atualizado, ignorado ou falhou.
        """
        chunk_size = chunk_size or self.settings.chunk_size
        chunk_overlap = chunk_overlap or self.settings.chunk_overlap
        log = progress or (lambda _msg: None)

        result = IngestionResult(indexed=[], updated=[], skipped=[], failed={}, total_chunks=0)
        files = self.list_local_documents()
        if not files:
            return result

        vector_store = self.pinecone.vector_store()
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        for path in files:
            name = path.name
            sha = _sha256(path)
            entry = self.manifest.get(name)

            if entry and entry.get("sha256") == sha:
                result.skipped.append(name)
                continue

            is_update = entry is not None
            log(f"{'Atualizando' if is_update else 'Indexando'} {name}…")
            try:
                if is_update:
                    self.pinecone.delete_by_filename(name)

                docs = SimpleDirectoryReader(
                    input_files=[str(path)],
                    filename_as_id=True,
                ).load_data()
                for doc in docs:
                    doc.metadata["file_name"] = name
                    # Evita que campos irrelevantes poluam o contexto enviado ao LLM.
                    doc.excluded_llm_metadata_keys = ["file_path", "file_type", "file_size",
                                                      "creation_date", "last_modified_date"]
                    doc.excluded_embed_metadata_keys = doc.excluded_llm_metadata_keys

                nodes = splitter.get_nodes_from_documents(docs)
                VectorStoreIndex(nodes=nodes, storage_context=storage_context, show_progress=False)

                self.manifest.set(name, sha, len(nodes))
                self.manifest.save()
                result.total_chunks += len(nodes)
                (result.updated if is_update else result.indexed).append(name)
            except Exception as exc:  # noqa: BLE001 — queremos reportar, não abortar o lote
                result.failed[name] = str(exc)
                log(f"Falha em {name}: {exc}")

        return result
