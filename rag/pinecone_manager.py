"""Gerenciamento do index no Pinecone.

Responsável por criar o index caso não exista, expor estatísticas e
construir o `PineconeVectorStore` usado pelo LlamaIndex.
"""

from __future__ import annotations

import time

from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from rag.config import Settings


class PineconeManager:
    """Encapsula as operações no index do Pinecone usado pelo sistema.

    Concentra criação do index, remoção de vetores (total ou por arquivo),
    consulta de estatísticas e a construção do `PineconeVectorStore` que o
    LlamaIndex usa para ler e escrever vetores.

    Attributes:
        settings: Configurações do projeto (nome do index, namespace, região etc.).
        client: Cliente autenticado da API do Pinecone.
    """

    def __init__(self, settings: Settings) -> None:
        """Cria o cliente do Pinecone a partir da chave de API configurada.

        Args:
            settings: Configurações do projeto.
        """
        self.settings = settings
        self.client = Pinecone(api_key=settings.pinecone_api_key)

    # ------------------------------------------------------------------ #
    # Index lifecycle
    # ------------------------------------------------------------------ #
    def index_exists(self) -> bool:
        """Verifica se o index configurado já existe na conta do Pinecone.

        Returns:
            `True` se o index existir, `False` caso contrário.
        """
        return self.client.has_index(self.settings.index_name)

    def ensure_index(self) -> bool:
        """Cria o index serverless caso ele ainda não exista.

        O index é criado com a dimensão do modelo de embedding configurado e
        métrica de cosseno. Após a criação, aguarda (até ~60 s) o index ficar
        pronto para receber upserts.

        Returns:
            `True` se um index novo foi criado, `False` se já existia.

        Raises:
            ValueError: se a dimensão do embedding não for conhecida.
        """
        if self.index_exists():
            return False

        self.client.create_index(
            name=self.settings.index_name,
            dimension=self.settings.embedding_dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud=self.settings.cloud, region=self.settings.region),
        )
        # Aguarda o index ficar pronto para receber upserts.
        for _ in range(60):
            status = self.client.describe_index(self.settings.index_name).status
            if status.get("ready"):
                break
            time.sleep(1)
        return True

    def delete_namespace(self) -> None:
        """Remove todos os vetores do namespace configurado.

        O index em si é preservado. Se o index ou o namespace não existirem,
        a operação é silenciosamente ignorada.
        """
        if not self.index_exists():
            return
        index = self.client.Index(self.settings.index_name)
        try:
            index.delete(delete_all=True, namespace=self.settings.namespace)
        except Exception as exc:  # namespace inexistente retorna 404
            if "404" not in str(exc) and "Namespace not found" not in str(exc):
                raise

    def delete_by_filename(self, file_name: str) -> None:
        """Remove os vetores de um único arquivo usando filtro de metadados.

        Depende do metadado `file_name` gravado em cada chunk durante a
        ingestão. Se o index não existir, nada é feito.

        Args:
            file_name: Nome do arquivo (sem diretório) cujos vetores serão removidos.
        """
        if not self.index_exists():
            return
        index = self.client.Index(self.settings.index_name)
        index.delete(filter={"file_name": {"$eq": file_name}}, namespace=self.settings.namespace)

    # ------------------------------------------------------------------ #
    # Informações
    # ------------------------------------------------------------------ #
    def stats(self) -> dict:
        """Retorna estatísticas resumidas do index e do namespace configurado.

        Returns:
            Dicionário com as chaves:
            - `exists`: se o index existe;
            - `dimension`: dimensão dos vetores (ausente se o index não existir);
            - `total_vectors`: total de vetores em todos os namespaces;
            - `namespace_vectors`: vetores apenas no namespace configurado.
        """
        if not self.index_exists():
            return {"exists": False, "total_vectors": 0, "namespace_vectors": 0}
        index = self.client.Index(self.settings.index_name)
        raw = index.describe_index_stats()
        namespaces = raw.get("namespaces", {}) or {}
        ns = namespaces.get(self.settings.namespace, {}) or {}
        return {
            "exists": True,
            "dimension": raw.get("dimension"),
            "total_vectors": raw.get("total_vector_count", 0),
            "namespace_vectors": ns.get("vector_count", 0),
        }

    # ------------------------------------------------------------------ #
    # Integração LlamaIndex
    # ------------------------------------------------------------------ #
    def vector_store(self) -> PineconeVectorStore:
        """Constrói o `PineconeVectorStore` do LlamaIndex apontando para o namespace.

        Garante que o index exista antes de criar o store.

        Returns:
            Vector store pronto para uso em `StorageContext` ou `VectorStoreIndex`.
        """
        self.ensure_index()
        pinecone_index = self.client.Index(self.settings.index_name)
        return PineconeVectorStore(
            pinecone_index=pinecone_index,
            namespace=self.settings.namespace,
        )
