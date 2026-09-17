# CLAUDE.md

Contexto para assistentes de IA trabalhando neste repositório.

## O que é este projeto

Sistema RAG (Retrieval-Augmented Generation) local com interface Streamlit.
Documentos ficam em `data/documents/`, são chunkados e embedados via LlamaIndex,
armazenados no Pinecone (serverless) e consultados por um chat engine com memória
conversacional usando modelos da OpenAI.

## Stack

| Camada          | Tecnologia                                  |
|-----------------|---------------------------------------------|
| Linguagem       | Python 3.13 (gerenciado por `uv`)            |
| Interface       | Streamlit                                    |
| Orquestração    | LlamaIndex (`llama-index-core` + integrações)|
| Vetores         | Pinecone serverless                          |
| LLM / Embedding | OpenAI (`OPENAI_MODEL_NAME` / `text-embedding-3-small`) |

## Comandos

```bash
uv sync                          # instala dependências
uv run streamlit run app.py      # sobe a interface
uv run ruff check app.py rag     # lint
```

## Estrutura

```
app.py                  # Interface Streamlit (única página)
rag/
  config.py             # Settings: lê .env, define pastas, dimensões de embedding
  llm.py                # Configura Settings globais do LlamaIndex (LLM + embed)
  pinecone_manager.py   # Cria index se não existir, stats, delete, vector_store()
  ingestion.py          # Ingestor: pasta local -> nós -> Pinecone; manifesto anti-duplicação
  engine.py             # RagEngine: chat engine condense_plus_context, fontes, streaming
data/
  documents/            # Documentos base (persistentes, versionados fora do git)
  manifest.json         # Gerado: hash SHA-256 + nº de chunks por arquivo indexado
docs/
  arquitetura.md        # Diagramas (Mermaid) dos fluxos + referência de cada função/classe
```

## Fluxos principais

**Inicialização** (`app.load_backend`, cacheado com `st.cache_resource`):
`get_settings()` → `PineconeManager.ensure_index()` cria o index com a dimensão
correta para o embedding configurado → `Ingestor` carrega o manifesto.

**Ingestão** (`Ingestor.sync`): percorre `data/documents/`, calcula SHA-256, ignora
arquivos já indexados com o mesmo hash, re-indexa arquivos alterados (apagando os
vetores antigos por filtro `file_name`), usa `SimpleDirectoryReader` +
`SentenceSplitter` e insere via `VectorStoreIndex(nodes, storage_context)`.

**Consulta** (`RagEngine`): `VectorStoreIndex.from_vector_store` + `as_chat_engine(
chat_mode="condense_plus_context")`. `top_k`, `similarity_cutoff` e `temperature`
vêm dos sliders da sidebar; mudar qualquer um reconstrói o engine (e zera a memória).

## Convenções e decisões

- Toda vetorização passa pelo metadado `file_name`; ele é a chave para deleção seletiva
  e para as citações mostradas na interface. Não remova esse campo.
- O manifesto é a fonte de verdade local sobre o que está no Pinecone. Se o namespace
  for limpo por fora, use "Reindexar tudo do zero" na interface.
- Dimensão do index é derivada de `EMBEDDING_DIMENSIONS` em `config.py`. Trocar de
  modelo de embedding exige um index novo (mude `PINECONE_INDEX_NAME`).
- Extensões aceitas: `SUPPORTED_EXTENSIONS` em `config.py`. Adicionar novas exige
  verificar se `llama-index-readers-file` cobre o formato.
- Respostas do LLM são em português do Brasil e restritas ao contexto
  (ver `SYSTEM_PROMPT` em `engine.py`).
- Variáveis de ambiente obrigatórias: `OPENAI_API_KEY`, `OPENAI_MODEL_NAME`,
  `PINECONE_API_KEY`. As demais têm defaults (ver a seção Instalação do README).
- `.env`, `data/documents/` e `data/manifest.json` estão no `.gitignore`. Nunca
  commite chaves de API nem documentos do usuário.
- Todas as funções, classes e métodos têm docstrings em português no estilo Google
  (`Args`/`Returns`/`Raises`). Ao criar ou alterar uma função, mantenha a docstring
  e atualize a tabela correspondente em `docs/arquitetura.md`.

## Ideias de evolução

- Reranker (ex.: `llama-index-postprocessor-cohere-rerank`) após a recuperação.
- Busca híbrida (Pinecone com sparse vectors) para termos exatos.
- Múltiplos namespaces por "coleção" de documentos, selecionáveis na sidebar.
- Avaliação automática com `llama-index` evaluators (faithfulness, relevancy).
