# RAG Local · Streamlit + LlamaIndex + Pinecone + OpenAI

Converse com seus próprios documentos. Os arquivos ficam em uma pasta local,
são indexados no Pinecone e consultados por um chat com memória, citações de
fonte e parâmetros ajustáveis pela interface.

## Funcionalidades

- **Index automático** — cria o index no Pinecone (serverless) na primeira execução, com a dimensão correta para o modelo de embedding.
- **Upload pela interface** — botão para adicionar PDF, DOCX, TXT, MD, CSV, JSON, HTML e PPTX; os arquivos são salvos em `data/documents/` e indexados na hora.
- **Sincronização da pasta** — arquivos copiados manualmente para a pasta são detectados e indexados; arquivos alterados são reindexados; nada é duplicado (controle por hash SHA-256).
- **Chat com memória** — perguntas de acompanhamento entendem o contexto da conversa (`condense_plus_context`).
- **Fontes** — cada resposta mostra os trechos recuperados, com arquivo, página e score de similaridade.
- **Parâmetros ao vivo** — Top-K, similaridade mínima e temperatura ajustáveis na sidebar.
- **Gestão** — remover um documento (arquivo + vetores), limpar a conversa e reindexar tudo do zero.

## Pré-requisitos

- Python 3.13 e [`uv`](https://docs.astral.sh/uv/)
- Conta na [OpenAI](https://platform.openai.com/) e no [Pinecone](https://www.pinecone.io/)

## Instalação

```bash
# dentro do diretório do projeto (com o ambiente uv já inicializado)
uv add streamlit python-dotenv pinecone \
       llama-index-core llama-index-llms-openai llama-index-embeddings-openai \
       llama-index-vector-stores-pinecone llama-index-readers-file
```

Crie o arquivo `.env` na raiz do projeto (ele está no `.gitignore` e nunca deve ser commitado):

```env
OPENAI_API_KEY=sk-...
OPENAI_MODEL_NAME=gpt-5.4
PINECONE_API_KEY=pcsk_...
```

Variáveis opcionais (com seus padrões): `OPENAI_EMBEDDING_MODEL=text-embedding-3-small`,
`PINECONE_INDEX_NAME=rag-local`, `PINECONE_NAMESPACE=default`, `PINECONE_CLOUD=aws`,
`PINECONE_REGION=us-east-1`, `CHUNK_SIZE=1024`, `CHUNK_OVERLAP=128`, `TOP_K=5`,
`LLM_TEMPERATURE=0.1`.

## Uso

```bash
uv run streamlit run app.py
```

1. Na sidebar, envie arquivos e clique em **Carregar e indexar** (ou copie arquivos para `data/documents/` e clique em **Sincronizar pasta local**).
2. Faça perguntas no campo de chat. Expanda **Fontes** para ver de onde a resposta veio.
3. Ajuste Top-K / similaridade / temperatura conforme necessário — a memória da conversa é reiniciada ao mudar parâmetros.

## Estrutura

```
app.py                  Interface Streamlit
rag/config.py           Configuração via .env
rag/llm.py              LLM e embeddings (OpenAI) no LlamaIndex
rag/pinecone_manager.py Ciclo de vida do index no Pinecone
rag/ingestion.py        Ingestão com manifesto anti-duplicação
rag/engine.py           Chat engine com recuperação e fontes
data/documents/         Seus documentos (fora do git)
data/manifest.json      Registro do que já foi indexado (gerado)
docs/arquitetura.md     Diagramas de arquitetura e referência de cada função
CLAUDE.md               Contexto da arquitetura para assistentes de IA
```

Para entender como os módulos se conectam e o que cada função faz, veja
[docs/arquitetura.md](docs/arquitetura.md).

## Solução de problemas

| Sintoma | Causa provável |
|---|---|
| `Variável de ambiente 'X' não encontrada` | `.env` ausente ou chave faltando |
| Erro de dimensão ao inserir vetores | Index criado com outro embedding; mude `PINECONE_INDEX_NAME` |
| Documento aparece ⚪ (não indexado) | Clique em **Sincronizar pasta local** |
| Respostas fora do contexto | Aumente a **similaridade mínima** ou reduza o Top-K |
| Vetores sumiram do Pinecone mas o manifesto ainda lista | Use **Reindexar tudo do zero** |
